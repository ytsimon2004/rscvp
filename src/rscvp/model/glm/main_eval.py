from typing import Union

import attrs
import numpy as np
from matplotlib.axes import Axes
from rscvp.model.glm.input import GLMInputData
from rscvp.util.cli import GLMOptions, SelectionOptions, DataOutput, NeuronID, get_neuron_list
from rscvp.util.cli.cli_camera import CameraOptions
from scipy.stats import pearsonr
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import Suite2PResult
from neuralib.io import csv_header
from neuralib.plot import plot_figure, ax_merge
from neuralib.util.unstable import unstable
from stimpyp import RiglogData

__all__ = ['BehavioralGLMOptions']


@unstable(method='run')
class BehavioralGLMOptions(AbstractParser, SelectionOptions, CameraOptions, GLMOptions):
    DESCRIPTION = 'LNP model fitting and predict the spike using different behavioral variables'

    predict_norm: bool = argument(
        '--pnorm',
        help='do the signal normalization for both predict and true spks',
    )

    cross_validation = 5
    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output(f'lnp_{self.var_type}')

        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()

        dat = self.get_lnp_inputs(s2p, rig)
        self.foreach_model_eval(s2p, dat, self.neuron_id, output_info)

    def get_lnp_inputs(self, s2p: Suite2PResult, rig: RiglogData) -> GLMInputData:

        args = dict(s2p=s2p, rig=rig, plane_index=self.plane_index, signal_type=self.signal_type, session=self.session)

        if self.var_type == 'pos':
            cov = GLMInputData.of_pos(**args)
        elif self.var_type == 'speed':
            cov = GLMInputData.of_speed(**args)
        elif self.var_type == 'lick_rate':
            if self.lick_event_source == 'facecam':
                args = {**args, **dict(lick_tracker=self.load_lick_tracker(rig))}
            cov = GLMInputData.of_lick(**args)
        elif self.var_type == 'acceleration':
            cov = GLMInputData.of_acceleration(**args)
        else:
            raise NotImplementedError('')

        return cov

    def foreach_model_eval(self,
                           s2p: Suite2PResult,
                           data: GLMInputData,
                           neuron_ids: NeuronID,
                           output: DataOutput) -> None:
        """Model evaluation using scikit-learn `PoissonRegressor` by cross validation of D2 score, MSE
        and plot the linear correlation"""
        neuron_list = get_neuron_list(s2p, neuron_ids)

        with csv_header(output.csv_output,
                        ['neuron_id',
                         f'mse_{self.var_type}',
                         f'r2_{self.var_type}',
                         f'cc_{self.var_type}',
                         f'score_{self.var_type}']) as csv:
            for neuron in tqdm(neuron_list, desc=f'lnp_{self.var_type}', unit='neurons', ncols=80):
                res = cross_validate(data, neuron, predict_norm=self.predict_norm, n_splits=self.cross_validation)

                act = res.y_test
                act_predict = res.y_predict

                if self.predict_norm:
                    act /= np.max(act)
                    act_predict /= np.max(act_predict)

                cc = pearsonr(act, act_predict)[0]

                output_file = output.figure_output(neuron, 'lnp_corr')
                with plot_figure(output_file, 1, 6, figsize=(12, 5)) as _ax:

                    ax = ax_merge(_ax)[:2]
                    y_hist, y_hist_predict, edg = get_act_hist(res.t_test, act, act_predict, do_norm=self.predict_norm)

                    mse = calc_mse(y_hist, y_hist_predict)
                    plot_act_cmp(ax, y_hist, y_hist_predict, edg, mse)

                    ax = ax_merge(_ax)[2:3]
                    plot_cv_value(ax, res.mse_all, n_splits=self.cross_validation, label='cv mse')
                    ax = ax_merge(_ax)[3:4]
                    plot_cv_value(ax, res.score_all, n_splits=self.cross_validation, label='cv D2 score')

                    ax = ax_merge(_ax)[4:]
                    r2 = plot_linear_corr(ax, y_hist, act_predict)

                    csv(neuron, mse, round(r2, 2), round(cc, 2), res.best_score)


@attrs.frozen
class GLMTestResult:
    """cross validation model predict result"""
    n_split: int

    t_test_all: list[np.ndarray] = attrs.field(factory=list)
    """timestamp for the test dataset (K,) """

    y_test_all: list[np.ndarray] = attrs.field(factory=list)
    """binned spks counts for the test dataset. (K,)"""

    y_predict_all: list[np.ndarray] = attrs.field(factory=list)
    """binned spks counts for the predicted results, (K,)"""

    mse_all: list[float] = attrs.field(factory=list)
    """mean square error for the model predict. (K,)"""

    score_all: list[float] = attrs.field(factory=list)
    """d^2 score for the model predict. (K,)"""

    def __iter__(self):
        iter_it = [self.t_test_all, self.y_test_all, self.y_predict_all, self.mse_all, self.score_all]
        for it in iter_it:
            yield it

    def __attrs_post_init__(self):
        for it in self:
            if isinstance(it, list):
                if len(it) != self.n_split:
                    raise RuntimeError('')

    @property
    def best_predict_idx(self) -> int:
        return np.argmax(self.score_all)

    @property
    def best_score(self) -> float:
        return self.score_all[self.best_predict_idx]

    @property
    def t_test(self) -> np.ndarray:
        return self.t_test_all[self.best_predict_idx]

    @property
    def y_test(self) -> np.ndarray:
        return self.y_test_all[self.best_predict_idx]

    @property
    def y_predict(self) -> np.ndarray:
        return self.y_predict_all[self.best_predict_idx]


def cross_validate(data: GLMInputData,
                   neuron_id: int,
                   predict_norm: bool,
                   n_splits: int) -> GLMTestResult:
    """
    Compute the cross-validation model prediction results

    :param data: `LNPCov`
    :param neuron_id
    :param predict_norm: normalization of y and y_predict
    :param n_splits: number of folds for k-fold validation

    """
    from sklearn.model_selection import KFold

    model = PoissonRegressor()

    ret = []
    mse_all = []
    score_all = []
    t_test_all = []
    y_test_all = []
    y_predict_all = []

    kfold_iterator = KFold(n_splits, shuffle=False)
    for i, (train_idx, test_index) in enumerate(kfold_iterator.split(data.cov)):
        # Split up the overall training data into cross-validation training and validation sets
        train_set = data[train_idx]
        test_set = data[test_index]

        X_train, Y_train, _ = train_set.prepare_XY(neuron_id)
        model.fit(X_train, Y_train)

        X_test, Y_test, tbins = test_set.prepare_XY(neuron_id)
        Y_predict = model.predict(X_test)

        y_test = test_set.neural_activity[neuron_id]

        if predict_norm:
            y_test = y_test / np.max(y_test)
            Y_predict = Y_predict / np.max(Y_predict)

        y_hist, y_hist_predict, _ = get_act_hist(test_set.time, y_test, Y_predict, do_norm=predict_norm)
        mse = calc_mse(y_hist, y_hist_predict)

        #
        mse_all.append(mse)
        score_all.append(model.score(X_test, Y_test))
        t_test_all.append(tbins[1:])
        y_test_all.append(Y_test)
        y_predict_all.append(Y_predict)

        ret.append(model)

    return GLMTestResult(n_splits,
                         t_test_all,
                         y_test_all,
                         y_predict_all,
                         mse_all,
                         score_all)


# ============= #

def plot_linear_corr(ax: Axes, y: np.ndarray, predict_y: np.ndarray):
    """
    linear regression for true signal versus predicted signal
    :param ax:
    :param y: true neural signal
    :param predict_y: model predict binned neural activity
    :return:
    """
    from scipy.stats import linregress

    res = linregress(y, predict_y)
    r_sqrt = res.rvalue ** 2

    ax.plot(y, predict_y, 'ko', markersize=4, label='raw', alpha=0.4)
    ax.plot(y, res.intercept + res.slope * y, 'r', label='linear fitting')
    ax.set_xlabel('spks')
    ax.set_ylabel('predicted spks')
    ax.set_title(f'R-squared: {r_sqrt:.4f}')
    ax.legend()

    return r_sqrt


def calc_mse(y: np.ndarray, y_predict: np.ndarray) -> float:
    """calculate the mean square error for `true y` and `model predicted y`"""
    return round(np.mean((y - y_predict) ** 2), 4)


def get_act_hist(t: np.ndarray,
                 y: np.ndarray,
                 predict_y: np.ndarray,
                 do_norm: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = len(predict_y)
    y_hist, edg = np.histogram(t, b, weights=y)
    if do_norm:
        y_hist /= np.max(y_hist)
        predict_y /= np.max(predict_y)

    return y_hist, predict_y, edg


def plot_act_cmp(ax: Axes,
                 y,
                 y_predict,
                 edg: np.ndarray,
                 mse: float):
    """
    compare the true neural signal and predicted signal (using glm lnp model fit)

    :param ax:
    :param y: true neural signal
    :param y_predict: model predict binned neural activity
    :param edg: histogram edg
    :param mse:
    :param predict_norm:
    :return:
    """

    ax.plot(edg[:-1], y, label='true_binned', alpha=0.8)
    ax.plot(edg[:-1], y_predict, color='r', alpha=0.5, label='predict')
    ax.set(xlabel='t', ylabel='act')
    ax.set_title(f'mse: {mse}')
    ax.legend()


def plot_cv_value(ax: Axes, value: Union[list[float], np.ndarray], n_splits: int, **kwargs):
    """ Plot the MSE values for the K_fold cross validation

    :param ax
    :param value: an array of size (number of splits,)

    :param n_splits

    """
    ax.boxplot(value)
    ax.set(**kwargs)

    ax.set_title(f'cv over {n_splits} data')


if __name__ == '__main__':
    BehavioralGLMOptions().main()
