from typing import NamedTuple
from typing import Union

import attrs
import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
from typing_extensions import Self

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import Suite2PResult, get_neuron_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import plot_figure, ax_merge
from neuralib.util.unstable import unstable
from rscvp.util.cli import BEHAVIOR_COVARIANT, get_neuron_list
from rscvp.util.cli import GLMOptions, SelectionOptions, DataOutput, NeuronID
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.util_lick import LickTracker
from rscvp.util.util_trials import TrialSelection
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

    def get_lnp_inputs(self, s2p: Suite2PResult, rig: RiglogData) -> 'GLMInputData':

        if self.var_type == 'pos':
            cov = GLMInputData.of_pos(self)
        elif self.var_type == 'speed':
            cov = GLMInputData.of_speed(self)
        elif self.var_type == 'lick_rate':
            lick_tracker = self.load_lick_tracker() if self.lick_event_source == 'facecam' else None
            cov = GLMInputData.of_lick(self, lick_tracker)
        elif self.var_type == 'acceleration':
            cov = GLMInputData.of_acceleration(self)
        else:
            raise NotImplementedError('')

        return cov

    def foreach_model_eval(self,
                           s2p: Suite2PResult,
                           data: 'GLMInputData',
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


def cross_validate(data: 'GLMInputData',
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


class DTypeParams(NamedTuple):
    temporal_res: float
    """Temporal resolution of the model in hz"""

    sampling_rate: float
    """Behavioral measure sampling rate in hz"""

    n_cov_bins: int
    """Number of bins for the value domain in the dtype"""


DEFAULT_DTYPE_PARAMS: dict[BEHAVIOR_COVARIANT, DTypeParams] = {
    'pos': DTypeParams(10, 30, 50),
    'speed': DTypeParams(10, 30, 30),
    'acceleration': DTypeParams(10, 30, 30),
    'lick_rate': DTypeParams(10, 30, 20)
}


@attrs.define
class GLMInputData:
    """
    `Dimension parameters`:

        N = Number of neurons

        S = Number of samples = sampling rate (hz) * total time (sec)

    """
    dtype: BEHAVIOR_COVARIANT
    """``BEHAVIOR_COVARIANT``"""

    time: np.ndarray
    """Timestamp of the covariant in sec. `Array[float, S]`"""

    cov: np.ndarray
    """Covariant values. `Array[float, S]`"""

    neural_activity: np.ndarray
    """Neural activity (i.e., spks, dff). `Array[float, [N, S]]`"""

    pars: DTypeParams = attrs.field(init=False, default=attrs.Factory(dict), kw_only=True)

    def __attrs_post_init__(self):
        self.pars = DEFAULT_DTYPE_PARAMS[self.dtype]

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> Self:
        """Train/Test dataset"""
        return GLMInputData(
            self.dtype,
            self.time[idx],
            self.cov[idx],
            self.neural_activity[:, idx],
        )

    @property
    def n_temporal_bins(self) -> int:
        return int((np.max(self.time) - np.min(self.time)) * self.pars.temporal_res)

    def prepare_XY(self, neuron_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param neuron_id:
        :return:
            X: binned covariant
            Y: spike counts
            t_bins: temporal bins
        """
        X, t_bins, x_bins = np.histogram2d(self.time, self.cov, bins=(self.n_temporal_bins, self.pars.n_cov_bins))
        Y = np.histogram(self.time, t_bins, weights=self.neural_activity[neuron_id])[0]

        return X, Y, t_bins

    @classmethod
    def of_pos(cls, opt) -> Self:
        """position covariates"""
        return cls.of_var('pos', opt, None)

    @classmethod
    def of_speed(cls, opt) -> Self:
        """speed covariates"""
        return cls.of_var('speed', opt, None)

    @classmethod
    def of_lick(cls, opt, lick_tracker) -> Self:
        return cls.of_var('lick_rate', opt, lick_tracker)

    @classmethod
    def of_acceleration(cls, opt) -> Self:
        return cls.of_var('acceleration', opt, None)

    @classmethod
    def of_var(cls, dtype: BEHAVIOR_COVARIANT,
               opt: BehavioralGLMOptions,
               lick_tracker: LickTracker | None = None) -> Self:
        """
        Create inputs for Linear-nonlinear Poisson GLM model

        :param dtype: ``BEHAVIOR_COVARIANT``
        :param opt: ``BehavioralGLMOptions``
        :param lick_tracker
        :return:
        """

        rig = opt.load_riglog_data()
        s2p = opt.load_suite_2p()
        plane_index = opt.plane_index
        signal_type = opt.signal_type

        trial_sel = TrialSelection.from_rig(rig, opt.session, use_virtual_space=opt.use_virtual_space)

        sampling_rate = DEFAULT_DTYPE_PARAMS[dtype].sampling_rate
        start_time = trial_sel.start_time
        end_time = trial_sel.end_time
        cov_time = np.linspace(start_time, end_time, int((end_time - start_time) * sampling_rate))

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, plane_index)

        # signal
        sig, _ = get_neuron_signal(s2p, get_neuron_list(s2p), signal_type=signal_type, normalize=False)
        imask = trial_sel.masking_time(image_time)
        image_time = image_time[imask]
        sig = sig[:, imask]  # (N, fs*t)

        #
        if sampling_rate < 10:
            # do histogram on signal peak
            from scipy.signal import find_peaks

            s = np.zeros((sig.shape[0], len(cov_time) - 1))
            for i in range(sig.shape[0]):
                s[i] = np.histogram(image_time[find_peaks(sig[i])[0]], cov_time)[0]
            sig = s
        else:
            sig = interp1d(image_time, sig,
                           axis=1, kind='nearest', copy=False, bounds_error=False, fill_value=0)(cov_time)

        #
        if dtype in ('pos', 'speed', 'acceleration'):
            pos = opt.load_position()
            pmask = trial_sel.masking_time(pos.t)
            ptime = pos.t[pmask]

            if dtype == 'pos':
                cov = pos.p[pmask]
            elif dtype == 'speed':
                cov = pos.v[pmask]
            elif dtype == 'acceleration':
                v = pos.v[pmask]
                dv = np.diff(v, prepend=v[0])
                cov = dv * sampling_rate
            else:
                raise ValueError(f'unknown covarient type: {dtype}')

            cov = interp1d(ptime, cov, kind='nearest', copy=False, bounds_error=False, fill_value=0)(cov_time)

        #
        elif dtype == 'lick_rate':

            if lick_tracker is None:
                lick_time = rig.lick_event.time
                time = np.linspace(start_time, end_time, int((end_time - start_time) * sampling_rate))
                value, edg = np.histogram(lick_time, time)
                t = edg[:-1]
            else:
                value = lick_tracker.pix_probability
                t = lick_tracker.camera_time

            cov = interp1d(t, value, kind='nearest', copy=False, bounds_error=False, fill_value=0)(cov_time)

        else:
            raise ValueError(f'unknown covarient type: {dtype}')

        return GLMInputData(dtype, cov_time, cov, sig)


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
