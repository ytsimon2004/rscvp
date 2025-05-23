import collections
import functools
import warnings
from pathlib import Path
from typing import Union, NamedTuple, Optional, Literal, Iterable, overload

import numpy as np
import scipy.optimize
import scipy.stats
from matplotlib.axes import Axes
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list
from scipy.interpolate import interp1d
from tqdm import tqdm

from argclz import AbstractParser, argument, int_tuple_type
from neuralib.imaging.suite2p import get_neuron_signal, Suite2PResult, sync_s2p_rigevent
from neuralib.persistence import *
from neuralib.plot import plot_figure
from neuralib.plot.setting import ax_log_setting
from neuralib.util.verbose import fprint
from stimpyp import GratingPattern

__all__ = ['SFTFModelCacheBuilder']

PLOT_INTERP_TYPE = Literal['heatmap', 'ellipse']


class SftfFitResult(NamedTuple):
    sf_set: np.ndarray
    """sf value (SF,)"""
    tf_set: np.ndarray
    """tf value (TF,)"""
    rawmat: np.ndarray
    """raw 2d array (SF, TF)"""
    fitmat: Optional[np.ndarray]
    """fit 2d array (SF, TF), might be None if not goodfit"""
    xhat: Optional[tuple[float, ...]]
    """optimal values for the parameters, shape: (6,), might be None if not goodfit"""
    flag_goodfit: bool

    def apply_model(self, opt: 'SFTFModelCacheBuilder') -> 'SftfFitResult':
        sf_range = opt.sf_interp_range
        tf_range = opt.tf_interp_range
        fitmat = opt.apply_model(self.xhat, sf_range, tf_range)
        fitmat = fitmat.reshape((len(sf_range), len(tf_range)))
        return self._replace(sf_set=sf_range, tf_set=tf_range, fitmat=fitmat, rawmat=fitmat)

    @classmethod
    def mean(cls, sftf_ls: list['SftfFitResult']) -> 'SftfFitResult':
        fitmats = [it.fitmat for it in sftf_ls]
        mean_fitmat = np.mean(fitmats, axis=0)
        return SftfFitResult(sftf_ls[0].sf_set, sftf_ls[0].tf_set, mean_fitmat, mean_fitmat, None, True)


@persistence.persistence_class
class SFTFModelCache(ETLConcatable):
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    plane_index: str = persistence.field(validator=False, filename=True, filename_prefix='plane')

    neuron_idx: np.ndarray
    """(N,)"""
    src_neuron_idx: np.ndarray
    """(N,), value domain as plane_index"""

    sf_set: np.ndarray
    """sf value (SF,)"""
    tf_set: np.ndarray
    """tf value (TF,)"""

    rawmat: np.ndarray
    """(N, SF, TF)"""
    fitmat: np.ndarray
    """(N, SF, TF)"""
    xhat: np.ndarray
    """(N, 6)"""
    flag: np.ndarray
    """bool array for model fit (N, )"""

    def __len__(self):
        return len(self.neuron_idx)

    @overload
    def __getitem__(self, i: int) -> SftfFitResult:
        pass

    @overload
    def __getitem__(self, i: Union[slice, np.ndarray, list[int]]) -> 'SFTFModelCache':
        pass

    def __getitem__(self, i):
        """for neuron(s)"""
        if isinstance(i, int) or np.issubdtype(type(i), int):
            return SftfFitResult(self.sf_set, self.tf_set, self.rawmat[i], self.fitmat[i], self.xhat[i], self.flag[i])
        else:
            return self._replace(neuron_idx=self.neuron_idx[i],
                                 rawmat=self.rawmat[i],
                                 fitmat=self.fitmat[i],
                                 xhat=self.xhat[i],
                                 flag=self.flag[i])

    def _replace(self, **kwargs) -> 'SFTFModelCache':
        raise RuntimeError('auto generated by persistence')

    def __iter__(self) -> Iterable[SftfFitResult]:
        for i in range(len(self.neuron_idx)):
            yield SftfFitResult(self.sf_set, self.tf_set, self.rawmat[i], self.fitmat[i], self.xhat[i], self.flag[i])

    @classmethod
    def concat_etl(cls, s2p: Suite2PResult, data: list['SFTFModelCache']) -> 'SFTFModelCache':
        """
        sftf gaussian fitting result

        fitmat: (N, sf, tf) -> (N x P, sf, tf)
        xhat: (N, 6) -> (N*P, 6)
        """
        n_planes = s2p.n_plane
        ret = SFTFModelCache(
            data[0].exp_date,
            data[0].animal,
            '_concat',
        )

        ret.neuron_idx = np.concatenate([data[i].neuron_idx for i in range(n_planes)])
        ret.src_neuron_idx = np.concatenate([data[i].src_neuron_idx for i in range(n_planes)])
        ret.sf_set = data[0].sf_set
        ret.tf_set = data[0].tf_set
        ret.rawmat = np.vstack([data[i].rawmat for i in range(n_planes)])
        ret.fitmat = np.vstack([data[i].fitmat for i in range(n_planes)])
        ret.xhat = np.vstack([data[i].xhat for i in range(n_planes)])
        ret.flag = np.concatenate([data[i].flag for i in range(n_planes)])

        return ret


class SFTFModelCacheBuilder(AbstractParser, SelectionOptions, PersistenceRSPOptions[SFTFModelCache]):
    DESCRIPTION = 'plot the visual sftf model fit tuning'

    fit: bool = argument(
        '-f', '--fitting',
        action='store_true',
        help='whether do the model fitting'
    )

    avg_dir: bool = argument(
        '--avg', '--avg_dir',
        action='store_true',
        help='whether do the resp. average toward all the direction, if not, pick up maximal resp.'
    )

    summary: bool = argument(
        '--summary',
        action='store_true',
        help='fitting result of all cells (only classified visual cells)'
    )

    interp_sftf_shape: tuple[int, int] = argument(
        '--sftf-shape',
        metavar='SF,TF',
        type=int_tuple_type,
        default=(300, 300),
        help='sftf shape for doing the interpolation'
    )

    plot_all: Optional[PLOT_INTERP_TYPE] = argument(
        '--plot-all',
        metavar='PLOT_TYPE',
        default=None,
        help='plot all the neuron sftf after model fit'
    )

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        if self.summary:
            self.reuse_output = True

    select_mask = None  #

    def run(self):
        output = self.get_data_output('vf')

        if self.plane_index is None:
            cache = self._compute_cache_concat()
        else:
            cache = self.load_cache()

        self.select_mask = self._select_mask(cache)

        if self.summary:  # for all cells
            self.interp_sftf_fit(output, cache)
        else:
            # for each cell
            self.plot_foreach_sftf_fit(output, cache)

    def _select_mask(self, cache: SFTFModelCache):
        """select and success fitting neurons"""
        cell_mask = self.get_selected_neurons()
        nan_mask = np.apply_along_axis(lambda it: not np.any(np.isnan(it)), 1, cache.xhat)
        return np.logical_and(cell_mask, nan_mask)

    # ============================================= #
    # Signal processing and visual stimulus methods #
    # ============================================= #

    @functools.cached_property
    def stim_para(self) -> GratingPattern:
        rig = self.load_riglog_data()
        return GratingPattern.of(rig)

    def prepare_raw_data(self, para: GratingPattern, signal: np.ndarray,
                         image_time: np.ndarray) -> np.ndarray:
        """
        collect the data in proper shape for fitting

        :param para:
        :param signal:
        :param image_time:
        :return:
            raw data (sf, tf)
        """
        sf_i = para.sf_i()
        tf_i = para.tf_i()

        # dict[(sf, tf, ori)] = sig  (N, t), N: number of sti. , t: time bins
        cy = collections.defaultdict(list)
        for si, st, sf, tf, ori in para.foreach_stimulus():
            tx = np.logical_and(st[0] <= image_time, image_time <= st[1])
            sig = signal[tx]

            x = np.linspace(0.1, 0.9, num=len(sig))
            y = sig / np.max(signal)  # normalize

            cy[(sf, tf, ori)].append((x, y))  # collect for doing average

        # dict[(sf, tf)] = list((ori, y_max))
        oy = collections.defaultdict(list)
        for p, xy in cy.items():
            x = np.linspace(0.1, 0.9, num=len(sig))  # (T, )
            y = np.array([interp1d(it[0], it[1])(x) for it in xy])  # (N, t)

            y_mean = np.mean(y, axis=0)  # trial avg (t,)
            y_max = np.max(y_mean)  # max resp. of certain sftf in certain direction

            oy[(p[0], p[1])].append((p[2], y_max))

        # dir_avg or dir_max (sf_i, tf_i)
        rawmat = np.zeros((len(sf_i), len(tf_i)))
        for sftf, v in oy.items():  # sftf:(sf, tf); v:(ori, y_max)
            v = np.array(v)
            r = np.mean(v[:, 1], axis=0) if self.avg_dir else np.max(v[:, 1], axis=0)
            rawmat[sf_i[sftf[0]], tf_i[sftf[1]]] = r

        return rawmat

    # ============= #
    # Cache methods #
    # ============= #

    def empty_cache(self) -> SFTFModelCache:
        return SFTFModelCache(exp_date=self.exp_date,
                              animal=self.animal_id,
                              plane_index=self.plane_index)

    def compute_cache(self, cache: SFTFModelCache) -> SFTFModelCache:
        neuron_idx = np.array(self.get_all_neurons())
        cache.neuron_idx = neuron_idx
        cache.src_neuron_idx = self.get_neuron_plane_idx(len(neuron_idx), self.plane_index)
        cache.sf_set = self.stim_para.sf_set
        cache.tf_set = self.stim_para.tf_set

        rawmat = []
        fitmat = []
        xhat = []
        flag = []
        for (i, res) in self.fit_result(cache.neuron_idx):  # type: int, SftfFitResult

            rawmat.append(res.rawmat)

            if res.flag_goodfit:
                fitmat.append(res.fitmat)
                xhat.append(res.xhat)
                flag.append(True)
            else:
                fprint(f'{i} fitting failure', vtype='warning')
                fitmat.append(np.full((len(res.sf_set), len(res.tf_set)), np.nan))
                xhat.append(np.full(6, np.nan))
                flag.append(False)

        cache.rawmat = np.array(rawmat)
        cache.fitmat = np.array(fitmat)
        cache.xhat = np.array(xhat)
        cache.flag = np.array(flag)
        return cache

    def _compute_cache_concat(self) -> SFTFModelCache:
        s2p = self.load_suite_2p()
        n_planes = s2p.n_plane

        data = []
        for i in range(n_planes):
            self.plane_index = i
            data.append(self.load_cache())

        self.plane_index = None  # reset to avoid selection error

        return SFTFModelCache.concat_etl(s2p, data)

    # ========== #
    # SFTF model #
    # ========== #

    @staticmethod
    def _fit(data: np.ndarray, para: GratingPattern):
        """
        fit sftf

        :param data: 2d np.ndarray of calcium responses, shape: (sf, tf)
        :param para:
        :return:
            fitting result
            xhat: Amp, mu_sf, sigma_sf, mu_tf, sigma_tf, Q(tan) (fitted)
            flag_goodfit: bool, whether is a good fitting result
        """
        flag_goodfit = True
        tf = para.tf_set
        sf = para.sf_set

        max_resp = np.max(data)
        tfmat, sfmat = np.meshgrid(tf, sf)
        data[data <= 0] = 0.001  # rectify calcium responses for log fitting
        peak_sf = np.nonzero(np.mean(data, 1) == np.max(np.mean(data, 1)))[0]
        peak_tf = np.nonzero(np.mean(data, 0) == np.max(np.mean(data, 0)))[0]

        x0 = [1, sf[peak_sf[0]], 4, tf[peak_tf[0]], 3, 0]  # Amp, mu_sf, sigma_sf, mu_tf, sigma_tf, Q(tan)
        lb = [0, np.min(sf), 0.5, np.min(tf), 0.5, -1.5]  # lower bound
        ub = [max(3 * max_resp, 1), np.max(sf), 4, np.max(tf), 5, 1.5]  # upper bound

        try:
            # pre-give tf to sftf_model, refer to f = @(par, sfmat)
            # OptimizeWarning if covariance of the parameters can not be estimated
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xhat, pcov = lsqcurvefit(functools.partial(sftf_model, len(tf), len(sf), tfmat), x0, sfmat, data, lb,
                                         ub)
        except RuntimeError as e:
            flag_goodfit = False
            warnings.warn(f'fitting failure from {e}')

            return None, None, flag_goodfit

        perr = np.sqrt(np.diag(pcov))
        # print('perr: ', perr)

        fitmat = sftf_model(len(tf), len(sf), tfmat, sfmat, *xhat)
        fitmat = fitmat.reshape(len(sf), len(tf))

        if np.max(data) / np.max(fitmat) > 2:  # threshold
            flag_goodfit = False
            fprint('amplitude do not match', vtype='error')

        return fitmat, xhat, flag_goodfit

    def fit_model(self, para: GratingPattern, signal: np.ndarray, image_time: np.ndarray):
        """fit raw data to model"""
        rawmat = self.prepare_raw_data(para, signal, image_time)
        fitmat, xhat, flag_goodgit = self._fit(rawmat, para)
        return SftfFitResult(para.sf_set, para.tf_set, rawmat, fitmat, xhat, flag_goodgit)

    def fit_result(self, neuron_list=None) -> Iterable[tuple[int, SftfFitResult]]:
        """generate neuron_id and sftf fit result"""
        s2p = self.load_suite_2p()
        if neuron_list is None:
            neuron_list = get_neuron_list(s2p, self.neuron_id)
        rig = self.load_riglog_data()
        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        for neuron_id in tqdm(neuron_list, desc='visual_sftf_tuning', unit='neuron', ncols=80):
            signal = get_neuron_signal(s2p, neuron_id)[0]
            yield neuron_id, self.fit_model(self.stim_para, signal, image_time)

    @property
    def sf_interp_range(self) -> np.ndarray:
        para = self.stim_para
        return np.linspace(np.min(para.sf_set), np.max(para.sf_set), self.interp_sftf_shape[0])

    @property
    def tf_interp_range(self) -> np.ndarray:
        para = self.stim_para
        return np.linspace(np.min(para.tf_set), np.max(para.tf_set), self.interp_sftf_shape[1])

    def apply_model(self,
                    xhat: np.ndarray,
                    sf_range: np.ndarray = None,
                    tf_range: np.ndarray = None) -> np.ndarray:
        """
        applying model by interpolation sftf shape

        :param xhat: fitted xhat
        :param sf_range
        :param tf_range
        :return:
        """
        sf_inp = sf_range if sf_range is not None else self.sf_interp_range
        tf_inp = tf_range if tf_range is not None else self.tf_interp_range
        tfmat, sfmat = np.meshgrid(tf_inp, sf_inp)

        return sftf_model(len(tf_inp), len(sf_inp), tfmat, sfmat, *xhat)

    def interp_sftf_fit(self, output: DataOutput, cache: SFTFModelCache):
        """
        interpolation plot of sftf shape by applying the model (for all selected neurons)
        """

        para = self.stim_para
        cache = cache[self.select_mask]
        sftf_ls = [it.apply_model(self) for it in cache]

        if self.plot_all:
            self.plot_sftf_interp_foreach(cache.xhat, sftf_ls, para, output)

        else:  # average of all vc (heatmap)
            with plot_figure(self.get_interp_output_file(output), 1, 3) as ax:
                self._plot_sftf_heatmap(ax[0], SftfFitResult.mean(sftf_ls))
                self._plot_sftf_ellipse(ax[1], para, cache.xhat)
                self._plot_sftf_dot(ax[2], sftf_ls)

    # =============== #
    # Plotting method #
    # =============== #

    @property
    def cell_id(self) -> np.ndarray:
        return np.nonzero(self.select_mask)[0]

    @property
    def red_select_mask(self) -> np.ndarray:
        red_cell_mask = self.select_red_neurons()
        return red_cell_mask & self.select_mask

    @property
    def red_cell_id(self) -> np.ndarray:
        return np.nonzero(self.red_select_mask)[0]

    def get_interp_output_file(self, output: DataOutput) -> Path:
        """for interpolation output plot"""
        return output.summary_figure_output(
            'pre' if self.pre_selection else None,
            f'{self.vc_selection}' if self.vc_selection is not None else None,
            'inp',
            f'{self.plot_all}_all' if self.plot_all is not None else None,
            f'{self.random}_random' if self.random is not None else None
        )

    def plot_foreach_sftf_fit(self, output: DataOutput, cache: SFTFModelCache):
        """
        fit and plot for each neuron
        :return
            "fit result" and "xhat"
        """

        for (i, res) in tqdm(enumerate(cache), desc='foreach_sftf', unit='neuron',
                             ncols=80):  # type: int, SftfFitResult
            if res.flag_goodfit:
                output_file = output.figure_output(i, 'fit' if self.fit else None)
                with plot_figure(output_file) as ax:
                    self._plot_sftf_heatmap(ax, res)

    def plot_sftf_interp_foreach(self,
                                 xhats: np.ndarray,
                                 sftfs: list[SftfFitResult],
                                 pattern: GratingPattern,
                                 output: DataOutput):
        """
        plot each neuron sftf after interpolation
        :param xhats: (N, 6)
        :param sftfs:
        :param pattern:
        :param output:
        :return:
        """
        n_neurons = len(self.cell_id)
        r = int(np.sqrt(n_neurons))
        c = int(n_neurons // r + 1)

        with plot_figure(self.get_interp_output_file(output), r, c) as ax:
            ax = ax.ravel()
            for i in range(r * c):
                if i < n_neurons:
                    if self.plot_all == 'ellipse':
                        self._plot_sftf_ellipse(ax[i], pattern, xhats[i])
                        self._plot_sftf_dot(ax[i], sftfs[i])
                    elif self.plot_all == 'heatmap':
                        self._plot_sftf_heatmap(ax[i], sftfs[i], show_info=False)
                        self._plot_sftf_dot(ax[i], sftfs[i])

                    if self.has_chan2 and self.cell_id[i] in self.red_cell_id:
                        ax[i].text(2, 0.1, f'{self.cell_id[i]}', fontsize=6, color='r', weight='bold')
                    else:
                        ax[i].text(2, 0.1, f'{self.cell_id[i]}', fontsize=6, color='k', weight='bold')

                    ax[i].axes.yaxis.set_visible(False)
                    ax[i].axes.xaxis.set_visible(False)
                else:
                    ax[i].set_visible(False)

    def _plot_sftf_heatmap(self, ax: Axes, sftf: SftfFitResult, show_info: bool = True):
        """
        plot the sftf tuning heatmap

        :param ax:
        :param sftf:
        :param show_info
        :return:
        """

        ax.imshow(
            sftf.fitmat if self.fit else sftf.rawmat,
            extent=[np.min(sftf.tf_set), np.max(sftf.tf_set), np.min(sftf.sf_set), np.max(sftf.sf_set)],
            cmap='jet',
            vmin=0,
            vmax=np.max(sftf.fitmat) if self.fit else np.max(sftf.rawmat),
            interpolation='none',
            aspect='auto',
            origin='lower'
        )

        ax_log_setting(ax, xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')

        if show_info:
            ax.set_title('fit' if self.fit else 'raw')

    def _plot_sftf_ellipse(self, ax: Axes, para: GratingPattern, xhat: np.ndarray):
        """elliptical form after model fit"""
        sf_inp = np.linspace(np.min(para.sf_set), np.max(para.sf_set), self.interp_sftf_shape[0])
        tf_inp = np.linspace(np.min(para.tf_set), np.max(para.tf_set), self.interp_sftf_shape[1])

        tfmat, sfmat = np.meshgrid(tf_inp, sf_inp)

        if xhat.ndim == 1:  # per cell
            R = sftf_model(len(tf_inp), len(sf_inp), tfmat, sfmat, *xhat, ravel=False)
            lv = np.array([np.percentile(R, 50), np.max(R)])
            ax.contourf(tfmat, sfmat, R, levels=lv)
        else:
            for x in xhat:
                R = sftf_model(len(tf_inp), len(sf_inp), tfmat, sfmat, *x, ravel=False)
                lv = np.array([np.percentile(R, 50), np.max(R)])
                ax.contourf(tfmat, sfmat, R, levels=lv, alpha=0.05, colors='g')
                ax.contour(tfmat, sfmat, R, levels=lv, linewidths=1, colors='g', alpha=0.5)

        ax_log_setting(ax, xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')

    @staticmethod
    def _plot_sftf_dot(ax: Axes, sftfs: Union[list['SftfFitResult'], 'SftfFitResult']):
        """dot plot of sftf fit, each dot represents single neuron"""

        def _plot_per_cell(result: SftfFitResult, show_info: bool = True, **kwargs):
            """ """
            from numpy import unravel_index
            i, j = unravel_index(result.fitmat.argmax(), result.fitmat.shape)
            x = result.tf_set[j]
            y = result.sf_set[i]
            ax.plot(x, y, 'ko', **kwargs)
            if show_info:
                ax.set_xlim(np.min(result.tf_set), np.max(result.tf_set))
                ax.set_ylim(np.min(result.sf_set), np.max(result.sf_set))
                ax_log_setting(ax, xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')

        if not isinstance(sftfs, list):  # per cell
            p = sftfs
            _plot_per_cell(p, show_info=False, markersize=3, alpha=0.4)
        else:
            for sftf in sftfs:
                _plot_per_cell(sftf, markersize=4, alpha=0.4)


def lsqcurvefit(func, x0, xdata, ydata, lb, ub):
    """
    Solve nonlinear curve-fitting (data-fitting) problems in least-squares sense
    https://www.mathworks.com/help/optim/ug/lsqcurvefit.html#d123e113051
    """
    xdata = xdata.ravel()
    ydata = ydata.ravel()
    popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata, x0, bounds=(lb, ub))
    return popt, pcov


# ydata = f(xdata, *params) + eps.
def sftf_model(tf: int, sf: int, tfmat, sfmat, amp, mu_sf, sigma_sf, mu_tf, sigma_tf, e,
               ravel: bool = True) -> np.ndarray:
    """
    Andermann 2011; Glickfeld 2013,  two-dimensional elliptical Gaussian

    .. seealso::

        :func:`matlab.visual.mymodel_sftf.m`

    :param tf: number of tf set
    :param sf: number of sf set
    :param tfmat:
    :param sfmat:
    :param amp: amplitude (neuron's peak response)
    :param mu_sf: center of sf (neuron's preferred sf)
    :param sigma_sf: sigma of sf tuning, standard deviation of log gaussian (sf tuning width)
    :param mu_tf: center of tf (neuron's preferred tf)
    :param sigma_tf: sigma of tf tuning standard deviation of log gaussian (tf tuning width)
    :param e:
    :param ravel: whether transform 2d to 1d
    :return:
    """
    tfmat = tfmat.reshape((sf, tf))
    sfmat = sfmat.reshape((sf, tf))

    TP = e * (np.log2(sfmat) - np.log2(mu_sf)) + np.log2(mu_tf)
    t1 = np.exp(-(np.log2(sfmat) - np.log2(mu_sf)) ** 2 / (2 * sigma_sf ** 2))
    t2 = np.exp(-(np.log2(tfmat) - TP) ** 2 / (2 * sigma_tf ** 2))
    R = amp * t1 * t2
    if ravel:
        R = R.ravel()  # for scipy curve fit
    return R


if __name__ == '__main__':
    SFTFModelCacheBuilder().main()
