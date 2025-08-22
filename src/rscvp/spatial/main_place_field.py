import itertools
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from typing_extensions import Self

from argclz import AbstractParser, argument, int_tuple_type
from neuralib.imaging.suite2p import SIGNAL_TYPE
from neuralib.io import csv_header
from neuralib.plot import plot_figure, ax_merge
from neuralib.util.verbose import fprint
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli import DataOutput, PlotOptions, SelectionOptions, get_neuron_list, NeuronID
from rscvp.util.util_trials import TrialSelection

__all__ = ['PlaceFieldsOptions',
           'split_flatten_lter']


class PlaceFieldResult(NamedTuple):
    start: int
    """start index"""
    end: int
    """end index"""
    bin_size: float
    """bin size in cm"""
    pf: list[tuple[int, int] | None]  # (pf_idx[0], pf_idx[1])...
    """place field"""
    act: np.ndarray
    """Trial averaged transient activity. `Array[float, B]`"""
    baseline: float
    """Trial averaged baseline activity. `Array[float, B]`"""
    threshold: float
    """Amplitude threshold"""

    @property
    def n_pf(self) -> int:
        """Number of place field"""
        return len(self.pf)

    @property
    def pf_width(self) -> list[float]:
        """Place field width in cm"""
        return [self.bin_size * float((it[1] - it[0])) for it in self.pf]

    @property
    def pf_peak(self) -> list[float]:
        """Place field peak location in cm"""
        pf_peak_ls = []
        for p in self.pf:
            i = np.arange(p[0], p[1]) % (self.end + 1)  # deal with PP case (over-range p[1])
            x = np.argmax(self.act[i])
            p_max = i[x]

            pf_peak_ls.append(self.bin_size * float(p_max))

        return pf_peak_ls

    def with_width_filter(self, place_field_range: tuple[int, int]) -> Self:
        """
        Place field width filter

        :param place_field_range: Place field width (lower_bound, upper_bound)
        :return: ``PlaceFieldResult``
        """
        if len(self.pf) != 0:
            ret = [it for it in self.pf if
                   place_field_range[0] < self.bin_size * (it[1] - it[0]) < place_field_range[1]]
        else:
            ret = []
        return self._replace(pf=ret)

    def with_reliability_filter(self, reliability: list[float], at_least: float = 0.33) -> Self:
        """
        Place field reliability filter. place fields must be presented above one-third of all trials

        :param reliability: Reliability for each place field across trials
        :param at_least: Fraction of the trials
        :return: ``PlaceFieldResult``
        """
        if len(self.pf) != len(reliability):
            raise ValueError('length not match')

        pf = np.array(self.pf)
        x = [r > at_least for r in reliability]
        return self._replace(pf=list(pf[x]))


class PlaceFieldsOptions(AbstractParser, ApplyPosBinActOptions, SelectionOptions, PlotOptions):
    DESCRIPTION = 'Place field properties calculations, including place field width, peak location, numbers'

    width_range: tuple[int, int] = argument(
        '--pf_range',
        type=int_tuple_type,
        default=(15, 120),
    )

    peak_baseline_thres: float = argument(
        '--threshold',
        metavar='VALUE',
        default=0.3,
        help='threshold for the difference between peak and baseline activity',
    )

    reliability_threshold: float | None = argument(
        '--reliability',
        metavar='VALUE',
        default=0.33,
        help='fraction of the trials presented the place field activity'
    )

    signal_type: SIGNAL_TYPE = 'spks'

    def post_parsing(self):
        if not isinstance(self.width_range, tuple) or len(self.width_range) != 2:
            raise ValueError()

        if self.plot_summary:
            self.pre_selection = True
            self.pc_selection = 'slb'
            self.reuse_output = True

        if self.is_virtual_protocol:
            self.session = 'close'

    def run(self):
        self.post_parsing()
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('pf', self.session, use_virtual_space=self.use_virtual_space)

        if self.plot_summary:
            self.place_field_summary(output_info)
        else:
            self.foreach_place_field(output_info, self.neuron_id)

    def place_field_summary(self, output: DataOutput):
        """place field summary (only classified place cells)"""
        pf_width = self.get_csv_data(f'pf_width_{self.session}', session=self.session, infer_schema_length=1000)
        pf_peak = self.get_csv_data(f'pf_peak_{self.session}', session=self.session, infer_schema_length=1000)
        n_pf = self.get_csv_data(f'n_pf_{self.session}', session=self.session)

        if np.max(n_pf) > 3:
            fprint('cells with more than 3 place fields, check in details', vtype='warning')

        cell_mask = self.get_selected_neurons()

        pf_width = pf_width[cell_mask]
        pf_peak = pf_peak[cell_mask]
        n_pf = n_pf[cell_mask]
        n_neuron = self.n_selected_neurons

        p1 = n_pf == 1
        p2 = n_pf == 2
        p3 = n_pf == 3
        p_all = n_pf != 0

        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            self.pc_selection if self.pc_selection is not None else None
        )

        with plot_figure(output_file, 3, 3, figsize=(12, 5)) as _ax:
            plot_place_field_hist(_ax[0, 0], pf_peak[p1], color='k')
            plot_place_field_hist(_ax[1, 0], pf_peak[p2], color='royalblue')
            plot_place_field_hist(_ax[2, 0], pf_peak[p3], color='maroon')
            plot_place_field_hist(_ax[0, 1], pf_peak[p_all], color='c')

            _ax[2, 0].set_xlabel('Position (cm)')
            _ax[1, 0].set_ylabel('Fraction')
            _ax[0, 2].set_visible(False)

            #
            ax = ax_merge(_ax)[1:, 1]
            plot_place_field_width(ax, pf_width[p1], label='place_field_1', color='k')
            plot_place_field_width(ax, pf_width[p2], label='place_field_2', color='royalblue')
            plot_place_field_width(ax, pf_width[p3], label='place_field_3', color='maroon')
            plot_place_field_width(ax, pf_width[p_all], label='place_field_all', color='c')
            ax.set_xlabel('Place-field width (cm)')
            ax.set_ylabel('Cum. Prob.')
            ax.set_xlim(*self.width_range)
            ax.set_ylim(0, 1)
            ax.legend()

            #
            ax = ax_merge(_ax)[1:, 2]
            x = [1, 2, 3]

            if n_neuron != 0:
                y = [np.count_nonzero(n_pf == i) / n_neuron for i in x]
            else:
                fprint('no place cells are selected for plotting', vtype='warning')
                y = np.zeros_like(x)

            ax.bar(x, y, color=['k', 'royalblue', 'maroon'])
            ax.set_xlabel('Number of place fields per place cell')
            ax.set_ylabel('Fraction')

    def foreach_place_field(self,
                            output: DataOutput,
                            neuron_ids: NeuronID):
        """
        place field width was calculated from the number of consecutive position bins in which
        the mean activity exceeded 20% of difference between peak and baseline activity
        """
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        rig = self.load_riglog_data()
        sig_all = self.apply_binned_act_cache().occ_activity
        sig_base = self.apply_binned_act_cache().occ_baseline

        sig_all[np.isnan(sig_all)] = 0.0  # exclude nan point that generated after speed filter
        sig_base[np.isnan(sig_base)] = 0.0
        signal_all = gaussian_filter1d(sig_all, 3, mode='wrap', axis=2)

        trial = (
            TrialSelection.from_rig(rig, self.session, use_virtual_space=self.use_virtual_space)
            .get_selected_profile()
            .trial_slice
        )
        signal_all = signal_all[:, trial, :]  # (N, L, B)
        signal_bas = sig_base[:, trial, :]

        headers = [
            'neuron_id',
            'thres',
            f'pf_reliability_{self.session}',
            f'pf_width_raw_{self.session}',
            f'pf_width_{self.session}',
            f'pf_peak_{self.session}',
            f'n_pf_{self.session}'
        ]

        with csv_header(output.csv_output, headers) as csv:
            for neuron in tqdm(neuron_list, desc='pf', unit='neurons', ncols=80):
                sig = signal_all[neuron]  # (L, B)
                pf_result_raw = calc_place_field(sig,
                                                 signal_bas[neuron],
                                                 self.peak_baseline_thres,
                                                 self.pos_bins,
                                                 self.belt_length)

                # width filter
                pf_width_raw = ' '.join(map(str, pf_result_raw.pf_width))  # for .csv storage
                pf_result = pf_result_raw.with_width_filter(self.width_range)  # width selection

                # reliability filter
                n_trials = sig.shape[0]
                bin_size = self.belt_length / sig.shape[1]
                peak = [int(p / bin_size) for p in pf_result.pf_peak]  # cm to bin index
                pf_reliability = [
                    float(np.mean([np.any(sig[i, p] > pf_result.threshold) for i in range(n_trials)]))
                    for p in peak
                ]

                pf_result = pf_result.with_reliability_filter(pf_reliability, at_least=self.reliability_threshold)

                #
                csv(
                    neuron,
                    pf_result.threshold,
                    ' '.join(map(str, pf_reliability)),
                    pf_width_raw,
                    ' '.join(map(str, pf_result.pf_width)),
                    ' '.join(map(str, pf_result.pf_peak)),
                    pf_result.n_pf
                )

                with plot_figure(output.figure_output(neuron)) as ax:
                    plot_place_field(ax, pf_result, self.pos_bins, self.signal_type)


def calc_place_field(signal: np.ndarray,
                     baseline: np.ndarray,
                     threshold: float,
                     window: int = 100,
                     belt_length: int = 150) -> PlaceFieldResult:
    """
    Calculate place field information.
    Position bins in which the activity exceeded 20% of the difference between peak and baseline activity.
    field width below 15cm or above 120cm were excluded

    .. seealso:: Mao et al., 2017. Nature Communications

    :param signal: 2D position-binned transient activity. `Array[float, [L,B]]`
    :param baseline: 2D position-binned baseline activity. `Array[float, [L,B]]`
    :param threshold: Threshold for the difference between peak and baseline activity
    :param window: Bin number for the belt
    :param belt_length: Length of the belt
    :return: ``PlaceFieldResult``
    """
    act = np.nanmean(signal, axis=0)  # trial avg (B, )
    baseline = np.nanmean(baseline, axis=0)  # trial avg (B,)
    baseline = np.mean(baseline)  # bin avg, scalar

    act_threshold = (np.max(act) - baseline) * threshold + baseline

    bin_size = belt_length / window  # cm

    intersect = np.nonzero(np.diff((act > act_threshold).astype(int)) != 0)[0]
    left = (act - act_threshold)[0]
    right = (act - act_threshold)[-1]

    # position idx
    start = 0
    end = window - 1

    # different cases of place field patterns on 1d track
    if left < 0 and right < 0:  # NN
        pf_idx = [
            (intersect[i], intersect[i + 1])
            for i in range(0, len(intersect), 2)
        ]
    elif left < 0 < right:  # NP
        pf_idx = [
            (intersect[i], end) if i + 1 == len(intersect) else (intersect[i], intersect[i + 1])
            for i in range(0, len(intersect), 2)
        ]
    elif right < 0 < left:  # PN
        pf_idx = [(start, intersect[0])] + [
            (intersect[i], intersect[i + 1])
            for i in range(1, len(intersect), 2)
        ]
    elif 0 < left and 0 < right:  # PP
        pf_idx = [
            (intersect[i], end + intersect[0]) if i + 1 == len(intersect) else (intersect[i], intersect[i + 1])
            for i in range(1, len(intersect), 2)
        ]
    else:
        fprint('zero activity might found', vtype='warning')
        pf_idx = []

    # noinspection PyTypeChecker
    return PlaceFieldResult(
        start,
        end,
        bin_size,
        pf_idx,
        act,
        baseline,
        act_threshold,
    )


def plot_place_field(ax: Axes,
                     pf_result: PlaceFieldResult,
                     window: int = 100,
                     act_type: SIGNAL_TYPE = 'spks'):
    """plot the place field for individual neurons"""
    ax.plot(np.linspace(0, 150, window), pf_result.act, color='k', label='mean_act')
    ax.axhline(pf_result.threshold, color='r', label='threshold')
    ax.axhline(pf_result.baseline, color='b', ls='--', label='baseline')
    ax.set_title(f'width: {pf_result.pf_width} \n n_pf: {pf_result.n_pf}, pf_peak: {pf_result.pf_peak}')
    ax.set(xlabel='position(cm)', ylabel=f'normalized activity ({act_type})')
    ax.legend()


def split_flatten_lter(x: np.ndarray | list[str],
                       sep: str = ' ',
                       to_numpy: bool = True,
                       dtype: type = str) -> list | np.ndarray:
    """
    split each component in x and flatten as 1d array

    :param x: array-like iterable
    :param sep: pattern use for separate each component
    :param to_numpy
    :param dtype
    :return:
    """
    ls = [i.split(sep) for i in x]  # remove space in str to transfer to float array due to the csv pf format
    ret = list(itertools.chain(*ls))

    if dtype == float:
        ret = list(map(float, ret))

    return np.array(ret).astype(dtype) if to_numpy else ret


def plot_place_field_hist(ax: Axes,
                          pf_peak: np.ndarray,
                          bins: int = 40,
                          r: tuple[int, int] = (0, 150),
                          **kwargs):
    """Histogram with normalized yaxis"""
    label = kwargs.pop('label', f'n={len(pf_peak)}')
    pf_peak = split_flatten_lter(pf_peak, dtype=float).ravel()

    hist, bin_edges = np.histogram(pf_peak, bins=bins, range=r)
    hist = hist / np.sum(hist)

    ax.hist(bin_edges[:-1], bins=bin_edges, weights=hist, label=label, **kwargs)
    ax.legend()


def plot_place_field_width(ax: Axes, pf_width: np.ndarray, **kwargs):
    """
    Cumulative plot for the place-field width
    """
    pf_width = split_flatten_lter(pf_width, dtype=float).ravel()
    x1 = np.sort(pf_width)
    y1 = np.linspace(0, 1, len(x1))
    ax.plot(x1, y1, lw=2, **kwargs)


if __name__ == '__main__':
    PlaceFieldsOptions().main()
