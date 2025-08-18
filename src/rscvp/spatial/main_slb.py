from functools import cached_property

import numpy as np
import polars as pl
import scipy.ndimage
import scipy.stats
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import SIGNAL_TYPE
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.util.tqdm import tqdm_joblib
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.spatial.util import PositionSignal
from rscvp.spatial.util_plot import plot_tuning_heatmap
from rscvp.util.cli import MultiProcOptions, DataOutput, PlotOptions, FIG_MODE, SelectionOptions, ShuffleBaseOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from rscvp.util.util_trials import TrialSelection

__all__ = ['PositionLowerBoundOptions']


@publish_annotation('sup', project='rscvp', figure='fig.S1', as_doc=True)
class PositionLowerBoundOptions(AbstractParser,
                                ApplyPosBinActOptions,
                                ShuffleBaseOptions,
                                SelectionOptions,
                                PlotOptions,
                                MultiProcOptions):
    DESCRIPTION = """
    Calculate the spatial lower bound activity, and binned the shuffled activity to see the given percentile threshold
    """

    do_signal_smooth: bool = argument(
        '--do-smooth',
        help='whether do bin-smoothing for trial-averaged binned signal'
    )

    do_shuffle_smooth: bool = argument(
        '--do-shuffle-smooth',
        help='whether do bin-smoothing for each shuffle signal'
    )

    with_heatmap: bool = argument(
        '--heatmap',
        help='whether plot the heatmap above the shuffle curve'
    )

    with_place_field_info: bool = argument(
        '--pf',
        help='whether plot the place field threshold on the lower bound plot'
    )

    signal_type: SIGNAL_TYPE = 'spks'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('slb', self.session)
        self.foreach_lower_bound(output_info, self.neuron_id)
        self.aggregate_output_csv(output_info)

    # ============ #
    # Plot methods #
    # ============ #

    def foreach_lower_bound(self, output: DataOutput, neuron_ids: NeuronID):
        """
        Adapted from Mao et al., 2020. cur. bio.

        the linear track was divided into 100 position bins (1.5 cm per bin) and the occupancy-normalized activity
        was smoothed using a Gaussian window (SD = 3 position bins) for each neuron.
        First circularly shifted the neuronal activity for a random time between 20 and session duration less than 20s,
        then repeated this process for 1000 times and obtained the shuffled distribution.
        If the lower bound of the actual activity (mean-SEM) in any position bin across trials was greater than the 97.5
        percentile of the shuffled distribution, then the neuron was considered to carry significant spatial activity.

        :param output: ``DataOutput``
        :param neuron_ids: ``NeuronID``
        :return:
        """
        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()
        cp = PositionSignal(s2p, rig,
                            window_count=self.pos_bins,
                            signal_type=self.signal_type,
                            plane_index=self.plane_index)

        neuron_list = get_neuron_list(s2p, neuron_ids)
        signal_all = self.apply_binned_act_cache().occ_activity
        signal_all[np.isnan(signal_all)] = 0.0  # exclude nan points that generated after speed filter

        if self.do_signal_smooth:
            signal_all = gaussian_filter1d(signal_all, 3, mode='wrap', axis=2)

        trials = TrialSelection(rig, self.session).get_selected_profile().trial_range

        #
        if self.with_place_field_info:
            pf_info = self.get_data_output('pf', self.session, latest=True).csv_output
            r = f'pf_reliability_{self.session}'

            def _dtype_map(x: str) -> str:
                return ' '.join(f"{round(float(r), 2)}" for r in x.split())

            df = pl.read_csv(pf_info).with_columns(
                pl.col(r).map_elements(lambda it: _dtype_map(it), return_dtype=pl.Utf8)
            )
            reliability = df[r].to_list()
            thres = df['thres'].to_numpy()
        else:
            thres = None
            reliability = None

        #
        with tqdm_joblib(tqdm(desc="lower bound", unit='neuron', ncols=80)) as _:
            Parallel(n_jobs=self.parallel_jobs, backend='multiprocessing', verbose=True)(
                delayed(self._foreach_spatial_lower_bound)(
                    cp,
                    signal_all,
                    n,
                    trials,
                    thres[n] if self.with_place_field_info else None,
                    reliability[n] if self.with_place_field_info else None,
                    output
                )
                for n in neuron_list
            )

    def _foreach_spatial_lower_bound(self, cp, signal_all, neuron, trials, pf_thres, pf_rel, output: DataOutput):
        """pure (independent) function for parallel computing"""
        output_file = output.data_output(f'slb-tmp-{neuron}', ext='.csv')
        signal = signal_all[neuron, slice(*trials), :]  # shape (L', B)

        with csv_header(output_file, ['neuron_id', f'nbins_exceed_{self.session}'], append=True) as csv:
            act_mean = np.mean(signal, axis=0)  # (B,)
            act_sem = scipy.stats.sem(signal, axis=0)  # (B,)
            lower_bound = act_mean - act_sem  # can be changed depends on criteria

            act_shuffle = self.shuffle_binned_activity(cp, neuron, trials)
            thres = np.percentile(act_shuffle, self.percentage, axis=0)  # (B,)
            bottom_shuffle = np.percentile(act_shuffle, 2.5, axis=0)  # (B,)

            # how many percentage of lower bound that exceed the shuffled data
            nbin_exceed = np.count_nonzero(lower_bound > thres) / self.pos_bins * 100
            nbin_exceed = round(nbin_exceed)
            csv(neuron, nbin_exceed)

        # plot
        plot_args = {
            "mean_act": act_mean,
            "lower_bound": lower_bound,
            "shuffled_act": act_shuffle,
            "threshold": thres,
            "bottom_shuffle": bottom_shuffle,
            "nbin_exceed": nbin_exceed,
            "belt_length": self.belt_length,
            "window": self.pos_bins,
            "percentile": self.percentage,
            "pf_threshold": pf_thres,
            "pf_reliability": pf_rel,
            "mode": self.mode
        }
        #
        if self.with_heatmap:
            with plot_figure(output.figure_output(neuron), 2, 1, sharex=True) as ax:
                plot_tuning_heatmap(signal, ax=ax[0])
                plot_spatial_lower_bound(ax=ax[1], **plot_args)
        else:
            with plot_figure(output.figure_output(neuron)) as ax:
                plot_spatial_lower_bound(ax=ax, **plot_args)

    # =============== #
    # Shuffle methods #
    # =============== #

    @cached_property
    def shuffle_time_window(self) -> float:
        """numbers of activity temporal bins in 20s"""
        return 20 * self.load_suite_2p().fs

    def shuffle_binned_activity(self, cp: PositionSignal,
                                neuron: int,
                                lap_range: tuple[int, int]) -> np.ndarray:
        """
        Circularly shifted the neuronal activity for a random time between 20 and a session duration less than 20s.
        Refer to Mao et al., 2020. cur. bio.

        :param cp:
        :param neuron: individual neuron
        :param lap_range:

        :return: shape: (S, B), where S is shuffle_times
        """

        ret = np.zeros((self.shuffle_times, self.pos_bins), dtype=float)
        t, sig, _ = cp.get_signal(neuron, lap_range, dff=True)

        for i in range(self.shuffle_times):
            low = int(self.shuffle_time_window)
            high = int(len(sig) - self.shuffle_time_window)
            shu_sig = np.roll(sig, np.random.randint(low, high))
            binned_shu_sig = cp.binned_sig.calc_binned_signal(t, shu_sig, lap_range)  # shape (L, B)

            if self.do_shuffle_smooth:
                binned_shu_sig = gaussian_filter1d(binned_shu_sig, 3, mode='wrap', axis=1)

            ret[i] = np.mean(binned_shu_sig, axis=0)  # (B,)

        return ret


def plot_spatial_lower_bound(mean_act: np.ndarray,
                             lower_bound: np.ndarray,
                             shuffled_act: np.ndarray,
                             threshold: np.ndarray,
                             bottom_shuffle: np.ndarray,
                             nbin_exceed: int,
                             *,
                             belt_length: int = 150,
                             window: int = 100,
                             percentile: float = 97.5,
                             pf_threshold: float | None = None,
                             pf_reliability: float = None,
                             mode: FIG_MODE = 'simplified',
                             ax: Axes | None = None):
    """

    `Dimension parameters`:

        S = number of shuffled

        B = number of position bins

    :param mean_act: Trial-averaged neural activity. `Array[float, B]`
    :param lower_bound: Mean-SEM lower bound activity. `Array[float, B]`
    :param shuffled_act: Shuffled neural activity. `Array[float, [S, B]]`
    :param threshold: Upper percentile of shuffled distribution. `Array[float, B]`
    :param bottom_shuffle: Lower percentile of shuffled distribution. `Array[float, B]`
    :param nbin_exceed: Number of bins exceed the threshold
    :param belt_length: Linear track length
    :param window: Number of position bins
    :param percentile: Criteria percentage
    :param pf_threshold: Place field Threshold
    :param pf_reliability: Place field Reliability
    :param mode: ``FIG_MODE``
    :param ax: ``Axes``
    :return:
    """
    x = np.linspace(0, belt_length, num=window)

    if mode == 'presentation':
        for act in shuffled_act:
            ax.plot(x, act, color='r', alpha=0.1)

    ax.plot(x, mean_act, color='k', label='mean', ls='--', alpha=0.6)
    ax.plot(x, lower_bound, color='k', label='lower bound')
    ax.fill_between(x, threshold, bottom_shuffle, color='grey', alpha=0.6, edgecolor='none')

    title = ''
    if pf_threshold is not None:
        ax.axhline(pf_threshold, color='r', label='place_field')
        title += f'pf_reliability: {pf_reliability}. ' if pf_reliability is not None else ''

    title += f'nbins: {nbin_exceed} in {percentile}%'
    ax.set(xlabel='position', ylabel='normalized activity', title=title)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    ax.legend()


if __name__ == '__main__':
    PositionLowerBoundOptions().main()
