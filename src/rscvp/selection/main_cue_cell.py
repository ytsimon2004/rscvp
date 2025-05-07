"""
1. find peaks that higher than 30 % of the highest peak in trial average signal
2. peak activity is at least three times higher than median activity
3. reliability > 20% of total trials
4. exclude peak around reward location (< ?? cm, > ?? cm on the track)
5. different between identified positions is similar to distance between two cues (50 +- 5, 10%) (modified)
----
cue cells are not further tested for LNP model?
"""

from typing import cast

import numpy as np
from matplotlib.axes import Axes
from rscvp.selection.utils import image_time_per_trial
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.typing import ArrayLike, array2str
from neuralib.util.unstable import unstable

__all__ = ['CueCLSOptions']


@unstable()
class CueCLSOptions(AbstractParser, ApplyPosBinActOptions, SelectionOptions):
    DESCRIPTION = 'cue cell identification'

    peak_threshold: float = argument(
        '--peak-thres',
        default=0.3,
        help='signal threshold for finding the peak responses',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('cue', self.session)

        self.calc_cue_resp(output_info, self.neuron_id)

    def calc_cue_resp(self, output: DataOutput, neuron_ids: NeuronID):
        """
        classify the cue cells
        """
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)
        rig = self.load_riglog_data()

        # raw
        signal, baseline = get_neuron_signal(s2p, neuron_list, signal_type='df_f', normalize=False)
        signal = gaussian_filter1d(signal, 5)

        bin_size = self.belt_length / self.window
        binned_sig = self.apply_binned_act_cache().occ_activity  # (N,L,B)
        binned_sig[np.isnan(binned_sig)] = 0
        binned_sig = gaussian_filter1d(binned_sig, 2, axis=2, mode='wrap')

        session = rig.get_stimlog().session_trials()[self.session]
        lap_mask = session.time_mask_of(rig.lap_event.time)
        lap_time = rig.lap_event.time
        lap_time = lap_time[lap_mask]

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)
        act_mask = session.time_mask_of(image_time)

        # raw
        baseline_mean = cast(float, np.mean(baseline))
        baseline_std = np.std(baseline)
        thres = baseline_mean + 3 * baseline_std

        # bin
        binned_sig = binned_sig[:, lap_mask, :]

        with csv_header(output.csv_output,
                        ['neuron_id', f'cue_points_{self.session}', f'cue_reliability_{self.session}',
                         f'n_cue_{self.session}', f'cue_diff_{self.session}',
                         'is_cue_cell']) as csv:
            pos = np.arange(0, self.belt_length, bin_size)
            for neuron in tqdm(neuron_list, desc='cue', unit='neurons', ncols=80):
                s = signal[neuron] if signal.shape[0] > 1 else signal[0]  # s.dim should be 1
                sig_trial = image_time_per_trial(image_time, lap_time, s, act_mask)

                p = find_peak_pos(binned_sig[neuron], thres=self.peak_threshold, bin_size=bin_size)
                r, p = self.check_reliability(sig_trial, thres, p)
                iscue = self.is_cue_cell(p, lower_bound=40, upper_bound=140, error_perc=0.1)

                _p = array2str(p)
                r = array2str(r)  # trial reliability each cue (nC,)
                d = array2str(crossdiff(p, only_unique=True))  # distance between each pairs of cue point

                with plot_figure(output.figure_output(neuron)) as ax:
                    plot_cue_points(ax, pos, binned_sig[neuron], p)

                csv(neuron, _p, r, len(p), d, iscue)

    def check_reliability(self,
                          sig_trial: list[np.ndarray],
                          threshold: float,
                          points: np.ndarray,
                          reliability: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """
        3. cue points exist in 20% of total trials (verified by raw signal instead of binned signal).
        Basically used the same criteria as `selection.main_trial_reliability.py`

        :param self
        :param sig_trial: signal per lap (L, S). Note that S has different len per lap.
            return from func: `selection.util.image_time_per_trial`
        :param threshold:
        :param points:
        :param reliability: percent of the total trials
        :return:
            reliability: for each point
            valid points:

        """
        rb = np.zeros(len(points))  # reliability of each points
        ret = np.zeros(len(points))
        trial = np.zeros(len(sig_trial))
        bin_size = self.belt_length / self.window

        for i, p in enumerate(points):

            for j, sig in enumerate(sig_trial):
                t = np.arange(0, len(sig) / self.window, 1 / self.window)
                s, _ = np.histogram(t, bins=self.window, weights=sig)
                if s[int(p / bin_size)] > threshold:
                    trial[j] = 1

            rb[i] = round(np.mean(trial), 2)
            if np.mean(trial) > reliability:
                ret[i] = 1

        return rb, points[ret.astype(bool)]

    def is_cue_cell(self, points: np.ndarray,
                    lower_bound: float,
                    upper_bound: float,
                    error_perc: float = 0.1) -> bool:
        """
        4. exclude peak around reward location (< 40 cm, > 140 cm on the track)
        5. different between identified positions is similar to distance between two cues (50 +- 5, 10%) (modified)

        :param points:
        :param lower_bound: cm along the linear track to avoid reward cell
        :param upper_bound: cm along the linear track to avoid reward cell
        :param error_perc:
        :return:
        """
        # exclude peak around reward location (< ?? cm, > ?? cm on the track)
        ret = np.zeros_like(points, dtype=bool)
        for p in points:
            np.logical_and(p > lower_bound, p < upper_bound, out=ret)

        points = points[ret]

        # make sure the distances between cue points
        delta = crossdiff(points)
        cue_distance = self.cue_loc[1] - self.cue_loc[0]
        cue_range = cue_distance * error_perc
        upper_limit = cue_distance + cue_range
        lower_limit = cue_distance - cue_range

        x = [np.logical_and(d > lower_limit, d < upper_limit)
             for d in delta]

        return np.any(x)


def plot_cue_points(ax: Axes, pos: np.ndarray, sig: np.ndarray, cue_point: np.ndarray):
    """

    :param ax:
    :param pos:
    :param sig: binned signal, (L,B)
    :param cue_point:
    :return:
    """
    trial_avg_sig = np.mean(sig, axis=0)
    ax.plot(pos, trial_avg_sig)
    for i in cue_point:
        ax.axvline(i, color='r', alpha=0.5, ls='--')
    ax.set_title(f'cue points: {cue_point}')


def find_peak_pos(sig: np.ndarray, thres: float = 0.3, bin_size: float = 1.5, fold: float = 2) -> np.ndarray:
    """
    1. find peaks that higher than 30 % of the highest peak in trial average signal
    2. peak activity is at least three times higher than median activity (modified)

    :param sig: binned signal, (L,B)
    :param thres: threshold for searching the peak
    :param bin_size: length for each position bin
    :param fold: times larger than `baseline`
    :return:
    """
    trial_avg_sig = np.nanmean(sig, axis=0)
    from_height = (np.max(trial_avg_sig) - np.min(trial_avg_sig)) * thres
    peak_idx, _ = find_peaks(trial_avg_sig, height=from_height)

    # is_larger_than x fold of baseline signal
    bas = np.median(trial_avg_sig)

    ret = []
    for p in peak_idx:
        if trial_avg_sig[p] > bas * fold:
            ret.append(p * bin_size)

    return np.array(ret)


def crossdiff(x: ArrayLike, only_unique: bool = False) -> np.ndarray:
    """
    Calculate the diff across all the pairs

    :param x: 1D
    :param only_unique: return only unique value's array
    :return:
    """
    import itertools
    ret = [y - x for x, y in itertools.combinations(x, 2)]
    if only_unique:
        return np.unique(ret)

    return np.array(ret)


if __name__ == '__main__':
    CueCLSOptions().main()
