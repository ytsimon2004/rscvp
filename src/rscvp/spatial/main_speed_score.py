from typing import Literal

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from tqdm import trange

from argclz import AbstractParser, as_argument, argument
from neuralib.imaging.suite2p import get_neuron_signal, SIGNAL_TYPE, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.locomotion import CircularPosition
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli import PlotOptions, DataOutput
from rscvp.util.position import PositionBinnedSig, load_interpolated_position
from rscvp.util.util_trials import TrialSelection
from stimpyp import RiglogData

__all__ = ['SpeedScoreOptions']


@publish_annotation('appendix', project='rscvp', caption='rev')
class SpeedScoreOptions(AbstractParser, ApplyPosBinActOptions, PlotOptions):
    DESCRIPTION = 'Calculate the speed score for each cell (Kropff et al., 2015)'

    plot_summary: bool = as_argument(PlotOptions.plot_summary).with_options(help='plot speed score summary histogram')

    plot_raw: bool = argument('--raw', help='plot raw correlation, otherwise plot the binned trial-averaged')

    signal_type: SIGNAL_TYPE = 'df_f'

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.plot_summary:
            self.reuse_output = True

    def run(self):
        self.post_parsing()
        output = self.get_data_output('sc', self.session, running_epoch=self.running_epoch)

        if self.plot_summary:
            self.plot_summary_hist(output)
        else:
            self.foreach_speed_score(output)

    def plot_summary_hist(self, output: DataOutput):
        field = f'speed_score_{self.session}'
        sc = pl.read_csv(output.csv_output)[field]

        with plot_figure(None) as ax:
            ax.hist(sc, bins=30)
            ax.set(xlabel='speed_score', ylabel='neuron #')

    def foreach_speed_score(self, output: DataOutput):
        rig = self.load_riglog_data()
        s2p = self.load_suite_2p()
        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)
        pos = load_interpolated_position(rig)
        speed = pos.v
        position_time = pos.t

        ix, px = self.trial_time_masking(rig, image_time, position_time)
        image_time = image_time[ix]
        speed = speed[px]
        position_time = position_time[px]

        binned_dff, binned_speed = self.load_binned_data(rig, pos)
        x = np.linspace(0, self.belt_length, num=self.pos_bins)

        with csv_header(output.csv_output, ['neuron_id', f'speed_score_{self.session}']) as csv:
            for n in trange(s2p.n_neurons, desc='calculate speed score', unit='neurons', ncols=80):
                dff = get_neuron_signal(s2p, n, signal_type=self.signal_type)[0]
                dff = dff[ix]
                score = corr_signal_helper(dff, speed, image_time, position_time)

                with plot_figure(output.figure_output(n)) as ax:
                    title = f'Neuron {n} | Speed score: {score:.2f}'
                    if self.plot_raw:
                        self.plot_corr_raw(ax, dff, speed, image_time, position_time, title=title)
                    else:
                        self.plot_corr_trial_averaged(ax, x, binned_dff[n], binned_speed, title=title)

                csv(n, score)

    def load_binned_data(self, rig: RiglogData, pos: CircularPosition) -> tuple[np.ndarray, np.ndarray]:
        """Load trial-averaged activity and speed signals"""
        indices = TrialSelection.from_rig(rig, self.session).get_time_profile().trial_range
        trial_range = np.arange(*indices)

        dff = self.apply_binned_act_cache().with_trial(trial_range).occ_activity  # shape (N, L, B)

        pbs = PositionBinnedSig(rig, bin_range=(0, self.belt_length, self.pos_bins))
        speed = pbs.calc_binned_signal(pos.t, pos.v, desc='calculate binned speed')
        speed = speed[slice(*indices), :]  # shape (L, B)

        return dff.mean(axis=1), speed.mean(axis=0)

    def trial_time_masking(self, rig, ta, tb) -> tuple[np.ndarray, np.ndarray]:
        trial = TrialSelection.from_rig(rig, self.session)
        mx_a = trial.masking_time(ta)
        mx_b = trial.masking_time(tb)

        return mx_a, mx_b

    @staticmethod
    def plot_corr_trial_averaged(ax: Axes,
                                 x: np.ndarray,
                                 dff: np.ndarray,
                                 speed: np.ndarray,
                                 title: str = ''):
        ax1 = ax
        ax1.plot(x, dff, 'g-', label='dFF')
        ax1.set_ylabel('norm dFF (%)')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(x, speed, 'gray', label='Speed')
        ax2.set_ylabel('Speed (cm/s)')
        ax2.tick_params(axis='y')

        ax1.set_xlabel('position (cm)')
        ax1.set_title(title)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

    @staticmethod
    def plot_corr_raw(ax: Axes,
                      x: np.ndarray,
                      y: np.ndarray,
                      tx: np.ndarray,
                      ty: np.ndarray,
                      title: str = ''):
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)

        ax.plot(tx, x_norm, 'g-', alpha=0.7, label='dFF (normalized)')
        ax.plot(ty, y_norm, 'gray', alpha=0.7, label='Speed (normalized)')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Signal')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)


def corr_signal_helper(x: np.ndarray,
                       y: np.ndarray,
                       tx: np.ndarray,
                       ty: np.ndarray,
                       interp_method: Literal['a2b', 'b2a'] = 'b2a') -> float:
    """
    This function helps correlate two signals by aligning their time stamps
    and optionally interpolating one signal to match the other. It supports
    two modes of interpolation: 'a2b' and 'b2a'. The function ensures the
    input signal lengths match their respective time arrays and computes
    the Pearson correlation coefficient between the aligned signals.

    :param x: The first signal as a numeric array. Must match the length of tx.
    :param y: The second signal as a numeric array. Must match the length of ty.
    :param tx: Time array corresponding to the first signal.
    :param ty: Time array corresponding to the second signal.
    :param interp_method: Interpolation mode, either 'a2b' to interpolate the
                          first signal to align with the second, or 'b2a' to
                          interpolate the second signal to align with the first.
                          Default is 'b2a'.
    :type interp_method: Literal['a2b', 'b2a']
    :raises ValueError: If the length of x and tx, or y and ty, do not match.
    :raises RuntimeError: If the standard deviation of either aligned signal
                          is zero, making correlation undefined.
    :return: The Pearson correlation coefficient between the aligned signals x
             and y after time alignment and interpolation.
    """
    if len(x) != len(tx):
        raise ValueError("Length of 'a' and 'ta' must match")
    if len(y) != len(ty):
        raise ValueError("Length of 'b' and 'tb' must match")

    match interp_method:
        case 'b2a':
            mx = (tx >= ty[0]) & (tx <= ty[-1])
            tx = tx[mx]
            x = x[mx]
            y = np.interp(tx, ty, y)
        case 'a2b':
            mx = (ty >= tx[0]) & (ty <= tx[-1])
            ty = ty[mx]
            y = y[mx]
            x = np.interp(ty, tx, x)
        case _:
            raise ValueError(f'invalid {interp_method}')

    if np.std(x) == 0 or np.std(y) == 0:
        raise RuntimeError('')

    return np.corrcoef(x, y)[0, 1]


if __name__ == '__main__':
    SpeedScoreOptions().main()
