from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from tqdm import trange

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import get_neuron_signal, SIGNAL_TYPE, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.locomotion import CircularPosition, running_mask1d
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_occ import ApplyPosBinCache
from rscvp.util.cli import DataOutput
from rscvp.util.position import PositionBinnedSig
from rscvp.util.util_trials import TrialSelection
from stimpyp import RiglogData

__all__ = ['SpeedScoreOptions']


@publish_annotation('appendix', project='rscvp', caption='rev')
class SpeedScoreOptions(AbstractParser, ApplyPosBinCache):
    DESCRIPTION = 'Calculate the speed score for each cell (Kropff et al., 2015)'

    plot_type: Literal['time_course', 'pos_binned', 'scatter', 'hist_summary'] = argument(
        '--plot',
        default='scatter',
        help='plot type',
    )

    signal_type: SIGNAL_TYPE = 'df_f'
    reuse_output = True

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.plot_type == 'hist_summary':
            self.reuse_output = True

    def run(self):
        self.post_parsing()
        output = self.get_data_output('sc', self.session, running_epoch=self.running_epoch)

        match self.plot_type:
            case 'hist_summary':
                self.plot_summary_hist(output)
            case 'time_course' | 'pos_binned' | 'scatter':
                self.foreach_speed_score(output)
            case _:
                raise ValueError('')

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

        dff = get_neuron_signal(s2p, signal_type=self.signal_type)[0]
        pos = self.load_position().interp_time(image_time)

        if self.running_epoch:
            mx = running_mask1d(pos.t, pos.v)
            pos = pos.with_run_mask1d()
            dff = dff[:, mx]
            image_time = image_time[mx]

        speed = pos.v
        position_time = pos.t

        trial = TrialSelection.from_rig(rig, self.session, use_virtual_space=self.use_virtual_space)
        mx = trial.masking_time(image_time)
        dff = dff[:, mx]
        image_time = image_time[mx]
        speed = speed[mx]
        position_time = position_time[mx]

        x = np.linspace(0, self.track_length, num=self.pos_bins)

        # csv + plot
        header = 'speed_score'
        if self.running_epoch:
            header += f'_run'

        if self.session is not None:
            header += f'_{self.session}'

        with csv_header(output.csv_output, ['neuron_id', header]) as csv:
            for n in trange(s2p.n_neurons, desc='calculate speed score', unit='neurons', ncols=80):
                xx, yy, score = corr_signal_helper(dff[n], speed, image_time, position_time, interp_method='none')

                with plot_figure(output.figure_output(n)) as ax:
                    title = f'Neuron {n} | Speed score: {score:.2f}'

                    match self.plot_type:
                        case 'time_course':
                            self.plot_corr_time(ax, dff[n], speed, image_time, position_time, title=title)
                        case 'scatter':
                            self.plot_corr_scatter(
                                ax, xx, yy,
                                title=title, xlabel='∆F/F', ylabel='speed (cm/s)'
                            )
                        case 'pos_binned':
                            self.load_binned_data(rig, pos)
                            self.plot_corr_trial_averaged(ax, x, self.binned_dff[n], self.binned_speed, title=title)

                csv(n, score)

    binned_dff = None
    binned_speed = None

    def load_binned_data(self, rig: RiglogData, pos: CircularPosition) -> tuple[np.ndarray, np.ndarray]:
        """Load trial-averaged activity and speed signals"""

        if self.binned_dff is None or self.binned_speed is None:
            indices = (
                TrialSelection
                .from_rig(rig, self.session, use_virtual_space=self.use_virtual_space)
                .get_selected_profile().trial_range
            )
            trial_range = np.arange(*indices)

            dff = self.get_occ_cache().with_trial(trial_range).occ_activity  # shape (N, L, B)

            pbs = PositionBinnedSig(rig, bin_range=(0, self.track_length, self.pos_bins),
                                    use_virtual_space=self.use_virtual_space)
            speed = pbs.calc_binned_signal(pos.t, pos.v, desc='calculate binned speed')
            speed = speed[slice(*indices), :]  # shape (L, B)

            self.binned_dff = dff.mean(axis=1)
            self.binned_speed = speed.mean(axis=0)

        return self.binned_dff, self.binned_speed

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
    def plot_corr_time(ax: Axes,
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

    @staticmethod
    def plot_corr_scatter(ax: Axes,
                          x: np.ndarray,
                          y: np.ndarray,
                          **kwargs):

        ax.scatter(x, y, s=4, alpha=0.7, c='gray', edgecolor='none')
        ax.set(**kwargs)
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


class CorrResult(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    score: float


def corr_signal_helper(x: np.ndarray,
                       y: np.ndarray,
                       tx: np.ndarray,
                       ty: np.ndarray,
                       interp_method: Literal['a2b', 'b2a', 'none'] = 'b2a') -> CorrResult:
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
        case 'none':
            pass
        case _:
            raise ValueError(f'invalid {interp_method}')

    if np.std(x) == 0 or np.std(y) == 0:
        raise RuntimeError('')

    return CorrResult(x, y, float(np.corrcoef(x, y)[0, 1]))


if __name__ == '__main__':
    SpeedScoreOptions().main()
