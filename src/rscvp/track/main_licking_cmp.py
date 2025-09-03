import numpy as np
from matplotlib.axes import Axes

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import fprint
from rscvp.util.cli import Suite2pOptions, DataOutput
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.util_lick import peri_reward_raster_hist, LickTracker

__all__ = ['LickingCmpOptions']


class LickingCmpOptions(AbstractParser, CameraOptions, Suite2pOptions):
    DESCRIPTION = 'Licking comparison between video tracking and electrical sensing'

    limit: float = argument(
        '-w', '--window',
        metavar='SEC',
        default=5.0,
        help='window size for peri-event plotting',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('lick', output_type='behavior')
        self.licking_cmp(output_info)

    def licking_cmp(self, output: DataOutput):
        """using MiceLick output.npy to validate the electrical sensing lick detector
        make comparison between them and do the plot. cross-correlation, peri-reward and histogram

        ** NOTE: After 210925, lag issue solved due to the labcam version

        .. seealso::

            :func:`code2prig.behavior_summary.licking_cmp`

        :param self:
        :param output:
        :return:
        """
        riglog = self.load_riglog_data()
        reward_time = riglog.reward_event.time

        track = self.load_lick_tracker()

        lick_per_trial_v, hist_vt, _ = peri_reward_raster_hist(
            track.prob_to_event().time,
            reward_time,
            self.limit
        )

        lick_per_trial, hist_lt, _ = peri_reward_raster_hist(track.electrical_timestamp, reward_time, self.limit)

        with plot_figure(output.summary_figure_output(), 3, 1, figsize=(10, 10)) as ax:
            if track.offset_flag:
                plot_corr_lag(ax[0], track)
            else:
                fprint('Camera-based lick without offset', vtype='warning')

            plot_lick_hist(ax[1], self, hist_lt, hist_vt)
            plot_lick_raster(ax[2], self, lick_per_trial, lick_per_trial_v)


def plot_corr_lag(ax: Axes, tracker: LickTracker):
    """
    plot the cross correlation, and find the time lag between *cam riglog* and *facecam*
    negative lag value represents the delay of facecam signal (riglog has to do the subtraction)
    """
    corr = tracker.signal_corr
    ax.plot(corr.lag_value, corr.corr)
    ax.axvline(corr.lag, ls='--', color='r')
    ax.set_xlabel('lag(s)')
    ax.set_ylabel('correlation')
    ax.set_xlim(-5, 5)
    ax.set_title(f'max_correlation on lag(s): {corr.lag}\n'
                 f'offset: {tracker.offset_flag}')


def plot_lick_hist(ax: Axes,
                   opt: LickingCmpOptions,
                   hist: np.ndarray,
                   hist_video: np.ndarray):
    """peri-reward(laps) lick percentage """
    ax.hist(np.linspace(-opt.limit, opt.limit, 100), 100,
            weights=hist, label='electrical', color='k', alpha=0.5)
    ax.hist(np.linspace(-opt.limit, opt.limit, 100), 100,
            weights=hist_video, label='video tracking', color='r', alpha=0.5)
    ax.set_ylabel('Count(%)')
    ax.set_xlim(-opt.limit, opt.limit)
    ax.axvline(0, color='k', ls='--', zorder=1)
    ax.legend()


def plot_lick_raster(ax: Axes,
                     opt: LickingCmpOptions,
                     lick_per_trial: list[np.ndarray],
                     lick_per_trial_video: list[np.ndarray]):
    """
    peri-reward(laps) lick time points
    :param ax:
    :param opt:
    :param lick_per_trial: len of list equal to trial numbers, len of array equal to lick numbers
    :param lick_per_trial_video: same results but from 'VIDEO'
    :return:
    """
    ax.eventplot(lick_per_trial, color='k', alpha=0.5)
    ax.eventplot(lick_per_trial_video, color='r', alpha=0.5)
    ax.set_xlabel('Time relative to reward (s)')
    ax.set_ylabel('Trial #')
    ax.set_xlim(-opt.limit, opt.limit)
    ax.axvline(0, color='k', ls='--', zorder=1)


if __name__ == '__main__':
    LickingCmpOptions().main()
