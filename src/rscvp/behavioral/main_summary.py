from argclz import AbstractParser, union_type, str_tuple_type, argument
from neuralib.plot import plot_figure, ax_merge, plot_peri_onset_1d
from rscvp.behavioral.util import *
from rscvp.behavioral.util import check_treadmill_trials
from rscvp.behavioral.util_plot import *
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_stimpy import StimpyOptions
from rscvp.util.cli.cli_suite2p import Suite2pOptions
from rscvp.util.cli.cli_treadmill import TreadmillOptions
from rscvp.util.position import load_interpolated_position
from rscvp.util.util_lick import peri_reward_raster_hist

__all__ = ['BehaviorSumOptions']


class BehaviorSumOptions(AbstractParser, StimpyOptions, Suite2pOptions, TreadmillOptions):
    DESCRIPTION = 'Plot single animal in treadmill behavioral overview'

    session_selection: str | tuple[str, ...] | None = argument(
        '-SL', '--ssl',
        metavar='SESSION',
        type=union_type(str_tuple_type, str),
        default=None,
        help='select single OR multiple behavioral sessions'
    )

    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('bs', output_type='behavior')
        self.behavior_sum_plot(output_info)

    def behavior_sum_plot(self, output: DataOutput):
        """plot for behavioral overview, including running speed, licking information"""
        riglog = self.load_riglog_data()
        interp_pos = load_interpolated_position(riglog)

        if self.session_selection is not None:
            riglog = riglog.with_sessions(self.session_selection)
            t0 = riglog.dat[0, 2] / 1000
            t1 = riglog.dat[-1, 2] / 1000
            interp_pos = interp_pos.with_time_range(t0, t1)

        lick_event = riglog.lick_event
        reward_event = riglog.reward_event
        lap_event = riglog.lap_event

        pos = interp_pos.p
        pos_time = interp_pos.t
        vel = interp_pos.v

        if self.cutoff_vel is not None:
            vel[(vel < self.cutoff_vel)] = 0

        sep = check_treadmill_trials(riglog)

        # running speed heatmap
        m = get_velocity_per_trial(lap_event.time, interp_pos, self.belt_length, self.smooth_vel)

        output_file = output.summary_figure_output(
            self.session_selection if self.session_selection is not None else None
        )

        with plot_figure(output_file, 6, 2, figsize=(8, 12), tight_layout=False) as _ax:
            # vel heatmap
            ax = ax_merge(_ax)[0:3, 0]
            plot_velocity_heatmap(ax, m, sep)

            # velocity line chart
            ax = ax_merge(_ax)[3:4, 0]
            plot_velocity_line(ax, m, show_all_trials=False)

            # peri-reward lick and histogram
            ax = _ax[0, 1]
            lick_per_trial, lick_hist, _ = peri_reward_raster_hist(lick_event.time, reward_event.time, self.psth_sec)
            plot_peri_reward_lick_raster(ax, lick_per_trial, limit=self.psth_sec)
            ax = _ax[1, 1]
            plot_peri_reward_lick_hist(ax, lick_hist, limit=self.psth_sec)

            # peri-reward velocity
            ax = _ax[2, 1]
            plot_peri_onset_1d(reward_event.time, pos_time, vel, pre=self.psth_sec, post=self.psth_sec, ax=ax)

            # lap interval to check dj mice
            ax = _ax[3, 1]
            plot_lap_time_interval(ax, lap_event.time)

            # position
            ax = ax_merge(_ax)[4, :]
            plot_position_and_event_raster(ax, pos_time, pos, lick_event, reward_event, lap_event)

            # velocity
            ax = ax_merge(_ax)[5, :]
            plot_value_time(ax, pos_time, vel, ylabel='Velocity (cm/s)')


if __name__ == '__main__':
    BehaviorSumOptions().main()
