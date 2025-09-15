import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import interp1d

from argclz import AbstractParser, argument
from neuralib.locomotion import CircularPosition
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import TreadmillOptions
from stimpyp import RiglogData

__all__ = ['LinearVRTaskOptions']


@publish_annotation('sup', project='rscvp', as_doc=True)
class LinearVRTaskOptions(AbstractParser, TreadmillOptions):
    DESCRIPTION = 'Plot lick raster in the linear VR task'

    pre_reward: float = argument(
        '--pre',
        default=0,
        help='pre-reward time (s) for reward zone plot',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.plot_concat()

    def plot_concat(self):
        """Plot two lick rasters side by side: position space (left) + reward zone time (right)"""

        rig = self.load_riglog_data()
        pos = self.load_position()

        with plot_figure(None, 1, 2, gridspec_kw={'width_ratios': [2, 1]}) as ax:
            self.plot_position_space(ax[0], rig, pos)
            self.plot_reward_zone(ax[1], rig)

    def plot_position_space(self, ax: Axes, rig: RiglogData, pos: CircularPosition):
        """Plot lick raster in position space with x=position, y=trial number"""
        lap = self.get_lap_event(rig)
        lick = rig.lick_event
        lick_positions = interp1d(pos.t, pos.p, bounds_error=False, fill_value=np.nan)(lick.time)

        # remove licks outside position range
        valid = ~np.isnan(lick_positions)
        lick_times = lick.time[valid]
        lick_positions = lick_positions[valid]

        trial_numbers = []
        lick_trial_positions = []

        n_trials = len(lap.time) - 1
        for trial in range(n_trials):
            t_start = lap.time[trial]
            t_end = lap.time[trial + 1]

            # Find licks in this trial
            trial_mask = (lick_times >= t_start) & (lick_times < t_end)
            trial_licks = lick_positions[trial_mask]

            # Add trial number for each lick
            trial_numbers.extend([trial] * len(trial_licks))
            lick_trial_positions.extend(trial_licks)

        if len(trial_numbers) > 0:
            ax.scatter(lick_trial_positions, trial_numbers, s=5, alpha=0.7, c='black', marker='|')

        # Add landmarks if available
        if self.track_landmarks:
            for landmark in self.track_landmarks:
                ax.axvline(landmark, color='red', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Trial Number')
        ax.set_title('Lick Raster - Position Space')
        ax.set_xlim(0, self.track_length)
        ax.set_ylim(-0.5, n_trials - 0.5)

    def plot_reward_zone(self, ax: Axes, rig: RiglogData):
        """Plot lick raster aligned to reward events with x=post-reward time, y=trial number"""
        reward = rig.reward_event
        lick = rig.lick_event

        pre = self.pre_reward
        post = rig.get_protocol().reward_duration

        trial_numbers = []
        relative_lick_times = []

        n_trials = len(reward.time)
        for trial in range(n_trials):
            reward_time = reward.time[trial]

            # find licks in the time window around this reward
            lick_mask = (lick.time >= reward_time - pre) & (lick.time <= reward_time + post)
            trial_licks = lick.time[lick_mask]

            # to relative time (reward = 0)
            relative_times = trial_licks - reward_time

            trial_numbers.extend([trial] * len(relative_times))
            relative_lick_times.extend(relative_times)

        if len(trial_numbers) > 0:
            ax.scatter(relative_lick_times, trial_numbers, s=3, alpha=0.7, c='black', marker='|')

        ax.axvspan(0, post, alpha=0.2, color='red', label='Reward Duration')

        ax.set_xlabel('Time relative to reward onset (s)')
        ax.set_title('Lick Raster - Reward Zone')
        ax.set_xlim(-pre, post)
        ax.set_ylim(-0.5, n_trials - 0.5)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)


if __name__ == '__main__':
    LinearVRTaskOptions().main()
