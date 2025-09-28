import numpy as np

from argclz import AbstractParser
from neuralib.locomotion import CircularPosition
from neuralib.plot import plot_figure
from rscvp.behavioral.util import get_binned_velocity
from rscvp.behavioral.util_plot import plot_velocity_heatmap
from rscvp.util.cli import TreadmillOptions
from stimpyp import RiglogData

__all__ = ['VRRunningOptions']


class VRRunningOptions(AbstractParser, TreadmillOptions):
    DESCRIPTION = 'Plot the running speed in the linear VR task (close + open loop)'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        rig = self.load_riglog_data()

        close_act = self.get_running_closeloop(rig)
        open_act = self.get_running_openloop(rig)
        act = np.vstack([close_act, open_act])

        with plot_figure(None) as ax:
            plot_velocity_heatmap(
                ax, act,
                sep=[close_act.shape[0]],
                extent=(0, self.track_length, 0, act.shape[0])
            )

    def get_running_closeloop(self, rig: RiglogData) -> np.ndarray:
        self.position_session = 'close'
        lap = self.get_lap_event(rig)
        mx = lap.time < rig.get_pygame_stimlog().passive_start_time

        pos = self.load_position()
        act = get_binned_velocity(lap.time[mx], pos, track_length=self.track_length, bins=50)

        return act

    def get_running_openloop(self, rig: RiglogData) -> np.ndarray:
        # set encoder based for physical position data
        self.invalid_riglog_cache = True
        self.use_virtual_space = False
        self.position_session = None

        pos = self.load_position()
        stim = rig.get_pygame_stimlog()
        t0 = stim.passive_start_time
        t1 = stim.exp_end_time
        mx = (pos.t >= t0) & (pos.t <= t1)
        v = pos.v[mx]  # actual physical velocity from encoder
        t = pos.t[mx]  # actual time stamps

        return self._openloop_velocity_virtual_space(rig, t, v, t0, t1)

    def _openloop_velocity_virtual_space(self, rig: RiglogData,
                                         actual_time: np.ndarray,
                                         actual_velocity: np.ndarray,
                                         t0: float, t1: float) -> np.ndarray:
        """
        Create 2D binned velocity array using actual physical velocity binned by virtual space position

        :param rig: RiglogData object
        :param actual_time: time stamps for actual velocity data
        :param actual_velocity: actual physical velocity from encoder
        :param t0: start time of open-loop period
        :param t1: end time of open-loop period
        :return: 2D array of binned velocities [laps, spatial_bins]
        """
        # virtual space
        self.use_virtual_space = True
        self.position_session = 'open'  # Use open-loop session

        virtual_pos = self.load_position()

        # interpolate actual velocity onto virtual position timepoints
        actual_vel_interp = np.interp(virtual_pos.t, actual_time, actual_velocity)

        hybrid_pos = CircularPosition(
            virtual_pos.t,
            virtual_pos.p,
            virtual_pos.d,
            actual_vel_interp,
            virtual_pos.trial_time_index
        )

        lap_events = self.get_lap_event(rig)

        lap_mask = (lap_events.time >= t0) & (lap_events.time <= t1)
        open_loop_laps = lap_events.time[lap_mask]

        if len(open_loop_laps) < 2:
            print(f"Warning: Only {len(open_loop_laps)} lap events found in open-loop period")
            return np.array([])

        binned_vel = get_binned_velocity(
            open_loop_laps,
            hybrid_pos,
            track_length=self.track_length,
            bins=50,
        )

        return binned_vel


if __name__ == '__main__':
    VRRunningOptions().main()
