from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from rscvp.util.cli import TreadmillOptions
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.position import PositionBinnedSig
from rscvp.util.util_camera import truncate_video_to_pulse

from argclz import AbstractParser, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from neuralib.plot.plot import grid_subplots
from neuralib.tracking.facemap import FaceMapResult
from neuralib.tracking.facemap.plot import plot_cmap_time_series
from neuralib.util.interp import interp1d_nan
from stimpyp import RiglogData

__all__ = ['PupilTrackOptions']


# TODO handle camera lagging legacy issue
class PupilTrackOptions(AbstractParser, TreadmillOptions, CameraOptions, Dispatch):
    DESCRIPTION = "See the pupil location and movement as a function of animal's position"

    dispatch_plot: Literal['location', 'movement', 'area'] = argument(
        '--plot', '--plot-type',
        required=True,
        help='which plot type'
    )

    track_type = 'pupil'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        rig = self.load_riglog_data()
        fmap = self.load_facemap_result()
        self.invoke_command(self.dispatch_plot, fmap, rig)

    # =============== #
    # Movement / Area #
    # =============== #

    @dispatch('area')
    def plot_pupil_area(self, fmap: FaceMapResult, rig: RiglogData):
        """Plot the position binned pupil area across trials"""
        area = fmap.get_pupil_area()
        self._plot_pupil_signals(area, rig)

    @dispatch('movement')
    def plot_pupil_movement(self, fmap: FaceMapResult, rig: RiglogData):
        """Plot the position binned pupil movement across trials"""
        mov = fmap.get_pupil_location_movement()
        self._plot_pupil_signals(mov, rig)

    def _plot_pupil_signals(self, sig: np.ndarray, rig: RiglogData):
        t = rig.camera_event['eyecam'].time
        sig = truncate_video_to_pulse(sig, t)
        sig = interp1d_nan(sig)

        dat = self.calc_position_binned_movement(rig, sig, t)
        n_trials = dat.shape[0]
        with plot_figure(None, tight_layout=False) as ax:
            im = ax.imshow(dat,
                           extent=[0, self.belt_length, 0, n_trials],
                           cmap='cividis',
                           aspect='auto',
                           origin='lower')

            ax.set(xlabel='position (cm)', ylabel='trials')
            cbar = insert_colorbar(ax, im)
            cbar.ax.set_ylabel('Norm. pixel changes')

    @staticmethod
    def calc_position_binned_movement(
            rig: RiglogData,
            signal: np.ndarray,
            tracking_time: np.ndarray,
            bin_range: tuple[int, int, int] = (0, 150, 50),
            lap_range: tuple[int, int] = None
    ) -> np.ndarray:
        """(L, B) position-binned movement/area"""
        pbs = PositionBinnedSig(rig, bin_range=bin_range)

        return pbs.calc_binned_signal(
            tracking_time,
            signal,
            lap_range=lap_range,
            occ_normalize=True,
            smooth=False,
            norm=True
        )

    # ======== #
    # Location #
    # ======== #

    @dispatch('location')
    def plot_pupil_location(self, fmap: FaceMapResult,
                            rig: RiglogData,
                            do_trial_averaged: bool = False,
                            foreach_trial: bool = True):
        """
        Plot the pupil location in 2D space per trial(lap)

        :param fmap: ``FaceMapResult``
        :param rig: ``RiglogData``
        :param do_trial_averaged: if plot the trial averaged or not
        :param foreach_trial: if not plot the trial averaged, whether plot the `com` in same plot or `grid` subplots
        :return:
        """
        loc = fmap.get_pupil_center_of_mass()
        t = rig.camera_event['eyecam'].time
        loc = truncate_video_to_pulse(loc, t)

        data = self.calc_position_binned_location(rig, loc, t, do_trial_averaged=do_trial_averaged)

        if do_trial_averaged:
            with plot_figure(None) as ax:
                plot_cmap_time_series(data[:, 0], data[:, 1], ax=ax, color_bar_label='position bins')
        else:
            #
            if foreach_trial:
                n = data.shape[0]
                imgs_per_row = int(np.sqrt(n))
                grid_subplots(data,
                              imgs_per_row,
                              plot_func=plot_cmap_time_series,
                              dtype='xy',
                              with_color_bar=False,
                              s=1.5)
            else:
                with plot_figure(None) as ax:

                    for trial_data in data:
                        plot_cmap_time_series(trial_data[:, 0], trial_data[:, 1], ax=ax, with_color_bar=False)

                    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=50), cmap='viridis'), ax=ax)
                    cbar.set_label('position bins')
                    ax.set(xlabel='x', ylabel='y')

    @staticmethod
    def calc_position_binned_location(
            rig: RiglogData,
            location: np.ndarray,
            tracking_time: np.ndarray,
            bin_range: tuple[int, int, int] = (0, 150, 50),
            lap_range: tuple[int, int] = None,
            do_trial_averaged: bool = False
    ):
        """
        Trial averaged (B, 2)
        OR
        Non-trial averaged (L, B, 2)
        """
        pbs = PositionBinnedSig(rig, bin_range=bin_range)

        location = location.T  # (2, F)

        # (2, L, B)
        binned_sig = pbs.calc_binned_signal(
            tracking_time,
            location,
            lap_range=lap_range,
            occ_normalize=True,
            smooth=True,
            norm=True,
        )

        if do_trial_averaged:
            return np.mean(binned_sig, axis=1).T
        else:
            return np.moveaxis(binned_sig, [0, 1, 2], [2, 0, 1])


if __name__ == '__main__':
    PupilTrackOptions().main()
