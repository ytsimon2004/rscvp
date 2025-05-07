from typing import Literal

import numpy as np
from rscvp.util.cli import TreadmillOptions
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.position import PositionBinnedSig

from argclz import AbstractParser, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from stimpyp import RiglogData

__all__ = ['LickProbOptions']


# TODO handle camera lagging legacy issue
class LickProbOptions(AbstractParser, TreadmillOptions, CameraOptions, Dispatch):
    DESCRIPTION = 'Show the lick'

    dispatch_plot: Literal['position', 'peri-reward', 'lickscore'] = argument(
        '--plot', '--plot-type',
        required=True,
        help='which plot type'
    )

    track_type = 'lick'

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    lick_time: np.ndarray = None
    lick_prob: np.ndarray = None

    def run(self):
        rig = self.load_riglog_data()
        self.lick_time, self.lick_prob = self.get_lick_event(rig)
        self.invoke_command(self.dispatch_plot, rig)

    # ======== #
    # Position #
    # ======== #

    @dispatch('position')
    def plot_position_bins(self, rig: RiglogData):
        """lick probability as function of positions bins across trials"""
        dat = self.calc_position_binned_lick_prob(rig, self.lick_prob, self.lick_time)
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
    def calc_position_binned_lick_prob(
            rig: RiglogData,
            signal: np.ndarray,
            tracking_time: np.ndarray,
            bin_range: tuple[int, int, int] = (0, 150, 50),
            lap_range: tuple[int, int] = None
    ) -> np.ndarray:
        """(L, B) position-binned lick probability"""
        pbs = PositionBinnedSig(rig, bin_range=bin_range)

        return pbs.calc_binned_signal(
            tracking_time,
            signal,
            lap_range=lap_range,
            occ_normalize=True,
            smooth=False,
            norm=True
        )

    # =========== #
    # Peri-Reward #
    # =========== #

    @dispatch('peri-reward')
    def plot_peri_reward_raster(self, rig: RiglogData):
        """lick probability raster (binarized) as function of positions bins across trials"""
        pass


if __name__ == '__main__':
    LickProbOptions().main()
