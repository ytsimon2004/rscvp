import itertools

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy import median

from argclz import AbstractParser
from neuralib.plot import plot_figure, ax_merge
from rscvp.util.cli import TreadmillOptions, Suite2pOptions, DataOutput
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.position import load_interpolated_position
from rscvp.util.util_lick import calc_lick_pos_trial, peri_reward_transformation, LickingPosition

__all__ = ['LickScoreOptions']


class LickScoreOptions(AbstractParser, CameraOptions, Suite2pOptions, TreadmillOptions):
    DESCRIPTION = 'licking precision in certain vop behavioral task'

    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('lick', output_type='behavior')
        self.lick_precision(output_info)

    def lick_precision(self, output: DataOutput):
        riglog = self.load_riglog_data()

        if self.lick_event_source == 'lickmeter':
            lick_time = riglog.lick_event.time
        elif self.lick_event_source == 'facecam':
            track = self.load_lick_tracker(riglog)
            lick_time = track.prob_to_event().time
        else:
            raise ValueError('specify correct lick event signal source')

        #
        interp_pos = load_interpolated_position(
            riglog,
            use_virtual_space=self.use_virtual_space,
            norm_length=self.track_length
        )

        lickscore = calc_lick_pos_trial(interp_pos, riglog.lap_event.time, lick_time)

        output_file = output.summary_figure_output()
        with plot_figure(output_file,
                         1, 3,
                         figsize=(12, 5)) as _ax:

            ax = ax_merge(_ax)[:2]
            self.plot_lick_raster_hist(ax, lickscore.lick_position, lickscore.boundary_limit)

            ax = ax_merge(_ax)[2:]
            self.plot_lick_loc_ci(ax, lickscore)

    def plot_lick_raster_hist(self,
                              ax: Axes,
                              data: list[np.ndarray],
                              limit: float):
        """plot lick raster and histogram as a function of location"""
        if self.track_landmarks is not None:
            for i, c in enumerate(self.track_landmarks):
                cc = peri_reward_transformation(c, limit)
                ax.axvspan(cc - 2.5, cc + 2.5, facecolor='y', alpha=0.6, label=f'tac. cue_{i}')

        ax.eventplot(data, color='k', alpha=0.5)

        _ax = ax.twinx()
        d = list(itertools.chain(*data))
        _ax.hist(d, 30, color='c', alpha=0.5, edgecolor='b')
        _ax.set_ylabel('# Numbers')

        ax.set_xlim(-limit, limit)
        ax.set_xlabel('position to reward')
        ax.set_ylabel('# Trials')
        ax.axvline(0, color='r', ls='--', zorder=1, label='reward')

        ax.legend()

    @staticmethod
    def plot_lick_loc_ci(ax: Axes, lick_pos: LickingPosition):
        sns.pointplot(ax=ax, data=lick_pos.anticipatory_lick_loc, estimator=median, errorbar=('ci', 95))
        ax.set_title(f'number of laps show anticipatory licking: {len(lick_pos.anticipatory_lick_loc)}'
                     f'\n range: {lick_pos.as_range()}')
        ax.set(xlabel='# trials', ylabel='First lick location')


if __name__ == '__main__':
    LickScoreOptions().main()
