from pathlib import Path
from typing import Literal, Optional, Final

import numpy as np
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.util_camera import truncate_video_to_pulse
from rscvp.visual.main_reliability import plot_cat_trial
from rscvp.visual.util_cache import AbstractVisualTuningOptions, VisualTuningCache, plot_visual_pattern_trace
from scipy.stats import zscore

from argclz import AbstractParser, argument
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure
from neuralib.util.interp import interp1d_nan
from stimpyp import RiglogData, GratingPattern

__all__ = ['VisualPatternPupilOptions',
           'ApplyVisualPupilOptions']


class VisualPatternPupilOptions(AbstractParser,
                                AbstractVisualTuningOptions,
                                CameraOptions,
                                PersistenceRSPOptions[VisualTuningCache]):
    DESCRIPTION = 'Plot and cache the pupil size in response to different direction'

    plot_type: Literal['preview', 'tuning', 'reliability'] = argument(
        '--plot', '--plot-type',
        required=True,
        help='which plot type'
    )

    sig_cutoff: Optional[float] = argument(
        '--cut',
        default=None,
        help='whether do manual cutoff and interpolation for false tracking'
    )

    rig: RiglogData
    pattern: GratingPattern
    pupil: np.ndarray
    camera_time: np.ndarray

    signal_type: Final[str] = 'pupil'

    def set_attrs(self):
        self.rig = self.load_riglog_data()
        self.pattern = GratingPattern.of(self.rig)
        self.camera_time = self.rig.camera_event['eyecam'].time

        #
        self.track_type = 'pupil'
        pupil = self.load_facemap_result().get_pupil_area()
        pupil = zscore(pupil)

        if self.sig_cutoff is not None:
            pupil[pupil > self.sig_cutoff] = np.nan
            pupil = interp1d_nan(pupil)

        self.pupil = truncate_video_to_pulse(pupil, self.camera_time)

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.set_attrs()

        if self.plot_type == 'preview':
            self.plot_pupil_preview()
        elif self.plot_type == 'tuning':
            self.load_cache()  # TODO if plot again, use --invalid-cache flag
        elif self.plot_type == 'reliability':
            self.plot_pupil_reliability()
        else:
            raise ValueError('')

    def empty_cache(self) -> VisualTuningCache:
        return VisualTuningCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            signal_type=self.signal_type,
            value_type=self.value_type,
            plane_index='Null',
            direction_invert=self.direction_invert
        )

    def compute_cache(self, cache: VisualTuningCache) -> VisualTuningCache:
        cache.neuron_idx = None
        cache.src_neuron_idx = None
        cache.pre_post = self.pre_post
        self.plot_pupil_pattern_tuning(cache)

        return cache

    @property
    def fps(self) -> float:
        return self.rig.camera_event['eyecam'].fps

    @property
    def output_file(self) -> Path:
        output = self.get_data_output('pupil', output_type='behavior')
        return output.summary_figure_output(self.plot_type)

    def plot_pupil_preview(self):
        with plot_figure(self.output_file) as ax:
            ax.plot(self.camera_time, self.pupil)

    def plot_pupil_pattern_tuning(self, cache: VisualTuningCache):
        """Plot pupil area in different sf,tf,dir conditions.
        ** Note that the direction is inverted based on KS's definition"""
        with plot_figure(self.output_file) as ax:
            # noinspection PyTypeChecker
            dat = plot_visual_pattern_trace(
                ax,
                self.pattern,
                self.pupil,
                self.camera_time,
                pre_post=self.pre_post,
                direction_invert=self.direction_invert,
                sig_type=self.signal_type,
                value_type=self.value_type
            )

            patterns = []  # (P,)
            act = []  # (P,)
            stim_indices = []
            pad = 0  # for index accumulate
            for it in dat:
                it = it.rollback_actual_value()
                patterns.append(tuple([it.direction, it.sftf]))
                act.append(it.signal)
                stim_index = it.stim_epoch_index(self.pre_post, self.rig.camera_event['eyecam'].fps) + pad
                stim_indices.append(stim_index)
                pad += it.n_frames

            cache.stim_pattern = patterns,
            cache.dat = np.array(act).flatten()
            cache.stim_index = np.array(stim_indices)

    def plot_pupil_reliability(self):
        """Plot concatenated pupil area in only the visual stimuli epoch across different trials"""
        with plot_figure(self.output_file) as ax:
            plot_cat_trial(
                ax,
                self.pattern,
                self.pupil,
                self.camera_time,
                self.fps
            )


# ====== #

class ApplyVisualPupilOptions(AbstractVisualTuningOptions):
    """Apply VisualTuningCache in pupil activity"""

    def apply_visual_pupil_cache(self, error_when_missing=True) -> VisualTuningCache:
        return get_options_and_cache(VisualPatternPupilOptions, self, error_when_missing)


if __name__ == '__main__':
    VisualPatternPupilOptions().main()
