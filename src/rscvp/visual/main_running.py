import numpy as np

from argclz import AbstractParser
from neuralib.plot import plot_figure
from rscvp.util.cli import PersistenceRSPOptions, TreadmillOptions
from rscvp.visual.util_cache import AbstractVisualTuningOptions, VisualTuningCache, plot_visual_pattern_trace
from stimpyp import GratingPattern

__all__ = ['RunningVisualOptions']


class RunningVisualOptions(AbstractParser,
                           AbstractVisualTuningOptions,
                           TreadmillOptions,
                           PersistenceRSPOptions[VisualTuningCache]):
    DESCRIPTION = 'Calculate binned running speed during different sftf stimulation'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.load_cache()

    def empty_cache(self) -> VisualTuningCache:
        return VisualTuningCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            signal_type='speed',
            value_type=self.value_type,
            plane_index='_all',
            direction_invert=self.direction_invert
        )

    def compute_cache(self, cache: VisualTuningCache) -> VisualTuningCache:
        cache.neuron_idx = None
        cache.src_neuron_idx = None
        cache.pre_post = self.pre_post
        self.plot_speed_trace(cache)

        return cache

    def get_frame_rate(self) -> float:
        self.plane_index = 0  # adhoc for non-imaging value
        s2p = self.load_suite_2p()
        return s2p.fs

    def plot_speed_trace(self, cache: VisualTuningCache):
        rig = self.load_riglog_data()
        pos = self.load_position()
        pattern = GratingPattern.of(rig)

        with plot_figure(None) as ax:
            signals = plot_visual_pattern_trace(
                ax,
                pattern,
                pos.v,
                pos.t,
                pre_post=self.pre_post,
                direction_invert=self.direction_invert,
                color=self.line_color,
                sig_type='speed',
                value_type=self.value_type
            )

            stim_pattern = []
            act = []
            stim_indices = []
            pad = 0

            for sig in signals:
                sig = sig.rollback_actual_value()
                stim_pattern.append(tuple([sig.direction, sig.sftf]))
                act.append(sig.signal)
                stim_indices.append([sig.stim_epoch_index(self.pre_post, self.get_frame_rate()) + pad])
                pad += sig.n_frames

        cache.stim_pattern = stim_pattern
        cache.signal = np.array(act)  # (P, F)
        cache.stim_index = np.array(stim_indices)


if __name__ == '__main__':
    RunningVisualOptions().main()
