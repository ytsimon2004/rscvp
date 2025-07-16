import numpy as np

from argclz import AbstractParser
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent, Suite2PResult
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import DataOutput, PersistenceRSPOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list
from rscvp.visual.util_cache import AbstractVisualTuningOptions, VisualTuningCache, plot_visual_pattern_trace
from stimpyp import GratingPattern

__all__ = ['VisualTuningOptions',
           'ApplyVisualActOptions']


@publish_annotation('main', project='rscvp', figure='fig.5A & 5E-H lower & fig.S4', as_doc=True)
class VisualTuningOptions(AbstractParser, AbstractVisualTuningOptions, PersistenceRSPOptions[VisualTuningCache]):
    DESCRIPTION = """
    Plot the calcium transients traces across different condition of visual stimulation(12 dir, 2tf, 3sf).
    Also make persistence cache for other analysis usage.
    """

    s2p: Suite2PResult
    neuron_list: list[int]
    output_info: DataOutput

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.set_background()

        self.s2p = self.load_suite_2p()
        self.neuron_list = get_neuron_list(self.s2p, self.neuron_id)
        self.output_info = self.get_data_output('ta')
        self.load_cache()

    def empty_cache(self) -> VisualTuningCache:
        return VisualTuningCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            signal_type=self.signal_type,
            value_type=self.value_type,
            plane_index=self.plane_index,
            direction_invert=self.direction_invert
        )

    def compute_cache(self, cache: VisualTuningCache) -> VisualTuningCache:
        cache.neuron_idx = self.neuron_list
        cache.src_neuron_idx = self.get_neuron_plane_idx(len(self.neuron_list), self.plane_index)
        cache.pre_post = self.pre_post
        self.foreach_visual_trace(cache)

        return cache

    def foreach_visual_trace(self, cache: VisualTuningCache):
        """Plot the calcium transients traces in response to visual stimulation for each neuron"""
        from tqdm import tqdm

        riglog = self.load_riglog_data()
        image_time = riglog.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, self.s2p, self.plane_index)

        pattern = GratingPattern.of(riglog)

        act_all = []
        for n in tqdm(self.neuron_list, desc='visual_trace', unit='neurons', ncols=80):
            signal = get_neuron_signal(self.s2p, n, signal_type=self.signal_type, normalize=False)[0]

            # noinspection PyTypeChecker
            output = self.output_info.figure_output(self.signal_type, n)
            with plot_figure(output) as ax:
                dat = plot_visual_pattern_trace(
                    ax,
                    pattern,
                    signal,
                    image_time,
                    pre_post=self.pre_post,
                    direction_invert=self.direction_invert,
                    color=self.line_color,
                    sig_type=self.signal_type,
                    value_type=self.value_type
                )

                #
                patterns = []  # (P,)
                act = []  # (P,)
                stim_indices = []
                pad = 0  # for index accumulate
                for it in dat:
                    it = it.rollback_actual_value()
                    patterns.append(tuple([it.direction, it.sftf]))
                    act.append(it.signal)
                    stim_index = it.stim_epoch_index(self.pre_post, self.s2p.fs) + pad
                    stim_indices.append(stim_index)
                    pad += it.n_frames

                act_all.append(np.array(act).flatten())

        #
        cache.stim_pattern = patterns
        cache.signal = np.array(act_all)
        cache.stim_index = np.array(stim_indices)


class ApplyVisualActOptions(AbstractVisualTuningOptions):
    """Apply VisualTuningCache in 2P cellular neural activity"""

    def apply_visual_tuning_cache(self, error_when_missing=True) -> VisualTuningCache:
        if self.plane_index is not None:
            return self._apply_single_plane(error_when_missing)
        else:
            return self._apply_concat_plane(error_when_missing)

    def _apply_single_plane(self, error_when_missing=True) -> VisualTuningCache:
        return get_options_and_cache(VisualTuningOptions, self, error_when_missing)

    def _apply_concat_plane(self, error_when_missing=True) -> VisualTuningCache:
        n_planes = self.load_suite_2p().n_plane

        etl_dat = []
        for i in range(n_planes):
            cache = get_options_and_cache(VisualTuningOptions, self, error_when_missing, plane_index=i)
            etl_dat.append(cache)

        ret = VisualTuningCache.concat_etl(etl_dat)

        return ret


if __name__ == '__main__':
    VisualTuningOptions().main()
