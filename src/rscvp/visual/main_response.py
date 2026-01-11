import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
from typing_extensions import Self

from argclz import AbstractParser, argument, float_tuple_type
from neuralib.persistence import persistence, ETLConcatable, validate_concat_etl_persistence
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from neuralib.plot.psth import peri_onset_1d
from neuralib.suite2p import get_neuron_signal, sync_s2p_rigevent
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import get_neuron_list, Suite2pOptions, PersistenceRSPOptions, StimpyOptions
from rscvp.util.typing import SIGNAL_TYPE
from stimpyp import RiglogData, GratingPattern, Direction, SFTF

__all__ = ['VisualPatternCache',
           'AbstractPatternResponseOptions',
           'PatternResponseOptions',
           'ApplyPatternResponseCache']


@persistence.persistence_class
class VisualPatternCache(ETLConcatable):
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    plane_index: int | str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    signal_type: SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    sftf: SFTF | None = persistence.field(validator=True, filename=True, filename_prefix='sftf_')
    direction: Direction | None = persistence.field(validator=True, filename=True, filename_prefix='direction_')

    neuron_idx: np.ndarray
    src_neuron_idx: np.ndarray
    data: np.ndarray
    """Array[float, [N, T]]"""
    time: np.ndarray
    """Array[float, T]"""

    @classmethod
    def concat_etl(cls, data: list[Self]) -> Self:
        validate_concat_etl_persistence(data, ('signal_type', 'sftf', 'direction'))

        const = data[0]
        ret = VisualPatternCache(
            exp_date=const.exp_date,
            animal=const.animal,
            plane_index='_concat',
            signal_type=const.signal_type,
            sftf=const.sftf,
            direction=const.direction
        )

        ret.neuron_idx = np.concatenate([it.neuron_idx for it in data])
        ret.src_neuron_idx = np.concatenate([it.src_neuron_idx for it in data])
        ret.data = np.vstack([it.data for it in data])
        ret.time = const.time

        return ret


class AbstractPatternResponseOptions(StimpyOptions, Suite2pOptions):
    EX_PATTERN_GROUP = 'pattern'

    sftf: SFTF | None = argument(
        '--sftf',
        type=float_tuple_type,
        default=None,
        ex_group=EX_PATTERN_GROUP,
        help='the sftf of the visual stimulation'
    )

    direction: Direction | None = argument(
        '--direction',
        default=None,
        ex_group=EX_PATTERN_GROUP,
        help='the direction of the visual stimulation'
    )

    # peri event t domain
    pre = 1
    post = 4


@publish_annotation('appendix', project='rscvp', caption='rev')
class PatternResponseOptions(AbstractParser, AbstractPatternResponseOptions, PersistenceRSPOptions[VisualPatternCache]):
    DESCRIPTION = 'Plot any activity during different stimulus patterns'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        cache = self.load_cache()

        with plot_figure(None, 1, 2) as ax:
            self.plot_heatmap(cache.time, cache.data, ax=ax[0])
            self.delta_histogram(cache.time, cache.data, ax=ax[1])

    def empty_cache(self) -> VisualPatternCache:
        return VisualPatternCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            signal_type=self.signal_type,
            plane_index=self.plane_index,
            sftf=self.sftf,
            direction=self.direction,
        )

    def compute_cache(self, cache: VisualPatternCache) -> VisualPatternCache:
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, self.neuron_id)
        n = len(neuron_list)

        cache.neuron_idx = np.array(neuron_list)
        cache.src_neuron_idx = self.get_neuron_plane_idx(n, self.plane_index)

        # peri event data
        rig = self.load_riglog_data()
        t = self.select_stimulus_segment(rig)

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)
        signal = get_neuron_signal(s2p, neuron_list)[0]

        data = []
        for n in tqdm(neuron_list, desc='response_pattern', unit='neurons', ncols=80):
            val = peri_onset_1d(t, image_time, signal[n], pre=self.pre, post=self.post).mean(axis=0)  # (P, B) -> (B,)
            data.append(val)

        cache.data = np.vstack(data)
        cache.time = np.linspace(-self.pre, self.post, len(cache.data[0]))

        return cache

    def select_stimulus_segment(self, rig: RiglogData) -> np.ndarray:
        match (self.sftf, self.direction):
            case (None, None):
                return rig.get_stimlog().stimulus_segment[:, 0]
            case (_, None):
                ret = np.vstack([
                    grating.time for grating in GratingPattern.of(rig).foreach_stimulus(name=True)
                    if grating.sf == self.sftf[0] and grating.tf == self.sftf[1]]
                )
                return ret[:, 0]
            case (None, _):
                ret = np.vstack([
                    grating.time for grating in GratingPattern.of(rig).foreach_stimulus(name=True)
                    if grating.direction == self.direction]
                )
                return ret[:, 0]

    def plot_heatmap(self, time: np.ndarray, data: np.ndarray, ax: Axes):
        # baseline substraction
        mx = time < 0
        pre_data = data[:, mx]
        baseline = np.median(pre_data, axis=1, keepdims=True)
        data -= baseline

        # sort
        index = np.argsort(np.mean(data, axis=1))
        data = data[index][::-1]

        # norm = TwoSlopeNorm(vmin=(-np.abs(data).max()), vcenter=0, vmax=(np.abs(data).max()))
        norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
        im = ax.imshow(
            data,
            cmap='seismic',
            norm=norm,
            aspect='auto',
            origin='lower',
            extent=(float(time[0]), float(time[-1]), 0, data.shape[0]),
            interpolation='none'
        )
        insert_colorbar(ax, im)

    def delta_histogram(self, time: np.ndarray, data: np.ndarray, ax: Axes):
        mx = time < 0
        baseline = data[:, mx].mean(axis=1)

        stim_mx = np.logical_and(time >= 0, time <= (self.post + self.pre))
        stim = data[:, stim_mx].mean(axis=1)

        delta = stim - baseline
        ax.hist(delta, bins=50, histtype='step', weights=np.ones_like(delta) / len(delta))
        ax.set(xlabel='delta change', ylabel='fraction', xlim=(-0.1, 0.2), ylim=(0, 0.2))


class ApplyPatternResponseCache(AbstractPatternResponseOptions):

    def get_pattern_cache(self):
        if self.plane_index is None:
            return self._get_cache_concat()
        else:
            return self._get_cache_single()

    def _get_cache_single(self, error_when_missing=True):
        return get_options_and_cache(PatternResponseOptions, self, error_when_missing)

    def _get_cache_concat(self, error_when_missing=True) -> VisualPatternCache:
        n_planes = self.load_suite_2p().n_plane

        caches = []
        for i in range(n_planes):
            cache = get_options_and_cache(PatternResponseOptions, self, error_when_missing, plane_index=i)
            caches.append(cache)

        return VisualPatternCache.concat_etl(caches)


if __name__ == '__main__':
    PatternResponseOptions().main()
