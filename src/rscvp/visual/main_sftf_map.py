import collections
from typing import Literal, Union, Optional, Final

import numpy as np
from rscvp.util.cli import PersistenceRSPOptions, SelectionOptions
from scipy.interpolate import interp1d
from tqdm import tqdm
from typing_extensions import Self

from argclz import AbstractParser, argument, str_tuple_type
from neuralib.imaging.suite2p import SIGNAL_TYPE, Suite2PResult, sync_s2p_rigevent, get_neuron_signal
from neuralib.persistence import *
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from stimpyp import SFTF, GratingPattern

__all__ = ['VisualSFTFMapCache',
           'ApplyVisualMapOptions',
           'VisualSFTFMapOptions']

_VALUE_TYPE = Literal['mean', 'median']
_DIR_AGG_TYPE = Literal['mean', 'max']


@persistence.persistence_class
class VisualSFTFMapCache(ETLConcatable):
    """
    `Dimension parameters`:

        N = number of neurons

        B = number of temporal bins (include visual off-on)

    """
    exp_date: str = persistence.field(validator=True, filename=True)
    """Experimental date"""
    animal: str = persistence.field(validator=True, filename=True)
    """Animal ID"""
    plane_index: Union[int, str] = persistence.field(validator=False, filename=True, filename_prefix='plane')
    """Optical imaging plane"""
    signal_type: SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    """Signal type"""
    sftf: SFTF = persistence.field(validator=True, filename=True)
    """SFTF group"""
    value_type: _VALUE_TYPE = persistence.field(validator=True, filename=True, filename_prefix='trial_')
    """Trial-averaged(mean) or trial-median"""
    dir_avg_type: _DIR_AGG_TYPE = persistence.field(validator=True, filename=True, filename_prefix='dir_')
    """Whether take averaging or maximal across direction"""

    #
    neuron_idx: np.ndarray
    """Neuron index of neuronal activity. `Array[int, N]`"""
    src_neuron_idx: np.ndarray
    """Source optic plane of neuronal activity. `Array[int, N]`"""
    dat: np.ndarray
    """Visual on-off neural responses. `Array[float [N, B]]`"""

    def load_data(self) -> np.ndarray:
        return self.dat

    @classmethod
    def concat_etl(cls, data: list['VisualSFTFMapCache']) -> Self:
        validate_concat_etl_persistence(data, ('signal_type', 'sftf', 'value_type', 'dir_avg_type'))
        ret = VisualSFTFMapCache(
            exp_date=data[0].exp_date,
            animal=data[0].animal,
            signal_type=data[0].signal_type,
            plane_index='concat',
            sftf=data[0].sftf,
            value_type=data[0].value_type,
            dir_avg_type=data[0].dir_avg_type,
        )

        ret.neuron_idx = np.concatenate([it.neuron_idx for it in data])
        ret.src_idx = np.concatenate([it.src_neuron_idx for it in data])
        ret.dat = np.vstack([it.dat for it in data])

        return ret


class AbstractSFTFMapOptions(SelectionOptions):
    sftf_type: SFTF = argument(
        '--sftf',
        type=str_tuple_type,
        required=True,
        help='which sftf group'
    )

    value_type: _VALUE_TYPE = argument(
        '--VT', '--value-type',
        default='mean',
        help='plot and cache is trial mean or trial median'
    )

    dir_agg_type: _DIR_AGG_TYPE = argument(
        '--dir-agg',
        default='mean',
        help='average or maximal across direction'
    )

    # selection
    pre_selection: Final = True
    vc_selection: Final = 0.3


class VisualSFTFMapOptions(AbstractParser, AbstractSFTFMapOptions, PersistenceRSPOptions[VisualSFTFMapCache]):
    DESCRIPTION = 'Plot the population activity heatmap (N, B) of pre/post visual stimulation windows in given sftf'

    s2p: Suite2PResult
    neuron_list: np.ndarray

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        self.set_attr()
        cache: VisualSFTFMapCache = self.load_cache()
        self.plot_sftf_map(cache.load_data())

    def set_attr(self):
        self.s2p = self.load_suite_2p()
        self.neuron_list = np.nonzero(self.get_selected_neurons())[0]

    def empty_cache(self) -> VisualSFTFMapCache:
        return VisualSFTFMapCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            signal_type=self.signal_type,
            plane_index=self.plane_index,
            value_type=self.value_type,
            dir_avg_type=self.dir_agg_type,
            sftf=self.sftf_type,
        )

    def compute_cache(self, cache: VisualSFTFMapCache) -> VisualSFTFMapCache:
        self.set_attr()
        cache.neuron_idx = self.neuron_list
        cache.src_neuron_idx = self.get_neuron_plane_idx(len(self.neuron_list), self.plane_index)

        cache.dat = self.compute_sftf_map()

        return cache

    def compute_sftf_map(self) -> np.ndarray:
        riglog = self.load_riglog_data()
        image_time = riglog.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, self.s2p, self.plane_index)

        pattern = GratingPattern.of(riglog)

        ret = []
        for neuron_id in tqdm(self.neuron_list, desc='visual_map', unit='neurons', ncols=80):
            signal = get_neuron_signal(self.s2p, neuron_id, signal_type=self.signal_type, normalize=True)[0]
            ext = extract_sftf_pattern(self.sftf_type, pattern, signal, image_time, dir_agg_type=self.dir_agg_type)
            ret.append(ext)

        return np.array(ret)

    @staticmethod
    def plot_sftf_map(data: np.ndarray):
        with plot_figure(None) as ax:
            im = ax.imshow(data, cmap='Reds', vmin=0, interpolation='none')
            insert_colorbar(ax, im)


def extract_sftf_pattern(sftf: SFTF,
                         pattern: GratingPattern,
                         signal: np.ndarray,
                         time: np.ndarray,
                         pre_stim_time: float = 1,
                         nbins: Optional[int] = 120,
                         dir_agg_type: _DIR_AGG_TYPE = 'mean') -> np.ndarray:
    """
    Extract the neural activity from a specific sftf group, aggregate across directions

    :param sftf: ``SFTF``
    :param pattern: ``StimPattern``
    :param signal: neural activity (T,)
    :param time: neural activity time (T,)
    :param pre_stim_time: time in sec for pre-stim epoch (visual-off)
    :param nbins: number of bins for the activity. used for fixed number for persistence concatenation purpose
    :param dir_agg_type: direction aggregate method. {'mean', 'max'}
    :return: (F',)
    """
    target_sf = float(sftf[0])
    target_tf = int(sftf[1])

    #
    dy = collections.defaultdict(list)
    for si, st, sf, tf, dire in pattern.foreach_stimulus():

        on, off = st
        tx_pre = np.logical_and(on - pre_stim_time <= time, on)
        tx = np.logical_and(on - pre_stim_time <= time, time <= off)

        if sf == target_sf and tf == target_tf:
            baseline = np.median(signal[tx_pre])
            sig = signal[tx] - baseline
            dy[(sf, tf, dire)].append(sig)  # *5
        else:
            continue

    if nbins is None:
        nbins = np.count_nonzero(tx)
    xx = np.linspace(0, 1, num=nbins)

    #
    cy = {}  # for average dir: dict[dire, np.ndarray]
    for (_, _, dire), sig in dy.items():

        trials = []
        for s in sig:
            x = np.linspace(0, 1, num=len(s))
            y = interp1d(x, s)(xx)
            trials.append(y)

        trial_avg = np.mean(trials, axis=0)
        cy[dire] = trial_avg

    func = getattr(np, dir_agg_type)

    return func(list(cy.values()), axis=0)


class ApplyVisualMapOptions(AbstractSFTFMapOptions):
    """Apply VisualSFTFMapCache in 2P cellular neural activity"""

    def apply_visual_map_cache(self, error_when_missing=False) -> VisualSFTFMapCache:
        if self.plane_index is not None:
            return self._apply_single_plane(error_when_missing)
        else:
            return self._apply_concat_plane(error_when_missing)

    def _apply_single_plane(self, error_when_missing=False) -> VisualSFTFMapCache:
        return get_options_and_cache(VisualSFTFMapOptions, self, error_when_missing)

    def _apply_concat_plane(self, error_when_missing=False) -> VisualSFTFMapCache:
        n_planes = self.load_suite_2p().n_plane

        etl_dat = []
        for i in range(n_planes):
            cache = get_options_and_cache(VisualSFTFMapOptions, self, error_when_missing, plane_index=i)
            etl_dat.append(cache)

        ret = VisualSFTFMapCache.concat_etl(etl_dat)

        return ret


if __name__ == '__main__':
    VisualSFTFMapOptions().main()
