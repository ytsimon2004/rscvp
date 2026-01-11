from __future__ import annotations

import numpy as np
import scipy
from functools import cached_property
from scipy.interpolate import interp1d
from typing import NamedTuple

from argclz import argument, AbstractParser, int_tuple_type, union_type, as_argument
from neuralib.decoding.position import place_bayes
from neuralib.locomotion import running_mask1d, CircularPosition
from neuralib.persistence import *
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.suite2p import get_neuron_signal, SIGNAL_TYPE, Suite2PResult, sync_s2p_rigevent
from neuralib.util.verbose import fprint
from rscvp.model.bayes_decoding.util import calc_wrap_distance
from rscvp.spatial.main_cache_occ import ApplyPosBinCache, AbstractPosBinActOptions
from rscvp.util.cli.cli_model import ModelOptions, trial_cross_validate
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.position import PositionBinnedSig
from rscvp.util.util_trials import TrialSelection
from stimpyp import RiglogData

__all__ = ['BayesDecodeData',
           'BayesDecodeCache',
           'BayesDecodeCacheBuilder',
           'ApplyBayesDecodeCache']


class BayesDecodeData(NamedTuple):
    """
    `Dimension parameters`:

        N = number of neurons

        T = number of temporal bins

        X = number of spatial bins

        L = number of trials

        P = number of optic planes

    """

    n_neurons: int
    """Number of neurons"""
    fr: np.ndarray
    """Firing rate 2D array. `Array[float, [T, N]]`"""
    fr_time: np.ndarray
    """Firing rate time 1D array. might be discontinuous. `Array[float, T]`"""
    actual_position: np.ndarray
    """Position bin as a function of time in cm. `Array[float, [T, N]]`"""
    rate_map: np.ndarray
    """Firing rate template. `Array[float, [X, N]]`"""
    spatial_bin_size: float
    """Spatial bin size in cm"""
    temporal_bin_size: float
    """Temporal bin size in second"""

    trial_range: tuple[int, int]
    """Trial selected range"""
    trial_index: np.ndarray
    """`Array[float, L]`, 1 for training, 2 for testing, 3 for both. only for selecting behavioral session"""

    posterior: np.ndarray | None = None
    """`Array[float, [T, X]]`"""
    predicted_position: np.ndarray | None = None
    """`Array[float, T]`"""
    decode_error: np.ndarray | None = None
    """`Array[float, T]`"""
    binned_decode_error: np.ndarray | None = None
    """`Array[float, [L, B] | [P, L, B]]`. value in median and sem err."""

    @property
    def binned_error_median(self) -> np.ndarray:
        """Trial average median. `Array[float, B]`"""
        if self.binned_decode_error.ndim == 3:
            ret = np.nanmean(self.binned_decode_error, axis=0)  # average across plane
        else:
            ret = self.binned_decode_error

        return np.nanmedian(ret, axis=0)

    @property
    def binned_error_sem(self) -> np.ndarray:
        """Trial average SEM. `Array[float, B]`"""
        if self.binned_decode_error.ndim == 3:
            ret = np.nanmean(self.binned_decode_error, axis=0)  # average across plane
        else:
            ret = self.binned_decode_error

        return np.asarray(scipy.stats.sem(ret, axis=0, nan_policy='omit'))

    def get_test_trial(self) -> np.ndarray:
        """Get test trial indices"""
        return np.nonzero(self.trial_index == 2)[0]


@persistence.persistence_class
class BayesDecodeCache(ETLConcatable):
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    # false validator for concat plane
    plane_index: int | str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    session: str = persistence.field(validator=True, filename=True)
    cv_info: str = persistence.field(validator=True, filename=True)
    bins: int = persistence.field(validator=True, filename=True, filename_prefix='bins_')
    signal_type: SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    selection: str = persistence.field(validator=True, filename=False)
    run_epoch: bool = persistence.field(validator=True, filename=False)
    random: str | None = persistence.field(validator=False, filename=False)
    use_virtual_space: bool = persistence.field(validator=True, filename=True, filename_prefix='vr_')
    version: int = persistence.autoinc_field(filename_prefix='#')

    #
    neuron_idx: np.ndarray
    """(N')"""
    src_neuron_idx: np.ndarray
    """(N'), value domain in plane_idx"""
    trial_range: tuple[int, int]
    """(L1, L2) range"""
    trial_index: np.ndarray
    """(L,), 1 for training, 2 for testing, 3 for both"""

    fr: np.ndarray
    """(T, N)"""
    fr_time: np.ndarray
    """(T,)"""
    actual_position: np.ndarray
    """(T,)"""
    rate_map: np.ndarray
    """(X, N)"""
    spatial_bin_size: float
    """cm"""
    temporal_bin_size: float
    """sec"""

    posterior: np.ndarray
    """(T, X)"""
    predicted_position: np.ndarray
    """(T,)"""
    decode_error: np.ndarray
    """(T,)"""

    binned_decode_error: np.ndarray
    """`Array[float, [L, B] | [P, L, B]]`"""

    def __len__(self) -> int:
        """numbers of discontinuous time bin"""
        return len(self.fr_time)

    def load_result(self) -> BayesDecodeData:
        # noinspection PyTypeChecker
        return BayesDecodeData(
            n_neurons=len(self.neuron_idx),
            fr=self.fr,
            fr_time=self.fr_time,
            actual_position=self.actual_position,
            rate_map=self.rate_map,
            spatial_bin_size=self.spatial_bin_size,
            temporal_bin_size=self.temporal_bin_size,
            trial_range=self.trial_range,
            trial_index=self.trial_index,
            posterior=self.posterior,
            predicted_position=self.predicted_position,
            decode_error=self.decode_error,
            binned_decode_error=self.binned_decode_error
        )

    def get_signal(self, opt: AbstractBayesDecodeOptions,
                   time: np.ndarray,
                   value: np.ndarray,
                   position_down_sampling: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Get behavioral event time and signal

        :param opt: ``AbstractBayesDecodeOptions``
        :param time: Signal time array. `Array[float, V]`
        :param value: Signal value array. `Array[float, V]`
        :param position_down_sampling: Whether down sample the position data to imaging(physiological) data
        :return: tuple of time and value array
        """
        rig = opt.load_riglog_data()
        s2p = opt.load_suite_2p()

        #
        trial = TrialSelection.from_rig(rig, self.session, use_virtual_space=opt.use_virtual_space)
        start_t = trial.start_time
        end_t = trial.end_time

        #
        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, opt.plane_index)
        image_mask = np.logical_and(start_t < image_time, image_time < end_t)
        image_time = image_time[image_mask]

        #
        pos = opt.load_position()

        match self.run_epoch, position_down_sampling:
            case (True, True):
                x = np.logical_and(start_t < pos.t, pos.t < end_t)
                pt = pos.t[x]
                pv = pos.v[x]

                # down-sampling
                vv = interp1d(pt, pv, bounds_error=False, fill_value=0)(image_time)
                signal = interp1d(time, value, bounds_error=False, fill_value=0)(image_time)

                run_mask = running_mask1d(image_time, vv)
                signal_time = image_time[run_mask]
                signal = signal[run_mask]

            case _:
                raise NotImplementedError('')

        return signal_time, signal

    @classmethod
    def concat_etl(cls, data: list[BayesDecodeCache]) -> BayesDecodeCache:
        validate_concat_etl_persistence(data, ('exp_date', 'animal', 'session', 'signal_type'))

        n_planes = len(data)

        ret: BayesDecodeCache = BayesDecodeCache(
            exp_date=data[0].exp_date,
            animal=data[0].animal,
            plane_index='_concat',
            session=data[0].session,
            cv_info=data[0].cv_info,
            bins=data[0].bins,
            signal_type=data[0].signal_type,
            selection=data[0].selection,
            run_epoch=data[0].run_epoch,
            random=None,
            use_virtual_space=data[0].use_virtual_space
        )

        ret.trial_range = data[0].trial_range
        ret.trial_index = data[0].trial_index

        ret.neuron_idx = np.concatenate([data[i].neuron_idx for i in range(n_planes)])
        ret.src_neuron_idx = np.concatenate([data[i].src_neuron_idx for i in range(n_planes)])

        # different T domain, not able to concat
        ret.fr = None
        ret.fr_time = None
        ret.actual_position = np.concatenate([data[i].actual_position for i in range(n_planes)])
        ret.predicted_position = np.concatenate([data[i].predicted_position for i in range(n_planes)])
        ret.posterior = None

        ret.rate_map = np.hstack([data[i].rate_map for i in range(n_planes)])
        ret.spatial_bin_size = data[0].spatial_bin_size
        ret.temporal_bin_size = data[0].temporal_bin_size

        # accumulate (sum(Tp) for p in P). i.e., for calculating median decoding error
        ret.decode_error = np.concatenate([data[i].decode_error for i in range(n_planes)])

        # (L, B) -> (P, L, B)
        ret.binned_decode_error = np.stack([data[i].binned_decode_error for i in range(n_planes)])

        return ret


class AbstractBayesDecodeOptions(SelectionOptions, ModelOptions):
    spatial_bin_size: float | None = argument(
        '--spatial-bin',
        metavar='VALUE',
        type=float,
        default=None,
        help='spatial bin size in cm',
    )

    temporal_bin_size: float | None = argument(
        '--temporal-bin',
        metavar='VALUE',
        type=float,
        default=None,
        help='temporal bin size in second',
    )

    keep_position_dim: bool = argument(
        '--keep-pos-dim',
        action='store_true',
        help='whether keep the position sample shape. '
             'If not, down-sampling the temporal resolution of position parameters to the same dim as fr',
    )

    cache_version: int | tuple = argument(
        '--load', '--load-version',
        type=union_type(int, int_tuple_type),
        metavar='VERSION',
        help='load which version of bayes cache'
    )

    pre_selection = True
    running_epoch = True
    reuse_output = True


class BayesDecodeCacheBuilder(AbstractParser,
                              AbstractBayesDecodeOptions,
                              ApplyPosBinCache,
                              PersistenceRSPOptions[BayesDecodeCache]):
    DESCRIPTION = "Generate the cache for bayes decoding animal's position, used for further analysis/plotting"

    random_iter: int = argument(
        '--random-iter',
        default=1,
        help='loop multiple times for fixed amount of randomized neurons. ONLY specify if not CV'
    )

    rig: RiglogData
    s2p: Suite2PResult

    image_time: np.ndarray
    image_mask: np.ndarray
    pos: CircularPosition

    trial_selection: TrialSelection
    train_test_list: list[tuple[TrialSelection, TrialSelection]]
    _current_train_test_index: int = 0
    number_iter: int | None = None

    def setattr(self):
        self.rig = self.load_riglog_data()
        self.s2p = self.load_suite_2p()

        #
        if self.spatial_bin_size is None:
            self.spatial_bin_size = self.track_length / self.pos_bins
            fprint(f'spatial bins using: {self.spatial_bin_size} cm')
        self.pos_bins = int(self.track_length / self.spatial_bin_size)
        fprint(f'number of position bins using: {self.pos_bins}')

        #
        if self.temporal_bin_size is None:
            self.temporal_bin_size = (1 / self.s2p.fs)
        else:
            raise NotImplementedError('check further for fr in different temporal bin size')

        #
        self._prepare_image_time()
        self.pos = self.load_position()

        #
        self.trial_selection = TrialSelection.from_rig(self.rig, self.session, use_virtual_space=self.use_virtual_space)
        self.train_test_list = trial_cross_validate(self.trial_selection, self.cross_validation, self.train_fraction)

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        match self.cross_validation:
            case int(cv) if cv > 0:
                self.number_iter = cv
            case str():
                self.number_iter = 1
            case 0:
                self.number_iter = self.random_iter
            case _:
                raise RuntimeError(f'cv invalid: {self.cross_validation=}')
        #
        for i in range(self.number_iter):
            self._current_train_test_index = i
            self.load_cache()

    @property
    def train_test(self) -> tuple[TrialSelection, TrialSelection]:
        sz = len(self.train_test_list)
        return self.train_test_list[self._current_train_test_index % sz]

    def _prepare_image_time(self):
        trial = TrialSelection.from_rig(self.rig, self.session, use_virtual_space=self.use_virtual_space)
        start_time = trial.start_time
        end_time = trial.end_time

        image_time = self.rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, self.s2p, self.plane_index)
        self.image_mask = np.logical_and(start_time < image_time, image_time < end_time)
        self.image_time = image_time[self.image_mask]

    def empty_cache(self) -> BayesDecodeCache:
        return BayesDecodeCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            plane_index=self.plane_index,
            session=self.session,
            cv_info=self.cv_info,
            bins=self.pos_bins,
            signal_type=self.signal_type,
            selection=self.selection_prefix(),
            run_epoch=self.running_epoch,
            random=self.n_selected_neurons if self.random else 'None',
            use_virtual_space=self.use_virtual_space,
            version=self.cache_version
        )

    @property
    def select_neurons_handler(self) -> np.ndarray:
        return self.selected_neurons if self.cross_validation == 0 else self._selected_neurons

    @cached_property
    def _selected_neurons(self) -> np.ndarray:
        """keep the same neuronal population (seeding) during cv"""
        return self.selected_neurons

    @property
    def n_selected_neurons(self) -> int:
        return len(self._selected_neurons)

    def compute_cache(self, cache: BayesDecodeCache) -> BayesDecodeCache:
        self.setattr()

        cache.neuron_idx = self.select_neurons_handler
        cache.src_neuron_idx = self.get_neuron_plane_idx(self.n_selected_neurons, self.plane_index)

        data = self.prepare_decode_data(cache.neuron_idx)

        cache.fr = data.fr
        cache.fr_time = data.fr_time
        cache.actual_position = data.actual_position
        cache.rate_map = data.rate_map
        cache.spatial_bin_size = data.spatial_bin_size
        cache.temporal_bin_size = data.temporal_bin_size
        cache.trial_range = data.trial_range
        cache.trial_index = data.trial_index

        pr = place_bayes(data.fr, data.rate_map, data.spatial_bin_size)
        predicted_position = np.argmax(pr, axis=1) * data.spatial_bin_size
        cache.posterior = pr
        cache.predicted_position = predicted_position

        cache.decode_error = calc_wrap_distance(predicted_position, data.actual_position, upper_bound=self.track_length)
        cache.binned_decode_error = self.calc_position_binned_error(data, predicted_position)

        return cache

    # ==================== #
    # Prepare Decode Input #
    # ==================== #

    def prepare_decode_data(self, neuron_list: np.ndarray) -> BayesDecodeData:
        """
        prepare the input data for the decoding analysis

        X = n_spatial_bins
        T = n_temporal_bins_image
        PT = n_temporal_bins_position
        N = n_neurons

        :return: BayesDecodeData
        """
        normalize = True if self.act_normalized == 'local' else False
        # (N, T)
        fr = get_neuron_signal(self.s2p, neuron_list,
                               signal_type=self.signal_type,
                               normalize=normalize)[0][:, self.image_mask]

        index = self.trial_selection.trial_range
        start_time = self.trial_selection.start_time
        end_time = self.trial_selection.end_time

        # speed filter
        if self.running_epoch:
            fr, fr_time, position = self._prepare_running_epoch(self.pos, start_time, end_time, self.image_time, fr)
        else:
            fr_time = self.image_time  # (T,)
            position = self.pos.p

        train, test = self.train_test

        # rate_map cache apply
        rate_map = self.get_occ_cache().occ_activity  # (N, L, X)
        rate_map = train.masking_trial_matrix(rate_map, 1)  # (N, L', X)
        rate_map = np.nanmean(rate_map, axis=1)[neuron_list]  # trial average  (N, X)

        #
        print(f'{rate_map.shape=}, {int(self.track_length / self.spatial_bin_size)}')
        if rate_map.shape[1] != int(self.track_length / self.spatial_bin_size):
            raise RuntimeError('spatial bin size inconsistent in rate_map and decoding parameter, '
                               'spatial bin size should be:'
                               f'{self.track_length / rate_map.shape[1]}')

        assert train.trial_range == index
        trial_index = np.zeros((index[1] - index[0]), dtype=int)  # (L')

        trial_index[train.selected_trials - index[0]] += 1  # train
        trial_index[test.selected_trials - index[0]] += 2  # test

        # test set
        t_mask = test.masking_time(fr_time)
        fr = fr[:, t_mask]
        fr_time = fr_time[t_mask]
        actual_pos = position[t_mask]

        return BayesDecodeData(
            n_neurons=fr.shape[0],
            fr=fr.T,
            fr_time=fr_time,
            actual_position=actual_pos,
            rate_map=rate_map.T,
            spatial_bin_size=self.spatial_bin_size,
            temporal_bin_size=self.temporal_bin_size,
            trial_range=index,
            trial_index=trial_index,
        )

    def _prepare_running_epoch(self,
                               pos: CircularPosition,
                               start_time: float,
                               end_time: float,
                               image_time: np.ndarray,
                               fr: np.ndarray) -> tuple[np.ndarray, ...]:
        """prepare activity only in the running epoch"""
        position_mask = np.logical_and(start_time < pos.t, pos.t < end_time)
        position_time = pos.t[position_mask]
        position = pos.p[position_mask]
        velocity = pos.v[position_mask]

        if not self.keep_position_dim:  # down-sampling of position to imaging
            interp_vel = interp1d(position_time, velocity, bounds_error=False, fill_value=0)(image_time)
            interp_pos = interp1d(position_time, position, bounds_error=False, fill_value=0)(image_time)

            run_mask = running_mask1d(image_time, interp_vel)  # (T,)
            fr = fr[:, run_mask]  # (N, T')
            fr_time = image_time[run_mask]  # (T')
            position = interp_pos[run_mask]  # (T')

        else:
            run_mask = running_mask1d(position_time, velocity)  # (PT', )
            fr_time = position_time[run_mask]  # (PT', )
            fr = interp1d(image_time,
                          fr,
                          axis=fr.ndim - 1,
                          bounds_error=False,
                          fill_value=0.0)(fr_time)
            position = position[run_mask]  # (PT')

        return fr, fr_time, position

    # =========================== #
    # Compute Binned Decode Error #
    # =========================== #

    def err_to_accuracy(self, err: float) -> float:
        max_err = self.track_length / 2
        return np.abs(err - max_err) / max_err * 100

    def calc_position_binned_error(self, result: BayesDecodeData, predicted_position: np.ndarray) -> np.ndarray:
        """
        :return: Position-binned decoding error across trials. `Array[float, [L, B]]`
        """
        rig = self.load_riglog_data()
        pbs = PositionBinnedSig(
            rig,
            bin_range=(0, self.track_length, self.pos_bins),
            position_sample_rate=self.position_sampling_rate,
            use_virtual_space=self.use_virtual_space,
            position_cache_file=self.position_cache
        )

        # trial range for model testing
        if self.cross_validation != 0:
            trial_test = np.nonzero(result.trial_index == 2)[0]
        else:
            trial_test = result.trial_range

        binned_actual_pos = pbs.calc_binned_signal(
            result.fr_time,
            result.actual_position,
            trial_test,
            running_epoch=self.running_epoch
        )
        binned_predicted_pos = pbs.calc_binned_signal(
            result.fr_time,
            predicted_position,
            trial_test,
            running_epoch=self.running_epoch
        )

        n_laps = binned_actual_pos.shape[0]
        ret = np.zeros_like(binned_actual_pos)
        for lap in range(n_laps):
            ret[lap] = calc_wrap_distance(binned_actual_pos[lap], binned_predicted_pos[lap])

        return ret


class ApplyBayesDecodeCache(AbstractBayesDecodeOptions, AbstractPosBinActOptions):
    cache_version = as_argument(AbstractBayesDecodeOptions.cache_version).with_options(required=True)

    def get_decoding_cache(self, version: int | tuple = None, error_when_missing=False) -> BayesDecodeCache:
        if self.plane_index is not None:
            if not isinstance(version, int):
                raise TypeError('for single plane, cache --load must be int type')
            return self._get_cache_single(version, error_when_missing)
        else:
            return self._get_cache_concat()

    def _get_cache_single(self, version: int, error_when_missing=False) -> BayesDecodeCache:
        if version is None:
            version = self.cache_version

        return get_options_and_cache(BayesDecodeCacheBuilder, self, error_when_missing, version=version)

    def _get_cache_concat(self, error_when_missing=False) -> BayesDecodeCache:
        data = []

        try:
            n_planes = self.load_suite_2p().n_plane
        except FileNotFoundError as e:
            n_planes = 4
            fprint(repr(e) + f' -> use n_planes: {n_planes}', vtype='warning')

        version = self._get_version_concat(n_planes)
        for i in range(n_planes):
            cache = get_options_and_cache(BayesDecodeCacheBuilder, self, error_when_missing,
                                          plane_index=i, version=version[i])
            data.append(cache)

        ret = BayesDecodeCache.concat_etl(data)
        return ret

    def _get_version_concat(self, n_planes: int) -> tuple[int, ...]:
        """versions for cache files in concat data"""
        match self.cache_version:
            case int():
                return (self.cache_version,) * n_planes
            case tuple():
                return self.cache_version
            case _:
                raise TypeError(f'{type(self.cache_version)}')


if __name__ == '__main__':
    BayesDecodeCacheBuilder().main()
