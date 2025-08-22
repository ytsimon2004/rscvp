import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from neuralib.locomotion import running_mask1d, interp_pos1d, CircularPosition
from neuralib.util.verbose import fprint
from stimpyp import RiglogData, RigEvent

__all__ = [
    'PositionBinnedSig',
    'load_interpolated_position'
]


class PositionBinnedSig:
    """
    Calculation of Position Binned Signal

    `Dimension parameters`:

        N = number of neurons

        L = number of trials (laps)

        B = number of position bins for each trials

        T = number of sample points for signal acquisition (i.e., neural signal)

        PT = number of sample points for position data acquisition (i.e., Encoder readout)

    """

    def __init__(
            self,
            riglog: RiglogData,
            *,
            bin_range: int | tuple[int, int] | tuple[int, int, int] = (0, 150, 150),
            smooth_kernel: int = 3,
            position_sample_rate: int = 300,
            use_virtual_space: bool = False
    ):
        """
        :param riglog: ``RiglogData``
        :param bin_range: END or tuple of (start, end, number)
        :param smooth_kernel: Smoothing gaussian kernel after binned
        :param position_sample_rate: Position sampling rate for the interpolation
        :param use_virtual_space: If run on virtual environment
        """
        self._riglog = riglog

        match bin_range:
            case int():
                bin_range = (0, bin_range, bin_range)
            case [start, end] | (start, end):
                bin_range = (start, end, end - start)
            case [start, end, num] | (start, end, num):
                pass  # already right format
            case _:
                raise TypeError(f'wrong bin_range type or format, got {bin_range}')

        #
        self.smooth_kernel = smooth_kernel
        self.position_sample_rate = position_sample_rate
        self.use_virtual_space = use_virtual_space

        # running epoch
        self._running_velocity_threshold = 5
        self._running_merge_gap = 0.5
        self._running_duration_threshold = 1

        # cache
        self._lap_event: RigEvent | None = None

        self._bin_range = bin_range
        self._pos_cache: CircularPosition | None = None
        self._run_mask = None
        self._pos_mask_cache: dict[int, np.ndarray] = {}  # lap: pos_mask(bool arr)
        self._occ_map_cache: dict[(int, bool), np.ndarray] = {}  # (lap, running_epoch): occ_map

    @property
    def riglog(self) -> RiglogData:
        """Riglog data"""
        return self._riglog

    @property
    def lap_event(self) -> RigEvent:
        if self._lap_event is None:
            if self.use_virtual_space:
                self._lap_event = self._riglog.get_pygame_stimlog().virtual_lap_event
            else:
                self._lap_event = self._riglog.lap_event

        return self._lap_event

    @property
    def bin_range(self) -> tuple[int, int]:
        """Bin range (start, end). in cm"""
        return self._bin_range[0], self._bin_range[1]

    @property
    def n_bins(self) -> int:
        """Number of bins"""
        return self._bin_range[2]

    @property
    def position_time(self) -> np.ndarray:
        """Position Time"""
        if self._pos_cache is None:
            self._pos_cache = load_interpolated_position(
                self._riglog,
                sample_rate=self.position_sample_rate,
                use_virtual_space=self.use_virtual_space,
                norm_length=self.bin_range[1]
            )
        return self._pos_cache.t

    @property
    def position(self) -> np.ndarray:
        """Position in cm"""
        _ = self.position_time  # trigger getter
        return self._pos_cache.p

    @property
    def velocity(self) -> np.ndarray:
        """Velocity in cm/s"""
        _ = self.position_time
        return self._pos_cache.v

    @property
    def run_mask(self) -> np.ndarray:
        if self._run_mask is None:
            _ = self.position_time
            p = self._pos_cache
            self._run_mask = running_mask1d(
                p.t, p.v,
                threshold=self._running_velocity_threshold,
                merge_gap=self._running_merge_gap,
                minimal_time=self._running_duration_threshold
            )
        return self._run_mask

    def _position_mask(self, lap: int) -> np.ndarray:
        """
        position mask based on initial lap number
        :param lap: 0-base lap index
        :return:
        """
        if lap not in self._pos_mask_cache:
            t = self.position_time
            lap_event = self.lap_event
            t1 = lap_event.time[lap]
            t2 = lap_event.time[lap + 1]
            self._pos_mask_cache[lap] = np.logical_and(t1 < t, t < t2)

        return self._pos_mask_cache[lap]

    def _occupancy_map(self, lap: int,
                       bins: np.ndarray,
                       pos: np.ndarray,
                       running_epoch=False) -> np.ndarray:
        """

        :param lap: 0-base lap_index
        :param bins:
        :param pos:
        :param running_epoch:
        :return:
        """
        key = (lap, running_epoch)
        if key not in self._occ_map_cache:
            self._occ_map_cache[key] = np.histogram(pos, bins)[0]

        return self._occ_map_cache[key]

    def calc_binned_signal(self,
                           t: np.ndarray,
                           signal: np.ndarray,
                           lap_range: tuple[int, int] | np.ndarray | None = None,
                           occ_normalize: bool = True,
                           smooth: bool = False,
                           running_epoch: bool = False,
                           enable_tqdm: bool = False,
                           norm: bool = False,
                           desc: str | None = None) -> np.ndarray:
        """
        Calculate binned signal

        :param t: Time array. `Array[float, T]`
        :param signal: Signal array. `Array[float, [N, T] | T]`
        :param lap_range:
            tuple type: lap (trial) index (0-based) from start to end (excluded)
            array type: trial index (0-based) array. i.e., for trial-wise cross validation
            None: all trials from all recording sessions
        :param occ_normalize: If do occupancy normalize
        :param smooth: If do the gaussian kernel smoothing after binned
        :param running_epoch: If only take running epoch
        :param enable_tqdm: enable tqdm progress bar
        :param norm: whether do the maximal normalization
        :param desc: description in tqdm
        :return: Position binned signal. `Array[float, [N, L, B] | [L, B]]`
        """

        # activity on position time
        act = self._activity(t, signal)
        act_1d = act.ndim == 1
        run_mask = self.run_mask if running_epoch else None
        bins = np.linspace(*self.bin_range, num=self.n_bins + 1, endpoint=True)

        if lap_range is None or isinstance(lap_range, tuple):

            if lap_range is None:
                lap_index = self.lap_event.value_index
                lap_range = lap_index[0], lap_index[-1]

            n_lap = lap_range[1] - lap_range[0] + 1
            enum = enumerate(range(*lap_range))

        elif isinstance(lap_range, np.ndarray):
            n_lap = len(lap_range)
            enum = enumerate(lap_range)
        else:
            raise TypeError(f'type:{type(lap_range)}')

        #
        if act_1d:
            ret = np.zeros((n_lap, self.n_bins))  # (L, B)
        else:
            ret = np.zeros((act.shape[0], n_lap, self.n_bins), dtype=np.float32)  # (N, L, B)

        if enable_tqdm:
            from tqdm import tqdm
            lap_iter = tqdm(enum, desc=desc if desc is not None else 'get_binned_signal', unit='laps', ncols=80)
        else:
            lap_iter = enum

        occ = None
        for i, lap in lap_iter:
            pos_mask = self._position_mask(lap)
            if running_epoch:
                pos_mask = np.logical_and(pos_mask, run_mask)

            pos = self.position[pos_mask]

            if occ_normalize:  # occ_normalize is constant in this function
                occ = self._occupancy_map(lap, bins, pos, running_epoch=running_epoch)

            if act_1d:
                ret[i] = self._binned_signal(pos, bins, act[pos_mask], occ, smooth)  # (L, B)
            else:
                for n in range(act.shape[0]):
                    ret[n, i] = self._binned_signal(pos, bins, act[n, pos_mask], occ, smooth)  # (N, L, B)

        if norm:
            ret /= np.max(ret)

        return ret

    def _activity(self, t: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Interpolation numbers of image time to the position time

        :param t: Time array. `Array[float, T]`
        :param signal: Signal array. `Array[float, [N, T] | T]`
        :return: Interpolated neural signal. `Array[float, [N, PT] | PT]`
        """
        if t.ndim != 1:
            raise RuntimeError(f'wrong time dimension : {t.ndim}')

        if signal.ndim == 1:
            n_samples = signal.shape[0]
        elif signal.ndim == 2:
            n_neuron, n_samples = signal.shape
            if n_neuron == 0:
                raise RuntimeError(f'empty signal. shape : {signal.shape}')
        else:
            raise RuntimeError(f'wrong signal dimension : {signal.ndim}')

        if len(t) != n_samples:
            raise RuntimeError(f't {t.shape} and signal {signal.shape} shape not matched')

        # activity on position time
        while True:
            try:
                ret = interp1d(
                    t,
                    signal,
                    axis=signal.ndim - 1,
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True
                )(self.position_time)
                return ret
            except MemoryError as e:  # not able to allocate, might not catch successfully in macOS
                fprint(e, vtype='error')
                input('press to continue')

    def _binned_signal(self, pos, bins, act, occ: np.ndarray | None, smooth: bool):
        """

        :param pos: position
        :param bins: bins
        :param act: weight
        :param occ: occupancy
        :param smooth:
        :return:
        """
        r, _ = np.histogram(pos, bins, weights=act)

        if occ is not None:
            r = np.divide(r, occ, out=np.zeros_like(r), where=occ != 0)

        if smooth:
            r[np.isnan(r)] = 0.0
            r = gaussian_filter1d(r, self.smooth_kernel, mode='wrap')

        return r


def load_interpolated_position(rig: RiglogData,
                               sample_rate: int = 1000,
                               force_compute: bool = False,
                               save_cache: bool = True,
                               use_virtual_space: bool = False,
                               norm_length: float = 150) -> CircularPosition:
    """
    get 'CircularPosition' and save as cache

    :param rig: ``RiglogData``
    :param sample_rate: sampling rate for interpolation
    :param force_compute: force recalculate and save as a new cache
    :param save_cache: save cache in the same directory as riglog file
    :param use_virtual_space: if used virtual environment position space
    :param norm_length: maximal length for normalization for each trial
    :return: ``CircularPosition``
    """

    file = rig.riglog_file
    suffix = '_position_cache.npy' if not use_virtual_space else '_virtual_position_cache.npy'
    cache_file = file.with_name(file.stem + suffix)

    if cache_file.exists() and not force_compute:
        d = np.load(cache_file)
        lt = d[:, 4]
        lt = lt[~np.isnan(lt)].astype(int)
        return CircularPosition(d[:, 0], d[:, 1], d[:, 2], d[:, 3], lt)
    else:
        if use_virtual_space:
            p = rig.get_pygame_stimlog().virtual_position_event
        else:
            p = rig.position_event

        d = interp_pos1d(p.time, p.value, sampling_rate=sample_rate, remove_nan=True, norm_max_value=norm_length)

        lt = np.full_like(d.t, np.nan, dtype=np.double)  # make (N,) array
        lt[:len(d.trial_time_index)] = d.trial_time_index

        if save_cache:
            np.save(cache_file, np.vstack([d.t, d.p, d.d, d.v, lt]).T)

        return d
