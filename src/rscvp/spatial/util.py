from typing import NamedTuple, Callable

import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import trange

from neuralib.imaging.suite2p import (
    Suite2PResult,
    get_neuron_signal,
    normalize_signal,
    sync_s2p_rigevent,
    CALCIUM_TYPE
)
from neuralib.locomotion import running_mask1d
from rscvp.util.cli.cli_shuffle import SHUFFLE_METHOD, PositionShuffleOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID, NORMALIZE_TYPE
from rscvp.util.position import load_interpolated_position, PositionBinnedSig
from rscvp.util.typing import SIGNAL_TYPE
from stimpyp import RiglogData

__all__ = [
    'SiResult',
    'calculate_si',
    'SiShuffleResult',
    'calculate_si_shuffle',
    #
    'SiSrcData',
    'prepare_si_data',
    #
    'sort_neuron',
    'PositionSignal',
    #
    'normalized_trial_avg'
]


class SiResult(NamedTuple):
    position: np.ndarray  # shape (B,)
    occupancy: np.ndarray  # shape (B - 1,)
    activity: np.ndarray  # shape (B - 1,)
    spatial_info: float


def calculate_si(t: np.ndarray,
                 x: np.ndarray,
                 ta: np.ndarray,
                 a: np.ndarray,
                 x_bins: int = 100,
                 epoch: np.ndarray = None) -> SiResult:
    """
    Calculate the spatial information for determine the place cell in 1d linear treadmill.
    Note: the absolute value of si might be sensitive to df_f normalization method.

    :param t: 1D position time array. `Array[float, P]`
    :param x: 1D position array. `Array[float, P]`
    :param ta: activities time array. `Array[float, F]`
    :param a: activities value array. `Array[float, F]`
    :param x_bins: position bin number
    :param epoch: specify certain epoch (running epoch) for calculation, shape is equal to x and t.
    :return: ``SiResult``
    """

    bins = np.linspace(0, 1, num=x_bins + 1, endpoint=True)
    at = interp1d(ta, a, bounds_error=False, fill_value=0)(t)
    dt = np.diff(t, prepend=t[0])

    if epoch is not None:
        x = x[epoch]
        at = at[epoch]
        dt = dt[epoch]

    # map.count
    count = gaussian_filter1d(np.histogram(x, bins)[0], 3, mode='wrap')
    # map.time
    occupancy = gaussian_filter1d(np.histogram(x, bins, weights=dt)[0], 3, mode='wrap')
    # map.z
    activity = gaussian_filter1d(np.histogram(x, bins, weights=at)[0], 3, mode='wrap') / count

    si = _spatial_info(occupancy, activity)

    return SiResult(bins, occupancy, activity, si)


class SiShuffleResult(NamedTuple):
    position: np.ndarray  # shape (B,)

    occupancy: np.ndarray  # shape (B - 1,)

    activity: np.ndarray  # shape (T, B-1)
    """activity for every shuffle time"""

    si_shuffled: np.ndarray  # shape (S,)
    """si for every shuffle time"""

    spatial_info: float
    """percentile of shuffled si"""

    percentage: float

    shuffle_method: SHUFFLE_METHOD | Callable[[np.ndarray], np.ndarray] | str

    @property
    def shuffle_times(self) -> int:
        return self.activity.shape[0]


def calculate_si_shuffle(t: np.ndarray,
                         x: np.ndarray,
                         ta: np.ndarray,
                         a: np.ndarray,
                         shuffle_times: int = 1000,
                         shuffle_method: SHUFFLE_METHOD | Callable[[np.ndarray], np.ndarray] = 'cyclic',
                         percentage: float = 97.5,
                         x_bins: int = 100,
                         epoch: np.ndarray | None = None) -> SiShuffleResult:
    """

    :param t: position time stamp array
    :param x: Normalized position array [0, 1]
    :param ta: activities time array
    :param a: activities value array
    :param shuffle_times:
    :param shuffle_method:
    :param percentage:
    :param x_bins: position bin number
    :param epoch: specify certain epoch (running epoch) for calculation, shape is equal to x and t.
    :return: ``SiShuffleResult``
    """
    bins = np.linspace(0, 1, num=x_bins + 1, endpoint=True)

    dt = np.diff(t, prepend=t[0])

    if epoch is not None:
        dt = dt[epoch]
        x = x[epoch]

    # map.count
    count = gaussian_filter1d(np.histogram(x, bins)[0], 3, mode='wrap')

    # map.time
    occupancy = gaussian_filter1d(np.histogram(x, bins, weights=dt)[0], 3, mode='wrap')

    ret = np.zeros((shuffle_times,))
    ret_act = np.zeros((shuffle_times, len(bins) - 1))

    a = interp1d(ta, a, bounds_error=False, fill_value=0)(t)

    # since sampling rate of a is currently transforming from ta to t, therefore recalculate shuffle_tw
    shuffle_tw = int(20 / np.median(dt))  # shuffle time window, how many time steps (imaging fs) in 20s

    if epoch is not None:
        a = a[epoch]

    def _calculate_si(i: int, activity: np.ndarray):
        activity = gaussian_filter1d(np.histogram(x, bins, weights=activity)[0], 3, mode='wrap') / count
        ret_act[i] = activity
        ret[i] = _spatial_info(occupancy, activity)

    #
    match shuffle_method:
        case 'cyclic':
            for i in trange(shuffle_times, desc='shuffle', unit='times', ncols=80, leave=False):
                _calculate_si(i, np.roll(a, np.random.randint(shuffle_tw, len(a) - shuffle_tw)))
        case 'random':
            b = a.copy()
            for i in trange(shuffle_times, desc='shuffle', unit='times', ncols=80, leave=False):
                np.random.shuffle(b)
                _calculate_si(i, b)
        case callable():
            b = a.copy()
            for i in range(shuffle_times):
                _calculate_si(i, shuffle_method(b))
            shuffle_method = str(shuffle_method)
        case _:
            raise ValueError(f'unknown shuffle method: {shuffle_method}')

    return SiShuffleResult(
        bins,
        occupancy,
        ret_act,
        ret,
        np.percentile(ret, percentage),
        percentage,
        shuffle_method,
    )


def _spatial_info(occupancy: np.ndarray, activity: np.ndarray) -> float:
    r"""
    spatial information score (bits/event) (Skaggs et al., 1996)
        .. math:: \sum_{i=1}^n P_i \frac{\lambda_i}{\lambda} log_{2}(\frac{\lambda_i}{\lambda})
       :label: math-sample

    :param occupancy: (B-1,)
    :param activity: (B-1,)
    :return:
    """
    occp = occupancy / np.sum(occupancy)  # (Pi) is the probability the mouse stays in the i bin
    f = np.sum(occp * activity)  # average (expected value) occupancy-normalized activity across the trial
    fif = activity / f  # fi/f
    occp = occp[fif > 0]
    fif = fif[fif > 0]

    return np.sum(occp * fif * np.log2(fif))


class SiSrcData(NamedTuple):
    opt: PositionShuffleOptions
    neuron_list: list[int]

    rig: RiglogData
    s2p: Suite2PResult

    pos_time: np.ndarray
    pos_value: np.ndarray
    image_time: np.ndarray

    act_mask: np.ndarray
    run_epoch: np.ndarray

    def signal(self, neuron: int, signal_type: SIGNAL_TYPE | None = None):
        if signal_type is None:
            signal_type = self.opt.signal_type
        return get_neuron_signal(self.s2p, neuron, signal_type=signal_type, normalize=True, dff=True)[0][self.act_mask]

    def calc_si(self, neuron: int | np.ndarray,
                window: int = None) -> SiResult:
        """

        :param neuron: `neuron id` or `neuron activity signal`
        :param window:
        :return:
        """
        if window is None:
            window = self.opt.pos_bins  # bin size, from BeltOptions

        if isinstance(neuron, int):
            signal = self.signal(neuron)
        else:
            signal = neuron
            if self.image_time.shape != signal.shape:
                raise RuntimeError('not yet alignment between image_time and signal')

        return calculate_si(self.pos_time, self.pos_value, self.image_time, signal, window, self.run_epoch)

    def calculate_si_shuffle(self,
                             neuron: int | np.ndarray,
                             shuffle_method: SHUFFLE_METHOD,
                             window: int = None):
        if window is None:
            window = self.opt.pos_bins

        if isinstance(neuron, int):
            signal = self.signal(neuron)
        else:
            signal = neuron
            if self.image_time.shape != signal.shape:
                raise RuntimeError()

        return calculate_si_shuffle(
            self.pos_time,
            self.pos_value,
            self.image_time,
            signal,
            self.opt.shuffle_times,
            shuffle_method,
            self.opt.percentage,
            window,
            self.run_epoch
        )


def prepare_si_data(opt: PositionShuffleOptions, neuron_ids: NeuronID) -> SiSrcData:
    s2p = opt.load_suite_2p()
    neuron_list = get_neuron_list(s2p, neuron_ids)

    rig = opt.load_riglog_data()
    image_time = rig.imaging_event.time
    image_time = sync_s2p_rigevent(image_time, s2p, opt.plane_index)

    ip = load_interpolated_position(rig)

    if opt.running_epoch:
        run_epoch = running_mask1d(ip.t, ip.v)
    else:
        run_epoch = None

    session = rig.get_stimlog().session_trials()[opt.session]
    pos_mask = session.time_mask_of(ip.t)
    pos_time = ip.t[pos_mask]
    pos_value = normalize_signal(ip.p)[pos_mask]
    run_epoch = run_epoch[pos_mask] if run_epoch is not None else None

    act_mask = session.time_mask_of(image_time)
    image_time = image_time[act_mask]

    return SiSrcData(
        opt,
        neuron_list,
        rig,
        s2p,
        pos_time,
        pos_value,
        image_time,
        act_mask,
        run_epoch
    )


# =============== #
# Position-binned #
# =============== #


class PositionSignal:

    def __init__(self, s2p: Suite2PResult,
                 riglog: RiglogData,
                 *,
                 bin_range: tuple[int, int] = (0, 150),
                 window_count: int = 100,
                 signal_type: SIGNAL_TYPE = 'df_f',
                 plane_index: int | None = 0):
        """
        :param s2p: ``Suite2PResult``
        :param riglog: ``RiglogData``
        :param bin_range: (start, end)
        :param window_count: Number of position bins
        :param signal_type: ``SIGNAL_TYPE``. {'df_f', 'spks', 'cascade_spks'}
        :param plane_index: Index of optical plane
        """
        self.s2p = s2p
        self.rig = riglog
        self.binned_sig = PositionBinnedSig(riglog, bin_range=(*bin_range, window_count))

        self._signal_type = signal_type
        self._plane_index = plane_index

        # cache
        self.__image_time = None

    @property
    def signal_type(self) -> SIGNAL_TYPE:
        return self._signal_type

    @property
    def window_count(self) -> int:
        return self.binned_sig.n_bins

    @property
    def plane_index(self) -> int:
        return self._plane_index

    @property
    def image_time(self) -> np.ndarray:
        if self.__image_time is None:
            it = self.rig.imaging_event.time
            self.__image_time = sync_s2p_rigevent(it, self.s2p, self._plane_index)
        return self.__image_time

    def get_signal(
            self,
            neuron: int | np.ndarray,
            lap_range: tuple[int, int] | None = None,
            normalize: bool = True,
            dff: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        activities, baseline activities and imaging time in certain interval of laps (nlaps)

        :param neuron:
        :param lap_range: lap (trial) number from start to end (excluded)
        :param normalize: 01 normalize
        :param dff: df/f normalize
        :return:
            image_time: (S,)
            signal: (S,) if single neuron; (N, S) as multiple neurons
            baseline signal: (S,) if single neuron; (N, S) as multiple neurons
        """
        signal_type = self.signal_type
        match signal_type:
            case 'df_f' | 'spks':
                # noinspection PyTypeChecker
                sig, bas = get_neuron_signal(self.s2p, neuron, signal_type=signal_type, normalize=normalize, dff=dff)
            case 'cascade_spks':
                from rscvp.util.util_cascade import get_neuron_cascade_spks
                sig = get_neuron_cascade_spks(self.s2p, neuron)
                bas = np.full_like(sig, 0, dtype=int)
            case _:
                raise ValueError(f'Unknown signal type: {signal_type}')

        #
        imt = self.image_time
        if lap_range is None:
            return imt, sig, bas

        lt = self.rig.lap_event.time
        lt0 = lt[lap_range[0]]
        lt1 = lt[lap_range[1] - 1]  # lap_event value start from 1, avoid out of bound 'IndexError'
        ltx = np.logical_and(lt0 <= imt, imt <= lt1)

        if sig.ndim == 1:
            return imt[ltx], sig[ltx], bas[ltx]
        else:
            return imt[ltx], sig[:, ltx], bas[:, ltx]

    def load_binned_data(self, do_norm: NORMALIZE_TYPE = 'local',
                         running_epoch: bool = False,
                         ret_type: CALCIUM_TYPE = 'transient',
                         smooth: bool = False) -> np.ndarray:
        """save or load the binned calcium activity data in all neurons (in single etl plane)

        :param do_norm:
        :param running_epoch:
        :param ret_type: return signal type. {'act', 'bas'}
        :param smooth
        :return: shape (N, L, B) where N = total neurons, L = diff(lap) - 1, B = spatial bins
        """
        return self._load_binned_data(do_norm, running_epoch, ret_type, smooth)

    def _load_binned_data(self,
                          do_norm: NORMALIZE_TYPE = 'local',
                          running_epoch: bool = False,
                          calcium_type: CALCIUM_TYPE = 'transient',
                          smooth: bool = False) -> np.ndarray:

        if do_norm == 'local':
            normalize = True
        else:
            normalize = False

        n_neurons = self.s2p.n_neurons
        total_neuron = np.arange(n_neurons)
        t, signal, sig_bas = self.get_signal(total_neuron, None, normalize=normalize, dff=True)

        if calcium_type == 'transient':
            act = self.binned_sig.calc_binned_signal(
                t, signal, running_epoch=running_epoch, enable_tqdm=True, smooth=smooth
            )
        elif calcium_type == 'baseline':
            act = self.binned_sig.calc_binned_signal(
                t, sig_bas, running_epoch=running_epoch, enable_tqdm=True, smooth=smooth
            )
        else:
            raise TypeError(f'{calcium_type}')

        if do_norm == 'global':
            act = act / np.max(act)
        return act


# ======= #

def sort_neuron(data: np.ndarray) -> np.ndarray:
    """
    sort neurons based on maximal activity along the belt

    :param data: 2d binned calactivity data. (N, B)
    :return: sorted indices
    """
    m_filter = scipy.ndimage.gaussian_filter1d(data, 3, axis=1)
    m_argmax = np.argmax(m_filter, axis=1)
    return np.argsort(m_argmax)


def normalized_trial_avg(signal_all: np.ndarray) -> np.ndarray:
    """
    Trial-averaged the position binned activity for all neurons

    :param signal_all: (N', L, B)
    :return: (N', B)
    """
    signal = np.nanmean(signal_all, axis=1)  # (N', B)
    return normalize_signal(signal, axis=1)  # (N', B)  normalize across spatial bins
