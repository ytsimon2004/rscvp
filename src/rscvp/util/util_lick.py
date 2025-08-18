from typing import NamedTuple, Literal, ClassVar, cast

import attrs
import numpy as np
import scipy
from typing_extensions import Self

from neuralib.locomotion import CircularPosition
from neuralib.typing import PathLike
from neuralib.util.interp import interp_timestamp
from neuralib.util.verbose import fprint
from rscvp.util.pixviz import PixVizResult
from rscvp.util.util_camera import truncate_video_to_pulse
from rscvp.util.util_trials import TrialSelection
from stimpyp import Session, RiglogData, RigEvent

__all__ = ['LICK_EVENT_TYPE',
           'LickTracker',
           #
           'LickingPosition',
           'peri_reward_transformation',
           'calc_lick_pos_trial',
           'peri_reward_raster_hist']

LICK_EVENT_TYPE = Literal['facecam', 'lickmeter']


class SignalCorrResult(NamedTuple):
    lag_value: np.ndarray
    """Lag values in sec. (2t-1, )"""
    corr: np.ndarray
    """cross-correlate result. (2t-1, )"""
    lag: float
    """lag in sec"""


@attrs.define
class LickTracker:
    """Handling the licking probability from imaging analysis versus electrical sensing"""

    rig: RiglogData
    """source riglog data"""

    camera_time: np.ndarray
    """camera time foreach image pixel"""

    pix_probability: np.ndarray
    """lick probability analyzed by pixel changes"""

    prob_threshold: float | None
    """set after calculation"""

    offset_flag: bool = attrs.field(default=False)
    """If do the time offset"""

    signal_corr: SignalCorrResult | None = attrs.field(default=None)
    """set if offset"""

    offset_time_lag: float | None = attrs.field(default=None)
    """offset time lag in sec"""

    SAMPLING_RATE: ClassVar[int] = 30
    """approximate sampling rate"""

    def __attrs_post_init__(self):
        if self.prob_threshold is None:
            self.prob_threshold = self._calc_auto_thres(self.pix_probability)
            fprint(f'Auto calculate and set lick pixel threshold: {self.prob_threshold}!')

    @staticmethod
    def _calc_auto_thres(data: np.ndarray,
                         bins: int = 50,
                         debug: bool = False,
                         vmin: float | None = None) -> float:
        """
        simple subtract the mode peak (background) and use percentile value as a cutoff

        :param data: pixel intensity 1D vector
        :param bins: histogram for finding the mode
        :param debug: debug plot
        :param vmin: minimal pixel value. i.e., recording in dark exclude
        :return:
        """
        if debug:
            import matplotlib.pyplot as plt
            plt.hist(data, bins)
            plt.show()

        if vmin is not None:
            data = data[data > vmin]

        hist, edg = np.histogram(data, bins)
        mode_index = np.argmax(hist)

        mode = (edg[mode_index] + edg[mode_index + 1]) / 2
        adjusted_data = data - mode

        return float(np.percentile(adjusted_data, 99.5))

    @classmethod
    def load_from_rig(cls, rig: RiglogData,
                      file: PathLike, *,
                      meta: PathLike | None = None,
                      threshold: float | None = None) -> Self:
        """
        Load info from riglog

        :param rig: ``RiglogData``
        :param file: filepath from the pixel-based analysis result
        :param meta: If pixviz backend, specify the meta filepath
        :param threshold: threshold for pixel cutoff. If None then auto calculate after `init()`
        :return: ``LickTracker``
        """

        time = rig.camera_event['facecam'].time
        pix = PixVizResult.load(file, meta)
        prob = pix.get_data('lick')

        prob = truncate_video_to_pulse(prob, time)
        fprint(f'set lick pixel threshold: {threshold}!')

        return cls(rig, time, prob, threshold)

    @property
    def electrical_timestamp(self) -> np.ndarray:
        """timestamp for the electrical-sensing signals"""
        return self.rig.lick_event.time

    @property
    def electrical_event(self) -> RigEvent:
        """interpolated ``RigEvent`` for the electrical-sensing signals"""
        ret = interp_timestamp(self.electrical_timestamp,
                               self.rig.exp_start_time,
                               self.rig.exp_end_time,
                               sampling_rate=self.SAMPLING_RATE)

        return RigEvent('lick_electrical', np.vstack([*ret]).T)

    def prob_to_event(self,
                      binarized: bool = True,
                      interp: bool = False) -> RigEvent:
        """
        Set threshold of probability

        :param binarized: value change to 1
        :param interp
        :return:
        """
        mx = self.pix_probability >= self.prob_threshold
        t = self.camera_time[mx]
        v = self.pix_probability[mx]

        if binarized:
            v = np.full_like(t, 1)

        if interp:
            t, v = interp_timestamp(t, self.rig.exp_start_time, self.rig.exp_end_time, sampling_rate=self.SAMPLING_RATE)

        return RigEvent('lick_track', np.vstack([t, v]).T)

    # ==================== #
    # Handle Lagging Issue #
    # ==================== #

    def with_offset(self) -> Self:
        """time offset between .riglog and facecam acquisition using cross correlation of lick event.
        Mostly due to labcam start recording (initial frames) without waiting for stimpy pulse

        *NOTE*: After 210925, lag issue solved due to the labcam version
        """
        corr = self.calculate_signal_corr()
        lag = corr.lag
        fprint(f'Camera lag offset: {lag}!')
        return attrs.evolve(self,
                            camera_time=self.camera_time + lag,
                            offset_flag=True,
                            signal_corr=corr,
                            offset_time_lag=lag)

    def calculate_signal_corr(self) -> SignalCorrResult:
        """synchronize the cam actual acquisition and .riglog time using cross-correlation
        method refer to: https://gist.github.com/mdhk/ad0725cf494385d699aef6d6c40131be
        """
        sig_electrical = self.electrical_event.value
        sig_track = self.prob_to_event(binarized=True, interp=True).value

        #
        n = len(sig_track)
        lag_value = (np.arange(2 * n - 1) - n + 1) / self.SAMPLING_RATE
        corr = scipy.signal.correlate(sig_electrical, sig_track)  # (2 * n_frame - 1)
        assert len(lag_value) == len(corr)

        corr_max_i = np.argmax(corr)
        lag = lag_value[corr_max_i]

        return SignalCorrResult(lag_value, corr, cast(float, lag))


# ================ #
# LickScore Metric #
# ================ #

class LickingPosition(NamedTuple):
    """Licking as a function of different position"""

    lick_position: list[np.ndarray]
    """len of list equal to lap number, len of array is lick numbers. in cm"""

    boundary_limit: float
    """left and right boundary for peri-event plot"""

    trial_range_index: tuple[int, int]
    """trial index (START, STOP). NOTE index start from 0"""

    def with_session(self, rig: RiglogData, session: Session) -> Self:
        indices = TrialSelection.from_rig(rig, session).get_selected_profile().trial_range
        start, end = indices

        return self._replace(lick_position=self.lick_position[start:end], trial_range_index=(start, end))

    # ================ #
    # LickScore Metric #
    # ================ #

    @property
    def anticipatory_lick_loc(self) -> np.ndarray:
        """anticipatory licking location (L',)"""
        ret = []
        for i, it in enumerate(self.lick_position):
            if len(it) != 0 and it[0] < 0:  # showing lick and lick before reward
                ret.append(it[0])

        return np.array(ret)

    def as_range(self) -> tuple[float, float, float]:
        """Value range across trials. Refer to: Fisher et al., 2020"""
        from scipy.stats import bootstrap
        data = (self.anticipatory_lick_loc,)  # samples must be in a sequence
        res = bootstrap(data, np.median)
        lb = float(res.confidence_interval.low)
        ub = float(res.confidence_interval.high)
        median_value = float(np.median(self.anticipatory_lick_loc))

        return lb, median_value, ub


def calc_lick_pos_trial(position: CircularPosition,
                        trial_time: np.ndarray,
                        lick_time: np.ndarray) -> LickingPosition:
    """
    Calculate the licking positions across trials.

    This function determines the licking positions (mapped to a circular space)
    for each lap within a set of trials. It calculates the licking positions relative
    to trial start and end times and returns the licking positions for each trial
    (grouped into laps), along with the boundary limit and trial index ranges.

    :param position: ``CircularPosition`` object, representing the circular spatial positions
                     with associated timestamps. Contains `position.t` (timestamps)
                     and `position.p` (associated positions).
    :param trial_time: 1D array of timestamps representing the start of each trial (lap).
                       Array shape: `Array[float, L]`, where `L` is the number of trials.
    :param lick_time: 1D array of timestamps indicating when lick events occurred.
                      Array shape: `Array[float, K]`, where `K` is the number of lick events

    :return: A ``LickingPosition`` object
    """
    from scipy.interpolate import interp1d

    lick_pos = interp1d(position.t, position.p, bounds_error=False, fill_value=0)(lick_time)  # (L, ) -> (K, )
    limit = np.max(position.p) / 2
    lick_pos = peri_reward_transformation(lick_pos, limit)  # (K,)

    ret = [np.array([])]  # pre-give for concat

    for i, (left_t, right_t) in enumerate(zip(trial_time[:-1], trial_time[1:])):
        m = np.logical_and(left_t <= lick_time, lick_time <= right_t)
        pm = lick_pos[m] >= 0  # positive
        nm = lick_pos[m] < 0  # negative

        ret[-1] = np.concatenate((ret[-1], lick_pos[m][pm]))
        ret.append(lick_pos[m][nm])

    trial_range_index = (0, len(trial_time) - 1)

    return LickingPosition(ret, limit, trial_range_index)


def peri_reward_transformation(position: np.ndarray, limit: float) -> np.ndarray:
    """
    transform the position value to peri-reward position

    :param position: (N, )
    :param limit:
    :return: transformed position (N, )
    """
    return (position + limit) % (2 * limit) - limit


def peri_reward_raster_hist(t: np.ndarray,
                            reward_time: np.ndarray,
                            limit: float,
                            nbins: int = 100) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Calculate peri-reward raster and histogram.

    :param t: 1D array of event times (e.g., licks).
    :param reward_time: 1D array of reward times.
    :param limit: time window around each reward (single-sided, e.g., 2.0s means [-2, +2]).
    :param nbins: number of histogram bins.
    :return: tuple containing:
        - event_per_trial: list of arrays of event times relative to each reward.
        - hist: normalized histogram (% per bin).
        - bin_edges: edges of the histogram bins.
    """
    event_per_trial = [t[(rt - limit <= t) & (t <= rt + limit)] - rt for rt in reward_time]

    all_aligned_events = np.concatenate(event_per_trial) if event_per_trial else np.array([])

    hist, bin_edges = np.histogram(all_aligned_events, bins=nbins, range=(-limit, limit))
    hist_percent = 100 * hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist, dtype=float)

    return event_per_trial, hist_percent, bin_edges
