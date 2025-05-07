import numpy as np

from neuralib.locomotion import CircularPosition
from neuralib.util.verbose import fprint
from rscvp.util.position import load_interpolated_position
from stimpyp import RiglogData

__all__ = [
    'get_velocity_per_trial',
    'peri_reward_velocity',
    'check_treadmill_trials'
]


def get_velocity_per_trial(lap_time: np.ndarray,
                           interp_pos: CircularPosition,
                           n_position_bins: int = 150,
                           smooth: bool = False) -> np.ndarray:
    """

    :param lap_time:
    :param interp_pos:
    :param n_position_bins:
    :param smooth:
    :return:
        (L, B)
    """

    p = interp_pos.p
    pt = interp_pos.t
    vel = interp_pos.v

    pt_trial = np.zeros_like(pt, dtype=bool)
    p_bin = np.linspace(0, n_position_bins, num=n_position_bins + 1, endpoint=True)

    ret = []
    left_t = lap_time[0]

    for i, lt in enumerate(lap_time[1:]):
        right_t = lt
        if right_t - left_t < 1:
            fprint(f'Warning >>> DJ mice happened in {i} lap!!!', vtype='warning')
            continue

        np.logical_and(left_t < pt, pt < right_t, out=pt_trial)
        v = vel[pt_trial]

        occ = np.histogram(p[pt_trial], p_bin)[0]
        hist_vel = np.histogram(p[pt_trial], p_bin, weights=v)[0] / occ

        if smooth:
            from scipy.ndimage import gaussian_filter1d
            hist_vel = gaussian_filter1d(hist_vel, 3)

        ret.append(hist_vel)
        left_t = right_t

    # ret[np.isnan(ret)] = 0

    return np.array(ret)


def peri_reward_velocity(reward_time: np.ndarray,
                         position_time: np.ndarray,
                         velocity: np.ndarray,
                         n_bins_trial: int = 100,
                         limit: float = 5) -> np.ndarray:
    """

    :param reward_time:
    :param position_time:
    :param velocity:
    :param n_bins_trial: number of bins per trial in a given time bins (peri-left + peri-right)
    :param limit: peri-event time (left / right)
    :return:
        (L, BT) peri-reward velocity
            BT, number of bins per trial in a given time bins (peri-left + peri-right)
    """
    ret = np.zeros((len(reward_time), n_bins_trial))
    for i, rt in enumerate(reward_time):
        left = rt - limit
        right = rt + limit
        time_mask = np.logical_and(left < position_time, position_time < right)
        t = position_time[time_mask]
        v = velocity[time_mask]

        hist, edg = np.histogram(t, n_bins_trial, range=(left, right), weights=v)
        occ = np.histogram(t, edg)[0]
        hist /= occ
        hist[np.isnan(hist)] = 0
        ret[i] = hist

    return ret


def check_treadmill_trials(rig: RiglogData, error_when_abnormal: bool = False) -> list[int] | None:
    """
    Warning verbose of the abnormal behaviors in the linear treadmill setup,
    and get the list of trial number for each behavioral session

    :param rig: ``RiglogData``
    :param error_when_abnormal: Raise error when rig hardware problem, otherwise, give warning
    :return: behavior sessions lap separation (trial numbers)
    """
    vel = load_interpolated_position(rig).v

    # printout warning msg for the abnormal animal's behavior
    if np.any(vel < -10):
        fprint("animal might run in an opposite direction, check further...", vtype='warning')

    n_rt = len(rig.reward_event.time)
    n_lt = len(rig.lap_event.time)
    if n_rt != n_lt:
        fprint(f'reward counts: {n_rt} mismatch with lap counts: {n_lt}, '
               f'might give reward manually during the recording, check further...', vtype='warning')

    err = ''
    try:
        sep = _get_lap_sep(rig)
    except RuntimeError as e:
        err += f'legacy fname: {repr(e)}'
        sep = None
    except ValueError as e:
        err += f'abnormal trial selection across sessions: {repr(e)}'
        sep = None

    finally:
        fprint(f'CHECKED {rig.riglog_file.parent.name} in treadmill task', vtype='pass')
        if len(err) != 0:
            if error_when_abnormal:
                raise RuntimeError(err)
            else:
                fprint(f'WARNING: {err}')

    return sep


def _get_lap_sep(rig: RiglogData) -> list[int]:
    """get the lap numbers cutoff in different behavioral sessions"""
    session = rig.get_stimlog().session_trials()

    val: list[slice] = [
        info.in_slice(rig.lap_event.time, rig.lap_event.value.astype(int), error=False)
        for _, info in zip(range(3), session.values())
    ]

    return [s.stop for s in val][:-1]
