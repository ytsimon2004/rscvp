import numpy as np

from neuralib.locomotion import CircularPosition
from neuralib.util.verbose import fprint
from rscvp.util.position import load_interpolated_position
from stimpyp import RiglogData

__all__ = [
    'get_velocity_per_trial',
    'check_treadmill_trials'
]


def get_velocity_per_trial(lap_time: np.ndarray,
                           pos: CircularPosition,
                           bins: int = 150,
                           smooth: bool = False) -> np.ndarray:
    """
    Get running velocity per trial

    :param lap_time: time array foreach lap. `Array[float, L]`
    :param pos: :class:`~neuralib.locomotion.position.CircularPosition`
    :param bins: number of position bins for each trial(lap)
    :param smooth: do gaussian smoothing
    :return: velocity 2D numpy array. `Array[float, [L, B]]`
    """

    p = pos.p
    pt = pos.t
    vel = pos.v

    pt_trial = np.zeros_like(pt, dtype=bool)
    p_bin = np.linspace(0, bins, num=bins + 1, endpoint=True)

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

    return np.array(ret)


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
