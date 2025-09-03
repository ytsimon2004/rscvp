import numpy as np

from neuralib.locomotion import CircularPosition
from neuralib.util.verbose import fprint

__all__ = ['get_velocity_per_trial']


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
