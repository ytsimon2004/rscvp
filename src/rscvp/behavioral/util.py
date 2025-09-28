import numpy as np

from neuralib.locomotion import CircularPosition
from neuralib.util.verbose import fprint

__all__ = ['get_binned_velocity']


def get_binned_velocity(lap_time: np.ndarray,
                        pos: CircularPosition,
                        track_length: int = 150,
                        bins: int = 50,
                        smooth: bool = False) -> np.ndarray:
    """
    Get running velocity per trial

    :param lap_time: time array foreach lap. `Array[float, L]`
    :param pos: :class:`~neuralib.locomotion.position.CircularPosition`
    :param track_length: number of position bins for each trial(lap)
    :param bins: number of position bins for each trial(lap)
    :param smooth: do gaussian smoothing
    :return: velocity 2D numpy array. `Array[float, [L, B]]`
    """

    p = pos.p
    pt = pos.t
    vel = pos.v

    mx = np.zeros_like(pt, dtype=bool)
    p_bin = np.linspace(0, track_length, num=bins + 1, endpoint=True)

    ret = []
    left_t = lap_time[0]
    for i, lt in enumerate(lap_time[1:]):
        right_t = lt

        if right_t - left_t < 1:
            fprint(f'wrong detection in {i} lap', vtype='warning')
            continue

        np.logical_and(left_t < pt, pt < right_t, out=mx)
        v = vel[mx]

        occ = np.histogram(p[mx], p_bin)[0]
        binned_vel = np.divide(
            np.histogram(p[mx], p_bin, weights=v)[0],
            occ,
            out=np.full_like(occ, np.nan, dtype=float),
            where=occ != 0
        )

        if smooth:
            from scipy.ndimage import gaussian_filter1d
            binned_vel = gaussian_filter1d(binned_vel, 3)

        ret.append(binned_vel)
        left_t = right_t

    return np.array(ret)
