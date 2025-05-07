from typing import Optional

import numpy as np

__all__ = ['image_time_per_trial',
           'moving_average']


def image_time_per_trial(image_time: np.ndarray,
                         lap_time: np.ndarray,
                         signal: np.ndarray,
                         act_mask: Optional[np.ndarray] = None) -> list[np.ndarray]:
    """
    create a mask for per trial activity

    :param image_time: (S, ) sampling
    :param lap_time: (L, )  laps
    :param signal:  (S, )
    :param act_mask: (S,)
    :return:
        signal per lap (L, S). Note that S has different len per lap
    """
    if not (signal.ndim == 1 and image_time.ndim == 1):
        raise ValueError('signal and image_time should be 1d array')

    if act_mask is None:
        act_mask = np.ones_like(signal, dtype=bool)
    else:
        try:
            signal[act_mask]
        except:
            raise ValueError('act_mask with wrong shape')

    image_time = image_time[act_mask]

    ret = []
    for (left_t, right_t) in (zip(lap_time[:-1], lap_time[1:])):
        x = np.logical_and(
            left_t < image_time,
            image_time < right_t
        )
        ret.append(signal[act_mask][x])

    return ret


def moving_average(arr: np.ndarray, num: int) -> np.ndarray:
    """do the average every `number` of element
    ** Down-sampling
    """
    if arr.ndim != 1:
        raise RuntimeError('input arr needs to be 1d')

    return np.array([np.mean(arr[i:i + num]) for i in range(0, len(arr), num)])
