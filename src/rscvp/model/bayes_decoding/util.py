import numpy as np

__all__ = ['calc_wrap_distance']


def calc_wrap_distance(x: np.ndarray,
                       y: np.ndarray,
                       upper_bound: int = 150) -> np.ndarray:
    """calculate the distance between two points in the wrapped environment"""

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('')

    points = np.sort([*zip(x, y)])
    distances = points[:, 1] - points[:, 0]
    distances_wrap = upper_bound - distances

    return np.minimum(distances, distances_wrap)
