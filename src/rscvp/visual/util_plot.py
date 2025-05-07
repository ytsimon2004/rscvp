from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from neuralib.typing import ArrayLike, ArrayLikeStr

__all__ = ['selective_pie', 'dir_hist']


def selective_pie(n: ArrayLike,
                  labels: ArrayLikeStr,
                  ax: Optional[Axes] = None):
    """
    Plot orientation / direction selective cell proportion

    :param n: Numbers. `Arraylike[float, N]`
    :param labels: Labels. `ArrayLikeStr[str, N]`
    :param ax: ``Axes``
    """
    if len(n) != len(labels):
        raise ValueError('')

    if ax is None:
        _, ax = plt.subplots()

    ax.pie(n, explode=[0.05, 0], labels=labels, autopct='%1.1f%%', startangle=90, radius=0.5)
    ax.axis('equal')


def dir_hist(data: np.ndarray,
             weights: Optional[ArrayLike] = None,
             *,
             label: Optional[str] = None,
             thres: float = None,
             bins: int = 10,
             xlim: tuple[int, int] = (0, 1),
             color: str = 'k',
             ax: Optional[Axes] = None,
             **kwargs):
    """
    Generic histogram plotting function for orientation/direction selectivity or preferred degree

    :param data: `Array[float, N]`
    :param weights: `Array[float, N]`
    :param label: legend label
    :param thres: threshold axvline cutoff
    :param bins: number of bins
    :param xlim: x limit for the histogram plot
    :param color: color
    :param ax: ``Axes``
    :param kwargs: Additional arguments pass to ``ax.set()``
    """

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(data,
            bins=bins,
            range=xlim,
            weights=weights,
            histtype='step',
            color=color,
            alpha=0.7,
            label=label)

    ax.set_xlim(*xlim)
    ax.set(**kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    if thres is not None:
        ax.axvline(thres, ls='--', color='r')

    if label is not None:
        ax.legend()
