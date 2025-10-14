import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from neuralib.plot.colormap import insert_colorbar
from neuralib.typing import ArrayLike, ArrayLikeStr

__all__ = ['selective_pie',
           'dir_hist',
           'mismatch_hist']


def selective_pie(n: ArrayLike,
                  labels: ArrayLikeStr,
                  ax: Axes | None = None):
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

    ax.pie(n, explode=[0.05, 0.05, 0], labels=labels, autopct='%1.1f%%', startangle=90, radius=0.5)
    ax.axis('equal')


def dir_hist(data: np.ndarray,
             weights: ArrayLike | None = None,
             *,
             label: str | None = None,
             thres: float = None,
             bins: int = 10,
             xlim: tuple[int, int] = (0, 1),
             color: str = 'k',
             ax: Axes | None = None,
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


def mismatch_hist(data: np.ndarray,
                  bins: int = 60,
                  ax: Axes | None = None,
                  **kwargs):
    """
    Generate a 2D histogram to visualize mismatches between two direction visual stimulation

    :param data: Array of two datasets to be compared. Array[float, [2, N]]
    :param bins: Number of bins to use for the histogram along each axis.
    :param ax: Matplotlib Axes object where the histogram will be rendered.
        If None, a new figure and axes will be created.
    :param kwargs: Additional keyword arguments to configure the Axes settings.
    """
    if ax is None:
        _, ax = plt.subplots()

    xmin, xmax = data[0].min(), data[0].max()
    ymin, ymax = data[1].min(), data[1].max()

    data = np.histogram2d(data[0], data[1], bins=(bins, bins))[0]

    row_sums = np.sum(data, axis=1, keepdims=True)
    data = np.divide(data, row_sums, where=row_sums != 0, out=np.zeros_like(data))

    im = ax.imshow(
        data.T,
        cmap='gray',
        aspect='equal',
        origin='lower',
        extent=(xmin, xmax, ymin, ymax)
    )
    cbar = insert_colorbar(ax, im)
    cbar.ax.set(ylabel='probability')

    lim = [min(xmin, ymin), max(xmax, ymax)]
    ax.plot(lim, lim, 'w--', alpha=0.5, linewidth=1, zorder=10)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), **kwargs)
