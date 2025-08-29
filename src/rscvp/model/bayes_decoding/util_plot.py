import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from neuralib.plot.colormap import insert_colorbar

__all__ = [
    'plot_decoding_err_position',
    'plot_confusion_scatter',
    'plot_confusion_heatmap'
]


def plot_decoding_err_position(mean_err: np.ndarray,
                               sem_err: np.ndarray,
                               *,
                               total_length: int = 150,
                               window: int = 100,
                               color: str | None = 'k',
                               ax: Axes | None = None,
                               **kwargs):
    """
    Plot decoding error as a function of position bins

    B = number of position bins = window

    :param mean_err: Mean decoding error. `Array[float, B]`
    :param sem_err: Standard error of mean decoding error. `Array[float, B]`
    :param total_length: Total length of the 1D environment (in cm)
    :param window: Number of position bins for each trial, must the same as the length of the ``mean_err`` and ``sem_err``
    :param color: Color for the mean curve.
    :param ax: ``Axes``
    """
    if len(mean_err) != window or len(sem_err) != window:
        raise ValueError(f'check shape')

    if ax is None:
        _, ax = plt.subplots()

    x = np.linspace(0, total_length, window)
    ax.plot(x, mean_err, color=color, **kwargs)
    ax.fill_between(x, mean_err + sem_err, mean_err - sem_err, color='grey', alpha=0.5)
    ax.set(xlabel='position(cm)', ylabel='decoding error(cm)', ylim=(0, 60))


def plot_confusion_scatter(actual_position: np.ndarray,
                           predicted_position: np.ndarray,
                           *,
                           total_length: float = 150,
                           landmarks: tuple[int, ...] | None = None,
                           ax: Axes | None = None):
    """
    Plot `scatter confusion matrix` of the decoding results

    T = number of temporal bins

    :param actual_position: Animal actual position. `Array[float, T]`
    :param predicted_position: Position by model prediction. `Array[float, T]`
    :param total_length: Total length of the 1D environment (in cm)
    :param landmarks: Cue location in the environment (in cm)
    :param ax: ``Axes``
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(actual_position, predicted_position, s=3, alpha=0.7, color='grey', edgecolor='none')

    if landmarks is not None:
        for c in landmarks:
            ax.axvline(c, ls='--', color='k', alpha=0.5)
            ax.axhline(c, ls='--', color='k', alpha=0.5)

    ax.axis('square')
    ax.set(xlabel='predicted', ylabel='actual', xlim=(0, total_length), ylim=(0, total_length))


def plot_confusion_heatmap(actual_position: np.ndarray,
                           predicted_position: np.ndarray,
                           nbins: int = 30,
                           *,
                           total_length: int = 150,
                           landmarks: tuple[int, ...] | None = None,
                           ax: Axes | None = None):
    """
    Plot `heatmap confusion matrix` of the decoding results

    T = number of temporal bins

    :param actual_position: Animal actual position. `Array[float, T]`
    :param predicted_position: Position by model prediction. `Array[float, T]`
    :param nbins: Number of position bins
    :param total_length: Total length of the 1D environment (in cm)
    :param landmarks: Cue location of labeling
    :param ax: ``Axes``
    :return:
    """
    pos = np.array([(e, a) for e, a in zip(predicted_position, actual_position)])  # (T, 2)

    if nbins is not None:
        pos = np.histogram2d(pos[:, 0], pos[:, 1], bins=(nbins, nbins))[0]
        pos /= np.sum(pos, axis=1, keepdims=True)  # normalize to actual position

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(pos,
                   cmap='Blues',
                   aspect='auto',
                   origin='lower',
                   extent=(0, total_length, 0, total_length),
                   vmin=0, vmax=1)  # for visualization

    if landmarks is not None:
        for c in landmarks:
            ax.axvline(c, ls='--', color='k', alpha=0.5)
            ax.axhline(c, ls='--', color='k', alpha=0.5)

    ax.set(xlabel='predicted', ylabel='actual')
    cbar = insert_colorbar(ax, im)
    cbar.ax.set(ylabel='probability')
    ax.axis('square')
