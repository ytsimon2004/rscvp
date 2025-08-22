from typing import cast, Sequence

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import sem

from neuralib.plot.colormap import insert_colorbar
from neuralib.typing import AxesArray
from rscvp.util.typing import SIGNAL_TYPE

__all__ = [
    'plot_sorted_trial_averaged_heatmap',
    'plot_fraction_active',
    'plot_alignment_map',
    'plot_tuning_heatmap'
]


def plot_sorted_trial_averaged_heatmap(signal: np.ndarray,
                                       signal_type: SIGNAL_TYPE,
                                       *,
                                       smooth_sigma: float | None = None,
                                       cmap: str = 'Greys',
                                       interpolation: str = 'none',
                                       total_length: int = 150,
                                       cue_loc: tuple[float, ...] | None = None,
                                       n_selected_neurons: int | None = None,
                                       n_total_neurons: int | None = None,
                                       ax: Axes | None = None):
    """
    Plot the heatmap of trial-averaged position-binned responses for all neurons

    `Dimension parameters`:

        N = number of selected neurons (sorted)

        B = Number of position bins

    :param signal: calcium signal. `Array[float, [N, B]]`
    :param signal_type: {"df_f", "spks", 'cascade_spks'}
    :param smooth_sigma: Gaussian smoothing sigma
    :param cmap: ``plt.imshow()`` cmap
    :param interpolation: ``plt.imshow()`` interpolation
    :param total_length: Total length of the 1D environment (in cm)
    :param cue_loc: Cue location in the environment (in cm)
    :param n_selected_neurons: Number of selected neurons for title label
    :param n_total_neurons: Number of total neurons for title label
    :param ax: ``Axes``
    """
    if ax is None:
        _, ax = plt.subplots()

    if smooth_sigma is not None:
        signal = scipy.ndimage.gaussian_filter1d(signal, sigma=3, axis=1)

    im = ax.imshow(
        signal,
        extent=(0, total_length, 0, len(signal)),
        cmap=cmap,
        interpolation=interpolation,
        aspect='auto',
        origin='lower',
    )

    if cue_loc is not None:
        for i in cue_loc:
            ax.axvline(i, ls='--', color='r', alpha=0.5)

    ax.set(xlabel='Position (cm)', ylabel='Neurons #')
    cbar = insert_colorbar(ax, im)

    if signal_type == 'spks':
        cbar.ax.set_ylabel('Norm. deconv')
    elif signal_type == 'df_f':
        cbar.ax.set_ylabel('Norm. ∆F/F')
    else:
        raise ValueError(f'unknown signal type:{signal_type}')

    if n_selected_neurons and n_total_neurons:
        ax.set_title(f'num of neurons used: {n_selected_neurons} / {n_total_neurons}')


def plot_fraction_active(ax: Axes,
                         signal: np.ndarray,
                         *,
                         n_bins: int = 15,
                         belt_length: int = 150,
                         cue_loc: tuple[float, ...] | None = None) -> None:
    """
    Plot fraction of active cell along the belt

    :param ax:
    :param signal: calcium signal. (N'(sorted selected neuron idx), B)
    :param n_bins: sample points (bin numbers) from histogram
    :param belt_length
    :param cue_loc
    """
    n_neurons = signal.shape[0]
    n_spatial_bin = signal.shape[1]

    max_resp_bin = np.argmax(signal, axis=1)  # (B,)

    b = np.linspace(0, n_spatial_bin, n_bins + 1)
    hist, _ = np.histogram(max_resp_bin, bins=b)

    x = np.linspace(0, belt_length, n_bins)
    y = hist / n_neurons  # fraction

    ax.plot(x, y, color='k', lw=3)

    if cue_loc is not None:
        for i in cue_loc:
            ax.axvline(i, ls='--', color='r', alpha=0.5)

    ax.set_xlim(0, belt_length)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Frac. active cell')


def plot_alignment_map(signal: np.ndarray,
                       signal_type: SIGNAL_TYPE,
                       *,
                       total_length: int = 150,
                       select_top: int | None = None,
                       neuron_norm: bool = True,
                       interpolation: str = 'none',
                       axes: AxesArray | None = None):
    """
    Align the position-binned data (N, B) with the peak response, and sorted by the spatial information

    :param signal: `Array[float, [N, B]]`
    :param signal_type: {"df_f", "spks"}
    :param total_length: Total length of the 1D environment (in cm)
    :param select_top: Only pick up the top cells for plotting, used if the neuron is sorted by spatial information
    :param neuron_norm: Trial averaged activity per neuron 01 normalization
    :param interpolation: Method for ``cmap`` interpolation
    :param axes: `Array[Axes, *]`
    """
    # check
    if axes is None:
        _, axes = plt.subplots(2, 1)
        axes = cast(AxesArray, axes)
    elif isinstance(axes, np.ndarray):
        pass
    else:
        raise TypeError(f'{type(axes)}')

    # selection
    if select_top is not None:
        signal = signal[:select_top]

    # heatmap
    half_lag = total_length / 2

    if neuron_norm:
        kw = dict(axis=1, keepdims=True)
        _sig_norm = (signal - signal.min(**kw)) / (signal.max(**kw) - signal.min(**kw))
    else:
        _sig_norm = signal

    ax = axes[0]
    im = ax.imshow(
        _sig_norm,
        extent=(-half_lag, half_lag, 0, signal.shape[0]),
        cmap='hot',
        interpolation=interpolation,
        aspect='auto',
    )
    ax.set(xlabel='Lag(cm)', ylabel='Neurons #')
    insert_colorbar(ax, im)

    # curve tuning
    ax = axes[1]
    avg_s = np.mean(signal, axis=0)  # (B,)
    sem_s = sem(signal, axis=0)
    x = np.linspace(-half_lag, half_lag, len(avg_s))
    ax.plot(x, avg_s)
    ax.fill_between(x, avg_s + sem_s, avg_s - sem_s, alpha=0.3)

    ax.set(xlabel='Lag(cm)',
           ylabel=f'{signal_type}',
           xlim=(-half_lag, half_lag))
    ax.sharex(axes[0])


def plot_tuning_heatmap(signal: np.ndarray, *,
                        belt_length: int = 150,
                        colorbar: bool = False,
                        session_line: Sequence[int] | None = None,
                        ax: Axes | None = None):
    """
    Plot the heatmap tuning for x(position bins) and y(trials)

    `Dimension parameters`:

        L = Number of laps(trials)

        B = Number of position bins

    :param signal: Binned calcium activity. `Array[float, [L, B]]`
    :param belt_length: Belt length in cm
    :param colorbar: If show colorbar
    :param session_line: Lines for separate different behavioral sessions
    :param ax: ``Axes``
    """
    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(signal,
                   extent=(0, belt_length, 0, len(signal)),
                   cmap='viridis',
                   interpolation='none',
                   aspect='auto',
                   origin='lower')

    # add session hline
    if session_line is not None:
        acc = 0
        for h in session_line:
            acc += h
            ax.axhline(acc, ls='--', color='w', lw=0.5)

    if colorbar:
        insert_colorbar(ax, im, inset=True)

    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
