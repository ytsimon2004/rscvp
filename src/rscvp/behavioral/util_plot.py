import numpy as np
from matplotlib.axes import Axes

from neuralib.plot.colormap import insert_colorbar
from stimpyp import RigEvent

__all__ = [
    'plot_velocity_heatmap',
    'plot_velocity_line',
    #
    'plot_peri_reward_lick_raster',
    'plot_peri_reward_lick_hist',
    #
    'plot_lap_time_interval',
    #
    'plot_position_and_event_raster',
    'plot_value_time'

]


def plot_velocity_heatmap(ax: Axes,
                          velocity: np.ndarray,
                          sep: list[int] | None = None,
                          **kwargs):
    """
    velocity heatmap (laps versus position)

    :param ax:
    :param velocity: (L, B)
    :param sep: trial numbers cut-off across behavioral sessions
    :return:
    """
    im = ax.imshow(
        velocity,
        cmap='hot',
        interpolation='none',
        origin='lower',
        aspect='auto',
        **kwargs
    )
    ax.set_ylabel('Trial #')

    if sep is not None:
        for n in sep:
            ax.axhline(n, ls='--', color='w', lw=3, alpha=0.8)

    insert_colorbar(ax, im)


def plot_velocity_line(ax: Axes,
                       velocity: np.ndarray,
                       *,
                       show_all_trials: bool = True,
                       with_fill_between: bool = True,
                       color: str | None = 'k',
                       **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    velocity line chart.

    `Dimension parameters`:

        L = number of laps

        B = number of position bins

    :param ax: ``Axes``
    :param velocity: `Array[float, [L, B]]`
    :param show_all_trials: if True, plot all the trajectory. If False, plot the std around avg
    :param with_fill_between: fill between velocity std across trials
    :param color
    :param kwargs: additional arguments to ``ax.plot()``
    :return: x (`Array[float, B]`) and average velocity (`Array[float, B]`)
    """
    x = np.linspace(0, 150, velocity.shape[1])
    avg_v = np.mean(velocity, axis=0)

    if show_all_trials:
        for i in velocity:
            ax.plot(x, i, color=color, alpha=0.05, **kwargs)

    if with_fill_between:
        std = np.std(velocity, axis=0)
        ax.fill_between(x, avg_v + std, avg_v - std, alpha=0.5)

    ax.plot(x, avg_v, color=color, **kwargs)
    ax.set_xlabel('Position (cm)')
    ax.set_xlim(0, 150)
    ax.set_ylabel('speed (cm/s)')

    return x, avg_v


def plot_peri_reward_lick_raster(ax: Axes,
                                 lick_per_trial: list[np.ndarray],
                                 limit: float,
                                 **kwargs):
    """peri-reward licking raster in function of time (s)"""
    ax.eventplot(lick_per_trial, linelengths=1, linewidths=0.5, **kwargs)
    ax.axvline(0, color='r', linestyle='--', zorder=1)
    ax.set_ylabel('Trial #')
    ax.set_xlim(-limit, limit)
    ax.set_xticklabels([])


def plot_peri_reward_lick_hist(ax: Axes,
                               lick_hist: np.ndarray,
                               limit: float,
                               nbins: int = 100,
                               **kwargs):
    """peri-reward licking histogram in function of time (s)"""
    x = np.linspace(-limit, limit, nbins)
    ax.hist(x, nbins, weights=lick_hist, color='orange', **kwargs)
    ax.axvline(0, color='r', linestyle='--', zorder=1)
    ax.set_xlim(-limit, limit)
    ax.set_ylabel('Count (%)')


def plot_lap_time_interval(ax: Axes, lap_time: np.ndarray):
    """plot time interval of each lap (to see if dj mice issue)"""
    ax.plot(np.diff(lap_time))
    ax.set_xlabel('lap number')
    ax.set_ylabel('time(s)')
    ax.set_ylim(-5, 20)


def plot_position_and_event_raster(ax: Axes,
                                   position_time: np.ndarray,
                                   position: np.ndarray,
                                   lick_event: RigEvent,
                                   reward_event: RigEvent,
                                   lap_event: RigEvent):
    """plot position and other events with color codes"""

    ax.plot(position_time, position, linewidth=0.5)

    lap_event = lap_event.with_pseudo_value(160)
    ax.scatter(lap_event.time, lap_event.value, s=5, c="red", marker='|', linewidths=0.3)

    reward_event = reward_event.with_pseudo_value(170)
    ax.scatter(reward_event.time, reward_event.value, s=5, c="green", marker='|', linewidths=0.3)

    lick_event = lick_event.with_pseudo_value(180)
    ax.scatter(lick_event.time, lick_event.value, s=5, c="orange", marker='|', linewidths=0.3)

    ax.set_ylabel('Position (cm)')


def plot_value_time(ax: Axes, time: np.ndarray, value: np.ndarray, **kwargs):
    """plot the time and its corresponding value"""
    ax.plot(time, value, linewidth=0.5)
    ax.set(xlabel='Time(s)', **kwargs)
