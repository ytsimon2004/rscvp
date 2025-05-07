from __future__ import annotations

import collections
from typing import NamedTuple, Literal

import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
from typing_extensions import Self

from argclz import argument, float_tuple_type
from neuralib.persistence import *
from neuralib.plot.tools import AnchoredScaleBar
from neuralib.util.utils import keys_with_value
from rscvp.util.cli import StimpyOptions, Suite2pOptions, PlotOptions
from stimpyp import Direction, SFTF, GratingPattern

__all__ = [
    'AbstractVisualTuningOptions',
    'VisualTuningCache',
    #
    'VisualTuningResult',
    'plot_visual_pattern_trace'
]

BINNED_SIGNAL_TYPE = Literal['pupil', 'df_f', 'spks']
TRIAL_VALUE_TYPE = Literal['mean', 'median']


class AbstractVisualTuningOptions(Suite2pOptions, StimpyOptions, PlotOptions):
    pre_post: tuple[float, float] = argument(
        '--pre-post',
        type=float_tuple_type,
        default=(1, 1),
        help='time foreach stim epoch'
    )

    direction_invert: bool = argument(
        '--direction-invert',
        help='direction invert for the same orientation (different temporal/nasal direction).'
             'i.e., stimpy & KS paper had different definition:'
             'In stimpy: 180 degree present horizontal from temporal to nasal direction, but 0 degree in KS paper'
    )

    value_type: TRIAL_VALUE_TYPE = argument(
        '--VT', '--value-type',
        default='mean',
        help='plot and cache is trial mean or trial median'
    )


class VisualTuningResult(NamedTuple):
    stim_pattern: list[tuple[Direction, SFTF]]
    """stimulus pattern. list of ``P``"""
    pre_post: tuple[int, int]
    """pre post in second"""
    stim_index: np.ndarray
    """`Array[int, [P, F]]`"""
    dat: np.ndarray
    """`Array[float, [N, C] | C]` -> C if not neural activity"""
    signal_type: BINNED_SIGNAL_TYPE
    """{'pupil', 'df_f', 'spks'}"""
    value_type: TRIAL_VALUE_TYPE
    """trial-averaged(mean) or trial-median(median)"""

    @property
    def n_frames(self) -> int:
        """frame number for entire pre + stim + post"""
        starts = [st[0] for st in self.stim_index]  # every start stim epoch frame
        uni = np.unique(np.diff(starts))

        if len(uni) == 1:
            # noinspection PyTypeChecker
            return uni[0]
        else:
            raise RuntimeError('error cache saving')

    def with_mask(self, mask: np.ndarray) -> Self:
        """cell mask"""
        return self._replace(dat=self.dat[mask])


@persistence.persistence_class
class VisualTuningCache(ETLConcatable):
    """
    `Dimension parameters`:

        P = number of stimuli pattern(type)

        F = number of on(stim) epoch frame

        N = number of neurons

        C = concat frames (included all P on-off)


    """
    exp_date: str = persistence.field(validator=True, filename=True)
    """experimental date"""
    animal: str = persistence.field(validator=True, filename=True)
    """animal ID"""
    plane_index: int | str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    """optical imaging plane"""
    signal_type: BINNED_SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    """{'pupil', 'df_f', 'spks'}"""
    value_type: TRIAL_VALUE_TYPE = persistence.field(validator=True, filename=True, filename_prefix='trial_')
    """trial-averaged(mean) or trial-median(median)"""
    direction_invert: bool = persistence.field(validator=True, filename=False)
    """If True, invert the direction for 180 degree. Otherwise, the same temporal/nasal direction as stimpy"""

    #
    neuron_idx: np.ndarray | None
    """neuron idx if neural activity"""
    src_neuron_idx: np.ndarray | None
    """source optic plane if neural activity"""

    pre_post: tuple[int, int]
    """pre post in second"""
    stim_pattern: list[tuple[Direction, SFTF]]
    """[``P``]"""
    stim_index: np.ndarray
    """`Array[int, [P, F]]`"""
    dat: np.ndarray
    """`Array[float, [N, C] | C]` -> C if not neural activity"""

    def load_result(self) -> VisualTuningResult:
        return VisualTuningResult(
            stim_pattern=self.stim_pattern,
            pre_post=self.pre_post,
            stim_index=self.stim_index,
            dat=self.dat,
            signal_type=self.signal_type,
            value_type=self.value_type
        )

    @classmethod
    def concat_etl(cls, data: list[VisualTuningCache]) -> Self:
        data = cls._align_stim_index(data)
        validate_concat_etl_persistence(
            data,
            ('direction_invert',
             'signal_type',
             'value_type',
             'pre_post',
             'stim_pattern')
        )

        const = data[0]
        ret = VisualTuningCache(
            exp_date=const.exp_date,
            animal=const.animal,
            plane_index='_concat',
            signal_type=const.signal_type,
            value_type=const.value_type,
            direction_invert=const.direction_invert
        )

        ret.neuron_idx = np.concatenate([it.neuron_idx for it in data])
        ret.src_idx = np.concatenate([it.src_neuron_idx for it in data])
        ret.pre_post = const.pre_post
        ret.stim_pattern = const.stim_pattern
        ret.stim_index = const.stim_index
        ret.dat = np.vstack([it.dat for it in data])

        return ret

    @classmethod
    def _align_stim_index(cls, data: list[VisualTuningCache]) -> list[VisualTuningCache]:
        """trimmed the frame that captured different by ETL scan"""
        lower_bound = np.min([it.stim_index.shape[1] for it in data])  # min stim frame number
        prestim = np.min([it.stim_index[0][0] for it in data])  # prestim frame number
        per_stim = np.min([it.stim_index[1][0] for it in data]) - prestim  # pre stim frame number

        ret = data.copy()

        for i, it in enumerate(data):
            if it.stim_index.shape[1] != lower_bound:  # find the planes have more frames
                ret[i].stim_index = it.stim_index[:, :lower_bound]
                mask = np.full(it.dat.shape[1], 1, dtype=bool)
                st = lower_bound + prestim + 1
                mask[st::per_stim] = 0
                ret[i].dat = it.dat[:, mask]

        return ret


# =============================== #
# Stim Pattern Activity Container #
# =============================== #


class StimPatternSignal(NamedTuple):
    """
    `Dimension parameters`:

        F = number of on-off frames
    """

    y_index: int
    """for plotting purpose, need to be rollback"""
    direction: Direction
    """stimulus direction in deg"""
    sftf: SFTF
    """tuple of spatial and temporal frequencies"""
    signal: np.ndarray
    """`Array[float, F]`"""
    signal_type: BINNED_SIGNAL_TYPE
    """{'pupil', 'df_f', 'spks'}"""

    @property
    def n_frames(self) -> int:
        return len(self.signal)

    def stim_epoch_index(self, pre_post: tuple[float, float], frame_rate: float) -> np.ndarray:
        pre_frame = int(pre_post[0] * frame_rate)
        post_frame = int(pre_post[1] * frame_rate)
        return np.arange(pre_frame, self.n_frames - post_frame)

    def rollback_actual_value(self) -> Self:
        if self.y_index > 0:
            act = self.signal - self.y_index
            return self._replace(signal=act)
        else:
            return self


def plot_visual_pattern_trace(
        ax: Axes,
        pattern: GratingPattern,
        signal: np.ndarray,
        time: np.ndarray, *,
        foreach_normalized: bool = False,
        pre_post: tuple[float, float] = (1, 1),
        block_xrange: tuple[float, float] = (0.1, 0.9),
        direction_invert: bool = False,
        color: str = 'k',
        sig_type: BINNED_SIGNAL_TYPE = 'df_f',
        value_type: TRIAL_VALUE_TYPE = 'mean',
) -> list[StimPatternSignal]:
    """
    Plot activity tuning in different visual stimulation pattern and cache the signal

    :param ax: ``Axes``
    :param pattern: ``GratingPattern``
    :param signal: `Array[float, F]`
    :param time: `Array[float, F]`
    :param foreach_normalized: each windows (i.e.,5s) normalized. to see the subtle foreach stimuli
    :param pre_post: stimulus pre post time in plot (in sec)
    :param direction_invert: direction invert. i.e., 0 <-> 180, 30 <-> 210, 60 <-> 240.... (used for karoline's exp)
    :param block_xrange: minor block x range (foreach tuning)
    :param color: line color
    :param sig_type: ``BINNED_SIGNAL_TYPE``
    :param value_type: ``TRIAL_VALUE_TYPE``

    :return list of ``StimPatternSignal``
    """
    dir_x = pattern.dir_i()
    sftf_y = pattern.sftf_i()

    pre, post = pre_post

    cy = collections.defaultdict(list)  # dict[(s_x,s_y)] = list((x, y))
    y_limit = []  # max dff(%) for the all stimuli condition. aka. top grey line in the plot

    # foreach trials
    for si, st, sf, tf, dire in pattern.foreach_stimulus():
        tx = np.logical_and(st[0] - pre <= time, time <= st[1] + post)

        s_df = signal[tx]  # shape: (pre + sti + post windows)
        dire = (dire + 180) % 360 if direction_invert else dire  # if invert
        s_x = dir_x[dire]  # value in major x, origin of minor x
        s_y = sftf_y[(sf, tf)]  # value in major y, origin of minor y

        # normalization
        dv = s_df if foreach_normalized else signal
        y = s_df / np.max(dv)

        # plot
        plot_x = np.linspace(*block_xrange, num=len(s_df)) + s_x
        plot_y = y + s_y
        ax.plot(plot_x, plot_y, color='gray', alpha=0.5)

        # container
        cy[(s_x, s_y)].append((plot_x, plot_y))  # collect for doing average
        y_limit.append(np.max(s_df))

    #
    func = getattr(np, value_type)
    ret = []  # for collect trial averaged result
    for p, xy in cy.items():  # p = s_x, s_y
        x = np.linspace(*block_xrange, num=len(s_df)) + p[0]
        y = [interp1d(it[0], it[1])(x) for it in xy]  # it[0], it[1] -> x, y in previous for loop
        sig = func(y, axis=0)
        ax.plot(x, sig, color=color)

        ret.append(
            StimPatternSignal(
                p[1],
                keys_with_value(dir_x, p[0], to_item=True),
                keys_with_value(sftf_y, p[1], to_item=True),
                sig,
                sig_type
            )
        )

    # stim epoch
    stim_time = pattern.get_stim_time()
    factor = block_xrange[1] - block_xrange[0]
    left_bound = pre / (pre + stim_time + post) * factor
    right_bound = post / (pre + stim_time + post) * factor
    for x in range(pattern.n_dir):
        ls = x + block_xrange[0]
        rs = x + block_xrange[1]
        ax.axvspan(ls + left_bound, rs - right_bound, 0, 1, color='mistyrose')

    ax.set_xlabel('Direction')
    ax.set_ylabel('SF/TF')

    #
    y_val = np.max(y_limit).astype(float)
    sbar = AnchoredScaleBar(ax.transData,
                            sizey=1,
                            labely=f'{y_val:.2f}%',
                            pad=0.1,
                            color=color,
                            color_txt=color)

    ax.add_artist(sbar)

    ax.set_xticks([it + 0.5 for it in range(pattern.n_dir)])  # draw ticks at the middle of the stimuli area
    ax.set_yticks([it for it in range(pattern.n_sftf)])
    ax.set_xticklabels([d for d in pattern.dir_i().keys()])
    ax.set_yticklabels(list(sftf_y.keys()))

    return ret
