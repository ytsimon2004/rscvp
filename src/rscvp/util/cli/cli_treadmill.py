from typing import ClassVar

from argclz import argument, int_tuple_type
from .cli_stimpy import StimpyOptions

__all__ = ['TreadmillOptions']


class TreadmillOptions(StimpyOptions):
    """Options for linear treadmill task"""

    GROUP_TREADMILL: ClassVar = 'Belt Options'
    """group treadmill options"""

    # ----- Position ----- #

    pos_bins: int = argument(
        '--pbins',
        metavar='VALUE',
        default=100,
        validator=lambda it: it > 0,
        group=GROUP_TREADMILL,
        help='bin number for the belt',
    )

    belt_length: int = argument(
        '--belt-length',
        default=150,
        validator=lambda it: it > 0,
        group=GROUP_TREADMILL,
        help='length of the belt'
    )

    cue_loc: tuple[float, ...] = argument(
        '--cue',
        metavar='VALUE',
        type=int_tuple_type,
        default=(50, 100),
        group=GROUP_TREADMILL,
        help='cue location',
    )

    # ----- Running ----- #

    running_epoch: bool = argument(
        '--run',
        group=GROUP_TREADMILL,
        help='whether select only the running epoch',
    )

    smooth_vel: bool = argument(
        '--vsmooth',
        group=GROUP_TREADMILL,
        help='do velocity smoothing',
    )

    cutoff_vel: float = argument(
        '--vcut',
        metavar='VALUE',
        default=-20,
        group=GROUP_TREADMILL,
        help='velocity cutoff for transient negative calculation',
    )

    # ----- Plot ----- #

    psth_sec: float = argument(
        '--psth',
        metavar='VALUE',
        group=GROUP_TREADMILL,
        default=3,
        help='one side limit for peri-event raster(s)',
    )

    @property
    def track_length(self) -> int:
        """get track length depending on either physical or virtual track"""
        if self.use_virtual_space:
            if self._riglog is None:
                self.load_riglog_data()
            return int(self._riglog.get_pygame_stimlog().get_virtual_length())
        else:
            return self.belt_length
