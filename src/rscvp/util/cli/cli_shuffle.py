from typing import ClassVar, Literal

from argclz import argument
from .cli_suite2p import Suite2pOptions
from .cli_treadmill import TreadmillOptions

__all__ = ['ShuffleBaseOptions',
           'PositionShuffleOptions',
           'SHUFFLE_METHOD']

SHUFFLE_METHOD = Literal['cyclic', 'random']


class ShuffleBaseOptions:
    """Data shuffle options"""

    GROUP_SHUFFLE: ClassVar[str] = 'Data Shuffle Options'
    """group shuffle options"""

    shuffle_times: int = argument(
        '--shuffle-times',
        metavar='TIMES',
        default=100,
        validator=lambda it: it > 0,
        group=GROUP_SHUFFLE,
        help='set shuffle times',
    )

    shuffle_method: SHUFFLE_METHOD = argument(
        '--shuffle-method',
        default='cyclic',
        group=GROUP_SHUFFLE,
        help="shuffle method, {'cyclic', 'random'}",
    )

    percentage: float = argument(
        '--percentage',
        metavar='VALUE',
        default=97.5,
        validator=lambda it: 95 < it < 99.5,
        group=GROUP_SHUFFLE,
        help='threshold percentage for the comparison between shuffled data',
    )


class PositionShuffleOptions(ShuffleBaseOptions, Suite2pOptions, TreadmillOptions):
    pass
