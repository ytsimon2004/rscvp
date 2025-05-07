from argclz import parse_command_args

from .main_lick_score import LickScoreOptions
from .main_licking_cmp import LickingCmpOptions

parse_command_args(
    usage='python -m rscvp.track CMD...',
    description='post processing of tracking data using DeepLabCut',
    parsers=dict(
        lc=LickingCmpOptions,
        ls=LickScoreOptions,
    )
)
