from argclz.commands import parse_command_args
from .main_tactile_summary import TactileSummaryOptions

parse_command_args(
    usage='python -m rscvp.behavioral CMD ...',
    description='Analysis of behavioral results',
    parsers=dict(bs=TactileSummaryOptions)
)
