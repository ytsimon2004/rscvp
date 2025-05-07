from argclz.commands import parse_command_args
from .main_summary import BehaviorSumOptions

parse_command_args(
    usage='python -m rscvp.behavioral CMD ...',
    description='Analysis of behavioral results',
    parsers=dict(bs=BehaviorSumOptions)
)
