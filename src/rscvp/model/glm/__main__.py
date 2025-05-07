from argclz.commands import parse_command_args
from .main_eval import BehavioralGLMOptions

parse_command_args(
    usage='python -m rscvp.model CMD ...',
    description='do the related analysis for the modeling',
    parsers=dict(
        lnp=BehavioralGLMOptions
    )
)
