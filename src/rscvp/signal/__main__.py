from argclz.commands import parse_command_args
from .main_dff_session import DffSesOption

parse_command_args(
    usage='python -m rscvp.signal CMD ...',
    description='raw calcium signal information',
    parsers=dict(ds=DffSesOption)
)
