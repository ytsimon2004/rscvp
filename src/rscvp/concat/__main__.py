from argclz.commands import parse_command_args
from .main_csv_cell import ConcatCellCSVOptions

parse_command_args(
    usage='python -m rscvp.concate_plane CMD ...',
    description='concate analysis file across multiple planes',
    parsers=dict(
        csv=ConcatCellCSVOptions,
    )
)
