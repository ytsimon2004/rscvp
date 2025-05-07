from argclz.commands import parse_command_args
from .main_roi_quant import RoiQuantOptions
from .main_roi_quant_batch import RoiQuantBatchOptions
from .main_roi_query import RoiQueryOptions
from .main_roi_query_batch import RoiQueryBatchOptions
from .main_roi_top_view import RoiTopViewOptions
from .main_ternary import TernaryPercOptions

parse_command_args(
    usage='python -m rscvp.atlas [CMD] ...',
    description='atlas cmd',
    parsers=dict(
        quant=RoiQuantOptions,
        ternary=TernaryPercOptions,
        query=RoiQueryOptions,
        top=RoiTopViewOptions,
        batch_quant=RoiQuantBatchOptions,
        batch_query=RoiQueryBatchOptions
    )
)
