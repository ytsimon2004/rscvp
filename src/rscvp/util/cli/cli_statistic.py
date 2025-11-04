from typing import ClassVar

from argclz import argument, as_argument
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE
from .cli_core import CommonOptions
from .cli_io import HEADER
from .cli_selection import SelectionOptions

__all__ = ['StatisticOptions']


class StatisticOptions(SelectionOptions):
    GROUP_STAT: ClassVar[str] = 'statistical options'

    exp_date = as_argument(CommonOptions.exp_date).with_options(required=False)
    animal_id = as_argument(CommonOptions.animal_id).with_options(required=False)

    # ----Header(Variable)---- #

    header: HEADER = argument(
        '-H', '--header',
        required=True,
        group=GROUP_STAT,
        help='which kind of variables (header) for statistic without the session'
    )

    # ----Aggregate---- #

    truncate_session_agg: bool = argument(
        '--trunc-session',
        group=GROUP_STAT,
        help='whether truncate the session for aggregate the csv'
             'use case: in csv has session <HEADER>_light, but in gspread only <HEADER> (i.e., dark_parq)'
    )

    # ----SpreadSheet---- #

    sheet_name: GSPREAD_SHEET_PAGE = argument(
        '--page',
        group=GROUP_STAT,
        help='which google spread sheet page name, especially used for agg'
    )

    update: bool = argument(
        '--update',
        group=GROUP_STAT,
        help='whether update the spreadsheet and save the parquet file for summary data',
    )

    @property
    def variable(self) -> str:
        if self.session is not None and not self.truncate_session_agg:
            parts = self.session.split(',')
            if len(set(parts)) > 1:
                raise RuntimeError('different session not supported')

            return f'{self.header}_{parts[0]}'
        return self.header
