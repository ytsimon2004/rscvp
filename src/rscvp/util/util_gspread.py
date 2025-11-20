from pathlib import Path
from typing import Literal, ClassVar, Final

import polars as pl
from typing_extensions import Self

from neuralib.tools.gspread import GoogleWorkSheet, GoogleSpreadSheet
from neuralib.typing import DataFrame
from neuralib.util.verbose import fprint
from rscvp.util.io import IOConfig

__all__ = [
    'GSPREAD_SHEET_PAGE',
    'USAGE_TYPE',
    'RSCGoogleWorkSheet',
    'GoogleSpreadSheet',
    #
    'filter_tdhash',
    'skip_comment_primary_key',
]

GSPREAD_SHEET_PAGE = Literal[
    'fov_table',
    'visual_parq', 'spatial_parq', 'dark_parq', 'generic_parq', 'vr_parq',

    'FieldOfViewDB',
    'BaseClassDB', 'DarkClassDB', 'BlankClassDB', 'VRClassDB',
    'BayesDecodeDB', 'VisualSFTFDirDB'
]

USAGE_TYPE = Literal['base', 'blank', 'dark', 'vr']


class RSCGoogleWorkSheet(GoogleWorkSheet):
    _service_account_path: ClassVar[Path] = IOConfig.DEFAULT_GSPREAD_AUTH

    FIRST_COLUMN_NAME: Final[str] = 'Data'

    @classmethod
    def of_work_page(cls, page: GSPREAD_SHEET_PAGE,
                     primary_key: str | tuple[str, ...] = 'Data',
                     debug: bool = False) -> Self:

        sh = GoogleSpreadSheet(
            'Test_YWAnalysis' if debug else 'YWAnalysis',
            cls._service_account_path
        )

        # noinspection PyProtectedMember
        for ws in sh._worksheets:
            if ws.title == page:
                return RSCGoogleWorkSheet(ws, primary_key)

        raise ValueError(f'page not found: {page}')

    @property
    def valid_primary_key(self) -> list[str]:
        df = filter_tdhash(self.to_polars(), self.FIRST_COLUMN_NAME)
        return df[self.FIRST_COLUMN_NAME].to_list()

    def mark_cell(self, primary_list: list[str],
                  header: str,
                  mark_as: str | None = None):
        """
        Mark google spreadsheet as `PARQUET` after updating(merging) the pickle file

        :param primary_list: list of `primary key` for updating
        :param header: variable
        :param mark_as: information showed in the Google spreadsheet after pickled
        :return:
        """

        if mark_as is None:
            from datetime import datetime
            mark_as = f"parq_{datetime.today().strftime('%y%m%d')}"

        values = [mark_as] * len(primary_list)
        self.update_cell(primary_list, header, values)


def filter_tdhash(df: DataFrame,
                  index_field: str = 'Data',
                  return_index: bool = False) -> DataFrame | tuple[int, DataFrame]:
    """Try truncate the dataframe/gspread before **#TODO** in the first field of dataframe"""
    try:
        idx = df[index_field].to_list().index('#TODO')
        return (idx, df[:idx]) if return_index else df[:idx]
    except ValueError:
        return (None, df) if return_index else df


def skip_comment_primary_key(df: pl.DataFrame, primary_key: str = 'Data') -> pl.DataFrame:
    """return list of index(data) had #"""
    expr = pl.col(primary_key).cast(pl.Utf8).str.starts_with('#')
    ignore_primary = df.filter(expr)[primary_key].to_list()
    if len(ignore_primary) > 0:
        fprint(f'Skip -> comment data: {ignore_primary}', vtype='warning')
        return df.filter(~expr)
    else:
        return df
