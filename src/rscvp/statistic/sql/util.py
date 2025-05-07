import pandas as pd
import polars as pl
from neuralib.typing import DataFrame
from rscvp.util.util_gspread import (
    GSPREAD_SHEET_PAGE,
    RSCGoogleWorkSheet,
    truncate_before_todo_hash,
    skip_comment_primary_key
)

__all__ = ['as_validate_sql_table']


def as_validate_sql_table(df: DataFrame, src_page: GSPREAD_SHEET_PAGE) -> pl.DataFrame:
    """
    Filter database table to valid data based on source gspread page

    :param df: db table or gspread sheet table
    :param src_page: source gspread page name
    :return: filtered table
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    src = RSCGoogleWorkSheet.of_work_page(src_page, primary_key='Data').to_polars()
    src = truncate_before_todo_hash(src, 'Data')
    src = skip_comment_primary_key(src, 'Data')
    valid_keys = src['Data'].to_list()

    return df.filter((pl.col('date') + '_' + pl.col('animal')).is_in(valid_keys))
