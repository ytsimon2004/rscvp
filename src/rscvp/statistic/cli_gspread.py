import typing
from pathlib import Path

import pandas as pd
import polars as pl

from neuralib.typing import DataFrame
from neuralib.util.verbose import fprint, print_load
from rscvp.spatial.main_place_field import split_flatten_lter
from rscvp.util.util_dataframe import check_null, to_numeric
from rscvp.util.util_gspread import filter_tdhash, RSCGoogleWorkSheet, skip_comment_primary_key, \
    GSPREAD_SHEET_PAGE

__all__ = ['GSPExtractor']


# TODO might not need this layer and move to other module path
@typing.final
class GSPExtractor:
    """Extract dataset from gspread OR local parquet file"""

    def __init__(self, sheet_name: GSPREAD_SHEET_PAGE,
                 cols: list[str] | None = None):
        """
        Class for extract the partial data from Google spreadsheet for statistic purpose

        :param sheet_name: page name of the google sheet.
        :param cols: column(s) to be extracted. If None, extract all columns
        """
        self.sheet_name = sheet_name
        self._cols = cols

    # ========== #
    # Preprocess #
    # ========== #

    def _preprocess(self, df: DataFrame, primary: str = 'Data') -> pl.DataFrame:
        """
        preprocess the dataset. get rid of the illegal pattern foreach cell.

        :param df: if not specified, use data from the gspread
        :param primary:
        :return:
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        if self._cols is not None:
            df = df.select(*self._cols)

        df = filter_tdhash(df, primary)
        df = skip_comment_primary_key(df, primary)
        df = self._parse_string_to_numerical_array(df)

        check_null(df)

        return df

    @staticmethod
    def _parse_string_to_numerical_array(df: pl.DataFrame) -> pl.DataFrame:
        """multiple values for each cell, then originally store as string array with space in csv"""
        _special = ('pf_width', 'pf_peak')
        # replace original col
        for it in _special:
            if it in df.columns:
                df = df.with_columns(
                    pl.col(it)
                    .map_elements(lambda x: split_flatten_lter(x, to_numpy=False, dtype=float),
                                  return_dtype=pl.List(pl.Float64))
                    .alias(it)
                )

        return df

    # ======================= #
    # Test GS / Parquet files #
    # ======================= #

    def load_from_gspread(self, primary_key: str = 'Data',
                          auto_cast: bool = True) -> pl.DataFrame:
        """
        load from the gspread
        ** Note that each columns in spreadsheet used either str/numeric type. other auto-casting problem

        :param primary_key
        :param auto_cast: cast str to numeric
        :return:
        """
        work_sheet = RSCGoogleWorkSheet.of_work_page(self.sheet_name, primary_key=primary_key)
        df = work_sheet.to_polars()
        fprint('LOAD data from GSpread!', vtype='io')

        # preprocess
        if isinstance(primary_key, tuple):
            fprint('tuple type of primary key is not able to preprocess', vtype='warning')
        else:
            df = self._preprocess(df, primary=primary_key)

        # casting
        if auto_cast:
            df = (pl.select(to_numeric(s) for s in df))

            for null_series in df.null_count():
                if null_series.item() != 0:
                    fprint(f'{null_series.name} contain null', vtype='warning')

            # index field cast back
            if isinstance(df[primary_key].dtype, pl.Float64):
                df = df.cast({primary_key: pl.Int64}).cast({primary_key: pl.Utf8})

        return df

    def load_parquet_file(self, output: Path,
                          session_melt_header: list[str] | None = None,
                          primary_key: str | tuple[str, ...] = 'Data') -> pl.DataFrame:
        """
        Load directly from the local parquet files

        :param output: for the parquet file for gspread (page-dependent)
        :param session_melt_header: specify if melt_session is True, headers(vars) to be melted
        :param primary_key:
        :return:
        """
        file = output / f'{self.sheet_name}.parquet'
        df = pl.read_parquet(file)
        print_load(file)

        #
        if isinstance(primary_key, tuple):
            fprint('tuple type of primary key is not able to preprocess', vtype='warning')
        else:
            df = self._preprocess(df, primary=primary_key)

        if session_melt_header is not None:
            df = melt_session(df, session_melt_header)

        return df


def melt_session(df: pl.DataFrame, var: str | list[str]) -> pl.DataFrame:
    """
    (1, S) -> (S, 1) dataframe

    :param df: (1, S) dataframe
    :param var: variable name without session to be melted
    :return:
    """
    if isinstance(var, str):
        return _melt_session(df, var)
    elif isinstance(var, list):
        return pl.concat([_melt_session(df, v) for v in var])
    else:
        raise TypeError(f'var type: {type(var)}')


def _melt_session(df: pl.DataFrame, var: str) -> pl.DataFrame:
    """
    (1, S) -> (S, 1) dataframe

    **Example**

    :param df: (1, S) dataframe
    :param var: variable name without session to be melted
    :return:
    """
    value_vars = [c for c in df.columns if c.startswith(var)]
    df = (
        df.melt(id_vars='Data', value_vars=value_vars, value_name=var)
        .with_columns(
            pl.col('variable').map_elements(lambda x: x[len(var) + 1:], return_dtype=pl.Utf8).alias('session'))
        .drop('variable')
        .sort('Data')
    )

    return df
