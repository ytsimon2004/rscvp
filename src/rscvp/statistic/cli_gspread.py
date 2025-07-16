import subprocess
import sys
import typing
from pathlib import Path

import attrs
import pandas as pd
import polars as pl

from neuralib.typing import DataFrame
from neuralib.util.utils import keys_with_value
from neuralib.util.verbose import fprint, print_load
from rscvp.spatial.main_place_field import split_flatten_lter
from rscvp.util.cli import Region
from rscvp.util.util_dataframe import check_null, to_numeric
from rscvp.util.util_gspread import truncate_before_todo_hash, RSCGoogleWorkSheet, skip_comment_primary_key, \
    GSPREAD_SHEET_PAGE
from stimpyp import Session

__all__ = [
    'GSPExtractor',
    #
    'CliGspreadLUT',
    'CliGspreadGenerator'
]


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

        df = truncate_before_todo_hash(df, primary)
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
                          auto_cast: bool = True,
                          verbose: bool = True) -> pl.DataFrame:
        """
        load from the gspread
        ** Note that each columns in spreadsheet used either str/numeric type. other auto-casting problem

        :param primary_key
        :param auto_cast: cast str to numeric
        :param verbose:
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

        if verbose:
            print(df)

        return df

    def load_parquet_file(self, output: Path,
                          session_melt_header: list[str] | None = None,
                          primary_key: str | tuple[str, ...] = 'Data') -> pl.DataFrame:
        """
        Load directly from the local parquet files

        :param output: for the pickle file for gspread (page-dependent)
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


# ============= #
# CLI Generator #
# ============= #

@attrs.define
class CliGspreadLUT:
    page: GSPREAD_SHEET_PAGE
    """extracted spreadsheet page"""

    cols: list[str]
    """extracted column(s)"""

    module_prefix: str = attrs.field(default='rscvp.statistic.persistence_agg', kw_only=True)
    """module path until dir"""

    file: str = attrs.field(kw_only=True)
    """module filename"""

    group_mode: bool = attrs.field(default=True, kw_only=True)
    """manual group mode on"""

    opt_args: list[str] | str | None = attrs.field(default=None, kw_only=True)
    """optional argument(s)"""

    @property
    def module_path(self) -> str:
        return f'{self.module_prefix}.{self.file}'


@typing.final
class CliGspreadGenerator:
    """Based on info in spreadsheet"""

    def __init__(self,
                 lut: CliGspreadLUT,
                 *,
                 session: Session | None = None,
                 used_session: Session | None = None,
                 region: Region | None = None,
                 batch_mode: bool = True,
                 enable_loop_planes: bool = True,
                 multi_plane_only: bool = False,
                 remote_disk: str | None = None):
        """

        :param lut: `CliGspreadLUT` lookup table
        :param region: Pick up certain region dataset
        :param batch_mode: Batch mode: use ``,`` to run across dataset. non-batch mode use ``;``
        :param enable_loop_planes: Loop every ETL, otherwise, ignore the ``-P`` flag
        :param multi_plane_only: only run the data with ETL
        :param remote_disk: Remote disk name for remotely testing using locally resources
        """
        gs = GSPExtractor(lut.page, lut.cols)

        if region is not None:
            self.df = (gs.load_from_gspread().filter(pl.col('region') == region))
            fprint(f'filter region {region}')
        else:
            self.df = gs.load_from_gspread()

        self.lut = lut
        self.enable_group_mode = lut.group_mode
        self.batch_mode = batch_mode

        # non-batch mode
        self.enable_loop_planes = enable_loop_planes
        self.multiple_plane_only = multi_plane_only

        #
        self.session = session
        self.used_session = used_session
        self.remote_disk = remote_disk

    @property
    def n_dataset(self) -> int:
        return self.df.shape[0]

    def call(self, option_args: list[str] | None = None,
             debug: bool = False,
             **kwargs) -> None:
        """

        :param option_args: Extra options not specified in lut
        :param debug: Debug print
        :param kwargs: additional arguments for ``create_common_cli_batch()``
        :return:
        """
        if self.batch_mode:
            cmds = self._build_cli_batch(option_args, **kwargs)
            print(cmds)
            if not debug:
                subprocess.check_call(cmds)
        else:
            cmds = self._build_cli_foreach(self.enable_loop_planes, self.multiple_plane_only)
            print(cmds)
            if not debug:
                for cmd in cmds:
                    subprocess.check_call(cmd)

    def create_common_cli_batch(
            self,
            date: bool = True,
            animal: bool = True,
            plane: bool = True,
            region: bool = True,
    ) -> list[str]:
        """
        Create the selected cli argument(s) inferred from gspread

        :param date:
        :param animal:
        :param plane:
        :param region:
        :param session:
        :param used_session:
        :return:
        """
        cmds = []
        dat = self.df.get_column('Data').str.split('_').to_list()

        #
        if date:
            cmds.extend(['-D', ','.join([d[0] for d in dat])])
        #
        if animal:
            cmds.extend(['-A', ','.join([d[1] for d in dat])])
        #
        if plane:
            cmds.extend(['-P', self._arg_plane_index()])
        #
        if region:
            cmds.extend(['--region', ','.join(self.df['region'])])

        #
        if self.enable_group_mode:
            from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg
            cmds.extend(['--group'])
            dy = AbstractPersistenceAgg.GROUP_REPR
            cmds.extend(['--as-group'])
            group_int = [str(keys_with_value(dy, r, to_item=True)) for r in self.df['region']]
            cmds.extend([','.join(group_int)])

        #
        if self.session is not None:
            cmds.extend(['--session', self.session])

        if self.used_session is not None:
            cmds.extend(['--used_session', self.used_session])

        if self.remote_disk is not None:
            cmds.extend(['--disk', self.remote_disk])

        return cmds

    def _arg_plane_index(self) -> str:
        p = self.df['n_planes']
        ret = ['0' if it == 1 else '' for it in p]
        return ','.join(ret)

    def _build_cli_foreach(self, loop_planes: bool, multiple_plane_only: bool) -> list[list[str]]:
        """
        Non-batch mode, use ``;`` to sep each bash command

        :param loop_planes: For the ETL image dataset, the command line need to loop through every optic planes or not
                    i.e., True: -D 210315 -A YW006 -P 0;  -D 210315 -A YW006 -P 1; -D 210315 -A YW006 -P 2 ...
                          False: -D 210315 -A YW006
        :param multiple_plane_only:
        :return: list of subprocess command list
        """
        df = self.df
        if multiple_plane_only:
            df = df.filter(pl.col('n_planes') > 1)

        data = df['Data'].str.split('_').to_list()
        planes = df['n_planes']
        regions = df['region']

        ret = []
        for i, dat in enumerate(data):

            date = dat[0]
            animal = dat[1]
            n_planes = planes[i]
            region = regions[i]

            cmds = ['-D', date, '-A', animal, '--region', region]

            opt = self.lut.opt_args
            if loop_planes:
                for plane in range(int(n_planes)):
                    cmds.extend(['-P', f'{plane}'])

                    if opt is not None:
                        opt_args = [opt] if isinstance(opt, str) else opt
                        cmds += opt_args

                    ret.append([sys.executable, '-m', self.lut.module_path] + cmds)
            else:
                if n_planes == 1:
                    cmds.extend(['-P', '0'])

                if opt is not None:
                    opt_args = [opt] if isinstance(opt, str) else opt
                    cmds += opt_args

                ret.append([sys.executable, '-m', self.lut.module_path] + cmds)

        return ret

    def _build_cli_batch(self, option_args: list[str] | None = None, **kwargs) -> list[str]:
        """
        Batch mode, mainly used in statistic
        i.e., -D 210315,210402 -A YW006,YW006 -P 0,0 .....

        :param option_args: more optional arguments for cmds call
        :param kwargs: pass to ``create_common_cli_batch()``
        :return: subprocess command list
        """
        cmds = [sys.executable, '-m', self.lut.module_path]
        cmds.extend(self.create_common_cli_batch(**kwargs))

        # optional
        opt = self.lut.opt_args
        if opt is not None:
            if isinstance(opt, str):
                opt = [opt]
            cmds += opt

        # more optional
        if option_args is not None:
            cmds += option_args

        return cmds
