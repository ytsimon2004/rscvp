from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from rscvp.statistic.csv_agg.collector import CSVCollector
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_gspread import RSCGoogleWorkSheet, truncate_before_todo_hash, GSPREAD_SHEET_PAGE

from neuralib.util.verbose import fprint

__all__ = ['LocalSpreadsheetSync']

# headers to sync from gspread to parqueted files while init (first time copy), not the pickled cell
# workpage: range
_INIT_DEFAULT_COLS_GSPREAD_PARQUET = {
    'ap_vz': 'B1:E1',
    'ap_place': 'B1:E1',
    'ap_generic': 'B1:E1',
    #
    'ap_ldl': 'B1:E1'
}


def _get_default_header(work_page: GSPREAD_SHEET_PAGE) -> list[str]:
    """for init parquet file from gspread"""
    wp = RSCGoogleWorkSheet.of_work_page(work_page)
    a1_range = _INIT_DEFAULT_COLS_GSPREAD_PARQUET[work_page]

    return wp.get_range_value(a1_range)


class LocalSpreadsheetSync:
    """Class for Spreadsheet and local parquet dataframe batch data synchronization"""

    def __init__(self, opt: StatisticOptions,
                 collector: CSVCollector,
                 sheet_page: GSPREAD_SHEET_PAGE):
        """
        Concat the certain csv file (based on code) and do the cell mask between different datasets

        :param opt:
        :param collector:
        :param sheet_page: page name

        """

        self.opt = opt
        self.code = collector.code
        self.sheet_page = sheet_page
        self.df = collector.find_all(opt)

    def __contains__(self, item: str):
        return item in self.df['Data']

    def __getitem__(self, item: Literal['region', 'session', 'Data']) -> list[str]:
        """Get unique column value from the aggregated dataset"""
        return self.df[item].unique().to_list()

    @property
    def worksheet(self) -> RSCGoogleWorkSheet:
        return RSCGoogleWorkSheet.of_work_page(self.sheet_page)

    @property
    def local_parquet_file(self) -> Path:
        """dataframe parquet synced with gspread sheet (but storing arr/list in cell)"""
        return (self.opt.statistic_dir / self.sheet_page).with_suffix('.parquet')

    def filter_concat_df(self, **kwargs) -> pl.DataFrame:
        """if the df for sync from concat csv to gspread/parquet file NEED to process,
        i.e., different headers, filter information...
        """
        pass

    def get_pars(self, primary: str, header: str) -> np.ndarray:
        """
        get population data based on `primary key` and `header`

        :param primary: data for updating
        :param header: variable
        :return:
        """
        filter_df = self.filter_concat_df()
        df = filter_df if filter_df is not None else self.df
        return df.filter(pl.col('Data') == primary)[header].to_numpy()

    def _save_parquet(self,
                      df: pl.DataFrame,
                      parquet_path: Path | None = None,
                      verbose: bool = False) -> None:
        if parquet_path is None:
            parquet_path = self.local_parquet_file
        df.write_parquet(parquet_path)

        if verbose:
            fprint(f'save parquet in {parquet_path}...', vtype='io')
            print(df)

    def load_parquet(self,
                     filepath: Path | None = None,
                     verbose: bool = False) -> pl.DataFrame:
        if filepath is None:
            filepath = self.local_parquet_file

        df = pl.read_parquet(filepath)

        if verbose:
            fprint(f'Load parquet in {filepath}...', vtype='io')
            print(df)

        return df

    # ================== #
    # Aggregate / Update #
    # ================== #

    def update_spreadsheet(self,
                           header: str,
                           force_parquet: bool = False,
                           as_local: bool = True) -> None:
        """
        **NOTE that if ``filter_concat_df()`` was implemented by children, ``get_pars()`` will used filtered df instead**

        :param header: Variable for statistic
        :param force_parquet: If re-create a new local parquet file
        :param as_local: Whether sync the spreadsheet as local (rollback `TBP` if lack of data)
        """
        primary_list = self['Data']  # unique value to be updated (from self.df)
        sheet = self.worksheet
        parquet_path = self.local_parquet_file

        #
        if parquet_path.exists() and not force_parquet:
            df = self.load_parquet(parquet_path)
            self._validate(sheet, df)
        else:
            df = self._init_parquet(sheet)

        update_df = self._agg_dataframe(df, primary_list, header)

        self._save_parquet(update_df, parquet_path)
        sheet.mark_cell(primary_list, header)

        if as_local:
            self._as_local(sheet, update_df, header)

    def _agg_dataframe(self,
                       df: pl.DataFrame,
                       primary_list: list[str],
                       header: str) -> pl.DataFrame:
        """
        Aggregate batch dataset to a single dataframe (with header in ``pl.List``)

        :param df: Local parquet file to be updated
        :param primary_list: Primary list to be updated
        :param header: Variable name
        :return: Updated dataframe
        """

        for info in primary_list:
            pars = self.get_pars(info, header)

            if header in df.columns:
                df = df.with_columns(pl.when(pl.col('Data') == info)
                                     .then(pars.tolist())
                                     .otherwise(pl.col(header))  # avoid auto fill null
                                     .alias(header))
            else:  # first fill auto-cast list column dtype
                df = df.with_columns(pl.when(pl.col('Data') == info)
                                     .then(pars.tolist())
                                     .alias(header))

        return df

    @staticmethod
    def _init_parquet(sheet: RSCGoogleWorkSheet, error_when_empty: bool = True) -> pl.DataFrame:
        """used to add necessary information in parquet file while first generating"""
        parquet_df = pl.DataFrame(pl.Series(values=sheet.primary_key_list, name='Data'))
        idx, parquet_df = truncate_before_todo_hash(parquet_df, return_index=True)

        for default_col in _get_default_header(sheet.title):
            filled = sheet.values(default_col)[:idx]  # gspread existing
            if error_when_empty:
                if np.any([len(it) == 0 for it in filled]):
                    raise RuntimeError(f'empty cell were found while init the {sheet.title} >>> {filled}')

            parquet_df = parquet_df.with_columns(pl.Series(values=filled).alias(default_col))

        return parquet_df

    @staticmethod
    def _as_local(sheet: RSCGoogleWorkSheet, df: pl.DataFrame, header: str) -> None:
        expr = pl.col(header).is_null()
        rollback_primary_keys = df.filter(expr)['Data'].to_list()
        sheet.mark_cell(rollback_primary_keys, header, mark_as='TBP')

    @staticmethod
    def _validate(sheet: RSCGoogleWorkSheet, df: pl.DataFrame) -> None:
        """
        Check if any new data in google sheet but not in parquet file

        :param sheet: ``RSCGoogleWorkSheet``
        :param df: local parquet file
        """
        spreadsheet_data = set(sheet.valid_primary_key)
        local_data = set(df['Data'])

        if len(spreadsheet_data) != len(local_data):
            raise RuntimeError('gspread sheet has new data(info)'
                               f', sheet: {spreadsheet_data}\n'
                               f'parquet: {local_data}\n'
                               f'please remake a parquet file')
