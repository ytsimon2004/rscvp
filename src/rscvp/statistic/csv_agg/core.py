from pathlib import Path
from typing import Literal, Any

import numpy as np
import polars as pl
from polars.polars import SchemaFieldNotFoundError

from neuralib.util.verbose import fprint, print_save, printdf
from rscvp.util.cli import CodeAlias, HEADER, CELLULAR_IO, CommonOptions
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_gspread import RSCGoogleWorkSheet, filter_tdhash, GSPREAD_SHEET_PAGE

__all__ = [
    'NeuronDataAggregator',
    'ParquetSheetSync'
]


class NeuronDataAggregator:
    """Aggregate the neuronal csv file (based on code) and do the cell mask between different datasets"""

    def __init__(self, code: CodeAlias, *,
                 stat_col: list[str | HEADER],
                 exclude_col: list[str] | None = None,
                 fields: dict[str, Any] = None,
                 truncate_session: bool = True):
        """
        :param code: cli INFO` code
        :param stat_col: columns that used for concat across dataset for statistical purpose.
                         used for remove session info in the original csv
                         ** Note that stat_col should be in a specific order that corresponding to original csv
        :param exclude_col: columns that need to be excluded (useless for statistic)
        :param fields: foreach options, beside animal and exp date
        :param truncate_session: truncate the session for aggregate the csv
        """
        if code not in CELLULAR_IO:
            raise ValueError(f'unknown code: {code}')

        self.code = code
        self.fields = fields or {}

        self.stat_col = ['neuron_id'] + stat_col
        self.exclude_col = exclude_col
        self.truncate_session = truncate_session

    def to_polars(self, opt: StatisticOptions) -> pl.DataFrame:
        return self._to_polars(opt)

    def _to_polars(self, opt: StatisticOptions) -> pl.DataFrame:
        ret = []
        for i, _ in enumerate(opt.foreach_dataset(**self.fields)):

            if opt.plane_index is not None:
                df = self._csv_plane(opt)
            else:
                df = self._csv_concat_plane(opt)
            #
            if opt.session is not None:
                df = df.with_columns(pl.lit(opt.session).alias('session'))

            if opt.rec_region is not None:
                df = df.with_columns(pl.lit(opt.rec_region).alias('region'))

            ret.append(df)

        return pl.concat(ret)

    def _csv_plane(self, opt) -> pl.DataFrame:
        session = opt.session
        file = opt.get_data_output(self.code, session, running_epoch=opt.running_epoch, latest=True).csv_output
        mx = opt.get_selected_neurons()
        df = pl.read_csv(file).filter(mx)

        # exclude unnecessary info (only for non-concat csv)
        if self.exclude_col is not None:
            if session is not None:
                columns = [f'{col}_{session}' for col in self.exclude_col]
            else:
                columns = self.exclude_col

            df = self._try_drop_cols(opt, df, columns)

        # rename column and add info
        if len(df.columns) != len(self.stat_col):
            fprint(f'Mismatch columns {list(df.columns)} and {self.stat_col}', vtype='error')
            raise RuntimeError(f'{opt.exp_date}_{opt.animal_id}: cannot be set, check csv..')

        if self.truncate_session:
            df.columns = self._truncate_session_column(opt, df)

        df = (
            df.with_columns(pl.lit(f'{opt.exp_date}_{opt.animal_id}').alias('Data'))
            .with_columns(pl.lit(f'{opt.plane_index}').alias('plane'))
        )
        return df

    def _csv_concat_plane(self, opt) -> pl.DataFrame:
        session = opt.session
        mx = opt.get_selected_neurons()

        ns = ['neuron_id']
        if session is not None:  # i.e., visual session parameter
            ns.extend(f'{n}_{session}' for n in self.stat_col if n != 'neuron_id')
        else:
            ns.extend(n for n in self.stat_col if n != 'neuron_id')

        pidx = opt.get_csv_data('plane_idx', to_numpy=True)[mx]
        df = opt.get_csv_data(cols=ns, to_numpy=False).filter(mx)

        if self.truncate_session:
            df.columns = self._truncate_session_column(opt, df)

        df = (
            df.with_columns(pl.lit(f'{opt.exp_date}_{opt.animal_id}').alias('Data'))
            .with_columns(pl.Series(pidx, dtype=pl.Utf8).alias('plane'))
        )

        return df

    def _truncate_session_column(self, opt: StatisticOptions, df: pl.DataFrame) -> list[str]:
        """truncate session info in the certain header"""
        s = opt.session
        if s is not None:
            cols = df.columns

            # remove neuron then add back (legacy)
            if 'neuron_id' in cols:
                cols.remove('neuron_id')
            elif 'n' in cols:
                cols.remove('n')
            ret = list(map(lambda x: x[:x.index(s) - 1], cols))
            ret = ['neuron_id'] + ret
        else:
            ret = df.columns

        assert set(ret) == set(self.stat_col)
        return ret

    @staticmethod
    def _try_drop_cols(opt: CommonOptions,
                       df: pl.DataFrame,
                       columns: list[str]) -> pl.DataFrame:
        """i.e., drop_col exclude column (with some legacy naming in csv)"""
        drop_col = []
        for ec in columns:
            if '*' in ec:
                parts = ec.split('*', 1)

                if len(parts) > 2:
                    raise ValueError(f'invalid pattern in {ec}')

                for c in df.columns:
                    if c.startswith(parts[0]):
                        drop_col.append(c)
            else:
                drop_col.append(ec)
        #
        try:
            df = df.drop(drop_col)
        except (KeyError, SchemaFieldNotFoundError):
            msg = f'{opt.exp_date}_{opt.animal_id} >> exc col:{drop_col} not fully found, df columns: {list(df.columns)}'
            fprint(msg, vtype='warning')
        else:
            fprint(f'Drop! {drop_col} in {opt.stimpy_filename}')

        return df


class ParquetSheetSync:
    """Class for Spreadsheet and local parquet dataframe batch data synchronization"""

    _PARQUET_INIT_FROM_GSPREAD = {
        'visual_parq': 'B1',
        'spatial_parq': 'B1',
        'generic_parq': 'B1',
        'dark_parq': 'B1',
        'vr_parq': 'B1:D1'
    }
    """single or range column(s) while init (first time copy from gspread to local parquet)"""

    def __init__(self, opt: StatisticOptions,
                 aggregator: NeuronDataAggregator,
                 sheet_page: GSPREAD_SHEET_PAGE):
        """
        Concat the certain csv file (based on code) and do the cell mask between different datasets

        :param opt:
        :param aggregator:
        :param sheet_page: page name

        """

        self.opt = opt
        self.sheet_page = sheet_page
        self.df = aggregator.to_polars(opt)

        self._aggregator = aggregator

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

    def _save_parquet(self, df: pl.DataFrame,
                      parquet_path: Path | None = None) -> None:
        if parquet_path is None:
            parquet_path = self.local_parquet_file
            verb = 'SAVE'
        else:
            verb = 'UPDATE'

        df.write_parquet(parquet_path)  # TODO check logic
        printdf(df)
        print_save(parquet_path, verb)

    def load_parquet(self, filepath: Path | None = None,
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

    def run_sync(self, header: str,
                 force_parquet: bool = False,
                 as_local: bool = True,
                 dtype: pl.DataType = pl.List(pl.Float64)) -> None:
        """
        **NOTE that if ``filter_concat_df()`` was implemented by children, ``get_pars()`` will used filtered df instead**

        :param header: Variable for statistic
        :param force_parquet: If re-create a new local parquet file
        :param as_local: Whether sync the spreadsheet as local (rollback `TBP` if lack of data)
        :param dtype: expected list of dtype for casting of aggregated data
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

        update_df = self._agg_dataframe(df, primary_list, header, dtype=dtype)

        self._save_parquet(update_df, parquet_path)
        sheet.mark_cell(primary_list, header)

        if as_local:
            self._as_local(sheet, update_df, header)

    def _agg_dataframe(self, df: pl.DataFrame,
                       primary_list: list[str],
                       header: str,
                       dtype: pl.DataType = pl.List(pl.Float64)) -> pl.DataFrame:
        """
        Aggregate batch dataset to a single dataframe (with header in ``pl.List``)

        :param df: Local parquet file to be updated
        :param primary_list: Primary list to be updated
        :param header: Variable name
        :param dtype: expected list of dtype for casting of aggregated data
        :return: Updated dataframe
        """
        for primary in primary_list:
            pars = self.get_pars(primary, header)

            if header in df.columns:  # when column
                # cast existing column to List type if needed to avoid type mismatch
                if df[header].dtype != dtype:
                    # Replace non-null values with empty lists, then update
                    df = df.with_columns(
                        pl.when(pl.col(header).is_null())
                        .then(None)
                        .otherwise([])
                        .cast(dtype)
                        .alias(header)
                    )

                df = df.with_columns(
                    pl.when(pl.col('Data') == primary)
                    .then(pars.tolist())
                    .otherwise(pl.col(header))  # avoid auto fill null
                    .alias(header)
                )
            else:  # first fill auto-cast list column dtype
                df = df.with_columns(
                    pl.when(pl.col('Data') == primary)
                    .then(pars.tolist())
                    .otherwise([])
                    .alias(header)
                )

        return df

    def _init_parquet(self, sheet: RSCGoogleWorkSheet, error_when_empty: bool = True) -> pl.DataFrame:
        """used to add necessary information in parquet file while first generating"""
        parquet_df = pl.DataFrame(pl.Series(values=sheet.primary_key_list, name='Data'))
        idx, parquet_df = filter_tdhash(parquet_df, return_index=True)

        wp = RSCGoogleWorkSheet.of_work_page(sheet.title)
        a1_range = self._PARQUET_INIT_FROM_GSPREAD[sheet.title]
        parquet_cols = wp.get_range_value(a1_range)

        for col in parquet_cols:
            filled = sheet.values(col)[:idx]
            if error_when_empty:
                if np.any([len(it) == 0 for it in filled]):
                    raise RuntimeError(f'empty cell were found while init the {sheet.title} >>> {filled}')

            parquet_df = parquet_df.with_columns(pl.Series(values=filled).alias(col))

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
