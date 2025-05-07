import abc
import datetime
from typing import Any, TypeVar, Generic

import polars as pl
from rscvp.util.database import RSCDatabase
from rscvp.util.util_dataframe import to_numeric

from argclz import argument
from neuralib.sqlp.util import datetime_to_str
from neuralib.tools.gspread import WorkPageName
from neuralib.util.verbose import fprint
from .cli_core import CommonOptions

__all__ = ['SQLDatabaseOptions']

T = TypeVar('T')


class SQLDatabaseOptions(CommonOptions, Generic[T], metaclass=abc.ABCMeta):
    """Generic T = supporting GenericDB | BayesDecodeDB | VisualSFTFDirDB"""

    db_commit: bool = argument('--commit', help='commit database operations')

    db_debug_mode: bool = argument('--db-debug', help='debug mode for database operations')

    _gspread_dataframe: pl.DataFrame = None  # cache

    @property
    def cur_time(self) -> str:
        return datetime_to_str(datetime.datetime.now().replace(microsecond=0))

    def get_primary_key_field(self, field: str, auto_cast: bool = True, page: WorkPageName = 'apcls_tac') -> Any:
        """
        Specify a `field name` and get a cell from the primary key

        :param field: field(header) in the gspread worksheet
        :param auto_cast: auto cast numerical
        :param page: Page name
        :return:
        """
        if self._gspread_dataframe is None:
            from rscvp.util.util_gspread import RSCGoogleWorkSheet
            self._gspread_dataframe = RSCGoogleWorkSheet.of_work_page(page).to_polars()

        df = self._gspread_dataframe
        if field not in df.columns:
            raise KeyError(f'{field} field not found in the gspread')

        if auto_cast:
            df = pl.select(to_numeric(s) for s in df)

        primary_key = f'{self.exp_date}_{self.animal_id}'

        return df.filter(pl.col('Data') == primary_key)[field].item()

    @abc.abstractmethod
    def populate_database(self, *args, **kwargs) -> None:
        pass

    def add_data(self, db: T):
        """
        Add data to the database

        :param db: Type of the database
        """
        database = RSCDatabase()
        if self.db_debug_mode:
            database._debug_mode = True
            fprint('Database DEBUG ENABLED!', vtype='debug')

        database.add_data(db)

    def update_data(self, db: T, *arg):
        database = RSCDatabase()
        if self.db_debug_mode:
            database._debug_mode = True
            fprint('Database DEBUG ENABLED!', vtype='debug')

        database.update_data(db, *arg)
