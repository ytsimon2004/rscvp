import abc
import datetime
from typing import Any, TypeVar, Generic

import polars as pl
from rich.pretty import pprint

from argclz import argument
from neuralib.tools.gspread import WorkPageName
from neuralib.util.verbose import fprint
from rscvp.util.database import RSCDatabase
from rscvp.util.util_dataframe import to_numeric
from rscvp.util.util_gspread import filter_tdhash
from sqlclz.util import datetime_to_str
from .cli_core import CommonOptions

__all__ = ['SQLDatabaseOptions']

T = TypeVar('T')


class SQLDatabaseOptions(CommonOptions, Generic[T], metaclass=abc.ABCMeta):
    db_commit: bool = argument('--commit', help='commit database operations')

    db_debug_mode: bool = argument('--db-debug', help='debug mode for database operations')  # TODO maybe disable

    _gspread_dataframe: pl.DataFrame | None = None  # cache

    @property
    def cur_time(self) -> str:
        """current time, used for database populate/update"""
        return datetime_to_str(datetime.datetime.now().replace(microsecond=0))

    def fetch_gspread(self, field: str,
                      auto_cast: bool = True,
                      page: WorkPageName = 'fov_table',
                      default_value: Any | None = None) -> Any:
        """
        Specify a ``field name`` and get a cell from the primary key

        :param field: field(header) in the gspread worksheet
        :param auto_cast: auto cast numerical
        :param page: Page name
        :param default_value: default value if error
        :return:
        """
        if self._gspread_dataframe is None:
            from rscvp.util.util_gspread import RSCGoogleWorkSheet
            self._gspread_dataframe = RSCGoogleWorkSheet.of_work_page(page).to_polars()

        df = self._gspread_dataframe
        df: pl.DataFrame = filter_tdhash(df, return_index=False)

        if field not in df.columns:
            raise KeyError(f'{field} field not found in the gspread')

        if auto_cast:
            df = pl.select(to_numeric(s) for s in df)

        data = f'{self.exp_date}_{self.animal_id}'
        ret = df.filter(pl.col('Data') == data)[field].item()

        return ret or default_value

    @abc.abstractmethod
    def write_database(self, *args, **kwargs) -> None:
        """write results into SQL database"""
        pass

    def replace_data(self, db: T):
        """add data to the database

        :param db: Type of the database
        """
        database = RSCDatabase()
        if self.db_debug_mode:
            database._debug_mode = True
            fprint('Database DEBUG ENABLED!', vtype='debug')
        else:
            database.replace_data(db)

    def update_data(self, db: T, *arg):
        """update data in the database

        :param db: Type of the database
        """
        database = RSCDatabase()
        if self.db_debug_mode:
            database._debug_mode = True
            fprint('Database DEBUG ENABLED!', vtype='debug')
        else:
            database.update_data(db, *arg)

    def print_replace(self, db: T):
        print(f'# ----- REPLACE DATA IN: {db.__class__.__name__} ----- #')
        pprint(db)

        if self.db_commit:
            self.replace_data(db)
        else:
            print('use "--commit" to perform database operations')

    def print_update(self, db: T, **update_fields):
        print(f'# ----- UPDATE DATA TO: {db.__class__.__name__} ----- #')
        pprint(update_fields)

        if self.db_commit:
            self.update_data(db, *update_fields.keys())
        else:
            print('use --commit to perform database operations')
