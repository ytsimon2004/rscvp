from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import NamedTuple, Optional, Annotated, Type, Literal, overload

import polars as pl

from neuralib import sqlp
from neuralib.tools.gspread import upload_dataframe_to_spreadsheet, SpreadSheetName
from neuralib.util.verbose import fprint
from rscvp.util.io import IOConfig

__all__ = [
    'DB_TYPE',
    'DataBaseType',
    #
    'PhysiologyDB',
    'GenericDB',
    'DarknessGenericDB',
    'BlankBeltGenericDB',
    'VRGenericDB',
    #
    'BayesDecodeDB',
    'VisualSFTFDirDB',
    #
    'RSCDatabase',
]

DB_TYPE = Literal[
    'GenericDB', 'DarknessGenericDB', 'BlankBeltGenericDB', 'VRGenericDB',
    'BayesDecodeDB', 'VisualSFTFDirDB'
]


@sqlp.named_tuple_table_class
class PhysiologyDB(NamedTuple):
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    @property
    def dir_name(self) -> str:
        return f'{self.date}_{self.animal}__{self.rec}_{self.user}_{self.optic}'

    @classmethod
    def parse_dir_structure(cls, directory: Path) -> list[PhysiologyDB]:
        matches = re.match(r'(\d+)_(\w+)__(\w+)_(\w+)', directory.name)
        if matches:
            date, animal, rec, user = matches.groups()

            optics = ['all']  # for concat data
            p = list(directory.glob('suite2p/plane*'))
            if len(p) != 0:
                for pp in p:
                    optics.extend(pp.name[5:])
                return [PhysiologyDB(date, animal, rec, user, o) for o in optics]
            else:
                fprint(f'no registered file found in {directory.name}', vtype='warning')
                return []
        else:
            fprint(f'{directory.name} invalid', vtype='error')
            return []


@sqlp.named_tuple_table_class
class GenericDB(NamedTuple):
    """Generic Value Data"""
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    # Statistic
    n_planes: Optional[int] = None
    region: Optional[str] = None
    pair_wise_group: Optional[int] = None

    n_total_neurons: Optional[int] = None
    n_selected_neurons: Optional[int] = None
    n_visual_neurons: Optional[int] = None
    n_spatial_neurons: Optional[int] = None
    n_overlap_neurons: Optional[int] = None

    update_time: Optional[datetime.datetime] = None

    @sqlp.foreign(PhysiologyDB)
    def _animal(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlp.named_tuple_table_class
class BayesDecodeDB(NamedTuple):
    """Bayes Decode Data"""
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    # Statistic
    region: Optional[str] = None
    pair_wise_group: Optional[int] = None

    n_neurons: Optional[int] = None
    spatial_bins: Optional[float] = None
    temporal_bins: Optional[float] = None
    median_decode_error: Optional[float] = None
    cross_validation: Optional[str] = None
    """CrossValidateType {'odd', 'even', 'random_split', int}"""
    update_time: Optional[datetime.datetime] = None

    @sqlp.foreign(PhysiologyDB)
    def _animal(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlp.named_tuple_table_class
class VisualSFTFDirDB(NamedTuple):
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    # Statistic
    region: Optional[str] = None
    pair_wise_group: Optional[int] = None

    # Order followed by ``SFTF_ARRANGEMENT``
    sftf_amp_group1: Optional[float] = None
    sftf_amp_group2: Optional[float] = None
    sftf_amp_group3: Optional[float] = None
    sftf_amp_group4: Optional[float] = None
    sftf_amp_group5: Optional[float] = None
    sftf_amp_group6: Optional[float] = None
    sftf_frac_group1: Optional[float] = None
    sftf_frac_group2: Optional[float] = None
    sftf_frac_group3: Optional[float] = None
    sftf_frac_group4: Optional[float] = None
    sftf_frac_group5: Optional[float] = None
    sftf_frac_group6: Optional[float] = None

    n_ds_neurons: Optional[int] = None
    """Number of direction selective neurons"""
    n_os_neurons: Optional[int] = None
    """Number of orientation selective neurons"""
    ds_frac: Optional[float] = None
    """Fraction of direction selective"""
    os_frac: Optional[float] = None
    """Fraction of orientation selective"""

    update_time: Optional[datetime.datetime] = None

    @sqlp.foreign(PhysiologyDB)
    def _animal(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlp.named_tuple_table_class
class DarknessGenericDB(NamedTuple):
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    #
    n_planes: Optional[int] = None
    region: Optional[str] = None
    n_total_neurons: Optional[int] = None
    n_selected_neurons: Optional[int] = None
    n_spatial_neurons_light_bas: Optional[int] = None
    n_spatial_neurons_dark: Optional[int] = None
    n_spatial_neurons_light_end: Optional[int] = None
    update_time: Optional[datetime.datetime] = None

    @sqlp.foreign(PhysiologyDB)
    def _animal(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlp.named_tuple_table_class
class BlankBeltGenericDB(NamedTuple):
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    region: Optional[str] = None
    pair_wise_group: Optional[int] = None
    n_total_neurons: Optional[int] = None
    n_selected_neurons: Optional[int] = None
    n_spatial_neurons: Optional[int] = None
    update_time: Optional[datetime.datetime] = None

    @sqlp.foreign(PhysiologyDB)
    def _animal(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlp.named_tuple_table_class
class VRGenericDB(NamedTuple):
    date: Annotated[str, sqlp.PRIMARY]
    animal: Annotated[str, sqlp.PRIMARY]
    rec: Annotated[str, sqlp.PRIMARY]
    user: Annotated[str, sqlp.PRIMARY]
    optic: Annotated[str, sqlp.PRIMARY]

    region: str | None = None
    pair_wise_group: int | None = None
    n_total_neurons: int | None = None
    n_selected_neurons: int | None = None
    n_spatial_neurons: int | None = None
    update_time: datetime.datetime | None = None

    @sqlp.foreign(PhysiologyDB)
    def _animal(self):
        return self.date, self.animal, self.rec, self.user, self.optic


DataBaseType = GenericDB | BayesDecodeDB | VisualSFTFDirDB | DarknessGenericDB | BlankBeltGenericDB | VRGenericDB


class RSCDatabase(sqlp.Database):
    _debug_mode = False

    @property
    def database_file(self) -> Path:
        directory = Path(__file__).parents[3] / 'res' / 'database'
        if not self._debug_mode:
            return directory / 'rscvp.db'
        else:
            return directory / '_test_rscvp.db'

    @property
    def database_tables(self) -> list[type]:
        return [PhysiologyDB, GenericDB, BayesDecodeDB, VisualSFTFDirDB, DarknessGenericDB, BlankBeltGenericDB]

    # ========= #
    # AnimalExp #
    # ========= #

    def list_animal_names(self) -> list[str]:
        with self.open_connection():
            results = sqlp.select_from(PhysiologyDB.animal, distinct=True).fetchall()
        return sqlp.util.take(0, results)

    def list_date_animals(self, date: str) -> list[str]:
        with self.open_connection():
            results = sqlp.select_from(PhysiologyDB.animal, distinct=True).where(
                PhysiologyDB.date == date
            ).fetchall()
        return sqlp.util.take(0, results)

    def list_animal_dates(self, animal: str) -> list[str]:
        with self.open_connection():
            results = sqlp.select_from(PhysiologyDB.date, distinct=True).where(
                PhysiologyDB.animal == animal
            ).fetchall()
        return sqlp.util.take(0, results)

    @overload
    def find_physiological_data(self, *, date: str = None,
                                animal: str = None,
                                rec: str = None,
                                user: str = None,
                                optic: str = None) -> list[PhysiologyDB]:
        pass

    def find_physiological_data(self, **kwargs) -> list[PhysiologyDB]:
        with self.open_connection():
            return (
                sqlp.select_from(PhysiologyDB)
                .where(*[getattr(PhysiologyDB, k) == v for k, v in kwargs.items()])
                .fetchall()
            )

    def list_dirs(self) -> list[PhysiologyDB]:
        with self.open_connection():
            return sqlp.select_from(PhysiologyDB).fetchall()

    def import_new_animals(self, animals: list[PhysiologyDB]):
        with self.open_connection():
            sqlp.insert_into(PhysiologyDB, policy='REPLACE').submit(animals)

    def import_new_animal_from_directory(self, root: Path):
        animals = []
        for file in root.iterdir():
            animals.extend(PhysiologyDB.parse_dir_structure(file))

        self.import_new_animals(animals)

    # ================== #
    # Calcium Image Data #
    # ================== #

    @staticmethod
    def select_foreign_from_source(foreign_db: type[DataBaseType],
                                   source: PhysiologyDB) -> sqlp.Cursor[DataBaseType]:
        return sqlp.util.pull_foreign(foreign_db, source)

    def add_data(self, data: DataBaseType):
        """add new data"""
        with self.open_connection():
            sqlp.insert_into(type(data), policy='REPLACE').submit([data])

    def update_data(self, data: DataBaseType, *args: str):
        """
        update the data in database for those with matched primary keys.

        If you want to change the primary keys for any data, please use
        raw sql statements, because this method does not allow this action.

        :param data: updating data.
        :param args: name of field that needs to be updated.
            If empty, update all non-primary, null-able keys with non-None value.
        """
        table = type(data)

        primary = [it.name for it in sqlp.table_primary_fields(table)]

        if len(args) == 0:
            args = []
            for field in sqlp.table_fields(table):
                if field.name not in primary and not field.not_null and getattr(data, field.name) is not None:
                    args.append(field.name)
        elif len(ill := [it for it in args if it in primary]):
            raise RuntimeError(f'It is illegal to update primary key {ill}')

        #
        if len(args):
            with self.open_connection():
                where = [getattr(table, f) == getattr(data, f) for f in primary]
                update = [getattr(table, f) == getattr(data, f) for f in args]
                sqlp.update(table, *update).where(*where).submit()

    @overload
    def get_data(self,
                 db: Type[DataBaseType], *,
                 date: str,
                 animal: str,
                 rec: str,
                 user: str,
                 optic: str) -> pl.DataFrame:
        pass

    def get_data(self, db: Type[DataBaseType], **kwargs: str) -> pl.DataFrame:
        """
        get db as polars dataframe

        :param db:
        :param kwargs:
        :return:
        """
        sources = self.find_physiological_data(**kwargs)
        results = []
        with self.open_connection():
            for src in sources:
                df = self.select_foreign_from_source(db, src).fetch_polars()
                if not df.is_empty():
                    results.append(df)

            try:
                ret = pl.concat(results)
            except ValueError:
                raise RuntimeError(f'DataBase: {db} is empty!')

        return ret

    def upload_to_gspread(self, db: Type[DataBaseType],
                          gspread_name: SpreadSheetName,
                          **kwargs) -> None:
        """upload db to a spreadsheet"""
        df = self.get_data(db, **kwargs)
        upload_dataframe_to_spreadsheet(
            df,
            gspread_name,
            db.__name__,
            service_account_path=IOConfig.DEFAULT_GSPREAD_AUTH,
            primary_key=('date', 'animal', 'rec', 'user', 'optic')
        )
        fprint(f'PUSH {db.__name__} to {gspread_name}')
