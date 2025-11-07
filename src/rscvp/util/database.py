from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Annotated, Type, Literal, overload

import polars as pl

import sqlclz
from neuralib.tools.gspread import upload_dataframe_to_spreadsheet, SpreadSheetName
from neuralib.util.verbose import fprint
from rscvp.util.io import IOConfig

__all__ = [
    'DB_TYPE',
    'ResultDB',
    #
    'PhysiologyDB',
    'FieldOfViewDB',
    'BaseClassDB',
    'DarkClassDB',
    'BlankClassDB',
    'VRClassDB',
    #
    'BayesDecodeDB',
    'VisualSFTFDirDB',
    #
    'RSCDatabase',
]

DB_TYPE = Literal[
    'BaseClassDB', 'FieldOfViewDB',
    'DarkClassDB', 'BlankClassDB', 'VRClassDB',
    'BayesDecodeDB', 'VisualSFTFDirDB'
]


@sqlclz.named_tuple_table_class
class PhysiologyDB(NamedTuple):
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    @property
    def dir_name(self) -> str:
        return f'{self.date}_{self.animal}__{self.rec}_{self.user}_{self.optic}'

    @classmethod
    def parse_dir_structure(cls, directory: Path) -> list[PhysiologyDB]:
        name = directory.name
        matches = re.match(r'(\d+)_(\w+)__(\w+)_(\w+)', name)
        if matches:
            date, animal, rec, user = matches.groups()

            if '_' in rec:  # 210315_YW006__2P_YW_s2p_16-03-25
                fprint(f'{name} not analysed yet', vtype='warning')
                return []

            # find planes
            optics = ['all']  # for concat data
            p = list(directory.glob('suite2p/plane*'))
            if len(p) != 0:
                for pp in p:
                    optics.extend(pp.name[5:])
                return [PhysiologyDB(date, animal, rec, user, o) for o in optics]
            else:
                fprint(f'no suite2p registered file found in {name}', vtype='error')
                return []
        else:
            fprint(f'{name} pattern invalid', vtype='error')
            return []


@sqlclz.named_tuple_table_class
class FieldOfViewDB(NamedTuple):
    """Two-photon field of view imaging parameters and registration coordinates"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]

    usage: Literal['base', 'dark', 'blank', 'vr']
    """data usage"""

    region: str
    """imaging region"""
    max_depth: str
    """depth of the most ventral optic plane in um | Literal['ETL']"""
    n_planes: int
    """number of optic planes"""
    objective_rotation: float
    """objective rotation in degree"""
    objective_magnification: int
    """objective magnification"""

    # registration - stored as JSON TEXT '[x, y]'
    medial_anterior: str
    """medial anterior coordinate"""
    medial_posterior: str
    """medial posterior coordinate"""
    lateral_posterior: str
    """lateral posterior coordinate"""
    lateral_anterior: str
    """lateral anterior coordinate"""

    @sqlclz.foreign(PhysiologyDB.date, PhysiologyDB.animal, PhysiologyDB.user)
    def _foreign(self):
        return self.date, self.animal, self.user


@sqlclz.named_tuple_table_class
class BaseClassDB(NamedTuple):
    """Neuron classification database from baseline condition (without manipulation)"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    pair_wise_group: int | None = None
    n_total_neurons: int | None = None
    n_selected_neurons: int | None = None
    n_visual_neurons: int | None = None
    n_spatial_neurons: int | None = None
    n_overlap_neurons: int | None = None

    update_time: str | None = None

    @sqlclz.foreign(PhysiologyDB)
    def _foreign(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlclz.named_tuple_table_class
class BayesDecodeDB(NamedTuple):
    """Bayes position decode database"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    pair_wise_group: int | None = None
    n_neurons: int | None = None
    spatial_bins: float | None = None
    temporal_bins: float | None = None
    median_decode_error: float | None = None
    cross_validation: str | None = None
    """{'odd', 'even', 'random_split', int}"""

    update_time: str | None = None

    @sqlclz.foreign(PhysiologyDB)
    def _foreign(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlclz.named_tuple_table_class
class VisualSFTFDirDB(NamedTuple):
    """Visual SF/TF/Direction database"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    pair_wise_group: int | None = None

    # Order followed by ``SFTF_ARRANGEMENT``
    sftf_amp_group1: float | None = None
    sftf_amp_group2: float | None = None
    sftf_amp_group3: float | None = None
    sftf_amp_group4: float | None = None
    sftf_amp_group5: float | None = None
    sftf_amp_group6: float | None = None
    sftf_frac_group1: float | None = None
    sftf_frac_group2: float | None = None
    sftf_frac_group3: float | None = None
    sftf_frac_group4: float | None = None
    sftf_frac_group5: float | None = None
    sftf_frac_group6: float | None = None

    n_ds_neurons: int | None = None
    """Number of direction selective neurons"""
    n_os_neurons: int | None = None
    """Number of orientation selective neurons"""
    ds_frac: float | None = None
    """Fraction of direction selective"""
    os_frac: float | None = None
    """Fraction of orientation selective"""

    update_time: str | None = None

    @sqlclz.foreign(PhysiologyDB)
    def _foreign(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlclz.named_tuple_table_class
class DarkClassDB(NamedTuple):
    """Neuron classification database from darkness condition"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    n_total_neurons: int | None = None
    n_selected_neurons: int | None = None
    n_spatial_neurons_light_bas: int | None = None
    n_spatial_neurons_dark: int | None = None
    n_spatial_neurons_light_end: int | None = None

    update_time: str | None = None

    @sqlclz.foreign(PhysiologyDB)
    def _foreign(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlclz.named_tuple_table_class
class BlankClassDB(NamedTuple):
    """Neuron classification database from blank treadmill (without tactile cue) condition"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    pair_wise_group: int | None = None
    n_total_neurons: int | None = None
    n_selected_neurons: int | None = None
    n_spatial_neurons: int | None = None

    update_time: str | None = None

    @sqlclz.foreign(PhysiologyDB)
    def _foreign(self):
        return self.date, self.animal, self.rec, self.user, self.optic


@sqlclz.named_tuple_table_class
class VRClassDB(NamedTuple):
    """Neuron classification database from VR environment"""
    date: Annotated[str, sqlclz.PRIMARY]
    animal: Annotated[str, sqlclz.PRIMARY]
    rec: Annotated[str, sqlclz.PRIMARY]
    user: Annotated[str, sqlclz.PRIMARY]
    optic: Annotated[str, sqlclz.PRIMARY]

    virtual_map: str | None = None
    protocol: str | None = None

    pair_wise_group: int | None = None
    n_total_neurons: int | None = None
    n_selected_neurons: int | None = None
    n_spatial_neurons: int | None = None
    """number of position selective neurons"""
    n_spatial_persist: int | None = None
    """number of position selective neurons persisted in open-loop from closed-loop"""
    n_spatial_remap: int | None = None
    """number of position selective neurons remap in open-loop from closed-loop"""
    update_time: str | None = None

    @sqlclz.foreign(PhysiologyDB)
    def _foreign(self):
        return self.date, self.animal, self.rec, self.user, self.optic


SourceDB = PhysiologyDB | FieldOfViewDB
"""Source database for read"""
ResultDB = BaseClassDB | BayesDecodeDB | VisualSFTFDirDB | DarkClassDB | BlankClassDB | VRClassDB
"""Analysis result database for write"""


class RSCDatabase(sqlclz.Database):

    @property
    def database_file(self) -> Path:
        return Path(__file__).parents[3] / 'res' / 'database' / 'rscvp.db'

    @property
    def database_tables(self) -> list[type]:
        return [
            PhysiologyDB,
            FieldOfViewDB,
            BaseClassDB,
            DarkClassDB,
            BlankClassDB,
            VRClassDB,
            BayesDecodeDB,
            VisualSFTFDirDB
        ]

    def list_animal_names(self) -> list[str]:
        with self.open_connection():
            results = sqlclz.select_from(PhysiologyDB.animal, distinct=True).fetchall()
        return sqlclz.util.take(0, results)

    def list_date_animals(self, date: str) -> list[str]:
        with self.open_connection():
            results = sqlclz.select_from(PhysiologyDB.animal, distinct=True).where(
                PhysiologyDB.date == date
            ).fetchall()
        return sqlclz.util.take(0, results)

    def list_animal_dates(self, animal: str) -> list[str]:
        with self.open_connection():
            results = sqlclz.select_from(PhysiologyDB.date, distinct=True).where(
                PhysiologyDB.animal == animal
            ).fetchall()
        return sqlclz.util.take(0, results)

    @overload
    def find_physiological_data(self, *,
                                date: str = None,
                                animal: str = None,
                                rec: str = None,
                                user: str = None,
                                optic: str = None) -> list[PhysiologyDB]:
        pass

    def find_physiological_data(self, **kwargs) -> list[PhysiologyDB]:
        with self.open_connection():
            return (
                sqlclz
                .select_from(PhysiologyDB)
                .where(*[getattr(PhysiologyDB, k) == v for k, v in kwargs.items()])
                .fetchall()
            )

    def list_dirs(self) -> list[PhysiologyDB]:
        with self.open_connection():
            return sqlclz.select_from(PhysiologyDB).fetchall()

    def import_new_animals(self, animals: list[PhysiologyDB]):
        with self.open_connection():
            sqlclz.insert_into(PhysiologyDB, policy='REPLACE').submit(animals)

    def import_new_animal_from_directory(self, root: Path):
        animals = []
        for file in root.iterdir():
            animals.extend(PhysiologyDB.parse_dir_structure(file))

        self.import_new_animals(animals)

    @staticmethod
    def select_foreign_from_source(foreign_db: type[ResultDB], source: PhysiologyDB) -> sqlclz.Cursor[ResultDB]:
        return sqlclz.util.pull_foreign(foreign_db, source)

    def replace_data(self, data: ResultDB):
        """Replace data (insert if new, replace if exists)"""
        with self.open_connection():
            sqlclz.replace_into(type(data)).submit([data])

    def update_data(self, data: ResultDB, *args: str):
        """
        Update the data in database for those with matched primary keys.

        If you want to change the primary keys for any data, please use
        raw sql statements, because this method does not allow this action.

        :param data: updating data.
        :param args: name of field that needs to be updated.
            If empty, update all non-primary, null-able keys with non-None value.
        """
        table = type(data)

        primary = [it.name for it in sqlclz.table_primary_fields(table)]

        if len(args) == 0:
            args = []
            for field in sqlclz.table_fields(table):
                if field.name not in primary and not field.not_null and getattr(data, field.name) is not None:
                    args.append(field.name)
        elif len(ill := [it for it in args if it in primary]):
            raise RuntimeError(f'It is illegal to update primary key {ill}')

        #
        if len(args):
            with self.open_connection():
                where = [getattr(table, f) == getattr(data, f) for f in primary]
                update = [getattr(table, f) == getattr(data, f) for f in args]
                sqlclz.update(table, *update).where(*where).submit()

    @overload
    def get_data(self, db: Type[ResultDB], *,
                 date: str,
                 animal: str,
                 rec: str,
                 user: str,
                 optic: str) -> pl.DataFrame:
        pass

    def get_data(self, db: Type[ResultDB], **kwargs: str) -> pl.DataFrame:
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

    def submit_gspread(self, db: Type[ResultDB],
                       gspread_name: SpreadSheetName, **kwargs):
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
        print(df)
