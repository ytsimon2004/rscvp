from pathlib import Path
from typing import Type

import polars as pl
from rscvp.util.database import RSCDatabase, PhysiologyDB, GenericDB, BayesDecodeDB, VisualSFTFDirDB, DataBaseType

from argclz import AbstractParser, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib import sqlp

__all__ = ['DatabaseViewOptions']


class DatabaseViewOptions(RSCDatabase, AbstractParser, Dispatch):
    action: str = argument(
        metavar='ACTION',
        help='dispatch options'
    )

    args: list[str] = argument(metavar='ARGS', nargs='*', action='extend')

    show_all: bool = argument('--all', help='show all the db')
    show_rich_table: bool = argument('--rich', help='show rich table instead of polars')

    def run(self):
        self.invoke_command(self.action, *self.args)

    @dispatch('test-init')
    def init_database_examples(self):
        with self.open_connection():
            sqlp.insert_into(PhysiologyDB).submit([
                PhysiologyDB('210315', 'YW006', '2P', 'YW', '0'),
                PhysiologyDB('210315', 'YW006', '2P', 'YW', 'all'),
                PhysiologyDB('210401', 'YW006', '2P', 'YW', '0'),
                PhysiologyDB('210401', 'YW006', '2P', 'YW', 'all'),
                PhysiologyDB('210402', 'YW008', '2P', 'YW', '0'),
                PhysiologyDB('210407', 'YW008', '2P', 'YW', '0'),
                PhysiologyDB('210101', 'SH00', 'no_cam', 'SH', '0'),
                PhysiologyDB('220202', 'SH00', 'no_cam', 'SH', '0'),
            ])

    @dispatch('list')
    def list_animal_names(self):
        print(super().list_animal_names())

    @dispatch('date')
    def list_date_animals(self, date: str):
        print(super().list_date_animals(date))

    @dispatch('animal')
    def list_animal_dates(self, animal: str):
        print(super().list_animal_dates(animal))

    @dispatch('find')
    def find_physiological_data(self, *args: str):
        kwargs = {}
        for arg in args:
            k, _, v = arg.partition('=')
            kwargs[k] = v
        results = super().find_physiological_data(**kwargs)
        sqlp.util.rich_sql_table(PhysiologyDB, results)

    @dispatch('import')
    def import_new_animal_from_directory(self, root: str):
        super().import_new_animal_from_directory(Path(root))

    @dispatch('diagram')
    def show_database_diagram(self, output_file: str = None):
        from neuralib.sqlp.dot import generate_dot
        ret = generate_dot(self, output_file)
        if ret is not None:
            print(ret)

    @dispatch('generic')
    def list_animal_data(self, *args: str):
        kwargs = {}
        for arg in args:
            k, _, v = arg.partition('=')
            kwargs[k] = v

        data = super().find_physiological_data(**kwargs)
        self.show_foreign_db(data, GenericDB)

    @dispatch('decode')
    def list_decode_data(self, *args: str):
        kwargs = {}
        for arg in args:
            k, _, v = arg.partition('=')
            kwargs[k] = v

        data = super().find_physiological_data(**kwargs)
        self.show_foreign_db(data, BayesDecodeDB)

    @dispatch('visual')
    def list_visual_data(self, *args: str):
        kwargs = {}
        for arg in args:
            k, _, v = arg.partition('=')
            kwargs[k] = v

        data = super().find_physiological_data(**kwargs)
        self.show_foreign_db(data, VisualSFTFDirDB)

    def show_foreign_db(self, sources: list[PhysiologyDB], db: Type[DataBaseType]) -> None:
        if self.show_rich_table:
            self._show_foreign_db_rich(sources, db)
        else:
            self._show_foreign_db_polars(sources, db)

    def _show_foreign_db_polars(self, sources, db: Type[DataBaseType]) -> None:
        results = []
        with self.open_connection():
            for src in sources:
                df = self.select_foreign_from_source(db, src).fetch_polars()
                if not df.is_empty():
                    results.append(df)
            #
            try:
                ret = pl.concat(results, how='vertical_relaxed')
            except ValueError as e:
                raise RuntimeError(f'primary key not found in source db: {repr(e)}')

            #
            if self.show_all:
                from neuralib.util.verbose import printdf
                printdf(ret)
            else:
                print(ret)

    def _show_foreign_db_rich(self, sources: list[PhysiologyDB], db: Type[DataBaseType]) -> None:
        results = []
        with self.open_connection():
            for src in sources:
                results.extend(self.select_foreign_from_source(db, src).fetchall())
            sqlp.util.rich_sql_table(db, results)


if __name__ == '__main__':
    DatabaseViewOptions().main()
