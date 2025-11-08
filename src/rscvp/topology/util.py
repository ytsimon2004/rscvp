from pathlib import Path
from typing import final

import attrs
import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.imaging.registration import FieldOfView

__all__ = ['RSCObjectiveFOV']

COORDINATES_FIELD = ['medial_anterior', 'medial_posterior', 'lateral_anterior', 'lateral_posterior']


@final
@attrs.define
class RSCObjectiveFOV(FieldOfView):

    @classmethod
    def load_from_gspread(cls, exp_date: str | None = None,
                          animal_id: str | None = None,
                          usage: str = 'base') -> Self | list[Self]:
        """Load fov coordinates from google spreadsheet"""
        from rscvp.statistic.cli_gspread import GSPExtractor

        df = GSPExtractor('fov_table').load_from_gspread().filter(pl.col('usage') == usage)

        return _to_fov(df, exp_date, animal_id, use_db=False)

    @classmethod
    def load_from_database(cls, exp_date: str | None = None,
                           animal_id: str | None = None,
                           usage: str = 'base'):
        """Load fov coordinates from local project sqlite database"""
        import sqlite3
        db_file = Path(__file__).parents[3] / 'res' / 'database' / 'rscvp.db'
        conn = sqlite3.connect(db_file)

        df = pl.read_database(
            query='SELECT * FROM "FieldOfViewDB"',
            connection=conn
        ).filter(pl.col('usage') == usage)

        return _to_fov(df, exp_date, animal_id, use_db=True)


def _to_fov(df: pl.DataFrame,
            exp_date: str | None = None,
            animal_id: str | None = None,
            use_db: bool = False) -> list[RSCObjectiveFOV]:
    """load all recording FOV"""
    region = df['region'].to_list()
    rot = df['objective_rotation'].to_list()

    #
    if exp_date is not None:
        if use_db:
            df = df.filter(pl.col('date') == exp_date)
        else:
            df = df.filter(pl.col('Data').str.contains(exp_date))

    #
    if animal_id is not None:
        if use_db:
            df = df.filter(pl.col('animal') == animal_id)
        else:
            df = df.filter(pl.col('Data').str.contains(animal_id))

    #
    df = df.with_columns([
        pl.col(c)
        .str.strip_chars("[]")
        .str.split(by=",")
        .list.eval(pl.element().str.strip_chars().cast(pl.Float64))
        .alias(c)
        for c in COORDINATES_FIELD
    ])

    points = (
        df.select(COORDINATES_FIELD)
        .drop_nulls()
        .to_numpy()
    )

    return [RSCObjectiveFOV(np.vstack([*p]), rotation_ml=rot[i], region_name=region[i])
            for i, p in enumerate(points)]
