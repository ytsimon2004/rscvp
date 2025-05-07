from typing import final

import attrs
import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.imaging.registration import FieldOfView
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

__all__ = ['RSCObjectiveFOV']

COORDINATES_FIELD = ['loc_MA', 'loc_MP', 'loc_LA', 'loc_LP']


@final
@attrs.define
class RSCObjectiveFOV(FieldOfView):

    @classmethod
    def load_from_gspread(cls, exp_date: str | None,
                          animal_id: str | None,
                          page: GSPREAD_SHEET_PAGE = 'apcls_tac') -> Self | list[Self]:
        """Load fov coordinates using gspread record"""
        from rscvp.statistic.cli_gspread import GSPExtractor

        df = GSPExtractor(page).load_from_gspread()

        if exp_date is None and animal_id is None:
            return _load_fov_all(df)
        elif exp_date is None and animal_id is not None:
            return _load_fov_animal(df, animal_id)
        else:
            return _load_fov_date_animal(df, exp_date, animal_id)


def _load_fov_date_animal(df: pl.DataFrame,
                          exp_date: str,
                          animal_id: str) -> RSCObjectiveFOV:
    """load a specific recording FOV"""
    data = f'{exp_date}_{animal_id}'
    df = df.filter(pl.col('Data') == data)
    region = df.select('region').item()
    rot = df.select('rotation').item()

    loc = (df.select(COORDINATES_FIELD)
           .to_numpy()
           .flatten()
           .astype(str))

    loc = [np.array([x.split(';')[0], x.split(';')[1]], dtype=float)
           for x in loc]

    corners = np.vstack([*loc])

    return RSCObjectiveFOV(corners, rotation_ml=rot, region_name=region)


def _load_fov_animal(df: pl.DataFrame, animal_id: str) -> list[RSCObjectiveFOV]:
    """load one animal across different dates FOV"""
    df = df.filter(pl.col('Data').str.contains(animal_id))
    region = df['region']
    rot = df['rotation']
    points = df.select(COORDINATES_FIELD).to_numpy()

    points = np.array([list(map(lambda x: list(map(float, x.split(';'))), row)) for row in points])

    return [RSCObjectiveFOV(np.vstack([*p]), rotation_ml=rot[i], region_name=region[i])
            for i, p in enumerate(points)]


def _load_fov_all(df: pl.DataFrame) -> list[RSCObjectiveFOV]:
    """load all recording FOV"""
    region = df['region'].to_list()
    rot = df['rotation'].to_list()
    points = (
        df.select(COORDINATES_FIELD)
        .drop_nulls()
        .to_numpy()
    )

    points = np.array([list(map(lambda x: list(map(float, x.split(';'))), row)) for row in points])

    return [RSCObjectiveFOV(np.vstack([*p]), rotation_ml=rot[i], region_name=region[i])
            for i, p in enumerate(points)]
