from typing import ClassVar, Literal

import polars as pl

__all__ = ['to_numeric',
           'check_null']


def to_numeric(s: pl.Series) -> pl.Series:
    try:
        ret = s.cast(pl.Float64)
    except pl.exceptions.InvalidOperationError:
        try:
            ret = s.cast(pl.Int64)
        except pl.exceptions.InvalidOperationError:
            ret = s

    return ret


def check_null(df: pl.DataFrame) -> None:
    """check if any null(empty) elements in the dataframe"""
    count = df.null_count()
    for series in count:
        if series.item() != 0:
            raise RuntimeError(f'{series.name} contain null value')


@pl.api.register_dataframe_namespace('alter')
class AlterFrame:
    _mouseline_thy1: ClassVar[tuple[str, ...]] = ('YW006', 'YW008', 'YW010', 'YW017')
    _mouseline_camk2: ClassVar[tuple[str, ...]] = ('YW022', 'YW032', 'YW033', 'YW048')

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def split_primary(self, drop: bool = True) -> pl.DataFrame:
        ret = (
            self._df
            .with_columns(pl.col("Data").str.split("_").list.get(0).alias("date"))
            .with_columns(pl.col("Data").str.split("_").list.get(1).alias("animal"))
        )
        return ret.drop('Data') if drop else ret

    def with_mouseline(self, col: Literal['animal', 'Data'] = 'animal') -> pl.DataFrame:
        return self._df.with_columns(
            pl.when(pl.col(col).str.contains('|'.join(self._mouseline_thy1)))
            .then(pl.lit('thy1'))
            .when(pl.col(col).str.contains('|'.join(self._mouseline_camk2)))
            .then(pl.lit('camk2'))
            .otherwise(pl.lit('other'))
            .alias('mouseline')
        )
