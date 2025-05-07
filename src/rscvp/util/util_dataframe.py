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
