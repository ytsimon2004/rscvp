from typing import Any, NamedTuple

import polars as pl
from polars import SchemaFieldNotFoundError
from rscvp.util.cli import CommonOptions
from rscvp.util.cli.cli_io import CELLULAR_IO, HEADER, CodeAlias
from rscvp.util.cli.cli_statistic import StatisticOptions

from neuralib.util.utils import joinn
from neuralib.util.verbose import fprint
from stimpyp import Session

__all__ = ['CSVCollector']


class DataName(NamedTuple):
    exp_date: str
    animal_id: str
    plane_idx: int | None = None
    neuron_id: int | None = None
    session: Session | None = None

    @property
    def name(self) -> str:
        plane = None if self.plane_idx is None else f'plane{self.plane_idx}'
        return joinn('_', self.exp_date, self.animal_id, plane, self.neuron_id, self.session)


class CSVCollector:
    """Concat the certain csv file (based on code) and do the cell mask between different datasets"""

    def __init__(self,
                 code: CodeAlias, *,
                 stat_col: list[str | HEADER],
                 exclude_col: list[str] | None = None,
                 fields: dict[str, Any] = None,
                 truncate_session_agg: bool = True):
        """
        :param code: cli INFO` code
        :param stat_col: columns that used for concat across dataset for statistical purpose.
                         used for remove session info in the original csv
                         ** Note that stat_col should be in a specific order that corresponding to original csv
        :param exclude_col: columns that need to be excluded (useless for statistic)
        :param fields: foreach options, beside animal and exp date
        :param truncate_session_agg: truncate the session for aggregate the csv
        """
        if code not in CELLULAR_IO:
            raise ValueError(f'unknown{code}')

        self.code = code
        self.fields = fields or {}

        self.stat_col = ['neuron_id'] + stat_col
        self.exclude_col = exclude_col
        self.truncate_session_agg = truncate_session_agg

    def find_all(self, opt: StatisticOptions) -> pl.DataFrame:
        return self._find_all(opt)

    def _find_all(self, opt: StatisticOptions) -> pl.DataFrame:

        df_ls = []
        for i, _ in enumerate(opt.foreach_dataset(**self.fields)):
            cell_mask = opt.get_selected_neurons()

            # workaround for NoneType wrong parse
            s = opt.session if opt.session != 'None' else None
            opt.session = s

            if opt.plane_index is not None:
                p = opt.get_data_output(self.code, s, latest=True).csv_output
                df = pl.read_csv(p).filter(cell_mask)

                # exclude unnecessary info
                if self.exclude_col is not None:
                    if s is not None:
                        columns = [f'{col}_{s}' for col in self.exclude_col]
                    else:
                        columns = self.exclude_col

                    df = self._try_drop_cols(opt, df, columns)

                # rename column and add info
                if len(df.columns) != len(self.stat_col):
                    fprint(f'Mismatch columns {list(df.columns)} and {self.stat_col}', vtype='error')
                    raise RuntimeError(f'{opt.exp_date}_{opt.animal_id}: cannot be set, check csv..')

                if self.truncate_session_agg:
                    df.columns = self._truncate_session_in_headers(opt, df)

                df = (
                    df.with_columns(pl.lit(DataName(opt.exp_date, opt.animal_id).name).alias('Data'))
                    .with_columns(pl.lit(f'{opt.plane_index}').alias('plane'))
                )

            else:
                ns = ['neuron_id']
                if s is not None:  # i.e., visual session parameter
                    ns.extend(f'{n}_{s}' for n in self.stat_col if n != 'neuron_id')
                else:
                    ns.extend(n for n in self.stat_col if n != 'neuron_id')

                pidx = opt.get_csv_data('plane_idx', to_numpy=True)[cell_mask]
                df = opt.get_csv_data(cols=ns, to_numpy=False).filter(cell_mask)

                if self.truncate_session_agg:
                    df.columns = self._truncate_session_in_headers(opt, df)

                df = (
                    df.with_columns(pl.lit(DataName(opt.exp_date, opt.animal_id).name).alias('Data'))
                    .with_columns(pl.Series(pidx, dtype=pl.Utf8).alias('plane'))
                )

            if opt.session is not None:
                df = df.with_columns(pl.lit(s).alias('session'))

            if opt.rec_region is not None:
                df = df.with_columns(pl.lit(opt.rec_region).alias('region'))

            df_ls.append(df)

        return pl.concat(df_ls)

    def _truncate_session_in_headers(self, opt: StatisticOptions, df: pl.DataFrame) -> list[str]:
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
            fprint(f'Drop! {drop_col} in {opt.filename}')

        return df
