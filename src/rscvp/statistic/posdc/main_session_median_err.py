import pingouin as pg
import polars as pl
import seaborn as sns
from rscvp.statistic.core import print_var
from rscvp.util.cli import CommonOptions

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import print_load, printdf

__all__ = ['SessionMedianErr']


class SessionMedianErr(AbstractParser, CommonOptions):
    DESCRIPTION = 'Plot median decoding error in different behavioral sessions from batch CV dataset'

    foreach_plane: bool = argument(
        '--foreach-plane',
        help='foreach plane for individual points, otherwise, average across optic plane (to per recording)'
    )

    def run(self):
        df = self.foreach_results()
        if self.foreach_plane:
            df = (df.with_columns(pl.col('filename') + '_' + pl.col('plane_idx').alias('filename'))
                  .drop('plane_idx'))

        print(df)
        self._print_var(df)
        self._print_statistic(df)

        with plot_figure(None) as ax:
            sns.pointplot(df, x='session', y='decode_err', hue='filename',
                          order=['light', 'dark', 'light_end'],
                          ax=ax)

            ax.set(xlabel='session', ylabel='median error (cm)')

    def foreach_results(self) -> pl.DataFrame:
        """
        ::

            ┌─────────┬────────────┬───────────┬──────────────┐
            │ session ┆ decode_err ┆ plane_idx ┆ filename     │
            │ ---     ┆ ---        ┆ ---       ┆ ---          │
            │ str     ┆ f64        ┆ str       ┆ str          │
            ╞═════════╪════════════╪═══════════╪══════════════╡
            │ light   ┆ 4.299381   ┆ 1         ┆ 211210_YW022 │
            │ dark    ┆ 9.204736   ┆ 1         ┆ 211210_YW022 │
            │ dark    ┆ 9.854965   ┆ 2         ┆ 211210_YW022 │
            │ light   ┆ 4.327917   ┆ 2         ┆ 211210_YW022 │
            └─────────┴────────────┴───────────┴──────────────┘

        OR

        ::

            ┌─────────┬──────────────┬────────────┐
            │ session ┆ filename     ┆ decode_err │
            │ ---     ┆ ---          ┆ ---        │
            │ str     ┆ str          ┆ f64        │
            ╞═════════╪══════════════╪════════════╡
            │ dark    ┆ 211210_YW022 ┆ 9.529851   │
            │ light   ┆ 211210_YW022 ┆ 4.313649   │
            └─────────┴──────────────┴────────────┘


        """
        ret = pl.DataFrame()

        for _ in self.foreach_dataset():
            caches = (self.cache_directory / 'posdc').glob('*.parquet')
            for file in caches:
                filename = file.stem
                plane_idx = filename[filename.find('plane') + 5]
                df = (
                    pl.read_parquet(file)
                    .select('session', 'decode_err')
                    .group_by('session')
                    .agg(pl.col('decode_err').mean())
                    .with_columns(pl.lit(plane_idx).alias('plane_idx'))
                    .with_columns(pl.lit(f'{self.exp_date}_{self.animal_id}').alias('filename'))
                )

                print_load(file)

                ret = pl.concat([ret, df])

        if not self.foreach_plane:
            ret = ret.group_by('session', 'filename').agg(pl.col('decode_err').mean())

        return ret

    @staticmethod
    def _print_var(df: pl.DataFrame) -> None:
        df = df.pivot(on='session', index='filename', values='decode_err')
        for col in df.drop('filename').columns:
            print_var(df[col], prefix=col)

    @staticmethod
    def _print_statistic(df: pl.DataFrame) -> None:
        """paired/non-parametric test"""
        post_hocs = pg.pairwise_tests(data=df.to_pandas(), dv='decode_err', within='session', parametric=False,
                                      subject='filename')
        printdf(post_hocs)


if __name__ == '__main__':
    SessionMedianErr().main()
