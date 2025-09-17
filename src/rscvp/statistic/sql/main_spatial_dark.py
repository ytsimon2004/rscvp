from typing import Final, Literal

import pingouin as pg
import polars as pl

from argclz import as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation, printdf
from rscvp.statistic.core import StatPipeline, print_var
from rscvp.statistic.sql.util import as_validate_sql_table
from rscvp.util.cli import CommonOptions

__all__ = ['SpatialFractionDarkStat']


@publish_annotation('sup', project='rscvp', figure='fig.S2C', as_doc=True)
class SpatialFractionDarkStat(StatPipeline):
    DESCRIPTION = 'Compare fraction of spatial cells across behavioral sessions from multiple animals'

    header = as_argument(StatPipeline.header).with_options(required=False)
    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))
    rec_region: Literal['aRSC', 'pRSC', 'all'] = as_argument(CommonOptions.rec_region).with_options(default='all')

    sheet_name: Final = 'DarknessGenericDB'
    db_table: Final = 'DarknessGenericDB'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False, concat_plane_only=True)
        self.df = as_validate_sql_table(self.df, 'ap_ldl')
        self.plot()

    def plot(self):
        df = self.df

        select_col = [
            'date',
            'animal',
            'n_selected_neurons',
            'n_spatial_neurons_light_bas',
            'n_spatial_neurons_dark',
            'n_spatial_neurons_light_end'
        ]

        if self.rec_region != 'all':
            df = df.filter(pl.col('region') == self.rec_region)

        df = (
            df.select(select_col)
            .with_columns((pl.col('date') + '_' + pl.col('animal')).alias('name'))
            .with_columns((pl.col('n_spatial_neurons_light_bas') / pl.col('n_selected_neurons')).alias('light_bas'))
            .with_columns((pl.col('n_spatial_neurons_dark') / pl.col('n_selected_neurons')).alias('dark'))
            .with_columns((pl.col('n_spatial_neurons_light_end') / pl.col('n_selected_neurons')).alias('light_end'))
            .drop(select_col)
        )

        printdf(df)
        self._print_var(df)
        self._print_statistic(df)

        with plot_figure(None) as ax:
            for val in df.iter_rows():
                ax.plot(val[1:], label=val[0])

            ax.legend()

    @staticmethod
    def _print_var(df: pl.DataFrame) -> None:
        for col in df.drop('name').columns:
            print_var(df[col], prefix=col)

    @staticmethod
    def _print_statistic(df: pl.DataFrame) -> None:
        df = df.unpivot(index='name', variable_name='session', value_name='fraction')
        post_hocs = pg.pairwise_tests(data=df.to_pandas(), dv='fraction', within='session', parametric=False,
                                      subject='name')

        printdf(post_hocs)


if __name__ == '__main__':
    SpatialFractionDarkStat().main()
