from typing import Literal, Final

import polars as pl

from argclz import as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.database import DB_TYPE
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

__all__ = ['VisuoSpatialFractionStat']


@publish_annotation('main', project='rscvp', figure='fig.2A right & fig.4D right', as_doc=True)
class VisuoSpatialFractionStat(StatPipeline):
    DESCRIPTION = 'Compare and plot the fraction of visuospatial cell types in aRSC versus pRSC'

    header: Literal['visual_frac', 'spatial_frac'] = as_argument(StatisticOptions.header).with_options(...)

    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))

    #
    sheet_name: Final[GSPREAD_SHEET_PAGE] = 'GenericClassDB'
    db_table: Final[DB_TYPE] = 'GenericClassDB'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False)
        self.plot()

    def plot(self):
        df = (
            self.df
            .with_columns(
                (pl.col('n_spatial_neurons') / pl.col('n_selected_neurons')).alias('spatial_frac'),
                (pl.col('n_visual_neurons') / pl.col('n_selected_neurons')).alias('visual_frac')
            )
            .with_columns(
                pl.when(pl.col('animal').str.contains('|'.join(self._mouseline_thy1)))
                .then(pl.lit('thy1'))
                .when(pl.col('animal').str.contains('|'.join(self._mouseline_camk2)))
                .then(pl.lit('camk2'))
                .otherwise(pl.lit('other'))
                .alias('mouseline')
            )
            .sort('pair_wise_group', 'region')
        )

        # statistic
        value_a = df.filter(pl.col('region') == 'aRSC')[self.header].to_list()
        value_b = df.filter(pl.col('region') == 'pRSC')[self.header].to_list()
        with plot_figure(None, figsize=(3, 6)) as ax:
            self.plot_connect_datapoints(ax, value_a, value_b, df=df)


if __name__ == '__main__':
    VisuoSpatialFractionStat().main()
