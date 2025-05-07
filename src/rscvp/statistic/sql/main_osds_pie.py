from typing import Literal, Final

import polars as pl
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.database import DB_TYPE
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE
from rscvp.visual.util_plot import selective_pie

from argclz import argument, as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation

__all__ = ['OriDirPieStat']


@publish_annotation('main', project='rscvp', caption='only in result text?', as_doc=True)
class OriDirPieStat(StatPipeline):
    DESCRIPTION = 'Pool the fraction of orientation/direction selective cells according to gspread'

    header = as_argument(StatisticOptions.header).with_options(required=False)

    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))

    pool_type: Literal['avg', 'sum'] = argument(
        '--pool',
        default='sum',
        help='the way to pool the data'
    )

    sheet_name: Final[GSPREAD_SHEET_PAGE] = 'VisualSFTFDirDB'
    db_table: Final[DB_TYPE] = 'VisualSFTFDirDB'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False, concat_plane_only=True)
        self.plot()

    def plot(self):
        if self.pool_type == 'avg':
            self.average_pie()
        elif self.pool_type == 'sum':
            self.sum_pie()

    def sum_pie(self):
        """Get the expression formula in gspread and sum up foreach animal"""
        df = self.df

        # restore total cells based on fraction
        df = df.with_columns((pl.col('n_ds_neurons') / pl.col('ds_frac')).alias('total_neurons'))

        region = df['region']
        dfrac = df['n_ds_neurons']
        ofrac = df['n_os_neurons']
        total = df['total_neurons']

        #
        aidx = (region == 'aRSC').arg_true()
        pidx = (region == 'pRSC').arg_true()

        da = dfrac[aidx].sum() / total[aidx].sum()
        dp = dfrac[pidx].sum() / total[pidx].sum()
        oa = ofrac[aidx].sum() / total[aidx].sum()
        op = ofrac[pidx].sum() / total[pidx].sum()

        with plot_figure(None, 1, 4) as ax:
            selective_pie([da, 1 - da], ['ds_aRSC', None], ax=ax[0])
            selective_pie([dp, 1 - dp], ['ds_pRSC', None], ax=ax[1])
            selective_pie([oa, 1 - oa], ['os_aRSC', None], ax=ax[2])
            selective_pie([op, 1 - op], ['os_pRSC', None], ax=ax[3])

    def average_pie(self):
        """Do average for each animal and plot overall pie"""

        def collect(col: Literal['ds_frac', 'os_frac']) -> list[float]:
            dataset = []
            for area in ('aRSC', 'pRSC'):
                avg = self.df.filter(pl.col('region') == area)[col].mean()
                dataset.append(avg)

            return dataset

        da, dp = collect('ds_frac')
        oa, op = collect('os_frac')

        with plot_figure(None, 1, 4) as ax:
            selective_pie([da, 1 - da], ['ds_aRSC', None], ax=ax[0])
            selective_pie([dp, 1 - dp], ['ds_pRSC', None], ax=ax[1])
            selective_pie([oa, 1 - oa], ['os_aRSC', None], ax=ax[2])
            selective_pie([op, 1 - op], ['os_pRSC', None], ax=ax[3])


if __name__ == '__main__':
    OriDirPieStat().main()
