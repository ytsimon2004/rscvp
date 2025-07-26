import polars as pl
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.stats._morestats import WilcoxonResult

from argclz import as_argument
from neuralib.plot import plot_figure
from rscvp.statistic.core import StatPipeline
from rscvp.statistic.sql.util import as_validate_sql_table
from rscvp.util.util_gspread import get_statistic_key_info

__all__ = ['SpatialBlankStat']


class SpatialBlankStat(StatPipeline):
    DESCRIPTION = 'Fraction of spatial cells in anterior v.s. posterior RSC in blank belt treadmill'

    header = as_argument(StatPipeline.header).with_options(required=False)
    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))

    sheet_name = 'BlankBeltGenericDB'
    db_table = 'BlankBeltGenericDB'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False, concat_plane_only=True)
        self.plot()

    def plot(self):
        df = self.preprocess()

        with plot_figure(None) as ax:
            sns.barplot(data=df, x="region", y="fraction", errorbar="se", capsize=0.2, ax=ax)
            sns.stripplot(data=df, x="region", y="fraction", color="black", size=6, jitter=False, ax=ax)

            self._plot_connected_line(df, ax)

    @staticmethod
    def _plot_connected_line(df, ax):
        df = (
            df.cast({'pair_wise_group': pl.Int64})
            .filter(pl.col('pair_wise_group') != -1)
            .sort('pair_wise_group', 'region')
        )

        value_a = df.filter(pl.col('region') == 'aRSC')['fraction']
        value_b = df.filter(pl.col('region') == 'pRSC')['fraction']

        res: WilcoxonResult = wilcoxon(value_a, value_b)

        for pair in list(zip(value_a, value_b)):
            ax.plot(pair, color='k')

        ax.set_title(f'p-val: {res.pvalue}')

    def preprocess(self) -> pl.DataFrame:
        df = as_validate_sql_table(self.df, 'apcls_blank')
        select_col = ['date', 'animal', 'n_selected_neurons', 'n_spatial_neurons']
        df = (
            df.select(select_col + ['region'])
            .with_columns((pl.col('date') + '_' + pl.col('animal')).alias('Data'))
            .with_columns((pl.col('n_spatial_neurons') / pl.col('n_selected_neurons')).alias('fraction'))
            .drop(select_col)
        )

        pair_df = get_statistic_key_info('apcls_blank')

        df = df.join(pair_df, on='Data')

        return df


if __name__ == '__main__':
    SpatialBlankStat().main()
