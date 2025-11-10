import polars as pl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from argclz import as_argument
from neuralib.plot import plot_figure
from neuralib.typing import AxesArray, ArrayLikeStr, ArrayLike
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline

__all__ = ['VRClassStat']


@publish_annotation('sup', project='rscvp', figure='fig.S4H', as_doc=True)
class VRClassStat(StatPipeline):
    DESCRIPTION = ('Fraction of position cells, and proportion of persistent place field / remap '
                   'in anterior v.s. posterior RSC in closed-loop VR')

    header = as_argument(StatPipeline.header).with_options(required=False)
    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))

    sheet_name = 'VRClassDB'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False, concat_plane=True)
        self.plot()

    def plot(self):
        with plot_figure(None, 1, 3) as _ax:
            ax = _ax[0]
            self.plot_connect(ax)

            ax = _ax[1:]
            self.plot_pie(ax)

    def plot_connect(self, ax):
        df = (
            self.df
            .with_columns((pl.col('n_spatial_neurons') / pl.col('n_selected_neurons')).alias('fraction'))
            .select('region', 'pair_wise_group', 'fraction')
            .sort('region', 'pair_wise_group')
        )

        print(df)

        value_a = df.filter(pl.col('region') == 'aRSC')['fraction'].to_numpy()
        value_b = df.filter(pl.col('region') == 'pRSC')['fraction'].to_numpy()

        self.plot_connect_datapoints(ax, value_a, value_b, with_bar=True)

    def plot_pie(self, axes: AxesArray):
        for i, df in enumerate(self.df.partition_by('region')):
            total = df['n_spatial_neurons'].sum()
            n_persist = df['n_spatial_persist'].sum() / total
            n_remap = df['n_spatial_remap'].sum() / total
            n_abolish = 1 - n_persist - n_remap

            self._pie_chart(n=[n_persist, n_remap, n_abolish], labels=['persist', 'remap', 'abolish'], ax=axes[i])
            axes[i].set_title(df['region'].unique().item())

    @staticmethod
    def _pie_chart(n: ArrayLike,
                   labels: ArrayLikeStr,
                   ax: Axes | None = None):
        """
        Plot spatial neuron type proportion with enhanced visualization

        :param n: Numbers. `Arraylike[float, N]`
        :param labels: Labels. `ArrayLikeStr[str, N]`
        :param ax: ``Axes``
        """
        if len(n) != len(labels):
            raise ValueError('')

        if ax is None:
            _, ax = plt.subplots()

        explode = [0.08, 0.08, 0.05]

        ax.pie(
            n,
            explode=explode,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            radius=0.8,
            pctdistance=0.7,
            labeldistance=1.1
        )

        ax.axis('equal')


if __name__ == '__main__':
    VRClassStat().main()
