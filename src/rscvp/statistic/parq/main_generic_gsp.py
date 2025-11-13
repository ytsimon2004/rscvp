from typing import Literal, Final

from argclz import as_argument, argument
from neuralib.plot import plot_figure, violin_boxplot
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_plot import REGION_COLORS_PHY

__all__ = ['GenericParQ']


@publish_annotation('main', project='rscvp', figure='fig.3D & fig.4C', as_doc=True)
class GenericParQ(StatPipeline):
    DESCRIPTION = 'Plot the generic header. i.e., session independent, without cell masking'

    header: Literal['perc95_dff', 'max_dff'] = as_argument(StatisticOptions.header).with_options(...)

    value_cutoff: float | None = argument(
        '--cut',
        default=1000,
        help='lower bound value to disable the long tail (outlier)'
    )

    load_source: Final = 'parquet'
    sheet_name: Final = 'generic_parq'

    def run(self):
        self.load_table(to_pandas=False)

        if self.animal_based_comp:
            self.plot_pairwise_mean()
        else:
            self.run_pipeline()

    def plot(self):
        dataset = self.get_collect_data()
        dat = dataset.data

        # cutoff
        if self.value_cutoff is not None:
            update = {}
            for group, val in dat.items():
                update[group] = val[val < self.value_cutoff]
            dat = update

        with plot_figure(self.output_figure, figsize=(3, 6)) as ax:
            violin_boxplot(ax=ax, data=dat, scatter_alpha=0.4, scatter_size=2, palette=REGION_COLORS_PHY)
            self.insert_pval(ax)
            ax.set(ylabel=dataset.name)


if __name__ == '__main__':
    GenericParQ().main()
