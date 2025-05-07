from typing import Literal, Any

import seaborn as sns
from matplotlib.axes import Axes

from argclz import argument, as_argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure, diag_histplot
from neuralib.typing import PathLike
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_stat import DataSetType, CollectDataSet

__all__ = ['ValStatGSP']


@publish_annotation('main', project='rscvp', figure='fig.2D-2F', as_doc=True)
class ValStatGSP(StatPipeline, Dispatch):
    DESCRIPTION = 'see the values from individual neurons across animal'

    plot_type: Literal[
        'cumulative',
        'box_mean',
        'histogram',
        'violin',
        'diag'
    ] = argument(
        '--plot', '--plot-type',
        default='box_mean',
        help='plot type'
    )

    sheet_name = as_argument(StatisticOptions.sheet_name).with_options(required=True)

    #
    load_source = 'parquet'

    #
    dependent = False
    ttest_parametric_infer = True

    def run(self):
        self.load_table(to_pandas=False if self.animal_based_comp else True)

        if self.animal_based_comp:
            self.plot_pairwise_mean()
        else:
            self.run_pipeline()

    def plot(self):
        dat = self.get_collect_data()
        output_file = self.get_output_figure_type(self.plot_type)

        if self.plot_type == 'diag':
            self.plot_diag_histplot(dat, output_file)
        else:
            with plot_figure(output_file) as ax:
                self.invoke_command(self.plot_type, ax, dat.data)
                self.insert_pval(ax)
                ax.set(**self.default_fig_settings())
                ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    def default_fig_settings(self) -> dict[str, Any]:
        return {
            'xlabel': self.group_header,
            'ylabel': self.header
        }

    @dispatch('box_mean')
    def plot_box_stat(self, ax: Axes, dat: DataSetType):
        sns.boxplot(ax=ax, data=dat, showfliers=False, width=0.5, palette=self.color_palette)
        sns.pointplot(ax=ax, data=dat, estimator='mean', color='red')

    @dispatch('violin')
    def plot_violin_stat(self, ax: Axes,
                         dat: DataSetType, *,
                         show_points=False):
        sns.violinplot(ax=ax, data=dat, alpha=0.6, bw='silverman', palette=self.color_palette)
        if show_points:
            sns.stripplot(ax=ax, data=dat, jitter=True, size=2, alpha=0.5)

    @dispatch('cumulative')
    def plot_cumulative_stat(self, ax: Axes, dat: DataSetType):
        for k, v in dat.items():
            sns.ecdfplot(ax=ax, data=v, color=self.color_palette[k], label=k)

        ax.legend()

    def plot_diag_histplot(self, dat: CollectDataSet, output: PathLike):
        from itertools import combinations

        if dat.group_header != 'session':
            raise NotImplementedError('')

        pairs = [list(pair) for pair in combinations(['light_bas', 'dark', 'light_end'], 2)]
        n_pairs = len(pairs)
        with plot_figure(output, 1, n_pairs) as ax:
            for i, pair in enumerate(pairs):
                xc = pair[0]
                yc = pair[1]
                x = dat.data[xc]
                y = dat.data[yc]
                diag_histplot(
                    x, y,
                    bins=35,
                    hist_width=0.4,
                    scatter_kws={'s': 3, 'c': 'gray', 'marker': '.', 'edgecolors': 'none'},
                    polygon_kws={'facecolor': 'silver', 'edgecolor': 'none', 'zorder': -1},
                    ax=ax[i]
                )
                ax[i].set(xlabel=xc, ylabel=yc)


if __name__ == '__main__':
    ValStatGSP().main()
