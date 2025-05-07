from typing import Literal

import seaborn as sns
from matplotlib import pyplot as plt
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli import HistOptions, PlotOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

from argclz import AbstractParser, as_argument, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.util import iter_source_coordinates
from neuralib.plot import plot_figure

__all__ = ['RoisViewOptions']


class RoisViewOptions(AbstractParser, HistOptions, Dispatch, PlotOptions):
    DESCRIPTION = '3d roi for particular region(s)'

    hemisphere = as_argument(HistOptions.hemisphere).with_options(default='ipsi')

    dispatch_plot: Literal['3d', 'histogram'] = argument(
        '--analysis',
        required=True,
        help='type of analysis'
    )

    ccf_dir: AbstractCCFDir

    def run(self):
        self.set_background()

        self.ccf_dir = self.get_ccf_dir()
        self.invoke_command(self.dispatch_plot)

    @property
    def title(self) -> str:
        return f'{self.area}_{self.hemisphere}_{self.dispatch_plot}'

    @dispatch('3d')
    def plot_roi_3d(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        iter_source = iter_source_coordinates(self.ccf_dir.parse_csv,
                                              only_areas=self.area,
                                              hemisphere=self.hemisphere,
                                              to_brainrender=False)
        for sc in iter_source:
            ax.scatter(sc.ap,
                       sc.ml,
                       sc.dv,
                       color=REGION_COLORS_HIST[sc.source],
                       alpha=0.8,
                       edgecolor='none',
                       marker='.')

        # same brain render orientation
        ax.invert_zaxis()
        ax.invert_yaxis()
        ax.set(xlabel='AP(mm)', ylabel='ML(mm)', zlabel='DV(mm)')

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))

        ax.set_title(self.title)

        if self.debug_mode:
            plt.show()
        else:
            plt.savefig(self.ccf_dir.output_folder / f'{self.title}.pdf')

    @dispatch('histogram')
    def plot_topographical_across_axis(self, bins=20):
        if self.debug_mode:
            output_file = None
        else:
            output_file = self.ccf_dir.output_folder / f'{self.title}.pdf'

        iter_source = iter_source_coordinates(
            self.ccf_dir.parse_csv,
            only_areas=self.area,
            hemisphere=self.hemisphere,
            to_brainrender=False
        )

        labels = ('ap', 'dv', 'ml')

        with plot_figure(output_file, 1, 3, set_square=True) as ax:
            for i, sc in enumerate(iter_source):
                for j in range(len(labels)):
                    sns.histplot(data=sc.coordinates[:, j],
                                 bins=bins,
                                 kde=True,
                                 ax=ax[j],
                                 element="step",
                                 stat='percent',
                                 color=REGION_COLORS_HIST[sc.source])

                ax[i].set_aspect(1.0 / ax[i].get_data_ratio(), adjustable='box')
                ax[i].set(xlabel=labels[i])


if __name__ == '__main__':
    RoisViewOptions().main()
