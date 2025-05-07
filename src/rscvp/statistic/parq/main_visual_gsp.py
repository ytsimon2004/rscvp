from typing import Final

import numpy as np
import seaborn as sns

from argclz import as_argument, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic._var import VIS_HEADERS
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_stat import CollectDataSet

__all__ = ['VisStatGSP']


@publish_annotation('main', project='rscvp', figure='fig.4B', as_doc=True)
class VisStatGSP(StatPipeline):
    DESCRIPTION = 'Compare the individual neurons visual responses in aRSC versus pRSC'

    nbins: int = argument(
        '--nbins',
        default=40,
        help='number of bins for histogram',
    )

    dff_cutoff: float | None = argument(
        '--cutoff',
        help='cutoff for removing transient dff',
    )

    hist_norm: bool = argument(
        '--hist-norm',
        help='number of neuron normalization in histogram',
    )

    header = as_argument(StatisticOptions.header).with_options(choices=VIS_HEADERS)

    sheet_name: Final = 'ap_vz'

    dependent = False
    ttest_parametric_infer = True

    def run(self):
        self.load_table()
        self.run_pipeline()

    def get_collect_data(self) -> CollectDataSet:
        if self._collect_data is None:
            self._collect_data = self._get_collect_data(self.group_header, self.header)

            if self.header in ('max_vis_resp', 'perc95_vis_resp') and self.dff_cutoff is not None:
                mask = {
                    g: np.array([val < self.dff_cutoff])
                    for g, val in self._collect_data.data.items()
                }

                self._collect_data = self._collect_data.with_selection(mask)

        return self._collect_data

    def plot(self):
        """plot visual reliability / max visual calcium resp. histogram"""
        dataset = self.get_collect_data()
        with plot_figure(self.output_figure) as ax:
            ylabel = 'Fraction' if self.hist_norm else 'Neurons #'
            self._plot_histogram(ax,
                                 dataset[0],
                                 label=dataset.group_vars[0],
                                 color='g',
                                 ylabel=ylabel,
                                 xlabel=self.header)
            self._plot_histogram(ax,
                                 dataset[1],
                                 label=dataset.group_vars[1],
                                 color='violet',
                                 ylabel=ylabel,
                                 xlabel=self.header)

    def _plot_histogram(self,
                        ax,
                        data: np.ndarray,
                        label: str,
                        color='g',
                        **kwargs):
        if self.dff_cutoff is not None:
            data = data[data < self.dff_cutoff]

        method = 'percent' if self.hist_norm else 'count'
        sns.histplot(ax=ax,
                     data=data,
                     bins=self.nbins,
                     label=label,
                     stat=method,
                     kde=True,
                     alpha=0.7,
                     color=color,
                     element='step',
                     fill=False)

        self.insert_pval(ax)
        ax.set(**kwargs)
        ax.legend()


if __name__ == '__main__':
    VisStatGSP().main()
