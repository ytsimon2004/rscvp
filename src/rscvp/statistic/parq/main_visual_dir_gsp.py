import pickle
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from rich.pretty import pprint

from argclz import as_argument, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic._var import VIS_DIR_HEADERS
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_stat import CollectDataSet
from rscvp.visual.main_polar import BaseVisPolarOptions
from rscvp.visual.util_plot import dir_hist

__all__ = ['VisualDirParQ']


@publish_annotation('main', project='rscvp', figure='fig.5F-G,I-J', as_doc=True)
class VisualDirParQ(StatPipeline, BaseVisPolarOptions):
    DESCRIPTION = 'direction / orientation selectivity in individual cells across animals'

    header: str = as_argument(StatisticOptions.header).with_options(choices=VIS_DIR_HEADERS)

    hist_unit: Literal['counts', 'fraction'] = argument(
        '--unit',
        default='fraction',
        help='histogram unit. counts(cell counts) or fraction(fraction of neurons)'
    )

    sheet_name = 'visual_parq'

    dependent = False
    parametric = False

    def run(self):
        self.load_table(to_pandas=False if self.animal_based_comp else True)

        if self.animal_based_comp:
            self.plot_pairwise_mean()
        else:
            self.run_pipeline()

    def get_collect_data(self) -> CollectDataSet:
        if self._collect_data is None:
            self._collect_data = self._get_collect_data(self.group_header, self.header)

            if self.header in ('pori', 'pdir'):
                mask = self._get_selectivity_mask()
                self._collect_data = self._collect_data.with_selection(mask)

            pprint(self._collect_data)

        return self._collect_data

    def _get_selectivity_mask(self) -> dict[str, np.ndarray]:
        """return dict of bool array"""
        if self.header == 'pori':
            file = self.callback_pickle_file('osi')
        elif self.header == 'pdir':
            file = self.callback_pickle_file('dsi')
        else:
            raise ValueError('')

        #
        with open(file, 'rb') as f:
            dat = pickle.load(f)['data']

        return {
            g: np.array([val >= self.selective_thres])
            for g, val in dat.items()
        }

    def plot(self):
        dataset = self.get_collect_data()
        with plot_figure(self.output_figure) as ax:
            self._plot_hist_opt(ax, dataset[0], color='g', label=dataset.group_vars[0])
            self._plot_hist_opt(ax, dataset[1], color='magenta', label=dataset.group_vars[1])

    def _plot_hist_opt(self, ax: Axes,
                       data: np.ndarray, *,
                       label: str | None = None,
                       color: str = 'k'):

        if self.hist_unit == 'fraction':
            weights = [1 / len(data)] * len(data)
            ylabel = 'fraction'
        else:
            weights = None
            ylabel = '#counts'

        #
        kwargs = dict(label=label, color=color, weights=weights, xlabel=f'{self.header}', ylabel=ylabel)

        if self.header in ('dsi', 'osi'):
            dir_hist(data, thres=self.selective_thres, bins=10, ax=ax, **kwargs)
        elif self.header == 'pdir':
            dir_hist(data, bins=12, xlim=(0, 360), ax=ax, **kwargs)
            ax.set_xticks([i * 90 for i in range(5)])
        elif self.header == 'pori':
            dir_hist(data, bins=6, xlim=(0, 180), ax=ax, **kwargs)
            ax.set_xticks([i * 45 for i in range(5)])
        else:
            raise ValueError('')


if __name__ == '__main__':
    VisualDirParQ().main()
