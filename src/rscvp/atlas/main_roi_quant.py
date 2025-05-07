from typing import Literal, get_args

import numpy as np
import polars as pl
from matplotlib_venn import venn2
from rscvp.atlas.core import RSCRoiClassifierDataFrame
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.atlas.util_plot import (
    plot_foreach_channel_pie,
    plot_rois_bar,
    plot_categorical_region,
    prepare_venn_data
)
from rscvp.util.cli import HistOptions, ROIOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

from argclz import AbstractParser, as_argument, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.ccf.dataframe import ROIS_NORM_TYPE
from neuralib.plot import plot_figure
from neuralib.plot.colormap import DiscreteColorMapper, get_customized_cmap
from neuralib.plot.tools import insert_latex_equation

__all__ = ['RoiQuantOptions']


class RoiQuantOptions(AbstractParser, ROIOptions, Dispatch):
    DESCRIPTION = 'plot the summary plot of labelled roi through classification strategies'

    area = as_argument(HistOptions.area).with_options(
        help='only show area(s), only applicable in cat/bar/parallel plots'
    )

    dispatch_plot: Literal['cat', 'pie', 'bar', 'venn', 'bias', 'parallel'] = argument(
        '-g', '--graph',
        required=True,
        help='graph type',
    )

    ccf_dir: AbstractCCFDir
    roi: RSCRoiClassifierDataFrame

    def run(self):
        self.ccf_dir = self.get_ccf_dir()
        self.roi = self.load_roi_dataframe(self.ccf_dir)
        self.invoke_command(self.dispatch_plot, self.roi)

    @property
    def filename(self) -> str:
        return f'roi_{self.dispatch_plot}_L{self.merge_level}_T{self.top_area}_H{self.hemisphere}'

    @dispatch('cat')
    def plot_roi_categorical(self, roi: RSCRoiClassifierDataFrame):
        """
        Catplot per region
        y-axis could be either `channel-wise normalized percent` OR `counts`
        top region are picked up for all channels, thus should set higher for overview
        """
        norm = roi.to_normalized(self.roi_norm, self.merge_level,
                                 top_area=self.top_area,
                                 hemisphere=self.hemisphere)

        filename = self.filename + f'_{norm.normalized}'
        output = self.ccf_dir.figure_output(filename) if not self.debug_mode else None

        with plot_figure(output) as _:
            plot_categorical_region(
                norm.dataframe(),
                norm.classified_column,
                norm.value_column,
                show_area_name=False,
                ylabel=norm.normalized_unit
            )

    @dispatch('parallel')
    def plot_parallel_normalized(self, roi: RSCRoiClassifierDataFrame):
        """Specify an area, and see the values of different normalization methods"""
        if self.area is None:
            raise RuntimeError('specify area(s)')

        n_plots = len(get_args(ROIS_NORM_TYPE))
        output = self.ccf_dir.figure_output(f'{self.filename}') if not self.debug_mode else None

        with plot_figure(output, 1, n_plots) as ax:
            for i, norm in enumerate(get_args(ROIS_NORM_TYPE)):
                norm = (
                    roi.to_normalized(norm, self.merge_level, top_area=self.top_area, hemisphere=self.hemisphere)
                    .filter_areas(self.area)
                )

                idx = {val: idx for idx, val in enumerate(['aRSC', 'pRSC', 'overlap'])}
                df = norm.dataframe().sort(pl.col('source').replace(idx))

                ch = df['source'].to_list()
                values = df[norm.value_column].to_list()

                plot_rois_bar(ax[i], ch, values,
                              color=[REGION_COLORS_HIST[c] for c in ch],
                              fullname=False,
                              legend=False,
                              ylabel=norm.normalized_unit)
                ax[i].set_title(self.area)

    @dispatch('bar')
    def plot_roi_bar(self, roi: RSCRoiClassifierDataFrame):
        """Bar plot for each channel, y-axis could be either `channel-wise normalized percent` OR `counts`"""
        output = self.ccf_dir.figure_output(f'{self.filename}_{self.dispatch_plot}') if not self.debug_mode else None

        with plot_figure(output, 3, 1) as ax:
            for i, src in enumerate(roi.sources):
                norm = roi.to_normalized(self.roi_norm, self.merge_level, source=src, hemisphere=self.hemisphere)
                df = norm.dataframe().sort(norm.value_column, descending=True)  # re-sort

                plot_rois_bar(ax[i],
                              df[norm.classified_column].to_list(),
                              df[norm.value_column],
                              color=REGION_COLORS_HIST[src],
                              legend=False, xlabel=src,
                              ylabel=norm.normalized_unit)

    @dispatch('pie')
    def plot_roi_pie(self, roi: RSCRoiClassifierDataFrame):
        """Pie chart for each channel, top region and merge level were force set"""
        self.top_area = 6
        self.merge_level = 1

        cmapper = DiscreteColorMapper('PiYG', 10)
        output = self.ccf_dir.figure_output(self.filename) if not self.debug_mode else None

        with plot_figure(output, 1, roi.n_channels, figsize=(10, 6)) as _ax:
            for i, src in enumerate(roi.sources):
                norm = roi.to_normalized(self.roi_norm, self.merge_level, top_area=self.top_area, source=src,
                                         hemisphere=self.hemisphere, rest_as_others=True)
                plot_foreach_channel_pie(_ax[i], norm.dataframe(), norm.classified_column, src, self.merge_level, cmapper=cmapper)

    @dispatch('venn')
    def plot_roi_venn(self, roi: RSCRoiClassifierDataFrame):
        """Venn diagram of each region counts"""

        norm = roi.to_normalized(self.roi_norm, self.merge_level, top_area=self.top_area, hemisphere=self.hemisphere)
        d = prepare_venn_data(norm.dataframe(), norm.classified_column)

        n_areas = len(d)
        nr = int(np.sqrt(n_areas))
        nc = int(n_areas // nr + 1)
        output = self.ccf_dir.figure_output(self.filename) if not self.debug_mode else None

        with plot_figure(output, nr, nc) as ax:
            ax = ax.ravel()

            for i, (k, v) in enumerate(d.items()):
                venn2(subsets=v, set_labels=('aRSC', 'pRSC'), ax=ax[i], set_colors=('gold', 'purple'))
                ax[i].set_title(k)
                ax[i].set_axis_off()
                ax[i].set_xticks([])
                ax[i].set_yticks([])

            for n in range(nr * nc):
                if n > i:
                    ax[n].set_visible(False)

    @dispatch('bias')
    def plot_bias_index(self, roi: RSCRoiClassifierDataFrame, *,
                        uni_color: bool = False,
                        exclude_index_range: tuple[float, float] | None = (-0.5, 0.5)):
        """Plot bias index using `percentage` (per channel) normalization method
        note that opt.supply_overlap default is set as True

        :param roi:  ``RSCRoiClassifierDataFrame``
        :param uni_color: uni bar color, otherwise gradient color changes
        :param exclude_index_range: exclude range with low abs index
        :return:
        """
        norm = roi.to_normalized(self.roi_norm, self.merge_level, top_area=self.top_area, hemisphere=self.hemisphere)
        df = norm.to_bias_index('pRSC', 'aRSC')

        if exclude_index_range is not None:
            lb, ub = exclude_index_range
            df = df.filter((pl.col('bias_index') < lb) | (pl.col('bias_index') > ub))

        x = df.get_column(norm.classified_column).to_list()
        values = df['bias_index']

        if uni_color:
            clist = ['magenta' if v >= 0 else 'y' for v in values]
        else:
            clist = []
            na = sum(1 for it in values if it < 0)
            nb = sum(1 for it in values if it >= 0)
            clist.extend(get_customized_cmap('Wistia', (0.7, 0.3), na))
            clist.extend(get_customized_cmap('RdPu', (0.3, 0.8), nb))

        output = self.ccf_dir.figure_output(self.filename) if not self.debug_mode else None

        with plot_figure(output) as ax:
            plot_rois_bar(ax, x, values, color=clist, ylabel='bias_index')
            equ = r'$\log_2[P_{area(pRSC)} / P_{area(aRSC)}]$'
            insert_latex_equation(ax, equ)


if __name__ == '__main__':
    RoiQuantOptions().main()
