from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from rscvp.util.cli.cli_roi import ROIOptions
from rscvp.util.util_plot import ROIS_COLORS

from argclz import AbstractParser, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure
from neuralib.util.utils import joinn

__all__ = ['RoiExpressionOptions']


class RoiExpressionOptions(AbstractParser, ROIOptions, Dispatch):
    DESCRIPTION = 'Plot all the ROI expression histogram along the certain anatomical axis'
    LABEL = ['pRSC-proj', 'aRSC-proj', 'both-proj']

    dispatch_plot: Literal['1d', '2d'] = argument(
        '--plot-type', '--plot',
        default='1d',
        help='expression for 1d or 2d view'
    )

    axis: Literal['ap', 'ml', 'dv', 'ap-dv', 'ml-ap'] = argument(
        '--axis',
        default='ap-dv',
        help='which axis for plotting'
    )

    bins: int = argument(
        '--bins',
        default=30,
        help='bin number of each axis'
    )

    def run(self):
        ccf_dir = self.get_ccf_dir()
        output_file = ccf_dir.figure_output(self.fig_title) if not self.debug_mode else None
        df = self.load_roi_dataframe(ccf_dir).dataframe()

        self.invoke_command(self.dispatch_plot, df, output_file)

    @property
    def fig_title(self) -> str:
        return joinn('_', self.hemisphere, self.dispatch_plot, self.axis, 'expr')

    @cached_property
    def hemi_expr(self) -> pl.Expr:
        match self.hemisphere:
            case 'ipsi' | 'contra':
                return pl.col('hemi.') == self.hemisphere
            case _:
                return pl.col('hemi.').is_in(('ipsi', 'contra'))

    @dispatch('1d')
    def plot_expression_1d(self, df: pl.DataFrame, output: Path | None):

        axis = self.get_axes()
        dataset = self.get_coordinates(df, axis)

        with plot_figure(output, 1, 3, figsize=(7.5, 3.5), set_square=True) as axes:
            for i, data in enumerate(dataset):
                ax = axes[i]
                label = self.LABEL[i]
                sns.histplot(
                    data=data,
                    bins=self.bins,
                    kde=True,
                    element="step",
                    color=ROIS_COLORS[label],
                    label=label,
                    ax=ax
                )
                ax.set(xlabel=f'{self.axis} distance(mm)', ylabel='count')
                ax.legend()
                ax.invert_xaxis()

    @dispatch('2d')
    def plot_expression_2d(self,
                           df: pl.DataFrame,
                           output: Path | None):
        axis = self.get_axes()
        dataset = self.get_coordinates(df, axis)

        with plot_figure(output, 1, 3, figsize=(7.5, 3.5), set_square=True) as axes:
            for i, data in enumerate(dataset):
                ax = axes[i]
                sns.histplot(
                    x=data[:, 0],
                    y=data[:, 1],
                    bins=self.bins,
                    cbar=True,
                    cbar_kws=dict(shrink=0.25),
                    color=ROIS_COLORS[self.LABEL[i]],
                    ax=ax
                )

                self._ax_set_2d_expr(ax)

    def get_coordinates(self, df: pl.DataFrame,
                        axis: str | list[str]) -> tuple[np.ndarray, ...]:
        """
        extract coordinates from dataframe in given axis
        :param df:
        :param axis:
        :return:
            r: 1d | 2d coordinates (mm) array
            g: 1d | 2d coordinates (mm) array
            1d | 2d coordinates (mm) array
        """
        r = df.filter((pl.col('channel') == 'rfp') & self.hemi_expr)[axis].to_numpy()
        g = df.filter((pl.col('channel') == 'gfp') & self.hemi_expr)[axis].to_numpy()
        o = df.filter((pl.col('channel') == 'overlap') & self.hemi_expr)[axis].to_numpy()

        return r, g, o

    def get_axes(self) -> str | list[str]:
        match self.axis:
            case 'ap':
                return 'AP_location'
            case 'ml':
                return 'ML_location'
            case 'dv':
                return 'DV_location'
            case 'ap-dv':
                return ['AP_location', 'DV_location']
            case 'ml-ap':
                return ['ML_location', 'AP_location']
            case _:
                raise ValueError('')

    def _ax_set_2d_expr(self, ax: Axes):
        if self.axis == 'ap-dv':
            xlim = (-5, 3)
            ylim = (0, 8)
            xlabel = 'ap distance (mm)'
            ylabel = 'dv distance (mm)'
            ax.invert_yaxis()
        elif self.axis == 'ml-ap':
            xlim = (-5, 5)
            ylim = (-5, 3)
            xlabel = 'ml distance (mm)'
            ylabel = 'ap distance (mm)'
        else:
            raise ValueError(f'{self.axis}')

        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        ax.set_xticks(np.arange(*xlim, 1))
        ax.set_yticks(np.arange(*ylim, 1))
        ax.grid(which='both', alpha=0.7)
        ax.invert_xaxis()


if __name__ == '__main__':
    RoiExpressionOptions().main()
