from typing import Literal, Iterable

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from rscvp.util.cli.cli_celltype import SessionDataFrame, CellTypeSelectionOptions
from rscvp.util.cli.cli_output import DataOutput

from argclz import AbstractParser, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation

__all__ = ['SpatialSessionStat']


@publish_annotation('appendix', caption='visual impact in spatial activity')
class SpatialSessionStat(AbstractParser, CellTypeSelectionOptions, Dispatch):
    DESCRIPTION = 'Compare the spatial variable (foreach cells) across different sessions'

    dispatch_plot: Literal['scatter', 'box', 'violin', 'dot', 'connect'] = argument(
        '--plot',
        default='box',
        help='plot type',
    )

    norm_ses: str | None = argument(
        '--norm-ses',
        default=None,
        help='normalize each variable to its specific session value'
    )

    pre_selection = True
    cell_type = 'spatial'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        rig = self.load_riglog_data()
        df = self.get_session_dataframe(rig)
        if self.norm_ses is not None:
            df = df.with_session_norm(self.norm_ses)

        output_info = self.get_data_output('stat_ses')
        self.invoke_command(self.dispatch_plot, df, output_info)

    def fig_output(self, output: DataOutput, var: str):
        return output.summary_figure_output(var, f'{self.dispatch_plot}', 'norm' if self.norm_ses else None)

    def run_statistic(self,
                      ssdf: SessionDataFrame,
                      var: str,
                      output: DataOutput,
                      df: pd.DataFrame | None = None) -> pd.DataFrame:
        if df is None:
            df = self.get_var_dataframe(ssdf, var)

        pair_format_df = pd.melt(df,
                                 id_vars='neuron_id',
                                 value_vars=ssdf.sessions,
                                 value_name=var,
                                 var_name='session')
        stat = pg.pairwise_tests(pair_format_df,
                                 dv=var,
                                 within='session',
                                 subject='neuron_id',
                                 parametric=True,
                                 padjust='bonf')

        stat.to_csv(output.directory / f'{var}_statistic.csv')

        return stat

    @staticmethod
    def get_var_dataframe(ssdf: SessionDataFrame, var: str) -> pd.DataFrame:
        return ssdf.unpivot(var).to_pandas()

    @dispatch('box')
    def plot_session_box(self,
                         ssdf: SessionDataFrame,
                         output: DataOutput):
        for (df, var) in self._prepare_columns_statistic(ssdf, output):
            with plot_figure(self.fig_output(output, var), figsize=(6, 10)) as ax:
                plot_ses_box(ax, df, ylabel=var)

    @dispatch('violin')
    def plot_session_violin(self, ssdf: SessionDataFrame,
                            output: DataOutput):
        for (df, var) in self._prepare_columns_statistic(ssdf, output):
            with plot_figure(self.fig_output(output, var), figsize=(6, 10)) as ax:
                plot_ses_violin(ax, df, ylabel=var)

    @dispatch('connect')
    def plot_session_connect(self, ssdf: SessionDataFrame,
                             output: DataOutput):
        for (df, var) in self._prepare_columns_statistic(ssdf, output):
            with plot_figure(self.fig_output(output, var), figsize=(6, 10)) as ax:
                plot_ses_connect(ax, df, ylabel=var)

    def _prepare_columns_statistic(self, ssdf: SessionDataFrame,
                                   output: DataOutput) -> Iterable[tuple[pd.DataFrame, str]]:
        """run statistic & prepare column for plot"""
        for var in ssdf.variables:
            df = self.get_var_dataframe(ssdf, var)
            self.run_statistic(ssdf, var, output, df=df)
            df.drop(columns='neuron_id', inplace=True)
            yield df, var

    # ======== #
    # Dot Plot #
    # ======== #

    @DispatchOption.dispatch('dot')
    def plot_session_dot(self, ssdf: SessionDataFrame,
                         output: DataOutput):
        for var in ssdf.variables:
            df = self.get_var_dataframe(ssdf, var)
            self.run_statistic(ssdf, var, output, df=df)
            df.drop(columns='neuron_id', inplace=True)
            with plot_figure(self.fig_output(output, var), figsize=(6, 10)) as ax:
                plot_ses_dot(ax, df, ylabel=var)

    @DispatchOption.dispatch('scatter')
    def plot_pair_scatter(self, ssdf: SessionDataFrame,
                          output: DataOutput,
                          **kwargs):
        """plot the scatter to compare certain variable in individual neurons in different behavioral sessions"""
        for var in ssdf.variables:
            df = self.get_var_dataframe(ssdf, var)
            self.run_statistic(ssdf, var, output, df=df)

            with plot_figure(self.fig_output(output, var), 1, 3) as ax:
                s = ssdf.sessions
                self._plot_pair_scatter(ax[0], df, var, session=(s[0], s[1]), **kwargs)
                self._plot_pair_scatter(ax[1], df, var, session=(s[1], s[2]), **kwargs)
                self._plot_pair_scatter(ax[2], df, var, session=(s[0], s[2]), **kwargs)

    @staticmethod
    def _plot_pair_scatter(ax: Axes,
                           df: pd.DataFrame,
                           var: str,
                           session: tuple[str, str], *,
                           show_linear_reg=False,
                           show_kernel_density=True):
        value_a = df[session[0]].to_numpy()
        value_b = df[session[1]].to_numpy()
        ax.plot(value_a, value_b, 'ko', markersize=0.5, alpha=0.3)

        if show_linear_reg:
            from scipy.stats import linregress
            res = linregress(value_a, value_b)
            r_sqrt = res.rvalue ** 2
            ax.plot(value_a, res.intercept + res.slope * value_a, 'crimson', lw=1)
            ax.set_title(f'R2: {r_sqrt:.3f}')

        lb = np.min([value_a, value_b])
        ub = np.max([value_a, value_b])

        if show_kernel_density:
            Z = create_kernel_density_scatter(value_a, value_b)
            ax.imshow(np.rot90(Z),
                      cmap=plt.cm.gist_earth_r,
                      extent=[lb, ub, lb, ub])

        ax.set(xlim=(lb, ub), ylim=(lb, ub),
               xlabel=f'{var}_{session[0]}',
               ylabel=f'{var}_{session[1]}')

        ax.axline((0.5, 0.5), slope=1, color='r', alpha=0.5, lw=1)
        ax.set_aspect('equal', adjustable='box')


def plot_ses_box(ax: Axes,
                 df: pd.DataFrame,
                 **kwargs):
    """
    box plot with single data point and connected line

    :param ax:
    :param df:
    :return:
    """
    sns.boxplot(ax=ax, data=df, showfliers=False, width=0.5, color='white')
    ax.plot(np.mean(df.values, axis=0), color='r', ls='-', lw=1)
    ax.set(**kwargs)


def plot_ses_violin(ax: Axes,
                    df: pd.DataFrame,
                    **kwargs):
    """
    box plot with single data point and connected line

    :param ax:
    :param df:
    :return:
    """
    sns.violinplot(ax=ax, data=df, showfliers=False, width=0.5)
    ax.plot(np.mean(df.values, axis=0), color='r', ls='-', lw=1)
    ax.set(**kwargs)


def plot_ses_connect(ax: Axes,
                     df: pd.DataFrame,
                     **kwargs):
    """
    connected line with mean value

    :param ax:
    :param df:
    :return:
    """

    for n in df.values:
        ax.plot(n, color='k', alpha=0.2, ls='-', lw=0.4, zorder=-1)

    ax.plot(np.mean(df.values, axis=0), color='r', ls='-', lw=1)
    ax.set(**kwargs)


def plot_ses_dot(ax: Axes,
                 df: pd.DataFrame,
                 **kwargs):
    """
    plot single data point

    :param ax:
    :param df:

    :return:
    """
    sns.stripplot(ax=ax, data=df, jitter=True, size=2, alpha=0.3)
    ax.plot(np.mean(df.values, axis=0), color='r', ls='-', lw=1)
    ax.set(**kwargs)


def create_kernel_density_scatter(x: np.ndarray,
                                  y: np.ndarray,
                                  grid_xy: tuple[complex, complex] = (100j, 100j)) -> np.ndarray:
    """

    :param x: value a. i.e., condition A
    :param y: value b. i.e., condition B
    :param grid_xy: grid resolution
    :return:
    """
    import scipy.stats as stats
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xstep, ystep = grid_xy

    X, Y = np.mgrid[xmin:xmax:xstep, ymin:ymax:ystep]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)

    return np.reshape(kernel(positions).T, X.shape)


if __name__ == '__main__':
    SpatialSessionStat().main()
