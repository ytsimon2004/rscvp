from typing import ClassVar

import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from rscvp.util.cli.cli_hist import HistOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

from argclz import AbstractParser, as_argument, str_tuple_type
from neuralib.plot import plot_figure

__all__ = ['RoiExprRangeBatchOptions']


class RoiExprRangeBatchOptions(AbstractParser, HistOptions):
    DESCRIPTION = 'Plot the roi distribution range in AP/ML axis for batch animals'

    animal = as_argument(HistOptions.animal).with_options(
        type=str_tuple_type,
        help='multiple animals. e.g. YW001,YW002'
    )

    def run(self):
        df = self.collect_range_data()

        with plot_figure(None, 1, 2) as ax:
            self.plot_violin_ap(ax[0], df)
            self.plot_violin_ml(ax[1], df)

    def collect_range_data(self) -> pl.DataFrame:
        ret = []
        for ccf_dir in self.foreach_ccf_dir(self.animal):
            df = pl.read_csv(ccf_dir.parse_csv)
            concat = (
                df.filter(pl.col('source').is_in(['aRSC', 'pRSC']))
                .select('AP_location', 'ML_location', 'source')
                .with_columns(pl.lit(self.animal).alias('animal'))
            )

            ret.append(concat)

        return pl.concat(ret)

    _plot_args: ClassVar[dict] = dict(
        hue='source',
        split=True,
        fill=False,
        gap=.1,
        inner="quart",
        palette=REGION_COLORS_HIST
    )

    def plot_violin_ap(self, ax: Axes, df: pl.DataFrame):
        sns.violinplot(
            ax=ax,
            data=df,
            x='animal',
            y='AP_location',
            **self._plot_args
        )

        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    def plot_violin_ml(self, ax: Axes, df: pl.DataFrame):
        sns.violinplot(
            ax=ax,
            data=df,
            x='ML_location',
            y='animal',
            **self._plot_args
        )

        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


if __name__ == '__main__':
    RoiExprRangeBatchOptions().main()
