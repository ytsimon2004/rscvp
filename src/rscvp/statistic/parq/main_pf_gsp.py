from typing import Literal, Final

import numpy as np
import pingouin as pg
import polars as pl
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from rich.pretty import pprint

from argclz import as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import fprint, publish_annotation
from rscvp.statistic._var import PF_HEADERS
from rscvp.statistic.core import StatPipeline
from rscvp.util.util_plot import REGION_COLORS_HIST
from rscvp.util.util_stat import DataSetType, CollectDataSet

__all__ = ['PlaceFieldParQ']


@publish_annotation('main', project='rscvp', figure='fig.2G', as_doc=True)
class PlaceFieldParQ(StatPipeline):
    DESCRIPTION = 'place-field statistic across animals'

    header: str = as_argument(StatPipeline.header).with_options(choices=PF_HEADERS)

    dependent = False
    parametric = False
    load_source: Final = 'parquet'

    def post_parsing(self):
        if self.header == 'n_pf' and self.animal_based_comp:
            raise RuntimeError('non scientific comparison')

    def run(self):
        self.post_parsing()

        if self.animal_based_comp:
            self.load_table(to_pandas=False)
            self.plot_pairwise_mean()
        else:
            self.load_table(to_pandas=False if self.header == 'n_pf' else True)
            self.mkdir_structure()
            self.plot()

    # ============ #
    # Neuron-based #
    # ============ #

    def plot(self):

        match self.header:
            case 'n_pf':
                self.plot_npf_grouping()
            case 'pf_peak':
                data = self.get_collect_data().data
                self.res = super().generate_stat_result()
                self.plot_pf_peak_loc(data)
            case 'pf_width':
                data = self.get_collect_data().data
                self.res = super().generate_stat_result()
                self.plot_pf_width(data)
            case _:
                raise ValueError(f'invalid header: {self.header}')

    # ====================== #
    # Numbers of place field #
    # ====================== #

    def _get_collect_data_npf(self, df: pl.DataFrame) -> CollectDataSet:
        """separated pipeline"""
        n_eval = 4
        df = df.group_by('n_pf', 'region').agg(pl.col('fraction'))

        data = {}
        for it in df.iter_rows(named=True):
            name = f"{it['region']}_{it['n_pf']}"
            data[name] = np.array(it['fraction'])

        return CollectDataSet(
            name=[f'npf_{i}' for i in range(n_eval)],
            data=data,
            group_header=self.group_header,
            test_type='pairwise_ttest',
            data_type='categorical',
            n_categories=n_eval
        )

    def _pg_pairwise_data(self) -> pl.DataFrame:
        """
        return Example ::

            ┌──────────────┬────────┬────────┬──────────┐
            │ Data         ┆ region ┆ n_pf   ┆ fraction │
            │ ---          ┆ ---    ┆ ---    ┆ ---      │
            │ str          ┆ str    ┆ str    ┆ f64      │
            ╞══════════════╪════════╪════════╪══════════╡
            │ 210315_YW006 ┆ aRSC   ┆ n_pf_1 ┆ 0.433121 │
            │ 210401_YW006 ┆ aRSC   ┆ n_pf_1 ┆ 0.572    │
            │ 210402_YW006 ┆ pRSC   ┆ n_pf_1 ┆ 0.42381  │
            │ 210409_YW006 ┆ pRSC   ┆ n_pf_1 ┆ 0.587121 │
            │ 210402_YW008 ┆ aRSC   ┆ n_pf_1 ┆ 0.5      │
            │ …            ┆ …      ┆ …      ┆ …        │
            │ 211208_YW032 ┆ pRSC   ┆ n_pf_4 ┆ 0.0      │
            │ 211202_YW033 ┆ aRSC   ┆ n_pf_4 ┆ 0.004886 │
            │ 211208_YW033 ┆ pRSC   ┆ n_pf_4 ┆ 0.0      │
            │ 221018_YW048 ┆ aRSC   ┆ n_pf_4 ┆ 0.00274  │
            │ 221019_YW048 ┆ pRSC   ┆ n_pf_4 ┆ 0.0      │
            └──────────────┴────────┴────────┴──────────┘
        """
        n_neurons = pl.col('n_pf').list.len()
        pf_expr = pl.col('n_pf').list.count_matches

        df = (
            self.df
            .with_columns((pf_expr(1) / n_neurons).alias('n_pf_1'))
            .with_columns((pf_expr(2) / n_neurons).alias('n_pf_2'))
            .with_columns((pf_expr(3) / n_neurons).alias('n_pf_3'))
            .with_columns((pf_expr(4) / n_neurons).alias('n_pf_4'))
            .drop('n_pf')
            .unpivot(index=['Data', 'region'],
                     on=['n_pf_1', 'n_pf_2', 'n_pf_3', 'n_pf_4'],
                     variable_name='n_pf',
                     value_name='fraction')
            .drop_nans()
        )

        stat_result = pg.pairwise_tests(
            data=df.to_pandas(),
            dv='fraction',
            within='n_pf',
            between='region',
            subject='Data',
            parametric=False
        )

        pl.DataFrame(stat_result).filter(pl.col('n_pf') != '-').write_csv(self.output_statistic_csv)

        return df

    def plot_npf_grouping(self):
        df = self._pg_pairwise_data()
        collect_data = self._get_collect_data_npf(df)
        pprint(collect_data)

        g = sns.catplot(
            data=df,
            kind='bar',
            x='n_pf',
            y='fraction',
            hue='region',
            errorbar=('se', 1),
            palette=REGION_COLORS_HIST
        )

        ax = g.ax

        for f in ax.patches:
            try:
                txt = str(round(f.get_height(), 2)) + '%'
            except AttributeError as e:
                fprint(repr(e), vtype='warning')
                pass
            else:
                txt_x = f.get_x()
                txt_y = f.get_height()
                ax.text(txt_x, txt_y, txt)

        #
        sns.swarmplot(
            data=df,
            x='n_pf',
            y='fraction',
            hue='region',
            dodge=True,
            palette=['g', 'magenta'],
            ax=ax
        )

        self._print_npf_info(df)
        ax.set(ylabel='fraction', xlabel='n_pf')
        ax.tick_params(axis='x', labelrotation=45)

        if self.debug_mode:
            plt.show()
        else:
            plt.savefig(self.output_figure)

    @staticmethod
    def _print_npf_info(df: pl.DataFrame) -> None:
        for key, part in df.partition_by('n_pf', 'region', as_dict=True).items():
            m = part['fraction'].mean()
            se = scipy.stats.sem(part['fraction'])
            print(f'{key}: mean:{m:.3f}, sem:{se:.3f}')

    # ================= #
    # Place Field Width #
    # ================= #

    def plot_pf_width(self, data: DataSetType,
                      plot_type: Literal['cumulative', 'box_mean', 'half_violin_box'] = 'cumulative'):

        with plot_figure(self.output_figure) as ax:

            match plot_type:
                case 'cumulative':
                    for i, (var, width) in enumerate(data.items()):
                        sns.ecdfplot(ax=ax, data=width, color=REGION_COLORS_HIST[var], label=var)
                        ax.set(xlabel='pf_width', ylabel='Cum. Prob.')
                case 'box_mean':
                    sns.boxplot(ax=ax, data=data, showfliers=False, width=0.5, palette=REGION_COLORS_HIST)
                    sns.pointplot(ax=ax, data=data, estimator='mean', color='red')
                    ax.set(ylabel='Place field width (cm)')
                case 'half_violin_box':
                    from neuralib.plot.plot import violin_boxplot
                    violin_boxplot(ax=ax, data=data)
                case _:
                    raise ValueError('Unknown plot type')

            self.insert_pval(ax)
            ax.legend()
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    # ==================== #
    # Place Field Location #
    # ==================== #

    def plot_pf_peak_loc(self, data: DataSetType):
        bin_range = (0, 210) if self.sheet_name == 'vr_parq' else (0, 150)

        with plot_figure(self.output_figure) as ax:
            for i, (var, peak_loc) in enumerate(data.items()):
                n = len(peak_loc)
                sns.histplot(
                    ax=ax,
                    data=peak_loc,
                    binrange=bin_range,
                    bins=50,
                    stat='percent',
                    kde=True,
                    alpha=0.7,
                    color=REGION_COLORS_HIST[var],
                    element='step',
                    label=f'{var} (n={n})'
                )

            ax.set(xlabel='position (cm)', ylabel='Fraction')
            self.insert_pval(ax)
            ax.legend()


if __name__ == '__main__':
    PlaceFieldParQ().main()
