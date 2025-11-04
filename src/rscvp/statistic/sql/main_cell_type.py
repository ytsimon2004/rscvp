import math
from typing import Literal, Final

import polars as pl

from argclz import as_argument, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure
from neuralib.plot.venn import VennDiagram
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.database import DB_TYPE
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

__all__ = ['CellTypeStat']


@publish_annotation('sup', project='rscvp', figure='fig.S5A', caption='@dispatch(stacked)', as_doc=True)
class CellTypeStat(StatPipeline, Dispatch):
    DESCRIPTION = 'Cell type venn diagram/stacked bar, or print as a table'

    header = as_argument(StatisticOptions.header).with_options(required=False)

    show_type: Literal['foreach', 'combine', 'table', 'stacked'] = argument(
        '--show',
        required=True,
        default='combine',
        help='whether foreach dataset, Or combine as a big venn'
    )

    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))

    sheet_name: Final[GSPREAD_SHEET_PAGE] = 'GenericClassDB'
    db_table: Final[DB_TYPE] = 'GenericClassDB'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False)
        self.plot()

    def plot(self):
        self.invoke_command(self.show_type, self.df)

    @dispatch('foreach')
    def plot_foreach_venn(self, df: pl.DataFrame):
        row, col = math.ceil(df.shape[0] / 5), 5

        with plot_figure(None, row, col) as ax:
            ax = ax.ravel()
            for i, fields in enumerate(df.iter_rows(named=True)):
                info = f"{fields['date']}_{fields['animal']}"
                region = fields['region']

                t = int(fields['n_selected_neurons'])
                v = int(fields['n_visual_neurons'])
                s = int(fields['n_spatial_neurons'])
                o = int(fields['n_overlap_neurons'])

                #
                subsets = {'visual': v - o, 'spatial': s - o}
                vd = VennDiagram(subsets, ax=ax[i])
                vd.add_total(t)
                vd.add_intersection('visual & spatial', o)
                vd.plot(add_title=False)
                vd.ax.set_title(f'{info}_{region}')

    @dispatch('combine')
    def plot_combine_venn(self, df: pl.DataFrame):
        diagram_list: list[VennDiagram] = []
        areas = ('aRSC', 'pRSC')
        for area in areas:
            data = (
                df.filter(pl.col('region') == area)
                .select('n_selected_neurons', 'n_visual_neurons', 'n_spatial_neurons', 'n_overlap_neurons').sum()
                .to_numpy()
                .astype(int)
            )

            data = data.flatten()
            t, v, s, o = data
            subsets = {'visual': v - o, 'spatial': s - o}

            vd = VennDiagram(subsets)
            vd.add_intersection('visual & spatial', data[3])
            vd.add_total(data[0])
            diagram_list.append(vd)

        #
        with plot_figure(None, 2, 1) as ax:
            for i, vd in enumerate(diagram_list):
                vd.ax = ax[i]
                vd.plot()
                title = areas[i] + '\n' + vd.ax.get_title()
                vd.ax.set_title(title)

    @dispatch('stacked')
    def plot_stacked_bar(self, df: pl.DataFrame):
        """Plot the stacked bar for each cell type"""
        with plot_figure(None, 1, 2) as ax:
            for i, df in enumerate(df.partition_by('region')):
                # get fraction of each type
                t = df['n_selected_neurons'].sum()
                v = df['n_visual_neurons'].sum() / t
                s = df['n_spatial_neurons'].sum() / t
                o = df['n_overlap_neurons'].sum() / t
                u = 1 - v - s - o

                bottom = 0
                val = (s, v, o, u)
                names = ('spatial', 'visual', 'overlap', 'unclassified')
                colors = ('palegreen', 'violet', 'gray', 'silver')
                for j, v in enumerate(val):
                    ax[i].bar(df['region'][0], v, bottom=bottom, color=colors[j], label=names[j])
                    bottom += v

                ax[i].legend()
                ax[i].set_title('\n'.join([f"{name}: {value:.3f}" for name, value in zip(names[::-1], val[::-1])]))

    @dispatch('table')
    def show_statistic_table(self, df: pl.DataFrame):
        from neuralib.util.table import rich_data_frame_table
        expr = pl.sum('n_total_neurons',
                      'n_selected_neurons',
                      'n_visual_neurons',
                      'n_spatial_neurons',
                      'n_overlap_neurons')

        df = df.group_by('region').agg(pl.len().alias('n_dataset'), expr)
        rich_data_frame_table(df)


if __name__ == '__main__':
    CellTypeStat().main()
