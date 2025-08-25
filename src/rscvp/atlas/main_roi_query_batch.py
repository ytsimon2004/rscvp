from typing import Final, Literal

import numpy as np
import polars as pl

from argclz import AbstractParser, as_argument, str_tuple_type, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.ccf import ROIS_NORM_TYPE
from neuralib.atlas.typing import Area, TreeLevel
from neuralib.plot import plot_figure, dotplot
from neuralib.util.verbose import printdf, fprint
from rscvp.atlas.util_plot import DEFAULT_AREA_SORT
from rscvp.util.cli import HistOptions, ROIOptions

__all__ = ['RoiQueryBatchOptions']


class RoiQueryBatchOptions(AbstractParser, ROIOptions, Dispatch):
    DESCRIPTION = 'Dotplot for subareas from a single area (foreach channel, animal)'

    animal = as_argument(HistOptions.animal).with_options(
        type=str_tuple_type,
        help='multiple animals. e.g. YW001,YW002'
    )

    area: Area = as_argument(HistOptions.area).with_options(
        type=str,
        help='single area query only',
    )

    force_set_show_col: TreeLevel | None = argument(
        '--show-col',
        metavar='LEVEL',
        default=None,
        help='force set show col to which level'
    )

    dispatch_plot: Literal['dot', 'hbar'] = argument(
        '--plot',
        default='hbar',
        help='graph type options',
    )

    roi_norm: Final[ROIS_NORM_TYPE] = 'channel'

    def run(self):
        df = self.get_batch_subregion_data()
        print(df)
        self.invoke_command(self.dispatch_plot, df)
        self.print_var(df)

    def get_batch_subregion_data(self) -> pl.DataFrame:
        ret = []
        for ccf_dir in self.foreach_ccf_dir():
            subregion = (
                self.load_roi_dataframe(ccf_dir)
                .to_subregion(self.area, source_order=('aRSC', 'pRSC', 'overlap'),
                              show_col=self.force_set_show_col,
                              animal=ccf_dir.animal)
            )

            ret.append(subregion.dataframe())

        return pl.concat(ret, how='diagonal').fill_null(0)

    @dispatch('hbar')
    def hbar_batch(self, df: pl.DataFrame):
        df = df.filter(~pl.col('source').is_in(['overlap']))
        brain_regions = [col for col in df.columns if col not in ['source', 'animal']]

        # sort
        if self.area in DEFAULT_AREA_SORT:
            area_order = DEFAULT_AREA_SORT[self.area]
            sorted_regions = [region for region in area_order if region in brain_regions]
            remaining_regions = [region for region in brain_regions if region not in area_order]
            brain_regions = sorted_regions + remaining_regions

        source_data = {}

        for source_name, source_df in df.group_by('source'):
            source_name = source_name[0]
            source_region_data = source_df.select(brain_regions)
            individual_data = source_region_data.to_numpy()
            n_animals = len(source_df)

            means = source_region_data.mean().to_numpy().flatten()
            stds = source_region_data.std().to_numpy().flatten()

            source_data[source_name] = {
                'means': means,
                'sems': stds / np.sqrt(n_animals),
                'individual': individual_data,
                'n_animals': n_animals
            }

        with plot_figure(None, figsize=(12, 6), tight_layout=True) as ax:
            y_pos = np.arange(len(brain_regions))
            colors = {'aRSC': '#1f77b4', 'pRSC': '#ff7f0e'}

            for source, data in source_data.items():
                color = colors.get(source, '#2ca02c')
                sign = -1 if source == 'aRSC' else 1
                x_values = sign * data['means']

                ax.barh(y_pos, x_values, 0.6, label=f'{source} (n={data["n_animals"]})', color=color, alpha=0.8)
                ax.errorbar(x_values, y_pos, xerr=data['sems'], fmt='none', color='black', capsize=3, capthick=1)

                individual_points = data['individual']
                for region_idx in range(len(brain_regions)):
                    region_values = individual_points[:, region_idx]
                    y_scatter = np.full(len(region_values), y_pos[region_idx])
                    ax.scatter(sign * region_values, y_scatter,
                               c='white', s=40, edgecolors='black', linewidths=0.5, alpha=0.9, zorder=3)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(brain_regions)
            ax.set_xlabel(f'fraction from {self.area} (%)')
            ax.set_ylabel(f'{self.area} subregions')
            ax.legend(loc='best')
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{abs(tick):.2f}' for tick in ticks])

    @dispatch('dot')
    def dotplot_batch(self, df: pl.DataFrame):
        with plot_figure(None, 1, 3) as ax:
            for i, (name, dat) in enumerate(df.group_by(['source'])):
                areas = dat.drop('source', 'animal').columns
                size = dat.select(areas).to_numpy()  # (Animal, Area)
                dotplot(dat['animal'], areas, size, scale='area', ax=ax[i])
                ax[i].set_title(dat['source'][0])

    @staticmethod
    def print_var(concat_df: pl.DataFrame):
        """statistic info"""
        for k, df in concat_df.partition_by('source', as_dict=True).items():
            dfx = df.drop('source', 'animal')

            mean = dfx.mean().with_columns(pl.lit('MEAN').alias('measurement'))
            sem = dfx.select([
                (pl.std(col) / pl.count(col).cast(float).sqrt()).alias(col)
                for col in dfx.columns
            ]).with_columns(pl.lit('STDDEV').alias('measurement'))

            fprint(f'{k}')
            printdf(pl.concat([mean, sem], how="diagonal"))


if __name__ == '__main__':
    RoiQueryBatchOptions().main()
