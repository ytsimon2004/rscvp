import colorsys
import random
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from argclz import AbstractParser, str_tuple_type, as_argument, argument
from neuralib.atlas.map import ALLEN_FAMILY_TYPE
from neuralib.atlas.typing import Area
from neuralib.io.json import save_json, load_json
from neuralib.plot import plot_figure, ax_merge
from neuralib.plot.colormap import insert_colorbar
from neuralib.typing import AxesArray
from neuralib.util.logging import LOGGING_IO_LEVEL
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import HistOptions, ROIOptions
from rscvp.util.util_ibl import IBLAtlasPlotWrapper

__all__ = ['RoiTopViewOptions']


@publish_annotation('main', project='rscvp', figure='fig.6G', as_doc=True)
class RoiTopViewOptions(AbstractParser, ROIOptions):
    """Plot ROIs on the top dorsal cortex view"""

    DESCRIPTION = 'Plot ROIs on the top dorsal cortex view'

    animal = as_argument(HistOptions.animal).with_options(
        type=str_tuple_type,
        help='multiple animals. e.g. YW001,YW002'
    )

    merge_level = as_argument(ROIOptions.merge_level).with_options(default=1)

    legend_number_limit: int | None = argument(
        '--limit',
        type=int,
        default=None,
        help='number of roi should larger than how many, then show in legend'
    )

    area_family: ALLEN_FAMILY_TYPE | None = argument(
        '--family',
        default='ISOCORTEX',
        help='only plot rois in allen family'
    )

    as_histogram: bool = argument(
        '--histogram',
        help='plot individual dots as the 2D histogram, otherwise, show individual dots'
    )

    def run(self):
        self.setup_logger(Path(__file__).name)
        df = self.get_roi_dataframe()
        ibl = IBLAtlasPlotWrapper(res_um=10)

        try:
            if self.as_histogram:
                self.plot_top_view_histogram(df, ibl)
            else:
                lut = self._get_area_colormap_lut(df)
                self.plot_top_view_scatter(df, lut, ibl, legend_number_limit=self.legend_number_limit)
        finally:
            del ibl

    @property
    def area_color_lut(self) -> Path:
        """cached file for **area:color** lookup table (reproducing figures)"""
        file = 'area_color_lut.json'
        return self.histology_cache_directory / file

    def get_roi_dataframe(self) -> pl.DataFrame:
        """get roi dataframe for plotting"""
        if len(self.animal) == 1:
            self.animal = self.animal[0]
            ccf_dir = self.get_ccf_dir()
            df = self.load_roi_dataframe(ccf_dir).dataframe()
            self.logger.info(f'reconstruct single animal {self.animal}')
        else:
            df_list = []
            for ccf_dir in self.foreach_ccf_dir():
                df = self.load_roi_dataframe(ccf_dir).dataframe()
                df_list.append(df)

            df = pl.concat(df_list)
            self.logger.info(f'reconstruct multiple animals <{self.animal}>')

        #
        if self.area_family is not None:
            self.logger.info(f'only reconstruct *{self.area_family}* family')
            df = df.filter(pl.col('family') == self.area_family)

        # extract
        df = (
            df.select([self.classified_column, 'source', 'ML_location', 'AP_location'])
            .rename({'ML_location': 'x', 'AP_location': 'y'})
            .with_columns(pl.col('x') * 1000)
            .with_columns(pl.col('y') * 1000)
        )

        return df

    # ============ #
    # 2D histogram #
    # ============ #

    def plot_top_view_histogram(self, df: pl.DataFrame, ibl: IBLAtlasPlotWrapper, bin_size: int = 200):
        """
        Plot the top view roi 2D histogram for each source, with the colormap lookup table for each area.

        :param df: ROI dataframe
        :param ibl: IBLAtlasPlotWrapper instance
        :param bin_size: bin_size in um
        """
        source = df['source'].unique().to_list()
        lut = {'aRSC': 'YlOrBr', 'pRSC': 'RdPu', 'overlap': 'Greys'}

        with plot_figure(None, 1, 3, figsize=(12, 9), sharex=True, sharey=True, tight_layout=False) as _ax:
            for i, src in enumerate(source):
                ax = _ax[i]
                df_src = df.filter(pl.col('source') == src)
                x, y = df_src['x'], df_src['y']

                cmap = matplotlib.colormaps[lut.get(src, 'Greys')]
                cmap.set_under(color='none')

                xbin = max(1, int((x.max() - x.min()) / bin_size))
                ybin = max(1, int((y.max() - y.min()) / bin_size))

                counts, xedges, yedges = np.histogram2d(x, y, bins=(xbin, ybin))
                masked_counts = np.ma.masked_where(counts == 0, counts)

                im = ax.pcolormesh(
                    xedges,
                    yedges,
                    masked_counts.T,
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=1),
                    zorder=-1
                )

                self.plot_ibl_view(ax, ibl)

                insert_colorbar(ax, im)
                ax.set_title(f'{src}: {df_src.shape[0]} ROIs')

    # ========= #
    # Dot ROIs  #
    # ========= #

    def _get_area_colormap_lut(self, df: pl.DataFrame,
                               seed: int = 3,
                               force_create: bool = True,
                               verbose: bool = False) -> dict[Area, str | tuple[float, float, float]]:
        """
        Get the seeding colormap lookup table for the unique areas

        :param df: parsed dataframe
        :param seed: For generate the same color map for brain areas
        :param force_create: force recreate the lookup table
        :return:
        """
        colors = {'VIS': 'magenta', 'MO': 'green', 'SS': 'cyan'}  # default
        areas = df[self.classified_column].unique().sort().to_numpy()

        for a in list(colors.keys()):
            if a not in areas:
                del colors[a]

        if not self.area_color_lut.exists() or force_create:
            random.seed(seed)

            for area in areas:
                if area not in colors:
                    hue = random.uniform(0, 1)
                    saturation = 0.7
                    lightness = 0.6
                    rgb_color = colorsys.hls_to_rgb(hue, saturation, lightness)
                    colors[area] = rgb_color

            save_json(self.area_color_lut, colors)
            self.logger.log(LOGGING_IO_LEVEL, f'SAVE ColorLUT in {self.area_color_lut}')
        else:
            colors = load_json(self.area_color_lut)
            self.logger.log(LOGGING_IO_LEVEL, f'LOAD ColorLUT from {self.area_color_lut}')

            if set(colors) != set(areas):
                colors = self._get_area_colormap_lut(df, seed, force_create=True)

        #
        areas = list(colors.keys())
        assert len(areas) != 0, 'Using wrong merge level, or wrong allen family'

        if verbose:
            self.logger.info(f'AREA: {areas}')

        return colors

    def _filter_rois_counts(self, ext_dataframe: pl.DataFrame,
                            number: int,
                            verbose: bool = True) -> list[Area]:
        """Get the list of area name which less than `number` of ROIs, i.e., for legend show limit"""
        df = (
            ext_dataframe.group_by(self.classified_column)
            .agg(pl.col('source').len().alias('count'))
            .filter(pl.col('count') <= number)
            .sort('count', descending=True)
        )

        if verbose:
            self.logger.warning(f'See counts less than {number}')
            print(df)

        return df[self.classified_column].to_list()

    def plot_top_view_scatter(self, df: pl.DataFrame,
                              colormap_lut: dict[Area, str],
                              ibl: IBLAtlasPlotWrapper,
                              with_legend: bool = True,
                              legend_number_limit: int | None = None):
        """
        Plot the top view roi scatter plot for each source, with the colormap lookup table for each area.

        :param df: ROI dataframe
        :param colormap_lut: brain area:color dict
        :param ibl: IBLAtlasPlotWrapper instance
        :param with_legend: with legend or not, default True
        :param legend_number_limit: number of roi should larger than how many, then show in legend. If None then show all.
        """

        source = df['source'].unique().to_list()

        with plot_figure(None, 3, 3, figsize=(12, 9), sharex=True, sharey=True, dpi=800) as _ax:  # type: AxesArray

            for i, src in enumerate(source):
                ax = ax_merge(_ax)[:2, i]
                self.plot_ibl_view(ax, ibl)

                df_src = df.filter(pl.col('source') == src)
                n_rois = df_src.shape[0]

                for area, dat in df_src.group_by([self.classified_column]):
                    a = area[0]
                    ax.scatter(dat['x'], dat['y'], color=colormap_lut[a], s=0.3, alpha=0.15, label=a, zorder=-1)

                ax.set_title(f'{src}: {n_rois} ROIs')

            if with_legend:
                if legend_number_limit is not None:
                    areas = self._filter_rois_counts(df, legend_number_limit)
                    for a in areas:
                        colormap_lut.pop(a)
                #
                table = [
                    Line2D([0], [0],
                           marker='o',
                           color='none',
                           markeredgecolor='none',
                           markerfacecolor=color,
                           markersize=10,
                           label=label)
                    for label, color in colormap_lut.items()
                ]

                ax = ax_merge(_ax)[2:, :]
                ax.axis('off')
                ax.legend(handles=table, loc='center', ncol=10)

    def plot_ibl_view(self, ax: Axes, ibl: IBLAtlasPlotWrapper):
        """plot ibl dorsal cortex view with given mpl.axes"""
        if self.area_family is None or self.area_family == 'ISOCORTEX':
            ibl.plot_scalar_on_slice(['root'], ax=ax, coord=-2000, plane='top', background='boundary')
        else:
            # Example only, not scientifically appropriate
            ibl.plot_scalar_on_slice(['root'], ax=ax, coord=-2500, plane='horizontal', background='boundary')


if __name__ == '__main__':
    RoiTopViewOptions().main()
