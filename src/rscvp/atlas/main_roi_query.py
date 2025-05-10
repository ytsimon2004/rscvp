from pathlib import Path
from typing import Literal, ClassVar

import numpy as np

from argclz import AbstractParser, as_argument, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.ccf.dataframe import RoiSubregionDataFrame
from neuralib.atlas.typing import Area
from neuralib.plot import dotplot
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.atlas.util_plot import plot_rois_bar
from rscvp.util.cli.cli_hist import HistOptions
from rscvp.util.cli.cli_roi import ROIOptions, RSCRoiClassifierDataFrame
from rscvp.util.util_plot import REGION_COLORS_HIST

__all__ = ['RoiQueryOptions']


@publish_annotation('main', project='rscvp', figure='fig.7E-F stacked', caption='@dispatch(stacked)', as_doc=True)
class RoiQueryOptions(AbstractParser, ROIOptions, Dispatch):
    """Plot the fraction of subregions in a queried region"""

    DESCRIPTION = 'Plot the fraction of subregions in a queried region'

    DEFAULT_AREA_SORT: ClassVar[dict[Area, list[Area]]] = {
        'VIS': ['VISp', 'VISam', 'VISpm', 'VISl', 'VISal', 'VISpor', 'VISli', 'VISpl', 'VISC'],
        'RHP': ['ENT', 'SUB', 'PRE', 'POST', 'Pros', 'PAR', 'HATA', 'APr'],
        'ATN': ['AM', 'AV', 'AD', 'IAD', 'IAM', 'LD']
    }
    """Dict of area:list of subarea, used for sorting to make plot consistent"""

    area = as_argument(HistOptions.area).with_options(
        metavar='BRAIN_AREA',
        required=True,
        help='region name for query'
    )

    dispatch_plot: Literal['bar', 'dot', 'stacked'] = argument(
        '-g', '--graph',
        default='stacked',
        help='graph type options',
    )

    source_order = ('aRSC', 'pRSC', 'overlap')
    disable_overlap_in_plot = True
    roi_norm = 'channel'

    def run(self):
        self.setup_logger(Path(__file__).name)

        ccf_dir = self.get_ccf_dir()
        roi = self.load_roi_dataframe(ccf_dir)

        # single region
        if len(self.area) == 1:
            result = self._get_single_query(roi)
            file = f'{self.area}_{self.dispatch_plot}.pdf'
            out = ccf_dir.output_folder / file if not self.debug_mode else None
            self.invoke_command(self.dispatch_plot, result, out)

        # multiple regions
        else:
            results = self._get_multiple_query(roi)
            out = ccf_dir.output_folder / 'stacked_multiple.pdf' if not self.debug_mode else None
            self.plot_stacked_multiregions(results, out)

    def _get_single_query(self, df: RSCRoiClassifierDataFrame) -> RoiSubregionDataFrame:
        area = self.area[0]
        sub = df.to_subregion(area)
        if self.disable_overlap_in_plot:
            sub = sub.filter_overlap()
        return sub

    def _get_multiple_query(self, df: RSCRoiClassifierDataFrame) -> list[RoiSubregionDataFrame]:
        if self.disable_overlap_in_plot:
            return [df.to_subregion(a, source_order=self.source_order) for a in self.area]
        else:
            return [df.to_subregion(a, source_order=self.source_order).filter_overlap() for a in self.area]

    @dispatch('bar')
    def plot_bar_foreach(self, result: RoiSubregionDataFrame, output: Path | None):
        """Plot the bar plot for each source"""
        nr = 2 if self.disable_overlap_in_plot else 3
        with plot_figure(output, nr, 1, sharey=True) as ax:
            for i, row in enumerate(result.dataframe().iter_rows()):
                source = row[0]
                values = row[1:]
                plot_rois_bar(ax[i], result.subregion, values, color=REGION_COLORS_HIST[source], title=source)

    @dispatch('dot')
    def plot_dot_foreach(self, result: RoiSubregionDataFrame, output: Path | None):
        """Plot the dot plot for each source"""
        with plot_figure(output, default_style=False) as ax:
            dotplot(result.sources,
                    result.subregion,
                    result.to_numpy(),
                    scale='area',
                    ax=ax)

    @dispatch('stacked')
    def plot_stacked_foreach(self, result: RoiSubregionDataFrame,
                             output: Path | None,
                             with_width: bool = False,
                             fill: bool = True):
        """
        Plot the stacked bar in a single region

        :param result: :class:`~neuralib.atlas.ccf.dataframe.RoiSubregionDataFrame`
        :param output: fig output path
        :param with_width: if X axis indicates the proportion of inputs in channels, otherwise, used default value
        :param fill: if fill the stacked bar with all regions
        """
        dy = result.to_dict(as_series=False)
        # sort
        sort_list = self.DEFAULT_AREA_SORT.get(self.area[0], None)
        if sort_list is not None:
            dy = {area: dy[area] for area in sort_list if area in dy}

        # bar width
        if with_width:
            width = [w * 5 for w in result.profile['total_fraction']]
        else:
            width = 0.4

        with plot_figure(output, figsize=(4, 8)) as ax:
            bottom = np.zeros(len(result.sources))
            for area, value in dy.items():
                ax.bar(result.sources, value, bottom=bottom, width=width, fill=fill)
                bottom += value

                for icol, loc in enumerate(bottom):
                    ax.text(icol, loc - 2, area)

            ax.set(ylabel=f'fraction of inputs from ({result.region})')

    def plot_stacked_multiregions(self, results: list[RoiSubregionDataFrame],
                                  output: Path | None,
                                  fill: bool = True):
        """
        Plot the stacked bar in separated channels
        yaxis is the percentage of subregions within each region
        xaxis is each region, width indicates the proportion of inputs in different regions.

        Shape Info:

            R: number of regions (queried)
            C: number of channels
            sR: number of subregions (common max for all the queried regions)

        :param results: list of :class:`~neuralib.atlas.ccf.dataframe.RoiSubregionDataFrame`
        :param output:
        :param fill:

        """
        n_channels = 3
        n_regions = len(results)
        n_subregions = max([res.n_subregion for res in results])  # common max

        dat = np.zeros((n_regions, 3, n_subregions))  # (R, C, sR)
        subregion_name = np.full((n_regions, n_subregions), '', dtype=object)  # (R, sR)
        width = np.zeros((n_regions, 3))  # (R, C)

        for i, res in enumerate(results):
            val = res.to_numpy()
            n = val.shape[1]
            dat[i, :, :n] = val
            subregion_name[i, :n] += res.subregion
            width[i] = np.array(res.profile['total_fraction'])

        width *= 2
        subregion_name = subregion_name.reshape(n_regions, n_subregions)
        with plot_figure(output, n_channels, 1) as ax:
            for ich in range(n_channels):
                bottom = np.zeros(n_regions)
                for isub in range(n_subregions):
                    value = dat[:, ich, isub]
                    ax[ich].bar(self.area, value, bottom=bottom, width=width[:, ich], fill=fill)
                    bottom += value

                    for icol, loc in enumerate(bottom):
                        ax[ich].text(icol, loc - 2, subregion_name[icol, isub])

                    ax[ich].set(ylabel=self.source_order[ich])


if __name__ == '__main__':
    RoiQueryOptions().main()
