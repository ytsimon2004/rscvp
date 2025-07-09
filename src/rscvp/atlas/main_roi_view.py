import collections
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem, kstest

from argclz import AbstractParser, as_argument, argument, str_tuple_type
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.util import iter_source_coordinates, SourceCoordinates
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.statistic.core import pval_verbose
from rscvp.util.cli import HistOptions, PlotOptions, ROIOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

__all__ = ['RoisViewOptions']


@publish_annotation('appendix', project='rscvp', caption='intra subregion topology (i.e., SUB)')
class RoisViewOptions(AbstractParser, ROIOptions, Dispatch, PlotOptions):
    DESCRIPTION = '3d roi for particular region(s), multiple animals data supported'

    animal = as_argument(HistOptions.animal).with_options(
        type=str_tuple_type,
        help='multiple animals. e.g. YW001,YW002'
    )

    hemisphere = as_argument(HistOptions.hemisphere).with_options(default='ipsi')

    dispatch_plot: Literal['3d', 'hist'] = argument(
        '--analysis',
        required=True,
        help='type of analysis'
    )

    ccf_dir: AbstractCCFDir

    def run(self):
        self.set_background()
        self.invoke_command(self.dispatch_plot)

    @property
    def title(self) -> str:
        return f'{self.area}_{self.hemisphere}_{self.dispatch_plot}'

    def get_batch_data(self) -> dict[str, list[SourceCoordinates]]:
        ret = {}
        for i, ccf_dir in enumerate(self.foreach_ccf_dir()):

            if not ccf_dir.parse_csv.exists():
                self.load_roi_dataframe(ccf_dir)

            coord = iter_source_coordinates(
                ccf_dir.parse_csv,
                only_areas=self.area,
                hemisphere=self.hemisphere,
                to_brainrender=False,
                source_order=('aRSC', 'pRSC')
            )
            ret[ccf_dir.animal] = list(coord)

        return ret

    @dispatch('3d')
    def plot_roi_3d(self):
        """plot the roi in a 3D view"""
        batch_data = self.get_batch_data()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        for animal, coord in batch_data.items():
            for sc in coord:
                ax.scatter(sc.ap, sc.ml, sc.dv,
                           color=REGION_COLORS_HIST[sc.source],
                           s=20,
                           alpha=0.7,
                           edgecolor='none',
                           marker='.')

        ax.invert_zaxis()
        ax.invert_yaxis()
        ax.set(xlabel='AP (mm)', ylabel='ML (mm)', zlabel='DV (mm)')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))

        ax.set_title(self.title)

        if self.debug_mode:
            plt.show()
        else:
            plt.savefig(self.ccf_dir.output_folder / f'{self.title}.pdf')

    @dispatch('hist')
    def plot_topographical_across_axis(self, bins: int = 40):
        """plot histogram for three axes"""
        batch_data = self.get_batch_data()
        axes = ('ap', 'dv', 'ml')
        source_labels = set()

        # all values to compute global bins
        all_values = {axis: [] for axis in axes}
        for coords_list in batch_data.values():
            for sc in coords_list:
                for i, axis in enumerate(axes):
                    all_values[axis].extend(sc.coordinates[:, i])

        # shared bin edges for each axis
        bin_edges = {
            axis: np.histogram_bin_edges(all_values[axis], bins=bins)
            for axis in axes
        }

        # collect histograms per source per animal
        histograms = collections.defaultdict(lambda: {axis: [] for axis in axes})

        for coords_list in batch_data.values():
            for sc in coords_list:

                source = sc.source
                source_labels.add(source)

                for i, axis in enumerate(axes):
                    vals = sc.coordinates[:, i]
                    hist, _ = np.histogram(vals, bins=bin_edges[axis], density=False)
                    hist = hist / hist.sum() * 100

                    histograms[source][axis].append(hist)

        self.kstest_verbose(histograms)
        #
        output_file = None if self.debug_mode else self.ccf_dir.output_folder / f'{self.title}.pdf'

        with plot_figure(output_file, 1, 3, figsize=(10, 6), set_square=True) as ax:
            for i, axis in enumerate(axes):
                centers = (bin_edges[axis][:-1] + bin_edges[axis][1:]) / 2

                for source in sorted(source_labels):
                    hists = np.array(histograms[source][axis])  # (n_animals, n_bins)

                    mean_hist = hists.mean(axis=0)
                    error = sem(hists, axis=0) if hists.shape[0] > 1 else None

                    ax[i].plot(
                        centers, mean_hist,
                        label=source,
                        color=REGION_COLORS_HIST.get(source, 'gray'),
                        linewidth=2
                    )

                    if error is not None:
                        ax[i].fill_between(
                            centers, mean_hist - error, mean_hist + error,
                            alpha=0.3,
                            color=REGION_COLORS_HIST.get(source, 'gray')
                        )

                ax[i].set(xlabel=f'{axis.upper()} (mm)', ylabel='percent (%)')
                ax[i].legend(title="Source")
                ax[i].set_aspect(1.0 / ax[i].get_data_ratio(), adjustable='box')

    @staticmethod
    def kstest_verbose(histograms):
        source_labels = list(histograms.keys())
        source1, source2 = sorted(source_labels)
        axes = ('ap', 'dv', 'ml')

        print(f'K-S Test between sources: {source1} vs {source2}')
        for axis in axes:
            h1 = np.array(histograms[source1][axis])  # shape: (n_animals, n_bins)
            h2 = np.array(histograms[source2][axis])

            v1 = h1.reshape(-1)
            v2 = h2.reshape(-1)

            stat, pval = kstest(v1, v2)
            print(f'{axis.upper()}: {pval_verbose(pval)}')


if __name__ == '__main__':
    RoisViewOptions().main()
