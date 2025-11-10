from typing import Final, Literal

import numpy as np

from argclz import argument
from neuralib.imaging.registration import CellularCoordinates
from neuralib.plot import plot_figure
from neuralib.plot.tools import AxesExtendHelper
from neuralib.util.verbose import fprint, publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.topology.util_plot import plot_topo_variable, plot_topo_histogram
from rscvp.visual.main_polar import BaseVisPolarOptions

__all__ = ['TopoMetricOptions']


@publish_annotation('main', project='rscvp', figure=['fig.2C', 'fig.4D left', 'fig.5K', 'fig.S2F'], as_doc=True)
class TopoMetricOptions(StatPipeline, BaseVisPolarOptions):
    DESCRIPTION = 'topological value average for the batch dataset'

    scaled: bool = argument('--scaled', help='If use scaled coordinates')

    plot_type: Literal['overview', 'hist1d'] = argument(
        '--plot',
        default='hist1d',
        help='plot type',
    )

    ibl_res: int = argument(
        '--res',
        default=10,
        choices=(10, 25, 50),
        help='resolution of IBL atlas dorsal map'
    )

    load_source: Final = 'parquet'

    ap_field: str
    ml_field: str

    restrict_rsc: bool = True
    """restrict the top dorsal cortex only in RSC"""

    def post_parsing(self):
        super().post_parsing()

        self.ap_field = 'ap_cords'
        self.ml_field = 'ml_cords'
        if self.scaled:
            self.ap_field += '_scale'
            self.ml_field += '_scale'

    def run(self):
        """overwrite"""
        self.post_parsing()
        self.load_table(to_pandas=False)
        self.plot()

    @property
    def gspread_field(self) -> str:
        """Handle case in ldl gspread with header(field): [HEADER]_[SESSION]"""
        if self.session is None:
            return self.header
        else:
            return f'{self.header}_{self.session}'

    def post_processing(self, remove_neg: bool = True):
        field = self.gspread_field

        if 'pf_peak' in field or 'pf_width' in field:
            ap, ml = self._place_field_handler()
            val = np.concatenate(self.df[field].to_numpy())
        else:
            ap = np.concatenate(self.df[self.ap_field].to_numpy())
            ml = np.concatenate(self.df[self.ml_field].to_numpy())
            val = np.concatenate(self.df[field].to_numpy())

        # masking case
        if self.header in ('pdir', 'pori'):
            ap, ml, val = self._selectivity_masking(ap, ml, val)

        #
        if remove_neg:
            n_outliers = np.count_nonzero(val < 0)
            if n_outliers != 0:
                fprint(f'{n_outliers} abnormal value < 0, clip negative', vtype='warning')
                val = np.clip(val, 0, None)

        cords = CellularCoordinates(np.arange(len(ap)), ap=ap, ml=ml)

        return cords, val

    def _place_field_handler(self) -> tuple[np.ndarray, np.ndarray]:
        """Handler for place field header (single cell might have multiple place fields)"""
        ap = self.df[self.ap_field].to_numpy()
        ml = self.df[self.ml_field].to_numpy()
        n = self.df['n_pf'].to_numpy()

        ap = np.concatenate([np.repeat(ap[i], n[i]) for i in range(len(n))])
        ml = np.concatenate([np.repeat(ml[i], n[i]) for i in range(len(n))])

        return ap, ml

    def _selectivity_masking(self, ap, ml, val, visual_only: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """direction/orientation selective neurons masking"""
        if visual_only:
            mx = np.concatenate(self.df['reliability'].to_numpy()) >= 0.3
            ap = ap[mx]
            ml = ml[mx]

        match self.header:
            case 'pdir':
                index = np.concatenate(self.df['dsi'].to_numpy())
            case 'pori':
                index = np.concatenate(self.df['osi'].to_numpy())
            case _:
                raise ValueError('')

        smx = index >= self.selective_thres
        ap = ap[smx]
        ml = ml[smx]
        val = val[smx]

        return ap, ml, val

    def plot(self):
        """
        Plot the value topography in regression (AP, ML), and soma location, and 2d histogram

        **Note to enhance color visualization, pass vmin/vmax in to :meth:`plot_topo_variable()`**
        """
        if self.plot_type == 'overview':
            self._plot_overview()
        elif self.plot_type == 'hist1d':
            self._plot_hist1d()

    def _plot_hist1d(self):
        cords, val = self.post_processing()

        bin_size = 50
        ml_bins = int(abs(np.max(cords.ml) - np.min(cords.ml)) // bin_size)
        ap_bins = int(abs(np.max(cords.ap) - np.min(cords.ap)) // bin_size)

        h_ml, ml_edges = np.histogram(cords.ml, bins=ml_bins, weights=val)
        h_ml_count, _ = np.histogram(cords.ml, bins=ml_bins)
        h_ml_mean = h_ml / h_ml_count
        h_ml_mean[np.isnan(h_ml_mean)] = 0
        ml_centers = (ml_edges[:-1] + ml_edges[1:]) / 2

        h_ap, ap_edges = np.histogram(cords.ap, bins=ap_bins, weights=val)
        h_ap_count, _ = np.histogram(cords.ap, bins=ap_bins)
        h_ap_mean = h_ap / h_ap_count
        h_ap_mean[np.isnan(h_ap_mean)] = 0
        ap_centers = (ap_edges[:-1] + ap_edges[1:]) / 2

        with plot_figure(None) as ax:
            plot_topo_histogram(ax, cords, val, self.gspread_field, ibl_res=self.ibl_res)

            if self.restrict_rsc:
                ax.set(xlim=(-1500, 50), ylim=(-4200, -500))

            helper = AxesExtendHelper(ax, y_position='left')
            helper.xbar(ml_centers, h_ml_mean, width=bin_size, color='gray', edgecolor='none', alpha=0.7)
            helper.ybar(ap_centers, h_ap_mean, height=bin_size, color='gray', edgecolor='none', alpha=0.7)

    def _plot_overview(self):
        cords, val = self.post_processing()

        with plot_figure(None, 2, 2, figsize=(6, 8), tight_layout=False) as ax:
            ax = ax.ravel()

            kwargs = {'ibl_res': self.ibl_res}
            cmap_range = (0, 97.5) if 'si' in self.header else None
            if cmap_range is not None:
                vmin, vmax = cmap_range
                kwargs.setdefault('vmin', np.percentile(val, vmin))
                kwargs.setdefault('vmax', np.percentile(val, vmax))

            plot_topo_variable(ax, cords, val, self.gspread_field,
                               scatter_size=5,
                               with_top_view=True,
                               zorder=-1, **kwargs)

            if self.restrict_rsc:
                ax[2].set(xlim=(-1500, 50), ylim=(-4200, -500))


if __name__ == '__main__':
    TopoMetricOptions().main()
