from typing import Final

import numpy as np

from argclz import argument
from neuralib.imaging.registration import CellularCoordinates
from neuralib.plot import plot_figure
from neuralib.util.verbose import fprint, publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.topology.util_plot import plot_topo_variable, TopoPlotArgs
from rscvp.visual.main_polar import BaseVisPolarOptions

__all__ = ['TopoMetricOptions']


@publish_annotation('main', project='rscvp', figure=['fig.2C', 'fig.4D left', 'fig.5K', 'fig.S2F'], as_doc=True)
class TopoMetricOptions(StatPipeline, BaseVisPolarOptions):
    DESCRIPTION = 'topological value average for the batch dataset'

    scaled: bool = argument('--scaled', help='If use scaled coordinates')

    load_source: Final = 'parquet'

    ap_field: str
    ml_field: str

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

    def plot(self, restrict_rsc: bool = True,
             remove_neg: bool = True):
        """
        Plot the value topography in regression (AP, ML), and soma location, and 2d histogram

        **Note to enhance color visualization, pass vmin/vmax in to :meth:`plot_topo_variable()`**

        :param restrict_rsc: restrict the top dorsal cortex only in RSC
        :param remove_neg: remove negative value
        """

        field = self.gspread_field

        if 'pf_peak' in field or 'pf_width' in field:
            ap, ml = self._place_field_handler()
            val = np.concatenate(self.df[field].to_numpy())
            args = TopoPlotArgs(field, 'cyclic', 2, 'twilight')
        else:
            ap = np.concatenate(self.df[self.ap_field].to_numpy())
            ml = np.concatenate(self.df[self.ml_field].to_numpy())
            val = np.concatenate(self.df[field].to_numpy())
            args = TopoPlotArgs.infer(val, field)

        # masking case
        if self.header in ('pdir', 'pori'):
            ap, ml, val = self._selectivity_masking(ap, ml, val)

        #
        if remove_neg:
            n_outliers = np.count_nonzero(val < 0)
            if n_outliers != 0:
                fprint(f'{n_outliers} abnormal value < 0, clip negative', vtype='warning')
                val = np.clip(val, 0, None)
                args = args._replace(dtype='uniform', cmap='magma')

        #
        cords = CellularCoordinates(np.arange(len(ap)), ap=ap, ml=ml)  # as container
        with plot_figure(None, 2, 2, figsize=(6, 8), tight_layout=False) as ax:
            ax = ax.ravel()

            kwargs = {}
            cmap_range = self._default_camp_range()
            if cmap_range is not None:
                vmin, vmax = cmap_range
                kwargs.setdefault('vmin', np.percentile(val, vmin))
                kwargs.setdefault('vmax', np.percentile(val, vmax))

            plot_topo_variable(ax, cords, args, val,
                               scatter_size=5,
                               with_top_view=True,
                               zorder=-1, **kwargs)

            if restrict_rsc:
                ax[2].set(xlim=(-1500, 50), ylim=(-4200, -500))

    def _default_camp_range(self) -> tuple[float, float] | None:
        if 'si' in self.header:
            return 0, 97.5
        else:
            return None

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


if __name__ == '__main__':
    TopoMetricOptions().main()
