import abc
from typing import Literal, Iterable

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from argclz import argument, AbstractParser, as_argument
from neuralib.imaging.registration import CellularCoordinates
from neuralib.imaging.suite2p import Suite2PResult, get_s2p_coords
from neuralib.imaging.suite2p.plot import get_soma_pixel
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar, insert_cyclic_colorbar
from neuralib.plot.plot import scatter_binx_plot
from neuralib.typing import AxesArray, ArrayLike
from rscvp.util.cli import PlotOptions, SBXOptions, SelectionOptions
from rscvp.util.util_ibl import IBLAtlasPlotWrapper
from .util import RSCObjectiveFOV

__all__ = ['Metric',
           'AbstractTopoPlotOptions',
           'plot_topo_variable',
           'plot_topo_histogram',
           'plot_registered_fov']

Metric = str
"""metric for see the topographical difference. i.e., si, tcc, vc, pdire, sftf_0.04_1., etc"""


class AbstractTopoPlotOptions(AbstractParser, SelectionOptions, SBXOptions, PlotOptions, metaclass=abc.ABCMeta):
    metric: Metric | None = argument(
        '--metric', '-M',
        default=None,
        help='which metric will be used. if None, do iter run'
    )

    plane_index: int = as_argument(SelectionOptions.plane_index).with_options(
        required=True,
        help='topographical plot need to implement in a certain optic plane'
    )

    #
    pre_selection = True
    reuse_output = True

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    def run(self):
        s2p = self.load_suite_2p()
        fov = RSCObjectiveFOV.load_from_gspread(self.exp_date, self.animal_id, page=self.gspread_page)
        cords = (
            get_s2p_coords(s2p, self.get_all_neurons(), self.plane_index, self.pixel2distance_factor(s2p))
            .relative_origin(fov)
        )

        mask = self.get_selected_neurons()
        output = self.get_data_output('topo')

        iter_metric = [self.metric] if self.metric is not None else self.foreach_metric
        for met in iter_metric:
            value = self._get_value(met)

            output_file = output.summary_figure_output(met)
            with plot_figure(output_file, 2, 2) as _ax:
                ax = _ax.ravel()
                plot_topo_variable(ax, cords, value, self.metric, mask=mask, with_top_view=True)

    @property
    @abc.abstractmethod
    def foreach_metric(self) -> Iterable[Metric]:
        """multiple metric for iteration"""
        pass

    @abc.abstractmethod
    def _get_value(self, metric: Metric) -> np.ndarray:
        """csv query for all cell (WITHOUT selection)"""
        pass


def plot_topo_variable(ax: AxesArray,
                       cords: CellularCoordinates,
                       value: np.ndarray,
                       name: str,
                       mask: np.ndarray | None = None,
                       log2_value: bool = False,
                       with_top_view: bool = True,
                       scatter_size: float = 2.5,
                       scatter_alpha: float = 0.8,
                       order: int = 1,
                       **kwargs):
    """
    1d domain to see linear (AP, ML for panel 1, 2) and 2d see the whole fov in space

    :param ax: AxesArray (4,)
    :param cords: ``CellularCoordinates``
    :param value: metric value domain
    :param name: metric name
    :param mask: cell selection mask
    :param log2_value: show value and color bar as log2
    :param with_top_view
    :param scatter_size:
    :param scatter_alpha:
    :param order:
    :param kwargs: to `plot_topo_scatter` and `plot_topo_histogram`
    """
    if mask is not None:
        value = value[mask]
        cords = cords.with_masking(mask)

    scatter_binx_plot(ax[0], cords.ap, value, xlabel='AP distance(um)', order=order, ylabel=name)
    scatter_binx_plot(ax[1], cords.ml, value, xlabel='ML distance(um)', order=order, ylabel=name)
    plot_topo_scatter(ax[2], cords, value, name,
                      log2_value=log2_value,
                      with_top_view=with_top_view,
                      scatter_size=scatter_size,
                      scatter_alpha=scatter_alpha,
                      **kwargs)
    plot_topo_histogram(ax[3], cords, value, name, log2_value=log2_value, with_top_view=with_top_view, **kwargs)
    ax[2].sharex(ax[3])
    ax[2].sharey(ax[3])


def plot_topo_scatter(ax: Axes,
                      cords: CellularCoordinates,
                      value: np.ndarray,
                      name: str,
                      *,
                      log2_value: bool = True,
                      cbar_range: tuple[float, float] | None = None,
                      with_top_view: bool = True,
                      scatter_size: float = 2.5,
                      scatter_alpha: float = 0.8,
                      **kwargs):
    """Plot the metric value(color) in 2d space (AP, ML) with scatter plot"""
    norm = mcolors.SymLogNorm(linthresh=0.001, base=2) if log2_value else None
    im = ax.scatter(cords.ml, cords.ap,
                    s=scatter_size,
                    alpha=scatter_alpha,
                    edgecolors='none',
                    c=value,
                    cmap='magma',
                    norm=norm,
                    **kwargs)

    # colorbar
    if name == 'pdir':
        insert_cyclic_colorbar(ax, im, vmin=0, vmax=360)
    else:
        cbar = insert_colorbar(ax, im)
        cbar.ax.set(ylabel=name)

        if cbar_range is not None:
            cbar.ax.set(ylim=cbar_range)
            im.set_clim(vmin=cbar_range[0], vmax=cbar_range[1])

    #
    if with_top_view:
        ibl = IBLAtlasPlotWrapper()
        ibl.plot_scalar_on_slice(['root'], ax=ax, coord=-2000, plane='top', background='boundary')

    ax.set_aspect('equal', adjustable='box')
    ax.set(xlabel='ML distance(um)', ylabel='AP distance(um)')


def plot_topo_histogram(ax: Axes,
                        coords: CellularCoordinates,
                        value: np.ndarray,
                        name: str,
                        *,
                        log2_value: bool = False,
                        with_top_view: bool = True,
                        bins: int | ArrayLike | None = None,
                        bin_size: int = 50,
                        **kwargs):
    """
    Plot the metric value(color) in 2d space (AP, ML) with 2d histogram plot

    :param ax:
    :param coords:
    :param value:
    :param name:
    :param log2_value
    :param with_top_view:
    :param bins: number of bins,
            if int, scalar with xy
            Or ArrayLike [x, y],
            Or None, then inferred by `bin size` args
    :param bin_size:  bins size in micron (um)
    :param kwargs:
    :return:
    """
    if bins is None:
        xb = int(abs(np.max(coords.ml) - np.min(coords.ml)) // bin_size)
        yb = int(abs(np.max(coords.ap) - np.min(coords.ap)) // bin_size)
        bins = (yb, xb)

    H, yedges, xedges = np.histogram2d(coords.ap, coords.ml, bins=bins, weights=value)
    H_count, _, _ = np.histogram2d(coords.ap, coords.ml, bins=bins)

    H_mean = H / H_count
    H_mean[np.isnan(H_mean)] = np.nan

    norm = mcolors.SymLogNorm(linthresh=0.001, base=2) if log2_value else None

    im = ax.pcolormesh(xedges, yedges, H_mean, cmap='magma', norm=norm, **kwargs)
    vmin, vmax = im.get_clim()

    #
    if name == 'pdir':
        insert_cyclic_colorbar(ax, im, vmin=0, vmax=360)
    else:
        cbar = insert_colorbar(ax, im)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set(ylabel=name)

    #
    if with_top_view:
        IBLAtlasPlotWrapper().plot_scalar_on_slice(
            ['root'],
            ax=ax,
            coord=-2000,
            plane='top',
            background='boundary'
        )

    ax.set_aspect('equal', adjustable='box')
    ax.set(xlabel='ML distance(um)', ylabel='AP distance(um)')


def plot_registered_fov(ax: Axes,
                        s2p: Suite2PResult,
                        cell_mask: np.ndarray,
                        *,
                        value: np.ndarray | None = None,
                        val_cmap: str = 'hsv',
                        uni_color: str = 'Greys',
                        axis_type: Literal['pixel', 'um', 'bregma_um'] = 'pixel',
                        p2d_factor: float | None = None,
                        fov_bregma: RSCObjectiveFOV | None = None,
                        **kwargs) -> AxesImage:
    """
    draw the color in cell shape (yx pixel).
    the color could be based on value (use val_cmap).
    or uniform in a certain neuronal population (use uni_color)

    :param ax
    :param s2p:
    :param cell_mask: bool array of the selected cells
    :param value: if needs to create own colormap based on the value range
    :param val_cmap: cmap based on the value type.
        i.e., Cyclic(hsv..) for direction
              monotonically(viridis) for DSI
    :param uni_color: if not None, use this cmap (maximal color) for all the non-nan pixel 2D array
                check the https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :param axis_type: xy axis unit
    :param p2d_factor: factor of pixel to distance, specify if axis are shown in `mm`
    :param fov_bregma: specify if axis are shown in ('bregma_mm', 'mm')

    :return:
        pix: AxesImage
    """

    neuron_pix = get_soma_pixel(s2p, cell_mask, color_diff=True)
    neuron_pix[neuron_pix == 0] = np.nan  # not draw

    #
    y, x = neuron_pix.shape
    if axis_type in ('um', 'bregma_um'):
        y *= p2d_factor / 1000
        x *= p2d_factor / 1000
        if axis_type == 'mm':
            extent = (0, x, y, 0)
        else:
            left = 0 + fov_bregma.pl[0]
            right = x + fov_bregma.pm[0]
            bottom = y + fov_bregma.pl[1]
            top = 0 + fov_bregma.al[1]
            extent = (left, right, bottom, top)
    else:
        extent = (0, x, y, 0)  # pixel

    #
    if value is not None:
        if value.ndim != 1:
            raise ValueError(f'value array should be single dimension, got {value.ndim} instead!')

        if len(value) != np.count_nonzero(cell_mask):
            raise ValueError('value numbers not equal to cell numbers')

        for i, neuron_id, in enumerate(np.nonzero(cell_mask)[0]):
            neuron_pix[neuron_pix == neuron_id + 1] = value[i]

        pix = ax.imshow(neuron_pix, cmap=val_cmap, alpha=0.8,
                        vmin=np.min(value), vmax=np.max(value), zorder=1, extent=extent, **kwargs)

    else:
        for i, neuron_id, in enumerate(np.nonzero(cell_mask)[0]):
            neuron_pix[neuron_pix == neuron_id + 1] = 1
        pix = ax.imshow(neuron_pix, cmap=uni_color, vmin=0, vmax=1, extent=extent, **kwargs)

    return pix
