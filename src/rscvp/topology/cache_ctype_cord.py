from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from rscvp.topology.util import RSCObjectiveFOV
from rscvp.topology.util_plot import plot_registered_fov
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_sbx import SBXOptions
from rscvp.util.cli.cli_selection import SelectionOptions, SelectionMask

from argclz import AbstractParser, as_argument, argument
from neuralib.imaging.registration import CellularCoordinates
from neuralib.imaging.suite2p import Suite2PResult
from neuralib.persistence import persistence
from neuralib.plot import plot_figure
from neuralib.plot.tools import AxesExtendHelper

__all__ = [
    'CellTypeCordCache',
    'CellTypeCordCacheBuilder'
]


@persistence.persistence_class
class CellTypeCordCache:
    exp_date: str = persistence.field(validator=True, filename=True)
    """Experimental date"""
    animal: str = persistence.field(validator=True, filename=True)
    """Animal ID"""
    plane_index: str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    """Optical imaging plane index"""

    neuron_idx: np.ndarray
    """Neuron index. `Array[int, N]`"""

    ml: np.ndarray
    """Medial-lateral coordinates (in um, relative to bregma. incremental from M to L). `Array[float, N]`"""
    ap: np.ndarray
    """Anterior-posterior coordinates (in um, relative to bregma. incremental from A to P). `Array[float, N]`"""

    ap_hist_visual: np.ndarray
    """AP counts for visual cells. `Array[float, B]`"""
    ap_bins_visual: np.ndarray
    """AP bins coordinates for visual cells. `Array[float, B]`"""
    ml_hist_visual: np.ndarray
    """ML counts for visual cells. `Array[float, B]`"""
    ml_bins_visual: np.ndarray
    """ML bins coordinates for visual cells. `Array[float, B]`"""

    ap_hist_place: np.ndarray
    """AP counts for place cells. `Array[float, B]`"""
    ap_bins_place: np.ndarray
    """AP bins coordinates for place cells. `Array[float, B]`"""
    ml_hist_place: np.ndarray
    """ML counts for place cells. `Array[float, B]`"""
    ml_bins_place: np.ndarray
    """ML bins coordinates for place cells. `Array[float, B]`"""


class CellTypeCordCacheBuilder(AbstractParser, SelectionOptions, SBXOptions, PersistenceRSPOptions[CellTypeCordCache]):
    DESCRIPTION = """
    Cache for collection of coordinates, cell type information. 
    AFTER PRESELECTION, and plot each cell type fraction histogram in ap/ml space
    """

    nbins_xy: int = argument('--nbins', default=20, help='number of bins in xy of FOV')

    plane_index: int = as_argument(SelectionOptions.plane_index).with_options(
        required=True,
        help='topographical plot need to implement in a certain optic plane'
    )

    #
    vc_selection = 0.3
    used_session = 'light'
    reuse_output = True

    #
    s2p: Suite2PResult
    select_mask: SelectionMask
    cords: CellularCoordinates

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        self.s2p = self.load_suite_2p()
        self.select_mask = self.get_selection_mask(with_preselection=True)
        cache = self.load_cache()
        output_file = self.get_data_output('topo').summary_figure_output('fraction')
        self.plot_cell_type_histogram(cache, output_file)

    # ==== #
    # Plot #
    # ==== #

    def plot_cell_type_histogram(self, cache: CellTypeCordCache, output: Path):
        with plot_figure(output, 1, 2) as ax:
            _helper = AxesExtendHelper(ax[0])
            plot_topo_cell_type(
                _helper.ax,
                self.s2p,
                self.exp_date,
                self.animal_id,
                self.p2d_factor,
                'spatial',
                self.pre_select(),
                self.select_mask.place_mask
            )
            _helper.xbar(cache.ml_bins_place[:-1], cache.ml_hist_place)
            _helper.ybar(cache.ap_bins_place[:-1], cache.ap_hist_place)

            _helper = AxesExtendHelper(ax[1])
            plot_topo_cell_type(
                _helper.ax,
                self.s2p,
                self.exp_date,
                self.animal_id,
                self.p2d_factor,
                'visual',
                self.pre_select(),
                self.select_mask.visual_mask
            )

            _helper.xbar(cache.ml_bins_visual[:-1], cache.ml_hist_visual)
            _helper.ybar(cache.ap_bins_visual[:-1], cache.ap_hist_visual)

    # =========== #
    # Build Cache #
    # =========== #

    def empty_cache(self) -> CellTypeCordCache:
        return CellTypeCordCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            plane_index=self.plane_index
        )

    def compute_cache(self, cache: CellTypeCordCache) -> CellTypeCordCache:
        cache.neuron_idx = np.nonzero(self.select_mask.pre_select_mask)[0]
        cache.ml = self.all_ml
        cache.ap = self.all_ap

        cache.ap_hist_place, cache.ap_bins_place = self.calculate_hist_fraction(self.place_ap, self.all_ap)
        cache.ml_hist_place, cache.ml_bins_place = self.calculate_hist_fraction(self.place_ml, self.all_ml)
        cache.ap_hist_visual, cache.ap_bins_visual = self.calculate_hist_fraction(self.visual_ap, self.all_ap)
        cache.ml_hist_visual, cache.ml_bins_visual = self.calculate_hist_fraction(self.visual_ml, self.all_ml)

        return cache

    # ====== #
    @cached_property
    def all_ap(self) -> np.ndarray:
        return self.get_csv_data('ap_cords')

    @cached_property
    def all_ml(self) -> np.ndarray:
        return self.get_csv_data('ml_cords')

    @cached_property
    def place_ap(self) -> np.ndarray:
        return self.all_ap[self.select_mask.place_mask]

    @cached_property
    def place_ml(self) -> np.ndarray:
        return self.all_ml[self.select_mask.place_mask]

    @cached_property
    def visual_ap(self) -> np.ndarray:
        return self.all_ap[self.select_mask.visual_mask]

    @cached_property
    def visual_ml(self) -> np.ndarray:
        return self.all_ml[self.select_mask.visual_mask]

    @cached_property
    def p2d_factor(self) -> float:
        return self.pixel2distance_factor(self.s2p)

    def calculate_hist_fraction(self, x: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO unify if need to run statistically in batch dataset.
        bin_range = np.linspace(np.min(x_all), np.max(x_all), num=self.nbins_xy)

        counts_cell_type = np.histogram(x, bins=bin_range)[0]
        counts_all = np.histogram(x_all, bins=bin_range)[0]

        frac = counts_cell_type / counts_all

        return frac, bin_range


def plot_topo_cell_type(ax: Axes,
                        s2p: Suite2PResult,
                        exp_date: str,
                        animal_id: str,
                        p2d_factor: float,
                        classification_type: Literal['visual', 'spatial'],
                        all_mask: np.ndarray,
                        cell_type_mask: np.ndarray,
                        set_square: bool = False):
    """
    plot the topographical distribution of cell types in s2p cell's segmentations

    :param ax: ``Axes``
    :param s2p ``Suite2PResult``
    :param exp_date: Experimental date
    :param animal_id: Animal ID
    :param p2d_factor: pixel to distance factor
    :param classification_type: cell type classification
    :param all_mask: all cell mask (i.e., after preselection)
    :param cell_type_mask: cell type classification mask
    :param set_square
    """
    fov_bregma = RSCObjectiveFOV.load_from_gspread(exp_date, animal_id).to_um()

    plot_registered_fov(ax, s2p, all_mask,
                        uni_color='Greys',
                        axis_type='bregma_um',
                        p2d_factor=p2d_factor,
                        fov_bregma=fov_bregma,
                        alpha=0.3,
                        interpolation='none')

    c = 'winter' if classification_type == 'spatial' else 'cool'
    plot_registered_fov(ax, s2p, cell_type_mask,
                        uni_color=c,
                        axis_type='bregma_um',
                        p2d_factor=p2d_factor,
                        fov_bregma=fov_bregma,
                        alpha=0.6, interpolation='none')

    ax.set(xlabel='ML distance(mm)', ylabel='AP distance(mm)')

    if set_square:
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


if __name__ == '__main__':
    CellTypeCordCacheBuilder().main()
