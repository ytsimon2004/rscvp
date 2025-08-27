from __future__ import annotations

from typing import Optional, Iterable, final

import attrs
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.atlas.ccf.matrix import CCFTransMatrix, load_transform_matrix
from neuralib.imglib.array import ImageArrayWrapper
from neuralib.plot import plot_figure
from neuralib.util.utils import uglob
from neuralib.util.verbose import publish_annotation
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli import HistOptions

__all__ = ['RoiAtlasOptions']


@publish_annotation('main', project='rscvp', figure='fig.6E', as_doc=True)
class RoiAtlasOptions(AbstractParser, HistOptions):
    """Plot the local maxima roi selection and atlas annotation based on transformed matrix & images"""

    DESCRIPTION = 'Plot the local maxima roi selection and atlas annotation based on transformed matrix & images'

    affine_transform: bool = argument(
        '--affine',
        help='whether do affine transformation for the roi atlas'
    )

    def run(self):
        """runs the main execution logic of the object"""
        ccf_dir = self.get_ccf_dir()
        n_images = len(list(ccf_dir.transformed_folder.glob('*.mat')))
        out = ccf_dir.roi_atlas_output

        foreach_atlas = tqdm(self.foreach_slice(ccf_dir), total=n_images, unit='images', desc='plot roi atlas')
        for ratlas in foreach_atlas:
            output_file = None if self.debug_mode else out / f'{ratlas.name}.pdf'
            with plot_figure(output_file, figsize=(8, 8)) as ax:
                ratlas.plot(ax, affine_transform=self.affine_transform)

    def foreach_slice(self, ccf_dir: AbstractCCFDir, overlap: bool = True) -> Iterable[RoiAtlas]:
        """
        Iterates through the slices in a transformed directory, applying a series
        of operations to each identified image transformation file. Each operation
        is responsible for loading transformation matrices, processing image data,
        and constructing an object representing the region of interest (ROI) atlas.
        When the overlap flag is enabled, an additional overlapping image is
        processed and included in the ROI atlas.

        :param ccf_dir: An object representing the transformed directory structure: :class:`~rscvp.atlas.dir.AbstractCCFDir`
        :param overlap: A boolean indicating whether to process overlap data for the images. Defaults to True.
        :return: An iterable of :class:`RoiAtlas` objects, each representing the
            processed atlas for a region of interest identified in an image.
        """
        directory = ccf_dir.transformed_folder
        mats = sorted(list(directory.glob('*mat')))

        o = None
        for f in mats:
            fidx = f.name.index('_resize')
            name = f.name[:fidx]
            im = uglob(f.parent, f'{name}*.tif')
            mat = load_transform_matrix(f, self.cut_plane)
            img = ImageArrayWrapper(im)
            r = img.local_maxima('red')
            g = img.local_maxima('green')

            if overlap:
                overlap_file = uglob(ccf_dir.transformed_folder_overlap, f'{name}*.tif')
                img_overlap = ImageArrayWrapper(overlap_file)
                o = img_overlap.local_maxima('red')

            yield RoiAtlas(name, mat, r, g, o)


@final
@attrs.define
class RoiAtlas:
    name: str
    """filename of the slice"""
    transform: CCFTransMatrix
    """CCFTransMatrix"""

    red_channel: np.ndarray
    """Red channel local maxima image"""
    green_channel: np.ndarray
    """Green channel local maxima image"""
    overlap_channel: np.ndarray | None = attrs.field(default=None)
    """Overlap channel (pseudo-red) local maxima image"""

    red_coordinates: np.ndarray = attrs.field(init=False)
    """Red channel coordinates. `Array[float, [2, R]]`"""
    green_coordinates: np.ndarray = attrs.field(init=False)
    """Green channel coordinates. `Array[float, [2, R]]`"""
    overlap_coordinates: Optional[np.ndarray] = attrs.field(default=None)
    """Overlap channel coordinates. `Array[float, [2, R]]`"""

    def __attrs_post_init__(self):
        def img_to_coord(img: np.ndarray, to_actual: bool = True):
            x, y = np.nonzero(img)
            if to_actual:
                y = y * 10 - (self.transform.get_slice_plane().slice_view.width_mm * 1000 / 2)
                x *= 10
            return np.vstack([x, y])

        self.red_coordinates = img_to_coord(self.red_channel)
        self.green_coordinates = img_to_coord(self.green_channel)
        if self.overlap_channel is not None:
            self.overlap_coordinates = img_to_coord(self.overlap_channel)

    @property
    def title(self) -> str:
        v = [self.name,
             f'{self.transform.get_slice_plane().reference_value}mm from Bregma',
             f'index: {self.transform.slice_index}',
             f'dw: {self.transform.delta_xy[0]}',
             f'dh: {self.transform.delta_xy[1]}']
        return '\n'.join(v)

    def plot(self, ax: Axes, *,
             s: float = 20,
             a: float = 0.8,
             with_overlap: bool = True,
             set_axis_visible: bool = True,
             boundaries: bool = True,
             affine_transform: bool = False) -> None:

        if affine_transform:
            import matplotlib.transforms as mtransforms
            aff = mtransforms.Affine2D().skew_deg(-20, 0)
            t = aff + ax.transData
        else:
            t = ax.transData

        plane = self.transform.get_slice_plane()
        plane.plot(ax, to_um=True, boundaries=boundaries, reference_bg_value=10, transform=t)

        self._plot_rois_scatter(ax, self.red_coordinates, 'violet', s, a, t)
        self._plot_rois_scatter(ax, self.green_coordinates, 'palegreen', s, a, t)

        if with_overlap and self.overlap_channel is not None:
            self._plot_rois_scatter(ax, self.overlap_coordinates, 'gold', s, a, t)

        ax.set_title(self.title)
        ax.axes.yaxis.set_visible(set_axis_visible)
        ax.axes.xaxis.set_visible(set_axis_visible)

    @staticmethod
    def _plot_rois_scatter(ax, xy, c, s, a, t):
        kwargs = dict(color=c, s=s, alpha=a, edgecolor='black', linewidth=0.4, transform=t)
        ax.scatter(xy[1, :], xy[0, :], **kwargs)


if __name__ == '__main__':
    RoiAtlasOptions().main()
