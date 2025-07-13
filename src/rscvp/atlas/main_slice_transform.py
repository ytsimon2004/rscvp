from pathlib import Path
from typing import get_args

import cv2
import numpy as np
from scipy.io import loadmat

from argclz import AbstractParser, argument, as_argument, str_tuple_type
from neuralib.atlas.ccf.matrix import load_transform_matrix
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.imglib.transform import apply_transformation
from neuralib.plot import plot_figure
from neuralib.typing import PathLike
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import HistOptions

__all__ = ['SliceTransformOptions', 'slice_transform_helper']

DEFAULT_DIMENSION: dict[PLANE_TYPE, tuple[int, int]] = {
    'coronal': (1140, 800),
    'sagittal': (1320, 800)
}
"""{P: (W, H)} for 10 um mouse atlas"""


@publish_annotation('appendix', project='rscvp', as_doc=True, caption='check slice transform and expression')
class SliceTransformOptions(AbstractParser, HistOptions):
    DESCRIPTION = 'Plot the 1) orig image 2) overlay transformed image + boundary 3) annotation brain region'

    animal = as_argument(HistOptions.animal).with_options(required=False)

    raw_image: Path = argument('-R', '--raw', help='raw image path')
    trans_matrix: Path = argument('-T', '--trans', help='transform matrix path (3 x 3)')
    ccf_matrix: Path = argument('--ccf', help='ccf matrix .mat file')

    annotation_region: tuple[str, ...] = argument(
        '--annotation',
        type=str_tuple_type,
        default=('LD',),
        help='annotation brain region'
    )
    overlay_only: bool = argument('--overlay', help='only show image')

    def run(self):
        raw, trans = slice_transform_helper(self.raw_image, self.trans_matrix, plane_type=self.cut_plane)
        x, y = DEFAULT_DIMENSION[self.cut_plane]
        extent = (-x / 2, x / 2, -y / 2, y / 2)

        if self.overlay_only:
            with plot_figure(None) as ax:
                ax.imshow(trans, extent=extent)

                matrix = load_transform_matrix(self.ccf_matrix, self.cut_plane)
                plane = matrix.get_slice_plane()
                plane._plot_boundaries(ax=ax, extent=extent, cmap='binary_r', alpha=0.7)

        else:
            with plot_figure(None, 1, 3) as ax:
                ax[0].imshow(raw)
                ax[1].imshow(trans, extent=extent)

                matrix = load_transform_matrix(self.ccf_matrix, self.cut_plane)
                plane = matrix.get_slice_plane()
                plane._plot_boundaries(ax=ax[1], extent=extent, cmap='binary_r', alpha=0.7)
                plane.plot(ax=ax[2], annotation_region=list(self.annotation_region), extent=extent, boundaries=True)


def slice_transform_helper(raw_image: PathLike,
                           trans_matrix: np.ndarray | PathLike, *,
                           plane_type: PLANE_TYPE = 'coronal',
                           flip_lr: bool = False,
                           flip_ud: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms an input image according to the specified transformation matrix,
    plane orientation, and optional flipping parameters. This function reads a raw
    image, optionally flips it horizontally or vertically, applies a transformation
    matrix, and resizes it to a defined dimension based on the plane type. The
    result is a tuple containing the raw image and the transformed image.

    :param raw_image: Path to the raw input image. The image is loaded and
        processed to apply transformations. Must be a valid path-like object.
    :param trans_matrix: Transformation matrix to apply to the image. It can be a
        numpy array or the path to a valid file containing the matrix. Supported
        files are `.mat` for MATLAB files and `.npy` for NumPy files.
    :param plane_type: Defines the anatomical plane for transformation.
        Defaults to 'coronal'.
    :param flip_lr: Boolean flag to determine whether to flip the image
        horizontally (left to right).
    :param flip_ud: Boolean flag to determine whether to flip the image
        vertically (up to down).
    :return: A tuple containing two numpy arrays: the raw image as processed,
        and the transformed image.
    """
    raw_image = cv2.cvtColor(cv2.imread(raw_image), cv2.COLOR_BGR2RGB)

    if flip_ud:
        raw_image = np.flipud(raw_image)

    if flip_lr:
        raw_image = np.fliplr(raw_image)

    if isinstance(trans_matrix, get_args(PathLike)):
        s = Path(trans_matrix).suffix
        if s == '.mat':
            mtx = loadmat(trans_matrix)['t'].T
        elif s == '.npy':
            mtx = np.load(trans_matrix)
        else:
            raise ValueError(f'invalid file type for the transformation matrix: {s}')
    else:
        mtx = trans_matrix

    resize = cv2.resize(raw_image, DEFAULT_DIMENSION[plane_type])
    transform_image = apply_transformation(resize, mtx)

    return raw_image, transform_image


if __name__ == '__main__':
    SliceTransformOptions().main()
