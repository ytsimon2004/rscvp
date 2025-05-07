from pathlib import Path
from typing import get_args

import cv2
import numpy as np
from scipy.io import loadmat

from neuralib.atlas.ccf.matrix import load_transform_matrix
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.imglib.transform import apply_transformation
from neuralib.plot import plot_figure
from neuralib.typing import PathLike
from neuralib.util.unstable import unstable

# {P: (W, H)}
DEFAULT_DIMENSION: dict[PLANE_TYPE, tuple[int, int]] = {
    'coronal': (1140, 800),
    'sagittal': (1320, 800)
}


@unstable()
def plot_slice_view_transform(raw_image: PathLike,
                              trans_matrix: np.ndarray | PathLike,
                              annotation: PathLike,
                              plane_type: PLANE_TYPE,
                              *,
                              resize_dim: tuple[int, int] | None = None,
                              flip_lr_raw: bool = True,
                              flip_ud_raw: bool = False,
                              output: PathLike | None = None):
    """

    :param raw_image: raw brain slice
    :param trans_matrix: .mat file transformation array, note that `t` while using 2dccf matlab `save` func
                (due to `projective2d` seems runtime call,
                thus not save in transformation.mat, need to save separately)
    :param plane_type: slice cutting orientation
    :param resize_dim: image down-sampling (W, H). Predefine by allen mouse atlas in default settings
    :param annotation: provide the .mat file after the 2dccf registration
    :param flip_lr_raw: horizontally flip the raw image. i.e, matching the transformation matrix
    :param flip_ud_raw: vertically flip the raw image. i.e, matching the transformation matrix
    :param output: image output
    :return:
    """

    raw_image = cv2.cvtColor(cv2.imread(raw_image), cv2.COLOR_BGR2RGB)

    #
    if flip_ud_raw:
        raw_image = np.flipud(raw_image)

    if flip_lr_raw:
        raw_image = np.fliplr(raw_image)

    #
    if resize_dim is None:
        resize = cv2.resize(raw_image, DEFAULT_DIMENSION[plane_type])
    else:
        resize = cv2.resize(raw_image, resize_dim)

    #
    if isinstance(trans_matrix, get_args(PathLike)):
        s = Path(trans_matrix).suffix
        if s == '.mat':
            trans_matrix = loadmat(trans_matrix)['t'].T
        elif s == '.npy':
            trans_matrix = np.load(trans_matrix)
        else:
            raise ValueError(f'invalid file type for the transformation matrix: {s}')

    trans = apply_transformation(resize, trans_matrix)

    with plot_figure(output, 1, 2) as ax:
        x, y = DEFAULT_DIMENSION[plane_type]
        extent = (-x / 2, x / 2, -y / 2, y / 2)
        ax[0].imshow(raw_image)
        ax[1].imshow(trans, extent=extent)

        #
        matrix = load_transform_matrix(annotation, plane_type)
        matrix.get_slice_plane()._plot_boundaries(ax=ax[1], extent=extent, cmap='binary_r', alpha=1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-R', '--raw', metavar='FILE', type=Path, help='raw image', dest='raw')
    ap.add_argument('-T', '--transform', metavar='FILE', type=Path, help='transformation matrix array', dest='trans')
    ap.add_argument('-M', metavar='FILE', type=Path, help='transformation matrix', dest='mat')
    ap.add_argument('-P', '--plane', choices=get_args(PLANE_TYPE), help='plane type', dest='plane')
    ap.add_argument('-O', '--output', metavar='FILE', type=Path, help='figure output path', dest='output')

    opt = ap.parse_args()

    plot_slice_view_transform(raw_image=opt.raw,
                              trans_matrix=opt.trans,
                              annotation=opt.mat,
                              plane_type=opt.plane,
                              output=opt.output)


if __name__ == '__main__':
    main()
