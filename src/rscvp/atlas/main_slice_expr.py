import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian

from argclz import AbstractParser, as_argument, argument, float_tuple_type
from neuralib.atlas.ccf import slice_transform_helper, load_transform_matrix, SLICE_DIMENSION_10um
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import CCFOptions, HistOptions


@publish_annotation('appendix', project='rscvp')
class SliceExprOptions(AbstractParser, CCFOptions):
    DESCRIPTION = 'Plot slice expression as contour'

    animal = as_argument(HistOptions.animal).with_options(required=False)

    gaussian_sigma: float = argument('--sigma', default=1, help='Gaussian smoothing sigma')
    percentile: tuple[float, float] = argument('--perc', type=float_tuple_type, default=(0, 99.5),
                                               help='percentile normalization of image')
    gamma: float = argument('--gamma', default=0.6, help='gamma value for suppressing oversaturation')

    level_range: tuple[float, float] = argument('--range', type=float_tuple_type, default=(0.1, 0.9),
                                                help='contour plot level')
    level_number: int = argument('--level', default=9, help='contour plot level')

    def run(self):
        image = iio.imread(self.raw_image)

        if image.ndim == 3:
            image = np.mean(image, axis=2)

        # constrast adjust
        image = gaussian(image, sigma=self.gaussian_sigma)
        lb, ub = np.percentile(image, self.percentile)
        image = np.clip((image - lb) / (ub - lb), 0, 1)
        image = image ** self.gamma

        # backgroud substraction
        h, w = image.shape
        center_crop = image[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50]
        center_median = np.median(center_crop)
        image = np.clip(image - center_median, 0, None)
        image = image / (image.max() + 1e-6)

        # apply transform and plot
        image, trans = slice_transform_helper(image, self.trans_matrix, plane_type=self.cut_plane)
        matrix = load_transform_matrix(self.ccf_matrix, self.cut_plane)
        plane = matrix.get_slice_plane()
        x, y = SLICE_DIMENSION_10um[self.cut_plane]
        extent = (-x / 2, x / 2, -y / 2, y / 2)

        fig, ax = plt.subplots()
        levels = np.linspace(*self.level_range, self.level_number)
        trans = np.flipud(trans)
        ax.contour(trans, levels=levels, extent=extent, cmap='viridis')
        plane.plot_boundaries(ax=ax, extent=extent, alpha=0.7)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    SliceExprOptions().main()
