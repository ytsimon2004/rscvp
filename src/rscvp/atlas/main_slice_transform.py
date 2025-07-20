from argclz import AbstractParser, argument, as_argument
from neuralib.atlas.ccf.matrix import load_transform_matrix, SLICE_DIMENSION_10um, slice_transform_helper
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import HistOptions, CCFOptions

__all__ = ['SliceTransformOptions']


@publish_annotation('appendix', project='rscvp', as_doc=True, caption='check slice transform and expression')
class SliceTransformOptions(AbstractParser, CCFOptions):
    DESCRIPTION = 'Plot the 1) orig image 2) overlay transformed image + boundary 3) annotation brain region'

    animal = as_argument(HistOptions.animal).with_options(required=False)

    overlay_only: bool = argument('--overlay', help='only show image')

    def run(self):
        raw, trans = slice_transform_helper(self.raw_image, self.trans_matrix, plane_type=self.cut_plane)
        x, y = SLICE_DIMENSION_10um[self.cut_plane]
        extent = (-x / 2, x / 2, -y / 2, y / 2)

        if self.overlay_only:
            with plot_figure(None) as ax:
                ax.imshow(trans, extent=extent)

                matrix = load_transform_matrix(self.ccf_matrix, self.cut_plane)
                plane = matrix.get_slice_plane()
                plane.plot_boundaries(ax=ax, extent=extent, cmap='binary_r', alpha=0.7)

        else:
            with plot_figure(None, 1, 3) as ax:
                ax[0].imshow(raw)
                ax[1].imshow(trans, extent=extent)

                matrix = load_transform_matrix(self.ccf_matrix, self.cut_plane)
                plane = matrix.get_slice_plane()
                plane.plot_boundaries(ax=ax[1], extent=extent, cmap='binary_r', alpha=0.7)
                plane.plot(ax=ax[2], annotation_region=list(self.annotation_region), extent=extent, boundaries=True)


if __name__ == '__main__':
    SliceTransformOptions().main()
