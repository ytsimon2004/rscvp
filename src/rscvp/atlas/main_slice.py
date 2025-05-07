from typing import Final

from argclz import AbstractParser
from neuralib.atlas.view import get_slice_view
from neuralib.plot import plot_figure


class SliceViewOptions(AbstractParser):
    DESCRIPTION = 'Plot the slice view of the injection site'

    anterior_index: Final = 690
    posterior_index: Final = 863
    resolution: Final = 10
    coords: Final = (500, 800)  # ml, dv in um

    def run(self):
        with plot_figure(None, 1, 2, sharex=True, sharey=True) as ax:
            kwargs = dict(boundaries=True,
                          annotation_region=['RSPd', 'RSPv', 'RSPagl'],
                          annotation_cmap='PiYG')

            view = get_slice_view('reference', plane_type='coronal', resolution=self.resolution)
            x, y = self.coords

            view.plane_at(slice_index=self.anterior_index).plot(ax=ax[0], **kwargs)
            ax[0].plot(x, y, 'ro')

            view.plane_at(slice_index=self.posterior_index).plot(ax=ax[1], **kwargs)
            ax[1].plot(x, y, 'ro')


if __name__ == '__main__':
    SliceViewOptions().main()
