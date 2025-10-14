from typing import Final, Literal

from argclz import AbstractParser, argument
from neuralib.atlas.view import get_slice_view
from neuralib.plot import plot_figure

__all__ = ['SliceViewOptions']


class SliceViewOptions(AbstractParser):
    DESCRIPTION = 'Plot the slice view of the injection site'

    resolution: Final = 10

    region: Literal['inj', 'SS', 'MO', 'ACA', 'PTLp', 'ATN', 'POST', 'LD'] = argument(
        '--region',
        default='inj',
        help='region to plot',
    )

    def run(self):

        match self.region:
            case 'inj':
                self.plot_injection()
            case 'ACA':
                index = 552
                self.plot_coronal(self.region, index)
            case 'SS':
                index = 708
                self.plot_coronal(self.region, index)
            case 'MO':
                index = 438
                self.plot_coronal(self.region, index)
            case 'PTLp':
                index = 726
                self.plot_coronal(self.region, index)
            case 'ATN':
                index = 615
                self.plot_coronal(['AD', 'AM', 'AV'], index)
            case 'POST':
                index = 960
                self.plot_coronal(self.region, index)
            case 'LD':
                index = 669
                self.plot_coronal(self.region, index)
            case _:
                raise NotImplementedError(f'{self.region}')

    def plot_injection(self):
        anterior_index = 690
        posterior_index = 863
        coords = (500, 800)  # ml, dv in um

        with plot_figure(None, 1, 2, sharex=True, sharey=True) as ax:
            kwargs = dict(boundaries=True,
                          annotation_region=['RSPd', 'RSPv', 'RSPagl'],
                          annotation_cmap='copper')

            view = get_slice_view('reference', plane_type='coronal', resolution=self.resolution)
            x, y = coords

            view.plane_at(slice_index=anterior_index).plot(ax=ax[0], **kwargs)
            ax[0].plot(x, y, 'ro')

            view.plane_at(slice_index=posterior_index).plot(ax=ax[1], **kwargs)
            ax[1].plot(x, y, 'ro')

    def plot_coronal(self, region: str | list[str], index: int):
        with plot_figure(None) as ax:
            if isinstance(region, str):
                region = [region]
            kwargs = dict(boundaries=True, annotation_region=region, annotation_cmap='copper')

            view = get_slice_view('reference', plane_type='coronal', resolution=self.resolution)
            view.plane_at(slice_index=index).plot(ax=ax, reference_bg_value=10, **kwargs)


if __name__ == '__main__':
    SliceViewOptions().main()
