from rscvp.topology.util import RSCObjectiveFOV
from rscvp.util.cli import CommonOptions, SBXOptions, StatisticOptions
from rscvp.util.util_ibl import IBLAtlasPlotWrapper

from argclz import AbstractParser, as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation

__all__ = ['FieldOfViewOptions']


@publish_annotation('main', project='rscvp', figure='fig.1C', as_doc=True)
class FieldOfViewOptions(AbstractParser, SBXOptions):
    DESCRIPTION = 'Based on gspread, plot 2p FOV corresponding location in dorsal cortex view'

    exp_date: str = as_argument(CommonOptions.exp_date).with_options(required=False)
    animal_id: str = as_argument(CommonOptions.animal_id).with_options(required=False)
    header = as_argument(StatisticOptions.header).with_options(required=False)

    def run(self):
        fovs = RSCObjectiveFOV.load_from_gspread(self.exp_date, self.animal_id, page=self.gspread_page)

        with plot_figure(None) as ax:
            ibl = IBLAtlasPlotWrapper()
            ibl.plot_scalar_on_slice(
                ['root'],
                plane='top',
                hemisphere='left',
                background='boundary',
                cmap='Purples',
                mapping='Beryl',
                ax=ax
            )

            if not isinstance(fovs, list):
                fovs = [fovs]

            cy = {'aRSC': 'g', 'pRSC': 'r'}
            for i, fov in enumerate(fovs):
                fov = fov.to_um()
                ax.add_patch(fov.to_polygon(ec=cy.get(fov.region_name, 'k'), alpha=0.8))


if __name__ == '__main__':
    FieldOfViewOptions().main()
