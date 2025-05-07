from rscvp.util.cli import StimpyOptions
from rscvp.util.cli.cli_sbx import SBXOptions
from rscvp.util.cli.cli_suite2p import Suite2pOptions

from argclz import AbstractParser
from neuralib.imaging.registration import CellularCoordinates
from neuralib.imaging.suite2p.core import get_s2p_coords
from neuralib.io import csv_header
from neuralib.util.verbose import publish_annotation
from .util import RSCObjectiveFOV

__all__ = ['RoiLocOptions']


@publish_annotation('main', project='rscvp', as_doc=True)
class RoiLocOptions(AbstractParser, Suite2pOptions, StimpyOptions, SBXOptions):
    DESCRIPTION = 'Get the xy actual coordinates (relative to Bregma, in um) for each cell'

    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        fov = RSCObjectiveFOV.load_from_gspread(self.exp_date, self.animal_id, page=self.gspread_reference)
        s2p = self.load_suite_2p()
        neuron_list = self.get_neuron_list()

        def get_roi(neuron, plane, factor) -> CellularCoordinates:
            return get_s2p_coords(s2p, neuron, plane, factor).relative_origin(fov)

        # raw cords
        self.brain_mapping = False
        roi_raw = get_roi(neuron_list, self.plane_index, self.pixel2distance_factor(s2p))

        # scaled cords
        self.brain_mapping = True
        roi_scale = get_roi(neuron_list, self.plane_index, self.pixel2distance_factor(s2p))

        output = self.get_data_output('cord')
        fields = ['neuron_id', 'ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords']
        depth = self.image_depth
        with csv_header(output.csv_output, fields) as csv:
            for i, n in enumerate(neuron_list):
                csv(n, roi_raw.ap[i], roi_raw.ml[i], roi_scale.ap[i], roi_scale.ml[i], depth)


if __name__ == '__main__':
    RoiLocOptions().main()
