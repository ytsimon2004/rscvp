from rscvp._side_proj.ks_proj.ks_mat_io import KSVisualPatternData
from rscvp.util.cli.cli_camera import CameraOptions

from argclz import AbstractParser


class KSVisualMatOptions(AbstractParser, CameraOptions):
    DESCRIPTION = 'Create a .mat file for KS paper'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        dat = KSVisualPatternData.load(self, direction_invert=True)
        output_file = self.data_output / self.filename / f'{self.exp_date}_{self.animal_id}_KS.mat'
        dat.write_mat(output_file)


if __name__ == '__main__':
    KSVisualMatOptions().main()
