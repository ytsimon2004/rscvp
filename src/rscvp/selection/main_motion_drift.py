from argclz import AbstractParser, argument
from neuralib.plot import plot_figure
from neuralib.suite2p import Suite2PResult
from rscvp.selection.utils import moving_average
from rscvp.util.cli import DataOutput, StimpyOptions, Suite2pOptions

__all__ = ['MotionDriftOptions']


class MotionDriftOptions(AbstractParser, Suite2pOptions, StimpyOptions):
    DESCRIPTION = 'see the motion correction by suite2p'

    binned_size: int = argument(
        '--bin-size',
        default=60,
        help='binned size for all the frames'
    )

    non_rigid_block: int | None = argument(
        '--non-rigid-block',
        help='which non-rigid registration block for plotting',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('md')
        s2p = self.load_suite_2p()
        self.plot_motion_correction(s2p, output_info)

    def plot_motion_correction(self, s2p: Suite2PResult,
                               output: DataOutput):
        """for visualizing the motion drifting in rigid registration or non-rigid registration"""

        if self.non_rigid_block is None:
            corr = s2p.rigid_xy_offset
            label = 'rigid_offset'
        else:
            corr = s2p.nonrigid_xy_offsets[:, self.non_rigid_block]  # pick up specific block
            label = f'non_rigid_offset_{self.non_rigid_block}'

        corr = moving_average(corr, self.binned_size)

        output_file = output.summary_figure_output(label)
        with plot_figure(output_file) as ax:
            ax.plot(corr, label=label)
            ax.legend()


if __name__ == '__main__':
    MotionDriftOptions().main()
