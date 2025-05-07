from typing import Literal

from rscvp.util.cli.cli_rastermap import RasterMapOptions
from rscvp.util.cli.cli_suite2p import Suite2pOptions

from argclz import AbstractParser, as_argument, argument
from argclz.dispatch import Dispatch, dispatch


class LaunchRasterMapGUIOptions(AbstractParser, RasterMapOptions, Suite2pOptions, Dispatch):
    DESCRIPTION = 'Launch RasterMap GUI based on different data type'

    neuron_bins = as_argument(RasterMapOptions.neuron_bins).with_options(required=False)

    load_mode: Literal['2p', 'wfield', 'proc'] = argument(
        '--load',
        required=True,
        help='Mode of loading data type'
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.invoke_command(self.load_mode)

    @dispatch('2p')
    def _load_suite2p_raw(self):
        """Raw data from suite2p folder"""
        directory = self.suite2p_directory
        if self.signal_type == 'df_f':
            act = directory / 'F.npy'
        elif self.signal_type == 'spks':
            act = directory / 'spks.npy'
        else:
            raise ValueError('')

        iscell = directory / 'iscell.npy'

        self.launch_gui(spike_file=str(act), iscell_file=str(iscell))

    @dispatch('proc')
    def _load_proc_embedding(self):
        """Data after run rastermap GUI (manually save) or
         customized pipeline (*embedding.npy) in the cache folder"""
        cached_path = self.rastermap_result_cache(self.plane_index)

        if not cached_path.exists():
            raise FileNotFoundError(f'{cached_path} not found')

        self.launch_gui(proc_file=str(cached_path))

    @dispatch('wfield')
    def _load_wfield_result(self):
        pass


if __name__ == '__main__':
    LaunchRasterMapGUIOptions().main()
