from os import PathLike

import numpy as np

from argclz import AbstractParser
from neuralib.imaging.suite2p import Suite2PResult, get_neuron_signal, sync_s2p_rigevent
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import publish_annotation, print_save
from rscvp.util.cli import TreadmillOptions, SelectionOptions
from rscvp.util.position import load_interpolated_position
from stimpyp import RiglogData

__all__ = ['PositionDecodeCacheBuilder']


@publish_annotation('sup', project='rscvp', figure='fig.S2', caption='alternative way to run bayes decoder (posdc side project)')
class PositionDecodeCacheBuilder(AbstractParser, TreadmillOptions, SelectionOptions):
    DESCRIPTION = """
    Build the cache for Bayes decoding, which used on posdc pipeline, for more comprehensive cross-validation
    """

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if not self.is_ldl_protocol:
            raise NotImplementedError('VOP protocol')

    def run(self):
        self.post_parsing()

        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()
        output_file = ensure_dir(self.cache_directory / 'posdc') / f'posdc_plane{self.plane_index}.npz'
        self._build_cache(s2p, rig, output_file)

    def _build_cache(self, s2p: Suite2PResult, rig: RiglogData, output_file: PathLike):
        #
        df_f = get_neuron_signal(s2p, signal_type='df_f')[0]
        spks = get_neuron_signal(s2p, signal_type='spks')[0]
        act_time = sync_s2p_rigevent(rig.imaging_event.time, s2p, self.plane_index)

        #
        pos = load_interpolated_position(rig)
        position = pos.p
        position_time = pos.t
        lap_time = rig.lap_event.time

        #
        session_trials = rig.get_stimlog().session_trials()
        dark_info = session_trials['dark']
        lights_off_lap = dark_info.in_range(rig.lap_event.time, rig.lap_event.value_index)[0]

        if self.is_ldl_protocol:
            light_end_info = session_trials['light_end']
            lights_off_time = (dark_info.time[0], light_end_info.time[0])
        else:
            lights_off_time = dark_info.time[0]

        np.savez(output_file,
                 df_f=df_f,
                 spks=spks,
                 act_time=act_time,
                 position=position,
                 position_time=position_time,
                 lap_time=lap_time,
                 lights_off_lap=lights_off_lap,
                 lights_off_time=lights_off_time,
                 trial_length=self.belt_length)

        print_save(output_file)


if __name__ == '__main__':
    PositionDecodeCacheBuilder().main()
