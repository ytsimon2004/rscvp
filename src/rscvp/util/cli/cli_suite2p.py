from pathlib import Path
from typing import Literal, ClassVar

import numpy as np

from argclz import argument, try_int_type
from neuralib.imaging.suite2p import SIGNAL_TYPE, Suite2PResult
from .cli_core import CommonOptions

__all__ = [
    'Suite2pOptions',
    'get_neuron_list',
    'NeuronID',
    'NORMALIZE_TYPE'
]

NORMALIZE_TYPE = Literal['global', 'local', 'none']
NeuronID = int | list[int] | slice | np.ndarray | None


class Suite2pOptions(CommonOptions):
    GROUP_SUITE2P: ClassVar[str] = 'Suite2P Options'

    plane_index: int | None = argument(
        '-P', '--plane',
        metavar='INDEX',
        type=try_int_type,
        group=GROUP_SUITE2P,
        help='plane_index for the imaging optic plane'
    )

    neuron_id: NeuronID | None = argument(
        '-N', '--neuron-id', '--neuron_id',
        metavar='NEURON',
        type=try_int_type,
        group=GROUP_SUITE2P,
        help='neuron ID. use all neurons if options not presented',
    )

    signal_type: SIGNAL_TYPE = argument(
        '-T', '--sig-type', '--signal-type',
        metavar='TYPE',
        default='df_f',
        group=GROUP_SUITE2P,
        help='signal type, either df_f or deconvolved spike',
    )

    act_normalized: NORMALIZE_TYPE = argument(
        '--norm',
        default='local',
        group=GROUP_SUITE2P,
        help='normalization method',
    )

    disable_ops_check: bool = argument(
        '--disable-ops-check',
        group=GROUP_SUITE2P,
        help='disable suite2p ops check',
    )

    _first_plane_depth: float = 300
    """default first image plane depth for rsc (um)"""

    _default_etl_interval: float = 80
    """default fixed interval between etl scanning (um)"""

    _has_chan2: bool = None  # examine after load, cache

    @property
    def suite2p_directory(self) -> Path:
        if self.plane_index is None:
            raise ValueError('')
        return self.get_src_path('suite2p') / f'plane{self.plane_index}'

    @property
    def has_chan2(self) -> bool:
        if self._has_chan2 is None:
            s2p = self.load_suite_2p()
            self._has_chan2 = s2p.has_chan2
        return self._has_chan2

    @property
    def image_depth(self) -> float:
        """imaging depth in um (fixed for approximation)"""
        if self.plane_index is None:
            raise ValueError('')
        return self._first_plane_depth + self.plane_index * self._default_etl_interval

    def load_suite_2p(self, cell_prob: bool | float = 0.5,
                      channel: int = 0,
                      force_load_plane: int | None = None) -> Suite2PResult:
        """
        Load suite2p result

        :param cell_prob: cell probability
        :param channel: which PMT channel
        :param force_load_plane: for load the setting usage (i.e., prevent empty `combined` folder)
        :return: ``Suite2PResult``
        """
        if force_load_plane is not None:
            self.plane_index = force_load_plane

        frate_check = None if self.disable_ops_check else 30.0
        ret = Suite2PResult.load(self.suite2p_directory, cell_prob, channel=channel, runtime_check_frame_rate=frate_check)

        self._has_chan2 = ret.has_chan2

        return ret

    def get_neuron_list(self) -> list[int]:
        if self.plane_index is None:
            raise ValueError('get neurons list should be a specific optical plane')

        n = Suite2PResult.load_total_neuron_number(self.suite2p_directory)
        return list(range(n))

    def launch_gui(self):
        from suite2p.gui import gui2p
        gui2p.run(str(self.suite2p_directory / 'stat.npy'))


def get_neuron_list(opt: Suite2pOptions | Suite2PResult,
                    neuron_ids: NeuronID | None = None) -> list[int]:
    match opt, neuron_ids:
        case Suite2PResult(), None:
            return list(range(opt.n_neurons))
        case Suite2pOptions(), None:
            return opt.get_neuron_list()
        case _, int():
            return [neuron_ids]
        case _:
            return list(neuron_ids)
