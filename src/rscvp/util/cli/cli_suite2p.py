from pathlib import Path
from typing import Literal, ClassVar

import cv2
import numpy as np
from matplotlib import pyplot as plt

from argclz import argument, try_int_type
from neuralib.imaging.suite2p import SIGNAL_TYPE, Suite2PResult
from neuralib.imglib.norm import enhance_blood_vessels
from neuralib.typing import PathLike
from .cli_core import CommonOptions

__all__ = [
    'Suite2pOptions',
    'get_neuron_list',
    'NeuronID',
    'NORMALIZE_TYPE',
    'suite2p_alignment_fov'
]

NORMALIZE_TYPE = Literal['global', 'local', 'none']
NeuronID = int | list[int] | slice | np.ndarray | None


class Suite2pOptions(CommonOptions):
    """Suite2P options for calcium imaging data"""

    GROUP_SUITE2P: ClassVar[str] = 'Suite2P Options'
    """group suite2p options"""

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
        """suite2p result directory with given plane index"""
        if self.plane_index is None:
            return self.get_src_path('suite2p') / 'combined'  # placeholder
        return self.get_src_path('suite2p') / f'plane{self.plane_index}'

    @property
    def has_chan2(self) -> bool:
        """if suite2p result has chan2 data"""
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
        :return: :class:`~neuralib.imaging.suite2p.core.Suite2PResult`
        """
        if force_load_plane is not None:
            self.plane_index = force_load_plane

        frate_check = None if self.disable_ops_check else 30.0
        ret = Suite2PResult.load(self.suite2p_directory, cell_prob, channel=channel,
                                 runtime_check_frame_rate=frate_check)

        self._has_chan2 = ret.has_chan2

        return ret

    def get_all_neurons(self) -> list[int]:
        """get list of all neuron indices"""
        if self.plane_index is None:
            raise ValueError('get neurons list should be a specific optical plane')

        n = Suite2PResult.load_total_neuron_number(self.suite2p_directory)
        return list(range(n))

    def launch_gui(self):
        """launch suite2p gui"""
        from suite2p.gui import gui2p
        gui2p.run(str(self.suite2p_directory / 'stat.npy'))


def get_neuron_list(opt: Suite2pOptions | Suite2PResult,
                    neuron_ids: NeuronID | None = None) -> list[int]:
    """
    Get a list of neuron IDs based on provided options and identifiers.

    :param opt: :class:`Suite2pOptions` or :class:`~neuralib.imaging.suite2p.core.Suite2PResult`.
    :param neuron_ids: Optional parameter identifying neurons to retrieve.
        It can be a single neuron ID, a list of neuron IDs, or ``None``. If ``None`` then all neurons are returned.
    :return: A list of integers representing the neuron IDs.
    """
    match opt, neuron_ids:
        case Suite2PResult(), None:
            return list(range(opt.n_neurons))
        case Suite2pOptions(), None:
            return opt.get_all_neurons()
        case _, int():
            return [neuron_ids]
        case _:
            return list(neuron_ids)


def suite2p_alignment_fov(ops_file: PathLike, bright_field_file: PathLike) -> None:
    """
    Align Suite2P FOV (from .sbx) with bright field FOV.

    :param ops_file: Path to Suite2P ops file
    :param bright_field_file: Path to bright field image
    """
    from neuralib.imaging.suite2p import Suite2pGUIOptions

    ops: Suite2pGUIOptions = np.load(ops_file, allow_pickle=True).tolist()
    actual_fov = ops['meanImg']
    bright_field_fov = cv2.imread(bright_field_file)

    actual_fov = cv2.rotate(actual_fov, cv2.ROTATE_90_COUNTERCLOCKWISE)
    actual_fov = cv2.flip(actual_fov, 0)

    actual_fov = enhance_blood_vessels(actual_fov)
    bright_field_fov = enhance_blood_vessels(bright_field_fov)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(actual_fov, cmap='gray')
    ax[1].imshow(bright_field_fov, cmap='gray')
    plt.show()
