import subprocess
from pathlib import Path
from typing import ClassVar, Optional

from argclz import argument, int_tuple_type
from neuralib.model.rastermap import RasterMapResult
from neuralib.util.verbose import fprint
from .cli_core import CommonOptions

__all__ = ['RasterMapOptions']


class RasterMapOptions(CommonOptions):
    GROUP_RMAP: ClassVar = 'RasterMap model Options'
    RESULT_CACHE_NAME: ClassVar = 'cache-rastermap_embedding'

    neuron_bins: int = argument(
        '--bins',
        group=GROUP_RMAP,
        required=True,
        help='number of neurons to bin over'
    )

    time_range: tuple[int, int] = argument(
        '--time',
        type=int_tuple_type,
        default=(0, 100),
        validator=lambda it: len(it) == 2,
        group=GROUP_RMAP,
        help='rastermap plot time range, in sec'
    )

    with_cue: bool = argument(
        '--with-cue',
        group=GROUP_RMAP,
        help='with cue label in the rastermap'
    )

    with_pupil: bool = argument(
        '--with-pupil',
        group=GROUP_RMAP,
        help='with pupil area in the rastermap'
    )

    force_compute: bool = argument(
        '--force',
        group=GROUP_RMAP,
        help='force re-compute rastermap'
    )

    def run_rastermap(self, **kwargs) -> RasterMapResult:
        """
        Run the rastermap model sorting.
        **Check the detail docs in source code

        :return:
        """
        pass

    def rastermap_result_cache(self, plane_index: Optional[int] = None) -> Path:
        d = self.get_src_path('cache')
        filename = self.RESULT_CACHE_NAME

        if plane_index is not None:
            filename += f'_plane{plane_index}'

        return (d / filename).with_suffix('.npy')

    @staticmethod
    def launch_gui(spike_file: Optional[str] = None,
                   proc_file: Optional[str] = None,
                   ops_file: Optional[str] = None,
                   iscell_file: Optional[str] = None) -> None:
        """

        :param spike_file: spike matrix (N, T)
        :param proc_file:
        :param ops_file:
        :param iscell_file: (N, 2) with [bool, probability]
        :return:
        """

        cmds = ['python', '-m', 'rastermap']

        if spike_file is not None:
            cmds.extend(['--S', spike_file])

        if proc_file is not None:
            cmds.extend(['--proc', proc_file])

        if ops_file is not None:
            cmds.extend(['--ops', ops_file])

        if iscell_file is not None:
            cmds.extend(['--iscell', iscell_file])

        fprint(f'{cmds=}')
        subprocess.check_call(cmds)
