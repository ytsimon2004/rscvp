"""
** Wfield analysed data structure

210302_YW006_1P_YW/
         ├── cache/ (e.g., rastermap persistence)
         ├── processed/ (e.g., processed .avi from raw image tif sequences)
         ├── run00_onep_retino_circling_squares/ (e.g., raw data, including tif sequences, .camlog)
         ├── retinotopic/ (e.g., storage trial_average.tif)
         └── *[analysis]/ (e.g., retinotopic, rastermap...)
"""
from pathlib import Path
from typing import ClassVar

from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.cli.cli_stimpy import StimpyOptions
from rscvp.util.wfield import WfieldResult

from argclz import argument, as_argument
from neuralib.util.utils import uglob
from stimpyp import LabCamlog, PyCamlog, STIMPY_SOURCE_VERSION

__all__ = ['WFieldOptions']


class WFieldOptions(CameraOptions):
    GROUP_WFIELD: ClassVar = 'Wfield imaging options'

    daq_type = '1P'

    source_version: STIMPY_SOURCE_VERSION = as_argument(StimpyOptions.source_version).with_options(default='pyvstim')

    load_raw: bool = argument(
        '-T', '--tif', '--tiff',
        group=GROUP_WFIELD,
        help='load sequences from raw tif'
    )

    load_avi: bool = argument(
        '--avi',
        group=GROUP_WFIELD,
        help='load sequences from from avi'
    )

    @property
    def phys_dir(self) -> Path:
        return self.get_io_config().phy_animal_dir

    @property
    def processed_dir(self) -> Path:
        """for general image processing, and trimmed/down-sampling .avi storage"""
        return self.phys_dir / 'processed'

    @property
    def retinotopic_directory(self) -> Path:
        """for retinotopic image processing"""
        return self.phys_dir / 'retinotopic'

    def _check_dir(self):
        for d in (self.processed_dir, self.retinotopic_directory):
            if not d.exists():
                d.mkdir(exist_ok=True)

    def load_wfield_result(self, **kwargs) -> WfieldResult:
        if not self.load_raw and not self.load_avi:
            raise ValueError(f'load sequences either using from --tif or --avi')

        self._check_dir()

        #
        log = self.load_riglog_data(**kwargs)
        camlog = self.load_camlog()
        if self.load_raw:
            raw_dir = uglob(self.phys_dir, 'run00*', is_dir=True)
            return WfieldResult.load_raw_tif(raw_dir, self.source_version, log, camlog)
        elif self.load_avi:
            avi = uglob(self.processed_dir, '*.avi')
            return WfieldResult.load_from_avi(avi, self.source_version, log, camlog)
        else:
            raise ValueError('')

    def load_camlog(self) -> LabCamlog | PyCamlog:
        """used for wfield alignment?"""
        camlog_dir = uglob(self.phys_dir, 'run00*', is_dir=True)

        match self.camera_version:
            case 'labcam':
                return LabCamlog.load(camlog_dir)
            case 'pycam':
                return PyCamlog.load(camlog_dir)
            case _:
                raise ValueError('')
