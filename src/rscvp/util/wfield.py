from pathlib import Path

import numpy as np
from tifffile import tifffile
from tqdm import tqdm
from typing_extensions import Self

from neuralib.typing import PathLike
from stimpyp import STIMPY_SOURCE_VERSION, PyCamlog, LabCamlog, RiglogData, PyVlog

__all__ = ['WfieldResult']


class WfieldResult:
    __slots__ = ('sequences', 'source_version', 'riglog', 'camlog')

    def __init__(
            self,
            sequences: np.ndarray,
            source_version: STIMPY_SOURCE_VERSION,
            riglog: PyVlog | RiglogData,
            camlog: LabCamlog | PyCamlog
    ):
        """

        :param sequences: `Array[float, [F, H, W]]`
        :param source_version:
        :param riglog:
        :param camlog:
        """
        self.sequences = sequences
        self.source_version = source_version
        self.riglog = riglog
        self.camlog = camlog

    @classmethod
    def load_from_avi(
            cls,
            avi_path: PathLike,
            source_version: STIMPY_SOURCE_VERSION,
            riglog: PyVlog | RiglogData,
            camlog: LabCamlog | PyCamlog
    ) -> Self:
        """

        :param avi_path:
        :param source_version:
        :param riglog:
        :param camlog:
        :return:
        """
        from neuralib.imglib.io import read_avi
        seq = read_avi(str(avi_path))

        return WfieldResult(seq, source_version, riglog, camlog)

    @classmethod
    def load_raw_tif(
            cls,
            tif_path: PathLike,
            source_version: STIMPY_SOURCE_VERSION,
            riglog: PyVlog | RiglogData,
            camlog: LabCamlog | PyCamlog
    ) -> Self:
        """ **If computational heavy for further image processing,
        considering do the FOV cropping or down-sampling resolution in ImageJ
        """
        files = sorted(list(Path(tif_path).glob('*.tif')))

        if len(files) == 0:
            raise FileNotFoundError(f'no tif files in {tif_path}')

        seq = []
        for f in tqdm(files, desc='load raw tif', unit='files'):
            seq.append(tifffile.memmap(f))

        return WfieldResult(np.vstack(seq), source_version, riglog, camlog)

    @property
    def n_frames(self) -> int:
        """F"""
        return self.sequences.shape[0]

    @property
    def height(self) -> int:
        """H"""
        return self.sequences.shape[1]

    @property
    def width(self) -> int:
        """W"""
        return self.sequences.shape[2]

    @property
    def camera_time(self) -> np.ndarray:
        """Do the interpolation and get the timing for each frames"""
        return self.camlog.get_camera_time(self.riglog, interpolate=True)
