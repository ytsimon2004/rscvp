"""
it's the dataformat for the collaborative project in KS's paper
and the .mat file structure is original made by SH (acquisition system using SpikeGLX)
"""
import os
import subprocess
from pprint import pprint

import attrs
import numpy as np
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.position import load_interpolated_position
from rscvp.util.util_camera import truncate_video_to_pulse
from typing_extensions import Self

from neuralib.typing import PathLike
from neuralib.util.verbose import fprint

__all__ = ['KSVisualPatternData']


@attrs.define
class KSVisualPatternData:
    """
    * Shape Info:
    S = number of visual stimuli
    C = number of camera pulse
    E = number of encoder signal
    """
    header: str
    version: str
    globals: list[str]

    exp_date: str
    animal: str

    fsl: float
    """rig sampling rate in Hz, from SpikeGLX"""
    stim_pattern: np.ndarray
    """(S, 5) with (direction, sf, tf, contrast, duration)"""
    stim_time: np.ndarray
    """(S, 2), with on-off time. time in sec"""
    pupil_dat: np.ndarray
    """(C, 2) with (time, pupil area). time in sec, area in ... """
    running_speed: np.ndarray | None
    """(E, 2) with (time, speed). time in sec, speed in cm/s"""

    @classmethod
    def load(cls, opt: CameraOptions,
             direction_invert: bool = True) -> Self:
        """
        Load data from rscvp pipeline

        :param opt: caller option
        :param direction_invert: direction invert 180 degree, which is invert from stimpy v.s KS paper
        :return:
        """

        rig = opt.load_riglog_data()
        stim = rig.get_stimlog()

        # stim_pattern
        pattern = stim.get_stim_pattern()
        direction = pattern.direction
        if direction_invert:
            direction = (direction + 180) % 360
        contrast = pattern.contrast
        sf = pattern.sf
        tf = pattern.tf
        duration = pattern.duration
        stim_pattern = np.stack([direction, sf, tf, contrast, duration], axis=1)

        # pupil
        fmap = opt.load_facemap_result()
        area_smooth = fmap.get_pupil_area()
        camera_time = rig.camera_event['facecam'].time
        area_smooth = truncate_video_to_pulse(area_smooth, camera_time)
        pupil_dat = np.stack([camera_time, area_smooth], axis=1)

        # locomotion
        pos = load_interpolated_position(rig)
        tv = np.stack([pos.t, pos.v], axis=1)

        return KSVisualPatternData(
            exp_date=opt.exp_date,
            animal=opt.animal_id,
            header='rscvp Python pipeline',
            version=get_commit_hash(),
            globals=[],
            fsl=np.nan,
            stim_pattern=stim_pattern,
            stim_time=stim.stimulus_segment,
            pupil_dat=pupil_dat,
            running_speed=tv
        )

    @classmethod
    def load_from_mat(cls, file: PathLike) -> Self:
        """Load SH's data for testing"""
        from scipy.io import loadmat
        dat = loadmat(file, squeeze_me=True)

        #
        fsl = dat['fsl']
        pupil_dat = dat['timepar']
        pupil_dat[:, 0] /= fsl

        #
        stim_time = dat['vstimon'].astype(np.float64)
        # not provided off
        if stim_time.ndim == 1:
            fprint(f'lack of stim off time', vtype='warning')
            off = np.full_like(stim_time, 0.0)
            stim_time = np.stack([stim_time, off], axis=1)
        stim_time /= fsl

        return KSVisualPatternData(
            exp_date='',
            animal='',
            header=dat.get('__header__', ''),
            version=dat.get('__version__', ''),
            globals=dat.get('__globals__', []),
            fsl=fsl,
            stim_pattern=dat['stimuli'],
            pupil_dat=pupil_dat,
            stim_time=stim_time,
            running_speed=dat.get('speed', None)
        )

    def write_mat(self, output_file: PathLike) -> None:
        """write as SH's format to KS"""
        from scipy.io import savemat

        field = {
            '__header__': self.header,
            '__version__': self.version,
            '__globals__': self.globals,

            'exp_date': self.exp_date,
            'animal': self.animal,
            'fsl': self.fsl,
            'stimuli': self.stim_pattern,
            'vstim_onoff': self.stim_time,
            'timepar': self.pupil_dat,
            'running_speed': self.running_speed
        }

        savemat(output_file, field)

    def as_info_dict(self) -> None:
        dy = {
            'stim_pattern': self.stim_pattern.shape,
            'stim_time': self.stim_time.shape,
            'pupil_dat': self.pupil_dat.shape,
            'running_speed': self.running_speed.shape if self.running_speed is not None else None
        }
        pprint(dy)


def get_commit_hash() -> str:
    codedir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit_ref = (
            subprocess
            .check_output(['git', 'rev-parse',
                           '--verify', 'HEAD',
                           '--short'], cwd=codedir)
            .decode()
            .strip('\n')
        )

    except subprocess.CalledProcessError:
        commit_ref = ''

    return commit_ref
