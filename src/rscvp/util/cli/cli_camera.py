from datetime import datetime
from pathlib import Path
from typing import ClassVar, Final, Literal

import numpy as np
from neuralib.tracking import read_facemap
from neuralib.tracking.facemap import FaceMapResult, KeyPoint
from neuralib.util.interp import interp_timestamp
from neuralib.util.utils import ensure_dir, uglob
from rscvp.util.pixviz import PixVizResult
from rscvp.util.util_camera import truncate_video_to_pulse
from rscvp.util.util_lick import LICK_EVENT_TYPE, LickTracker
from stimpyp import RiglogData, CAMERA_VERSION

from argclz import argument, int_tuple_type, str_tuple_type
from .cli_stimpy import StimpyOptions

__all__ = ['CameraOptions']


class CameraOptions(StimpyOptions):
    """Camera tracking options"""

    GROUP_CAM: ClassVar = 'Camera options'
    """group camera options"""

    LABCAM_OFFSET_ISSUE_DATE: Final[datetime] = datetime.strptime('210925', "%y%m%d").date()
    """labcam offset issue"""

    camera_version: CAMERA_VERSION = argument(
        '--camera',
        metavar='VERSION',
        default='labcam',
        group=GROUP_CAM,
        help='which camera acquisition system'
    )

    track_type: Literal['keypoints', 'pupil', 'lick'] = argument(
        '--track',
        group=GROUP_CAM,
        help='which track types'
    )

    keypoint: tuple[KeyPoint, ...] = argument(
        '-K', '--keypoint',
        type=str_tuple_type,
        group=GROUP_CAM,
        help='facemap keypoint',
    )

    frames: tuple[int, int] = argument(
        '--frames',
        metavar='FRAME,...',
        type=int_tuple_type,
        group=GROUP_CAM,
        help='start-end frame number'
    )

    not_outlier_filtering: bool = argument(
        '--no-filter',
        group=GROUP_CAM,
        help='Apply median filter to the keypoints data to remove outliers'
    )

    with_keypoints: bool = argument(
        '--with-keypoints',
        group=GROUP_CAM,
        help='whether has keypoints tracking result, otherwise, load svd result only'
    )

    lick_event_source: LICK_EVENT_TYPE = argument(
        '-e', '--event_type',
        group=GROUP_CAM,
        default='facecam',
        help='lick event source, can be either facecam or lickmeter',
    )

    alignment: bool = argument(
        '--align',
        group=GROUP_CAM,
        help='set true, if labcam and riglog are not sync while acquisition,'
             'if electrical signal is not reliable (loss too much..), set offset kwarg as false',
    )

    offset_time: float | None = argument(
        '--offset',
        group=GROUP_CAM,
        help='manual give offset value (in sec) if labcam and riglog are not sync while acquisition'
    )

    lick_thres: float | None = argument(
        '-t', '--threshold',
        metavar='VALUE',
        type=float,
        default=None,
        group=GROUP_CAM,
        help='threshold for licking probability to classify as licking behavior using "MiceLick", '
             'if None, use the automatically calculated value',
    )

    @property
    def camera_time(self) -> np.ndarray:
        """get camera time from riglog data"""
        rig = self.load_riglog_data()
        match self.track_type:
            case 'pupil':
                cam = 'eyecam'
            case 'keypoints' | 'lick':
                cam = 'facecam'
            case _:
                raise ValueError(f'unknown track type: {self.track_type}')
        # noinspection PyTypeChecker
        return rig.camera_event[cam].time

    # =============== #
    # PixViz analysis #
    # =============== #

    @property
    def pixviz_directory(self) -> Path:
        """customized package ``pixviz`` output directory, for tracking licking video with pixel changing"""
        ret = self.get_src_path('track') / 'pixviz'
        return ensure_dir(ret)

    def load_pixviz_result(self) -> PixVizResult:
        """load ``pixviz`` results from :attr:`pixviz_directory`"""
        file = uglob(self.pixviz_directory, '*.npy')
        meta = uglob(self.pixviz_directory, '*.json')
        return PixVizResult.load(file, meta)

    def load_lick_tracker(self, rig: RiglogData | None = None) -> LickTracker:
        if rig is None:
            rig = self.load_riglog_data()
        #
        file = uglob(self.pixviz_directory, '*.npy')
        meta = uglob(self.pixviz_directory, '*.json')

        ret = LickTracker.load_from_rig(rig, file, meta=meta, threshold=self.lick_thres)

        return ret.with_offset() if self.alignment else ret

    def get_lick_event(self, rig: RiglogData | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get lick event with time and signal

        :param rig: ``RiglogData`` or None
        :return: lick time: `Array[float, L]` and lick signal: `Array[float, L]`
        """
        if rig is None:
            rig = self.load_riglog_data()

        match self.lick_event_source:
            case 'facecam':
                px = self.load_pixviz_result()
                sig = px.get_data('lick')
                t = self.camera_time
                s = truncate_video_to_pulse(sig, t)
            case 'lickmeter':
                event = rig.lick_event.time
                t, s = interp_timestamp(event, t0=rig.exp_start_time, t1=rig.exp_end_time, sampling_rate=30)
            case _:
                raise ValueError(f'unknown lick event source: {self.lick_event_source}')

        return t, s

    # ================ #
    # Facemap analysis #
    # ================ #

    @property
    def facemap_directory(self) -> Path:
        """``facemap`` output directory for video tracking of keypoints and pupil size"""
        ret = self.get_src_path('track') / 'facemap'
        ensure_dir(ret)
        return ret

    def load_facemap_result(self) -> FaceMapResult:
        """load facemap result from :attr:`facemap_directory`"""
        return read_facemap(self.facemap_directory)
