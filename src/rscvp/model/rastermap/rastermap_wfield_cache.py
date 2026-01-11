import attrs
import numpy as np
from memory_profiler import profile
from scipy.interpolate import interp1d
from typing import cast

from argclz import AbstractParser
from neuralib.locomotion import CircularPosition
from neuralib.persistence import persistence
from neuralib.widefield import compute_singular_vector
from rscvp.util.cli import PersistenceRSPOptions, WFieldOptions, TreadmillOptions
from rscvp.util.wfield import WfieldResult
from stimpyp import STIMPY_SOURCE_VERSION, RiglogData, PyVlog

__all__ = [
    'RasterInputWfield',
    'RasterMapWfieldCacheBuilder'
]


@attrs.frozen
class RasterInputWfield:
    camera_fps: float
    n_frames: int
    height: int
    width: int

    #
    n_components: int
    sv: np.ndarray
    Vsv: np.ndarray
    U: np.ndarray

    #
    xpos: np.ndarray
    ypos: np.ndarray

    #
    image_time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    visual_stim_time: np.ndarray

    @property
    def visual_stim_start(self) -> float:
        return cast(float, self.visual_stim_time[0, 0])

    def visual_stim_trange(self, trange: tuple[float, float]) -> np.ndarray:
        """select visual stim time range segments

        :return (nStim, 2)
        """
        vt = self.visual_stim_time  # (N,2) -> (N*2)
        start_idx, end_idx = np.searchsorted(vt.ravel(), list(trange))

        # map to (N, 2)
        start_stim_idx = int(start_idx // 2)
        end_stim_idx = int(end_idx // 2)

        if start_idx % 2 != 0:
            vt[start_stim_idx, 0] = trange[0]

        if end_idx % 2 != 0:
            vt[end_stim_idx, 1] = trange[1]

        return vt[start_stim_idx: end_stim_idx]


@persistence.persistence_class
class RasterMapWfieldCache:
    """
    `Dimension parameters`:

        W = image width

        H = image height

        T = number of image pulse

        C = number of components after SVD reduction

        S = number of stimulation (optional)

    """
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    source_code: STIMPY_SOURCE_VERSION = persistence.field(validator=True, filename=True)

    #
    camera_fps: float
    n_frames: int
    height: int
    """Image height H"""
    width: int
    """Image width W"""

    # singular vector
    n_components: int
    """Number of components after SVD reduction"""
    sv: np.ndarray
    """Singular values. `Array[float, C]`"""
    Vsv: np.ndarray
    """Right singular vector. `Array[float, [T, C]]`"""
    U: np.ndarray
    """left singular vector. `Array[float, [W * H, C]]`"""

    #
    xpos: np.ndarray
    """X position. `Array[float, [W, H]]`"""
    ypos: np.ndarray
    """Y position. `Array[float, [W, H]]`"""

    # behavior
    image_time: np.ndarray
    """1D wide-field imaging acquisition time. `Array[float, T]`"""
    position: np.ndarray
    """1D animal position in the environment. `Array[float, T]`"""
    velocity: np.ndarray
    """1D animal velocity in the environment. `Array[float, T]`"""
    visual_stim_time: np.ndarray
    """2D on-off visual stimulation time. `Array[float, [S,2]]`"""

    def load_result(self) -> RasterInputWfield:
        return RasterInputWfield(
            camera_fps=self.camera_fps,
            n_frames=self.n_frames,
            height=self.height,
            width=self.width,
            n_components=self.n_components,
            sv=self.sv,
            Vsv=self.Vsv,
            U=self.U,
            xpos=self.xpos,
            ypos=self.ypos,
            image_time=self.image_time,
            position=self.position,
            velocity=self.velocity,
            visual_stim_time=self.visual_stim_time
        )


class RasterMapWfieldCacheBuilder(AbstractParser,
                                  WFieldOptions,
                                  TreadmillOptions,
                                  PersistenceRSPOptions[RasterMapWfieldCache]):
    """Build the Wfield data cache for RasterMap model"""

    wfield: WfieldResult
    rig: RiglogData | PyVlog
    pos: CircularPosition
    image_time: np.ndarray

    n_components = 128

    def post_parsing(self):
        del self.source_root['suite_2p']
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    def run(self):
        self.post_parsing()
        self.load_cache()

    def empty_cache(self) -> RasterMapWfieldCache:
        return RasterMapWfieldCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            source_code=self.source_version
        )

    @profile
    def compute_cache(self, cache: RasterMapWfieldCache) -> RasterMapWfieldCache:
        self._set_attrs()

        wf = self.wfield
        rig = self.rig

        # pos
        x = np.arange(wf.width)
        y = np.arange(wf.height)
        xpos, ypos = np.meshgrid(x, y)

        #
        cache.camera_fps = rig.camera_event['1P_cam'].fps
        cache.n_frames = wf.n_frames
        cache.height = wf.height
        cache.width = wf.width
        cache.n_components = self.n_components

        # compute SVD
        self._compute_singular_vector(cache)

        cache.xpos = xpos
        cache.ypos = ypos

        #
        cache.image_time = self.image_time
        v, p = self._prepare_position_data(self.pos)
        cache.velocity = v
        cache.position = p
        cache.visual_stim_time = rig.get_stimlog().stimulus_segment

        return cache

    def _set_attrs(self):
        self.wfield = self.load_wfield_result()
        self.rig = self.load_riglog_data()
        self.image_time = self.wfield.camera_time

        self.pos = self.load_position()

    def _compute_singular_vector(self, cache: RasterMapWfieldCache) -> None:
        sv = compute_singular_vector(self.wfield.sequences, self.n_components)

        cache.sv = sv.singular_value
        cache.Vsv = sv.right_vector
        cache.U = sv.left_vector

    def _prepare_position_data(self, pos: CircularPosition) -> tuple[np.ndarray, np.ndarray]:
        _vel = interp1d(
            pos.t, pos.v,
            bounds_error=False,
            fill_value=0
        )(self.image_time)

        _pos = interp1d(
            pos.t, pos.p,
            bounds_error=False,
            fill_value=0
        )(self.image_time)

        return _vel, _pos


if __name__ == '__main__':
    RasterMapWfieldCacheBuilder().main()
