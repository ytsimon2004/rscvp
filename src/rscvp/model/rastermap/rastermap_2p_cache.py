from typing import Optional

import attrs
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import zscore

from argclz import AbstractParser
from neuralib.imaging.suite2p import SIGNAL_TYPE, Suite2PResult, get_neuron_signal, sync_s2p_rigevent, get_s2p_coords
from neuralib.locomotion import CircularPosition
from neuralib.persistence import persistence
from rscvp.util.cli import RasterMapOptions, SBXOptions, SelectionOptions, CameraOptions, PersistenceRSPOptions, \
    TreadmillOptions
from rscvp.util.util_camera import truncate_video_to_pulse
from rscvp.util.util_trials import TrialSelection
from stimpyp import RiglogData

__all__ = [
    'RasterInput2P',
    'RasterMap2PCacheBuilder'
]


@attrs.frozen
class RasterInput2P:
    """
    `Dimension parameters`:

        N = number of neurons

        T = number of image pulse

        S = number of stimulation (optional)
    """

    xy_pos: np.ndarray
    """Soma central position.`Array[float, [2, N]]`"""
    neural_activity: np.ndarray
    """2D Calcium activity. `Array[float, [N, T]]`"""
    image_time: np.ndarray
    """1D Calcium imaging time. `Array[float, T]`"""
    position: np.ndarray
    """1D animal position. `Array[float, T]`"""
    velocity: np.ndarray
    """1D animal velocity. `Array[float, T]`"""
    lap_index: np.ndarray
    """1D trial index (laps in circular env). `Array[float, T]`"""
    pupil_area: Optional[np.ndarray]
    """1D animal pupil area. `Array[float, T]`"""
    visual_stim_time: np.ndarray
    """2D on-off visual stimulation time. `Array[float, [S,2]]`"""

    def __attrs_post_init__(self):
        assert self.neural_activity.shape[1] == len(self.position) == len(self.velocity)

    @property
    def n_neurons(self) -> int:
        return self.neural_activity.shape[0]

    @property
    def x_pos(self) -> np.ndarray:
        return self.xy_pos[0]

    @property
    def y_pos(self) -> np.ndarray:
        return self.xy_pos[1]

    def get_landmarks_index(self, time_mask: np.ndarray | None = None,
                            *,
                            tolerance: float = 0.5,
                            cue_loc: tuple[float, ...] = (50, 100)) -> np.ndarray:
        """
        Find cue(s) location indices.
        **Note that the return is relative value to the ``time_mask`` (start from zero)**

        :param time_mask: time mask. `Array[bool, T]`
        :param tolerance: tolerance for interpolation of position finding the diff
        :param cue_loc: Cue location
        :return: Cue indices
        """
        if time_mask is not None:
            lap_index = self.lap_index[time_mask]
            pos = self.position[time_mask]
        else:
            lap_index = self.lap_index
            pos = self.position

        split_index = np.where(np.diff(lap_index) > tolerance)[0]
        x = np.split(pos, split_index + 1)
        base_index = np.concatenate([np.array([0]), split_index])

        ret = []
        for loc in cue_loc:
            cue_index = [np.searchsorted(it, loc) + b for it, b in zip(x, base_index) if np.max(it) > loc]
            ret.append(cue_index)

        ret = np.sort(np.concatenate(ret))

        return ret

    @property
    def visual_stim_start(self) -> float:
        return float(self.visual_stim_time[0, 0])

    def visual_stim_trange(self, trange: tuple[float, float]) -> np.ndarray:
        """select visual stim time range segments

        :return: `Array[float, [S, 2]]`
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
class RasterMapCache:
    """
    `Dimension parameters`:

        N = number of neurons

        T = number of image pulse

        S = number of stimulation (optional)
    """
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    plane_index: int = persistence.field(validator=True, filename=True, filename_prefix='plane')
    signal_type: SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    session: str = persistence.field(validator=True, filename=True)
    selection: str = persistence.field(validator=True, filename=True)

    #
    xy_pos: np.ndarray
    """Soma central position.`Array[float, [2, N]]`"""
    neural_activity: np.ndarray
    """2D Calcium activity. `Array[float, [N,T]]`"""
    image_time: np.ndarray
    """1D Calcium imaging time. `Array[float, T]`"""
    position: np.ndarray
    """1D animal position. `Array[float, T]`"""
    velocity: np.ndarray
    """1D animal velocity. `Array[float, T]`"""
    lap_index: np.ndarray
    """1D trial index (laps in circular env). `Array[float, T]`"""
    pupil_area: np.ndarray | None
    """1D animal pupil area. `Array[float, T]`"""
    visual_stim_time: np.ndarray | None
    """2D on-off visual stimulation time. `Array[float, [S,2]]`"""

    def load_result(self) -> RasterInput2P:
        return RasterInput2P(
            xy_pos=self.xy_pos,
            neural_activity=self.neural_activity,
            image_time=self.image_time,
            position=self.position,
            velocity=self.velocity,
            lap_index=self.lap_index,
            pupil_area=self.pupil_area if hasattr(self, 'pupil_area') else None,
            visual_stim_time=self.visual_stim_time
        )


class RasterMap2PCacheBuilder(AbstractParser,
                              SelectionOptions,
                              SBXOptions,
                              RasterMapOptions,
                              CameraOptions,
                              TreadmillOptions,
                              PersistenceRSPOptions[RasterMapCache]):
    """Building the rastermap cache for 2p dataset"""

    rig: RiglogData
    s2p: Suite2PResult
    pos: CircularPosition

    trial: TrialSelection
    start_time: float
    end_time: float

    image_time: np.ndarray
    image_mask: np.ndarray

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.session is None:
            self.session = 'all'

    def run(self):
        self.post_parsing()
        self.load_cache()

    def empty_cache(self) -> RasterMapCache:
        return RasterMapCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            plane_index=self.plane_index,
            signal_type=self.signal_type,
            session=self.session,
            selection=self.selection_prefix(),
        )

    # ============= #
    # Compute Cache #
    # ============= #

    def compute_cache(self, cache: RasterMapCache) -> RasterMapCache:
        self._set_attrs()  # for the callers not build cache
        act = self._prepare_neural_activity()
        vel, pos, idx = self._prepare_position_data(self.pos)

        cache.xy_pos = self._prepare_roi_position()

        cache.neural_activity = act
        cache.image_time = self.image_time
        cache.sampling_rate = self.s2p.fs

        cache.velocity = vel
        cache.position = pos
        cache.lap_index = idx

        if self.is_vop_protocol:
            cache.visual_stim_time = self.rig.get_stimlog().get_stim_pattern().time
        else:
            cache.visual_stim_time = None

        if self.with_pupil:
            cache.pupil_area = self._prepare_pupil_area()

        return cache

    def _prepare_image_time(self) -> None:
        image_time = self.rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, self.s2p, self.plane_index)
        self.image_mask = np.logical_and(self.start_time < image_time, image_time < self.end_time)
        self.image_time = image_time[self.image_mask]

    def _set_attrs(self) -> None:
        self.rig = self.load_riglog_data()
        self.s2p = self.load_suite_2p()
        self.trial = TrialSelection.from_rig(self.rig, self.session, use_virtual_space=self.use_virtual_space)
        self.start_time = self.trial.session_info.time[0]
        self.end_time = self.trial.session_info.time[1]
        self.pos = self.load_position()

        self._prepare_image_time()

    def _prepare_roi_position(self) -> np.ndarray:
        c = get_s2p_coords(self.s2p,
                           self.selected_neurons,
                           plane_index=self.plane_index,
                           factor=self.pixel2distance_factor(self.s2p))
        x = c.ml
        y = c.ap

        return np.vstack([x, y])

    def _prepare_neural_activity(self, to_zscore: bool = True) -> np.ndarray:
        act = get_neuron_signal(self.s2p, self.get_selected_neurons(), signal_type=self.signal_type)[0]
        act = act[:, self.image_mask]

        if to_zscore:
            act = zscore(act, axis=1)

        return act

    def _prepare_position_data(self, pos: CircularPosition) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """interpolate the position related parameters to imaging sampling

        :param: ``CircularPosition``
        :return velocity, position and corresponding lap index
        """
        position_mask = np.logical_and(self.start_time < pos.t, pos.t < self.end_time)
        position_time = pos.t[position_mask]
        position = pos.p[position_mask]
        velocity = pos.v[position_mask]
        lap_idx = pos.trial_array[position_mask]

        _vel = interp1d(
            position_time, velocity,
            bounds_error=False,
            fill_value=0
        )(self.image_time)

        _pos = interp1d(
            position_time, position,
            bounds_error=False,
            fill_value=0
        )(self.image_time)

        _lap_idx = interp1d(
            position_time, lap_idx,
            bounds_error=False,
            fill_value=0
        )(self.image_time)

        return _vel, _pos, _lap_idx

    def _prepare_pupil_area(self) -> np.ndarray:
        """prepare pupil area data from facemap"""
        self.track_type = 'pupil'
        fmap = self.load_facemap_result()
        pupil = fmap.get_pupil_area()
        cam_time = self.rig.camera_event['eyecam'].time

        if self.alignment:  # manual give
            cam_time += self.offset_time

        pupil = truncate_video_to_pulse(pupil, cam_time)

        return interp1d(
            cam_time, pupil,
            bounds_error=False,
            fill_value=0
        )(self.image_time)


if __name__ == '__main__':
    RasterMap2PCacheBuilder().main()
