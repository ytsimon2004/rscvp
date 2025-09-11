from functools import cached_property
from pathlib import Path
from typing import ClassVar

from argclz import argument, int_tuple_type
from neuralib.locomotion import CircularPosition
from .cli_stimpy import StimpyOptions

__all__ = ['TreadmillOptions']


class TreadmillOptions(StimpyOptions):
    """Options for linear treadmill task"""

    GROUP_TREADMILL: ClassVar = 'Belt Options'
    """group treadmill options"""

    # ----- Position ----- #

    pos_bins: int = argument(
        '--pbins',
        metavar='VALUE',
        default=100,
        validator=lambda it: it > 0,
        group=GROUP_TREADMILL,
        help='bin number for the belt',
    )

    belt_length: int = argument(
        '--belt-length',
        default=150,
        validator=lambda it: it > 0,
        group=GROUP_TREADMILL,
        help='length of the belt'
    )

    tactile_cues: tuple[int, ...] = argument(
        '--cue',
        metavar='VALUE',
        type=int_tuple_type,
        default=(50, 100),
        group=GROUP_TREADMILL,
        help='cue location',
    )

    # ----- Running ----- #

    running_epoch: bool = argument(
        '--run',
        group=GROUP_TREADMILL,
        help='whether select only the running epoch',
    )

    smooth_vel: bool = argument(
        '--vsmooth',
        group=GROUP_TREADMILL,
        help='do velocity smoothing',
    )

    cutoff_vel: float = argument(
        '--vcut',
        metavar='VALUE',
        default=-20,
        group=GROUP_TREADMILL,
        help='velocity cutoff for transient negative calculation',
    )

    # ----- Plot ----- #

    psth_sec: float = argument(
        '--psth',
        metavar='VALUE',
        group=GROUP_TREADMILL,
        default=3,
        help='one side limit for peri-event raster(s)',
    )

    # ----- position ----- #

    invalid_position_cache: bool = argument(
        '--invalid-position-cache',
        group=GROUP_TREADMILL,
        help='force to re-compute the position cache'
    )

    _position_sampling_rate_encoder: int = 300
    _position_sampling_rate_virtual: int = 50

    # ----- Landmarks ----- #

    _track_landmarks: tuple[int, ...] | None = None
    _virtual_linear_row = 0
    _virtual_landmark_char = 'v'

    @cached_property
    def track_length(self) -> int:
        """get track length depending on either physical or virtual track"""
        if self.use_virtual_space:
            rig = self.load_riglog_data()
            return int(rig.get_pygame_stimlog().get_virtual_length())
        else:
            return self.belt_length

    @property
    def track_landmarks(self) -> tuple[int, ...]:
        """get track landmarks location (cm) depending on either physical or virtual track"""
        if self._track_landmarks is None:
            self._track_landmarks = self._get_landmarks()
        return self._track_landmarks

    def _get_landmarks(self) -> tuple[int, ...]:
        if self.use_virtual_space:
            rig = self.load_riglog_data()
            landmarks = rig.get_pygame_stimlog().get_landmarks(
                row=self._virtual_linear_row,
                char=self._virtual_landmark_char
            )

            if len(landmarks) == 0:
                return ()

            return tuple([int(it[0]) for it in landmarks])
        else:
            return self.tactile_cues

    @cached_property
    def position_sampling_rate(self) -> int:
        if self.use_virtual_space:
            return self._position_sampling_rate_virtual
        else:
            return self._position_sampling_rate_encoder

    @property
    def position_cache(self) -> Path:
        suffix = 'virtual_position_cache.npy' if self.use_virtual_space else 'position_cache.npy'
        return self.cache_directory / suffix

    def load_position(self) -> CircularPosition:
        from rscvp.util.position import load_interpolated_position

        rig = self.load_riglog_data()

        return load_interpolated_position(
            rig,
            sample_rate=self.position_sampling_rate,
            force_compute=self.invalid_position_cache,
            cache_file=self.position_cache,
            use_virtual_space=self.use_virtual_space,
            norm_length=self.track_length,
        )
