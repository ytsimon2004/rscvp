from typing import NamedTuple

import attrs
import numpy as np
from scipy.interpolate import interp1d
from typing_extensions import Self

from neuralib.imaging.suite2p import Suite2PResult, get_neuron_signal, sync_s2p_rigevent, SIGNAL_TYPE
from rscvp.util.cli import BEHAVIOR_COVARIANT, get_neuron_list
from rscvp.util.position import load_interpolated_position
from rscvp.util.util_lick import LickTracker
from rscvp.util.util_trials import TrialSelection
from stimpyp import RiglogData

__all__ = ['GLMInputData']


class DTypeParams(NamedTuple):
    temporal_res: float
    """Temporal resolution of the model in hz"""

    sampling_rate: float
    """Behavioral measure sampling rate in hz"""

    n_cov_bins: int
    """Number of bins for the value domain in the dtype"""


DEFAULT_DTYPE_PARAMS: dict[BEHAVIOR_COVARIANT, DTypeParams] = {
    'pos': DTypeParams(10, 30, 50),
    'speed': DTypeParams(10, 30, 30),
    'acceleration': DTypeParams(10, 30, 30),
    'lick_rate': DTypeParams(10, 30, 20)
}


@attrs.define
class GLMInputData:
    """
    `Dimension parameters`:

        N = Number of neurons

        S = Number of samples = sampling rate (hz) * total time (sec)

    """
    dtype: BEHAVIOR_COVARIANT
    """``BEHAVIOR_COVARIANT``"""

    time: np.ndarray
    """Timestamp of the covariant in sec. `Array[float, S]`"""

    cov: np.ndarray
    """Covariant values. `Array[float, S]`"""

    neural_activity: np.ndarray
    """Neural activity (i.e., spks, dff). `Array[float, [N, S]]`"""

    pars: DTypeParams = attrs.field(init=False, default=attrs.Factory(dict), kw_only=True)

    def __attrs_post_init__(self):
        self.pars = DEFAULT_DTYPE_PARAMS[self.dtype]

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> Self:
        """Train/Test dataset"""
        return GLMInputData(
            self.dtype,
            self.time[idx],
            self.cov[idx],
            self.neural_activity[:, idx],
        )

    @property
    def n_temporal_bins(self) -> int:
        return int((np.max(self.time) - np.min(self.time)) * self.pars.temporal_res)

    def prepare_XY(self, neuron_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param neuron_id:
        :return:
            X: binned covariant
            Y: spike counts
            t_bins: temporal bins
        """
        X, t_bins, x_bins = np.histogram2d(self.time, self.cov, bins=(self.n_temporal_bins, self.pars.n_cov_bins))
        Y = np.histogram(self.time, t_bins, weights=self.neural_activity[neuron_id])[0]

        return X, Y, t_bins

    @classmethod
    def of_pos(cls,
               s2p: Suite2PResult,
               rig: RiglogData,
               signal_type: SIGNAL_TYPE,
               plane_index: int,
               session: str) -> Self:
        """position covariates"""
        return cls.of_var('pos', s2p, rig, signal_type, plane_index, session)

    @classmethod
    def of_speed(cls,
                 s2p: Suite2PResult,
                 rig: RiglogData,
                 signal_type: SIGNAL_TYPE,
                 plane_index: int,
                 session: str) -> Self:
        """speed covariates"""
        return cls.of_var('speed', s2p, rig, signal_type, plane_index, session)

    @classmethod
    def of_lick(cls,
                s2p: Suite2PResult,
                rig: RiglogData,
                signal_type: SIGNAL_TYPE,
                plane_index: int,
                session: str,
                lick_tracker: LickTracker | None = None) -> Self:
        return cls.of_var('lick_rate', s2p, rig, signal_type, plane_index, session, lick_tracker)

    @classmethod
    def of_acceleration(cls,
                        s2p: Suite2PResult,
                        rig: RiglogData,
                        signal_type: SIGNAL_TYPE,
                        plane_index: int,
                        session: str) -> Self:
        return cls.of_var('acceleration', s2p, rig, signal_type, plane_index, session)

    @classmethod
    def of_var(
            cls,
            dtype: BEHAVIOR_COVARIANT,
            s2p: Suite2PResult,
            rig: RiglogData,
            signal_type: SIGNAL_TYPE,
            plane_index: int,
            session: str,
            lick_tracker: LickTracker | None = None,
    ) -> Self:
        """
        Create inputs for Linear-nonlinear Poisson GLM model

        :param dtype: ``BEHAVIOR_COVARIANT``
        :param s2p:
        :param rig:
        :param signal_type:
        :param plane_index:
        :param session:
        :param lick_tracker
        :return:
        """
        trial = TrialSelection.from_rig(rig, session)
        tprofile = trial.get_selected_profile()

        sampling_rate = DEFAULT_DTYPE_PARAMS[dtype].sampling_rate
        start_time = tprofile.start_time
        end_time = tprofile.end_time
        cov_time = np.linspace(start_time, end_time, int((end_time - start_time) * sampling_rate))

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, plane_index)

        # signal
        sig, _ = get_neuron_signal(s2p, get_neuron_list(s2p), signal_type=signal_type, normalize=False)
        imask = trial.masking_time(image_time)
        image_time = image_time[imask]
        sig = sig[:, imask]  # (N, fs*t)

        #
        if sampling_rate < 10:
            # do histogram on signal peak
            from scipy.signal import find_peaks

            s = np.zeros((sig.shape[0], len(cov_time) - 1))
            for i in range(sig.shape[0]):
                s[i] = np.histogram(image_time[find_peaks(sig[i])[0]], cov_time)[0]
            sig = s
        else:
            sig = interp1d(image_time, sig,
                           axis=1, kind='nearest', copy=False, bounds_error=False, fill_value=0)(cov_time)

        #
        if dtype in ('pos', 'speed', 'acceleration'):
            pos = load_interpolated_position(rig)
            pmask = trial.masking_time(pos.t)
            ptime = pos.t[pmask]

            if dtype == 'pos':
                cov = pos.p[pmask]
            elif dtype == 'speed':
                cov = pos.v[pmask]
            elif dtype == 'acceleration':
                v = pos.v[pmask]
                dv = np.diff(v, prepend=v[0])
                cov = dv * sampling_rate
            else:
                raise ValueError(f'unknown covarient type: {dtype}')

            cov = interp1d(ptime, cov, kind='nearest', copy=False, bounds_error=False, fill_value=0)(cov_time)

        #
        elif dtype == 'lick_rate':

            if lick_tracker is None:
                lick_time = rig.lick_event.time
                time = np.linspace(start_time, end_time, int((end_time - start_time) * sampling_rate))
                value, edg = np.histogram(lick_time, time)
                t = edg[:-1]
            else:
                value = lick_tracker.pix_probability
                t = lick_tracker.camera_time

            cov = interp1d(t, value, kind='nearest', copy=False, bounds_error=False, fill_value=0)(cov_time)

        else:
            raise ValueError(f'unknown covarient type: {dtype}')

        return GLMInputData(dtype, cov_time, cov, sig)
