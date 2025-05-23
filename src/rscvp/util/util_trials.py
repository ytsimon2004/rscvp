from typing import Iterable

import attrs
import numpy as np
from attr import field
from typing_extensions import Self

from neuralib.imaging.suite2p import Suite2PResult, get_neuron_signal, sync_s2p_rigevent
from neuralib.util.verbose import fprint
from rscvp.util.cli.cli_suite2p import NeuronID, get_neuron_list
from rscvp.util.position import load_interpolated_position
from stimpyp import Session, SessionInfo, RiglogData

__all__ = [
    'TrialSelection',
    'TrialSignal',
    'foreach_session_signals'
]


class TrialSelection:
    """Class for selecting behavioral trial. i.e., used for cross validation / train,test split"""

    session_trial: dict[Session, SessionInfo]

    def __init__(self, rig: RiglogData,
                 session: Session = 'all',
                 selected_trial: np.ndarray | None = None):
        """

        :param rig: ``RiglogData``
        :param session: ``Session``
        :param selected_trial: trial index, NOT trial number
        """
        self.rig = rig
        self.session_trial = self.rig.get_stimlog().session_trials()

        self._session = None  # reset

        if selected_trial is None:
            self.session_type = session  # trigger setter checking
        else:
            assert len(selected_trial) > 0, 'Empty selected trials'
            self._selected_trials = selected_trial
            self._session = session  # do not overwrite selected_trial

    @classmethod
    def from_rig(cls, rig: RiglogData, session: Session = 'all') -> Self:
        return TrialSelection(rig, session)

    @property
    def session_type(self) -> Session:
        return self._session

    @session_type.setter
    def session_type(self, session: Session):
        if session not in self.session_trial:
            raise ValueError(f'{session}')

        self._session = session

        if session == 'all':
            self._selected_trials = self.rig.lap_event.value_index
        else:
            session_info = self.session_trial[session]
            ret = session_info.in_range(self.rig.lap_event.time, self.rig.lap_event.value_index)
            self._selected_trials = np.arange(ret[0], ret[1])

    @property
    def session_info(self) -> SessionInfo:
        return self.session_trial[self.session_type]

    @property
    def selected_trials(self) -> np.ndarray:
        return self._selected_trials

    @property
    def selected_numbers(self) -> int:
        """Number of selected trials"""
        return len(self.selected_trials)

    @property
    def trials_time(self) -> np.ndarray:
        """Time for selected trials"""
        return self.rig.lap_event.time[self.selected_trials]

    @property
    def trial_range_in_session(self) -> tuple[int, int]:
        """Range within the session"""
        session_info = self.session_trial[self.session_type]
        ret = session_info.in_range(
            self.rig.lap_event.time,
            self.rig.lap_event.value_index
        )
        return ret

    # ================ #
    # Cross Validation #
    # ================ #

    def invert(self) -> Self:
        whole = np.arange(*self.trial_range_in_session)
        ret = np.setdiff1d(whole, self.selected_trials)
        return TrialSelection(self.rig, self.session_type, ret)

    def select_odd(self) -> Self:
        odd_trials = np.arange(self.trial_range_in_session[0] + 1, self.trial_range_in_session[1], 2)
        return TrialSelection(self.rig, self.session_type, odd_trials)

    def select_even(self) -> Self:
        even_trials = np.arange(*self.trial_range_in_session, 2)
        return TrialSelection(self.rig, self.session_type, even_trials)

    def select_range(self, trial_range: tuple[int, int],
                     session: Session | None = None) -> Self:
        """Select from trial range"""
        select_trials = np.arange(*trial_range)
        if session is None:
            session = self.session_type
        return TrialSelection(self.rig, session, select_trials)

    def select_odd_in_range(self, trial_range: tuple[int, int],
                            session: Session | None = None):
        """select odd trials within a range of trials"""
        if session is None:
            session = self.session_type

        t = self.select_range(trial_range, session=session)
        start, end = t.selected_trials[0], t.selected_trials[-1]
        odd_trials = np.arange(start + 1, end, 2)
        return TrialSelection(self.rig, self.session_type, odd_trials)

    def select_even_in_range(self, trial_range: tuple[int, int],
                             session: Session | None = None):
        """select even trials within a range of trials"""
        if session is None:
            session = self.session_type

        t = self.select_range(trial_range, session=session)
        start, end = t.selected_trials[0], t.selected_trials[-1]
        even_trials = np.arange(start, end, 2)
        return TrialSelection(self.rig, self.session_type, even_trials)

    def masking_trial_matrix(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Masking data with the given ``selected_trials``

        :param data: `Array[float, [..., L, ...]]`
        :param axis: default is 1
        :return: `Array[float, [..., L', ...]]`
        """
        return np.take(data, self.selected_trials, axis=axis)

    def masking_time(self, t: np.ndarray) -> np.ndarray:
        """
        Create a time mask

        :param t: Time array in sec. `Array[float, T]`
        :return: Mask `Array[bool, T]`
        """

        time = self.rig.lap_event.time  # (L+1,)
        index = self.selected_trials  # ranging from 0 to L-1

        # time index? find a trial index which interval include t
        trial_index = np.searchsorted(time, t) - 1  # (T,), ranging from 0 to L-1

        # trial index in selected_trial
        a = np.zeros_like(time, dtype=bool)  # (L+1,)
        a[index] = True
        ret = a[trial_index]  # (T)
        # two edge cases for trial index,
        # 1. t before first lap, trial_index = -1
        # 2. t after last lap , trial_index = L
        # a[L] always false since index's value range from 0 to L-1
        # a[edge_cases] always false

        return ret

    def kfold_cv(self, fold: int = 5) -> list[tuple[Self, Self]]:
        """list of train/test TrialSelection, respectively"""
        from sklearn.model_selection import KFold
        kfold_iter = KFold(fold, shuffle=False)

        ret = []
        for train_index, test_index in kfold_iter.split(self.selected_trials):
            ret.append(
                (
                    TrialSelection(self.rig, self.session_type, train_index),
                    TrialSelection(self.rig, self.session_type, test_index)
                )
            )

        return ret

    def select_fraction(self, train_fraction: float) -> tuple[Self, Self]:
        """Select fraction of the trials for training"""
        total = self.selected_numbers
        n_test = int(total * (1 - train_fraction))
        start = np.random.randint(total - n_test) + self.trial_range_in_session[0]
        trial_range = (start, start + n_test)

        test = self.select_range(trial_range)
        train = test.invert()

        return train, test

    def get_time_profile(self, verbose=True) -> 'TrialTimeProfile':
        """
        Get the time/trial profile for each session
        :param verbose:
        :return:
        """
        info = self.rig.get_stimlog().session_trials()[self.session_type]

        trial_range = info.in_range(
            self.rig.lap_event.time,
            self.rig.lap_event.value_index
        )

        start_time = info.time[0]
        end_time = info.time[1]

        if verbose:
            fprint(f'select trials in {self.session_type} session: within {trial_range},'
                   f'from {start_time} to {end_time}')

        return TrialTimeProfile(
            self,
            trial_range,
            start_time,
            end_time
        )


# TODO check if simplier way?
@attrs.define
class TrialTimeProfile:
    selection: TrialSelection
    trial_range: tuple[int, int]
    start_time: float
    end_time: float

    @property
    def session(self) -> Session:
        return self.selection.session_type

    @property
    def trial_slice(self) -> slice:
        return slice(*self.trial_range)

    @property
    def n_trials(self) -> int:
        return self.trial_range[1] - self.trial_range[0]

    def with_selected_range(self, ranging: tuple[int, int]) -> Self:
        """with only selection trial ranged in this session"""
        lap_time = self.selection.rig.lap_event.time

        n1 = self.trial_range[0] + ranging[0]
        n2 = self.trial_range[0] + ranging[1]
        t0 = lap_time[n1]
        t1 = lap_time[n2]

        # noinspection PyTypeChecker
        return attrs.evolve(self,
                            trial_range=(n1, n2),
                            start_time=t0,
                            end_time=t1)


# ======================== #
# Activity for each trials #
# ======================== #

@attrs.define
class TrialSignal:
    """Container for selected trial signals"""

    time_profile: TrialTimeProfile

    dff: np.ndarray
    """(N, IT)"""
    spks: np.ndarray
    """(N, IT)"""
    image_time: np.ndarray
    """(IT,)"""

    # optional
    position: np.ndarray = field(default=None, kw_only=True)
    """(PT,)"""
    velocity: np.ndarray = field(default=None, kw_only=True)
    """(PT,)"""
    position_time: np.ndarray = field(default=None, kw_only=True)
    """(PT,)"""
    vstim_pulse: np.ndarray = field(default=None, kw_only=True)
    """(VT,)"""
    vstim_time: np.ndarray = field(default=None, kw_only=True)
    """(VT,)"""

    @classmethod
    def of_calcium(
            cls,
            s2p: Suite2PResult,
            rig: RiglogData,
            neuron_ids: NeuronID,
            plane_index: int,
            session: Session,
            normalize: bool = False
    ) -> Self:
        neuron_list = get_neuron_list(s2p, neuron_ids)

        dff, _ = get_neuron_signal(s2p, neuron_list, signal_type='df_f', dff=True, normalize=normalize)
        spks, _ = get_neuron_signal(s2p, neuron_list, signal_type='spks', normalize=normalize)

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, plane_index)

        prf = TrialSelection(rig, session).get_time_profile()
        t0 = prf.start_time
        t1 = prf.end_time

        it_mask = np.logical_and(t0 < image_time, image_time < t1)
        image_time = image_time[it_mask]
        dff = dff[:, it_mask]
        spks = spks[:, it_mask]

        return TrialSignal(prf, dff, spks, image_time)

    @property
    def n_neurons(self) -> int:
        return self.dff.shape[0]


def foreach_session_signals(s2p: Suite2PResult,
                            rig: RiglogData,
                            neuron_ids: NeuronID,
                            plane_index: int,
                            normalize: bool = True,
                            do_smooth: bool = False,
                            trial_numbers: tuple[int, int] | None = None) -> Iterable[TrialSignal]:
    """

    :param s2p:
    :param rig:
    :param neuron_ids:
    :param plane_index:
    :param normalize: whether 01 normalize calcium raw signal
    :param do_smooth: do the smoothing of calcium raw signal
    :param trial_numbers: Only pick up the trial numbers from the beginning of each session
    :return:
    """
    neuron_list = get_neuron_list(s2p, neuron_ids)

    dff, _ = get_neuron_signal(s2p, neuron_list, signal_type='df_f', dff=True, normalize=normalize)
    spks, _ = get_neuron_signal(s2p, neuron_list, signal_type='spks', normalize=normalize)

    if do_smooth:
        from scipy.ndimage import gaussian_filter1d
        dff = gaussian_filter1d(dff, 3, axis=1)
        spks = gaussian_filter1d(spks, 3, axis=1)

    #
    pos = load_interpolated_position(rig)

    #
    stim = rig.get_stimlog().stim_square_pulse_event()

    #
    image_time = rig.imaging_event.time
    image_time = sync_s2p_rigevent(image_time, s2p, plane_index)

    #
    session_info = rig.get_stimlog().session_trials()
    session_info.pop('all', None)
    sessions = list(session_info.keys())

    for s in sessions:
        prf = TrialSelection(rig, s).get_time_profile()

        if trial_numbers is not None:
            prf = prf.with_selected_range(trial_numbers)

        t0 = prf.start_time
        t1 = prf.end_time

        # neural activity
        it_mask = np.logical_and(t0 < image_time, image_time < t1)
        _image_time = image_time[it_mask]
        _dff = dff[:, it_mask]
        _spks = spks[:, it_mask]

        # position
        pt_mask = np.logical_and(t0 < pos.t, pos.t < t1)
        position_time = pos.t[pt_mask]
        position = pos.p[pt_mask]
        velocity = pos.v[pt_mask]

        # visual
        vt_mask = np.logical_and(t0 < stim.time, stim.time < t1)
        vtime = stim.time[vt_mask]
        vpulse = stim.value[vt_mask]

        yield TrialSignal(
            prf, _dff, _spks, _image_time,
            position=position,
            velocity=velocity,
            position_time=position_time,
            vstim_pulse=vpulse,
            vstim_time=vtime
        )
