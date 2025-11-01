from typing import Literal, get_args

import attrs
import numpy as np
from attr import field
from typing_extensions import Self

from neuralib.imaging.suite2p import Suite2PResult, get_neuron_signal, sync_s2p_rigevent
from rscvp.util.cli.cli_suite2p import NeuronID, get_neuron_list
from stimpyp import Session, SessionInfo, RiglogData, RigEvent

__all__ = [
    'TRIAL_CV_TYPE',
    'TrialSelection',
    'TrialSignal',
    'signal_trial_cv_helper',
]

# @formatter:off
TRIAL_CV_TYPE = Literal[
    # visual exp (vop)
    'light', 'light-odd', 'light-even',
    'visual', 'visual-odd', 'visual-even',
    'dark', 'dark-odd', 'dark-even',

    # light-dark exp (ldl)
    'light-bas', 'light-bas-odd', 'light-bas-even',
    'dark', 'dark-odd', 'dark-even',
    'light-end', 'light-end-odd', 'light-end-even',

    # vr
    'close', 'close-odd', 'close-even',
    'open', 'open-odd', 'open-even',

    # all
    'all', 'all-odd', 'all-even'
]
# @formatter:on


class TrialSelection:
    """Class for selecting behavioral trial. i.e., used for cross validation / train,test split"""

    session_trial: dict[Session, SessionInfo]

    def __init__(self, rig: RiglogData,
                 session: Session = 'all',
                 selected_trial: np.ndarray | None = None,
                 use_virtual_space: bool = False):
        """

        :param rig: ``RiglogData``
        :param session: ``Session``
        :param selected_trial: trial index, NOT trial number
        """
        self.rig = rig
        self.use_virtual_space = use_virtual_space

        if self.use_virtual_space:
            self.session_trial = self.rig.get_pygame_stimlog().session_trials()
        else:
            self.session_trial = self.rig.get_stimlog().session_trials()

        self._session = None  # reset

        if selected_trial is None:
            self.session_type = session  # trigger setter checking
            self._complete_trial_session = True
        else:
            assert len(selected_trial) > 0, 'Empty selected trials'
            self._selected_trials = selected_trial
            self._session = session  # do not overwrite selected_trial
            self._complete_trial_session = False

    @classmethod
    def from_rig(cls, rig: RiglogData, session: Session = 'all', use_virtual_space: bool = False) -> Self:
        return TrialSelection(rig, session, use_virtual_space=use_virtual_space)

    @property
    def lap_event(self) -> RigEvent:
        if self.use_virtual_space:
            return self.rig.get_pygame_stimlog().virtual_lap_event
        else:
            return self.rig.lap_event

    @property
    def session_type(self) -> Session:
        return self._session

    @session_type.setter
    def session_type(self, session: Session):
        if session not in self.session_trial:
            raise ValueError(f'{session}')

        self._session = session

        if session == 'all':
            self._selected_trials = self.lap_event.value_index
        else:
            session_info = self.session_trial[session]
            ret = session_info.in_range(self.lap_event.time, self.lap_event.value_index)
            self._selected_trials = np.arange(ret[0], ret[1])

    @property
    def session_info(self) -> SessionInfo:
        """session info for selected session, regardless of selection"""
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
        return self.lap_event.time[self.selected_trials]

    @property
    def trial_range(self) -> tuple[int, int]:
        """Range within the session (full session range, not just selected trials)"""
        return self.session_info.in_range(
            self.lap_event.time,
            self.lap_event.value_index
        )

    @property
    def start_time(self) -> float:
        if self._complete_trial_session:
            return self.session_info.time[0]
        else:
            return self.trials_time[0]

    @property
    def end_time(self) -> float:
        if self._complete_trial_session:
            return self.session_info.time[1]
        else:
            return self.trials_time[-1]

    # ================ #
    # Cross Validation #
    # ================ #

    def invert(self) -> Self:
        whole = np.arange(*self.trial_range)
        ret = np.setdiff1d(whole, self.selected_trials)
        return TrialSelection(self.rig, self.session_type, ret, use_virtual_space=self.use_virtual_space)

    def select_odd(self) -> Self:
        odd_trials = np.arange(self.trial_range[0] + 1, self.trial_range[1], 2)
        return TrialSelection(self.rig, self.session_type, odd_trials, use_virtual_space=self.use_virtual_space)

    def select_even(self) -> Self:
        even_trials = np.arange(*self.trial_range, 2)
        return TrialSelection(self.rig, self.session_type, even_trials, use_virtual_space=self.use_virtual_space)

    def select_range(self, trial_range: tuple[int, int],
                     session: Session | None = None) -> Self:
        """Select from trial range"""
        select_trials = np.arange(*trial_range)
        if session is None:
            session = self.session_type
        return TrialSelection(self.rig, session, select_trials, use_virtual_space=self.use_virtual_space)

    def select_odd_in_range(self, trial_range: tuple[int, int],
                            session: Session | None = None) -> Self:
        """select odd trials within a range of trials"""
        if session is None:
            session = self.session_type

        t = self.select_range(trial_range, session=session)
        start, end = t.selected_trials[0], t.selected_trials[-1]
        odd_trials = np.arange(start + 1, end, 2)
        return TrialSelection(self.rig, self.session_type, odd_trials, use_virtual_space=self.use_virtual_space)

    def select_even_in_range(self, trial_range: tuple[int, int],
                             session: Session | None = None) -> Self:
        """select even trials within a range of trials"""
        if session is None:
            session = self.session_type

        t = self.select_range(trial_range, session=session)
        start, end = t.selected_trials[0], t.selected_trials[-1]
        even_trials = np.arange(start, end, 2)
        return TrialSelection(self.rig, self.session_type, even_trials, use_virtual_space=self.use_virtual_space)

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

        time = self.lap_event.time  # (L+1,)
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
                    TrialSelection(self.rig, self.session_type, train_index, use_virtual_space=self.use_virtual_space),
                    TrialSelection(self.rig, self.session_type, test_index, use_virtual_space=self.use_virtual_space)
                )
            )

        return ret

    def select_fraction(self, train_fraction: float) -> tuple[Self, Self]:
        """Select fraction of the trials for training"""
        total = self.selected_numbers
        n_test = int(total * (1 - train_fraction))
        start = np.random.randint(total - n_test) + self.trial_range[0]
        trial_range = (start, start + n_test)

        test = self.select_range(trial_range)
        train = test.invert()

        return train, test


# ======================== #
# Activity for each trials #
# ======================== #

@attrs.define
class TrialSignal:
    """Container for selected trial signals"""

    session: Session
    """session name"""

    time: np.ndarray
    """(T,)"""

    dff: np.ndarray
    """(N, T)"""
    spks: np.ndarray
    """(N, T)"""

    # optional
    position: np.ndarray = field(default=None, kw_only=True)
    """(T,)"""
    velocity: np.ndarray = field(default=None, kw_only=True)
    """(T,)"""

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
            normalize: bool = False,
            use_virtual_space: bool = False,
    ) -> Self:
        neuron_list = get_neuron_list(s2p, neuron_ids)

        dff, _ = get_neuron_signal(s2p, neuron_list, signal_type='df_f', dff=True, normalize=normalize)
        spks, _ = get_neuron_signal(s2p, neuron_list, signal_type='spks', normalize=normalize)

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, plane_index)

        trial_sel = TrialSelection(rig, session, use_virtual_space=use_virtual_space)
        t0 = trial_sel.start_time
        t1 = trial_sel.end_time

        it_mask = np.logical_and(t0 < image_time, image_time < t1)
        image_time = image_time[it_mask]
        dff = dff[:, it_mask]
        spks = spks[:, it_mask]

        return TrialSignal(session, image_time, dff, spks)

    @property
    def n_neurons(self) -> int:
        return self.dff.shape[0]


def signal_trial_cv_helper(rig: RiglogData,
                           signal: np.ndarray,
                           use_trial: TRIAL_CV_TYPE | tuple[int, int],
                           use_virtual_space: bool = False) -> np.ndarray:
    trial_literal = get_args(TRIAL_CV_TYPE)

    if use_trial in trial_literal:
        parts = use_trial.split('-')
        if len(parts) == 2:
            trial_name, cv_name = parts
        elif len(parts) == 1:
            trial_name = parts[0]
            cv_name = None
        else:
            raise ValueError(f'Invalid use_trial format: {use_trial}')

        ts = TrialSelection(rig, trial_name, use_virtual_space=use_virtual_space)

        match cv_name:
            case 'odd':
                ts = ts.select_odd()
            case 'even':
                ts = ts.select_even()
            case None:
                pass
            case _:
                raise ValueError(f'unknown trial cv name: {cv_name}')

        idx = ts.selected_trials

        return signal[:, idx, :]

    elif isinstance(use_trial, tuple):
        return signal[:, np.arange(*use_trial), :]

    else:
        raise TypeError(f'use_trial must be a tuple of trial number or {trial_literal}')
