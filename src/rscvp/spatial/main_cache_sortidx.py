from typing import Union

import numpy as np
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions, PosBinActCacheBuilder
from rscvp.spatial.util import sort_neuron, normalized_trial_avg
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.cli.cli_treadmill import TreadmillOptions
from rscvp.util.typing import SIGNAL_TYPE
from rscvp.util.util_trials import TrialSelection

from argclz import argument, AbstractParser, int_tuple_type, union_type
from neuralib.persistence import persistence
from neuralib.persistence.cli_persistence import get_options_and_cache


@persistence.persistence_class
class SortIdxCache:
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    plane_index: str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    signal_type: SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    selection: str = persistence.field(validator=True, filename=True)
    sortby: str = persistence.field(validator=True, filename=True, filename_prefix='SORTBY_')

    ###
    neuron_idx: np.ndarray
    """neuron_id (N')"""

    sort_idx: np.ndarray
    """sorted (np.argsort) neuron_id (N,). NOTE that the idx is after `cli.selection` procedure"""

    @property
    def sorted_neuron_idx(self) -> np.ndarray:
        return self.neuron_idx[self.sort_idx]


class AbstractSortIdxOptions(TreadmillOptions, SelectionOptions):
    use_trial: Union[str, tuple[int, int]] = argument(
        '-t', '--use-trial',
        type=union_type(int_tuple_type, str),
        default=None,
        help='plot the TRIAL_AVG activity within trial range (START, STOP) or session name'
             'NOTE that this can replace opt.session in general'
    )

    use_sorted_strategy: str = argument(
        '--sort',
        metavar='NAME',
        help='use which strategy for neuron sorting'
             'if not given, sorted by use_trial',
    )


class SortIdxCacheBuilder(AbstractParser, AbstractSortIdxOptions,
                          ApplyPosBinActOptions, PersistenceRSPOptions[SortIdxCache]):
    DESCRIPTION = 'compute the trial average activity sorting idx along position bins'

    def post_parsing(self):
        if self.session is not None:
            raise ValueError('opt.use_trial instead to avoid misunderstand')

        if self.use_trial is not None:
            if isinstance(self.use_trial, str):
                pass
            else:
                # (int, int)
                if len(self.use_trial) != 2:
                    raise ValueError(f'illegal use-trial, len != 2 : {self.use_trial}')
                if not all([isinstance(it, int) for it in self.use_trial]):
                    raise TypeError(f'illegal use-trial, not int : {self.use_trial}')

        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    def run(self):
        self.post_parsing()
        self.load_cache()

    def selected_signal(self) -> np.ndarray:
        """get binned data (N', L, B)"""
        mx = self.get_selected_neurons()
        signal_all = self.apply_binned_act_cache().occ_activity[mx]

        # trial selection (use_trial instead of opt.session)
        if isinstance(self.use_trial, str):
            rig = self.load_riglog_data()
            trial_range = TrialSelection(rig, self.use_trial).get_time_profile().trial_slice
        elif isinstance(self.use_trial, tuple):
            trial_range = slice(*self.use_trial)
        else:
            raise TypeError(f'use_trial: {type(self.use_trial)} type error')

        return signal_all[:, trial_range, :]  # (N', L', B)

    # ============= #
    # Cache methods #
    # ============= #

    def empty_cache(self) -> SortIdxCache:
        if self.use_sorted_strategy is None:
            if isinstance(self.use_trial, str):
                use_strategy = self.use_trial
            elif isinstance(self.use_trial, tuple):
                a, b = self.use_trial
                use_strategy = f'trial{a}-{b}'
            else:
                raise TypeError('')
        else:
            use_strategy = self.use_sorted_strategy

        return SortIdxCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            plane_index=self.plane_index if self.plane_index is not None else '_concat',
            signal_type=self.signal_type,
            selection=(self.selection_prefix()),
            sortby=use_strategy
        )

    def compute_cache(self, cache: SortIdxCache) -> SortIdxCache:
        return self._compute_cache_plane(cache)

    def _compute_cache_plane(self, cache: SortIdxCache) -> SortIdxCache:
        if self.use_sorted_strategy is not None:
            raise RuntimeError(f'cannot make cache from {self.use_sorted_strategy}')

        cache.neuron_idx = self.selected_neurons
        cache.sort_idx = sort_neuron(normalized_trial_avg(self.selected_signal()))
        return cache


class ApplySortIdxOptions(AbstractSortIdxOptions):

    def apply_sort_idx_cache(self, error_when_missing=False) -> SortIdxCache:
        return get_options_and_cache(SortIdxCacheBuilder, self, error_when_missing)


if __name__ == '__main__':
    SortIdxCacheBuilder().main()
