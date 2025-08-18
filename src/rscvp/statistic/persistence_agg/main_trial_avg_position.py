from pathlib import Path
from typing import Final

import numpy as np

from argclz import try_int_type, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation, fprint, print_load, print_save
from rscvp.spatial.main_cache_occ import PosBinActCache, ApplyPosBinActOptions
from rscvp.spatial.util import sort_neuron
from rscvp.spatial.util_plot import plot_sorted_trial_averaged_heatmap
from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg, data_region_dict
from rscvp.util.cli import Region, SelectionOptions, TreadmillOptions
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE
from rscvp.util.util_trials import TrialSelection
from stimpyp import Session

__all__ = ['PositionBinPersistenceAgg']


@publish_annotation('main', project='rscvp', figure=['fig.2B', 'fig.S2E'], as_doc=True)
class PositionBinPersistenceAgg(AbstractPersistenceAgg, ApplyPosBinActOptions, TreadmillOptions, SelectionOptions):
    DESCRIPTION = 'Plot the sorted position-binned trial averaged activity heatmap for batch dataset'

    sort_session: Session | None = argument(
        '--sort',
        default=None,
        help='Sorted index in which session then save cache, If None, then auto save based on `--session`'
    )

    force_sort: bool = argument(
        '--force-sort',
        help='Force sorting not matter there is index cache'
    )

    region_validate_page: GSPREAD_SHEET_PAGE = argument(
        '--page',
        required=True,
        help='Page to validate the region column to be consistent'
    )

    field: Final = dict(plane_index=try_int_type, region=str)

    signal_type = 'df_f'
    act_normalized = 'local'
    pre_selection = True
    pc_selection = 'slb'

    def post_parsing(self):
        if self.session is None:
            raise ValueError('specify a certain session for masking the position [N, L, B] cache')

        super().post_parsing()

    def run(self):
        self.post_parsing()
        caches = self.get_cache_list()
        data = self.get_cache_data(caches)
        self.plot(data)

    @property
    def index_cache(self) -> Path:
        region = self._validate_region()
        return self.statistic_dir / f'population_index_{self.sort_session}_{region}.npy'

    def _validate_region(self) -> Region:
        """Validate and get region (region-specific analysis)"""
        data = self.data_identity

        dy = data_region_dict(self.region_validate_page)
        val = [dy[d] for d in data]
        if len(set(val)) != 1:
            raise RuntimeError('not unique region')
        else:
            return val[0]

    def get_cache_list(self) -> list[PosBinActCache]:
        ret = []
        for i, _ in enumerate(self.foreach_dataset(**self.field)):
            self.exp_list.append(self.exp_date)
            self.animal_list.append(self.animal_id)

            cell_mask = self.get_selected_neurons()

            # trial
            trial = np.arange(*TrialSelection(self.load_riglog_data(), self.session).get_selected_profile().trial_range)
            ret.append(self.apply_binned_act_cache()
                       .with_mask(cell_mask)
                       .with_trial(trial))

        return ret

    def get_cache_data(self, cache_list: list[PosBinActCache]) -> np.ndarray:
        """(N, B) -> (N', B) for batch dataset"""
        ret = [cache.get_trial_avg_occ() for cache in cache_list]

        try:
            act = np.vstack(ret)
            fprint(f'occ shape: {act.shape}')
        except ValueError as e:
            print(repr(e))
            raise RuntimeError('check bins size are consistent')
        else:
            if self.sort_session is None:
                self.sort_session = self.session  # as current session

            if not self.index_cache.exists() or self.force_sort:
                idx = sort_neuron(act)
                np.save(self.index_cache, idx)
                print_save(self.index_cache)
            else:
                idx = np.load(self.index_cache)
                print_load(self.index_cache)

            return act[idx]

    def plot(self, data: np.ndarray):
        """
        :param data: (N', B)
        """
        with plot_figure(None, tight_layout=False) as ax:
            plot_sorted_trial_averaged_heatmap(
                data,
                cmap='cividis',
                interpolation='antialiased',
                total_length=self.belt_length,
                cue_loc=self.cue_loc,
                ax=ax
            )


if __name__ == '__main__':
    PositionBinPersistenceAgg().main()
