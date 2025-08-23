from functools import cached_property
from typing import Final

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Self

from argclz import AbstractParser, argument, copy_argument
from neuralib.persistence import *
from neuralib.persistence.cli_persistence import get_options_and_cache
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.util_trials import TrialSelection
from stimpyp import Session


@persistence.persistence_class
class SISortAlignPeakCache(ETLConcatable):
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    plane_index: str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    signal_type: str = persistence.field(validator=True, filename=True)
    selection: str = persistence.field(validator=True, filename=True)
    session_type: Session = persistence.field(validator=True, filename=True)
    act_normalized: bool = persistence.field(validator=True, filename=False)
    use_virtual_space: bool = persistence.field(validator=True, filename=True, filename_prefix='vr_')

    neuron_idx: np.ndarray
    """neuronal index. `Array[float, N]`"""
    src_idx: np.ndarray
    """which plane. i.e., 0-3. `Array[float, N]`"""
    spatial_info: np.ndarray
    """spatial information. `Array[float, N]`"""
    trial_avg_binned_data: np.ndarray
    """`Array[float, [N, B]]`"""

    @classmethod
    def concat_etl(cls, data: list[Self], do_sort: bool = True) -> Self:
        validate_concat_etl_persistence(data, ('exp_date', 'animal'))

        const = data[0]
        ret = SISortAlignPeakCache(
            exp_date=const.exp_date,
            animal=const.animal,
            plane_index='_concat',
            signal_type=const.signal_type,
            selection=const.selection,
            session_type=const.session_type,
            act_normalized=const.act_normalized,
            use_virtual_space=const.use_virtual_space,
        )

        ret.neuron_idx = np.concatenate([it.neuron_idx for it in data])
        ret.src_idx = np.concatenate([it.src_idx for it in data])
        ret.spatial_info = np.concatenate([it.spatial_info for it in data])
        ret.trial_avg_binned_data = np.vstack([it.trial_avg_binned_data for it in data])

        if do_sort:
            new_index = np.argsort(ret.spatial_info)[::-1]
            ret.neuron_idx = ret.neuron_idx[new_index]
            ret.src_idx = ret.src_idx[new_index]
            ret.spatial_info = ret.spatial_info[new_index]
            ret.trial_avg_binned_data = ret.trial_avg_binned_data[new_index]

        return ret


class AbstractAlignPeakOptions(SelectionOptions):
    no_sort: bool = argument(
        '--no-sort',
        help='whether sorted the plot based on si',
    )

    with_top: int | None = argument(
        '--top', '--with-top',
        type=int,
        default=None,
        help='only pick up top si cell numbers'
    )

    act_normalized: Final = 'none'  # for curve plot raw dff
    signal_type: Final = 'spks'

    @property
    def sort(self) -> bool:
        return not self.no_sort


class SISortAlignPeakCacheBuilder(AbstractParser, AbstractAlignPeakOptions,
                                  ApplyPosBinActOptions, PersistenceRSPOptions[SISortAlignPeakCache]):
    DESCRIPTION = 'Cache for (N, B) binned calcium data, N idx is sorted by spatial information'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.load_cache()

    def empty_cache(self) -> SISortAlignPeakCache:
        return SISortAlignPeakCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            plane_index=self.plane_index,
            signal_type=self.signal_type,
            selection=self.selection_prefix(),
            session_type=self.session,
            act_normalized=self.act_normalized,
            use_virtual_space=self.use_virtual_space,
        )

    def compute_cache(self, cache: SISortAlignPeakCache) -> SISortAlignPeakCache:
        cache.trial_avg_binned_data = self.prepare_align_data()
        cache.neuron_idx = self.selected_neurons[self.si_sorted_index]
        cache.src_idx = np.full(self.n_selected_neurons, self.plane_index)
        cache.spatial_info = self.si[self.si_sorted_index]

        return cache

    def prepare_align_data(self) -> np.ndarray:
        rig = self.load_riglog_data()
        binned_sig = self.apply_binned_act_cache().occ_activity

        # neuron selection
        cell_mask = self.get_selected_neurons()
        binned_sig = binned_sig[cell_mask]

        if self.sort:
            binned_sig = binned_sig[self.si_sorted_index]

        # trial selection
        trial_range = (
            TrialSelection(rig, self.session, use_virtual_space=self.use_virtual_space)
            .get_selected_profile()
            .trial_slice
        )
        binned_sig = binned_sig[:, trial_range, :]

        return circular_pos_shift(binned_sig, smooth=False)

    # ===== #

    @cached_property
    def si(self) -> np.ndarray:
        """spatial information after selection"""
        if self.plane_index is not None:
            file = self.get_data_output('si', self.used_session,
                                        use_virtual_space=self.use_virtual_space,
                                        latest=True).csv_output
            df = pl.read_csv(file)
            si = df[f'si_{self.used_session}'].to_numpy()
        else:
            si = self.get_csv_data(f'si_{self.used_session}')

        cell_mask = self.get_selected_neurons()

        return si[cell_mask]

    @cached_property
    def si_sorted_index(self) -> np.ndarray:
        """return the *descending* sorted idx based on `spatial information`"""
        return np.argsort(self.si)[::-1]


class ApplyAlignPeakOptions(AbstractAlignPeakOptions):

    def apply_align_peak_cache(self, error_when_missing=False) -> SISortAlignPeakCache:
        if self.plane_index is not None:
            return self._apply_single_plane(error_when_missing)
        else:
            return self._apply_concat_plane(error_when_missing)

    def _apply_single_plane(self, error_when_missing):
        return get_options_and_cache(SISortAlignPeakCacheBuilder, self, error_when_missing)

    def _apply_concat_plane(self, error_when_missing):
        data = []
        n_planes = self.load_suite_2p(force_load_plane=0).n_plane
        for i in range(n_planes):
            opt = copy_argument(SISortAlignPeakCacheBuilder(), self, plane_index=i)
            data.append(opt.load_cache(error_when_missing=error_when_missing))

        return SISortAlignPeakCache.concat_etl(data)


def circular_pos_shift(binned_sig: np.ndarray,
                       smooth: bool = False) -> np.ndarray:
    """
    do alignment of peak response to 0 (lag) by circular shifting

    :param binned_sig: (N, L, B)
    :param smooth: whether binned activity smoothing

    :return:
        (N, B)
    """
    sig = np.nanmean(binned_sig, axis=1)  # trial avg
    n_neurons = sig.shape[0]
    n_bins = sig.shape[1]

    ret = np.zeros((n_neurons, n_bins))
    for i, n in enumerate(range(n_neurons)):
        s = sig[n]  # (B, )

        mi = np.argmax(s)
        s = np.roll(s, int(n_bins / 2 - mi))

        if smooth:
            s = gaussian_filter1d(s, 3)

        ret[i] = s

    return ret


if __name__ == '__main__':
    SISortAlignPeakCacheBuilder().main()
