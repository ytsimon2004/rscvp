import numpy as np
from rscvp.spatial.util import PositionSignal, normalized_trial_avg
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_stimpy import StimpyOptions
from rscvp.util.cli.cli_suite2p import NORMALIZE_TYPE, Suite2pOptions
from rscvp.util.cli.cli_treadmill import TreadmillOptions
from rscvp.util.typing import SIGNAL_TYPE
from typing_extensions import Self

from argclz import AbstractParser, argument, as_argument, copy_argument
from argclz.core import as_dict
from neuralib.persistence import ETLConcatable, persistence
from neuralib.persistence.cli_persistence import get_options_and_cache

__all__ = [
    'PosBinActCache',
    'PosBinActCacheBuilder',
    'ApplyPosBinActOptions',
    'AbstractPosBinActOptions'
]


@persistence.persistence_class
class PosBinActCache(ETLConcatable):
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    plane_index: str = persistence.field(validator=False, filename=True, filename_prefix='plane')
    signal_type: SIGNAL_TYPE = persistence.field(validator=True, filename=True)
    smooth: str = persistence.field(validator=True, filename=False)
    normalized_method: NORMALIZE_TYPE = persistence.field(validator=True, filename=False)
    window: int = persistence.field(validator=True, filename=False)
    run_epoch: bool = persistence.field(validator=True, filename=False)

    #
    occ_activity: np.ndarray
    """(N, L, B) transient"""
    occ_baseline: np.ndarray
    """(N, L, B) baseline"""

    @property
    def n_neurons(self) -> int:
        return self.occ_activity.shape[0]

    @property
    def n_trials(self) -> int:
        return self.occ_activity.shape[1]

    @property
    def n_windows(self) -> int:
        return self.occ_activity.shape[2]

    @classmethod
    def concat_etl(cls, data: list[Self]) -> Self:
        if len(set([it.exp_date for it in data])) != 1:
            raise RuntimeError('')

        const = data[0]
        ret = PosBinActCache(
            exp_date=const.exp_date,
            animal=const.animal,
            plane_index='_concat',
            signal_type=const.signal_type,
            smooth=const.smooth,
            normalized_method=const.normalized_method,
            window=const.window,
            run_epoch=const.run_epoch,
        )
        ret.occ_activity = np.vstack([it.occ_activity for it in data])
        ret.occ_baseline = np.vstack([it.occ_baseline for it in data])

        return ret

    def with_mask(self, mask: np.ndarray) -> Self:
        """Cell mask selection"""
        return self._replace(
            occ_activity=self.occ_activity[mask],
            occ_baseline=self.occ_baseline[mask]
        )

    def with_trial(self, mask: np.ndarray) -> Self:
        """
        Trial selection

        :param mask: `Array[int, INDEX]`
        :return:
        """

        if mask.dtype != int:
            raise ValueError(f'invalid mask dtype: {mask.dtype}')

        return self._replace(
            occ_activity=self.occ_activity[:, mask, :],
            occ_baseline=self.occ_baseline[:, mask, :]
        )

    def _replace(self, **kwargs):
        raise RuntimeError('')

    def get_trial_avg_occ(self) -> np.ndarray:
        """
        :return (N, B)
        """
        return normalized_trial_avg(self.occ_activity)


class AbstractPosBinActOptions(TreadmillOptions, Suite2pOptions, StimpyOptions):
    binned_smooth: bool = argument(
        '--smooth',
        help='whether do the smoothing of binned signal'
    )

    signal_type: SIGNAL_TYPE = as_argument(Suite2pOptions.signal_type).with_options(...)
    """cascade spks available"""


class PosBinActCacheBuilder(AbstractParser, AbstractPosBinActOptions, PersistenceRSPOptions[PosBinActCache]):
    DESCRIPTION = 'Generate position binned occupancy-normalized calcium activity'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        print(as_dict(self))
        self.load_cache()

    def empty_cache(self) -> PosBinActCache:
        return PosBinActCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            plane_index=self.plane_index,
            signal_type=self.signal_type,
            smooth=self.binned_smooth,
            normalized_method=self.act_normalized,
            window=self.window,
            run_epoch=self.running_epoch,
        )

    def compute_cache(self, cache: PosBinActCache) -> PosBinActCache:
        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()

        ps = PositionSignal(s2p, rig, window_count=self.window,
                            signal_type=self.signal_type,
                            plane_index=self.plane_index)

        cache.occ_activity = ps.load_binned_data(
            self.act_normalized, self.running_epoch, 'transient', self.binned_smooth
        )
        cache.occ_baseline = ps.load_binned_data(
            self.act_normalized, self.running_epoch, 'baseline', self.binned_smooth
        )

        return cache


class ApplyPosBinActOptions(AbstractPosBinActOptions):
    """Apply `PosBinActOptions` after *PosBinActCache Generated*"""

    def apply_binned_act_cache(self, error_when_missing=False) -> PosBinActCache:
        if self.plane_index is not None:
            return self._apply_binned_act_cache_plane(error_when_missing)
        else:
            return self._apply_binned_act_cache_concat(error_when_missing)

    def _apply_binned_act_cache_plane(self, error_when_missing=False) -> PosBinActCache:
        return get_options_and_cache(PosBinActCacheBuilder, self, error_when_missing)

    def _apply_binned_act_cache_concat(self, error_when_missing=False) -> PosBinActCache:
        data = []
        n_planes = self.load_suite_2p().n_plane
        for i in range(n_planes):
            opt = copy_argument(PosBinActCacheBuilder(), self, plane_index=i)
            data.append(opt.load_cache(error_when_missing=error_when_missing))

        ret = PosBinActCache.concat_etl(data)
        return ret


if __name__ == '__main__':
    PosBinActCacheBuilder().main()
