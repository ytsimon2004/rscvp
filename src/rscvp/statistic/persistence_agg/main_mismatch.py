from typing import Final, Literal

import numpy as np

from argclz import try_int_type, argument
from neuralib.plot import diag_histplot, plot_figure
from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg
from rscvp.util.cli import Region
from rscvp.visual.main_response import VisualPatternCache, ApplyPatternResponseCache
from rscvp.visual.util_plot import mismatch_hist

__all__ = ['MismatchPersistenceAgg']


class MismatchPersistenceAgg(AbstractPersistenceAgg, ApplyPatternResponseCache):
    DESCRIPTION = 'plot the mismatch (nasal-temporal) or control (upper-lower direction) activity pairs in batch data'

    paired_group: Literal['mismatch', 'ctrl'] = argument(
        '--paired-group',
        default='mismatch',
        help='which group to compare: mismatch (temporal nasal) or ctrl (upper, lower)'
    )

    field: Final = dict(plane_index=try_int_type, region=str)

    def run(self):
        data = self.concat_data()
        self.plot(data)

    def get_cache_list(self) -> list[VisualPatternCache]:
        ret = []
        for i, _ in enumerate(self.foreach_dataset(**self.field)):
            self.exp_list.append(self.exp_date)
            self.animal_list.append(self.animal_id)
            ret.append(self.get_pattern_cache())

        return ret

    def get_cache_data(self, cache_list: list[VisualPatternCache]) -> np.ndarray:
        """(N',)"""
        ret = []
        for cache in cache_list:
            t = cache.time
            mx = np.logical_and(t >= 0, t <= int(self.post - self.pre))
            ret.extend(cache.data[:, mx].max(axis=1))

        return np.array(ret)

    _fig_kwargs = dict()

    def concat_data(self) -> np.ndarray:
        if self.paired_group == 'mismatch':
            self.direction = 0
            caches = self.get_cache_list()
            x = self.get_cache_data(caches)

            self.direction = 180
            caches = self.get_cache_list()
            y = self.get_cache_data(caches)

            self._fig_kwargs.setdefault('xlabel', 'N–T')
            self._fig_kwargs.setdefault('ylabel', 'T–N')

            return np.vstack([x, y])
        elif self.paired_group == 'ctrl':
            self.direction = 90
            caches = self.get_cache_list()
            x = self.get_cache_data(caches)

            self.direction = 270
            caches = self.get_cache_list()
            y = self.get_cache_data(caches)

            self._fig_kwargs.setdefault('xlabel', 'L-U')
            self._fig_kwargs.setdefault('ylabel', 'U–L')

            return np.vstack([x, y])
        else:
            raise ValueError(f'unknown paired group: {self.paired_group}')

    def plot(self, data: list[np.ndarray] | dict[Region, np.ndarray] | np.ndarray):
        with plot_figure(None, 1, 2, set_square=True) as ax:
            diag_histplot(data[0], data[1], ax=ax[0])
            ax[0].set(**self._fig_kwargs)
            mismatch_hist(data, ax=ax[1], **self._fig_kwargs)


if __name__ == '__main__':
    MismatchPersistenceAgg().main()
