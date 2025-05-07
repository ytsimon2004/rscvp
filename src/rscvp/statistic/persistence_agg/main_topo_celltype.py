import collections

import numpy as np

from argclz import try_int_type, argument
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure
from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg, data_region_dict
from rscvp.topology.cache_ctype_cord import CellTypeCordCache, CellTypeCordCacheBuilder
from rscvp.util.cli import Region
from rscvp.util.cli.cli_celltype import CellTypeSelectionOptions
from rscvp.util.cli.cli_sbx import SBXOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

__all__ = ['TopoCellTypePersistenceAgg']


class TopoCellTypePersistenceAgg(AbstractPersistenceAgg, CellTypeSelectionOptions, SBXOptions):
    DESCRIPTION = 'Cell type fraction along the anterior-posterior axis'

    smooth_kernel: float | None = argument('--smooth', default=None, help='kernel size for filter histogram')

    field = dict(plane_index=try_int_type)

    def run(self):
        cache = self.get_regions_cache()
        data = self.get_cache_data(cache)
        self.plot(data)

    def get_regions_cache(self) -> dict[Region, list[CellTypeCordCache]]:
        ret = collections.defaultdict(list)

        dy = data_region_dict('apcls_tac')
        for _ in self.foreach_dataset(**self.field):
            name = f'{self.exp_date}_{self.animal_id}'
            region = dy[name]
            ret[region].append(get_options_and_cache(CellTypeCordCacheBuilder, self, error_when_missing=True))

        return ret

    def get_cache_data(self, cache_list: dict[Region, list[CellTypeCordCache]]) -> dict[Region, list[np.ndarray]]:
        """Return paired region data (2, N), with ap, and fraction"""
        if self.cell_type == 'visual':
            attr = 'ap_hist_visual'
        elif self.cell_type == 'spatial':
            attr = 'ap_hist_place'
        else:
            raise ValueError(f'{self.cell_type} unsupported!')

        #
        ret = {}
        for r, caches in cache_list.items():
            agg = []
            for cache in caches:
                frac = getattr(cache, attr)

                if self.smooth_kernel is not None:
                    from scipy.ndimage import gaussian_filter
                    frac = gaussian_filter(frac, self.smooth_kernel)  # note value sensitive

                ap = cache.ap_bins_place[1:]
                agg.append(np.vstack([ap, frac]))

            ret[r] = agg
        return ret

    def plot(self, data: dict[Region, list[np.ndarray]]):
        with plot_figure(None) as ax:
            for r, d in data.items():
                for dd in d:
                    ax.plot(dd[0], dd[1], color=REGION_COLORS_HIST[r])

            ax.set(xlabel='ap(mm)', ylabel='fraction')


if __name__ == '__main__':
    TopoCellTypePersistenceAgg().main()
