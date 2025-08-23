from typing import NamedTuple, Final

import numpy as np
from typing_extensions import Self

from argclz import try_int_type
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_align_peak import ApplyAlignPeakOptions, SISortAlignPeakCache
from rscvp.spatial.util_plot import plot_alignment_map
from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg
from rscvp.util.cli.cli_treadmill import TreadmillOptions

__all__ = ['SISortAlignPersistenceAgg']


class SIOccData(NamedTuple):
    """
    `Dimension parameters`:

        N = number of neurons

        B = number of position bins

    """
    spatial_info: np.ndarray
    """Spatial information. `Array[float, N]`"""

    sig: np.ndarray
    """Signal all selected neurons. `Array[float, [N, B]]`"""

    def reorder(self) -> Self:
        idx = np.argsort(self.spatial_info)[::-1]
        return self._replace(
            spatial_info=self.spatial_info[idx],
            sig=self.sig[idx]
        )


@publish_annotation('test', caption='visualization only')
class SISortAlignPersistenceAgg(AbstractPersistenceAgg, ApplyAlignPeakOptions, TreadmillOptions):
    DESCRIPTION = """
    Align the position-binned data (N, B) with the peak response in batch dataset,
    and sorted by the spatial information
    """

    field: Final[dict] = dict(plane_index=try_int_type, region=str)

    session = 'light'
    used_session = 'light'  # for query the spatial info

    def run(self):
        caches = self.get_cache_list()
        dat = self.get_cache_data(caches)
        self.plot(dat)

    def get_cache_list(self) -> list[SISortAlignPeakCache]:
        ret = []
        for i, _ in enumerate(self.foreach_dataset(**self.field)):
            self.exp_list.append(self.exp_date)
            self.animal_list.append(self.animal_id)
            ret.append(self.apply_align_peak_cache())

        return ret

    def get_cache_data(self, cache_list: list[SISortAlignPeakCache]) -> SIOccData:
        si = []
        sig = []
        for cache in cache_list:
            si.append(cache.spatial_info)
            sig.append(cache.trial_avg_binned_data)

        return SIOccData(np.concatenate(si), np.vstack(sig)).reorder()

    def plot(self, data: SIOccData):
        with plot_figure(None, 2, 1, tight_layout=False) as axes:
            plot_alignment_map(
                data.sig,
                self.signal_type,
                track_length=self.track_length,
                select_top=self.with_top,
                interpolation='antialiased',
                axes=axes
            )


if __name__ == '__main__':
    SISortAlignPersistenceAgg().main()
