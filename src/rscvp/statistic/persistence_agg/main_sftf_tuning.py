from typing import Final

import numpy as np
from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg
from rscvp.visual.main_sftf_map import ApplyVisualMapOptions, VisualSFTFMapCache

from argclz import try_int_type
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from neuralib.util.verbose import publish_annotation

__all__ = ['SFTFMapPersistenceAgg']


@publish_annotation('sup')
class SFTFMapPersistenceAgg(AbstractPersistenceAgg, ApplyVisualMapOptions):
    DESCRIPTION = 'Use SFTF visual map persistence to plot batch heatmap'

    field: Final[dict] = dict(plane_index=try_int_type, region=str)
    used_session = 'light'

    def run(self):
        caches = self.get_cache_list()
        dat = self.get_cache_data(caches)
        self.plot(dat)

    def get_cache_list(self) -> list[VisualSFTFMapCache]:
        ret = []
        for i, _ in enumerate(self.foreach_dataset(**self.field)):
            self.exp_list.append(self.exp_date)
            self.animal_list.append(self.animal_id)
            ret.append(self.apply_visual_map_cache())

        return ret

    def get_cache_data(self, cache_list: list[VisualSFTFMapCache]) -> np.ndarray:
        dat = []
        for cache in cache_list:
            dat.append(cache.dat)

        return np.vstack(dat)

    def plot(self, data: np.ndarray):
        """
        resort based on mean activity per neurons and plot as heatmap

        :param data: concatenated visual on-off binned neuronal activity data. (N, B): (n_neurons, n_bins)
        :return:
        """
        index = np.argsort(np.mean(data, axis=1))
        data = data[index]

        with plot_figure(None, tight_layout=False) as ax:
            im = ax.imshow(data, cmap='Reds', vmin=0, interpolation='none')
            cbar = insert_colorbar(ax, im)
            cbar.ax.set(ylabel='Norm. dF/F')


if __name__ == '__main__':
    SFTFMapPersistenceAgg().main()
