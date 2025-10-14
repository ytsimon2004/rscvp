from typing import Literal

import numpy as np
from matplotlib.axes import Axes

from argclz import AbstractParser, argument
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import diag_histplot, plot_figure
from neuralib.plot.colormap import insert_colorbar
from rscvp.visual.main_response import PatternResponseOptions, VisualPatternCache, AbstractPatternResponseOptions

__all__ = ['MismatchActivityOptions']


class MismatchActivityOptions(AbstractParser, AbstractPatternResponseOptions):
    DESCRIPTION = 'plot the mismatch (nasal-temporal) or control (upper-lower direction) activity pairs'

    paired_group: Literal['mismatch', 'ctrl'] = argument(
        '--paired-group',
        default='mismatch',
        help='which group to compare: mismatch (temporal nasal) or ctrl (upper, lower)'
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        data = self.get_pair_data()

        with plot_figure(None, 1, 2, set_square=True) as ax:
            self.plot_hist(data, ax[0])
            self.plot_scatter(data, ax[1])

    _fig_kwargs = dict()

    def get_pair_data(self) -> np.ndarray:
        """(2, N)"""
        match self.paired_group:
            case 'mismatch':
                self.direction = 0
                cache = get_options_and_cache(PatternResponseOptions, self)
                dat1 = self._extract_visual_epoch(cache)
                self.direction = 180
                cache = get_options_and_cache(PatternResponseOptions, self)
                dat2 = self._extract_visual_epoch(cache)
                ret = np.vstack([dat1, dat2])

                self._fig_kwargs.setdefault('xlabel', 'N–T')
                self._fig_kwargs.setdefault('ylabel', 'T–N')

            case 'ctrl':
                self.direction = 90
                cache = get_options_and_cache(PatternResponseOptions, self)
                dat1 = self._extract_visual_epoch(cache)
                self.direction = 270
                cache = get_options_and_cache(PatternResponseOptions, self)
                dat2 = self._extract_visual_epoch(cache)
                ret = np.vstack([dat1, dat2])

                self._fig_kwargs.setdefault('xlabel', 'L-U')
                self._fig_kwargs.setdefault('ylabel', 'U–L')

            case _:
                raise ValueError(f'unknown paired group: {self.paired_group}')

        return ret

    def _extract_visual_epoch(self, cache: VisualPatternCache):
        t = cache.time
        v = cache.data
        mx = np.logical_and(t >= 0, t <= int(self.post - self.pre))
        return v[:, mx].max(axis=1)

    def plot_scatter(self, data: np.ndarray, ax: Axes):
        diag_histplot(data[0], data[1], ax=ax)
        ax.set(**self._fig_kwargs)

    def plot_hist(self, data: np.ndarray, ax: Axes, n: int = 60):
        xmin, xmax = data[0].min(), data[0].max()
        ymin, ymax = data[1].min(), data[1].max()

        data = np.histogram2d(data[0], data[1], bins=(n, n))[0]

        row_sums = np.sum(data, axis=1, keepdims=True)
        data = np.divide(data, row_sums, where=row_sums != 0, out=np.zeros_like(data))

        im = ax.imshow(
            data.T,
            cmap='gray',
            aspect='equal',
            origin='lower',
            extent=(xmin, xmax, ymin, ymax)
        )
        cbar = insert_colorbar(ax, im)
        cbar.ax.set(ylabel='probability')

        lim = [min(xmin, ymin), max(xmax, ymax)]
        ax.plot(lim, lim, 'w--', alpha=0.5, linewidth=1, zorder=10)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), **self._fig_kwargs)


if __name__ == '__main__':
    MismatchActivityOptions().main()
