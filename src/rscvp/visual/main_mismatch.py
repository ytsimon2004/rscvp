from typing import Literal

import numpy as np

from argclz import AbstractParser, argument
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import diag_histplot, plot_figure
from neuralib.util.deprecation import deprecated_class
from rscvp.visual.main_response import PatternResponseOptions, VisualPatternCache, AbstractPatternResponseOptions
from rscvp.visual.util_plot import mismatch_hist

__all__ = ['MismatchActivityOptions']


@deprecated_class(new='rscvp.statistic.persistence_agg.main_mismatch')
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
        output = self.get_data_output('mm').summary_figure_output(self.paired_group)

        with plot_figure(output, 1, 2, set_square=True) as ax:
            diag_histplot(data[0], data[1], ax=ax[0])
            ax[0].set(**self._fig_kwargs)
            mismatch_hist(data, ax=ax[1], **self._fig_kwargs)

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


if __name__ == '__main__':
    MismatchActivityOptions().main()
