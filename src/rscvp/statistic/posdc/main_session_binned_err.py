import collections

import numpy as np
import polars as pl

from argclz import AbstractParser
from neuralib.plot import plot_figure
from neuralib.typing import PathLike
from neuralib.util.verbose import print_load
from rscvp.util.cli import CommonOptions, TreadmillOptions
from stimpyp import Session

__all__ = ['SessionBinnedErr']


class SessionBinnedErr(AbstractParser, CommonOptions, TreadmillOptions):
    DESCRIPTION = 'Plot binned decoding error in different behavioral sessions from batch CV dataset'

    def run(self):
        self.plot_grid_binned_err()

    def plot_grid_binned_err(self, images_per_row: int = 3):
        dats = self.compute_foreach_session()

        color_dict = {
            'light': 'k',
            'dark': 'red',
            'light_end': 'gray'
        }

        n = len(dats)
        n_rows = np.ceil(n / images_per_row).astype(int)
        n_cols = min(images_per_row, n)
        with plot_figure(None, n_rows, n_cols, sharey=True) as _ax:
            for i, (file, dat) in enumerate(dats.items()):

                if n > images_per_row:
                    r, c = divmod(i, images_per_row)
                    ax = _ax[r, c]
                else:
                    ax = _ax[i]

                for s, d in dat.items():
                    x = np.linspace(0, self.track_length, len(d))
                    ax.plot(x, d, label=s, color=color_dict[s])

                ax.set_title(file)
                if i == 0:  # clean
                    ax.set(xlabel='position(cm)', ylabel='error(cm)')
                    ax.legend()

    def compute_foreach_session(self) -> dict[str, dict[Session, np.ndarray]]:
        """
        :return: {FILENAME: {SESSION: `Array[float, B]`}}
        """
        ret = {}
        for _ in self.foreach_dataset():
            caches = (self.cache_directory / 'posdc').glob('*.parquet')

            # loop plane
            _ret: dict[Session, list[np.ndarray]] = collections.defaultdict(list)
            for file in caches:
                # loop cv
                for s, val in get_cv_results(file).items():
                    _ret[s].append(val)

            # avg cross plane
            dat = {s: np.mean(vals, axis=0) for s, vals in _ret.items()}

            ret[self.stimpy_filename] = dat

        return ret


def get_cv_results(file: PathLike) -> dict[Session, np.ndarray]:
    """
    Compute the cv mean results

    :param file: parquet file from posdc result
    :return: {Session: `Array[float, B]`}
    """
    print_load(file)
    df = pl.read_parquet(file)
    return {
        s[0]: np.mean(val['binned_err'].to_numpy(), axis=0)
        for s, val in df.select('session', 'binned_err').partition_by('session', as_dict=True).items()
    }


if __name__ == '__main__':
    SessionBinnedErr().main()
