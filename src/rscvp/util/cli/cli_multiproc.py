from typing import ClassVar

import joblib
import polars as pl

from argclz import argument, validator
from neuralib.util.verbose import fprint
from .cli_output import DataOutput

__all__ = ['MultiProcOptions']


class MultiProcOptions:
    """Multiprocessing CPU options for parallel computation"""

    GROUP_MP: ClassVar[str] = 'Multiprocessing CPU Option'
    """Multiprocessing CPU Option"""

    f_jobs: float = argument(
        '-J', '--job',
        validator.float.in_range_closed(0, 1),
        type=float,
        default=0.5,
        group=GROUP_MP,
        help='fraction of CPU for parallel computations'
    )

    @property
    def parallel_jobs(self) -> int:
        if self.f_jobs == 0:
            n_jobs = 1
        else:
            if not 0 < self.f_jobs <= 1:
                raise ValueError('')
            else:
                n_jobs = int(joblib.cpu_count() * self.f_jobs)
        fprint(f'MultiProcess on {n_jobs} CPUs!', vtype='io', flush=True)

        return n_jobs

    @staticmethod
    def aggregate_output_csv(output: DataOutput, pattern: str = 'tmp'):
        """aggregate the `tmp csv` to a single csv, then delete the rest"""
        ret = []
        files = list(output.directory.glob(f'*{pattern}*.csv'))
        for csv in files:
            ret.append(pl.read_csv(csv))
            csv.unlink()

        if len(ret) == 0:
            fprint('empty file of csv aggregate', vtype='warning')
            return

        df = pl.concat(ret).sort('neuron_id')
        df.write_csv(output.csv_output)
