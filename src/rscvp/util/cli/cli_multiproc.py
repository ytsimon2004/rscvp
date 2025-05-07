from typing import ClassVar

import joblib
import polars as pl

from argclz import argument
from neuralib.util.verbose import fprint
from .cli_output import DataOutput

__all__ = ['MultiProcOptions']


class MultiProcOptions:
    GROUP_MP: ClassVar[str] = 'Multiprocessing CPU Option'

    CPU_COUNT: ClassVar[int] = joblib.cpu_count()

    f_jobs: float = argument(
        '-J', '--job',
        type=float,
        default=0.5,
        validator=lambda it: 0 < it < 1,
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
                n_jobs = int(self.CPU_COUNT * self.f_jobs)
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

        df = pl.concat(ret).sort('neuron_id')
        df.write_csv(output.csv_output)
