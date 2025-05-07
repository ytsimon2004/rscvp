from pathlib import Path
from typing import Iterable, get_args

from rscvp.util.cli import DAQ_TYPE
from rscvp.util.io import get_io_config

from neuralib.util.verbose import fprint

__all__ = ['PhysPathFinder']


class PhysPathFinder:

    def __init__(self,
                 root: Path | None = None,
                 remote_disk: str | None = None,
                 daq_type: DAQ_TYPE = '2P',
                 username: str = 'YW',
                 run_number: str | None = None):
        if root is None:
            root = get_io_config(remote_disk=remote_disk).stimpy

        self.root = root

        self._daq_type = daq_type
        self._username = username
        self._run_number = run_number

    @property
    def daq_type(self):
        return self._daq_type

    @daq_type.setter
    def daq_type(self, value: DAQ_TYPE):
        if value not in get_args(DAQ_TYPE):
            raise ValueError(f'{value}')
        self._daq_type = value

    def foreach_dir(self) -> Iterable[str]:
        """return list of ED_ID"""
        pattern = f'{self.daq_type}_{self._username}'
        target = list(self.root.glob(f'*{pattern}'))
        for it in target:
            idx = it.name.index(pattern)
            ei = it.name[:idx - 2]
            yield ei

    def find_list_animal(self) -> list[str]:
        """return set of ID"""
        animal = [it.split('_')[1] for it in self.foreach_dir()]
        return sorted(list(set(animal)))

    def find_list_exp_date(self, animal: str) -> list[str]:
        """return ED in given ID"""
        exp_date = []
        for it in self.foreach_dir():
            if animal in it:
                idx = it.index(animal)
                exp_date.append(it[:idx - 1])

        if len(exp_date) == 0:
            fprint(f'no animal{animal} were found', vtype='error')

        return sorted(exp_date)

    def find_riglog_path(self, animal: str, exp_date: str, run: str = '00') -> Path:
        pattern = f'{exp_date}_{animal}__{self.daq_type}_{self._username}'
        f = list(self.root.glob(pattern + '/*.riglog'))

        if len(f) == 1:
            return f[0]
        else:
            fsub = list(self.root.glob(pattern + f'/run{run}*'))
            f = list(fsub[0].glob('*.riglog'))
            if len(f) == 1:
                return f[0]

        fprint(f'riglog not found or more than one file in {self.root / pattern}', vtype='error')

    def find_list_riglog_path(self, animal: str, run: str = '00') -> list[Path]:
        f = list(self.root.glob(f'*{animal}*/*.riglog'))
        fsub = list(self.root.glob(f'*{animal}*/run{run}*/*.riglog'))  # for new stimpy folder structure
        f_all = f + fsub
        if len(f_all) != 0:
            return sorted(f_all)
        else:
            raise FileNotFoundError(f'{animal} data not found')
