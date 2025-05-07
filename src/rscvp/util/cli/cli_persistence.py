import abc
from pathlib import Path
from typing import Generic, TypeVar, get_origin, get_args

import numpy as np

from neuralib.persistence import PickleHandler, PersistenceHandler
from neuralib.persistence.cli_persistence import PersistenceOptions
from .cli_core import cast_opt, CommonOptions

__all__ = ['PersistenceRSPOptions']

T = TypeVar('T')


class PersistenceRSPOptions(PersistenceOptions, Generic[T], metaclass=abc.ABCMeta):

    @property
    def persistence_class(self) -> type[T]:
        # https://stackoverflow.com/a/50101934
        for t in type(self).__orig_bases__:
            if get_origin(t) == PersistenceRSPOptions:
                return get_args(t)[0]
        raise TypeError('unable to retrieve cache class T')

    def persistence_handler(self, dest: Path | None) -> PersistenceHandler[T]:
        if dest is None:
            dest = cast_opt(CommonOptions, self).cache_directory

        return PickleHandler(self.persistence_class, dest)

    @staticmethod
    def get_neuron_plane_idx(num: int, plane_index: int) -> np.ndarray:
        """get plane number of each neuron (N')"""
        return np.full(num, plane_index)
