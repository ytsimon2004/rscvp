import json
from pathlib import Path
from typing import Any

import numpy as np

from neuralib.typing import PathLike

__all__ = ['PixVizResult']


class PixVizResult:
    meta: dict[str, Any]
    dat: np.ndarray

    __slots__ = ('meta', 'dat')

    def __init__(self,
                 dat: Path | str,
                 meta: Path | str):
        """

        :param dat: .npy or .mat data path
        :param meta: .json data path
        """
        if Path(dat).suffix == '.npy':
            self.dat = np.load(dat)
        elif Path(dat).suffix == '.mat':
            raise NotImplementedError('')
        else:
            raise ValueError('')
        #
        with open(meta, 'r') as file:
            self.meta = json.load(file)

    @classmethod
    def load(cls,
             dat: PathLike,
             meta: PathLike):
        return cls(dat, meta)

    def __repr__(self):
        class_name = self.__class__.__name__
        roi_reprs = ", ".join(
            [f"{roi['name']} (index {roi['index']})" for roi in self.meta.values()]
        )
        return f"<{class_name}: [{roi_reprs}]>"

    __str__ = __repr__

    def get_index(self, name: str) -> int:
        """
        Get `index` from `roi name`

        :param name: roi name
        :return: roi index
        """
        return self.meta[name]['index']

    def get_data(self, source: int | str) -> np.ndarray:
        """
        Get roi data from either `index` or `name`

        :param source: if int type, get data from index;
            If string type, get data from roi name
        :return: data (F,)
        """
        if isinstance(source, int):
            pass
        elif isinstance(source, str):
            source = self.get_index(source)
        else:
            raise TypeError(f'invalid type: {type(source)}')

        return self.dat[source]

    def __getitem__(self, index: int) -> str:
        """
        Get `roi name` from index

        :param index: 0-based index
        :return: roi name
        """
        for name, roi in self.meta.items():
            if roi['index'] == index:
                return name

        raise IndexError(f'{index}')

    @property
    def n_rois(self) -> int:
        """number of roi selected"""
        return self.dat.shape[0]

    @property
    def n_frames(self) -> int:
        """number of frames (sequences)"""
        return self.dat.shape[1]
