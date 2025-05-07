from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import scipy
from attrs import field, define, evolve, asdict

from neuralib.typing import DataFrame

__all__ = ['GROUP_HEADER_TYPE',
           'DataSetType',
           #
           'get_stat_group_vars',
           'CollectDataSet']

# =============== #
# Collect DataSet #
# =============== #

GROUP_HEADER_TYPE = Literal['region', 'session']

IndVar = str
"""C | CxG"""

DataSetType = dict[IndVar, np.ndarray]


@define
class CollectDataSet:
    name: list[str]  # accept categorical comparison
    """value name"""

    data: DataSetType = field(repr=False)
    """group: {G: D}; categorical: {CxG: D}"""

    group_header: GROUP_HEADER_TYPE
    """header for independent variable. i.e., region, session"""

    test_type: str | None = field(default=None)
    """statistic test type. assign after run the test"""

    # ========= #
    # post_init #
    # ========= #

    group_vars: list[IndVar] = field(init=False)
    """independent variable. i.e., aRSC, pRSC or aRSC_*, pRSC_*"""

    group_mean_value: dict[IndVar, float] = field(init=False)
    """value mean with given group_vars"""

    group_sem_value: dict[IndVar, float] = field(init=False)
    """value sem with given group_vars"""

    n_categories: int = field(kw_only=True, default=1)
    """number of categories. nC"""

    n_groups: int = field(init=False)
    """number of group. nG"""

    n_datapoints: dict[IndVar, int] = field(init=False)
    """number of data value. nD"""

    data_type: Literal['group', 'categorical'] = field(kw_only=True, default='group')

    def __attrs_post_init__(self):
        self.group_vars = get_stat_group_vars(self.data, self.group_header)

        self.group_mean_value = {
            g: float(np.mean(self.data[g]))
            for g in self.group_vars
        }

        self.group_sem_value = {
            g: scipy.stats.sem(self.data[g])
            for g in self.group_vars
        }
        #
        self.n_groups = len(self.group_vars)
        self.n_datapoints = {k: len(v) for k, v in self.data.items()}

    def __getitem__(self, i: int | str | tuple[str, str]) -> np.ndarray:
        """

        :param i: key, G, (G, C), or 'G_C'
        :return:
        """
        if self.data_type == 'group':
            if isinstance(i, int):
                return self.data[self.group_vars[i]]
            elif isinstance(i, str):
                return self.data[i]
            else:
                raise TypeError('')

        elif self.data_type == 'categorical':
            if not isinstance(i, tuple):
                raise TypeError(f'categorical should be getitem from tuple type: {type(i)}')

            return self.data[f'{i[0]}_{i[1]}']

        else:
            raise ValueError(f'invalid data type: {self.data_type}')

    @property
    def n_datasets(self) -> int:
        return len(self.data)

    @property
    def is_two_samples(self) -> bool:
        return self.n_datasets == 2

    @property
    def is_multisample(self) -> bool:
        return self.n_datasets > 2 or self.data_type == 'categorical'

    def append(self, data: CollectDataSet) -> CollectDataSet:
        """append other same header CollectDataSet.

        :param data:
        :return:
        """
        if self.group_header != data.group_header:
            raise RuntimeError()

        if self.group_vars != data.group_vars:
            raise RuntimeError()

        if self.data_type != data.data_type:
            raise RuntimeError()

        ret = {}
        for i in self.data:
            for j in data.data:
                if i == j:
                    values = np.concatenate([self[i], data[i]])
                    ret[i] = values
                else:
                    ret[i] = ret[i]

        return CollectDataSet(
            self.name,
            ret,
            self.group_header,
        )

    def update(self, data: CollectDataSet,
               extend_name: bool = True) -> CollectDataSet:
        """Update other header from another CollectDataSet.

        :param data:
        :param extend_name: for categorical data comparison, dataset name extend
        :return:
        """
        #
        if extend_name:
            self.name.extend(data.name)
        else:
            if self.name != data.name:
                raise RuntimeError(f'{self.name} != {data.name}')

        if self.group_header != data.group_header:
            raise RuntimeError('')

        if self.data_type != data.data_type:
            raise RuntimeError(f'{self.data_type}, {data.data_type}')

        #
        ret = dict(self.data)
        ret.update(data.data)

        return CollectDataSet(
            self.name,
            ret,
            self.group_header,
        )

    def with_selection(self, mask: dict[IndVar, np.ndarray]) -> CollectDataSet:
        """

        :param mask: {key: array[bool]}
        :return:
        """
        _data = {
            group: self[group][tuple(_mask)]
            for group, _mask in mask.items()
        }

        return evolve(self, data=_data)

    def to_pickle(self, output: Path):
        import pickle
        with open(output, 'wb') as f:
            pickle.dump(asdict(self), f)

    def to_polars(self, melt: bool = False) -> pl.DataFrame:
        """
        Convert data(DataSetType) to polars dataframe

        :param melt: melt foreach ``group_header`` (i.e., session, region., etc.)
        :return:
        """
        df = pl.from_dict(self.data)

        if melt:
            name = self.name
            if len(name) != 1:
                raise RuntimeError(f'invalid name list length: {name}')
            df = df.unpivot(on=df.columns, variable_name=self.group_header, value_name=name[0])

        return df


# ============ #

def get_stat_group_vars(data: DataFrame | dict[str, np.ndarray],
                        group_header: GROUP_HEADER_TYPE) -> list[str]:
    """
    Get list of independent variables based on group header. i.e., aRSC, pRSC

    :param data:
    :param group_header: field of dataframe, or key of the dict. i.e., region
    :return:
    """
    if isinstance(data, pd.DataFrame):
        varz = list(data[group_header].unique().astype(str))

    elif isinstance(data, pl.DataFrame):
        varz = data[group_header].unique().to_list()

    elif isinstance(data, dict):
        varz = np.unique(list(data.keys())).tolist()
    else:
        raise TypeError('')

    #
    def sort_func(item: str) -> tuple | None:

        if group_header == 'region':
            seqs = ['aRSC', 'pRSC']
        elif group_header == 'session':
            seqs = ['light_bas', 'dark', 'light_end']
        else:
            return None

        for string in seqs:
            if string in item:
                return string, item
        return (item,)

    return sorted(varz, key=sort_func)
