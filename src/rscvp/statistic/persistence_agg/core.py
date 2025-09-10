import abc
from functools import cached_property
from typing import TypeVar, overload, Generic

import numpy as np
import polars as pl

from argclz import AbstractParser, argument, int_tuple_type
from rscvp.util.cli import Region, CommonOptions
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

__all__ = [
    'GroupInt',
    'GroupName',
    'AbstractPersistenceAgg',
    'data_region_dict'
]

C = TypeVar('C')
GroupInt = int
GroupName = str


class AbstractPersistenceAgg(AbstractParser, CommonOptions, Generic[C], metaclass=abc.ABCMeta):
    """Pipeline for aggregating population persistence/cache dataset (cache) and do the plot/statistic"""

    group_mode: bool = argument('--group', help='run as manual grouping mode')
    assign_group: tuple[int, ...] = argument('--as-group', type=int_tuple_type, help='assign int group')

    #
    exp_list: list[str] = []
    animal_list: list[str] = []
    field: dict | None = None

    GROUP_REPR: dict[GroupInt, GroupName] = {
        0: 'aRSC',
        1: 'pRSC',
        2: 'light',
        3: 'dark'
    }

    def post_parsing(self):
        self.get_io_config()
        if self.field is None:
            raise ValueError(f'specify fields for setting the data_identity')

        self._clean_data_identity()
        self._set_data_identity()

        if self.group_mode:
            n_groups = len(self.assign_group)
            n_animals = len(self.animal_list)
            assert n_groups == n_animals, f'grouping & data inconsistent: {n_groups} != {n_animals}'

    def _clean_data_identity(self):
        self.exp_list = []
        self.animal_list = []

    def _set_data_identity(self):
        for i, _ in enumerate(self.foreach_dataset(**self.field)):
            self.exp_list.append(self.exp_date)
            self.animal_list.append(self.animal_id)

    @property
    def data_identity(self) -> list[str]:
        return [f'{exp}_{animal}' for exp, animal in zip(self.exp_list, self.animal_list)]

    # ======== #
    # Pipeline #
    # ======== #

    def get_cache_list(self) -> list[C]:
        """get list of cache"""
        pass

    @overload
    def get_cache_data(self, cache_list: list[C]) -> list[np.ndarray]:
        pass

    @overload
    def get_cache_data(self, cache_list: dict[Region, list[C]]) -> dict[Region, np.ndarray]:
        pass

    @abc.abstractmethod
    def get_cache_data(self, cache_list):
        """get list of input data from cache for plotting

        **
        normal mode: aggregate list of persistence caches and create list of ndarray
        region mode: aggregate region dict of persistence cache and create list of ndarray
        """
        pass

    @abc.abstractmethod
    def plot(self, data: list[np.ndarray] | dict[Region, np.ndarray] | np.ndarray):
        pass

    # ========== #
    # Region Agg #
    # ========== #

    def get_regions_cache(self) -> dict[Region, list[C]]:
        """get list of cache corresponding to different regions"""
        pass

    # ================== #
    # Manual Groups Mode #
    # ================== #

    @cached_property
    def unique_groups(self) -> list[GroupName]:
        """ Get unique group name from :attr:`.data_grouping`.

        Be careful:
        unique() causes different orders result"""
        return self.data_grouping['group'].unique().to_list()

    @cached_property
    def data_grouping(self) -> pl.DataFrame:
        """
        cached the property to avoid changed for duplicated get from caller
        ::

            ┌───────────────┬───────┐
            │ data_identity ┆ group │
            │ ---           ┆ ---   │
            │ str           ┆ str   │
            ╞═══════════════╪═══════╡
            │ 210315_YW006  ┆ aRSC  │
            │ 210402_YW006  ┆ pRSC  │
            └───────────────┴───────┘
        """
        ret = pl.DataFrame(
            {'data_identity': self.data_identity,
             'group': [self.GROUP_REPR.get(i, None) for i in self.assign_group]}
        )
        return ret


# ======= #

def data_region_dict(page: GSPREAD_SHEET_PAGE) -> dict[str, Region]:
    from rscvp.statistic.cli_gspread import GSPExtractor
    df = GSPExtractor(page, ['Data', 'region']).load_from_gspread(verbose=False)

    return {d: r for d, r in df.iter_rows()}
