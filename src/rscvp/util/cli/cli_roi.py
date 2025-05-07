from __future__ import annotations

from typing import ClassVar, TypedDict

import polars as pl

from argclz import argument, try_int_type
from neuralib.atlas.ccf import RoiClassifierDataFrame
from neuralib.atlas.ccf.dataframe import ROIS_NORM_TYPE
from neuralib.atlas.typing import TreeLevel, Area, Channel, Source
from neuralib.util.verbose import fprint
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli import HistOptions

__all__ = ['ROIOptions',
           'RSCRoiClassifierDataFrame']


class ROIOptions(HistOptions):
    GROUP_ROI: ClassVar[str] = 'ROI Options'

    top_area: int | None = argument(
        '--top', '--top-area',
        group=GROUP_ROI,
        default=None,
        help='top value would like to show in the plot',
    )

    merge_level: TreeLevel | str | None = argument(
        '--level', '--lv',
        type=try_int_type,
        group=GROUP_ROI,
        default=1,
        help='level of acronym merge',
    )

    roi_norm: ROIS_NORM_TYPE = argument(
        '--norm',
        group=GROUP_ROI,
        default='none',
        help='roi normalization method'
    )

    separate_overlap: bool = argument(
        '--separate', '--separate-overlap',
        group=GROUP_ROI,
        help='if separate the overlap channel counts from other sources'
    )

    disable_overlap_in_plot: bool = argument(
        '--disable-overlap-plot',
        group=GROUP_ROI,
        help='whether disable overlap in the plotting'
    )

    invalid_post_processing: bool = argument(
        '--invalid',
        group=GROUP_ROI,
        help='invalid post processing cache for the roi csv'
    )

    def load_roi_dataframe(self, ccf_dir: AbstractCCFDir | None = None) -> RSCRoiClassifierDataFrame:

        if ccf_dir is None:
            ccf_dir = self.get_ccf_dir()

        df = RSCRoiClassifierDataFrame(ccf_dir, invalid_post_processing_cache=self.invalid_post_processing)

        return df.post_processing(
            filter_injection=(df.config['area'], 'ipsi'),
            copy_overlap=self.supply_overlap,
        )

    @property
    def classified_column(self) -> str:
        field = self.merge_level
        match field:
            case int():
                col = f'tree_{field}'
            case str():
                col = field
            case None:
                col = 'acronym'
            case _:
                raise TypeError(f'{type(field)}')

        return col

    @property
    def supply_overlap(self) -> bool:
        if self.separate_overlap:
            fprint('Separated sources from overlap channel!', vtype='warning')
        return not self.separate_overlap


class UserInjectionConfig(TypedDict):
    area: Area
    """injection area"""
    fluor_repr: dict[Channel, Source]
    """fluorescence color and tracing source alias pairs"""


class RSCRoiClassifierDataFrame(RoiClassifierDataFrame):
    DEFAULT_INVERT_ANIMALS: ClassVar[tuple[str, ...]] = ('YW063', 'YW064')
    """Swap viral vectors (avoid bias)"""
    DEFAULT_SAGITTAL_ANIMALS: ClassVar[tuple[str, ...]] = ('YW053',)
    """Sagittal slice animal"""

    def __init__(self, ccf_dir: AbstractCCFDir, invalid_post_processing_cache: bool = False):
        cached_dir = ccf_dir.parsed_data_folder
        self.config = self._get_user_config(ccf_dir)

        if not ccf_dir.parse_csv.exists() or invalid_post_processing_cache:
            files = ccf_dir.labelled_roi_folder.glob('*.csv')
            df = pl.concat([self._with_valid_columns(file, self.config) for file in files])
            super().__init__(df, cached_dir=cached_dir,
                             invalid_post_processing_cache=invalid_post_processing_cache)
        else:
            super().__init__(pl.read_csv(ccf_dir.parse_csv), cached_dir=cached_dir,
                             invalid_post_processing_cache=invalid_post_processing_cache)

    @staticmethod
    def _get_user_config(ccf_dir: AbstractCCFDir) -> UserInjectionConfig:
        if ccf_dir.animal in RSCRoiClassifierDataFrame.DEFAULT_INVERT_ANIMALS:
            return UserInjectionConfig(area='RSP', fluor_repr=dict(rfp='aRSC', gfp='pRSC', overlap='overlap'))
        else:
            return UserInjectionConfig(area='RSP', fluor_repr=dict(rfp='pRSC', gfp='aRSC', overlap='overlap'))

    @staticmethod
    def _with_valid_columns(file, config) -> pl.DataFrame:
        channel = file.stem.split('_')[1]
        source = config['fluor_repr'][channel]
        df = (pl.read_csv(file)
              .with_columns(pl.lit(channel).alias('channel'))
              .with_columns(pl.lit(source).alias('source')))
        return df
