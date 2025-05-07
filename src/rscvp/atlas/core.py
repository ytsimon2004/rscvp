from typing import ClassVar, TypedDict

import polars as pl
from rscvp.atlas.dir import AbstractCCFDir

from neuralib.atlas.ccf.dataframe import RoiClassifierDataFrame
from neuralib.atlas.typing import Area, Source, Channel

__all__ = ['RSCRoiClassifierDataFrame']


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
