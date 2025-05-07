from typing import ClassVar, TYPE_CHECKING

from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli import HistOptions

from argclz import argument, try_int_type
from neuralib.atlas.ccf.dataframe import ROIS_NORM_TYPE
from neuralib.atlas.typing import TreeLevel
from neuralib.util.verbose import fprint

if TYPE_CHECKING:
    from rscvp.atlas.core import RSCRoiClassifierDataFrame

__all__ = ['ROIOptions']


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

    def load_roi_dataframe(self, ccf_dir: AbstractCCFDir | None = None) -> 'RSCRoiClassifierDataFrame':
        from rscvp.atlas.core import RSCRoiClassifierDataFrame

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
