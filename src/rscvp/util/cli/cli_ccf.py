from pathlib import Path
from typing import ClassVar

from argclz import argument, str_tuple_type
from .cli_hist import HistOptions

__all__ = ['CCFOptions']


class CCFOptions(HistOptions):
    GROUP_CCF: ClassVar = 'CCF transform Options'

    raw_image: Path = argument('-R', '--raw', group=GROUP_CCF, help='raw image path')
    trans_matrix: Path = argument('-T', '--trans', group=GROUP_CCF, help='transform matrix path (3 x 3)')
    ccf_matrix: Path = argument('--ccf', group=GROUP_CCF, help='ccf matrix .mat file')

    annotation_region: tuple[str, ...] = argument(
        '--annotation',
        type=str_tuple_type,
        default=('LD',),
        group=GROUP_CCF,
        help='annotation brain region'
    )
