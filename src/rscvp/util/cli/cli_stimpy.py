from pathlib import Path
from typing import ClassVar

from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

from argclz import argument
from neuralib.util.verbose import fprint
from stimpyp import (
    STIMPY_SOURCE_VERSION,
    Session,
    PyVlog,
    RiglogData,
    get_protocol_name
)
from .cli_core import CommonOptions

__all__ = ['StimpyOptions']


class StimpyOptions(CommonOptions):
    GROUP_STIMPY: ClassVar[str] = 'Stimpy Options'

    session: Session | None = argument(
        '-s', '--session',
        metavar='NAME',
        group=GROUP_STIMPY,
        help='specify the which session',
    )

    no_diode_offset: bool = argument(
        '--no-offset',
        group=GROUP_STIMPY,
        help='not to do the time offset between diode (riglog) and stimlog',
    )

    source_version: STIMPY_SOURCE_VERSION = argument(
        '--version',
        default='stimpy-bit',
        group=GROUP_STIMPY,
        help='code version'
    )

    @property
    def stimpy_directory(self) -> Path:
        d = self.get_src_path('stimpy')
        ret = d / self.filename
        if not ret.exists():
            ret = d / f'{self._legacy_filename}'
            if ret.exists():
                fprint(f'legacy filename found: {ret}', vtype='warning')
            else:
                raise FileNotFoundError(f'cannot find {ret}')
        return ret

    @property
    def with_diode_offset(self) -> bool:
        return not self.no_diode_offset

    def load_riglog_data(self, **kwargs) -> RiglogData | PyVlog:
        if self.source_version == 'pyvstim':
            return PyVlog(self.stimpy_directory, **kwargs)
        else:
            return RiglogData(self.stimpy_directory, diode_offset=self.with_diode_offset, **kwargs)

    # ================== #
    # Protocol Dependent #
    # ================== #

    @property
    def session_list(self) -> list[Session]:
        if self.is_vop_protocol():
            return ['light', 'visual', 'dark']
        elif self.is_ldl_protocol():
            return ['light_bas', 'dark', 'light_end']
        else:
            raise RuntimeError('unsupported protocol')

    @property
    def gspread_reference(self) -> GSPREAD_SHEET_PAGE:
        if self.is_vop_protocol():
            return 'apcls_tac'
        elif self.is_ldl_protocol():
            return 'ap_ldl'
        else:
            raise ValueError('unsupported protocol')

    def get_protocol_name(self) -> str:
        return get_protocol_name(self.load_riglog_data().stimlog_file)

    def is_ldl_protocol(self) -> bool:
        return self.get_protocol_name() == 'light_dark_light'

    def is_vop_protocol(self) -> bool:
        return self.get_protocol_name() == 'visual_open_loop'
