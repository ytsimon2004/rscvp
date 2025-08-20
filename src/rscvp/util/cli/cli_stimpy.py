from typing import ClassVar

import numpy as np

from argclz import argument
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE
from stimpyp import (
    STIMPY_SOURCE_VERSION,
    Session,
    PyVlog,
    RiglogData,
    get_protocol_name, ProtocolAlias, SessionInfo
)
from .cli_core import CommonOptions

__all__ = ['StimpyOptions']


class StimpyOptions(CommonOptions):
    """Stimpy options"""

    GROUP_STIMPY: ClassVar[str] = 'Stimpy Options'
    """group stimpy options"""

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

    virtual_env: bool = argument(
        '--vr',
        group=GROUP_STIMPY,
        help='if the experiment is performed in VR environment or use virtual position space'
    )


    @property
    def with_diode_offset(self) -> bool:
        """if do the diode offset synchronization"""
        return not self.no_diode_offset

    def load_riglog_data(self, **kwargs) -> RiglogData | PyVlog:
        """Load :class:`~stimpyp.stimpy_core.RiglogData` or :class:`~stimpyp.pyvstim.PyVlog`
         object based on the source version"""
        log_directory = self.get_src_path('stimpy')

        match self.source_version:
            case 'pyvstim':
                log_directory = log_directory / self.pyvstim_filename
                return PyVlog(log_directory, **kwargs)
            case 'stimpy-bit' | 'stimpy-git':
                log_directory = log_directory / self.stimpy_filename
                return RiglogData(log_directory, diode_offset=self.with_diode_offset, **kwargs)
            case _:
                raise ValueError('')

    # ================== #
    # Protocol Dependent #
    # ================== #

    @property
    def session_list(self) -> list[Session]:
        """list of sessions based on the protocol type"""
        if self.is_vop_protocol:
            return ['light', 'visual', 'dark']
        elif self.is_ldl_protocol:
            return ['light_bas', 'dark', 'light_end']
        else:
            raise ValueError('unsupported protocol')

    @property
    def gspread_reference(self) -> GSPREAD_SHEET_PAGE:
        """get statistic google spreadsheet reference from the protocol type"""
        if self.is_vop_protocol:
            return 'apcls_tac'
        elif self.is_ldl_protocol:
            return 'ap_ldl'
        else:
            raise ValueError('unsupported protocol')

    @property
    def is_ldl_protocol(self) -> bool:
        """if is light dark light protocol"""
        return self.get_protocol_alias() == 'light_dark_light'

    @property
    def is_vop_protocol(self) -> bool:
        """if is visual open loop protocol"""
        return self.get_protocol_alias() == 'visual_open_loop'

    @property
    def is_virtual_env(self) -> bool:
        return 'vr' in self.get_protocol_alias()

    def get_protocol_alias(self) -> ProtocolAlias:
        """get the protocol type from the filename"""
        return get_protocol_name(self.load_riglog_data().riglog_file)

    def get_session_info(self, rig: RiglogData, session: Session) -> SessionInfo:
        if self.is_virtual_env:
            return rig.get_pygame_stimlog().session_trials()[session]
        else:
            return rig.get_stimlog().session_trials()[session]

    def masking_lap_time(self, rig: RiglogData) -> np.ndarray:

        if self.is_virtual_env:
            stim = rig.get_pygame_stimlog()
            session = stim.session_trials()[self.session]
            lap = stim.virtual_lap_event
            t = lap.time
            v = lap.value.astype(int)
        else:
            session = rig.get_stimlog().session_trials()[self.session]
            t = rig.lap_event.time
            v = rig.lap_event.value.astype(int)

        lap_index = session.in_slice(t, v)

        return t[lap_index]
