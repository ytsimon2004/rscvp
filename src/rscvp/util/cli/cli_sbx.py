from functools import cached_property
from typing import ClassVar

from argclz import argument
from neuralib.scanbox.core import SBXInfo
from neuralib.suite2p import Suite2PResult
from neuralib.util.utils import uglob
from neuralib.util.verbose import fprint
from .cli_core import CommonOptions

__all__ = ['SBXOptions']


class SBXOptions(CommonOptions):
    """Scanbox options for calcium imaging DAQ system"""

    GROUP_SBX: ClassVar = 'Scanbox Options'
    """group scanbox options"""

    GROUP_MAPPING: ClassVar = 'Brain Mapping Options'
    """group brain mapping options"""

    # -----SBX----- #
    lines: int = argument('--lines', group=GROUP_SBX, default=528, help='number of lines for the scanning fov')
    objective_type: int = argument('--obj-type', group=GROUP_SBX, default=16, help='objective magnification. i.e., 16')
    zoom: float = argument('--obj-zoom', group=GROUP_SBX, default=1.7, help='zoom setting during acquisition')

    # -----mapping----- #
    brain_mapping: bool = argument('--brain-mapping', group=GROUP_MAPPING, help='Do brain mapping scaling')

    def load_sbx(self) -> SBXInfo:
        f = uglob(self.get_io_config().phy_animal_dir, '*.mat')
        return SBXInfo.load(f)

    @property
    def map_brain_factor(self) -> float:
        """From theoretical XY(um) to actual brain atlas coordinate (brain size variation from individual animal)"""
        from rscvp.topology.util import RSCObjectiveFOV

        fov = RSCObjectiveFOV.load_from_gspread(self.exp_date, self.animal_id, usage=self.exp_usage).to_um()
        d = self.scanbox_distance
        fx = fov.ml_distance / d[0]
        fy = fov.ap_distance / d[1]

        return (fx + fy) / 2

    @cached_property
    def scanbox_distance(self) -> tuple[float, float]:
        """(X, Y) in um shown in the scanbox system"""
        try:
            sbx = self.load_sbx()
        except FileNotFoundError:
            fprint('Use default value for 2P FOV dimension', vtype='warning')
            return _get_default_scanbox_fov_dimension(self.lines, self.objective_type, self.zoom)
        else:
            return sbx.fov_distance

    def pixel2distance_factor(self, s2p: Suite2PResult) -> float:
        """2p image pixel unit to um"""
        x, y = self.scanbox_distance

        if self.brain_mapping:
            f = self.map_brain_factor
            fprint(f'Scale to fov to brain coordinates: {f}')
            x *= f
            y *= f

        factor_x = x / s2p.image_width
        factor_y = y / s2p.image_height

        return (factor_x + factor_y) / 2


def _get_default_scanbox_fov_dimension(lines: int, obj_type: int, zoom: float) -> tuple[float, float]:
    """
    Hardware/settings dependent fov size according to recording configuration

    :param lines: number of lines for the scanning fov
    :param obj_type: objective magnification. i.e., 16X
    :param zoom: zoom setting during acquisition
    :return: (X, Y) in um
    """
    # ~ 30hz
    if lines == 528 and obj_type == 16:
        if zoom == 1:
            return 1396, 1056
        elif zoom == 1.2:
            return 1284, 978
        elif zoom == 1.4:
            return 1023, 765
        elif zoom == 1.7:
            return 892, 667
        elif zoom == 2.0:
            return 716, 531
        elif zoom == 2.4:
            return 632, 463
        else:
            raise NotImplementedError('')

    else:
        raise NotImplementedError('')
