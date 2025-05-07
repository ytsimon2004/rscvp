from argclz.commands import parse_command_args
from .cache_ctype_cord import CellTypeCordCacheBuilder
from .main_cls import ClsTopoOptions
from .main_fov import FOVOptions
from .main_rois_loc import RoiLocOptions
from .main_spatial_topo import SpatialTopoPlotOptions
from .main_visual_topo import VisTopoPlotOptions

parse_command_args(
    usage='python -m rscvp.topology CMD ...',
    description='classification for each cell types',
    parsers=dict(
        fov=FOVOptions,
        loc=RoiLocOptions,
        ctopo=ClsTopoOptions,
        vtopo=VisTopoPlotOptions,
        stopo=SpatialTopoPlotOptions,
        ct=CellTypeCordCacheBuilder
    )
)
