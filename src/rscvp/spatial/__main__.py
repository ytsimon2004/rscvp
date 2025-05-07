from argclz.commands import parse_command_args
from .main_align_map import AlignPeakMapOptions
from .main_belt_sort import CPBeltSortOptions
from .main_belt_sort_trial import CPBeltSortTrialOptions
from .main_cache_occ import PosBinActCacheBuilder
from .main_cache_sortidx import SortIdxCacheBuilder
from .main_ev_pos import EVOptions
from .main_place_field import PlaceFieldsOptions
from .main_population_matrix import PopulationMTXOptions
from .main_position_map import PositionMapOptions
from .main_si import SiOptions
from .main_slb import PositionLowerBoundOptions
from .main_sparsity import SparsityOptions
from .main_trial_corr import TrialCorrOptions

parse_command_args(
    usage='python -m rscvp.spatial [CMD] ...',
    description='plot and store spatial-related parameters',
    parsers=dict(
        ev=EVOptions,
        pf=PlaceFieldsOptions,
        si=SiOptions,
        slb=PositionLowerBoundOptions,
        spr=SparsityOptions,
        tcc=TrialCorrOptions,
        am=AlignPeakMapOptions,
        ba=PositionMapOptions,
        sa=CPBeltSortOptions,
        spv=CPBeltSortTrialOptions,
        sa_cache=SortIdxCacheBuilder,
        pba_cache=PosBinActCacheBuilder,
        cm=PopulationMTXOptions
    )
)
