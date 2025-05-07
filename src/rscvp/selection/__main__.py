from argclz.commands import parse_command_args
from .main_active_place_cell import ActivePCOptions
from .main_cls_summary import ClsCellTypeOptions
from .main_motion_drift import MotionDriftOptions
from .main_neuropil_error import NeuropilErrOptions
from .main_trial_reliability import TrialReliabilityOptions

parse_command_args(
    usage='python -m rscvp.selection CMD ...',
    description='do the related analysis for the preselection',
    parsers=dict(
        apc=ActivePCOptions,
        md=MotionDriftOptions,
        np=NeuropilErrOptions,
        tr=TrialReliabilityOptions,
        cls=ClsCellTypeOptions
    )
)
