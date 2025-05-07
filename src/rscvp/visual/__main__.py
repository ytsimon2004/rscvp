from argclz.commands import parse_command_args
from .main_polar import VisualPolarOptions
from .main_reliability import VisualReliabilityOptions
from .main_sftf_fit import SFTFModelCacheBuilder
from .main_sftf_pref import VisualSFTFPrefOptions
from .main_tuning import VisualTuningOptions

parse_command_args(
    usage='python -m rscvp.visual CMD ...',
    description='plot visually-driven neural activity',
    parsers=dict(
        pa=VisualPolarOptions,
        vf=SFTFModelCacheBuilder,
        st=VisualSFTFPrefOptions,
        ta=VisualTuningOptions,
        vc=VisualReliabilityOptions
    )
)
