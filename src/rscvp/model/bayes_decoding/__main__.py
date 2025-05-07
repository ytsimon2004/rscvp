from argclz.commands import parse_command_args
from .cache_bayes import BayesDecodeCacheBuilder
from .main_decode_analysis import DecodeAnalysisOptions
from .main_disengagement import DecodeDisengageOptions

parse_command_args(
    usage='python -m rscvp.model CMD ...',
    description='do the related analysis for the modeling',
    parsers=dict(
        decode_cache=BayesDecodeCacheBuilder,
        analysis=DecodeAnalysisOptions,
        bayes_disengagement=DecodeDisengageOptions
    )
)
