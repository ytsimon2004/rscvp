from typing import Literal

from argclz import argument
from .cli_model import ModelOptions

__all__ = [
    'BEHAVIOR_COVARIANT',
    'GLMOptions'
]

BEHAVIOR_COVARIANT = Literal['pos', 'speed', 'lick_rate', 'acceleration']


class GLMOptions(ModelOptions):
    var_type: BEHAVIOR_COVARIANT = argument(
        '-V', '--var',
        required=True,
        help='Behavioral covariates used for predict the neural activity',
    )
