from typing import ClassVar, Literal, TYPE_CHECKING

from argclz import argument, union_type

if TYPE_CHECKING:
    from rscvp.util.util_trials import TrialSelection  # noqa: F401

__all__ = [
    'CrossValidateType',
    'ModelOptions',
    'trial_cross_validate'
]

TRAIN_TEST_SPLIT_METHOD = Literal['odd', 'even', 'random_split']
CrossValidateType = TRAIN_TEST_SPLIT_METHOD | int


class ModelOptions:
    GROUP_MODEL: ClassVar[str] = 'modeling options'

    cross_validation: CrossValidateType = argument(
        '--CV', '--cv-type',
        type=union_type(int, str),
        default=0,
        group=GROUP_MODEL,
        help='int type for nfold for model cross validation, otherwise, string type',
    )

    train_fraction: float = argument(
        '--train',
        default=0.8,
        validator=lambda it: 0 < it < 1,
        group=GROUP_MODEL,
        help='fraction of data for train set if `random_split` in cv, the rest will be utilized in test set'
    )

    @property
    def cv_info(self) -> str:
        """verbose, filename usage"""
        match self.cross_validation:
            case 0:
                method = 'false'
            case 'random_split':
                method = f'{self.cross_validation}_{self.train_fraction}train'
            case 'odd' | 'even':
                method = self.cross_validation
            case int():
                method = f'{self.cross_validation}fold'
            case _:
                raise ValueError(f'Unsupported type for cross_validation: {type(self.cross_validation)}')

        return f'CV_{method}'


def trial_cross_validate(trial: 'TrialSelection',
                         cross_validation: CrossValidateType,
                         train_fraction: float = 0.8) -> list[tuple['TrialSelection', 'TrialSelection']]:
    """
    :return:
        train: The training set indices for that split.
        test:  The testing set indices for that split.
    """
    match cross_validation:
        case str():
            match cross_validation:
                case 'even':
                    train_set = trial.select_even()
                    test_set = train_set.invert()
                case 'odd':
                    train_set = trial.select_odd()
                    test_set = train_set.invert()
                case 'random_split':
                    train_set, test_set = trial.select_fraction(train_fraction)
                case _:
                    raise ValueError('')

            return [(train_set, test_set)]

        case int():
            match cross_validation:
                case 0:  # no cv
                    return [(trial, trial)]
                case _:
                    return trial.kfold_cv(int(cross_validation))

        case _:
            raise TypeError('')
