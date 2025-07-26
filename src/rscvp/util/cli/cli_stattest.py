from typing import Literal, Any

import numpy as np
import polars as pl
from scipy.stats import cramervonmises_2samp, wilcoxon, mannwhitneyu
from scipy.stats._hypotests import CramerVonMisesResult
from scipy.stats._mannwhitneyu import MannwhitneyuResult
from scipy.stats._morestats import WilcoxonResult
from scipy.stats._stats_py import (
    KstestResult,
    kstest,
    ttest_rel,
    TtestResult,
    ttest_ind,
    normaltest,
    NormaltestResult
)

from argclz import argument
from neuralib.io import save_json
from neuralib.typing import PathLike, ArrayLike, DataFrame
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_stat import CollectDataSet

__all__ = [
    'StatResults',
    'StatisticTestOptions'
]

# ================= #
# StatMethodOptions #
# ================= #

STAT_TEST_TYPE = Literal['ttest', 'kstest', 'cvm', 'pairwise_ttest']
StatResults = dict[str, Any] | DataFrame


class StatisticTestOptions(StatisticOptions):
    dependent: bool = argument(
        '--dep',
        help='dependent or correlated test, only applicable for ttest'
    )

    parametric: bool = argument(
        '--para',
        help='assume parametric test, only applicable for ttest'
    )

    test_type: STAT_TEST_TYPE = argument(
        '--stat-test',
        help='performs which statistical test'
    )

    ttest_parametric_infer: bool = argument(
        '--infer-para',
        help='Run normality test and infer parametric of samples in ttest'
    )

    @property
    def options(self) -> dict[str, Any]:
        return {
            'header': self.header,
            'parametric': self.parametric,
            'dependent': self.dependent
        }

    # ----CramerVonMises(CVM)---- #

    def run_cramervonmises_test(self, dataset: CollectDataSet,
                                output: PathLike, **kwargs) -> StatResults:
        self.parametric = False
        ret: CramerVonMisesResult = cramervonmises_2samp(dataset[0], dataset[1], **kwargs)

        res = {}
        res.setdefault('test', 'cramervonmises')
        res = {**res, **ret.__dict__, **self.options}
        save_json(output, res)

        return res

    # ----KS test---- #

    def run_ks_test(self, dataset: CollectDataSet,
                    output: PathLike, **kwargs) -> StatResults:
        """performs the (one-sample or two-sample) Kolmogorov-Smirnov test for goodness of fit"""
        self.parametric = False
        if self.dependent:
            raise RuntimeError('consider other statistic methods')

        ret: KstestResult = kstest(dataset[0], dataset[1], **kwargs)

        res = {}
        res.setdefault('test', 'ks_test')
        res = {**res, **ret._asdict(), **self.options}
        save_json(output, res)

        return res

    # ----T test---- #

    @staticmethod
    def run_normality_test(data: ArrayLike) -> tuple[float, bool]:
        """
        Test if the variables for comparison are normal distribution

        :param data: ``ArrayLike``
        :return: tuple of pval and if ``data`` is normal distribution
        """
        ret: NormaltestResult = normaltest(data)
        pval = ret.pvalue
        return pval, pval >= 0.05

    def run_ttest(self, dataset: CollectDataSet | np.ndarray,
                  output: PathLike | None, **kwargs) -> StatResults:
        """

        :param dataset: ``CollectDataSet`` or `Array[float, [2, ...]]`
        :param output: Output json filepath
        :param kwargs: Additional arguments pass to *statistic test function*
        :return: ``StatResults``
        """

        self._validate_two_samples(dataset)
        x = dataset[0]
        y = dataset[1]

        # infer parametric
        if self.ttest_parametric_infer:
            xp, x_norm = self.run_normality_test(x)
            yp, y_norm = self.run_normality_test(y)
            self.parametric = np.all([x_norm, y_norm]).item()  # as numpy 2.x

        #
        res = {}
        match (self.parametric, self.dependent):
            case (True, True):
                res.setdefault('test', 'ttest_rel')
                ret: TtestResult = ttest_rel(x, y, **kwargs)
                res.setdefault('statistic', ret.statistic)
                res.setdefault('pvalue', ret.pvalue)
                res = {**res, **self.options}

            case (True, False):
                res.setdefault('test', 'ttest_ind')
                ret: TtestResult = ttest_ind(x, y, **kwargs)
                res.setdefault('statistic', ret.statistic)
                res.setdefault('pvalue', ret.pvalue)
                res = {**res, **self.options}

            case (False, True):
                res.setdefault('test', 'wilcoxon')
                ret: WilcoxonResult = wilcoxon(x, y, **kwargs)
                res.setdefault('statistic', ret.statistic)
                res.setdefault('pvalue', ret.pvalue)
                res = {**res, **self.options}

            case (False, False):
                res.setdefault('test', 'mannwhitneyu')
                ret: MannwhitneyuResult = mannwhitneyu(x, y, **kwargs)
                res = {**res, **ret._asdict(), **self.options}

            case _:
                raise RuntimeError(f'Statistic test not found: ({self.parametric=}, {self.dependent=})')

        if output is not None:
            save_json(output, res)

        return res

    @staticmethod
    def _validate_two_samples(dataset: CollectDataSet | np.ndarray):
        if isinstance(dataset, CollectDataSet):
            if not dataset.is_two_samples:
                raise ValueError('')
        elif isinstance(dataset, np.ndarray):
            if dataset.shape[0] != 2:
                raise ValueError('')
        else:
            raise TypeError('')

    # -----Multiple comparison (More than two data sample)----- #

    def run_pairwise_ttest(self, dataset: CollectDataSet,
                           output: PathLike,
                           *,
                           dv: str | None = None,
                           within: str | None = None,
                           between: str | None = None,
                           subject: str | None = None,
                           **kwargs) -> StatResults:
        import pingouin as pg

        self._validate_multisample(dataset)
        df = dataset.to_polars(melt=True)

        if dv is None:
            dv = dataset.name[0]

        if between is None:
            between = dataset.group_header

        post_hocs = pg.pairwise_tests(
            data=df.to_pandas(),
            dv=dv,
            within=within,
            between=between,
            subject=subject,
            parametric=self.parametric,
            padjust='bonf',
            **kwargs
        )

        post_hocs = pl.from_pandas(post_hocs)

        post_hocs.write_csv(output)

        return post_hocs

    @staticmethod
    def _validate_multisample(dataset: CollectDataSet | np.ndarray):
        if isinstance(dataset, CollectDataSet):
            if not dataset.is_multisample:
                raise ValueError('')
        elif isinstance(dataset, np.ndarray):
            if dataset.shape[0] <= 2:
                raise ValueError('')
        else:
            raise TypeError('')
