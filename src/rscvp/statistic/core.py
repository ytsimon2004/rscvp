import abc
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from polars.exceptions import ColumnNotFoundError
from rich.pretty import pprint
from scipy.stats import wilcoxon, mannwhitneyu

from argclz import argument, AbstractParser
from neuralib.plot import plot_figure
from neuralib.typing import DataFrame, PathLike, ArrayLike, is_numeric_arraylike
from neuralib.util.utils import joinn
from neuralib.util.utils import uglob
from neuralib.util.verbose import fprint
from rscvp.statistic.cli_gspread import GSPExtractor
from rscvp.util.cli.cli_stattest import StatisticTestOptions, StatResults
from rscvp.util.database import (
    DB_TYPE,
    GenericClassDB,
    BayesDecodeDB,
    VisualSFTFDirDB,
    RSCDatabase
)
from rscvp.util.util_gspread import get_statistic_key_info, GSPREAD_SHEET_PAGE
from rscvp.util.util_plot import REGION_COLORS_HIST, SESSION_COLORS
from rscvp.util.util_stat import GROUP_HEADER_TYPE, CollectDataSet, get_stat_group_vars

__all__ = [
    'StatPipeline',
    'print_pkl',
    'print_var',
    'pval_verbose'
]


class StatPipeline(AbstractParser, StatisticTestOptions, metaclass=abc.ABCMeta):
    """
    Folder Structure ::

        [STAT_DIR] /
            └── [SPREADSHEET_NAME] /
                    └── [HEADER]_[GROUP_HEADER]_[TEST_TYPE] /
                        ├── collect_dataset/
                        │       └── collected_dataset_*.pkl  (1)
                        │
                        ├── extracted_data/
                        │       └── extracted_data_*.parquet  (2)
                        │
                        ├── statistic_figure/
                        │        └── figures.pdf  (3)
                        │
                        └── statistic_result/
                            └── perc95_dff_region.json/csv (4)


    (1) pkl file for :class:`CollectDataSet` saving
    (2) extracted data with selected header(s) in parquet
    (3) figures output
    (4) statistic output json/csv

    """

    load_source: Literal['gspread', 'parquet', 'db'] = argument(
        '--load',
        default='parquet',
        help='whether load source from gspread (i.e., population average) or parquet file (i.e., individual neurons)'
    )

    db_table: DB_TYPE = argument(
        '--db',
        default=None,
        help='which db table'
    )

    group_header: GROUP_HEADER_TYPE = argument(
        '--group-var',
        default='region',
        help='which group variable for statistic'
    )

    animal_based_comp: bool = argument(
        '--animals-comp',
        help='do animal-based pair comparison'
    )

    df: DataFrame
    res: StatResults

    def post_parsing(self):
        if self.load_source == 'db' and self.db_table is None:
            raise ValueError('please specify the db table')

    @abc.abstractmethod
    def run(self):
        pass

    def run_pipeline(self):
        self.mkdir_structure()

        #
        df = pl.from_pandas(self.df) if isinstance(self.df, pd.DataFrame) else self.df
        df.write_parquet(self.output_extract_data)

        #
        dataset = self.get_collect_data()

        self.res = self.generate_stat_result()
        dataset.test_type = self.test_type
        dataset.to_pickle(self.pickled_file)

        self.plot()

    def load_table(self, primary_key: str | tuple[str, ...] = 'Data',
                   to_pandas: bool = True,
                   concat_plane_only: bool = True) -> None:
        """
        Load table from source

        - gspread: google spreadsheet, specify the worksheet using ``sheet_name``
        - parquet: local parquet file, used for array-like values in one cell (i.e., after csv aggregation)
        - db: local database (i.e., REPO/res/database/*.db)

        :param primary_key: used in preprocess (i.e., remove # for statistic...).
            only useful in ``gspread``, ``parquet`` load source
        :param to_pandas: polars to pandas dataframe
        :param concat_plane_only: in ETL dataset, only filter ``optic`` with ``all``(i.e., db pipeline)
        :return:
        """
        if self.group_header == 'session':
            melt_session_vars = self.header
        else:
            melt_session_vars = None

        #
        match self.load_source:
            case 'gspread':
                df = GSPExtractor(self.sheet_name).load_from_gspread(primary_key=primary_key)
            case 'parquet':
                df = GSPExtractor(self.sheet_name).load_parquet_file(self.statistic_dir, melt_session_vars,
                                                                     primary_key=primary_key)
            case 'db':
                df = self._load_database(self.db_table)
            case _:
                raise ValueError(f'Unknown load source: {self.load_source}')

        #
        if concat_plane_only:
            df = self._take_concat_planes(df)

        #
        if to_pandas:
            df = df.to_pandas()

        self.df = df

    @staticmethod
    def _load_database(db: DB_TYPE) -> pl.DataFrame:
        func = RSCDatabase().get_data
        if db == 'GenericDB':
            return func(GenericClassDB)
        elif db == 'BayesDecodeDB':
            return func(BayesDecodeDB)
        elif db == 'VisualSFTFDirDB':
            return func(VisualSFTFDirDB)
        else:
            raise ValueError('')

    @staticmethod
    def _take_concat_planes(df: pl.DataFrame) -> pl.DataFrame:
        try:
            expr = pl.struct('date', 'animal', 'rec', 'user')
            df = df.with_columns(expr.alias('primary'))
        except ColumnNotFoundError:
            return df
        else:
            ret = []
            for name, dat in df.group_by('primary'):
                if dat.shape[0] > 1:
                    primary_df = dat.filter(pl.col('optic') == 'all')
                else:
                    primary_df = dat

                ret.append(primary_df)

            return pl.concat(ret).drop('primary')

    # ================ #
    # Folder Structure #
    # ================ #

    @property
    def directory(self) -> Path:
        d = self.statistic_dir / self.sheet_name / f'{self.header}_{self.group_header}_{self.test_type}'
        if not d.exists():
            d.mkdir(exist_ok=True, parents=True)
        return d

    @property
    def extracted_data(self) -> Path:
        return self.directory / 'extracted_data'

    @property
    def collect_dataset(self) -> Path:
        return self.directory / 'collect_dataset'

    @property
    def statistic_result(self) -> Path:
        return self.directory / 'statistic_result'

    @property
    def statistic_figure(self) -> Path:
        return self.directory / 'statistic_figure'

    @property
    def output_extract_data(self) -> Path:
        return self.extracted_data / f'extracted_data_{self.header}.parquet'

    @property
    def output_figure(self) -> Optional[Path]:
        if self.debug_mode:
            return
        return (self.statistic_figure / f'{self.header}_{self.group_header}').with_suffix('.pdf')

    def get_output_figure_type(self, *ext) -> Path:
        """specific fig types"""
        output = self.output_figure
        prefix = output.stem + f"_{joinn('-', *ext)}"
        suffix = output.suffix
        return output.with_name(prefix + suffix)

    @property
    def output_statistic_json(self) -> Path:
        return (self.statistic_result / f'{self.header}_{self.group_header}').with_suffix('.json')

    @property
    def output_statistic_csv(self) -> Path:
        """Used for visualization purpose. i.e., pairwise test"""
        return (self.statistic_result / f'{self.header}_{self.group_header}').with_suffix('.csv')

    @property
    def pickled_file(self) -> Path:
        return self.directory / 'collect_dataset' / f'collected_dataset_{self.header}.pkl'

    def callback_pickle_file(self, header: str, page: GSPREAD_SHEET_PAGE | None = None) -> Path:
        """call previous analyzed CollectDataSet pkl path"""
        if page is None:
            page = self.sheet_name

        pattern = f'{header}_{self.group_header}_{self.test_type}'
        return uglob(self.statistic_dir / page, f'{pattern}/collect_dataset/*.pkl')

    def mkdir_structure(self):
        f = ('extracted_data', 'collect_dataset', 'statistic_result', 'statistic_figure')
        for it in f:
            (self.directory / it).mkdir(exist_ok=True, parents=True)

    # ============== #
    # CollectDataSet #
    # ============== #

    _collect_data: CollectDataSet = None

    def get_collect_data(self) -> CollectDataSet:
        if self._collect_data is None:
            self._collect_data = self._get_collect_data()

        return self._collect_data

    def _get_collect_data(self,
                          group_header: GROUP_HEADER_TYPE | None = None,
                          col: str | None = None,
                          key_prefix: str | None = None,
                          verbose: bool = True) -> CollectDataSet:
        """
        grouping variable

        :param group_header: `independent variable` for groupby
        :param col: `dependent variable`  for statistic
        :param key_prefix: for categorical dataset

        :return
            (G, D)
        """
        if group_header is None:
            group_header = self.group_header

        group_vars = get_stat_group_vars(self.df, group_header)

        if col is None:
            col = self.variable

        try:
            value = self.df[col][0]
        except IndexError as e:
            fprint(f'might be empty in header: {col} in {self.load_source} files!', vtype='error')
            raise e

        #
        if isinstance(self.df, pl.DataFrame):
            df = self.df.to_pandas()
        else:
            df = self.df

        gg = df.groupby(group_header)

        if isinstance(value, float):
            ret = {
                g: gg.get_group(g)[col].to_numpy().astype(float)
                for g in group_vars
            }

        elif isinstance(value, (np.ndarray, pl.Series)):
            ret = {
                g: np.concatenate(list(gg.get_group(g)[col].to_numpy()))
                for g in group_vars
            }
        else:
            raise TypeError(f'col type: {type(value)}')

        if key_prefix is not None:
            ret = {
                key + f'_{key_prefix}': ret[key]
                for key in ret
            }

        ret = CollectDataSet([col], ret, self.group_header, self.test_type)

        if verbose:
            pprint(ret)

        return ret

    def generate_stat_result(self) -> StatResults:
        """
        generate statistical result dataframes.
        could be overwritten by children
        i.e., `pingouin` OR scipy.stat outputs ...
        """
        data = self.get_collect_data()

        match self.test_type:
            case 'ttest':
                return self.run_ttest(data, self.output_statistic_json)
            case 'kstest':
                return self.run_ks_test(data, self.output_statistic_json)
            case 'cvm':
                return self.run_cramervonmises_test(data, self.output_statistic_json)
            case 'pairwise_ttest':
                return self.run_pairwise_ttest(data, self.output_statistic_csv)
            case _:
                raise NotImplementedError(f'stat test unknown: {self.test_type}')

    # ================ #
    # Plot / Statistic #
    # ================ #

    @abc.abstractmethod
    def plot(self):
        """generate plot"""
        pass

    @property
    def color_palette(self) -> dict[str, str]:
        match self.group_header:
            case 'region':
                return REGION_COLORS_HIST
            case 'session':
                return SESSION_COLORS
            case _:
                raise NotImplementedError(f'unknown group header: {self.group_header}')

    def insert_pval(self, ax: Axes) -> None:
        res = self.res

        if isinstance(res, dict):
            fields = ['p-val', 'pvalue']
            txt = [res[f] for f in fields if f in res]
        elif isinstance(res, DataFrame):  # pg df
            txt = []
            df = pl.DataFrame(res).select('A', 'B', 'p-unc')
            for r in df.iter_rows(named=True):
                txt.append(f"{r['A']} vs. {r['B']}: {r['p-unc']}")

            txt = '\n'.join(txt)
        else:
            raise TypeError(f'res: {type(res)}')

        ax.set_title(f'{txt}', fontstyle='italic')

    # ============ #
    # Animal-based #
    # ============ #

    def plot_pairwise_mean(self, with_bar: bool = True) -> None:
        h = self.variable

        info = get_statistic_key_info().drop('region')
        df = self.df.join(info, on='Data')
        df = (
            df.select('Data', 'region', 'pair_wise_group', h)
            .sort('pair_wise_group', 'region')
            .with_columns(pl.col(h).list.len().alias('n_neurons'))
            .with_columns(pl.col(h).list.mean())
            .with_columns(
                pl.when(pl.col('Data').str.contains('|'.join(self._mouseline_thy1)))
                .then(pl.lit('thy1'))
                .when(pl.col('Data').str.contains('|'.join(self._mouseline_camk2)))
                .then(pl.lit('camk2'))
                .otherwise(pl.lit('other'))
                .alias('mouseline')
            )
        )

        value_a = df.filter(pl.col('region') == 'aRSC')[h].to_list()
        value_b = df.filter(pl.col('region') == 'pRSC')[h].to_list()

        with plot_figure(None, figsize=(3, 8)) as ax:
            self.plot_connect_datapoints(ax, value_a, value_b, df=df, with_bar=with_bar)

    def plot_connect_datapoints(self, ax: Axes,
                                value_a: np.ndarray,
                                value_b: np.ndarray,
                                df: pl.DataFrame | None = None,
                                with_bar: bool = True,
                                errorbar: Literal['ci', 'pi', 'se', 'sd'] = 'se'):
        if self.dependent:
            stat = wilcoxon(value_a, value_b)
        else:
            stat = mannwhitneyu(value_a, value_b)

        name = type(stat).__name__
        p = stat.pvalue

        # if different mouseline connected line
        if df is not None:
            h = self.variable
            mouseline_colors = {'thy1': 'gray', 'camk2': 'black'}

            for mouseline in df['mouseline'].unique():
                subset = df.filter(pl.col('mouseline') == mouseline)
                sub_a = subset.filter(pl.col('region') == 'aRSC')[h].to_list()
                sub_b = subset.filter(pl.col('region') == 'pRSC')[h].to_list()

                color = mouseline_colors.get(mouseline, 'red')
                for pair in zip(sub_a, sub_b):
                    ax.plot([0, 1], pair, color=color, alpha=0.7)
        else:
            for pair in zip(value_a, value_b):
                ax.plot([0, 1], pair, color='k')

        #
        if with_bar:
            import seaborn as sns
            sns.barplot(data=[value_a, value_b], errorbar=errorbar, ax=ax)

        ax.set_title(f'{name}\np = {p}', fontstyle='italic')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['aRSC', 'pRSC'])

        print_var(value_a, t='std')
        print_var(value_b, t='std')


def print_pkl(file: PathLike):
    import pickle
    from rich.pretty import pprint

    with open(file, 'rb') as f:
        dat = pickle.load(f)
        pprint(dat)


def print_var(arr: ArrayLike,
              t: Literal['sem', 'std'] = 'sem',
              prefix: str | None = None):
    """print variability for """
    msg = ''
    if prefix is not None:
        msg += f'[{prefix}] -> '

    #
    arr = np.asarray(arr)
    if not is_numeric_arraylike(arr):
        raise TypeError('')
    if arr.ndim != 1:
        raise ValueError('')

    #
    mean = np.mean(arr)
    match t:
        case 'sem':
            from scipy.stats import sem
            v = sem(arr)
        case 'std':
            from numpy import std
            v = std(arr)
        case _:
            raise ValueError(f'invalid: {t}')

    msg += f'mean +/- {t}: {mean:.4f} +/- {v:.4f}'

    fprint(msg)


def pval_verbose(pval: float) -> str:
    if pval < 0.001:
        return f"*** (p={pval:.3g})"
    elif pval < 0.01:
        return f"** (p={pval:.3g})"
    elif pval < 0.05:
        return f"* (p={pval:.3g})"
    else:
        return f"n.s. (p={pval:.3g})"
