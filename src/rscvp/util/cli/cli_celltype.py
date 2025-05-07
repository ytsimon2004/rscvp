import collections
from typing import ClassVar, NamedTuple, Literal

import numpy as np
import polars as pl
from rscvp.util.cli import SelectionOptions, CodeAlias, HEADER, get_headers_from_code
from scipy.stats import zscore
from typing_extensions import Self

from argclz import argument
from neuralib.typing import flatten_arraylike
from stimpyp import Session, RiglogData

__all__ = [
    'CellTypeSelectionOptions',
    'SessionDataFrame'
]

CELL_TYPE = Literal['visual', 'spatial', 'overlap']


class CellTypeSelectionOptions(SelectionOptions):
    DEFAULT_CTYPE_CODEDICT: ClassVar[dict[CELL_TYPE, list[CodeAlias]]] = {
        'visual': ['pa', 'vc'],
        'spatial': ['si', 'pf', 'spr', 'ev', 'tcc', 'ds'],
        'overlap': ['cord'],
    }

    DEFAULT_VISUAL_EXCLUDE: ClassVar[list[HEADER]] = ['preferred_sftf', 'ori_resp']
    DEFAULT_SPATIAL_EXCLUDE: ClassVar[list[HEADER]] = ['pf_width_raw', 'pf_peak', 'mean_dff', 'median_dff',
                                                       'perc95_dff', 'max_dff']

    cell_type: CELL_TYPE = argument(
        '--CT', '--ctype',
        metavar='TYPE',
        required=True,
        help='cell type OR parameter type for running the analysis'
    )

    fraction_cell: bool = argument(
        '--fraction-cell',
        help='plot fraction of spatial cell across sessions',
    )

    def masking_celltype(self, cell_type: CELL_TYPE, preselect: bool = True) -> None:
        """force set opt for the certain cell type"""
        self.pre_selection = preselect

        if cell_type == 'spatial':
            self.pc_selection = 'slb'
        elif cell_type == 'visual':
            self.vc_selection = 0.3
        elif cell_type == 'overlap':
            self.pc_selection = 'slb'
            self.vc_selection = 0.3
        else:
            raise ValueError(f'{cell_type} unknown')

    def select_dataframe(self, session: Session,
                         to_zscore: bool = True,
                         verbose: bool = False) -> pl.DataFrame:
        """
        Get the selected header from csv results for a certain cell type.

        nRows represent neuron numbers (AFTER SELECTION)::

        :param session: behavioral session
        :param to_zscore: whether normalization of numerical data
        :param verbose: print output pl.Dataframe
        :return: Dataframe

        Example::

            ┌───────────┬───────────┬───────────┬───────────┬───┬───────────┬───────────┬───────────┬──────────┐
            │ si        ┆ shuffled_ ┆ pf_width  ┆ pf_peak   ┆ … ┆ ev_trial_ ┆ trial_cc  ┆ mean_dff  ┆ max_dff  │
            │ ---       ┆ si        ┆ ---       ┆ ---       ┆   ┆ avg       ┆ ---       ┆ ---       ┆ ---      │
            │ f64       ┆ ---       ┆ f64       ┆ f64       ┆   ┆ ---       ┆ f64       ┆ f64       ┆ f64      │
            │           ┆ f64       ┆           ┆           ┆   ┆ f64       ┆           ┆           ┆          │
            ╞═══════════╪═══════════╪═══════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪══════════╡
            │ 2.081837  ┆ 4.26826   ┆ -0.452336 ┆ -1.122843 ┆ … ┆ 1.13402   ┆ 2.491553  ┆ 4.149803  ┆ 4.094087 │
            │ …         ┆ …         ┆ …         ┆ …         ┆ … ┆ …         ┆ …         ┆ …         ┆ …        │
            │ -0.346977 ┆ -0.603281 ┆ -0.339952 ┆ 0.854614  ┆ … ┆ -0.453812 ┆ -0.428618 ┆ -0.455956 ┆ -0.73740 │
            │ -0.607127 ┆ -0.837634 ┆ -0.396144 ┆ 0.000415  ┆ … ┆ -0.781146 ┆ -0.79303  ┆ -0.937063 ┆ -1.27833 │
            │           ┆           ┆           ┆           ┆   ┆           ┆           ┆           ┆ 9        │
            └───────────┴───────────┴───────────┴───────────┴───┴───────────┴───────────┴───────────┴──────────┘

        """
        if self.fraction_cell:
            df = self._fraction_select(session)
        else:
            df = self._foreach_select(session)

        print(f'{df=}')

        #
        if to_zscore:
            zscore_dy = {series.name: zscore(series) for series in df}
            df = pl.DataFrame(zscore_dy)

        # mask based on cell type
        if self.cell_type is not None:
            self.masking_celltype(self.cell_type)

        cell_mask = self.get_selected_neurons()
        df = df.filter(cell_mask)

        if verbose:
            print(df)

        return df

    def _fraction_select(self, s: Session) -> pl.DataFrame:
        paras = [get_headers_from_code(c) for c in ('pf', 'slb')]
        paras = flatten_arraylike(paras)

        paras = {
            par: self.get_csv_data(f'{par}_{s}')
            for par in paras
            if par in ('n_pf', 'nbins_exceed')
        }

        return pl.from_dict(paras)

    def _foreach_select(self, s: Session) -> pl.DataFrame:

        code = self.DEFAULT_CTYPE_CODEDICT[self.cell_type]
        paras = [get_headers_from_code(c) for c in code]
        paras = flatten_arraylike(paras)

        match self.cell_type:
            case 'visual':
                paras = [par for par in paras if par not in self.DEFAULT_VISUAL_EXCLUDE]
                df = self.get_csv_data(paras, to_numpy=False, enable_use_session=False)
            case 'spatial':
                paras = [par for par in paras if par not in self.DEFAULT_SPATIAL_EXCLUDE]

                ret = {}
                for p in paras:
                    match p:
                        case 'pf_width' | 'pf_reliability':
                            val = self._parse_pf_pars(self.get_csv_data(f'{p}_{s}'))
                        case _:
                            val = self.get_csv_data(f'{p}_{s}')
                    ret[p] = val
                df = pl.from_dict(ret)
            case _:
                raise ValueError('')

        return df

    @staticmethod
    def _parse_pf_pars(values: np.ndarray) -> np.ndarray:
        """
        Parse place field-related parameters

        - transform nan/null to 0
        - do the average for multiple values per cell

        *Note that some parameters make no sense (e.g. pf_peak)*
        """
        ret = []
        for it in values:  # for each cell's pf parameters
            if it is None:  # no pf
                ret.append(0)
            else:
                ret.append(round(np.mean(list(map(float, it.split(' ')))), 1))

        return np.array(ret)

    def get_session_dataframe(self, rig: RiglogData) -> 'SessionDataFrame':
        """
        spatial variables in different behavioral session
        ** NOTE that row number represent all neurons (without selection)

        :return: Dict of key (session name), value (pl.DataFrame)
        """

        sy = collections.OrderedDict()  # dict('ses': df)
        session = [it for it in rig.get_stimlog().session_trials() if it != 'all']
        for s in session:
            df = self.select_dataframe(s, to_zscore=False)
            rename_cols = {}
            for var in df.columns:
                if var.endswith(f'_{s}'):
                    rename_cols[var] = var[:-len(s) - 1]

            df = df.rename(rename_cols)
            sy[s] = df

        return SessionDataFrame(sy)


class SessionDataFrame(NamedTuple):
    ses_dict: dict[Session, pl.DataFrame]
    """``Session``: df with all paras. keys: (N, V)"""

    @property
    def n_neurons(self) -> int:
        return self.ses_dict[self.sessions[0]].shape[0]

    @property
    def variables(self) -> list[str]:
        """variables list"""
        return self.ses_dict[self.sessions[0]].columns

    @property
    def sessions(self) -> list[Session]:
        return list(self.ses_dict)

    def with_session_norm(self, norm_session: str) -> Self:
        """Normalize values with baseline behavioral session"""
        base = self.ses_dict[norm_session].to_numpy()
        ret = {}
        for s, df in self.ses_dict.items():
            arr = df.to_numpy()
            arr /= base
            ret[s] = pl.from_numpy(arr)
            ret[s].columns = self.variables

        return self._replace(ses_dict=ret)

    def unpivot(self, var: str) -> pl.DataFrame:
        """
        Unpivot to long format table for a variable across ``Sessions``

        :param var: variable (header)
        :return:

        Example::

            ┌───────────┬───────────┬────────┬───────────┐
            │ neuron_id ┆ light_bas ┆ dark   ┆ light_end │
            │ ---       ┆ ---       ┆ ---    ┆ ---       │
            │ i64       ┆ f64       ┆ f64    ┆ f64       │
            ╞═══════════╪═══════════╪════════╪═══════════╡
            │ 0         ┆ 0.4311    ┆ 0.4082 ┆ 0.4028    │
            │ 1         ┆ 0.5051    ┆ 0.4358 ┆ 0.9136    │
            │ …         ┆ …         ┆ …      ┆ …         │
            │ 2769      ┆ 0.0997    ┆ 0.0137 ┆ 0.0311    │
            │ 2770      ┆ 0.1064    ┆ 0.0139 ┆ 0.0077    │
            └───────────┴───────────┴────────┴───────────┘
        """

        if var not in self.variables:
            raise ValueError(f'{var}')

        ret = {'neuron_id': np.arange(self.n_neurons)}
        for s, df in self.ses_dict.items():
            ret[s] = df[var].to_numpy()

        return pl.from_dict(ret)
