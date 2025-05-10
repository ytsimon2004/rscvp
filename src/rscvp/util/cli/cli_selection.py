from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, ClassVar

import attrs
import numpy as np
import polars as pl

from argclz import argument, try_int_type
from neuralib.util.verbose import fprint
from stimpyp import Session
from .cli_io import CELLULAR_IO
from .cli_stimpy import StimpyOptions
from .cli_suite2p import Suite2pOptions

__all__ = [
    'SelectionOptions',
    'SelectionMask',
]

PC_CLASSIFIER = Literal['si', 'slb', 'intersec']
"""place(spatial)-tuned selection classifier type"""


class SelectionOptions(Suite2pOptions, StimpyOptions):
    """Neuronal selection and masking options"""
    
    GROUP_SELECTION: ClassVar[str] = 'selection of neuron options'
    """selection of neuron options"""

    DEFAULT_NEUROPIL_THRES: ClassVar[float] = -5
    """default neuropiil error threshold for preselection"""

    DEFAULT_TRIAL_THRES: ClassVar[float] = 0.25
    """default trial (lap) reliability threshold for preselection"""

    DEFAULT_VISUAL_THRES: ClassVar[float] = 0.2
    """default visual reliability threshold for preselection"""

    pre_selection: bool = argument(
        '--pre',
        group=GROUP_SELECTION,
        help='with neuron preselection',
    )

    pc_selection: PC_CLASSIFIER | None = argument(
        '--classifier', '-c',
        group=GROUP_SELECTION,
        help='specify the place cell classifier type',
    )

    no_pf_limit: bool = argument(
        '--no-pf-limit',
        group=GROUP_SELECTION,
        help='whether limited ranges of place field width in place cell criteria',
    )

    vc_selection: float | None = argument(
        '--visual-reliability', '--vr',
        metavar='VALUE',
        type=float,
        group=GROUP_SELECTION,
        help='visual reliability for the selection threshold',
    )

    used_session: Session = argument(
        '--us', '--used_session',
        metavar='SESSION',
        group=GROUP_SELECTION,
        help='which session in filename and column name that used in preselection',
    )

    random: str | int | None = argument(
        '-r', '--random',
        metavar='VALUE',
        type=try_int_type,
        group=GROUP_SELECTION,
        help='how many percent (with %%) or number of randomly-selected neurons',
    )

    not_circular_env: bool = argument(
        '--not-circular',
        group=GROUP_SELECTION,
        help='if not in circular environment, then do the preselection without considering trial-to trial reliability'
    )

    def __init__(self):
        self.setup_logger(Path(__file__).name)

    @property
    def root(self) -> Path:
        return self.get_io_config().phy_animal_dir

    @property
    def n_total_neurons(self) -> int:
        """number of total neurons"""
        return len(self.get_selected_neurons())

    @property
    def n_selected_neurons(self) -> int:
        """number of selected neurons"""
        return np.count_nonzero(self.get_selected_neurons())

    @property
    def selected_neurons(self) -> np.ndarray:
        """selected neurons indices"""
        return np.nonzero(self.get_selected_neurons())[0]

    def get_csv_data(self, cols: str | list[str], *,
                     to_numpy: bool = True,
                     session: Session | None = None,
                     enable_use_session: bool = True,
                     verbose: bool = False,
                     **kwargs) -> np.ndarray | pl.DataFrame:
        """
        Get data from an output csv pipeline

        :param cols: Column names of the csv file
        :param to_numpy: Return as 1D numpy array. `Array[Any, N]`
        :param session: Specify the session to truncate :meth:`_check_file_found()`
        :param enable_use_session: If multiple files are fit with the pattern,
             whether use `self.used_session` as a reference, If False, directly used the latest version
        :param verbose: Verbose print
        :param kwargs: Additional args passed to ``pl.read_csv()``
        :return: 1D numpy array | polars Dataframe
        """
        plane_index = self.plane_index

        if plane_index is None:
            p = 'concat*/*csv*/*.csv'
            f = list(self.root.glob(p))
            file = self._check_file_found(f, p, cols, enable_use_session)
            ret = pl.read_csv(file, columns=[cols] if isinstance(cols, str) else cols, **kwargs)
        else:
            session = session if session is not None else None

            if isinstance(cols, str):
                if cols in ('cell_prob', 'red_cell_prob'):
                    # noinspection PyTypeChecker
                    ret = _find_single_plane_cell_prob(plane_index, self.suite2p_directory, cols)
                else:
                    filename = _get_filename_from_col(cols, session)
                    p = f'plane{plane_index}/{filename}*/*.csv'
                    f = list(self.root.glob(p))
                    file = self._check_file_found(f, p, cols, enable_use_session)
                    ret = pl.read_csv(file, columns=[cols] if isinstance(cols, str) else cols, **kwargs)

            elif isinstance(cols, list):
                ret = []
                for c in cols:
                    filename = _get_filename_from_col(c, session)
                    p = f'plane{plane_index}/{filename}*/*.csv'
                    f = list(self.root.glob(p))
                    ret.append(pl.read_csv(self._check_file_found(f, p, c, enable_use_session), columns=[c], **kwargs))

                ret = pl.concat(ret, how='horizontal')
            else:
                raise TypeError(f'{type(cols)}')

        #
        if to_numpy:
            ret = ret[cols].to_numpy()

        if verbose:
            fprint(f'GET CSV cols: {cols}', vtype='io')
            print(ret)

        return ret

    def _check_file_found(self,
                          f: list[Path],
                          glob_pattern: str,
                          col: str,
                          enable_use_session: bool = True) -> Path:
        if len(f) == 0:
            if '%' in str(self.root):
                raise RuntimeError('extend src path first')

            raise FileNotFoundError(f'{(self.root / glob_pattern).resolve()} in {self.exp_date}_{self.animal_id}')

        elif len(f) == 1:
            f = f[0]

        # if more than one file fit with 'filename' pattern, then session need to be specified for apply selection
        elif len(f) > 1:
            if enable_use_session:
                used_session = self.used_session
                if used_session is None:
                    raise RuntimeError('please provide session name that used to the do the preselection')

                for s in f:
                    if used_session in str(s.parent.stem):
                        f = s
                        break
                else:
                    raise RuntimeError(f'{col} in {used_session} not found in {self.exp_date}_{self.animal_id}')
            else:
                f = f[-1]  # latest version

            self.logger.warning(f'multiple files fit for glob pattern, use csv in {f.parent.stem}')

        return f

    # ========= #
    # Selection #
    # ========= #

    @property
    def pf_limit(self) -> bool:
        """if place field limit as selection criteria"""
        return not self.no_pf_limit

    def pre_select(self) -> np.ndarray:
        """do the preselection before analysis

        :return: mask array. `Array[bool, N]`
        """
        #
        if self.not_circular_env:
            t = False
        else:
            t = self.get_csv_data(f'trial_reliability_{self.used_session}') >= self.DEFAULT_TRIAL_THRES

        try:
            if self.is_vop_protocol():
                self.logger.info('do the preselection in vop...')
                n = self.get_csv_data('error_perc', enable_use_session=False) >= self.DEFAULT_NEUROPIL_THRES
                v = self.get_csv_data('reliability', enable_use_session=False) >= self.DEFAULT_VISUAL_THRES
                return n & (t | v)
            elif self.is_ldl_protocol():
                self.logger.info('do the preselection in non-visual prot')
                n = self.get_csv_data('error_perc', enable_use_session=False) >= self.DEFAULT_NEUROPIL_THRES
                return n & t

        except FileNotFoundError as e:
            raise RuntimeError(f'{e} -> unknown protocol')

        raise ValueError('')

    def select_place_neurons(self, classifier: PC_CLASSIFIER,
                             pf_limit: bool = True,
                             force_session: Session | None = None) -> np.ndarray:
        """Select place (spatially-tuned) neuron using different methods.

        :param classifier: ``PC_CLASSIFIER``
        :param pf_limit: whether included the place field limit, if any of the place field within 15-120 cm
        :param force_session: whether to force session selection. (used in session comparison, LDL protocol)
        :return: mask array. `Array[bool, N]`
        """
        _prev_used_session = self.used_session
        use_session = force_session or _prev_used_session
        self.used_session = use_session

        self.logger.info(f'select place cells using {classifier}, pf criteria: {pf_limit}')

        if pf_limit:
            pf = self.get_csv_data(f'n_pf_{use_session}') != 0
        else:
            pf = True

        match classifier:
            case 'si':
                si = self.get_csv_data('si')
                ss = self.get_csv_data('shuffled_si')
                ret = np.array(si > ss) & pf
            case 'slb':
                ret = (self.get_csv_data(f'nbins_exceed_{use_session}') >= 1) & pf
            case 'intersec':
                six = self.select_place_neurons('si', pf_limit=self.pf_limit)
                slbx = self.select_place_neurons('slb', pf_limit=self.pf_limit)
                ret = six & slbx
            case _:
                raise ValueError(f'invalid classifier {classifier}')

        self.used_session = _prev_used_session  # rollback

        return ret

    def select_visual_neurons(self, reliability: float = 0.3) -> np.ndarray:
        """Select visually-responsive neuron using visual reliability

        :param reliability: reliability threshold
        :return: mask array
        """
        self.logger.info(f'use visual reliability {reliability} to select visual cell...')
        return self.get_csv_data('reliability', enable_use_session=False) >= reliability

    def select_visuospatial_neurons(self) -> np.ndarray:
        if self.pc_selection is None:
            raise ValueError('')
        return (
                self.select_place_neurons(self.pc_selection, pf_limit=self.pf_limit) &
                self.select_visual_neurons(self.vc_selection)
        )

    def select_red_neurons(self, p: bool = 0.65) -> np.ndarray:
        """
        selection red cell using red cell probability >=0.65

        :param self:
        :param p:
        :return:
        """
        plane_index = self.plane_index
        if plane_index is None:
            return self.get_csv_data('red_cell_prob') >= p
        else:
            s2p = self.load_suite_2p()
            return s2p.red_cell_prob >= p

    # cannot cache due to batch foreach stat
    def get_selected_neurons(self) -> np.ndarray:
        """Do the selection, can be directly used in cli.

        :return: mask array. `Array[bool, N]`
        """
        # preselection
        if self.pre_selection:
            ret = self.pre_select()
        else:
            ret = np.ones_like(self.get_csv_data('error_perc', enable_use_session=False), dtype=bool)  # all neuron

        # spatial tuned selection
        if self.pc_selection is not None:
            ret = np.logical_and(ret, self.select_place_neurons(self.pc_selection, pf_limit=self.pf_limit))

        # visual tuned selection
        if self.vc_selection is not None:
            ret = np.logical_and(ret, self.select_visual_neurons(self.vc_selection))

        # random selection
        if self.random is not None:
            ret = self._random_value(ret, self.random)

        if np.count_nonzero(ret) == 0:
            raise RuntimeError('no cells were selected, something wrong! check all the preselection criteria files')

        return ret

    def get_selection_mask(self, with_preselection: bool = True) -> SelectionMask:
        """get selection mask for all neurons"""
        ps = self.pre_select() if with_preselection else np.full(self.get_all_neurons(), 0, dtype=bool)
        n_neurons = np.count_nonzero(ps)

        if self.vc_selection is None:
            raise ValueError('to get the mask, visual reliability cannot be None')

        vc = self.select_visual_neurons(self.vc_selection)
        pc = self.select_place_neurons('slb', pf_limit=True)

        return SelectionMask(
            n_neurons,
            ps,
            vc & ps,
            pc & ps,
            vc & pc & ps,
            ~vc & ~pc & ps,
            self.select_red_neurons() & ps if self.has_chan2 else None
        )

    def _random_value(self, mask: np.ndarray, value: str | int) -> np.ndarray:
        """Handle union random type and return a new mask"""
        selected_ids = np.nonzero(mask)[0]
        n_neurons = len(selected_ids)
        ret = np.zeros_like(mask)  # new indices

        if isinstance(value, int):
            n = value
            if n > n_neurons:
                n = n_neurons
                self.logger.warning(f'random number exceed: {n}, use {n_neurons}')
            else:
                self.logger.info(f'apply randomly selected neurons, use {n} / {n_neurons}')

        elif isinstance(value, str) and value.endswith('%'):
            i = value.rfind('%')
            p = int(value[:i])
            n = int(n_neurons * p / 100)
            self.logger.info(f'apply randomly selected neurons, use {p}% ({n}/ {n_neurons})')

        else:
            raise RuntimeError('')

        ret[random.sample(list(selected_ids), n)] = 1
        return ret

    # ===================== #
    # Verbose/Filename Info #
    # ===================== #

    def selection_info(self, sep: str = '\n') -> str:
        """selection information"""
        pre = f'preselection: True use session {self.used_session}' if self.pre_selection else 'preselection: False'
        vis = f'visual: {self.vc_selection}' if self.vc_selection is not None else 'visual: False'
        place = f'place: {self.pc_selection}' if self.pc_selection is not None else 'place: False'

        select_num = np.count_nonzero(self.get_selected_neurons())
        all_cell = len(self.get_selected_neurons())
        _random = f'random: True ({select_num}/{all_cell})' if self.random is not None else f'random: None ({all_cell})'

        return sep.join((pre, vis, place, _random))

    def selection_prefix(self, sep: str = '-') -> str:
        """selection prefix for filename"""
        place = self.pc_selection or None

        ret = ''
        if place is not None:
            ret += place
        else:
            sep = ''

        if self.pre_selection:
            ret += f'{sep}pre'

        return ret


@attrs.frozen(repr=False)
class SelectionMask:
    """Cell mask after preselection. **keep the all cell shape**"""

    n_neurons: int
    """N"""
    pre_select_mask: np.ndarray
    """if no pre-select, then all False. `Array[bool, N]`"""
    visual_mask: np.ndarray
    """`Array[bool, N]`"""
    place_mask: np.ndarray
    """`Array[bool, N]`"""
    overlap_mask: np.ndarray
    """`Array[bool, N]`"""
    unclass_mask: np.ndarray
    """`Array[bool, N]`"""
    ch2_mask: np.ndarray | None
    """`Array[bool, N]`"""

    def __post_init__(self):
        r = list(map(np.count_nonzero, [self.visual_mask, self.place_mask, self.overlap_mask, self.unclass_mask]))
        expect_num = r[0] + r[1] - r[2] + r[3]
        if not expect_num != self.n_neurons:
            raise RuntimeError(f'{self.n_neurons}')

    def __repr__(self):
        v = np.nonzero(self.visual_mask)[0]
        p = np.nonzero(self.place_mask)[0]
        o = np.nonzero(self.overlap_mask)[0]
        return f'visual: {v}\nplace: {p}\noverlap: {o}'

    @property
    def n_visual(self) -> int:
        """Number of visual neurons"""
        return np.count_nonzero(self.visual_mask)

    @property
    def n_place(self) -> int:
        """Number of place neurons"""
        return np.count_nonzero(self.place_mask)

    @property
    def n_overlap(self) -> int:
        """Number of overlap neurons"""
        return np.count_nonzero(self.overlap_mask)

    @property
    def n_unclass(self) -> int:
        """Number of unclass neurons"""
        return np.count_nonzero(self.unclass_mask)


def _find_single_plane_cell_prob(plane_index: int,
                                 root: Path,
                                 col: Literal['cell_prob', 'red_cell_prob']) -> pl.DataFrame:
    """
    Create cell probability related pl.Series for single plane data (not integrated into concat csv)

    :param plane_index:
    :param root: root path, normally
    :param col:
    :return:
    """
    from neuralib.imaging.suite2p import Suite2PResult

    s2p = Suite2PResult.load((root / f'plane{plane_index}'), 0.5)

    if col == 'cell_prob':
        return pl.DataFrame({col: s2p.cell_prob})
    elif col == 'red_cell_prob':
        return pl.DataFrame({col: s2p.red_cell_prob})
    else:
        raise ValueError('not reachable')


def _get_filename_from_col(col: str, session: Session | None = None) -> str:
    """Get filename from specific column.

    :param col: column name in csv result file
    :param session: ``Session``
    :return: csv result filename (without file extension)
    :raise ValueError: when field not found
    """
    ret = None

    for code, info in CELLULAR_IO.items():
        for h in info.headers:
            if h.endswith('*'):
                if col.startswith(h[:-1]):
                    ret = info.directory
                    break
            elif col == h:
                ret = info.directory
                break

        if ret is not None:
            return ret if session is None else f'{ret}_{session}'

    raise ValueError(f'{col} not found')
