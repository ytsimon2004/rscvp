import collections
from typing import NamedTuple, NewType, Final

import numpy as np
import polars as pl
import scipy
from scipy.interpolate import interp1d
from typing_extensions import Self

from neuralib.util.verbose import fprint
from rscvp.util.cli.cli_io import get_headers_from_code, CELLULAR_IO
from rscvp.util.cli.cli_selection import SelectionOptions
from stimpyp.stimulus import GratingPattern, VisualParas, Direction, SFTF

__all__ = [
    'SFTF_LIT',
    'SFTF_IDX',
    'SFTF_ARRANGEMENT',
    'get_polar_sftf_order',
    'get_sftf_mesh_order',
    #
    'PrefSFTFParas',
    #
    'SFTFDirCollections'
]

SFTF_LIT = NewType('SFTF', str)

# used in visual polar legacy
SFTF_IDX: dict[int, SFTF_LIT] = {
    0: SFTF_LIT('0.04 1'),
    1: SFTF_LIT('0.08 1'),
    2: SFTF_LIT('0.16 1'),
    3: SFTF_LIT('0.04 4'),
    4: SFTF_LIT('0.08 4'),
    5: SFTF_LIT('0.16 4')
}

# used in st order. e.g., sftf_amp_0.04 1
SFTF_ARRANGEMENT: list[str] = CELLULAR_IO['st'].headers


def get_polar_sftf_order() -> list[SFTF_LIT]:
    return list(SFTF_IDX.values())


def get_sftf_mesh_order() -> list[SFTF_LIT]:
    """follow `SFTF_ARRANGEMENT` order"""
    s = SFTF_LIT
    return [s('0.04 1'), s('0.04 4'), s('0.08 1'), s('0.08 4'), s('0.16 1'), s('0.16 4')]


class PrefSFTFParas(NamedTuple):
    pref_sftf: np.ndarray
    """(N,) unit: cyc/degree, hz"""
    pref_dir: np.ndarray
    """(N,) unit: degree"""
    pref_osi: np.ndarray
    """(N,) unit: VALUE"""
    pref_dsi: np.ndarray
    """(N,) unit: VALUE"""

    @property
    def pref_ori(self) -> np.ndarray:
        return self.pref_dir % 180

    @classmethod
    def load_from_csv(cls, opt: SelectionOptions, use_cpx_index: bool = False) -> Self:
        """pick up ori, OSI, DSI from preferred sftf combination"""
        df: pl.DataFrame = opt.get_csv_data(cols=get_headers_from_code('pa'),
                                            to_numpy=False,
                                            enable_use_session=False)
        return cls._compute_pref(df, use_cpx_index)

    @classmethod
    def load_dataframe(cls, df: pl.DataFrame, use_cpx_index: bool = False) -> Self:

        if use_cpx_index:
            fprint(f'Use complex selectivity index')
        else:
            fprint(f'Use Non-complex selectivity index')

        return cls._compute_pref(df, use_cpx_index)

    @staticmethod
    def _compute_pref(df: pl.DataFrame, use_cpx_index=False) -> Self:
        p_sftf = df['preferred_sftf'].to_numpy()
        _idx = list(SFTF_IDX.values())
        p_idx = [_idx.index(it) + 1 for it in p_sftf]  # named from '1'

        p_ori = []
        p_osi = []
        p_dsi = []
        for i, idx in enumerate(p_idx):
            p_ori.append(df[f'preferred ori_{idx}'][i])

            if use_cpx_index:
                p_osi.append(df[f'OSI_{idx}_cpx'][i])
                p_dsi.append(df[f'DSI_{idx}_cpx'][i])
            else:
                p_osi.append(df[f'OSI_{idx}'][i])
                p_dsi.append(df[f'DSI_{idx}'][i])

        return PrefSFTFParas(
            p_sftf,
            np.array(p_ori),
            np.array(p_osi),
            np.array(p_dsi)
        )

    def with_mask(self, mask: np.ndarray) -> Self:
        return self._replace(
            pref_sftf=self.pref_sftf[mask],
            pref_dir=self.pref_dir[mask],
            pref_osi=self.pref_osi[mask],
            pref_dsi=self.pref_dsi[mask]
        )


# ======== #


class OriDirSelectivity(NamedTuple):
    """

    `Dimension parameters`:

        D = number of direction

        SFTF = number of SFTF
    """

    odsi: dict[SFTF, tuple[Direction, float, float]]
    """dict[(sf, tf)] = (dir, osi, dsi). cmp method used by Mao's paper"""

    odsi_cpx: dict[SFTF, tuple[float, float]]
    """dict[(sf, tf)] = (osi, dsi).  complex number method used by Ben's paper"""


class SFTFDirCollections:
    """
    `Dimension parameters`:

        D = number of direction

        SFTF = number of SFTF
    """
    __slots__ = ('grating', 'signal', 'image_time', 'norm', 'direction_invert', '_responses')

    def __init__(
            self,
            grating: GratingPattern,
            signal: np.ndarray,
            image_time: np.ndarray,
            *,
            norm: bool = False,
            direction_invert: bool = False
    ):
        """

        :param grating: ``GratingPattern``
        :param signal:
        :param image_time:
        :param norm: do maximal signal normalization foreach cell
        """
        self.grating = grating
        self.signal = signal
        self.image_time = image_time
        self.norm = norm
        self.direction_invert: Final[bool] = direction_invert

        # cache
        self._responses: dict[SFTF, list[tuple[Direction, float, float]]] | None = None

    def get_sftfdir_signal(self) -> dict[VisualParas, list[np.ndarray]]:
        """
        :return: Dictionary [(sf, tf, dir)] = sig  (N, t), N: number of sti. , t: time bins
            len(sig) indicates the number of stim
            the shape of each array indicates the time bins (varied across stimuli)
        """
        cy = collections.defaultdict(list)
        for si, st, sf, tf, dire in self.grating.foreach_stimulus():
            #
            dire = (dire + 180) % 360 if self.direction_invert else dire
            tx = np.logical_and(st[0] <= self.image_time, self.image_time <= st[1])
            sig = self.signal[tx]
            x = np.linspace(0.1, 0.9, num=len(sig))
            y = sig / np.max(self.signal) if self.norm else sig

            cy[(sf, tf, dire)].append((x, y))  # collect for doing average

        return cy

    def responses(self) -> dict[SFTF, list[tuple[Direction, float, float]]]:
        """
        :return: Dictionary dict[(sf, tf)] = list((dir, y_max, *y_sem)).
        **Note that keys are sorted**
        """
        if self._responses is None:
            cy = self.get_sftfdir_signal()
            oy = collections.defaultdict(list)
            for p, xy in cy.items():
                x = xy[-1][0]
                y = np.array([interp1d(it[0], it[1])(x) for it in xy])

                y_mean = np.mean(y, axis=0)  # trial avg
                y_max = np.max(y_mean)  # max resp. of certain sftf in certain direction

                y_sem = scipy.stats.sem(np.max(y, axis=1))
                oy[(p[0], p[1])].append((p[2], y_max, y_sem))

            self._responses = oy

        return self._responses

    @property
    def pref_sftf(self) -> SFTF:
        act = self.find_max_across_dir()
        perf_index = np.argmax(act)
        return list(self.responses().keys())[perf_index]

    def find_max_across_dir(self) -> np.ndarray:
        """(SFTF, )"""
        ret = []
        for _, res in self.responses().items():
            dat = np.array(res)  # (D, 3)
            val = np.mean(dat, axis=0)  # across dir (3,)
            ret.append(val[1])

        return np.array(ret)

    def get_pref_sftf_dir_response(self) -> np.ndarray:
        """(D, )"""
        pref_resp = self.responses().get(self.pref_sftf)
        return np.array(pref_resp)[:, 1]

    def get_ori_dir_selectivity(self) -> OriDirSelectivity:
        py = self.responses()

        # CMP method
        oy = {}  # dict[(sf, tf)]= (ori, osi, dsi)
        for p, v in py.items():  # p: (sf, tf); v: (ori, y_max, y_sem)
            v = np.array(v)
            v = v[np.argsort(v[:, 0])]  # sorted based orientation
            p_o = np.argmax(v[:, 1])  # preferred orientation index
            o_o1 = (p_o + 3) % v.shape[0]  # orthogonal orientation
            o_o2 = (p_o - 3) % v.shape[0]
            n_o = (p_o + 6) % v.shape[0]  # null (180 degree from preferred) orientation

            r_p = float(v[p_o, 1])  # response in preferred orientation
            r_or = float(v[o_o1, 1] + v[o_o2, 1]) / 2  # response in preferred orientation + 90 and -90
            r_null = float(v[n_o, 1])  # response in preferred orientation + 180

            osi = round(polar_selectivity_cmp(r_p, r_or), 3)
            dsi = round(polar_selectivity_cmp(r_p, r_null), 3)

            oy[p] = v[p_o, 0], osi, dsi

            # Complex number method
        iy = {}  # dict[(sf, tf)] = (osi, dsi)
        for p, v in py.items():  # p: (sf, tf); v: (ori, y_max, y_sem)
            resp = np.array(v)[:, 1]
            osi = round(polar_osi_cpx(resp), 3)
            dsi = round(polar_dsi_cpx(resp), 3)

            iy[p] = osi, dsi

        return OriDirSelectivity(oy, iy)

    def get_meshgrid_data(self, do_dir_avg: bool = True) -> np.ndarray:
        """Get mesh 2d data in sftf"""
        sf_i = self.grating.sf_i()
        tf_i = self.grating.tf_i()

        ret = np.zeros((len(sf_i), len(tf_i)))
        for (sf, tf), val in self.responses().items():
            v = np.array(val)
            r = np.mean(v[:, 1], axis=0) if do_dir_avg else np.max(v[:, 1], axis=0)
            ret[sf_i[sf], tf_i[tf]] = r

        return ret


def polar_selectivity_cmp(r_p: float, r_cmp: float):
    """ Calculate the OSI and DSI, which followed by Dun's thesis.
    Kerlin et al., 2010; Niell and Stryker, 2008; Anderman et al., 2011

    :param r_p: response in preferred direction
    :param r_cmp: response in compared direction (either 90 or 180° away from preferred direction)
    :return: OSI or DSI depends on ``r_cmp``
    """
    return (r_p - r_cmp) / (r_p + r_cmp)


def _circ_r(resp: np.ndarray, complex_angles: np.ndarray) -> tuple[float, float]:
    """

    .. seealso::

        ``res/matlab/visual/circ_r.m``

    :param resp:
    :param complex_angles:
    :return:
    """
    f = np.sum(resp * complex_angles)
    norm_length = np.abs(f) / np.sum(resp)
    angle = np.angle(f)

    return norm_length, angle


def polar_osi_cpx(resp: np.ndarray) -> float:
    """
    Calculate the orientation selective index using complex number method

    .. seealso::

        ``matlab/visual/calc_OSI.m``

    :param resp:mean(trial average) response during certain ori stimulation period (n_ori, )
    :return:
    """

    resp = np.mean(resp.reshape(-1, 2), axis=1)
    angles = np.linspace(0, np.pi, len(resp), endpoint=False)
    complex_angles = np.exp(2 * 1j * angles)
    idx, _ = _circ_r(resp, complex_angles)

    return idx


def polar_dsi_cpx(resp: np.ndarray) -> float:
    """
    Calculate the direction selective index using complex number method

    .. seealso::

        ``matlab/visual/calc_DSI.m``

    :param resp: mean(trial average) response during certain ori stimulation period (n_ori, )
    :return:
    """
    n_ori = len(resp)
    angles = np.linspace(0, 2 * np.pi, n_ori, endpoint=False)
    complex_angles = np.exp(1j * angles)
    idx, _ = _circ_r(resp, complex_angles)

    return idx
