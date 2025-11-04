import collections
from typing import Final

import numpy as np
import pandas as pd
import polars as pl

from argclz import AbstractParser, try_int_type, as_argument
from neuralib.plot import dotplot
from neuralib.plot import plot_figure
from neuralib.util.verbose import fprint
from rscvp.statistic._var import VIS_SFTF_HEADERS
from rscvp.statistic.csv_agg.collector import CSVCollector
from rscvp.statistic.csv_agg.core import LocalSpreadsheetSync
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.visual.util import SFTF_ARRANGEMENT, get_sftf_mesh_order

__all__ = ['VZSFTFAggOption']


class VZSFTFStat(LocalSpreadsheetSync):
    """visual SFTF preference properties statistic"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='st',
            stat_col=SFTF_ARRANGEMENT,
            fields=dict(rec_region=str, plane_index=try_int_type)
        )

        super().__init__(opt, sheet_page='visual_parq', collector=collector)


class VZSFTFAggOption(AbstractParser, StatisticOptions):
    DESCRIPTION = 'compare the visual spatiotemporal preference properties in apRSC'

    SFTF = get_sftf_mesh_order()
    SF_SET: Final = ['0.04', '0.08', '0.16']  # hard-code for plotting
    TF_SET: Final = ['1', '4']

    header: str = as_argument(StatisticOptions.header).with_options(choices=VIS_SFTF_HEADERS)

    pre_selection = True
    vc_selection = 0.3

    def run(self):
        vzsftf = VZSFTFStat(self)
        match self.header:
            case 'fraction':
                self.plot_dot_fraction(vzsftf.df)
            case 'dff':
                self.plot_dot_dff(vzsftf.df)
            case s if s.startswith('sftf_amp_'):
                if self.update:
                    vzsftf.update_spreadsheet(self.variable)

    def plot_dot_dff(self, df: pl.DataFrame):
        """
        .. seealso::

            rscvp.visual.main_sftf_pref.SFTFPrefOptions.plot_dff_all
        """
        header = ['sftf_amp_' + it for it in self.SFTF]

        amp_a = df.filter(pl.col('region') == 'aRSC')[header].to_numpy()
        amp_b = df.filter(pl.col('region') == 'pRSC')[header].to_numpy()

        med_act_arsc = np.median(amp_a, axis=0).reshape(3, 2)  # avoid transient
        med_act_prsc = np.median(amp_b, axis=0).reshape(3, 2)

        output = self.statistic_dir / 'sftf_dot_plot' / 'sftf_dot_dff.pdf'
        output.parent.mkdir(parents=True, exist_ok=True)

        with plot_figure(output, 1, 2, set_square=True) as ax:
            dotplot(self.SF_SET,
                    self.TF_SET,
                    med_act_arsc,
                    with_color=True,
                    size_legend_as_int=False,
                    ax=ax[0])
            ax[0].set(title='aRSC', xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')

            dotplot(self.SF_SET,
                    self.TF_SET,
                    med_act_prsc,
                    with_color=True,
                    size_legend_as_int=False,
                    ax=ax[1])
            ax[1].set(title='pRSC', xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')

    def _pick_pref_sftf(self, df: pd.DataFrame, expr=None) -> list[str]:
        """pick up the preferred sftf based on vzsftf concat dataframe

        :return list of SFTF string. i.e., ['0.08 1', '0.04 4', ...]
        """
        header = ['sftf_amp_' + it for it in self.SFTF]
        df = df.filter(expr) if expr is not None else df

        dff = df[header].to_numpy()
        indices = np.argmax(dff, axis=1)
        return [self.SFTF[idx] for idx in indices]

    def _calc_fraction(self, df: pl.DataFrame, expr: pl.Expr = None) -> np.ndarray:
        """calculate the fraction of visual cells under each sftf condition

        :return fraction. (SF, TF)
        """
        n_neurons = df.filter(expr).shape[0] if expr is not None else df.shape[0]
        c = collections.Counter(self._pick_pref_sftf(df, expr))

        frac = {f'sftf_amp_{sftf}': (num / n_neurons)
                for sftf, num in c.items()}

        dy = {}
        for sftf in SFTF_ARRANGEMENT:
            try:
                fraction = frac[sftf]
            except KeyError as e:
                fraction = 0
                fprint(f'no cell count in sftf set: {e}', vtype='warning')

            dy[sftf] = fraction

        return np.array(list(dy.values())).reshape(3, 2)

    def plot_dot_fraction(self, df: pd.DataFrame):
        """
         ..seealso::

               rscvp.visual.main_sftf_pref.SFTFPrefOptions.plot_fraction_all
        """
        arsc_fraction = self._calc_fraction(df, pl.col('region') == 'aRSC')
        prsc_fraction = self._calc_fraction(df, pl.col('region') == 'pRSC')

        output = self.statistic_dir / 'sftf_dot_plot' / 'sftf_dot_fraction.pdf'
        output.parent.mkdir(parents=True, exist_ok=True)
        with plot_figure(output, 1, 2) as ax:
            dotplot(self.SF_SET, self.TF_SET, arsc_fraction, size_legend_as_int=False, ax=ax[0])
            ax[0].set(title='aRSC', xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')

            dotplot(self.SF_SET, self.TF_SET, prsc_fraction, size_legend_as_int=False, ax=ax[1])
            ax[1].set(title='pRSC', xlabel=f'TF (Hz)', ylabel=f'SF (cyc/deg)')


if __name__ == '__main__':
    VZSFTFAggOption().main()
