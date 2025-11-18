import numpy as np
import polars as pl

from argclz import AbstractParser, try_int_type, as_argument
from neuralib.plot import plot_figure
from rscvp.statistic._var import VIS_DIR_HEADERS
from rscvp.statistic.csv_agg.core import ParquetSheetSync, NeuronDataAggregator
from rscvp.util.cli.cli_io import get_headers_from_code
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.visual.main_polar import plot_osi_dsi_all, BaseVisPolarOptions
from rscvp.visual.util import get_polar_sftf_order, PrefSFTFParas

__all__ = ['VisDirAggOption']


class VisDirAggOption(AbstractParser, StatisticOptions, BaseVisPolarOptions):
    DESCRIPTION = 'OSI DSI preferred in aRSC and pRSC'

    header = as_argument(StatisticOptions.header).with_options(choices=VIS_DIR_HEADERS)

    pre_selection = True
    vc_selection = 0.3

    def run(self):
        vzdir = VisDirStat(self)

        if self.update:
            vzdir.update_sync(self.variable)

        self.plot_osi_dsi_regions(vzdir, 'aRSC')
        self.plot_osi_dsi_regions(vzdir, 'pRSC')

    def plot_osi_dsi_regions(self, vzdir: 'VisDirStat', region: str):
        """plot the osi, dsi plot for certain region (could be batch dataset)"""

        mask = vzdir.df['region'] == region
        vp = vzdir.process().with_mask(mask)

        dire = vp.pref_dir[(vp.pref_dsi >= self.selective_thres)]
        ori = vp.pref_ori[(vp.pref_osi >= self.selective_thres)]

        output_file = vzdir.opt.statistic_dir / 'visual_dir' / region
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with plot_figure(output_file, 2, 3, figsize=(12, 8)) as ax:
            plot_osi_dsi_all(ax, vp.pref_dsi, vp.pref_osi, dire, ori, self.selective_thres)


class VisDirStat(ParquetSheetSync):
    """Visual direction/orientation statistic"""

    # should be same order as SFTF_IDX in main_polar.py
    SFTF = get_polar_sftf_order()

    def __init__(self, opt: StatisticOptions, sftf: list[str] = None):
        collector = NeuronDataAggregator(
            'pa',
            stat_col=get_headers_from_code('pa'),
            fields=dict(rec_region=str, plane_index=try_int_type)
        )

        super().__init__(opt, sheet_page='visual_parq', aggregator=collector)

        # sort based on tf order, transform due to different orders as D,OSI_{x}
        self.sftf = sftf or self.SFTF

    def process(self) -> PrefSFTFParas:
        return PrefSFTFParas.load_dataframe(self.df)

    def filter_concat_df(self) -> pl.DataFrame:
        """pick up the preferred sftf"""
        df = self.df

        # preferred sftf idx for osi dsi in all cell, len: n_neurons
        p_sftf = df['preferred_sftf'].to_numpy()
        p_idx: list[int] = [self.sftf.index(j) + 1 for j in p_sftf]

        p_dir = np.zeros(len(p_idx))
        osi = np.zeros(len(p_idx))
        dsi = np.zeros(len(p_idx))
        for i, idx in enumerate(p_idx):
            p_dir[i] = df[f'preferred ori_{idx}'][i]  # legacy naming (ori -> dir)
            osi[i] = df[f'OSI_{idx}'][i]
            dsi[i] = df[f'DSI_{idx}'][i]

        ret = {
            'Data': df['Data'],
            'dsi': dsi,
            'osi': osi,
            'pdir': p_dir,
            'pori': p_dir % 180
        }

        return pl.DataFrame(ret)


if __name__ == '__main__':
    VisDirAggOption().main()
