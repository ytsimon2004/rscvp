from typing import Literal, cast, get_args

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from argclz import argument, AbstractParser
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.plot.colormap import get_customized_cmap
from neuralib.typing import AxesArray, PathLike
from neuralib.util.utils import keys_with_value
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import DataOutput, SQLDatabaseOptions, SelectionOptions
from rscvp.util.cli.cli_io import CELLULAR_IO
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from rscvp.util.database import VisualSFTFDirDB
from rscvp.visual.util import PrefSFTFParas, SFTFDirCollections, SFTF_IDX, SFTF_LIT
from rscvp.visual.util_plot import selective_pie, dir_hist
from stimpyp import GratingPattern, Direction

__all__ = [
    'VisualPolarOptions',
    'BaseVisPolarOptions',
    'plot_osi_dsi_all'
]


class BaseVisPolarOptions:
    direction_invert: bool = argument(
        '--direction-invert',
        help='direction invert for the same orientation (different temporal/nasal direction).'
             'i.e., stimpy & KS paper had different definition:'
             'In stimpy: 180 degree present horizontal from temporal to nasal direction, but 0 degree in KS paper'
    )

    use_cpx_selective_index: bool = argument(
        '--cpx',
        help='use direction/orientation selective index which calculated by complex circular variance'
    )

    _selective_thres: float = argument(
        '--thres',
        metavar='VALUE',
        default=0.33,
        help='threshold for osi, dsi'
    )

    @property
    def selective_thres(self) -> float:
        return 0.2 if self.use_cpx_selective_index else self._selective_thres


POLAR_STYLE = Literal['line_sem', 'color_angle']


@publish_annotation('main', project='rscvp', figure='fig.5B & fig.5E-H upper', caption='db usage', as_doc=True)
class VisualPolarOptions(AbstractParser, SelectionOptions, BaseVisPolarOptions, SQLDatabaseOptions):
    DESCRIPTION = """
    Plot the neural responses in different combination of sf-tf.
    Different direction of visual stimulus represents as polar plot.
    """

    summary: bool = argument('--summary', help='plot OSI, DSI hist')

    polar_style: POLAR_STYLE = argument('--ps', '--polar-style', default='line_sem', help='polar plot style')

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.summary:
            self.reuse_output = True
            self.pre_selection = True
            self.vc_selection = 0.3

        if self.polar_style not in get_args(POLAR_STYLE):
            raise ValueError(f'invalid polars style: {self.polar_style}')

    def run(self):
        self.post_parsing()
        output_info = self.get_data_output('pa')

        if self.summary:
            if self.plane_index is not None:
                df = pl.read_csv(output_info.csv_output)
            else:
                df = pl.read_csv(self.concat_csv_path)

            vp = PrefSFTFParas.load_dataframe(df, use_cpx_index=self.use_cpx_selective_index)
            self.plot_osi_dsi_sum(vp, output_info)
        else:
            self.foreach_visual_polars(output_info, self.neuron_id)

    def foreach_visual_polars(self, output: DataOutput, neuron_ids: NeuronID):
        """
        Plot the polar plot of neural responses in different combination of sf-tf, direction visual stimulus,
        print the OSI, DSI, and save as csv

        :param output: figure saving file path
        :param neuron_ids:
        :return:
        """
        from tqdm import tqdm

        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        riglog = self.load_riglog_data()
        image_time = riglog.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        grating = GratingPattern.of(riglog)

        headers = self._polars_header_handle(grating)
        with csv_header(output.csv_output, headers) as csv:
            for neuron_id in tqdm(neuron_list, desc='visual_polar', unit='neuron', ncols=80):
                signal = get_neuron_signal(s2p, neuron_id)[0]

                coll = SFTFDirCollections(grating, signal, image_time)

                ret = plot_sftf_dir_polar(coll,
                                          polar_style=self.polar_style,
                                          output_file=output.figure_output(neuron_id))
                csv(neuron_id, *ret)

    @staticmethod
    def _polars_header_handle(pattern: GratingPattern) -> list[str]:
        """legacy handling of the csv headers"""
        header = ['neuron_id']
        if pattern.n_sftf == 6:
            header += CELLULAR_IO['pa'].headers  # OSI_{*} idx referred to `SFTF_IDX`
        else:
            header += ['preferred_sftf', 'ori_resp']
            sftf = pattern.sftf_i()
            indices = []
            for st, i in sftf.items():
                value = " ".join(map(str, st))
                idx = keys_with_value(SFTF_IDX, SFTF_LIT(value)) + 1  # 1-base
                indices.append(idx)

            for i in sorted(indices):
                header += [f'preferred ori_{i}', f'OSI_{i}', f'DSI_{i}', f'OSI_{i}_cpx', f'DSI_{i}_cpx']

        return header

    def plot_osi_dsi_sum(self, vp: PrefSFTFParas, output: DataOutput):
        """
        Plot the osi dsi summary, and populate the results into the database

        :param vp: ``PrefSFTFParas``
        :param output: ``DataOutput``
        :return:
        """
        mx = self.get_selected_neurons()
        vp = vp.with_mask(mx)

        dmx = vp.pref_dsi >= self.selective_thres
        omx = vp.pref_osi >= self.selective_thres

        # database
        self.populate_database(dmx, omx)

        dire = vp.pref_dir[dmx]
        ori = vp.pref_ori[omx]

        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            f'{self.vc_selection}' if self.vc_selection is not None else None
        )

        with plot_figure(output_file, 2, 3, figsize=(12, 8)) as ax:
            plot_osi_dsi_all(ax, vp.pref_dsi, vp.pref_osi, dire, ori, self.selective_thres)

    # ======== #
    # Database #
    # ======== #

    def populate_database(self, dmx: np.ndarray, omx: np.ndarray):
        region = self.get_primary_key_field('region') if self.rec_region is None else self.rec_region

        update_fields = dict(
            n_ds_neurons=np.count_nonzero(dmx),
            n_os_neurons=np.count_nonzero(omx),
            os_frac=np.mean(omx),
            ds_frac=np.mean(dmx),
            update_time=self.cur_time
        )

        db = VisualSFTFDirDB(
            date=self.exp_date,
            animal=self.animal_id,
            rec=self.daq_type,
            user=self.username,
            optic=f'{self.plane_index}' if self.plane_index is not None else 'all',
            region=region,
            pair_wise_group=self.get_primary_key_field('pair_wise_group'),
            **update_fields
        )

        print('NEW', db)
        if self.db_commit:
            self.add_data(db)
        else:
            print('use --commit to perform database operations')


# ========== #
# Plot Polar #
# ========== #

def plot_sftf_dir_polar(coll: SFTFDirCollections,
                        polar_style: POLAR_STYLE = 'color_angle',
                        output_file: PathLike | None = None) -> list[str]:
    """

    :param coll: ``SFTFDirCollections``
    :param polar_style
    :param output_file:
    :return:
    """
    grating = coll.grating

    with plot_figure(output_file,
                     grating.n_tf,
                     grating.n_sf,
                     default_style=False,
                     tight_layout=False,
                     subplot_kw=dict(polar=True)) as ax:
        _plot_sftf_dir_polar(ax, coll, polar_style)
        fig = plt.gcf()
        fig.suptitle(f'preferred sftf: {coll.pref_sftf}')
        fig.supxlabel(f'SF: {grating.sf_set}')
        fig.supylabel(f'TF: {grating.tf_set}')

    #
    ret = []
    dire = coll.get_pref_sftf_dir_response()
    ret.extend([' '.join(map(str, coll.pref_sftf))])  # preferred sftf combination
    ret.extend([' '.join(map(str, dire))])  # all ori resp. in preferred sftf combination

    #
    selectivity = coll.get_ori_dir_selectivity()
    odsi = selectivity.odsi
    odsi_cpx = selectivity.odsi_cpx
    for tf in sorted(grating.tf_set):
        for sf in sorted(grating.sf_set):
            ret.extend(odsi[sf, tf])

    for tf in sorted(grating.tf_set):
        for sf in sorted(grating.sf_set):
            ret.extend(odsi_cpx[sf, tf])

    return [str(it) for it in ret]


def _plot_sftf_dir_polar(ax: AxesArray,
                         coll: SFTFDirCollections,
                         polar_style: POLAR_STYLE = 'color_angle',
                         with_grid: bool = True,
                         with_sem: bool = True):
    sf_x = coll.grating.sf_i()
    tf_y = coll.grating.tf_i()

    for tf_i, tf in enumerate(tf_y):
        for sf_i, sf in enumerate(sf_x):
            if isinstance(ax, np.ndarray):
                _ax = ax[-tf_i - 1, sf_i]  # axs[0] from top, but tf[0] from bottom
            else:
                _ax = cast(Axes, ax)

            ori_x, y, y_sem = zip(*sorted(coll.responses()[sf, tf]))
            p_ori, osi, dsi = coll.get_ori_dir_selectivity().odsi[sf, tf]
            _plot_polar(_ax, ori_x, y, y_sem, p_ori, osi, dsi, polar_style, with_sem)

            if not with_grid:
                _ax.grid(False)


def _plot_polar(ax: Axes,
                ori_x: tuple[Direction, ...],
                y: tuple[float, ...],
                y_sem: tuple[float, ...],
                p_ori: float,
                osi: float,
                dsi: float,
                polar_style: POLAR_STYLE = 'color_angle',
                with_sem: bool = False):
    ori_x = np.array(np.deg2rad(ori_x))
    y = np.array(y)

    if polar_style == 'line_sem':
        y_sem = np.array(y_sem)

        # for closed polars
        ori_x = np.append(ori_x, ori_x[0])
        y = np.append(y, y[0])
        y_sem = np.append(y_sem, y_sem[0])

        ax.plot(ori_x, y, color='r')

        if with_sem:
            ax.fill_between(ori_x, y - y_sem, y + y_sem, color='r', alpha=0.3)

    elif polar_style == 'color_angle':
        colors = get_customized_cmap('twilight', (0, 1), len(ori_x))
        ax.bar(ori_x, y, color=colors, width=0.5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    ax.set_title(f'p_ori: {p_ori}° \n OSI: {osi} \n DSI: {dsi}', fontsize=8)


def plot_osi_dsi_all(ax: AxesArray,
                     p_dsi: np.ndarray,
                     p_osi: np.ndarray,
                     dire: np.ndarray,
                     ori: np.ndarray,
                     selective_thres: float):
    """Plot all the osi/dsi histogram and pie chart"""
    n_neurons = len(p_dsi)
    n_dire_neurons = len(dire)
    n_ori_neurons = len(ori)

    dir_hist(p_dsi, thres=selective_thres, ax=ax[0, 0], xlabel='DSI')
    dir_hist(p_osi, thres=selective_thres, ax=ax[1, 0], xlabel='OSI')

    selective_pie([n_dire_neurons, n_neurons - n_dire_neurons], ['direction selective', None], ax=ax[0, 1])
    selective_pie([n_ori_neurons, n_neurons - n_ori_neurons], ['orientation selective', None], ax=ax[1, 1])

    dir_hist(dire, bins=12, xlim=(0, 360), ax=ax[0, 2], xlabel='Preferred direction(deg)')
    ax[0, 2].set_xticks([i * 90 for i in range(5)])
    ax[0, 2].set_title(f'{n_dire_neurons} / {n_neurons}')

    dir_hist(ori, bins=6, xlim=(0, 180), ax=ax[1, 2], xlabel='Preferred orientation(deg)')
    ax[1, 2].set_xticks([i * 45 for i in range(5)])
    ax[1, 2].set_title(f'{n_ori_neurons} / {n_neurons}')


if __name__ == '__main__':
    VisualPolarOptions().main()
