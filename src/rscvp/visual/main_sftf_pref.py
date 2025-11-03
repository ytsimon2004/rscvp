from typing import Literal

import numpy as np
import polars as pl
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import dotplot
from neuralib.plot import plot_figure
from neuralib.util.verbose import fprint, publish_annotation
from rscvp.util.cli import DataOutput, SelectionOptions, SQLDatabaseOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from rscvp.util.database import VisualSFTFDirDB
from rscvp.visual.util import SFTF_ARRANGEMENT, SFTFDirCollections
from stimpyp import GratingPattern

__all__ = ['VisualSFTFPrefOptions']


@publish_annotation('main', project='rscvp', figure='fig.5C left', caption='db usage', as_doc=True)
class VisualSFTFPrefOptions(AbstractParser, SelectionOptions, SQLDatabaseOptions):
    DESCRIPTION = """
    Plot the visual sftf preference in dot plot (either fraction of cell, or activity amplitude).
    NOTE that this script should run `main_polar` first.
    """

    max_dir: bool = argument(
        '--max-dir',
        help='pick up maximal resp. if not, do the resp. average toward all the direction',
    )

    summary_type: Literal['dff', 'fraction'] | None = argument(
        '--summary',
        default=None,
        help='plot summary from population neurons'
    )

    signal_type = 'df_f'

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.summary_type is not None:
            self.pre_selection = True
            self.vc_selection = 0.3
            self.reuse_output = True

    def run(self):
        self.post_parsing()

        riglog = self.load_riglog_data()
        pattern = GratingPattern.of(riglog)
        output_info = self.get_data_output('st')

        match self.summary_type:
            case 'dff':
                self.plot_dff_all(pattern, output_info)
            case 'fraction':
                self.plot_fraction_all(pattern, output_info)
            case None:
                self.foreach_sftf_preference(pattern, self.neuron_id, output_info)
            case _:
                raise ValueError(f'{self.summary_type} unknown')

    def foreach_sftf_preference(self, para: GratingPattern, neuron_ids: NeuronID, output: DataOutput):
        """
        plot the preferred sftf (df_f) in dot plot per cell
        ** DO NOT do the 01 normalization of the dff signal **
        """
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)
        riglog = self.load_riglog_data()
        image_time = sync_s2p_rigevent(riglog.imaging_event.time, s2p, self.plane_index)
        sig = get_neuron_signal(s2p, neuron_list, normalize=False)[0]

        # header
        sftf_ls = list(para.sftf_i())
        sftf_ls = ['sftf_amp_' + ' '.join(map(str, it)) for it in sftf_ls]
        assert sftf_ls == SFTF_ARRANGEMENT, "check legacy header issue"
        header = ['neuron_id'] + sftf_ls

        # plot/csv
        with csv_header(output.csv_output, header) as csv:
            for i in tqdm(neuron_list, desc='preferred_sftf', unit='neuron', ncols=80):
                i = 0 if sig.shape[0] == 1 else i  # index correction
                data = (
                    SFTFDirCollections(para, sig[i], image_time, norm=True)
                    .get_meshgrid_data(do_dir_avg=~self.max_dir)
                )
                data[data <= 0] = 0  # exclude special case with negative value (i.e., neuropil correction error)
                csv(i, *data.flatten())  # normalized dff (0-1) per cells

                with plot_figure(output.figure_output(i), default_style=False, set_square=True) as ax:
                    dotplot(para.sf_set.astype(str),
                            para.tf_set.astype(str),
                            data,
                            size_legend_as_int=False,
                            scale='area',
                            ax=ax)
                    ax.set(xlabel='TF (Hz)', ylabel='SF (cyc/deg)')

    def plot_dff_all(self, para: GratingPattern, output: DataOutput):
        """plot the dff response amplitude by averaging `all the visual neurons`
        in sftf condition (avg dir or max dir)"""
        if self.plane_index is not None:
            sftf_act = np.loadtxt(output.csv_output, skiprows=1, delimiter=',')[:, 1:]
        else:
            cname = SFTF_ARRANGEMENT
            sftf_act = self.get_csv_data(cname)

        cell_mask = self.get_selected_neurons()
        ret = np.median(sftf_act[cell_mask], axis=0)  # avoid transient
        med_act = ret.reshape(3, 2)  # (sf, tf)

        # for median dff value
        dy = dict(index='dff')
        dy.update({sftf: [ret[i]] for i, sftf in enumerate(SFTF_ARRANGEMENT)})
        tmp = pl.DataFrame(dy)
        tmp.write_csv(output.mk_subdir('tmp', 'preferred_sftf_dff', '.csv'))
        self.populate_database(tmp)

        #
        output_file = output.summary_figure_output(
            'dff',
            'pre' if self.pre_selection else None,
            f'vc{self.vc_selection}' if self.vc_selection is not None else None
        )
        with plot_figure(output_file, default_style=False, set_square=True) as ax:
            dotplot(para.sf_set.astype(str),
                    para.tf_set.astype(str),
                    med_act,
                    with_color=True,
                    scale='area',
                    size_legend_as_int=False,
                    ax=ax)
            ax.set_title(f'n_neurons: {np.count_nonzero(cell_mask)}')
            ax.set(xlabel='TF (Hz)', ylabel='SF (cyc/deg)')

    def plot_fraction_all(self, para: GratingPattern, output: DataOutput):
        """Plot the fraction of the preferred sftf in dot size"""
        df = self.get_csv_data('preferred_sftf', to_numpy=False, enable_use_session=False)
        cell_mask = self.get_selected_neurons()
        df = df.filter(cell_mask)
        n_neurons = df.shape[0]
        c = df.group_by("preferred_sftf", maintain_order=True).agg(pl.len())

        num_frac = {
            f'sftf_amp_{sftf}': [num, (num / n_neurons)]
            for sftf, num in c.iter_rows()
        }

        # for mean fraction value
        dy = dict(index=['n_visual_cells', 'fraction'])
        for sftf in SFTF_ARRANGEMENT:
            try:
                num, fraction = num_frac[sftf]
            except KeyError as e:
                num = 0
                fraction = 0
                fprint(f'no cell count in sftf set: {e}', vtype='warning')

            dy[sftf] = [num, fraction]

        tmp = pl.DataFrame(dy, strict=False)
        tmp.write_csv(output.mk_subdir('tmp', 'preferred_sftf_fraction', '.csv'))

        if not self.db_debug_mode:
            self.populate_database(tmp)

        #
        output_file = output.summary_figure_output(
            'fraction',
            'pre' if self.pre_selection else None,
            f'vc{self.vc_selection}' if self.vc_selection is not None else None
        )

        with plot_figure(output_file, default_style=False, set_square=True) as ax:
            frac = (tmp.filter(pl.col('index') == 'fraction')
                    .drop('index')
                    .to_numpy()
                    .reshape(3, 2))

            dotplot(para.sf_set.astype(str),
                    para.tf_set.astype(str),
                    frac,
                    scale='area',
                    size_legend_as_int=False,
                    ax=ax)

            ax.set_title(f'n_neurons: {n_neurons}')
            ax.set(xlabel='TF (Hz)', ylabel='SF (cyc/deg)')

    # ======== #
    # Database #
    # ======== #

    def populate_database(self, df: pl.DataFrame) -> None:
        region = self.get_primary_key_field('region') if self.rec_region is None else self.rec_region

        sftf = list(df.row(by_predicate=pl.col('index') == self.summary_type)[1:])

        update_fields = dict(update_time=self.cur_time)

        if self.summary_type == 'fraction':
            kwargs = {f'sftf_frac_group{i + 1}': sftf[i] for i in range(len(sftf))}
        elif self.summary_type == 'dff':
            kwargs = {f'sftf_amp_group{i + 1}': sftf[i] for i in range(len(sftf))}
        else:
            raise ValueError(f'{self.summary_type}')

        update_fields = {**update_fields, **kwargs}

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

        self.print_update(db, update_fields)


if __name__ == '__main__':
    VisualSFTFPrefOptions().main()
