import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from argclz import AbstractParser, argument
from neuralib.io import csv_header
from neuralib.plot import plot_figure, ax_set_default_style, VennDiagram
from neuralib.plot.plot import axvline_histplot
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import DataOutput, PlotOptions, SelectionOptions, SelectionMask, SQLDatabaseOptions
from rscvp.util.database import GenericDB, DarknessGenericDB, BlankBeltGenericDB

__all__ = ['ClsCellTypeOptions']


@publish_annotation('main', project='rscvp', caption='db usage', as_doc=True)
class ClsCellTypeOptions(AbstractParser, SelectionOptions, PlotOptions, SQLDatabaseOptions):
    DESCRIPTION = 'Quantification of proportion of visual/spatial/overlap/unclassified RSC neurons'

    blankbelt_db: bool = argument('--blankbelt-db', help='populate to blank belt db instead of protocol based')

    pre_selection = True
    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.populate_database()

    # ======== #
    # Database #
    # ======== #

    def populate_database(self):
        if self.blankbelt_db:
            db = self._populate_blankbelt_database()
        else:
            db = self._populate_protocol_database()
            self.plot()

        print('NEW', db)
        if self.db_commit:
            self.add_data(db)
        else:
            print('use --commit to perform database operations')

    def _populate_blankbelt_database(self) -> BlankBeltGenericDB:
        self.vc_selection = 0.2  # for active neuron purpose
        mask = self.get_selection_mask()
        region = self.get_primary_key_field('region', page='apcls_blank')

        return BlankBeltGenericDB(
            date=self.exp_date,
            animal=self.animal_id,
            rec=self.daq_type,
            user=self.username,
            optic=self.plane_index if self.plane_index is not None else 'all',
            region=region,
            n_total_neurons=self.n_total_neurons,
            n_selected_neurons=mask.n_neurons,
            n_spatial_neurons=np.count_nonzero(mask.place_mask),
            update_time=self.cur_time
        )

    def _populate_protocol_database(self) -> GenericDB | DarknessGenericDB:
        if self.is_ldl_protocol:
            self.vc_selection = None
            db = self._populate_database_ldl()
        elif self.is_vop_protocol:
            self.vc_selection = 0.3
            db = self._populate_database_vop()
        else:
            raise ValueError('unsupported protocol')

        return db

    def _populate_database_vop(self) -> GenericDB:
        mask = self.get_selection_mask()
        region = self.get_primary_key_field('region')
        n_planes = self.get_primary_key_field('n_planes')

        return GenericDB(
            date=self.exp_date,
            animal=self.animal_id,
            rec=self.daq_type,
            user=self.username,
            optic=self.plane_index if self.plane_index is not None else 'all',
            n_planes=n_planes,
            region=region,
            pair_wise_group=self.get_primary_key_field('pair_wise_group'),
            n_total_neurons=self.n_total_neurons,
            n_selected_neurons=mask.n_neurons,
            n_spatial_neurons=np.count_nonzero(mask.place_mask),
            n_visual_neurons=np.count_nonzero(mask.visual_mask),
            n_overlap_neurons=np.count_nonzero(mask.overlap_mask),
            update_time=self.cur_time
        )

    def _populate_database_ldl(self) -> DarknessGenericDB:
        region = self.get_primary_key_field('region', page='ap_ldl')
        n_planes = self.get_primary_key_field('n_planes')

        return DarknessGenericDB(
            date=self.exp_date,
            animal=self.animal_id,
            rec=self.daq_type,
            user=self.username,
            optic=self.plane_index if self.plane_index is not None else 'all',
            n_planes=n_planes,
            region=region,
            n_total_neurons=self.n_total_neurons,
            n_selected_neurons=self.n_selected_neurons,
            n_spatial_neurons_light_bas=np.count_nonzero(self.select_place_neurons('slb', force_session='light_bas')),
            n_spatial_neurons_dark=np.count_nonzero(self.select_place_neurons('slb', force_session='dark')),
            n_spatial_neurons_light_end=np.count_nonzero(self.select_place_neurons('slb', force_session='light_end')),
            update_time=self.cur_time
        )

    # ======== #
    # Plotting #
    # ======== #

    def plot(self):
        output_info = self.get_data_output('cls')
        if self.is_ldl_protocol:
            self.plot_lower_bound(output_info)
        else:
            self.plot_cell_type_summary(output_info)

    def plot_cell_type_summary(self, output: DataOutput, verbose: bool = True):
        mask = self.get_selection_mask()
        n_total = mask.n_neurons
        n_visual = mask.n_visual
        n_place = mask.n_place
        n_overlap = mask.n_overlap

        #
        visual_rel = self.get_csv_data('reliability', enable_use_session=False)
        spatial_info = self.get_csv_data(f'si_{self.session}')
        lower_bound = self.get_csv_data(f'nbins_exceed_{self.session}')

        if verbose:
            print(mask)

        output_file = output.summary_figure_output()
        with csv_header(output.csv_output, ['visual_cell(%)', 'place_cell(%)', 'overlap(%)', 'ch2(%)']) as csv:
            with plot_figure(output_file, 2, 2, figsize=(10, 10)) as ax:
                #
                fraction_vc = f'{n_visual} / {n_total}'
                axvline_histplot(ax[0, 0],
                                 visual_rel,
                                 cutoff=0.3,
                                 xlabel='Visual reliability',
                                 ylabel='population (%)',
                                 title=f'visual_cell: {n_visual}/{n_total}')
                #
                fraction_pc = f'{n_place} / {n_total}'
                axvline_histplot(ax[0, 1],
                                 lower_bound,
                                 cutoff=1,
                                 xlabel='lower bound activity exceed (bin)',
                                 ylabel='population (%)',
                                 title=f'place_cell: {n_place}/{n_total}')

                fraction_overlap = f'{n_overlap} / {n_total}'

                self.plot_scatter_hist(ax[1, 0], spatial_info, visual_rel, mask)
                self.venn_plot(ax[1, 1], mask)

                fraction_ch2 = f'{np.count_nonzero(mask.ch2_mask)} / {n_total}' if self.has_chan2 else 'None'
                csv(eval(fraction_vc), eval(fraction_pc), eval(fraction_overlap), eval(fraction_ch2))
                csv(fraction_vc, fraction_pc, fraction_overlap, fraction_ch2)

    def plot_lower_bound(self, output: DataOutput):
        n_total = self.n_total_neurons
        n_place_cells = np.count_nonzero(self.select_place_neurons('slb'))
        output_file = output.summary_figure_output()

        with plot_figure(output_file, figsize=(10, 10)) as ax:
            axvline_histplot(ax,
                             self.get_csv_data(f'nbins_exceed_{self.session}'),
                             cutoff=1,
                             xlabel='lower bound activity exceed (bin)',
                             ylabel='population (%)',
                             title=f'place_cell: {n_place_cells}/{n_total}')

    @staticmethod
    def venn_plot(ax, mask: SelectionMask):
        n_visual = mask.n_visual
        n_place = mask.n_place
        n_overlap = mask.n_overlap

        vd = VennDiagram({'visual': n_visual - n_overlap, 'spatial': n_place - n_overlap}, ax=ax)
        vd.add_intersection('visual & spatial', n_overlap)
        vd.add_total(mask.n_neurons)
        vd.plot()

    @staticmethod
    def plot_scatter_hist(ax: Axes, v1: np.ndarray, v2: np.ndarray, m: SelectionMask):
        # n include overlap
        ax.plot(v1[m.visual_mask], v2[m.visual_mask], 'ro', markersize=3, label=f'visual cell(n={m.n_visual})')
        ax.plot(v1[m.place_mask], v2[m.place_mask], 'go', markersize=3, label=f'place cell(n={m.n_place})')
        ax.plot(v1[m.unclass_mask], v2[m.unclass_mask], 'ko', markersize=3, label=f'unclassified(n={m.n_unclass})')
        ax.plot(v1[m.overlap_mask], v2[m.overlap_mask], 'gold', marker='o', markersize=3, ls='',
                label=f'overlap(n={m.n_overlap})')

        if m.ch2_mask is not None:
            ax.plot(v1[m.ch2_mask], v2[m.ch2_mask], 'bs', markersize=5, fillstyle='none',
                    label=f'red labeled cell(n={np.count_nonzero(m.ch2_mask)})')

        ax.set_xlabel('spatial info.')
        ax.set_ylabel('visual reliability')

        sp = 0.1
        ax.set_xlim(0 - sp, sp + np.max(v1))
        ax.set_ylim(0 - sp / 2, sp / 2 + np.max(v2))
        ax_set_default_style(ax)
        ax.legend()

        # hist
        # create new axes on the right and on the top of the current axes
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes('top', 1, pad=0.1, sharex=ax)
        ax_histy = divider.append_axes('right', 1, pad=0.1, sharey=ax)

        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        # bin size
        binwidth = 0.03
        xymax = max(np.max(np.abs(v1)), np.max(np.abs(v2)))
        lim = (int(xymax / binwidth) + 1) * binwidth
        x_bins = np.arange(0, lim + binwidth, binwidth / 2)
        y_bins = np.arange(0, lim + binwidth, binwidth / 2)

        ax_histx.hist(v1[m.visual_mask], x_bins, color='r', alpha=0.5)
        ax_histx.hist(v1[m.place_mask], x_bins, color='g', alpha=0.5)
        ax_histy.hist(v2[m.visual_mask], y_bins, color='r', alpha=0.5, orientation='horizontal')
        ax_histy.hist(v2[m.place_mask], y_bins, color='g', alpha=0.5, orientation='horizontal')

        ax_histx.set_ylabel('neurons')
        ax_histy.set_xlabel('neurons')

        ax_histx.tick_params(width=1.2)
        ax_histy.tick_params(width=1.2)
        for i in ['top', 'bottom', 'left', 'right']:
            ax_histx.spines[i].set_linewidth(1.2)
            ax_histy.spines[i].set_linewidth(1.2)


if __name__ == '__main__':
    ClsCellTypeOptions().main()
