from typing import get_args

import numpy as np

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.util_trials import TRIAL_CV_TYPE
from stimpyp import Session, SessionInfo

__all__ = ['PopulationMTXOptions']


class PopulationMTXOptions(AbstractParser, ApplyPosBinActOptions, SelectionOptions):
    DESCRIPTION = 'Plot the population (selected neurons) correlation matrix in different behavioral session (trial-averaged)'

    x_cond: TRIAL_CV_TYPE = argument(
        '-x',
        default='light',
        help='specify the condition in x-axis of correlation matrix',
    )

    y_cond: TRIAL_CV_TYPE = argument(
        '-y',
        default='dark',
        help='specify the condition in y-axis of correlation matrix',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('cm')
        self.population_corr_mtx(output_info)

    def population_corr_mtx(self, output: DataOutput, diverging_color: bool = False):
        """
        :param output: ``DataOutput``
        :param diverging_color: whether using diverging color map
        """
        from scipy.stats import pearsonr
        rig = self.load_riglog_data()

        signal_all = self.apply_binned_act_cache().occ_activity

        # neuron selection
        cell_mask = self.get_selected_neurons()
        signal_all = signal_all[cell_mask]  # (N', L, B)

        # trial selection
        lap_time = rig.lap_event.time
        session_info = rig.get_stimlog().session_trials()
        is_ldl = self.is_ldl_protocol

        mx = get_trial_mask(session_info, self.x_cond, lap_time, is_ldl)
        my = get_trial_mask(session_info, self.y_cond, lap_time, is_ldl)
        signal_x = signal_all[:, mx, :]  # (N', L', B)
        signal_y = signal_all[:, my, :]  # (N', L', B)

        mean_x = np.nanmean(signal_x, axis=1)  # (N', B) trial average
        mean_y = np.nanmean(signal_y, axis=1)  # (N', B)

        matrix = np.zeros((self.pos_bins, self.pos_bins))
        for x in range(self.pos_bins):
            for y in range(self.pos_bins):
                a = mean_x[:, x]  # population neural act in position x
                b = mean_y[:, y]
                corr_coef = pearsonr(a, b)[0]
                matrix[y, x] = corr_coef

        #
        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            self.pc_selection if self.pc_selection is not None else None,
            self.signal_type,
            self.x_cond,
            self.y_cond
        )

        if diverging_color:
            import matplotlib.colors as mcolors
            cmap = 'coolwarm'
            norm = mcolors.TwoSlopeNorm(vmin=np.min(matrix), vmax=np.max(matrix), vcenter=0)
        else:
            cmap = 'inferno'
            norm = None

        with plot_figure(output_file, set_square=True) as ax:
            im = ax.imshow(
                matrix,
                extent=[0, self.belt_length, 0, self.belt_length],
                cmap=cmap,
                interpolation='none',
                aspect='auto',
                origin='lower',
                norm=norm
            )

            if self.cue_loc is not None:
                for i in self.cue_loc:
                    ax.axvline(i, lw=0.5, color='w', alpha=0.5)
                    ax.axhline(i, lw=0.5, color='w', alpha=0.5)

            ax.set(xlabel=f'Position(cm) - {self.x_cond}', ylabel=f'Position(cm) - {self.y_cond}')
            ax.set_title(f'signal: {self.signal_type} \n'
                         f'num of neurons used: {self.n_selected_neurons} / {self.apply_binned_act_cache().n_neurons}')

            cbar = ax.figure.colorbar(im)
            cbar.ax.set_ylabel('Corr. Coef.')


def get_trial_mask(session_info: dict[Session, SessionInfo],
                   cond: TRIAL_CV_TYPE,
                   lap_time: np.ndarray,
                   is_ldl: bool) -> np.ndarray:
    """Get bool array mask for selected trials in condition"""
    # order sensitive
    if cond.startswith('light-bas') and is_ldl:
        session = session_info['light_bas']
    elif cond.startswith('light-end') and is_ldl:
        session = session_info['light_end']
    elif cond.startswith('light') and is_ldl:
        session = session_info['light_bas']
    #
    elif cond.startswith('light') and not is_ldl:
        session = session_info['light']
    elif cond.startswith('dark'):  # used in both vol and ldl prot
        session = session_info['dark']
    elif cond.startswith('visual'):
        session = session_info['visual']
    elif cond.startswith('all'):
        session = session_info['all']
    else:
        raise ValueError(f'condition: {cond} is not supported. check {get_args(TRIAL_CV_TYPE)}')

    x = session.time_mask_of(lap_time)

    if cond.endswith('-odd'):
        x[1::2] = False
    elif cond.endswith('-even'):
        x[0::2] = False

    return x


if __name__ == '__main__':
    PopulationMTXOptions().main()
