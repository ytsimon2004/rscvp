import numpy as np
from scipy.ndimage import gaussian_filter1d

from argclz import AbstractParser, argument, as_argument
from neuralib.imaging.suite2p import normalize_signal, get_neuron_signal, sync_s2p_rigevent
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from rscvp.spatial.main_cache_sortidx import ApplySortIdxOptions
from rscvp.util.cli import Suite2pOptions
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.position import load_interpolated_position
from rscvp.util.typing import SIGNAL_TYPE
from rscvp.util.util_trials import TrialSelection

__all__ = ['CPBeltSortTrialOptions']


class CPBeltSortTrialOptions(AbstractParser, ApplySortIdxOptions):
    DESCRIPTION = 'plot the sorted calcium activities of population neurons in the given laps'

    sort_lap: int | None = argument(
        '--sl', '--sort-lap',
        metavar='INDEX',
        default=None,
        help='sorting index in certain lap, otherwise use trial averaged sorting',
    )

    signal_type: SIGNAL_TYPE = as_argument(Suite2pOptions.signal_type).with_options(default='df_f')
    pre_selection = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('spv')
        self.calactivity_belt_trial(output_info)

    def calactivity_belt_trial(self, output: DataOutput):
        """plot the sorting calcium activity per lap, and corresponding position and running speed"""

        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)
        lap_time = rig.lap_event.time

        if isinstance(self.use_trial, str):
            indices = TrialSelection(rig, self.use_trial).get_time_profile().trial_range
            trial_range = slice(*indices)
            t1 = indices[0]
            t2 = indices[1]
        elif isinstance(self.use_trial, tuple):
            trial_range = slice(*self.use_trial)
            t1, t2 = self.use_trial
        else:
            raise ValueError('')

        p_result = load_interpolated_position(rig)
        pt = p_result.t
        p = p_result.p
        v = p_result.v

        # signal
        neuron_list = np.arange(s2p.n_neurons)
        if self.signal_type in ('df_f', 'spks'):
            signal = get_neuron_signal(s2p, neuron_list)[0]  # (N, it:image_time)
        elif self.signal_type == 'cascade_spks':
            from rscvp.util.util_cascade import get_neuron_cascade_spks
            signal = get_neuron_cascade_spks(s2p, neuron_list)
        else:
            raise ValueError(f'unknown signal type: {self.signal_type}')

        # neuron_selection
        cell_mask = self.get_selected_neurons()
        signal = signal[cell_mask]  # (N', it)

        # collect the result from different laps
        lap_time_ls = lap_time[trial_range]
        left_t = t0 = lap_time[t1]
        right_t = lap_time[t2]  # start time from next lap

        it_trial = np.logical_and(left_t < image_time, image_time < right_t)
        pt_trial = np.logical_and(left_t < pt, pt < right_t)

        s = signal[:, it_trial]  # (N', it')
        pt = pt[pt_trial]  # (pt,)
        p = p[pt_trial]
        v = v[pt_trial]
        v[v < -20] = 0  # recalculate laps
        v[v > 100] = 0

        # normalize sig within selected laps interval
        s = normalize_signal(s)

        # sorting of calcium transient across time
        if self.sort_lap is not None:
            ss = gaussian_filter1d(signal, 3, mode='wrap')  # used for sorting
            _it_trial = np.logical_and(lap_time[self.sort_lap] < image_time, image_time < lap_time[self.sort_lap + 1])
            m_argmax = np.argmax(ss[:, _it_trial], axis=1)
            sorted_idx = np.argsort(m_argmax)
        else:
            # i.e., used the trial average (across laps) neuronal index. refer to rscvp.spatial.main_belt_sort.py
            sorted_idx = self.apply_sort_idx_cache().sort_idx

        s = s[sorted_idx]

        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            self.pc_selection if self.pc_selection is not None else None,
            self.signal_type,
            self.use_trial,
        )

        with plot_figure(output_file, 3, 1, figsize=(16, 6), tight_layout=False) as _ax:
            #
            ax = _ax[0]
            im = ax.imshow(
                s,  # srate -> t
                aspect='auto',
                cmap='Greys',
                interpolation='none',
                origin='lower',
                extent=[0 + t0, (s.shape[1] / s2p.fs) + t0, 0, s.shape[0]]
            )

            insert_colorbar(ax, im)
            ax.set(ylabel='neuron no.')
            ax.axes.xaxis.set_visible(False)

            #
            ax = _ax[1]
            ax.plot(pt, p, color='k')
            ax.set_xlim(np.min(pt), np.max(pt))
            ax.set(ylabel='cm')
            ax.spines['bottom'].set_visible(False)
            ax.axes.xaxis.set_visible(False)

            #
            ax = _ax[2]
            ax.plot(pt, v, color='k')
            ax.set_xlim(np.min(pt), np.max(pt))
            ax.set_xlabel('time(s)')
            ax.set_ylabel('cm/s')

            #
            for i in range(len(_ax)):
                _ax[i].vlines(lap_time_ls, ymin=_ax[i].get_ylim()[0], ymax=_ax[i].get_ylim()[1],
                              color='r', linestyle='--', alpha=0.5, linewidth=1, zorder=1)


if __name__ == '__main__':
    CPBeltSortTrialOptions().main()
