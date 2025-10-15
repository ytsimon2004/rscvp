import numpy as np
from matplotlib.axes import Axes
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure, grid_subplots
from neuralib.typing import AxesArray
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_occ import ApplyPosBinCache
from rscvp.util.cli import SelectionOptions
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from rscvp.util.util_trials import TrialSelection
from .util_plot import plot_tuning_heatmap

__all__ = ['PositionMapOptions']


@publish_annotation('main', project='rscvp', figure='fig.2A & fig.S5B-C', as_doc=True)
class PositionMapOptions(AbstractParser, SelectionOptions, ApplyPosBinCache):
    DESCRIPTION = 'Plot normalized position binned calcium activity across trials'

    overview: bool = argument('--overview', help='plot batch overview')
    pc_only: bool = argument('--pc-only', help='overview only plot place cell')
    si_filter: float | None = argument('--si-filter', default=None, help='overview spatial information filter')

    binned_smooth = True
    reuse_output = True

    def post_parsing(self):
        if self.overview:
            self.pre_selection = True
            self.pc_selection = 'slb'

            if self.is_vop_protocol or self.is_ldl_protocol:
                self.used_session = 'light'
            elif self.is_virtual_protocol:
                self.used_session = 'close'

    def run(self):
        self.post_parsing()
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('ba', self.signal_type,
                                           running_epoch=self.running_epoch,
                                           use_virtual_space=self.use_virtual_space)

        if self.overview:
            self.plot_overview(output_info)
        else:
            self.foreach_position_map(output_info, self.neuron_id)

    # ============= #
    # Overview Plot #
    # ============= #

    def plot_overview(self, output: DataOutput, batch_size: int = 25):
        signal, orig_indices = self._extract_signals()
        n_neurons = signal.shape[0]

        n_batches = (n_neurons + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_neurons)

            batch_signal = signal[start:end]
            batch_ids = orig_indices[start:end]
            titles = [f'ID_{idx}' for idx in batch_ids]

            grid_subplots(
                batch_signal,
                images_per_row=5,
                plot_func='imshow',
                dtype='img',
                cmap='viridis',
                title=titles,
                figsize=(8, 8),
                output=output.summary_figure_output(),
                aspect='auto'
            )

    def _extract_signals(self) -> tuple[np.ndarray, np.ndarray]:
        """Get activity with neurons and trials selection

        :return: activity (N', L', B) and neuron indices (N',)
        """
        rig = self.load_riglog_data()
        signal = self.get_occ_cache().occ_activity
        orig_indices = np.ones(signal.shape[0], dtype=bool)

        # neuron selection
        if self.pc_only:
            mx = self.get_selected_neurons()
            orig_indices &= mx

        if self.si_filter is not None:
            si = self.get_csv_data(f'si_{self.session}')
            mx = si > self.si_filter
            orig_indices &= mx

        signal = signal[orig_indices]

        # trial selection
        indices = (
            TrialSelection
            .from_rig(rig, self.session, use_virtual_space=self.use_virtual_space)
            .get_selected_profile()
            .trial_range
        )
        signal = signal[:, slice(*indices), :]
        signal = signal / np.max(signal, axis=(1, 2), keepdims=True)

        if self.binned_smooth:
            signal = gaussian_filter1d(signal, 3, mode='wrap', axis=2)

        return signal, np.where(orig_indices)[0]

    # ============ #
    # Foreach Plot #
    # ============ #

    def foreach_position_map(self, output: DataOutput, neuron_ids: NeuronID):
        """
        plot normalized dff in the belt env

        :param output:
        :param neuron_ids:
        :return:
        """
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)
        signal_all = self.get_occ_cache().occ_activity
        if self.binned_smooth:
            signal_all = gaussian_filter1d(signal_all, 3, mode='wrap', axis=2)

        #
        protocol = self.get_protocol_alias
        n_sessions = len(self.session_list)
        rig = self.load_riglog_data()

        session = self.get_session_info(rig, ignore_all=True)

        if self.use_virtual_space:
            lap_event = rig.get_pygame_stimlog().virtual_lap_event
        else:
            lap_event = rig.lap_event

        # pre-compute session slices once
        session_slices = None
        if protocol in ['visual_open_loop', 'light_dark_light', 'vr']:
            session_slices = {
                k: v.in_slice(lap_event.time, lap_event.value_index)
                for _, (k, v) in zip(range(n_sessions), session.items())
            }

        for neuron_id in tqdm(neuron_list, desc='plot_calactivity_belt', unit='neurons', ncols=80):
            signal = signal_all[neuron_id]

            match protocol:
                case 'visual_open_loop' | 'light_dark_light' | 'vr':
                    with plot_figure(output.figure_output(neuron_id, self.signal_type),
                                     2, n_sessions + 1,
                                     tight_layout=False,
                                     figsize=(16, 6),
                                     gridspec_kw={'width_ratios': [1.618] + [1] * n_sessions}) as ax:
                        self.plot_multiple_session(ax, session_slices, signal)
                case 'grey':
                    with plot_figure(output.figure_output(neuron_id)) as ax:
                        self.plot_single_session(ax, signal)
                case _:
                    raise ValueError(f'plot is not implemented in {protocol}')

    def plot_multiple_session(self, axes: AxesArray,
                              session_slices: dict,
                              signal: np.ndarray):
        """
        Plot binned_activity across multiple session

        :param axes: ``Axes``
        :param session_slices: pre-computed session slices dict
        :param signal: binned cal-activity with shape (L, B) where L = diff(lap) - 1, B = spatial bins
        """
        n_sessions = len(self.session_list)
        session_sig = {
            k: signal[v]
            for k, v in session_slices.items()
        }

        names = list(session_sig.keys())
        sig = list(session_sig.values())
        label_color = ['k', 'r', 'gray']

        # all
        ax = axes[0, 0]
        plot_tuning_heatmap(
            signal,
            track_length=self.track_length,
            colorbar=True,
            session_line=[len(sig[i]) for i in range(n_sessions - 1)],
            ax=ax
        )
        ax.set_ylabel('trial #')

        ax = axes[1, 0]
        for i in range(n_sessions):
            plot_trial_avg_curve(
                ax,
                sig[i],
                bins=self.pos_bins,
                track_length=self.track_length,
                label=names[i],
                color=label_color[i]
            )
        ax.legend()

        # normalization per session plot / exclude zero run in particular session
        norm = [s / np.nanmax(s) if s.size != 0 else s for s in sig]

        for i in range(n_sessions):
            plot_tuning_heatmap(norm[i],
                                track_length=self.track_length,
                                colorbar=True if i == n_sessions - 1 else False,
                                ax=axes[0, i + 1])
            plot_trial_avg_curve(axes[1, i + 1],
                                 norm[i],
                                 bins=self.pos_bins,
                                 track_length=self.track_length,
                                 label=names[i], color=label_color[i])

    def plot_single_session(self, ax, signal: np.ndarray):
        """plot binned_activity in single session, used in 'grey' protocol"""
        plot_tuning_heatmap(signal, track_length=self.track_length, ax=ax)
        ax.set(xlabel='Position (cm)', ylabel='Neuron #')


def plot_trial_avg_curve(ax: Axes,
                         signal: np.ndarray, *,
                         bins: int = 100,
                         track_length: int = 150,
                         **kwargs):
    mean = np.nanmean(signal, axis=0)
    sem = stats.sem(signal, axis=0, nan_policy='omit')
    x = np.arange(bins) * (track_length / bins)
    ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean + sem, mean - sem, alpha=0.3, **kwargs)
    ax.set(xlim=(0, track_length), xlabel='Position (cm)', ylabel='Norm. dF/F')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


if __name__ == '__main__':
    PositionMapOptions().main()
