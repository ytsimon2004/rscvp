import numpy as np
from matplotlib.axes import Axes
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure, grid_subplots
from neuralib.typing import AxesArray
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli import SelectionOptions
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from rscvp.util.util_trials import TrialSelection
from stimpyp import RiglogData, Session
from .util_plot import plot_tuning_heatmap

__all__ = ['PositionMapOptions']


@publish_annotation('main', project='rscvp', figure='fig.2A & fig.S5B-C', as_doc=True)
class PositionMapOptions(AbstractParser, SelectionOptions, ApplyPosBinActOptions):
    DESCRIPTION = 'Plot normalized position binned calcium activity across trials'

    overview: bool = argument('--overview', help='plot batch overview')
    pc_only: bool = argument('--pc-only', help='overview only plot place cell')
    si_filter: float | None = argument('--si-filter', default=None, help='overview spatial information filter')

    binned_smooth = True
    reuse_output = True

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.overview:
            self.pre_selection = True
            self.pc_selection = 'slb'
            self.session = 'light'
            self.used_session = 'light'

    def run(self):
        self.post_parsing()
        output_info = self.get_data_output('ba', self.signal_type, running_epoch=self.running_epoch)

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
        signal = self.apply_binned_act_cache().occ_activity
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
        indices = TrialSelection.from_rig(rig, self.session).get_time_profile().trial_range
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

        rig = self.load_riglog_data()
        protocol = self.get_protocol_alias()

        signal_all = self.apply_binned_act_cache().occ_activity

        if self.binned_smooth:
            signal_all = gaussian_filter1d(signal_all, 3, mode='wrap', axis=2)

        for neuron_id in tqdm(neuron_list, desc='plot_calactivity_belt', unit='neurons', ncols=80):
            signal = signal_all[neuron_id]

            match protocol:
                case 'visual_open_loop' | 'light_dark_light':
                    with plot_figure(output.figure_output(neuron_id, self.signal_type), 2, 4,
                                     tight_layout=False,
                                     figsize=(16, 6),
                                     gridspec_kw={'width_ratios': [1.618, 1, 1, 1]}) as ax:
                        self.plot_three_sessions(ax, rig, signal)
                case 'grey':
                    with plot_figure(output.figure_output(neuron_id)) as ax:
                        self.plot_single_session(ax, signal)
                case _:
                    raise ValueError(f'plot is not implemented in {protocol}')

    def plot_three_sessions(self, axes: AxesArray,
                            rig: RiglogData,
                            signal: np.ndarray):
        """
        Plot binned_activity across three session, i.e., "LDL" and 'VOL' protocol types

        :param axes: ``Axes``
        :param rig: ``RiglogData``
        :param signal: binned cal-activity with shape (L, B) where L = diff(lap) - 1, B = spatial bins
        """

        session_sig = get_session_sig(rig, signal)
        names = list(session_sig.keys())
        sig_sep = list(session_sig.values())
        label_color = ['k', 'r', 'gray']

        #
        ax = axes[0, 0]
        plot_tuning_heatmap(
            signal,
            belt_length=self.belt_length,
            colorbar=True,
            session_line=(len(sig_sep[0]), len(sig_sep[1])),
            ax=ax
        )
        ax.set_ylabel('trial #')

        #
        ax = axes[1, 0]
        for i in range(3):
            plot_trial_avg_curve(
                ax,
                sig_sep[i],
                bins=self.pos_bins,
                belt_length=self.belt_length,
                label=names[i],
                color=label_color[i]
            )
        ax.legend()

        # normalization per session plot / exclude zero run in particular session
        norm = [s / np.max(s) if s.size != 0 else s for s in sig_sep]

        #
        plot_tuning_heatmap(norm[0], belt_length=self.belt_length, ax=axes[0, 1])
        plot_tuning_heatmap(norm[1], belt_length=self.belt_length, ax=axes[0, 2])
        plot_tuning_heatmap(norm[2], belt_length=self.belt_length, colorbar=True, ax=axes[0, 3])

        #
        for i in range(3):
            ax = axes[1, i + 1]
            plot_trial_avg_curve(ax, norm[i], bins=self.pos_bins, belt_length=self.belt_length,
                                 label=names[i], color=label_color[i])

    def plot_single_session(self, ax, signal: np.ndarray):
        """plot binned_activity in single session, used in 'grey' protocol"""
        plot_tuning_heatmap(signal, belt_length=self.belt_length, ax=ax)
        ax.set(xlabel='Position (cm)', ylabel='Neuron #')


def plot_trial_avg_curve(ax: Axes,
                         signal: np.ndarray, *,
                         bins: int = 100,
                         belt_length: int = 150,
                         **kwargs):
    mean = np.nanmean(signal, axis=0)
    sem = stats.sem(signal, axis=0, nan_policy='omit')
    x = np.arange(bins) * (belt_length / bins)
    ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean + sem, mean - sem, alpha=0.3, **kwargs)
    ax.set(xlim=(0, belt_length), xlabel='Position (cm)', ylabel='Norm. dF/F')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


def get_session_sig(rig: RiglogData, signal: np.ndarray) -> dict[Session, np.ndarray]:
    """
    get session name and activity in different behavioral sessions

    :param rig:
    :param signal: (L, B)
    :return:
        session_sig
    """
    session = rig.get_stimlog().session_trials()
    lap_event = rig.lap_event

    session_sig = {
        k: signal[v.in_slice(lap_event.time, lap_event.value_index)]
        for _, (k, v) in zip(range(3), session.items())
    }

    return session_sig


if __name__ == '__main__':
    PositionMapOptions().main()
