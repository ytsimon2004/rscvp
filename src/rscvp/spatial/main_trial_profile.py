import numpy as np
import pandas as pd
from tqdm import tqdm

from argclz import int_tuple_type, AbstractParser, argument
from neuralib.plot import plot_figure
from neuralib.typing import AxesArray
from rscvp.util.cli import DataOutput, StimpyOptions, Suite2pOptions
from rscvp.util.util_trials import foreach_session_signals, TrialSignal

__all__ = ['TrialActProfile']


class TrialActProfile(AbstractParser, Suite2pOptions, StimpyOptions):
    DESCRIPTION = 'Plot the activities (dff/spks) versus position/velocity/visual stim profiles in range of laps'

    filter: bool = argument(
        '-f', '--filter',
        help='if do the smoothing of dff signal',
    )

    trial_numbers: tuple[int, int] = argument(
        '--trange', '-t',
        type=int_tuple_type,
        required=True,
        help='trial range per session, i.e., (0,5) indicates the first 5 laps plot for different sessions',
    )

    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('at')
        self.foreach_session_signal(output_info)

    def foreach_session_signal(self, output: DataOutput):
        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()

        iter_session = list(foreach_session_signals(
            s2p,
            rig,
            self.neuron_id,
            self.plane_index,
            normalize=False,
            do_smooth=False,
            trial_numbers=self.trial_numbers
        ))

        try:
            si = self._get_spatial_info(iter_session)
        except FileNotFoundError:
            si = None

        #
        for n in tqdm(range(s2p.n_neurons)):
            with plot_figure(output.figure_output(n),
                             5, 3,
                             figsize=(10, 4),
                             gridspec_kw={'height_ratios': [1.5, 1, 1.5, 1.5, 1.5]}) as _ax:

                for i, ts in enumerate(iter_session):
                    ax = _ax[:, i]
                    self._plot_session_signals(ax, ts, n)

                    if si is not None:
                        ax[0].set_title(f'si:{si[i, n]}')

    def _get_spatial_info(self, trial_sig: list[TrialSignal]) -> np.ndarray:
        """(S, N)"""
        ret = []
        for t in trial_sig:
            s = t.time_profile.session
            f = pd.read_csv(self.get_data_output('si', s, latest=True).csv_output)
            ret.append(f[f'si_{s}'].to_numpy())

        return np.vstack(ret)

    @staticmethod
    def _plot_session_signals(axes: AxesArray,
                              trial_sig: TrialSignal,
                              neuron_id: int,
                              show_axis=True):

        dff = trial_sig.dff[neuron_id]
        spks = trial_sig.spks[neuron_id]

        axes[0].plot(trial_sig.image_time, dff, c='g', lw=0.5)

        axes[1].plot(trial_sig.image_time, spks, c='r', lw=0.5)

        axes[2].plot(trial_sig.position_time, trial_sig.position, c='k', lw=0.8)
        axes[3].plot(trial_sig.position_time, trial_sig.velocity, c='k', lw=0.8)
        axes[4].plot(trial_sig.vstim_time, trial_sig.vstim_pulse, c='r', lw=0.8)

        for i in range(axes.shape[0]):
            if not show_axis:
                axes[i].axes.xaxis.set_visible(False)
                axes[i].axes.yaxis.set_visible(False)
                axes[i].spines['bottom'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
            if i == (axes.shape[0] - 1):
                axes[i].set(xlabel='time(s)')


if __name__ == '__main__':
    TrialActProfile().main()
