import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Iterable

from argclz import int_tuple_type, AbstractParser, argument
from neuralib.plot import plot_figure
from neuralib.suite2p import get_neuron_signal, sync_s2p_rigevent, Suite2PResult
from neuralib.typing import AxesArray
from rscvp.util.cli import DataOutput, Suite2pOptions, TreadmillOptions, get_neuron_list
from rscvp.util.util_trials import TrialSignal

__all__ = ['TrialActProfile']


class TrialActProfile(AbstractParser, Suite2pOptions, TreadmillOptions):
    DESCRIPTION = 'Plot the activities (dff/spks) versus position/velocity/visual stim profiles in range of laps'

    filter: bool = argument(
        '-f', '--filter',
        help='if do the smoothing of dff signal',
    )

    trial_numbers: tuple[int, int] = argument(
        '--trange', '-t',
        type=int_tuple_type,
        default=(0, 5),
        help='trial range per session, i.e., (0,5) indicates the first 5 laps plot for different sessions',
    )

    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('at')
        self.foreach_session_signal(output_info)

    def foreach_session_signal(self, output: DataOutput):
        s2p = self.load_suite_2p()
        iter_session = list(self.foreach_signals(s2p))

        try:
            si = self._get_spatial_info(iter_session)
        except (FileNotFoundError, ValueError):  # debug mode
            si = None

        #
        for n in tqdm(range(s2p.n_neurons)):
            nrows = 5 if self.is_vop_protocol else 4
            ncols = len(self.session_list)
            with plot_figure(output.figure_output(n), nrows, ncols, figsize=(10, 4), sharex='col') as _ax:

                for i, ts in enumerate(iter_session):
                    ax = _ax[:, i]
                    self._plot_session_signals(ax, ts, n)

                    if si is not None:
                        ax[0].set_title(f'si:{si[i, n]}')

    def _get_spatial_info(self, trial_sig: list[TrialSignal]) -> np.ndarray:
        """(S, N)"""
        ret = []
        for t in trial_sig:
            s = t.session
            f = pd.read_csv(self.get_data_output('si', s, latest=True).csv_output)
            ret.append(f[f'si_{s}'].to_numpy())

        return np.vstack(ret)

    def _plot_session_signals(self, axes: AxesArray,
                              trial_sig: TrialSignal,
                              neuron_id: int,
                              show_axis=True):

        dff = trial_sig.dff[neuron_id]
        spks = trial_sig.spks[neuron_id]

        axes[0].plot(trial_sig.time, dff, c='g', lw=0.5)

        axes[1].plot(trial_sig.time, spks, c='r', lw=0.5)

        axes[2].plot(trial_sig.time, trial_sig.position, c='k', lw=0.8)
        axes[3].plot(trial_sig.time, trial_sig.velocity, c='k', lw=0.8)

        if self.is_vop_protocol:
            axes[4].plot(trial_sig.vstim_time, trial_sig.vstim_pulse, c='r', lw=0.8)

        for i in range(axes.shape[0]):
            if not show_axis:
                axes[i].axes.xaxis.set_visible(False)
                axes[i].axes.yaxis.set_visible(False)
                axes[i].spines['bottom'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
            if i == (axes.shape[0] - 1):
                axes[i].set(xlabel='time(s)')

    def foreach_signals(self, s2p: Suite2PResult) -> Iterable[TrialSignal]:
        rig = self.load_riglog_data()
        neuron_ids = self.neuron_id

        neuron_list = get_neuron_list(s2p, neuron_ids)

        dff, _ = get_neuron_signal(s2p, neuron_list, signal_type='df_f', dff=True, normalize=False)
        spks, _ = get_neuron_signal(s2p, neuron_list, signal_type='spks', normalize=False)
        stim = rig.get_stimlog().stim_square_pulse_event() if self.is_vop_protocol else None

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        pos = self.load_position().interp_time(image_time)

        lap_event = self.get_lap_event(rig)

        session_info = rig.get_stimlog().session_trials()
        session_info.pop('all', None)
        sessions = list(session_info.keys())

        for s in sessions:
            x = session_info[s].in_slice(lap_event.time, lap_event.value.astype(int))
            trial0 = x.start + self.trial_numbers[0]
            trial1 = x.start + self.trial_numbers[1]
            t0 = lap_event.time[trial0]
            t1 = lap_event.time[trial1]

            # neural activity
            mx = np.logical_and(t0 < image_time, image_time < t1)
            time = image_time[mx]
            _dff = dff[:, mx]
            _spks = spks[:, mx]

            position = pos.p[mx]
            velocity = pos.v[mx]

            # visual
            if self.is_vop_protocol:
                vt_mask = np.logical_and(t0 < stim.time, stim.time < t1)
                vtime = stim.time[vt_mask]
                vpulse = stim.value[vt_mask]
            else:
                vpulse = None
                vtime = None

            yield TrialSignal(
                s, time, _dff, _spks,
                position=position,
                velocity=velocity,
                vstim_pulse=vpulse,
                vstim_time=vtime
            )


if __name__ == '__main__':
    TrialActProfile().main()
