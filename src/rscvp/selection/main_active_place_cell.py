import numpy as np
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_suite2p import NeuronID, get_neuron_list
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.io import csv_header
from neuralib.plot import plot_figure

__all__ = ['ActivePCOptions']


class ActivePCOptions(AbstractParser, ApplyPosBinActOptions):
    DESCRIPTION = 'see spatial active cell by spks and dff cmp'

    threshold: float = argument(
        '-t', '--threshold',
        default=0.03,
        help='multiplicand of dff',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('apc', self.session)
        self.foreach_active_place_neuron(output_info, self.neuron_id)

    # this function is not commonly used, check again.
    # this method might be sensitive to spks calculation
    def foreach_active_place_neuron(self, output: DataOutput, neuron_ids: NeuronID):
        """
        Followed by Kandler et al., 2018, bioRxiv
        occupancy-normalized activity (spks) in any position bin exceed .03% df_f
        """

        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        self.signal_type = 'spks'
        sig_spk = self.apply_binned_act_cache().occ_activity
        self.signal_type = 'df_f'
        sig_dff = self.apply_binned_act_cache().occ_activity

        session = rig.get_stimlog().session_trials()[self.session]
        lap_mask = session.time_mask_of(rig.lap_event.time[-1])

        with csv_header(output.csv_output, ['neuron_id', 'spk_exceed']) as csv:
            for neuron_id in tqdm(neuron_list, desc='active_pc', unit='neurons', ncols=80):
                _sig_spks = sig_spk[neuron_id, lap_mask, :]  # (L', B)
                _sig_dff = sig_dff[neuron_id, lap_mask, :]

                # trial average
                _sig_spks = np.nanmean(_sig_spks, axis=0)  # (B,)
                _sig_dff = np.nanmean(_sig_dff, axis=0)

                dff_thres = _sig_dff * self.threshold
                spk_exceed = np.mean(_sig_spks > dff_thres)  # %

                with plot_figure(output.figure_output(neuron_id)) as ax:
                    plot_activate_pc(ax, _sig_spks, dff_thres)

                csv(neuron_id, spk_exceed)


def plot_activate_pc(ax, spks: np.ndarray, dff_threshold: np.ndarray):
    ax.plot(spks, color='b', label='spks')
    ax.plot(dff_threshold, color='k', alpha=0.3, label='dff_thres')
    ax.legend()


if __name__ == '__main__':
    ActivePCOptions().main()
