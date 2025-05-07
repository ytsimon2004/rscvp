import numpy as np
from matplotlib.axes import Axes
from scipy.signal import find_peaks
from tqdm import tqdm

from argclz import AbstractParser
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import DataOutput, StimpyOptions, Suite2pOptions, get_neuron_list
from rscvp.util.util_trials import TrialSelection


@publish_annotation('test')
class CalciumEventOptions(AbstractParser, Suite2pOptions, StimpyOptions):
    DESCRIPTION = 'Calculate the calcium transient numbers and median amplitude'

    std_fold = 3
    reuse_output = True

    def post_parsing(self):
        if self.signal_type != 'df_f':
            raise ValueError('')
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    def run(self):
        self.post_parsing()

        output = self.get_data_output('event', self.session)
        self.foreach_event_properties(output)

    def foreach_event_properties(self, output: DataOutput):
        rig = self.load_riglog_data()
        s2p = self.load_suite_2p()

        neuron_list = get_neuron_list(s2p, self.neuron_id)

        trial = TrialSelection.from_rig(rig, self.session)

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)
        image_mask = trial.masking_time(image_time)

        signal_all, baseline_all = get_neuron_signal(s2p, neuron_list, signal_type=self.signal_type, normalize=False)
        signal_all = signal_all[:, image_mask]
        baseline_all = baseline_all[:, image_mask]

        with csv_header(output.csv_output, ['neuron_id', 'transient_numbers', 'transient_median_amp']) as csv:
            for neuron_id in tqdm(neuron_list, desc='event eval', unit='neuron', ncols=80):
                signal = signal_all[neuron_id]
                baseline = baseline_all[neuron_id]
                std = np.std(signal)
                thres = baseline + self.std_fold * std

                peak_indices, _ = find_peaks(signal, height=thres, distance=s2p.fs / 2)
                sig_peak = signal[peak_indices]

                n_events = len(peak_indices)
                median_events = np.median(sig_peak)

                with plot_figure(output.figure_output(neuron_id)) as ax:
                    self.plot_event_peak(ax, peak_indices, signal, sig_peak)

                csv(neuron_id, n_events, median_events)

    def plot_event_peak(self, ax: Axes,
                        x: np.ndarray,
                        signal: np.ndarray,
                        peaks: np.ndarray):

        ax.plot(signal)
        ax.plot(x, peaks, 'x')
        ax.set(xlabel='sample points', ylabel=self.signal_type)


if __name__ == '__main__':
    CalciumEventOptions().main()
