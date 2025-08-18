from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.imaging.suite2p import dff_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.typing import AxesArray
from neuralib.typing.func import flatten_arraylike
from neuralib.util.verbose import publish_annotation
from rscvp.selection.utils import image_time_per_trial
from rscvp.util.cli import DataOutput, PlotOptions, FIG_MODE, StimpyOptions, Suite2pOptions, get_neuron_list, NeuronID, \
    TreadmillOptions

__all__ = ['TrialReliabilityOptions']


@publish_annotation('main', project='rscvp', caption='lap (trial) reliability', as_doc=True)
class TrialReliabilityOptions(AbstractParser, Suite2pOptions, StimpyOptions, TreadmillOptions, PlotOptions):
    DESCRIPTION = 'See fraction of active trials in the linear treadmill task'

    filter: bool = argument(
        '--filter',
        action='store_true',
        help='whether do the gaussian filter of the signal'
    )

    std_type: Literal['dff', 'neuropil'] = argument(
        '--std-type',
        default='dff',
        help='use which types of signal for the standard deviation calculation'
    )

    std_fold: float = argument(
        '--stdf',
        default=3,
        help='standard deviation fold',
    )

    def post_parsing(self):
        if self.vr_environment:
            self.session = 'all'

    def run(self):
        self.post_parsing()
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('tr', self.session)
        self.foreach_trial_reliability(output_info, self.neuron_id)

    def foreach_trial_reliability(self, output: DataOutput,
                                  neuron_ids: NeuronID):
        """
        Neurons were selected if exhibit one or more calcium transients (>baseline df/f + 3 * S.D)
        in at least 30% of the laps. basically consider the trial to trial reliability

        .. seealso:: Kandler et al., 2018, bioRxiv.
        """

        rig = self.load_riglog_data()
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        # specify session
        session = self.get_session(rig, self.session)
        lap_time = self.masking_lap_time(rig)
        act_mask = session.time_mask_of(image_time)

        with csv_header(output.csv_output, ['neuron_id', f'trial_reliability_{self.session}']) as csv:
            for neuron_id in tqdm(neuron_list, desc='trial reliability', unit='neuron', ncols=80):
                dff = dff_signal(s2p.f_raw[neuron_id], s2p.f_neu[neuron_id], s2p)
                signal = gaussian_filter1d(dff.dff, 3) if self.filter else dff.dff
                baseline = dff.dff_baseline

                match self.std_type:
                    case 'dff':
                        std = np.std(signal)
                    case 'neuropil':
                        std = np.std(dff.baseline_fluctuation)
                    case _:
                        raise ValueError(f'unknown type: {self.std_type}')

                thres = baseline + self.std_fold * std  # type: np.ndarray

                bas_trial = image_time_per_trial(image_time, lap_time, baseline, act_mask)
                sig_trial = image_time_per_trial(image_time, lap_time, signal, act_mask)
                thres_trial = image_time_per_trial(image_time, lap_time, thres, act_mask)
                active_trial = [np.any(sig_trial[i] > thres_trial[i]) for i in range(len(sig_trial))]

                with plot_figure(output.figure_output(neuron_id), 1, 2) as ax:
                    plot_trial_reliability_trace(ax, bas_trial, thres_trial, sig_trial, mode=self.mode)

                csv(neuron_id, np.mean(active_trial))


def plot_trial_reliability_trace(axes: AxesArray,
                                 baseline: list[np.ndarray],
                                 threshold: list[np.ndarray],
                                 sig_trial: list[np.ndarray],
                                 first: int = 5,
                                 last: int = 5,
                                 mode: FIG_MODE = 'simplified'):
    """
    Plot the first/last 5 laps signal within the certain behavioral session"
    
    :param axes: ``Axes``
    :param baseline: List of baseline signals foreach trial
    :param threshold: List of threshold signals foreach trial
    :param sig_trial: List of transient signals foreach trial
    :param first: First number of trial to plot
    :param last: Last number of trial to plot
    :param mode: {'simplified', 'presentation'}
    """

    ax = axes[0]
    ax.plot(flatten_arraylike(baseline[:first]), color='k', alpha=0.5, label='baseline')
    ax.plot(flatten_arraylike(threshold[:first]), color='r', alpha=0.5, label='threshold')
    ax.plot(flatten_arraylike(sig_trial[:first]), color='b', alpha=0.5, label='signal')

    ax = axes[1]
    ax.plot(flatten_arraylike(baseline[-last:]), color='k', alpha=0.5)
    ax.plot(flatten_arraylike(threshold[-last:]), color='r', alpha=0.5)
    ax.plot(flatten_arraylike(sig_trial[-last:]), color='b', alpha=0.5, label='signal')

    if mode != 'simplified':
        lap_boundary = 0
        for lap, sig in enumerate(baseline[:5]):
            lap_boundary += len(sig)
            axes[0].axvline(lap_boundary, color='g', ls='--', zorder=1, alpha=0.5)
            axes[1].axvline(lap_boundary, color='g', ls='--', zorder=1, alpha=0.5)

        axes[0].set(xlabel='frames', ylabel='dF/F')

    axes[0].legend()


if __name__ == '__main__':
    TrialReliabilityOptions().main()
