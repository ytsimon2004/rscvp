import collections
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from scipy.signal import butter, lfilter
from scipy.stats import pearsonr

from argclz import AbstractParser
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.plot.tools import AnchoredScaleBar
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_plot import PlotOptions
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID
from stimpyp import GratingPattern

__all__ = [
    'VisualReliabilityOptions',
    'plot_cat_trial'
]


@publish_annotation('main', project='rscvp', figure='fig.4A & fig.S4 & fig.S5B-C', as_doc=True)
class VisualReliabilityOptions(AbstractParser, SelectionOptions, PlotOptions):
    DESCRIPTION = """
    Concatenate the calcium traces per visual-stimuli epoch, and compute the pairwise cross-correlation (reliability) 
    to determine if is a visually-responsive cell
    """

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('vc')
        self.foreach_visual_reliability(output_info, self.neuron_id)

    def foreach_visual_reliability(self, output: DataOutput, neuron_ids: NeuronID):
        """Calculate the reliability and plot foreach neuron"""
        from tqdm import tqdm

        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        riglog = self.load_riglog_data()
        image_time = riglog.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        pattern = GratingPattern.of(riglog)
        headers = ['neuron_id', 'visual_cell', 'reliability', 'max_vis_resp', 'perc95_vis_resp']
        with csv_header(output.csv_output, headers) as csv:
            for neuron_id in tqdm(neuron_list, desc='visual_reliability', unit='neurons', ncols=80):
                signal = get_neuron_signal(s2p, neuron_id, normalize=False)[0]

                with plot_figure(output.figure_output(neuron_id)) as ax:
                    ret = plot_cat_trial(ax, pattern, signal, image_time, s2p.fs,
                                         threshold=self.DEFAULT_TRIAL_THRES)
                csv(neuron_id, *ret)


def calc_visual_reliability(data: np.ndarray,
                            idx_onoff: np.ndarray,
                            fps: float,
                            threshold: float = 0.3,
                            plot_matrix: bool = False) -> tuple[bool, float]:
    """
    Check if is a visually-evoked cell by calculating the cross-trials correlation.
    Refer to Xu et., 2022. Nature Comm.

    Trial averaged responses (dF/F) for each stimulus condition was offset by the median of pre-stimulus (5 frames) activity.
    Neural responses during the visual stimulus epochs were concatenated for computing the **visual reliability index(r)**
    , which defined as the 75th percentile of the cross-trial correlation coefficients of the de-randomized response time courses.
    Cell with r > 0.3 (threshold) was classified as a visually responsive neuron

    `Dimension parameters`:

        T = number of stimulation trials

        C = number of stimulation condition

        F = number of recording frames foreach stimulation


    :param data: Neural responses. `Array[float, [T, C, F]]`
    :param idx_onoff: On/Off visual stimulation bool array. `Array[bool, F]`
    :param fps: Frame rate of the ``data``
    :param threshold: Threshold for identifying whether it is a responsive cell
    :param plot_matrix: Preview to see the correlation coefficient matrix
    :return: tuple of (reliability > threshold, reliability).

            -(reliability > threshold): whether the calculated reliability greater than threshold

            -(reliability): visual reliability

    """
    # find pre-stim
    n_trial, n_cond, n_frame = data.shape
    idx_stim_1 = np.nonzero(idx_onoff)[0][0]
    pre_stim = data[:, :, idx_stim_1 - 5: idx_stim_1]  # 5 frames of data before the stimulus
    pre_stim = np.mean(np.median(pre_stim, axis=2))

    # preprocessing of the calcium signals and grouped by trials
    proc_data = data - pre_stim
    proc_data = butter_lowpass_filter(proc_data, cutoff=fps * 0.2, fs=fps)
    proc_data = proc_data.reshape(n_trial, n_frame * n_cond)

    # pairwise
    r = np.zeros((n_trial, n_trial))
    for i in range(n_trial):
        for j in range(n_trial):
            r[i, j], _ = pearsonr(proc_data[i], proc_data[j])

    # preview
    if plot_matrix:
        with plot_figure(None) as ax:
            ax.imshow(r)
            for (i, j), val in np.ndenumerate(r):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    reliability = np.percentile(r, 75)

    return reliability > threshold, reliability


class VisualEpochPars(NamedTuple):
    is_visual: bool
    reliability: float
    max_vis_resp: float
    perc95_vis_resp: float


def plot_cat_trial(ax: Axes,
                   pattern: GratingPattern,
                   signal: np.ndarray,
                   image_time: np.ndarray,
                   fps: float, *,
                   threshold: float = 0.3) -> VisualEpochPars:
    """
    plot concat stimulus epoch calcium trace (order followed by stimuli types)
    colors represent different repetitive trials.

    :param ax:
    :param pattern:
    :param signal:
    :param image_time:
    :param fps: frame rate (for filter)
    :param threshold: reliability threshold
    :return:
        flag: whether a visually-evoked neurons
        visual reliability
        max signal response
    """
    cond = pattern.sftfdir_i()

    ct = collections.Counter()  # counter of trial
    cy = {}  # dict[(trial, cond)] = y, for storage
    ry = {}  # dict[(trial, cond)] = y, for non-normalized data storage
    cs = {}  # dict[nstim] = idx_onoff

    for si, st, sf, tf, ori in pattern.foreach_stimulus():
        df_fx = np.logical_and(st[0] - 1 <= image_time, image_time <= st[1] + 1)
        idx_onoff = np.logical_and(st[0] <= image_time, image_time <= st[1])[df_fx]  # 5s per each frame

        sig_trial = signal[df_fx]  # shape: 5 x fs (3s stimuli window + 1s pre-/post-stimulus)
        c = cond[(sf, tf, ori)]  # which condition
        t = ct[c]  # which trial
        ct[c] += 1

        ry[(t, c)] = sig_trial
        sig_trial = sig_trial / (np.max(signal))  # normalization

        cy[(t, c)] = sig_trial  # store into cy dict
        cs[len(sig_trial)] = idx_onoff

    # if the last stimulus trial is incomplete
    if len(set(ct.values())) > 1:
        raise RuntimeError('stimulus trials are incomplete, check protocol file...')

    ntrial = max(ct.values())
    ncond = len(cond)
    nframe_stim = max(map(len, ry.values()))  # number of frame per stim epoch

    data = np.zeros((ntrial, ncond, nframe_stim))
    rawdata = np.zeros((ntrial, ncond, nframe_stim))

    for (t, c), sig_trial in cy.items():
        data[t, c, :len(sig_trial)] = sig_trial

    for (t, c), sig_trial in ry.items():
        rawdata[t, c, :len(sig_trial)] = sig_trial

    idx_onoff = cs[nframe_stim]
    is_visual, idx_reliability = calc_visual_reliability(data, idx_onoff, fps, threshold=threshold)

    rawdata = rawdata.reshape((ntrial, ncond * nframe_stim))
    data = data.reshape((ntrial, ncond * nframe_stim))

    # remove transient noise for plotting
    rawdata = butter_lowpass_filter(rawdata, cutoff=fps * 0.2, fs=fps)
    data = butter_lowpass_filter(data, cutoff=fps * 0.2, fs=fps)

    # Plot
    for y, t in enumerate(data):
        ax.plot(t + y, linewidth=0.5)
        if y == len(data) - 1:
            yval = np.max(signal).astype(int)
            sbar = AnchoredScaleBar(ax.transData,
                                    sizey=1,
                                    labely=f'{yval}%',
                                    pad=0.1,
                                    color='black')

            ax.add_artist(sbar)

    ax.set(ylabel='Trial #', xlabel='frames')
    ax.set_yticks([it for it in range(ntrial)])
    ax.set_yticklabels(it + 1 for it in range(ntrial))
    ax.set_title(f'reliability: {idx_reliability:.4f}', loc='right')

    # For visual responses from non-normalized raw dff
    trial_avg_resp = np.mean(rawdata, axis=0)
    max_vis_resp = np.max(trial_avg_resp)
    perc95_vis_resp = np.percentile(trial_avg_resp, 95)

    return VisualEpochPars(is_visual, idx_reliability, max_vis_resp, perc95_vis_resp)


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    VisualReliabilityOptions().main()
