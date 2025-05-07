import numpy as np
from matplotlib.axes import Axes

from argclz import AbstractParser
from neuralib.imaging.suite2p import Suite2PResult, get_neuron_signal
from neuralib.plot import plot_figure
from rscvp.util.cli.cli_suite2p import Suite2pOptions, get_neuron_list


class SimpleOptoOptions(AbstractParser, Suite2pOptions):
    DESCRIPTION = 'simple testing for asli exp of LP silencing'

    s2p: Suite2PResult

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.s2p = self.load_suite_2p(channel=0)

        self.peri_opto_heatmap()

    def peri_opto_heatmap(self):
        pmt2 = self.load_suite_2p(channel=1).f_raw[0]
        opto_mask = get_opto_segmentation_pmt2(pmt2)

        neuron_list = get_neuron_list(self.s2p)
        signal = get_neuron_signal(self.s2p, neuron_list)[0]

        act, cutoff = foreach_trial_activity(signal, opto_mask)
        cutoff = [c / self.s2p.fs for c in cutoff]

        with plot_figure(None, 1, len(cutoff)) as ax:
            for i, a in enumerate(act):
                self.plot_peri_opto_heatmap(ax[i], a)
                ax[i].axvline(cutoff[i], ls='--', color='r', lw=1, alpha=0.8)

    def plot_peri_opto_heatmap(self, ax: Axes,
                               act: np.ndarray):
        """

        :param ax
        :param act: activity (N, F)
        :return:
        """

        ax.imshow(act,
                  aspect='auto',
                  extent=[0, act.shape[1] / self.s2p.fs, 0, act.shape[0]],
                  cmap='Greys')

        ax.set(xlabel='# time(s)', ylabel='# neurons')


def foreach_trial_activity(signal: np.ndarray,
                           opto_mask: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """
    get the `foreach trial` activity, and frame index

    :param signal
    :param opto_mask:
    :return:
        act: list of activity: [(N, F), ...]. length equal to one onoff, n_frames(F) might different
        cutff: list of frame index: [Fidx, ...]
    """
    blocks = np.nonzero(np.diff(opto_mask.astype(int)) < 0)[0]
    blocks = np.concatenate([[0], blocks + 1])

    cutoffs = []
    act = []
    for (left, right) in zip(blocks[:-1], blocks[1:]):
        signal_per_trial = signal[:, left:right]  # (N, T)
        mask_per_trial = opto_mask[left:right]  # (T)
        baseline = signal_per_trial[:, ~mask_per_trial]
        opto = signal_per_trial[:, mask_per_trial]
        cutoffs.append(baseline.shape[1])
        act.append(np.hstack([baseline, opto]))

    return act, cutoffs


def get_opto_segmentation_pmt2(pmt2_signal: np.ndarray,
                               threshold: float = 10_000,
                               tolerance_frame: int = 50) -> np.ndarray:
    """
    based on pmt2 signal (saturation) to get the non-opto/opto signal segmentation

    :param pmt2_signal: pmt2_signal, could be from any cell
    :param threshold: signal greater than which value, considered as opto epoch
    :param tolerance_frame: due to manually laser switching artifact, oscillating disappear. thus merge for segmentation
    :return:
        bool array: 0, non-opto; 1, opto epoch
    """

    opto_epoch = (pmt2_signal > threshold).astype(int)
    epoch_index = np.cumsum(abs(np.diff(opto_epoch, prepend=opto_epoch[0])))

    if pmt2_signal[0] > threshold:
        epoch_index += 1

    # merge
    for frame in range(2, np.max(epoch_index) + 1, 2):
        nframe = np.count_nonzero(epoch_index == frame)
        if nframe < tolerance_frame:
            epoch_index[epoch_index == frame] += 1
            epoch_index[epoch_index == frame - 1] += 2

    opto_mask = epoch_index % 2 != 0

    return opto_mask


if __name__ == '__main__':
    SimpleOptoOptions().main()
