import numpy as np
from matplotlib.axes import Axes
from typing import Optional

from argclz import AbstractParser, argument, int_tuple_type
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from neuralib.suite2p import get_neuron_signal
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_stimpy import StimpyOptions
from rscvp.util.cli.cli_suite2p import Suite2pOptions, get_neuron_list

__all__ = ['NeuropilCorrOption',
           'plot_dff_heat']


class NeuropilCorrOption(AbstractParser, Suite2pOptions, StimpyOptions):
    DESCRIPTION = 'see the effect of neuropil correction'

    dff_range: tuple[int, int] | None = argument(
        '--dff-range', '--drange',
        type=int_tuple_type,
        default=(-10, 10),
        help='dff value for the heatmap range, for visualization',
    )

    time_range: tuple[int, int] = argument(
        '--time-range', '--trange',
        metavar='TIME(S)',
        type=int_tuple_type,
        default=(0, 20),
        help='time range (in sec) for plotting',
    )

    signal_type = 'df_f'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('npc')
        self.uncorr_corr_sig(output_info)

    def uncorr_corr_sig(self, output: DataOutput):
        """
        followed Bonin et al., 2011. JN - Fig.2C.
        To see if there is contamination of neuropil signal

        :param output:
        :return:
        """
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, None)  # all neurons

        dff_uncorr = get_neuron_signal(s2p, neuron_list, normalize=False, correct_neuropil=False)[0]
        dff_corr = get_neuron_signal(s2p, neuron_list, normalize=False, correct_neuropil=True)[0]

        start = int(self.time_range[0] * s2p.fs)
        end = int(self.time_range[1] * s2p.fs)

        dff_uncorr = dff_uncorr[:, start:end]
        dff_corr = dff_corr[:, start:end]

        output_file = output.summary_figure_output()

        with plot_figure(output_file, 1, 2, figsize=(8, 4)) as ax:
            plot_dff_heat(ax[0], dff_uncorr, s2p.fs, title='uncorrected', vmin=self.dff_range[0],
                          vmax=self.dff_range[1])
            plot_dff_heat(ax[1], dff_corr, s2p.fs, title='corrected', vmin=self.dff_range[0], vmax=self.dff_range[1])


def plot_dff_heat(ax: Axes,
                  signal: np.ndarray,
                  fs: float,
                  title: str,
                  cbar_label: str = 'dF/F (%)',
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None):
    """
    heatmap for neuropil-uncorrected / corrected dF/F visualization

    :param ax:
    :param signal: (N, fs*s)
    :param fs: sampling rate of imaging
    :param title
    :param cbar_label
    :param vmin
    :param vmax
    :return:
    """

    im = ax.imshow(
        signal,
        extent=(0, signal.shape[1] / fs, 0, signal.shape[0]),
        cmap='gray',
        interpolation='none',
        aspect='auto',
        origin='lower',
        vmin=vmin,
        vmax=vmax
    )

    ax.set(xlabel='time(s)', ylabel='neurons #')
    ax.set_title(title)

    cbar = insert_colorbar(ax, im)
    cbar.ax.set_ylabel(cbar_label)


if __name__ == '__main__':
    NeuropilCorrOption().main()
