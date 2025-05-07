from typing import cast, Final

import numpy as np
from matplotlib.axes import Axes
from rscvp.util.cli import DataOutput, PlotOptions, StimpyOptions, Suite2pOptions, get_neuron_list, NeuronID
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from argclz import AbstractParser
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation

__all__ = ['NeuropilErrOptions']


@publish_annotation('main', project='rscvp', caption='selection criteria', as_doc=True)
class NeuropilErrOptions(AbstractParser, Suite2pOptions, StimpyOptions, PlotOptions):
    DESCRIPTION = 'Check if any error in neuropil extraction'

    session: Final[str] = 'all'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        output_info = self.get_data_output('np', self.session)
        self.foreach_neuropil_error(output_info, self.neuron_id)

    def foreach_neuropil_error(self, output: DataOutput, neuron_ids: NeuronID):
        """
        Simply exclude the cell that has larger mean_Fneu than mean_F
        (overlapping pixels in neuropil estimation,resemble to issue, https://github.com/MouseLand/suite2p/issues/613),
        and calculate the percentage of exceed.
        """

        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        with csv_header(output.csv_output, ['neuron_id', 'error_perc']) as csv:
            for n in tqdm(neuron_list, desc='neuropil_err', unit='neuron', ncols=80):
                f = gaussian_filter1d(s2p.f_raw[n, :], 3)
                fneu = gaussian_filter1d(s2p.f_neu[n, :], 3)

                with plot_figure(output.figure_output(n), 1, 2) as ax:
                    err = plot_ffneu(ax[0], f, fneu)
                    plot_ffneu(ax[1], f, fneu, frames=slice(-5000, -1))

                csv(n, err)


def plot_ffneu(ax: Axes,
               f: np.ndarray,
               fneu: np.ndarray,
               frames: slice | None = None) -> float:
    """
    Plot extracted somata and neuropil fluorescence traces

    :param ax: ``Axes``
    :param f: Fluorescence traces. `Array[float, F]`
    :param fneu: Neuropil fluorescence trace. `Array[float, F]`
    :param frames: Frame interval for plotting. If none then plot the first 5000 frames.
    :return: error_perc: negative percentage value if the mean_fneu > mean_f
    """
    if frames is None:
        frames = slice(0, 5000)

    mean_f = cast(float, np.mean(f))
    mean_fneu = cast(float, np.mean(fneu))
    error_perc = round((mean_f - mean_fneu) / mean_fneu * 100, 2)

    ax.plot(f[frames], color='k', label='f', alpha=0.3)
    ax.plot(fneu[frames], color='r', label='fneu', alpha=0.3)
    ax.axhline(mean_f, color='k')
    ax.axhline(mean_fneu, color='r')
    ax.legend()

    return error_perc


if __name__ == '__main__':
    NeuropilErrOptions().main()
