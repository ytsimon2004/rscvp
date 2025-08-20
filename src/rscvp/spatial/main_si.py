import numpy as np
import polars as pl
from matplotlib.axes import Axes
from tqdm import tqdm

from argclz import AbstractParser, as_argument
from neuralib.io import csv_header
from neuralib.plot import plot_figure, ax_merge
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.util import prepare_si_data, SiResult, SiShuffleResult
from rscvp.util.cli import PlotOptions, PositionShuffleOptions, DataOutput, NeuronID

__all__ = ['SiOptions']


@publish_annotation('main', project='rscvp')
class SiOptions(AbstractParser, PositionShuffleOptions, PlotOptions):
    DESCRIPTION = 'Calculate the spatial information, and plot with shuffle activity foreach cell'

    plot_summary: bool = as_argument(PlotOptions.plot_summary).with_options(help='plot spatial cumulative summary')

    signal_type = 'spks'

    def post_parsing(self):
        if self.plot_summary:
            self.reuse_output = True

        if self.virtual_env:
            self.session = 'all'

    def run(self):
        self.post_parsing()
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('si', self.session,
                                           running_epoch=self.running_epoch,
                                           virtual_env=self.virtual_env)

        if self.plot_summary:
            self.plot_si_cumulative(output_info)
        else:
            self.foreach_spatial_info(output_info, self.neuron_id)

    def foreach_spatial_info(self, output: DataOutput, neuron_ids: NeuronID):

        data = prepare_si_data(self, neuron_ids, virtual_env=self.virtual_env)

        headers = ['neuron_id', f'si_{self.session}', f'shuffled_si_{self.session}', f'place_cell_si_{self.session}']
        with csv_header(output.csv_output, headers) as csv:
            for neuron in tqdm(data.neuron_list, desc='calculate_si', unit='neurons', ncols=80):
                signal = data.signal(neuron)
                si_result = data.calc_si(signal)
                shuffled_result = data.calculate_si_shuffle(signal, self.shuffle_method)

                csv(
                    neuron,
                    round(si_result.spatial_info, 4),
                    round(shuffled_result.spatial_info, 4),
                    si_result.spatial_info > shuffled_result.spatial_info
                )

                with plot_figure(output.figure_output(neuron)) as ax:
                    plot_si(ax, si_result, shuffled_result)

    def plot_si_cumulative(self, output: DataOutput,
                           xlim: tuple[float, float] = (0, 2.5)):

        df = pl.read_csv(output.csv_output)
        idx = df[f'si_{self.session}'].arg_sort()  # sorted based on si index
        si = df[f'si_{self.session}'][idx]
        ss = df[f'shuffled_si_{self.session}'][idx]
        is_place_cell = df[f'place_cell_si_{self.session}']
        n_neuron = len(si)
        n_place_cell = np.count_nonzero(is_place_cell)

        with plot_figure(output.summary_figure_output(), 2, 2, figsize=(16, 6)) as _ax:
            ax = ax_merge(_ax)[0:, 0]
            y = np.linspace(0, 1, n_neuron)
            ax.plot(si, y, label=f'si (n = {n_neuron})')
            ax.plot(ss, y, label=f'shuffled_si (n = {n_neuron})')
            ax.set(xlabel='spatial info.', ylabel='Cum. Prob.', xlim=xlim, ylim=(0, 1.05))
            ax.legend()

            #
            ax = _ax[0, 1]
            ax.hist(si, bins=round(n_neuron / 5), range=xlim, color='black', label='si')
            ax.set(ylabel='Num. cell')
            ax.set_title(f'place cell: {n_place_cell}/{n_neuron}', loc='right')
            ax.legend()

            ax = _ax[1, 1]
            ax.hist(ss, bins=round(n_neuron / 5), range=xlim, color='grey', label='shuffled_si')
            ax.set(ylabel='Num. cell')
            ax.legend()


def plot_si(ax: Axes, result: SiResult, shuffle: SiShuffleResult):
    """
    Plot occupancy-normalized trial average activity & shuffled activity

    :param ax: ``Axes``
    :param result: ``SiResult``
    :param shuffle: ``SiShuffleResult``
    """
    x = result.position[1:]
    ax.plot(x, result.activity, label='activity')

    for act in shuffle.activity:
        ax.plot(x, act, color='r', alpha=0.1)

    ax.set_title(f'si: {result.spatial_info:.3f}\n' f'97.5% shuffled si: {shuffle.spatial_info:.3f}', loc='right')
    ax.set(xlabel='normalized position', ylabel='normalized activity')
    ax.legend()


if __name__ == '__main__':
    SiOptions().main()
