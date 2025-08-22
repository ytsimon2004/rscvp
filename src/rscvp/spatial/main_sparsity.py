import numpy as np
from tqdm import tqdm

from argclz import AbstractParser
from neuralib.io import csv_header
from rscvp.spatial.util import prepare_si_data
from rscvp.util.cli import SelectionOptions, PositionShuffleOptions, DataOutput, NeuronID

__all__ = ['SparsityOptions']


class SparsityOptions(AbstractParser, PositionShuffleOptions, SelectionOptions):
    DESCRIPTION = 'Calculate spatial sparsity'

    signal_type = 'spks'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('spr', self.session,
                                           running_epoch=self.running_epoch,
                                           use_virtual_space=self.use_virtual_space)
        self.foreach_spatial_sparsity(output_info, self.neuron_id)

    def foreach_spatial_sparsity(self,
                                 output: DataOutput,
                                 neuron_ids: NeuronID):
        r"""
        measure of the fraction of the environment in which the cell is active (Skaggs et al., 1996)
        .. math::  \frac{\sum (p_i \lambda_i)^2}{\sum p_i \lambda_i ^2}

       :label: math-sample

        """
        data = prepare_si_data(self, neuron_ids, use_virtual_space=self.use_virtual_space)
        with csv_header(output.csv_output, ['neuron_id', f'sparsity_{self.session}']) as csv:
            for neuron in tqdm(data.neuron_list, desc='sparsity', unit='neuron', ncols=80):
                signal = data.signal(neuron, 'spks')
                si_result = data.calc_si(signal)
                occp = si_result.occupancy
                lda_i = si_result.activity
                pi = occp / np.sum(occp)
                sparsity = (np.sum(pi * lda_i) ** 2) / (np.sum(pi * (lda_i ** 2)))

                csv(neuron, sparsity)


if __name__ == '__main__':
    SparsityOptions().main()
