import numpy as np
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_stimpy import StimpyOptions
from rscvp.util.cli.cli_suite2p import Suite2pOptions
from rscvp.util.util_trials import TrialSignal
from tqdm import tqdm

from argclz import AbstractParser, argument
from neuralib.io import csv_header
from neuralib.util.verbose import publish_annotation

__all__ = ['DffSesOption']


@publish_annotation('main', project='rscvp', as_doc=True)
class DffSesOption(AbstractParser, Suite2pOptions, StimpyOptions):
    DESCRIPTION = 'Calculate the mean/median/percentile/max dff in every recording sessions'

    transient_cutoff: float | None = argument(
        '--t-cutoff', '--tcutoff',
        default=None,
        help='cutoff value for the transient calcium signal',
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('ds', self.session)
        self.get_dff_amplitude(output_info)

    def get_dff_amplitude(self, output: DataOutput):
        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()
        cal = TrialSignal.of_calcium(s2p, rig, self.neuron_id, self.plane_index, session=self.session)
        dff = cal.dff

        headers = ['neuron_id',
                   f'mean_dff_{self.session}',
                   f'median_dff_{self.session}',
                   f'perc95_dff_{self.session}',
                   f'max_dff_{self.session}']

        with csv_header(output.csv_output, headers) as csv:
            for n in tqdm(range(cal.n_neurons), unit='neurons'):
                act = dff[n]
                if self.transient_cutoff is not None:
                    act = act[act < self.transient_cutoff]

                mean_act = np.mean(act)
                median_act = np.median(act)
                perc_act = np.percentile(act, 95)
                max_act = np.max(act)

                csv(n, mean_act, median_act, perc_act, max_act)


if __name__ == '__main__':
    DffSesOption().main()
