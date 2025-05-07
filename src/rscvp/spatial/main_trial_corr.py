import numpy as np
import seaborn as sns
from tqdm import tqdm

from argclz import AbstractParser
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.util.cli import DataOutput, SelectionOptions, get_neuron_list
from rscvp.util.util_trials import TrialSelection

__all__ = ['TrialCorrOptions']


@publish_annotation('main', project='rscvp', as_doc=True)
class TrialCorrOptions(AbstractParser, ApplyPosBinActOptions, SelectionOptions):
    DESCRIPTION = 'Calculate median value of pairwise trial to trial activity correlation'

    signal_type = 'df_f'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        dat = self.select_trials_activity()
        output_info = self.get_data_output('tcc', self.session)
        self.foreach_trial_cc(dat, output_info)

    def select_trials_activity(self) -> np.ndarray:
        rig = self.load_riglog_data()

        if self.plane_index is not None:
            act = self.apply_binned_act_cache().occ_activity
        else:
            raise NotImplementedError('')

        indices = TrialSelection.from_rig(rig, self.session).get_time_profile().trial_range
        act = act[:, slice(*indices), :]

        return act

    def foreach_trial_cc(self, data: np.ndarray, output: DataOutput):
        """
        Reliability was defined as the pairwise Pearson correlation between the activity on each trial.
        Refer to Pettit et al., 2022. Nature Neuroscience

        `Dimension parameters`:

            N = Number of neurons

            L' = Selected trial activity

            B = Number of position bins

        :param data: `Array[float, [N, L', B]]`
        :param output: ``DataOutput``
        """

        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, self.neuron_id)

        with csv_header(output.csv_output, ['neuron_id', f'trial_cc_{self.session}']) as csv:
            for neuron in tqdm(neuron_list, desc='trial_cc', unit='neurons', ncols=80):
                corr_matrix = np.corrcoef(data[neuron])
                mx = np.triu(np.ones_like(corr_matrix, dtype=bool))

                # take upper for median across all laps
                upper_triangle = np.triu_indices_from(corr_matrix, k=1)
                x = corr_matrix[upper_triangle]
                csv(neuron, np.nanmedian(x))

                with plot_figure(output.figure_output(neuron), set_square=True) as ax:
                    sns.heatmap(
                        corr_matrix,
                        mask=mx,
                        cmap=(sns.diverging_palette(230, 20, as_cmap=True)),
                        center=0,
                        square=True,
                        linewidths=.5,
                        cbar_kws={"shrink": .5, 'label': 'corr. coef.'},
                        ax=ax
                    )
                    ax.set(xlabel='# Trial', ylabel='# Trial')


if __name__ == '__main__':
    TrialCorrOptions().main()
