import numpy as np
from rscvp.util.cli.cli_celltype import CellTypeSelectionOptions

from argclz import AbstractParser, as_argument
from neuralib.plot import plot_figure

__all__ = ['OverviewSessionStat']


class OverviewSessionStat(AbstractParser, CellTypeSelectionOptions):
    DESCRIPTION = 'Compare fraction of variables (population overview) across different sessions'

    cell_type = as_argument(CellTypeSelectionOptions.cell_type).with_options(required=False, default=None)

    pre_selection = True
    pc_selection = None

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        ret = []
        for s in self.session_list:
            total = self.n_selected_neurons
            pre_mask = self.get_selected_neurons()
            pf_mask = self.get_csv_data(f'n_pf_{s}', session=s) > 0
            mx = pre_mask & pf_mask
            spatial = self.get_csv_data(f'nbins_exceed_{s}', session=s)[mx]
            n_spatial = np.count_nonzero(spatial > 0)
            frac = n_spatial / total
            ret.append(frac)
            print(f'spatial fraction in {s}: {frac:.2f}% ({n_spatial}/{total})')

        with plot_figure(None) as ax:
            ax.plot(ret)
            ax.set(xticks=[0, 1, 2], xticklabels=self.session_list, ylabel='spatial fraction')


if __name__ == '__main__':
    OverviewSessionStat().main()
