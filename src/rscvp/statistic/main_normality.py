import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib.axes import Axes
from rscvp.util.cli.cli_celltype import CellTypeSelectionOptions
from rscvp.util.cli.cli_output import DataOutput

from argclz import AbstractParser, as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation

__all__ = ['NormTestOption']


@publish_annotation('appendix')
class NormTestOption(AbstractParser, CellTypeSelectionOptions):
    DESCRIPTION = 'See the distribution of each variables and do the normality test (all neurons)'

    cell_type = as_argument(CellTypeSelectionOptions.cell_type).with_options(
        default='spatial',
        required=False,
    )

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('var')
        self.normality_test(output_info)

    def normality_test(self, output: DataOutput):
        """

        :param output:
        :return:
        """
        df = self.select_dataframe(self.session, to_zscore=True)

        stats = pg.normality(data=df.to_pandas(), method='normaltest')
        stats.to_csv((output.directory / 'normality_result').with_suffix('.csv'),
                     float_format='%.4e')

        output_file = output.summary_figure_output(
            f'{self.cell_type}_var',
            'normality-all-neurons'
        )

        n_paras = len(stats)
        r = int(np.sqrt(n_paras))
        c = int(n_paras // r + 1)

        with plot_figure(output_file, r, c) as ax:
            ax = ax.ravel()
            i = 0
            for series in df:
                if i < n_paras:
                    plot_norm_hist(ax[i], series, stats['pval'].iloc[i], xlabel=series.name)
                    i += 1

            for i in range(n_paras, (r * c)):
                ax[i].set_visible(False)


def plot_norm_hist(ax: Axes,
                   data: pd.Series,
                   pval: float,
                   **kwargs):
    """

    :param ax:
    :param data: variable (N,)
    :param pval: p-value of the normality test
    :return:
    """
    sns.histplot(data=data, bins=20, kde=True, ax=ax)

    ax.set_title(f'p: {pval:.4f}', loc='right', fontsize=8)
    ax.set(ylabel='#counts', **kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


if __name__ == '__main__':
    NormTestOption().main()
