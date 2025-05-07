import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from rscvp.util.cli.cli_celltype import CellTypeSelectionOptions
from rscvp.util.cli.cli_output import DataOutput

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure

__all__ = ['ParaCorrMatOptions']


class ParaCorrMatOptions(AbstractParser, CellTypeSelectionOptions):
    DESCRIPTION = 'Pairwise pearson correlation between each selected parameters in a certain cell type'

    annot: bool = argument(
        '--annot',
        action='store_true',
        help='whether show cc in the matrix of individual cells',
    )

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.masking_celltype(self.cell_type)

    def run(self):
        self.post_parsing()
        output_info = self.get_data_output('var')
        self.pairwise_corr_matrix(output_info)

    def pairwise_corr_matrix(self, output: DataOutput):
        """plot the pairwise correlation matrix across each analyzed parameter"""
        df = self.select_dataframe(self.session, to_zscore=True).to_pandas()
        pcorr = df.corr(method='pearson')

        output_file = output.summary_figure_output(f'{self.cell_type}_var')
        with plot_figure(output_file) as ax:
            self.plot_corr_matrix(ax, pcorr)
            ax.set_title(f'{self.cell_type} neurons: {self.n_selected_neurons} / {self.n_total_neurons}')

    def plot_corr_matrix(self, ax: Axes, data: pd.DataFrame):
        """pairwise correlation matrix"""
        sns.heatmap(data,
                    ax=ax,
                    vmin=-1, vmax=1, center=0,
                    cmap=sns.diverging_palette(230, 20, n=200),
                    square=True,
                    xticklabels=True,
                    yticklabels=True,
                    annot=self.annot)

        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           horizontalalignment='right')

        ax.set_yticklabels(ax.get_yticklabels())


if __name__ == '__main__':
    ParaCorrMatOptions().main()
