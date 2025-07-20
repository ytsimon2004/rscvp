import numpy as np
from matplotlib import pyplot as plt

from argclz import as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli import StatisticOptions


@publish_annotation('appendix', project='rscvp', as_doc=True, caption='rev')
class SpeedStatGSP(StatPipeline):
    DESCRIPTION = 'Speed score histogram (note the negative value is meaningful, not use mean-based statistics)'

    header = as_argument(StatisticOptions.header).with_options(
        choices=['speed_score', 'speed_score_run'],
        required=False,
        default='speed_score'
    )

    sheet_name = as_argument(StatisticOptions.sheet_name).with_options(required=True)
    load_source = 'parquet'
    test_type = 'kstest'

    def run(self):
        self.load_table(to_pandas=False)
        self.run_pipeline()

    def plot(self):
        data = self.get_collect_data().data
        output = self.get_output_figure_type()

        with plot_figure(output) as ax:
            for k, v in data.items():
                n = len(v)
                weights = np.ones_like(v) * (100.0 / n)  # each value contributes equally to total 100%
                ax.hist(v, bins=50, weights=weights, histtype='step', label=k)

            ax.set(xlabel='Speed score', ylabel='percent (%)')
            plt.legend()


if __name__ == '__main__':
    SpeedStatGSP().main()
