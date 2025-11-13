import numpy as np
from matplotlib import pyplot as plt
from rich.pretty import pprint

from argclz import as_argument, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.cli_gspread import GSPExtractor
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli import StatisticOptions


@publish_annotation('appendix', project='rscvp', as_doc=True, caption='rev')
class SpeedParQ(StatPipeline):
    DESCRIPTION = 'Speed score histogram (note the negative value is meaningful, not use mean-based statistics)'

    header = as_argument(StatisticOptions.header).with_options(
        choices=['speed_score', 'speed_score_run'],
        required=False,
        default='speed_score'
    )

    visual_only: bool = argument('--vis', help='filter with only visual cells')

    sheet_name = as_argument(StatisticOptions.sheet_name).with_options(required=True)
    load_source = 'parquet'
    test_type = 'kstest'

    def run(self):
        self.load_table(to_pandas=False)

        self.run_pipeline()

    def plot(self):
        data = self.post_processing()

        with plot_figure(self.output_figure) as ax:
            for k, v in data.items():
                n = len(v)
                weights = np.ones_like(v) * (100.0 / n)  # each value contributes equally to total 100%
                ax.hist(v, bins=50, weights=weights, histtype='step', label=k)

            ax.set(xlabel='Speed score', ylabel='percent (%)')
            plt.legend()

    def post_processing(self):
        df_ss = self.df.select('Data', 'region', self.header)

        if not self.visual_only:
            ret = {}
            for subset in df_ss.partition_by('region'):
                ss = subset.explode(self.header)[self.header].to_numpy()
                ret[subset['region'].unique().item()] = ss

            return ret

        else:
            df_vis = GSPExtractor('visual_parq').load_parquet_file(
                self.statistic_dir,
                session_melt_header=None,
                primary_key='Data'
            ).select('Data', 'region', 'reliability')

            df = df_ss.join(df_vis, on=['Data', 'region'], how='inner')

            ret = {}
            for subset in df.partition_by('region'):
                rel = subset.explode('reliability')['reliability'].to_numpy()
                mx = rel > 0.3
                ss = subset.explode(self.header)[self.header].to_numpy()[mx]

                ret[subset['region'].unique().item()] = ss

            pprint(ret)

            return ret


if __name__ == '__main__':
    SpeedParQ().main()
