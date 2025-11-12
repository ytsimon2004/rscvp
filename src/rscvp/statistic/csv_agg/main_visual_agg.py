from argclz import AbstractParser, as_argument, try_int_type
from rscvp.statistic._var import VIS_HEADERS
from rscvp.statistic.csv_agg.core import ParquetSheetSync, NeuronDataAggregator
from rscvp.util.cli.cli_io import HEADER
from rscvp.util.cli.cli_statistic import StatisticOptions

__all__ = ['VisStatAggOptions']


class VisualStat(ParquetSheetSync):
    """used for general visual properties statistic"""

    def __init__(self, opt: StatisticOptions):
        collector = NeuronDataAggregator(
            code='vc',
            stat_col=[HEADER('reliability'), HEADER('max_vis_resp'), HEADER('perc95_vis_resp')],
            exclude_col=['visual_cell'],
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=False
        )

        super().__init__(opt, sheet_page='visual_parq', aggregator=collector)


class CordVisualStat(ParquetSheetSync):

    def __init__(self, opt: StatisticOptions):
        collector = NeuronDataAggregator(
            code='cord',
            stat_col=['ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=False
        )

        super().__init__(opt, sheet_page='visual_parq', aggregator=collector)


# ========== #

class VisStatAggOptions(AbstractParser, StatisticOptions):
    DESCRIPTION = 'find csv across dataset, update gspread, then aggregate to parquet file'

    header = as_argument(StatisticOptions.header).with_options(choices=VIS_HEADERS)

    pre_selection = True

    def post_parsing(self):
        match self.header:
            case 'reliability' | 'ap_cords' | 'ml_cords' | 'ap_cords_scale' | 'ml_cords_scale' | 'dv_cords':
                self.vc_selection = None  # for non-selected topology shown
            case _:
                self.vc_selection = 0.3

    def run(self):
        self.post_parsing()

        match self.header:
            case 'reliability' | 'max_vis_resp' | 'perc95_vis_resp':
                stat = VisualStat(self)
            case 'ap_cords' | 'ml_cords' | 'ap_cords_scale' | 'ml_cords_scale' | 'dv_cords':
                stat = CordVisualStat(self)
            case _:
                raise ValueError(f'unknown header: {self.header}')

        if self.update:
            stat.run_sync(self.variable)


if __name__ == '__main__':
    VisStatAggOptions().main()
