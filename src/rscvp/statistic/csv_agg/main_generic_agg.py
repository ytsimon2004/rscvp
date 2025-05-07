from rscvp.statistic._var import GENERIC_HEADERS
from rscvp.statistic.csv_agg.collector import CSVCollector
from rscvp.statistic.csv_agg.core import LocalSpreadsheetSync
from rscvp.util.cli.cli_statistic import StatisticOptions

from argclz import try_int_type, AbstractParser, as_argument


class DFFStat(LocalSpreadsheetSync):
    """DF/F amplitude"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='ds',
            stat_col=['perc95_dff', 'max_dff'],
            exclude_col=['mean_dff', 'median_dff'],
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, collector=collector, sheet_page='ap_generic')


class GenericStatAgg(AbstractParser, StatisticOptions):
    DESCRIPTION = 'Generic variables statistic aggregation'

    header = as_argument(StatisticOptions.header).with_options(choices=GENERIC_HEADERS)

    pre_selection = True
    stat: DFFStat

    def run(self):
        if self.header in ('perc95_dff', 'max_dff'):
            self.stat = DFFStat(self)
        else:
            raise ValueError('')

        if self.update:
            self.stat.update_spreadsheet(self.variable)


if __name__ == '__main__':
    GenericStatAgg().main()
