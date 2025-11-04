from argclz import try_int_type, AbstractParser, as_argument
from rscvp.statistic._var import GENERIC_HEADERS
from rscvp.statistic.csv_agg.collector import CSVCollector
from rscvp.statistic.csv_agg.core import LocalSpreadsheetSync
from rscvp.util.cli.cli_statistic import StatisticOptions


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

        super().__init__(opt, collector=collector, sheet_page='generic_parq')


class SpeedStat(LocalSpreadsheetSync):
    """speed score"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='sc',
            stat_col=['speed_score'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, collector=collector, sheet_page=opt.sheet_name)


class SpeedRunStat(LocalSpreadsheetSync):
    """speed score in run epoch"""

    def __init__(self, opt: StatisticOptions):
        opt.running_epoch = True

        collector = CSVCollector(
            code='sc',
            stat_col=['speed_score_run'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, collector=collector, sheet_page=opt.sheet_name)


class GenericStatAgg(AbstractParser, StatisticOptions):
    DESCRIPTION = 'Generic variables statistic aggregation'

    header = as_argument(StatisticOptions.header).with_options(choices=GENERIC_HEADERS)

    pre_selection = True

    def run(self):
        match self.header:
            case 'speed_score':
                stat = SpeedStat(self)
            case 'speed_score_run':
                stat = SpeedRunStat(self)
            case 'perc95_dff' | 'max_dff':
                stat = DFFStat(self)
            case _:
                raise ValueError(f'unknown header: {self.header}')

        if self.update:
            stat.update_spreadsheet(self.variable)


if __name__ == '__main__':
    GenericStatAgg().main()
