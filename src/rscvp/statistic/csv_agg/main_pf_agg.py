from rscvp.statistic._var import PF_HEADERS
from rscvp.statistic.csv_agg.collector import CSVCollector
from rscvp.statistic.csv_agg.core import LocalSpreadsheetSync
from rscvp.util.cli.cli_statistic import StatisticOptions

from argclz import AbstractParser, as_argument, try_int_type

__all__ = ['PFStatAggOptions']


class PFStat(LocalSpreadsheetSync):
    """place field properties"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='pf',
            stat_col=['n_pf', 'pf_width', 'pf_peak'],
            exclude_col=['pf_width_raw', 'pf_reliability', 'thres*'],  # * legacy no session in header
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, sheet_page='ap_place', collector=collector)


class PFStatAggOptions(AbstractParser, StatisticOptions):
    DESCRIPTION = 'place field properties comparison'

    header = as_argument(StatisticOptions.header).with_options(choices=PF_HEADERS)

    pc_selection = 'slb'
    pre_selection = True

    def run(self):
        pfs = PFStat(self)
        if self.update:
            pfs.update_spreadsheet(self.variable)


if __name__ == '__main__':
    PFStatAggOptions().main()
