import polars as pl

from argclz import AbstractParser, as_argument, try_int_type
from rscvp.statistic._var import PF_HEADERS
from rscvp.statistic.csv_agg.core import ParquetSheetSync, NeuronDataAggregator
from rscvp.util.cli.cli_statistic import StatisticOptions

__all__ = ['PFStatAggOptions']


class PFStat(ParquetSheetSync):
    """place field properties"""

    def __init__(self, opt: StatisticOptions):
        aggregator = NeuronDataAggregator(
            code='pf',
            stat_col=['n_pf', 'pf_width', 'pf_peak'],
            exclude_col=['pf_width_raw', 'pf_reliability', 'thres*'],  # * legacy no session in header
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=opt.truncate_session,
        )

        super().__init__(opt, sheet_page=opt.sheet_name, aggregator=aggregator)


class PFStatAggOptions(AbstractParser, StatisticOptions):
    DESCRIPTION = 'place field properties comparison'

    header = as_argument(StatisticOptions.header).with_options(choices=PF_HEADERS)

    pc_selection = 'slb'
    pre_selection = True

    def run(self):
        pfs = PFStat(self)
        if self.update:
            dtype = pl.List(pl.Int8) if 'n_pf' in self.variable else pl.List(pl.Utf8)
            pfs.run_sync(self.variable, dtype=dtype)


if __name__ == '__main__':
    PFStatAggOptions().main()
