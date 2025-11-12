from typing import Literal

from argclz import AbstractParser, as_argument, try_int_type
from rscvp.statistic._var import SPATIAL_HEADERS
from rscvp.statistic.csv_agg.core import ParquetSheetSync, NeuronDataAggregator
from rscvp.util.cli.cli_statistic import StatisticOptions


class SIStat(ParquetSheetSync):
    """spatial information"""

    def __init__(self, opt: StatisticOptions):
        collector = NeuronDataAggregator(
            code='si',
            stat_col=['si'],
            exclude_col=[f'shuffled_si', 'place_cell_si'],
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=opt.truncate_session
        )

        super().__init__(opt, aggregator=collector, sheet_page=opt.sheet_name)


class TCCStat(ParquetSheetSync):
    """median trial correlation coefficient"""

    def __init__(self, opt: StatisticOptions):
        collector = NeuronDataAggregator(
            code='tcc',
            stat_col=['trial_cc'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=opt.truncate_session
        )

        super().__init__(opt, sheet_page=opt.sheet_name, aggregator=collector)


class EVStat(ParquetSheetSync):
    """explained variance"""

    def __init__(self, opt: StatisticOptions):
        collector = NeuronDataAggregator(
            code='ev',
            stat_col=['ev_trial_avg'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=opt.truncate_session
        )

        super().__init__(opt, sheet_page=opt.sheet_name, aggregator=collector)


class TrStat(ParquetSheetSync):
    """trial reliability"""

    def __init__(self, opt: StatisticOptions):
        collector = NeuronDataAggregator(
            code='tr',
            stat_col=['trial_reliability'],
            exclude_col=['is_active*'],
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=opt.truncate_session
        )

        super().__init__(opt, sheet_page=opt.sheet_name, aggregator=collector)


class CordSpatialStat(ParquetSheetSync):

    def __init__(self, opt: StatisticOptions):
        opt.session = None

        collector = NeuronDataAggregator(
            code='cord',
            stat_col=['ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session=False
        )

        super().__init__(opt, sheet_page=opt.sheet_name, aggregator=collector)


# ====== #

class SpatialStatAggOptions(AbstractParser, StatisticOptions):
    DESCRIPTION = 'spatial related variables statistic, ' \
                  'accept for session statistic (ldl protocol)'

    header = as_argument(StatisticOptions.header).with_options(choices=SPATIAL_HEADERS)
    sheet_name: Literal['spatial_parq', 'dark_parq']

    pre_selection = True
    pc_selection = 'slb'

    def run(self):

        match self.header:
            case 'si':
                stat = SIStat(self)
            case 'trial_cc':
                stat = TCCStat(self)
            case 'ev_trial_avg':
                stat = EVStat(self)
            case 'trial_reliability':
                stat = TrStat(self)
            case 'ap_cords' | 'ml_cords' | 'ap_cords_scale' | 'ml_cords_scale' | 'dv_cords':
                stat = CordSpatialStat(self)
            case _:
                raise ValueError(f'unknown header: {self.header}')

        if self.update:
            stat.update_sync(self.variable)


if __name__ == '__main__':
    SpatialStatAggOptions().main()
