from typing import Literal

from argclz import AbstractParser, as_argument, try_int_type
from rscvp.statistic._var import SPATIAL_HEADERS
from rscvp.statistic.csv_agg.collector import CSVCollector
from rscvp.statistic.csv_agg.core import LocalSpreadsheetSync
from rscvp.util.cli.cli_statistic import StatisticOptions


class SIStat(LocalSpreadsheetSync):
    """spatial information"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='si',
            stat_col=['si'],
            exclude_col=[f'shuffled_si', 'place_cell_si'],
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, collector=collector, sheet_page=opt.sheet_name)


class SpeedStat(LocalSpreadsheetSync):
    """speed score"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='ss',
            stat_col=['speed_score'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, collector=collector, sheet_page=opt.sheet_name)


class TCCStat(LocalSpreadsheetSync):
    """median trial correlation coefficient"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='tcc',
            stat_col=['trial_cc'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, sheet_page=opt.sheet_name, collector=collector)


class EVStat(LocalSpreadsheetSync):
    """explained variance"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='ev',
            stat_col=['ev_trial_avg'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, sheet_page=opt.sheet_name, collector=collector)


class TrStat(LocalSpreadsheetSync):
    """trial reliability"""

    def __init__(self, opt: StatisticOptions):
        collector = CSVCollector(
            code='tr',
            stat_col=['trial_reliability'],
            exclude_col=['is_active*'],
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=opt.truncate_session_agg
        )

        super().__init__(opt, sheet_page=opt.sheet_name, collector=collector)


class CordSpatialStat(LocalSpreadsheetSync):

    def __init__(self, opt: StatisticOptions):
        opt.session = None

        collector = CSVCollector(
            code='cord',
            stat_col=['ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords'],
            exclude_col=None,
            fields=dict(rec_region=str, plane_index=try_int_type),
            truncate_session_agg=False
        )

        super().__init__(opt, sheet_page=opt.sheet_name, collector=collector)


# ====== #

class SpatialStatAggOptions(AbstractParser, StatisticOptions):
    DESCRIPTION = 'spatial related variables statistic, ' \
                  'accept for session statistic (ldl protocol)'

    header = as_argument(StatisticOptions.header).with_options(choices=SPATIAL_HEADERS)
    sheet_name: Literal['ap_place', 'ap_ldl']

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
            case 'speed_score':
                stat = SpeedStat(self)
            case _:
                raise ValueError(f'unknown header: {self.header}')

        if self.update:
            stat.update_spreadsheet(self.variable)


if __name__ == '__main__':
    SpatialStatAggOptions().main()
