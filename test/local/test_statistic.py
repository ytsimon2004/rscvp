import time
import unittest
import warnings
from datetime import datetime
from typing import ClassVar
from unittest.mock import patch

from rscvp.statistic.csv_agg.main_pf_agg import PFStatAggOptions
from rscvp.statistic.csv_agg.main_spatial_agg import SpatialStatAggOptions
from rscvp.statistic.csv_agg.main_visual_agg import VisStatAggOptions
from rscvp.statistic.csv_agg.main_visual_dir_agg import VisDirStatAggOption
from rscvp.statistic.csv_agg.main_visual_sftf_agg import VZSFTFAggOption
from rscvp.statistic.main_normality import NormTestOption
from rscvp.statistic.main_para_mtx import ParaCorrMatOptions
from rscvp.statistic.parq.main_generic_gsp import GenericGSP
from rscvp.statistic.parq.main_pf_gsp import PFStatGSP
from rscvp.statistic.sql.main_cell_type import CellTypeStat
from rscvp.statistic.sql.main_decode_err import MedianDecodeErrorStat
from rscvp.statistic.sql.main_vp_fraction import VisuoSpatialFractionStat
from .util import check_attr


# ============================== #
# Test in Generic Statistic Test #
# ============================== #

class TestStatisticModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_normality(self, *args):
        class Opt(NormTestOption):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            debug_mode = True

        check_attr(Opt, NormTestOption)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_para_corr_mtx(self, *args):
        class Opt(ParaCorrMatOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            cell_type = 'spatial'
            annot = True
            debug_mode = True

        check_attr(Opt, ParaCorrMatOptions)
        Opt().main([])

    def test_session_statistic(self):
        """TODO"""
        pass


# ======================== #
# Test in ``StatPipeline`` #
# ======================== #

class TestStatisticPipeline(unittest.TestCase):
    DATETIME: ClassVar = datetime.today().strftime('%y%m%d')

    def setUp(self):
        time.sleep(20)  # avoid API error in gspread (quote limit)

    @patch('matplotlib.pyplot.show')
    def test_parq_generic(self, *args):
        class Opt(GenericGSP):
            exp_date = '210315,210409'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            header = 'perc95_dff'
            session = 'light'
            ttest_parametric_infer = True
            test_type = 'kstest'
            debug_mode = True

        Opt().main([])

    def test_non_session_agg(self):
        class Opt(SpatialStatAggOptions):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            header = 'ap_cords'
            sheet_name = 'ap_place'
            used_session = 'light'
            update = True

        opt = Opt()
        opt.main([])

    def test_visual_header_agg(self):
        class Opt(VisStatAggOptions):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            header = 'reliability'
            used_session = 'light'
            update = True

        opt = Opt()
        opt.main([])

    def test_visual_sftf_agg(self):
        class Opt(VZSFTFAggOption):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            header = 'sftf_amp_0.04 1'
            used_session = 'light'
            update = True

        opt = Opt()
        opt.main([])

    def test_visual_sftf_fraction(self):
        class Opt(VZSFTFAggOption):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            header = 'fraction'
            rec_region = 'aRSC,pRSC'
            used_session = 'light'

        opt = Opt()
        opt.main([])

    def test_visual_dir_agg(self):
        class Opt(VisDirStatAggOption):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            rec_region = 'aRSC,pRSC'  # pseudo
            plane_index = '0,0'
            header = 'dsi'
            used_session = 'light'
            update = True

        opt = Opt()
        opt.main()

    def test_visual_dir_parq(self):
        pass

    def test_pf_width_agg(self):
        class Opt(PFStatAggOptions):
            exp_date = '210315,210409'
            animal_id = 'YW006,YW006'
            rec_region = 'aRSC,pRSC'  # pseudo
            plane_index = '0,0'
            header = 'pf_width'
            session = 'light'
            used_session = 'light'
            truncate_session_agg = True
            update = True

        opt = Opt()
        opt.main([])

    def test_pf_width_parq(self):
        class Opt(PFStatGSP):
            exp_date = '210315,210409'
            animal_id = 'YW006,YW006'
            rec_region = 'aRSC,pRSC'
            plane_index = '0,0'
            header = 'pf_width'
            test_type = 'ttest'

        opt = Opt()
        opt.main([])

    # ======================= #
    # Non Agg Or SQL pipeline #
    # ======================= #

    @patch('matplotlib.pyplot.show')
    def test_vp_fraction_gspread(self, *args):
        class Opt(VisuoSpatialFractionStat):
            header = 'visual_frac'
            load_source = 'gspread'

        check_attr(Opt, VisuoSpatialFractionStat)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_vp_fraction_sql(self, *args):
        class Opt(VisuoSpatialFractionStat):
            header = 'spatial_frac'
            load_source = 'db'

        check_attr(Opt, VisuoSpatialFractionStat)

        try:
            Opt().main([])
        except RuntimeError as e:
            print(e)

    @patch('matplotlib.pyplot.show')
    def test_celltype_venn(self, *args):
        class Opt(CellTypeStat):
            show_type = 'combine'

        check_attr(Opt, CellTypeStat)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_median_decode(self, *args):
        class Opt(MedianDecodeErrorStat):
            pass

        check_attr(Opt, MedianDecodeErrorStat)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
