import unittest
import warnings
from unittest.mock import patch

from rscvp.statistic.persistence_agg.main_topo_celltype import TopoCellTypePersistenceAgg
from rscvp.topology.main_cls import ClsTopoOptions
from rscvp.topology.main_cords import RoiLocOptions
from rscvp.topology.main_fov import FOVOptions
from rscvp.topology.main_spatial_topo import SpatialTopoPlotOptions
from rscvp.topology.main_visual_topo import VisTopoPlotOptions
from .util import check_attr


class TestTopologyModule(unittest.TestCase):
    """Test in ``topology`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_cls_topo(self, *args):
        class Opt(ClsTopoOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            debug_mode = True

        check_attr(Opt, ClsTopoOptions)
        Opt().main([])

    def test_fov(self):
        class Opt(FOVOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0

        check_attr(Opt, FOVOptions)
        Opt().main([])

    def test_roi_location(self):
        class Opt(RoiLocOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            gspread_page = 'apcls_tac'
            debug_mode = True

        check_attr(Opt, RoiLocOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_spatial_topo(self, *args):
        class Opt(SpatialTopoPlotOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            gspread_page = 'apcls_tac'
            used_session = 'light'
            debug_mode = True

        check_attr(Opt, SpatialTopoPlotOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_visual_topo(self, *args):
        class Opt(VisTopoPlotOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            gspread_page = 'apcls_tac'
            used_session = 'light'
            debug_mode = True

        check_attr(Opt, VisTopoPlotOptions)
        Opt().main([])


class TestTopologyBatchModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_topo_celltype_batch(self, *args):
        class Opt(TopoCellTypePersistenceAgg):
            exp_date = '210315,210402'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            cell_type = 'spatial'
            debug_mode = True

        check_attr(Opt, TopoCellTypePersistenceAgg)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
