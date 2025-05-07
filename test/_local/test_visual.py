import unittest
import warnings
from unittest.mock import patch

from rscvp.statistic.persistence_agg.main_sftf_tuning import SFTFMapPersistenceAgg
from rscvp.statistic.sql.main_osds_pie import OriDirPieStat
from rscvp.visual.main_polar import VisualPolarOptions
from rscvp.visual.main_reliability import VisualReliabilityOptions
from rscvp.visual.main_sftf_map import VisualSFTFMapOptions
from rscvp.visual.main_sftf_pref import VisualSFTFPrefOptions
from rscvp.visual.main_tuning import VisualTuningOptions
from ._util import check_attr


class TestVisualModule(unittest.TestCase):
    """Test in ``visual`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_visual_trace(self, *args):
        class Opt(VisualTuningOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            invalid_cache = True
            debug_mode = True

        check_attr(Opt, VisualTuningOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_visual_polar(self, *args):
        class Opt(VisualPolarOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            debug_mode = True

        check_attr(Opt, VisualPolarOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_visual_polar_summary(self, *args):
        class Opt(VisualPolarOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            summary = True

        check_attr(Opt, VisualPolarOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_visual_reliability(self, *args):
        class Opt(VisualReliabilityOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            debug_mode = True

        check_attr(Opt, VisualReliabilityOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_sftf_pref(self, *args):
        class Opt(VisualSFTFPrefOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            debug_mode = True

        check_attr(Opt, VisualSFTFPrefOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_sftf_pref_summary(self, *args):
        class Opt(VisualSFTFPrefOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            batch_type = 'dff'

        check_attr(Opt, VisualSFTFPrefOptions)
        Opt().main([])

        class Opt(VisualSFTFPrefOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            batch_type = 'fraction'

        check_attr(Opt, VisualSFTFPrefOptions)
        Opt().main([])

    def test_sftf_fit(self):
        pass

    @patch('matplotlib.pyplot.show')
    def test_sftf_map_cache(self, *args):
        class Opt(VisualSFTFMapOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            used_session = 'light'
            sftf_type = ('0.04', '4')
            plane_index = 0
            value_type = 'mean'
            dir_agg_type = 'max'
            invalid_cache = True

        Opt().main([])


class TestVisualBatchModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_sftf_tuning_batch(self, *args):
        class Opt(SFTFMapPersistenceAgg):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            sftf_type = ('0.04', '4')
            debug_mode = True

        check_attr(Opt, SFTFMapPersistenceAgg)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_osds_pie(self, *args):
        class Opt(OriDirPieStat):
            pass

        Opt().main([])


if __name__ == '__main__':
    unittest.main()
