import unittest
import warnings
from unittest.mock import patch

from rscvp.model.bayes_decoding.main_cache_bayes import BayesDecodeCacheBuilder
from rscvp.model.bayes_decoding.main_decode_analysis import DecodeAnalysisOptions
from rscvp.statistic.persistence_agg.main_decode_err import BayesDecodePersistenceAgg
from .util import check_attr


class TestPositionDecodeModule(unittest.TestCase):
    """Test in ``model.decoding`` module and relevant statistic locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    def test_cache_build(self):
        class Opt(BayesDecodeCacheBuilder):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            invalid_cache = True

        check_attr(Opt, BayesDecodeCacheBuilder)
        Opt().main([])

    def test_cache_build_perc(self):
        class Opt(BayesDecodeCacheBuilder):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            random = '50%'
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            invalid_cache = True

        check_attr(Opt, BayesDecodeCacheBuilder)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_bayes_overview(self, *args):
        class Opt(DecodeAnalysisOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'overview'
            cache_version = 0
            debug_mode = True

        check_attr(Opt, DecodeAnalysisOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_bayes_median_decode_error(self, *args):
        class Opt(DecodeAnalysisOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'median_decode_error'
            cache_version = 0
            debug_mode = True

        check_attr(Opt, DecodeAnalysisOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_bayes_median_confusion_matrix(self, *args):
        class Opt(DecodeAnalysisOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'confusion_matrix'
            cache_version = 0
            debug_mode = True

        check_attr(Opt, DecodeAnalysisOptions)
        Opt().main()

    @patch('matplotlib.pyplot.show')
    def test_bayes_median_position_bins_error(self, *args):
        class Opt(DecodeAnalysisOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'position_bins_error'
            cache_version = 0
            debug_mode = True

        check_attr(Opt, DecodeAnalysisOptions)
        Opt().main()


# ====================== #
# Batch Statistic Module #
# ====================== #


class TestPositionDecodeBatchModule(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_bayes_persistence_agg_cmtx(self, *args):
        class Opt(BayesDecodePersistenceAgg):
            exp_date = '210315,210402'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'confusion_matrix'
            debug_mode = True

        check_attr(Opt, BayesDecodePersistenceAgg)
        Opt().main()

    @patch('matplotlib.pyplot.show')
    def test_bayes_persistence_agg_group_cmtx(self, *args):
        class Opt(BayesDecodePersistenceAgg):
            exp_date = '210315,210402'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'confusion_matrix'
            group_mode = True
            assign_group = (0, 1)
            debug_mode = True

        check_attr(Opt, BayesDecodePersistenceAgg)
        Opt().main()

    @patch('matplotlib.pyplot.show')
    def test_bayes_persistence_agg_pb(self, *args):
        class Opt(BayesDecodePersistenceAgg):
            exp_date = '210315,210402'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'position_bins_error'
            debug_mode = True

        check_attr(Opt, BayesDecodePersistenceAgg)
        Opt().main()

    @patch('matplotlib.pyplot.show')
    def test_bayes_persistence_agg_group_pb(self, *args):
        class Opt(BayesDecodePersistenceAgg):
            exp_date = '210315,210402'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            session = 'light'
            used_session = 'light'
            random = 200
            pos_bins = 100
            spatial_bin_size = 1.5
            cross_validation = 'odd'
            pre_selection = True
            analysis_type = 'position_bins_error'
            group_mode = True
            assign_group = (0, 1)
            debug_mode = True

        check_attr(Opt, BayesDecodePersistenceAgg)
        Opt().main()


if __name__ == '__main__':
    unittest.main()
