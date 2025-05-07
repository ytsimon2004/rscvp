import unittest
import warnings
from unittest.mock import patch

from rscvp.spatial.main_align_map import AlignPeakMapOptions
from rscvp.spatial.main_ev_pos import EVOptions
from rscvp.spatial.main_place_field import PlaceFieldsOptions
from rscvp.spatial.main_population_matrix import PopulationMTXOptions
from rscvp.spatial.main_position_map import PositionMapOptions
from rscvp.spatial.main_si import SiOptions
from rscvp.spatial.main_slb import PositionLowerBoundOptions
from rscvp.spatial.main_sparsity import SparsityOptions
from rscvp.spatial.main_trial_corr import TrialCorrOptions
from rscvp.statistic.persistence_agg.main_si_sorted_occ import SISortAlignPersistenceAgg
from rscvp.statistic.persistence_agg.main_trial_avg_position import PositionBinPersistenceAgg
from ._util import check_attr


class TestSpatialModule(unittest.TestCase):
    """Test in ``spatial`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_linear_tuning(self, *args):
        class Opt(PositionMapOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            debug_mode = True

        check_attr(Opt, PositionMapOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_si(self, *args):
        class Opt(SiOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, SiOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_si_summary(self, *args):
        class Opt(SiOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            plot_summary = True

        check_attr(Opt, SiOptions)
        Opt().main([])

    class _LowerBoundOpt(PositionLowerBoundOptions):  # allow parallel computing serializing
        exp_date = '210315'
        animal_id = 'YW006'
        plane_index = 0
        neuron_id = 0
        session = 'light'
        debug_mode = False  # for csv aggregation

    @unittest.skip('Non DEBUG mode, might overwrite local current dataset')
    @patch('matplotlib.pyplot.show')
    def test_slb(self, *args):
        check_attr(self._LowerBoundOpt, PositionLowerBoundOptions)
        self._LowerBoundOpt().main([])

    def test_sparsity(self):
        class Opt(SparsityOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, SparsityOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_place_field(self, *args):
        class Opt(PlaceFieldsOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, PlaceFieldsOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_place_field_summary(self, *args):
        class Opt(PlaceFieldsOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            used_session = 'light'
            plot_summary = True

        check_attr(Opt, PlaceFieldsOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_align_map(self, *args):
        class Opt(AlignPeakMapOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, AlignPeakMapOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_trial_corr(self, *args):
        class Opt(TrialCorrOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, TrialCorrOptions)
        Opt().main([])

    def test_position_ev(self):
        class Opt(EVOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, EVOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_population_corr_mtx(self, *args):
        class Opt(PopulationMTXOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            pre_selection = True
            pc_selection = 'slb'
            x_cond = 'light-odd'
            y_cond = 'light-even'
            debug_mode = True
            reuse_output = True
            signal_type = 'spks'

        check_attr(Opt, PopulationMTXOptions)
        Opt().main([])


# ====================== #
# Batch Statistic Module #
# ====================== #


class TestSpatialBatchModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_alignment_map_batch(self, *args):
        class Opt(SISortAlignPersistenceAgg):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            with_top = 500

        check_attr(Opt, SISortAlignPersistenceAgg)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_position_bin_batch(self, *args):
        class Opt(PositionBinPersistenceAgg):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            plane_index = '0,0'
            session = 'light'
            used_session = 'light'
            debug_mode = True

        check_attr(Opt, PositionBinPersistenceAgg)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
