import unittest
import warnings
from unittest.mock import patch

from rscvp.selection.main_motion_drift import MotionDriftOptions
from rscvp.selection.main_neuropil_error import NeuropilErrOptions
from rscvp.selection.main_trial_reliability import TrialReliabilityOptions
from .util import check_attr


class TestSelectionModule(unittest.TestCase):
    """Test in ``selection`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_neuropil_error(self, *args):
        class Opt(NeuropilErrOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            debug_mode = True

        check_attr(Opt, NeuropilErrOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_trial_reliability(self, *args):
        class Opt(TrialReliabilityOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            session = 'light'
            debug_mode = True

        check_attr(Opt, TrialReliabilityOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_motion_drifting(self, *args):
        class Opt(MotionDriftOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            debug_mode = True

        check_attr(Opt, MotionDriftOptions)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
