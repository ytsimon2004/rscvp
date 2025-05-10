import unittest
import warnings
from unittest.mock import patch

from rscvp.model.glm.main_eval import BehavioralGLMOptions
from .util import check_attr


class TestGLMModule(unittest.TestCase):
    """Test in ``model.glm`` module and relevant statistic locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_glm_pos(self, *args):
        class Opt(BehavioralGLMOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            var_type = 'pos'
            session = 'light'
            signal_type = 'spks'
            debug_mode = True

        check_attr(Opt, BehavioralGLMOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_glm_speed(self, *args):
        class Opt(BehavioralGLMOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            var_type = 'speed'
            session = 'light'
            signal_type = 'spks'
            debug_mode = True

        check_attr(Opt, BehavioralGLMOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_glm_acc(self, *args):
        class Opt(BehavioralGLMOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 0
            var_type = 'acceleration'
            session = 'light'
            signal_type = 'spks'
            debug_mode = True

        check_attr(Opt, BehavioralGLMOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_glm_lick(self, *args):
        class Opt(BehavioralGLMOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            neuron_id = 14
            var_type = 'lick_rate'
            session = 'light'
            signal_type = 'spks'
            debug_mode = True
            lick_thres = 80

        check_attr(Opt, BehavioralGLMOptions)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
