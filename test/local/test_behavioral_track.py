import unittest
import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt

from argclz import AbstractParser
from rscvp.behavioral.main_tactile_batch import TactileBatchOptions
from rscvp.behavioral.main_tactile_summary import TactileSummaryOptions
from rscvp.track.main_lick_score import LickScoreOptions
from rscvp.track.main_licking_cmp import LickingCmpOptions
from rscvp.track.main_licking_prob import LickProbOptions
from rscvp.track.main_pupil_track import PupilTrackOptions
from rscvp.util.cli import StimpyOptions
from stimpyp import RiglogData
from .util import check_attr


class TestBehaviorTrackModule(unittest.TestCase):
    """Test in ``Behavioral`` and ``track`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    # ================= #
    # Behavioral Module #
    # ================= #

    @patch('matplotlib.pyplot.show')
    def test_trial_time(self, *arg):
        class Opt(AbstractParser, StimpyOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            source_version = 'stimpy-bit'

            def post_parsing(self):
                self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

            def run(self):
                rig = self.load_riglog_data()
                self.plot(rig)

            def plot(self, rig: RiglogData):
                plt.plot(rig.position_event.time, rig.position_event.value)
                for t in rig.lap_event.time:
                    plt.axvline(t, color='r', ls='--')
                plt.show()

        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_individual_summary(self, *args):
        class Opt(TactileSummaryOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True

        check_attr(Opt, TactileSummaryOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_peri_reward_vel(self, *args):
        class Opt(TactileBatchOptions):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            dispatch_plot = 'peri_reward_vel'
            debug_mode = True

        check_attr(Opt, TactileBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_vel_as_position(self, *args):
        class Opt(TactileBatchOptions):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            dispatch_plot = 'vel_as_position'
            debug_mode = True

        check_attr(Opt, TactileBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_peri_reward_lick(self, *args):
        class Opt(TactileBatchOptions):
            exp_date = '210315,210401'
            animal_id = 'YW006,YW006'
            dispatch_plot = 'peri_reward_lick'
            lick_thres = '80,80'
            debug_mode = True

        check_attr(Opt, TactileBatchOptions)
        Opt().main([])

    # =================== #
    # Track Module (Lick) #
    # =================== #

    @patch('matplotlib.pyplot.show')
    def test_lick_score(self, *args):
        class Opt(LickScoreOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            lick_thres = 70

        check_attr(Opt, LickScoreOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_lick_cmp_offset(self, *args):
        class Opt(LickingCmpOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            lick_thres = 70
            alignment = True

        check_attr(Opt, LickingCmpOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_lick_cmp_non_offset(self, *args):
        class Opt(LickingCmpOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            lick_thres = 70
            alignment = False

        check_attr(Opt, LickingCmpOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_lick_prob(self, *args):
        class Opt(LickProbOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            lick_thres = 70
            dispatch_plot = 'position'

        check_attr(Opt, LickProbOptions)
        Opt().main([])

    # ==================== #
    # Track Module (Pupil) #
    # ==================== #

    @patch('matplotlib.pyplot.show')
    def test_pupil_track_location(self, *args):
        class Opt(PupilTrackOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            dispatch_plot = 'location'

        check_attr(Opt, PupilTrackOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_pupil_track_movement(self, *args):
        class Opt(PupilTrackOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            dispatch_plot = 'movement'

        check_attr(Opt, PupilTrackOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_pupil_track_area(self, *args):
        class Opt(PupilTrackOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            debug_mode = True
            lick_thres = 70
            dispatch_plot = 'area'

        check_attr(Opt, PupilTrackOptions)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
