import unittest

import numpy as np
from numpy.testing import assert_array_equal

from argclz import AbstractParser
from rscvp.util.cli import StimpyOptions
from rscvp.util.util_trials import TrialSelection


class TestRiglogData(AbstractParser, StimpyOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0

    def __init__(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)


class TestTrialSelection(unittest.TestCase):
    trial: TrialSelection

    @classmethod
    def setUpClass(cls):
        rig = TestRiglogData().load_riglog_data()
        cls.trial = TrialSelection.from_rig(rig, session='light')
        cls.odd_trial = cls.trial.select_odd()
        cls.even_trial = cls.trial.select_even()

    def test_odd_even_invert(self):
        odd_trial = self.trial.select_odd()
        even_trial = self.trial.select_even()
        t0 = odd_trial.invert().selected_trials
        t1 = even_trial.selected_trials
        assert_array_equal(t0, t1)

    def test_select_range_invert(self):
        t1 = self.trial.select_range((1, 10))
        t2 = t1.invert()
        assert_array_equal(t1.selected_trials, np.arange(1, 10))

        exp = np.setdiff1d(self.trial.selected_trials, t1.selected_trials)
        assert_array_equal(t2.selected_trials, exp)

    def test_select_range_odd(self):
        t1 = self.trial.select_odd_in_range((0, 10))
        t2 = t1.invert()

        assert_array_equal(t1.selected_trials, np.arange(1, 9, 2))

        exp = np.setdiff1d(self.trial.selected_trials, t1.selected_trials)
        assert_array_equal(t2.selected_trials, exp)

    def test_fraction_selection(self):
        train, test = self.trial.select_fraction(0.8)
        self.assertEqual(len(test.selected_trials), int(0.2 * self.trial.selected_numbers))

    def test_kfold_selection(self):
        total_trials = set(self.trial._selected_trials)

        kfold_list = self.trial.kfold_cv()
        for k, (train, test) in enumerate(kfold_list):
            train_set = set(train.selected_trials)
            test_set = set(test.selected_trials)
            print(f'{k}fold:\ntrain:{train_set}\ntest:{test_set}')
            self.assertSetEqual(train_set.union(test_set), total_trials, 'not equal to total trials')

    def test_masking_data(self):
        n = len(self.trial.selected_trials)
        data = np.arange(n * 10).reshape(5, n, 2)  # (N, L, B)
        odd = self.trial.select_odd()
        res = odd.masking_trial_matrix(data)
        exp = data[:, 1::2, :]
        assert_array_equal(res, exp)

    def test_masking_time(self):
        pass


if __name__ == '__main__':
    unittest.main()
