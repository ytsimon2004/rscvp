import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from rscvp.model.bayes_decoding.main_decode_analysis import DecodeAnalysisOptions
from rscvp.selection.main_cls_summary import ClsCellTypeOptions
from rscvp.util.cli.cli_sql_view import DatabaseViewOptions
from rscvp.util.io import get_io_config
from rscvp.visual.main_polar import VisualPolarOptions
from rscvp.visual.main_sftf_pref import VisualSFTFPrefOptions
from .util import check_attr


class TestSQLDatabaseInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        file = Path(__file__).parents[2] / 'database' / '_test_rscvp.db'
        file.unlink(missing_ok=True)

    def test_import(self):
        class Opt(DatabaseViewOptions):
            _debug_mode = True
            action = 'import'
            args = [str(get_io_config().source_root['physiology'])]

        Opt().main([])

    def test_diagram(self):
        class Opt(DatabaseViewOptions):
            _debug_mode = True
            action = 'diagram'

        Opt().main([])


class TestSQLPopulate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @patch('matplotlib.pyplot.show')
    def test_generic_db_populate(self, *args):
        class Opt(ClsCellTypeOptions):
            exp_date = '210401'
            animal_id = 'YW006'
            plane_index = 0
            session = 'light'
            used_session = 'light'
            debug_mode = True
            db_commit = True
            db_debug_mode = True

        check_attr(Opt, ClsCellTypeOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_decode_db_populate(self, *args):
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
            cache_version = 2
            debug_mode = True
            db_commit = True
            db_debug_mode = True

        check_attr(Opt, DecodeAnalysisOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_dir_db_populate(self, *args):
        class Opt(VisualPolarOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            summary = True
            db_debug_mode = True
            db_commit = True

        check_attr(Opt, VisualPolarOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_sftf_db_frac_populate(self, *args):
        class Opt(VisualSFTFPrefOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            db_debug_mode = True
            db_commit = True
            summary_type = 'fraction'

        check_attr(Opt, VisualSFTFPrefOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_sftf_db_dff_populate(self, *args):
        class Opt(VisualSFTFPrefOptions):
            exp_date = '210315'
            animal_id = 'YW006'
            plane_index = 0
            used_session = 'light'
            db_debug_mode = True
            db_commit = True
            summary_type = 'dff'

        check_attr(Opt, VisualSFTFPrefOptions)
        Opt().main([])


class TestSQLView(unittest.TestCase):

    def test_generic_db_view(self):
        class Opt(DatabaseViewOptions):
            _debug_mode = True
            action = 'generic'
            show_all = True

        Opt().main([])

    def test_decode_db_view(self):
        class Opt(DatabaseViewOptions):
            _debug_mode = True
            action = 'decode'
            show_all = True

        Opt().main([])

    def test_sftf_dir_db_view(self):
        class Opt(DatabaseViewOptions):
            _debug_mode = True
            action = 'visual'
            show_all = True

        Opt().main([])


if __name__ == '__main__':
    unittest.main()
