import unittest
import warnings
from unittest.mock import patch

from neuralib.imglib.io import tif_to_gif, gif_show
from rscvp.retinotopic.cache_retinotopic import RetinotopicCacheBuilder
from rscvp.retinotopic.main_retinotopic_map import RetinotopicMapOptions
from .util import check_attr


class TestRetinotopicModule(unittest.TestCase):
    """Test in ``Retinotopic`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @unittest.skip('Calculated from raw tif sequence, filesize too large, thus need to be tested in workstation')
    def test_retinotopic_cache_pyvstim(self):
        class Opt(RetinotopicCacheBuilder):
            exp_date = '210302'
            animal_id = 'YW008'
            source_version = 'pyvstim'
            invalid_cache = True

        check_attr(Opt, RetinotopicCacheBuilder)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_retinotopic_result_pyvstim(self, *args):
        class Opt(RetinotopicMapOptions):
            exp_date = '210302'
            animal_id = 'YW008'
            source_version = 'pyvstim'
            debug_mode = True

        check_attr(Opt, RetinotopicMapOptions)
        Opt().main([])

    _play_gif = False

    @unittest.skipIf(_play_gif, reason='play .gif need manual test')
    def test_retinotopic_gif_pyvstim(self):
        class Opt(RetinotopicCacheBuilder):
            exp_date = '210302'
            animal_id = 'YW008'
            source_version = 'pyvstim'

            def __init__(self):
                self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        opt = Opt()
        tif_to_gif(opt.trial_averaged_tiff, opt.trial_averaged_gif)

        if self._play_gif:
            gif_show(opt.trial_averaged_gif)


if __name__ == '__main__':
    unittest.main()
