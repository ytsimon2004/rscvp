import unittest
import warnings
from unittest.mock import patch

from rscvp.model.rastermap.main_rastermap_2p import RunRasterMap2POptions
from rscvp.model.rastermap.main_rastermap_wfield import RunRasterMapWfieldOptions
from rscvp.model.rastermap.rastermap_2p_cache import RasterMap2PCacheBuilder
from rscvp.model.rastermap.rastermap_wfield_cache import RasterMapWfieldCacheBuilder
from .util import check_attr


class TestRastermapModule(unittest.TestCase):
    """Test in rastermap model locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    @unittest.skip('Calculated from raw tif sequence, filesize too large, thus need to be tested in workstation')
    def test_rastermap_wfield_cache(self):
        class Opt(RasterMapWfieldCacheBuilder):
            exp_date = '210302'
            animal_id = 'YW008'

        check_attr(Opt, RasterMapWfieldCacheBuilder)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_rastermap_wfield(self, *args):
        class Opt(RunRasterMapWfieldOptions):
            exp_date = '210302'
            animal_id = 'YW008'
            debug_mode = True

        check_attr(Opt, RunRasterMapWfieldOptions)
        Opt().main([])

    def test_rastermap_2p_cache(self):
        class Opt(RasterMap2PCacheBuilder):
            exp_date = '210401'
            animal_id = 'YW006'
            plane_index = 0
            invalid_cache = True
            neuron_bins = 30

        check_attr(Opt, RasterMap2PCacheBuilder)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_rastermap_2p(self, *args):
        class Opt(RunRasterMap2POptions):
            exp_date = '210401'
            animal_id = 'YW006'
            plane_index = 0
            neuron_bins = 30
            dispatch_analysis = 'sorting'
            debug_mode = True

        check_attr(Opt, RunRasterMap2POptions)
        Opt().main([])


if __name__ == '__main__':
    unittest.main()
