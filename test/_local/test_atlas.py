import unittest
import warnings
from unittest.mock import patch

from argclz import AbstractParser
# from rscvp.atlas._ccf.main_slice_transform import plot_slice_view_transform
from rscvp.atlas.main_expr_range import RoiExprRangeBatchOptions
from rscvp.atlas.main_roi_atlas import RoiAtlasOptions
from rscvp.atlas.main_roi_quant import RoiQuantOptions
from rscvp.atlas.main_roi_quant_batch import RoiQuantBatchOptions
from rscvp.atlas.main_roi_query import RoiQueryOptions
from rscvp.atlas.main_roi_query_batch import RoiQueryBatchOptions
from rscvp.atlas.main_roi_top_view import RoiTopViewOptions
from rscvp.atlas.main_roi_view import RoisViewOptions
from rscvp.atlas.main_ternary import TernaryPercOptions
from rscvp.util.cli import HistOptions
from ._util import check_attr


class TestAtlasModule(unittest.TestCase):
    """Test in ``atlas`` module locally"""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=ResourceWarning)

    # =============== #
    # RoiQuantOptions #
    # =============== #

    @patch('matplotlib.pyplot.show')
    def test_roi_quant_cat(self, *args):
        class Opt(RoiQuantOptions):
            animal = 'YW043'
            dispatch_plot = 'cat'
            debug_mode = True
            merge_level = 2
            top_area = 40
            roi_norm = 'volume'

        check_attr(Opt, RoiQuantOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_quant_pie(self, *args):
        class Opt(RoiQuantOptions):
            animal = 'YW043'
            dispatch_plot = 'pie'
            debug_mode = True

        check_attr(Opt, RoiQuantOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_quant_bar(self, *args):
        class Opt(RoiQuantOptions):
            animal = 'YW043'
            dispatch_plot = 'bar'
            debug_mode = True
            merge_level = 3
            top_area = 50
            roi_norm = 'none'

        check_attr(Opt, RoiQuantOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_quant_venn(self, *args):
        class Opt(RoiQuantOptions):
            animal = 'YW043'
            dispatch_plot = 'venn'
            debug_mode = True
            merge_level = 2
            top_area = 50

        check_attr(Opt, RoiQuantOptions)
        Opt().main([])

    @unittest.skip('latex plt issue')
    @patch('matplotlib.pyplot.show')
    def test_roi_quant_bias(self, *args):
        class Opt(RoiQuantOptions):
            animal = 'YW043'
            dispatch_plot = 'bias'
            debug_mode = True
            merge_level = 2
            top_area = 50

        check_attr(Opt, RoiQuantOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_quant_parallel(self, *args):
        class Opt(RoiQuantOptions):
            animal = 'YW043'
            dispatch_plot = 'parallel'
            debug_mode = True
            merge_level = 2
            top_area = 50
            area = 'VIS'

        check_attr(Opt, RoiQuantOptions)
        Opt().main([])

    # ==================== #
    # RoiQuantBatchOptions #
    # ==================== #

    @patch('matplotlib.pyplot.show')
    def test_batch_region_x(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'region_x'
            debug_mode = True
            merge_level = 2
            area = 'VIS'

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_x_flatten(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'region_x'
            debug_mode = True
            merge_level = 2
            area = 'VIS'
            plot_args = ['True']  # flatten

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_overall(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'region_overall'
            debug_mode = True
            merge_level = 2
            top_area = 40
            area = 'VIS'

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_expr_scatter(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW063', 'YW064')
            dispatch_plot = 'expr_scatter'
            debug_mode = True
            merge_level = 2
            top_area = 40

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_bias_index(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'bias_index'
            debug_mode = True
            merge_level = 2
            top_area = 70

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_heatmap(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'heatmap'
            debug_mode = True
            merge_level = 2
            top_area = 70
            roi_norm = 'channel'

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_hemi_diff(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'hemi_diff'
            debug_mode = True
            merge_level = 2
            top_area = 50
            roi_norm = 'channel'

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_batch_region_family_stacked(self, *args):
        class Opt(RoiQuantBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            dispatch_plot = 'family_stacked'
            debug_mode = True
            plot_args = ['True']  # foreach_animal

        check_attr(Opt, RoiQuantBatchOptions)
        Opt().main([])

    # =============== #
    # RoiQueryOptions #
    # =============== #

    @patch('matplotlib.pyplot.show')
    def test_roi_query_bar(self, *args):
        class Opt(RoiQueryOptions):
            animal = 'YW043'
            graph = 'bar'
            debug_mode = True
            area = ('VIS',)

        check_attr(Opt, RoiQueryOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_query_dot(self, *args):
        class Opt(RoiQueryOptions):
            animal = 'YW043'
            graph = 'dot'
            debug_mode = True
            area = ('VIS',)

        check_attr(Opt, RoiQueryOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_query_stacked(self, *args):
        class Opt(RoiQueryOptions):
            animal = 'YW043'
            graph = 'stacked'
            debug_mode = True
            area = ('VIS',)

        check_attr(Opt, RoiQueryOptions)
        Opt().main([])

    # ==================== #
    # RoiQueryBatchOptions #
    # ==================== #

    @patch('matplotlib.pyplot.show')
    def test_batch_query(self, *args):
        class Opt(RoiQueryBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            area = 'VIS'
            debug_mode = True

        check_attr(Opt, RoiQueryBatchOptions)
        Opt().main([])

    # ====== #
    # Others #
    # ====== #

    @patch('matplotlib.pyplot.show')
    def test_top_view(self, *args):
        class Opt(RoiTopViewOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')
            area_family = 'ISOCORTEX'
            legend_number_limit = 20

        check_attr(Opt, RoiTopViewOptions)
        Opt().main([])

    @patch('plotly.graph_objects.Figure.show')
    def test_ternary_plot(self, *args):
        class Opt(TernaryPercOptions):
            animal = 'YW043'
            roi_norm = 'volume'
            debug_mode = True

        check_attr(Opt, TernaryPercOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_expr_range(self, *args):
        class Opt(RoiExprRangeBatchOptions):
            animal = ('YW043', 'YW051', 'YW063', 'YW064')

        check_attr(Opt, RoiExprRangeBatchOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_view_3d(self, *args):
        class Opt(RoisViewOptions):
            animal = 'YW043'
            dispatch_plot = '3d'
            area = 'VIS'
            debug_mode = True

        check_attr(Opt, RoisViewOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_roi_view_hist(self, *args):
        class Opt(RoisViewOptions):
            animal = 'YW043'
            dispatch_plot = 'histogram'
            area = 'VIS'
            debug_mode = True

        check_attr(Opt, RoisViewOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    @unittest.skip('walk pipeline')
    def test_roi_atlas(self, *args):
        class Opt(RoiAtlasOptions):
            animal = 'YW043'
            debug_mode = True

        check_attr(Opt, RoiAtlasOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    @unittest.skip('walk pipeline')
    def test_roi_atlas_affine(self, *args):
        class Opt(RoiAtlasOptions):
            animal = 'YW043'
            glass_id = 4
            slice_id = 1
            debug_mode = True
            affine_transform = True

        check_attr(Opt, RoiAtlasOptions)
        Opt().main([])

    @patch('matplotlib.pyplot.show')
    def test_slice_transform(self, *args):
        class Test(AbstractParser, HistOptions):
            animal = 'YW051'

            def run(self):
                ccf_dir = self.get_ccf_dir()
                image = ccf_dir.zproj_folder / 'YW051_7_1_merge.jpeg'
                mtx = ccf_dir.transformed_folder / 'test_7_1.mat'
                ann = ccf_dir.transformed_folder / 'YW051_7_1_resize_processed_transform_data.mat'
                plot_slice_view_transform(image, mtx, ann, self.cut_plane)

        Test().run()


if __name__ == '__main__':
    unittest.main()
