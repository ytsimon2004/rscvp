from neuralib.util.utils import ensure_dir
from rscvp.atlas.main_roi_top_view import RoiTopViewOptions
from rscvp.util.io import RSCVP_CACHE_DIRECTORY
from rscvp.util.util_demo import run_demo


class ExampleRun(RoiTopViewOptions):
    SOURCE_ROOT = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset' / 'analysis' / 'hist'
    animal = ('YW043',)
    area_family = 'ISOCORTEX'
    legend_number_limit = 20


if __name__ == '__main__':
    run_demo(ExampleRun, clean_cached=False)
