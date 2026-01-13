from neuralib.util.utils import ensure_dir
from rscvp.atlas.main_roi_atlas import RoiAtlasOptions
from rscvp.util.io import RSCVP_CACHE_DIRECTORY
from rscvp.util.util_demo import run_demo


class ExampleRun(RoiAtlasOptions):
    SOURCE_ROOT = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset' / 'analysis' / 'hist'
    animal = 'YW043'
    debug_mode = True


if __name__ == '__main__':
    run_demo(ExampleRun, clean_cached=False)
