from demo.util import mkdir_test_dataset
from neuralib.util.utils import ensure_dir
from rscvp.atlas.main_roi_atlas import RoiAtlasOptions
from rscvp.util.io import RSCVP_CACHE_DIRECTORY


class ExampleRun(RoiAtlasOptions):
    SOURCE_ROOT = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset' / 'analysis' / 'hist'
    animal = 'YW043'
    debug_mode = True


def main():
    mkdir_test_dataset()
    ExampleRun().main()
    # clean_cache_dataset()  # clean all if needed


if __name__ == '__main__':
    main()
