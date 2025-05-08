from neuralib.atlas.brainrender.roi import RoiRenderCLI
from neuralib.util.utils import ensure_dir
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli import RSCRoiClassifierDataFrame
from rscvp.util.io import RSCVP_CACHE_DIRECTORY

SOURCE_ROOT = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset' / 'analysis' / 'hist' / 'YW043'
ccf = AbstractCCFDir(root=SOURCE_ROOT, hemisphere_type='both')
df = RSCRoiClassifierDataFrame(ccf, invalid_post_processing_cache=True)
df.post_processing(filter_injection=(df.config['area'], 'ipsi'), copy_overlap=True)
file = ccf.parse_csv


class ExampleRun(RoiRenderCLI):
    classifier_file = file
    regions = ("VISal", "VISam", "VISl", "VISli", "VISp", "VISpl", "VISpm", "VISpor")
    roi_region = "VIS"
    regions_alpha = 0.2
    radius = 20
    no_root = True
    camera_angle = "top"


def main():
    ExampleRun().main()


if __name__ == '__main__':
    main()
