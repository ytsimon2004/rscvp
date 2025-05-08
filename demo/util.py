import zipfile
from pathlib import Path

from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.utils import ensure_dir
from rscvp.util.io import RSCVP_CACHE_DIRECTORY

__all__ = ['mkdir_test_dataset',
           'clean_cache_dataset']

TOKEN = ...  # contact author since paper is not published yet


def mkdir_test_dataset() -> Path:
    output_dir = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset'

    if output_dir.exists():
        return output_dir
    else:
        data_url = 'https://zenodo.org/records/15363378/files/rscvp_dataset.zip?token='
        data_url += TOKEN

        zip_stream = download_with_tqdm(data_url)
        with zipfile.ZipFile(zip_stream) as zip_file:
            zip_file.extractall(output_dir)

    return output_dir


def clean_cache_dataset():
    output_dir = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset'
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
