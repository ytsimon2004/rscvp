import zipfile
from pathlib import Path
from typing import Type

from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.utils import ensure_dir
from rscvp.util.io import RSCVP_CACHE_DIRECTORY

from argclz import AbstractParser

__all__ = [
    'mkdir_test_dataset',
    'clean_cache_dataset',
    'run_demo'
]

CACHED_DEMO_DATASET = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset'


def mkdir_test_dataset(token: str, force_download: bool = False) -> Path:
    if CACHED_DEMO_DATASET.exists() and not force_download:
        return CACHED_DEMO_DATASET
    else:
        data_url = 'https://zenodo.org/records/15363378/files/rscvp_dataset.zip?token='
        data_url += token

        zip_stream = download_with_tqdm(data_url)
        with zipfile.ZipFile(zip_stream) as zip_file:
            zip_file.extractall(RSCVP_CACHE_DIRECTORY)

    return CACHED_DEMO_DATASET


def clean_cache_dataset():
    output_dir = CACHED_DEMO_DATASET
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)


def run_demo(cls: Type[AbstractParser], token: str, *,
             force_download: bool = False,
             clean_cached: bool = False):
    """
    Run demo for a class, download dataset from zenodo, and run main function.

    :param cls: Running class, must be subclass of AbstractParser
    :param token: token for downloading dataset from zenodo
    :param force_download: force re-download dataset from zenodo
    :param clean_cached: clean cached dataset after running demo
    """
    mkdir_test_dataset(token=token, force_download=force_download)
    cls().main([])

    if clean_cached:
        clean_cache_dataset()
