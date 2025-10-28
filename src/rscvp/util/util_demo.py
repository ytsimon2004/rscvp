import zipfile
from io import BytesIO
from pathlib import Path
from typing import Type

from argclz import AbstractParser
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.utils import ensure_dir
from rscvp.util.io import RSCVP_CACHE_DIRECTORY

__all__ = [
    'mkdir_test_dataset',
    'clean_cache_dataset',
    'run_demo'
]

CACHED_DEMO_DATASET = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset'


def mkdir_test_dataset(token: str,
                       force_download: bool = False,
                       aria2: bool = True) -> Path:
    """
    Creates the directory for the test dataset by downloading from zenodo.


    :param token: Authentication token to access the dataset.
    :param force_download: Boolean flag indicating whether to forcibly download
        the dataset even if it already exists in the cache. Defaults to False.
    :param aria2: Boolean flag indicating whether to use `aria2` for downloading the dataset. Defaults to True.
    :return: Path to the directory where the dataset is stored.
    """
    if CACHED_DEMO_DATASET.exists() and not force_download:
        return CACHED_DEMO_DATASET

    data_url = f"https://zenodo.org/records/17466243/files/rscvp_dataset.zip?token={token}"
    zip_path = Path(RSCVP_CACHE_DIRECTORY) / "rscvp_dataset.zip"

    if aria2:
        try:
            import subprocess
            print('Using aria2c for fast download...')
            subprocess.run(
                [
                    "aria2c", "-x", "16", "-s", "16", "-k", "1M",
                    "-o", str(zip_path.name),
                    "-d", str(RSCVP_CACHE_DIRECTORY),
                    data_url
                ],
                check=True
            )
        except BaseException:
            print('aria2c not found, falling back to Python downloader...')
            aria2 = False

    if not aria2:
        print("Downloading with Python (requests + tqdm)...")
        zip_stream: BytesIO = download_with_tqdm(data_url)
        with open(zip_path, "wb") as f:
            f.write(zip_stream.getbuffer())

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(RSCVP_CACHE_DIRECTORY)

    zip_path.unlink()

    return CACHED_DEMO_DATASET


def clean_cache_dataset():
    output_dir = CACHED_DEMO_DATASET
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)


def run_demo(cls: Type[AbstractParser],
             token: str, *,
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
