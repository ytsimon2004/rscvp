from pathlib import Path

import zipfile
from io import BytesIO
from typing import Type, Literal

from argclz import AbstractParser
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.utils import ensure_dir
from rscvp.util.io import RSCVP_CACHE_DIRECTORY

__all__ = [
    'DEMO_DATA_SOURCE',
    'mkdir_demo_dataset',
    'clean_cache_dataset',
    'run_demo'
]

CACHED_DEMO_DATASET = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset'
DEMO_DATA_SOURCE = Literal['zenodo', 'figshare']


def mkdir_demo_dataset(source: DEMO_DATA_SOURCE = 'zenodo',
                       force_download: bool = False,
                       aria2: bool = True) -> Path:
    """
    Creates the directory for the test dataset by downloading from zenodo.


    :param source: Source of the dataset
    :param force_download: Boolean flag indicating whether to forcibly download
        the dataset even if it already exists in the cache. Defaults to False.
    :param aria2: Boolean flag indicating whether to use `aria2` for downloading the dataset. Defaults to True.
    :return: Path to the directory where the dataset is stored.
    """
    if CACHED_DEMO_DATASET.exists() and not force_download:
        return CACHED_DEMO_DATASET

    match source:
        case 'zenodo':
            token = 'eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImYxZjA1ZDIyLTczMjgtNGQ4NS1iZTE2LWE2NTIzM2JiZjNmNCIsImRhdGEiOnt9LCJyYW5kb20iOiI4YWZlNTUxNjEwNDNhNjg2NWQxMzU2NzUwNjQ4YzQ5ZiJ9.O53R_ljkVh2ybl3J4bo1ZI7V42Y1M5c04nIpNTvKQp1NHPWqd232dRtKlk3_R3ZBk-_4mR3MB06fi40myyzqVQ'
            data_url = f"https://zenodo.org/api/records/17639283/files/rscvp_dataset.zip/content?token={token}"
        case 'figshare':
            private_link = '88802ef91b1f519f9075'
            data_url = f"https://figshare.com/ndownloader/files/60981004?private_link={private_link}"
        case _:
            raise ValueError(f"Unknown source: {source}")

    zip_path = Path(RSCVP_CACHE_DIRECTORY) / "rscvp_dataset.zip"

    if aria2:
        try:
            import subprocess
            print(f'Using aria2c for fast download data from {source}, takes few minutes ...')
            # fewer connections for figshare to avoid 502 errors
            connections = "2" if source == 'figshare' else "16"

            aria2c_cmd = [
                "aria2c",
                "-x", connections,
                "-s", connections,
                "-k", "1M",
                "-o", str(zip_path.name),
                "-d", str(RSCVP_CACHE_DIRECTORY),
            ]

            # zenodo requires User-Agent header
            if source == 'zenodo':
                aria2c_cmd.extend(
                    ["--user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"])

            aria2c_cmd.append(data_url)

            result = subprocess.run(
                aria2c_cmd,
                check=False,  # check manually
            )
            if result.returncode != 0:
                print(f'aria2c failed with code {result.returncode}, falling back to Python downloader...')
                aria2 = False
        except (FileNotFoundError, Exception) as e:
            print(f'aria2c error: {e}, falling back to Python downloader...')
            aria2 = False

    if not aria2:
        print(f"Downloading data from {source} ...")
        # zenodo requires User-Agent header for restricted records
        if source == 'zenodo':
            import requests
            from tqdm import tqdm
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            resp = requests.get(data_url, headers=headers, stream=True)
            resp.raise_for_status()
            file_size = int(resp.headers.get('content-length', 0))
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc='Downloading from Zenodo...')

            with open(zip_path, 'wb') as f:
                for data in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(data)
                    progress_bar.update(len(data))
            progress_bar.close()
        else:
            zip_stream: BytesIO = download_with_tqdm(data_url)
            with open(zip_path, 'wb') as f:
                f.write(zip_stream.getbuffer())

    print('Extracting dataset...')
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(RSCVP_CACHE_DIRECTORY)

    zip_path.unlink()

    return CACHED_DEMO_DATASET


def run_demo(cls: Type[AbstractParser],
             source: DEMO_DATA_SOURCE = 'zenodo', *,
             force_download: bool = False,
             clean_cached: bool = False):
    """
    Run demo for a class, download dataset from zenodo, and run main function.

    :param cls: Running class, must be subclass of AbstractParser
    :param source: source of dataset, default is figshare
    :param force_download: force re-download dataset from zenodo
    :param clean_cached: clean cached dataset after running demo
    """
    mkdir_demo_dataset(source=source, force_download=force_download)
    cls().main([])

    if clean_cached:
        clean_cache_dataset()


def clean_cache_dataset():
    output_dir = CACHED_DEMO_DATASET
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
