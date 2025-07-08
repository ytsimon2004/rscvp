import platform
from pathlib import Path
from typing import TypedDict, ClassVar, Literal, cast

from neuralib.typing import PathLike
from neuralib.util.utils import ensure_dir

__all__ = [
    'RSCVP_CACHE_DIRECTORY',
    'CONFIG_FOLDER',
    'HISTOLOGY_HOME_ROOT',
    #
    'DISK_TYPE',
    'DATA_SRC_TYPE',
    #
    'DataSourceRoot',
    'IOConfig',
    'get_io_config',
    'islocal'
]

CACHE_DIRECTORY = Path.home() / '.cache'
RSCVP_CACHE_DIRECTORY = CACHE_DIRECTORY / 'rscvp'
CONFIG_FOLDER = RSCVP_CACHE_DIRECTORY / 'config'
HISTOLOGY_HOME_ROOT = RSCVP_CACHE_DIRECTORY / 'histology'

#
DISK_TYPE = Literal[
    'default',
    'local', 'WD-2T', 'BigDATA',
    'bkrunch', 'bkrunch2',
    'bkrunch-linux', 'bkrunch-linux-bigdata'
]

DATA_SRC_TYPE = Literal['stimpy', 'suite2p', 'behavior', 'histology', 'track', 'cache']


class DataSourceRoot(TypedDict, total=False):
    stimpy: PathLike
    """Stimpy data presentation"""
    physiology: PathLike
    """Physiology data. i.e., calcium image, electrophysiology, etc."""
    histology: PathLike
    """Histology data. i.e., brain slice, probe track, tracing, etc."""


class IOConfig:
    DEFAULT_GSPREAD_AUTH: ClassVar[Path] = CONFIG_FOLDER / 'gspread' / 'service_account.json'

    source_root: DataSourceRoot
    gspread_auth: Path

    def __init__(self,
                 source_root: DataSourceRoot,
                 gspread_auth: Path | None = None):
        self.source_root = cast(
            DataSourceRoot,
            {src: Path(p) for src, p in source_root.items()}
        )

        self._gspread_auth = gspread_auth or self.DEFAULT_GSPREAD_AUTH
        self._phy_animal_dir = self.source_root['physiology'] / '%'

    @property
    def phy_base_dir(self) -> Path:
        """base physiology directory"""
        return self.source_root['physiology']

    @property
    def phy_animal_dir(self) -> Path:
        """physiology animal directory"""
        return self._phy_animal_dir

    @phy_animal_dir.setter
    def phy_animal_dir(self, path: PathLike):
        self._phy_animal_dir = Path(path)

    @property
    def stimpy(self) -> Path:
        """stimpy presentation directory"""
        return self.source_root['stimpy']

    @property
    def suite2p(self) -> Path:
        """suite2p analysis directory under physiology animal directory"""
        return ensure_dir(self.phy_animal_dir / 'suite2p')

    @property
    def cache(self) -> Path:
        """cached data directory under physiology animal directory"""
        return ensure_dir(self.phy_animal_dir / 'cache')

    @property
    def track(self) -> Path:
        """camera tracking data under physiology animal directory"""
        return ensure_dir(self.phy_animal_dir / 'track')

    @property
    def behavior(self) -> Path:
        """behavioral data under physiology animal directory"""
        return ensure_dir(self.phy_animal_dir / 'behavior')

    @property
    def statistic_dir(self) -> Path:
        """statistics directory under base physiology directory"""
        return ensure_dir(self.phy_base_dir / 'statistics')


DEFAULT_IO_CONFIG: dict[DISK_TYPE, IOConfig] = {
    'local': IOConfig(source_root=dict(
        stimpy='/Users/yuting/data/presentation',
        physiology='/Users/yuting/data/analysis/phys',
        histology='/Users/yuting/data/analysis/hist'
    )),

    # ============= #
    # Other Machine #
    # ============= #

    'bkrunch': IOConfig(source_root=dict(
        stimpy='j:/data/presentation',
        physiology='/Users/simon/code/Analysis/Analysis',
        histology='/Users/simon/code/Analysis/histology'
    )),

    'bkrunch2': IOConfig(source_root=dict(
        stimpy='e:/data/user/yu-ting/presentation',
        physiology='e:/data/user/yu-ting/analysis/phys',
        histology='e:/data/user/yu-ting/analysis/hist'
    )),

    'bkrunch-linux': IOConfig(source_root=dict(
        stimpy='/tmp_data/user/yuting/presentation',
        physiology='/tmp_data/user/yuting/analysis/phys',
        histology='/tmp_data/user/yuting/analysis/hist'
    )),

    # ========= #
    # HDD / SSD #
    # ========= #

    'WD-2T': IOConfig(source_root=dict(
        stimpy='/Volumes/WD-2T/physiology',
        physiology='/Volumes/WD-2T/physiology',
        histology='/Volumes/WD-2T/histology'
    )),

    # ============= #
    # Mounted Drive #
    # ============= #

    'BigDATA': IOConfig(source_root=dict(
        stimpy='/Volumes/BigDATA/data/user/yu-ting/presentation',
        physiology='/Volumes/BigDATA/data/user/yu-ting/analysis/phys',
        histology='/Volumes/BigDATA/data/user/yu-ting/analysis/hist'
    )),

    'default': IOConfig(source_root=dict(
        stimpy=RSCVP_CACHE_DIRECTORY / 'rscvp_dataset' / 'presentation',
        physiology=RSCVP_CACHE_DIRECTORY / 'rscvp_dataset' / 'analysis' / 'phys',
        histology=RSCVP_CACHE_DIRECTORY / 'rscvp_dataset' / 'analysis' / 'hist'
    )),

    'bkrunch-linux-bigdata': IOConfig(source_root=dict(
        stimpy='/mnt/bigdata/data/user/yu-ting/presentation',
        physiology='/mnt/bigdata/data/user/yu-ting/analysis/phys',
        histology='/mnt/bigdata/data/user/yu-ting/analysis/hist'
    ))

}


def get_io_config(config: dict[str, IOConfig] | None = None,
                  remote_disk: str | None = None,
                  mnt_prefix: str = '/Volumes',
                  force_use_default: bool = False) -> IOConfig:
    """
    Determines and retrieves the appropriate IO configuration for the current node and remote disk
    setup. This function evaluates the current machine's hostname and optionally, a specified remote
    disk to decide the corresponding `IOConfig` from the given or default configuration dictionary.

    :param config: A dictionary mapping from strings (node/disk identifiers) to `IOConfig` objects.
        If not provided, a predefined `DEFAULT_IO_CONFIG` dictionary is used.
        Defaults to None.
    :param remote_disk: The identifier of a remote disk to be checked. When provided, the function
        verifies that this disk is mounted under the specified `mnt_prefix` path. If absent or not
        specified, the function determines the configuration based only on the node. Defaults to None.
    :param mnt_prefix: A prefix path where a remote disk should be mounted. The allowed values are
        either '/mnt' or '/Volumes'. Defaults to '/Volumes'.
    :param force_use_default: A boolean flag to enforce the use of the default configuration, ignoring
        other criteria like node or remote disk. Defaults to False.
    :return: The `IOConfig` object corresponding to the current environment (node/disk combination).
    :raises RuntimeError: If `remote_disk` is specified but is not detected at the specified mount
        location under `mnt_prefix`.
    """
    if config is None:
        config = DEFAULT_IO_CONFIG

    if isinstance(remote_disk, str) and not (Path(mnt_prefix) / remote_disk).exists():
        raise RuntimeError(f'check remote disk connection: {remote_disk}')

    if force_use_default:
        return config['default']

    node = platform.node()
    match node, remote_disk:
        # local
        case ('Yu-Tings-MacBook-Pro.local', None):
            return config['local']
        case ('Simon-MacBook-Pro.local', None):
            return config['local']

        # mount
        case ('Yu-Tings-MacBook-Pro.local', 'WD-2T'):
            return config['WD-2T']
        case ('Yu-Tings-MacBook-Pro.local', 'BigDATA'):
            return config['BigDATA']

        # other
        case ('bkrunch', None):
            return config['bkrunch']
        case ('bkrunch2', None):
            return config['bkrunch2']
        case ('bkrunch-linux', None):
            return config['bkrunch-linux']
        case ('bkrunch-linux', 'bigdata'):
            return config['bkrunch-linux-bigdata']
        case _:
            return config['default']


def islocal() -> bool:
    """if running on current local machine"""
    return platform.node() == 'Yu-Tings-MacBook-Pro.local'
