import platform
from pathlib import Path
from typing import TypedDict, ClassVar, Literal

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
DISK_TYPE = Literal['local', 'WD-2T', 'BigDATA', 'bkrunch', 'bkrunch2', 'bkrunch-linux']
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
        self.source_root = source_root
        self._converter()

        #
        self._gspread_auth = gspread_auth or self.DEFAULT_GSPREAD_AUTH
        self._phy_animal_dir = self.source_root['physiology'] / '%'

    def _converter(self):
        for src, p in self.source_root.items():
            self.source_root[src] = Path(p)

    @property
    def phy_animal_dir(self) -> Path:
        """Daily usage"""
        return self._phy_animal_dir

    @phy_animal_dir.setter
    def phy_animal_dir(self, path: PathLike):
        self._phy_animal_dir = Path(path)

    @property
    def stimpy(self) -> Path:
        return self.source_root['stimpy']

    @property
    def histology(self) -> Path:
        return self.source_root['histology']

    @property
    def suite2p(self) -> Path:
        return ensure_dir(self.phy_animal_dir / 'suite2p')

    @property
    def cache(self) -> Path:
        return ensure_dir(self.phy_animal_dir / 'cache')

    @property
    def track(self) -> Path:
        return ensure_dir(self.phy_animal_dir / 'track')

    @property
    def behavior(self) -> Path:
        return ensure_dir(self.phy_animal_dir / 'behavior')

    @property
    def output_dir(self) -> Path:
        return self.source_root['physiology']

    @property
    def statistic_dir(self) -> Path:
        return ensure_dir(self.output_dir / 'statistics')


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
    ))

}


def get_io_config(config: dict[str, IOConfig] | None = None,
                  remote_disk: str | None = None,
                  mnt_prefix: Literal['/mnt', '/Volumes'] = '/Volumes') -> IOConfig:
    if config is None:
        config = DEFAULT_IO_CONFIG

    if isinstance(remote_disk, str) and not (Path(mnt_prefix) / remote_disk).exists():
        raise RuntimeError(f'check remote disk connection: {remote_disk}')

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
        case _:
            raise NotImplementedError('')


def islocal() -> bool:
    return platform.node() == 'Yu-Tings-MacBook-Pro.local'
