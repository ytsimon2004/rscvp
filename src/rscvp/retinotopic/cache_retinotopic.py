from pathlib import Path

import numpy as np
from rscvp.retinotopic.util import compute_wfield_trial_average, combine_cycles_within_trial
from rscvp.util.cli.cli_persistence import PersistenceRSPOptions
from rscvp.util.cli.cli_wfield import WFieldOptions
from tifffile import tifffile

from argclz import AbstractParser
from neuralib.imglib.io import tif_to_gif
from neuralib.persistence import persistence
from neuralib.util.logging import LOGGING_IO_LEVEL
from stimpyp import (
    STIMPY_SOURCE_VERSION,
    AbstractStimlog,
    RiglogData,
    StimpyProtocol,
    PyVlog,
    PyVProtocol,
    PyCamlog,
    LabCamlog
)

__all__ = ['RetinotopicCacheBuilder']


# =========== #
# Persistence #
# =========== #

@persistence.persistence_class
class RetinotopicCache:
    exp_date: str = persistence.field(validator=True, filename=True)
    animal: str = persistence.field(validator=True, filename=True)
    source_version: STIMPY_SOURCE_VERSION = persistence.field(validator=True, filename=True)

    #
    prot_name: str
    """protocol name"""
    trial_averaged_resp: np.ndarray
    """(S, H, W)"""
    stim_xy: np.ndarray
    """stimulation obj XY. (S, 2)"""
    n_trials: int
    """(TR,)"""
    n_cycles: list[int]
    """cycle foreach trials [(C,), ...]"""


class RetinotopicCacheBuilder(AbstractParser, WFieldOptions, PersistenceRSPOptions[RetinotopicCache]):
    """Build the cache for retinotopic analyzed results"""
    riglog: PyVlog | RiglogData
    stimlog: AbstractStimlog
    prot: PyVProtocol | StimpyProtocol
    camlog: LabCamlog | PyCamlog

    load_avi = False
    load_raw = True

    def post_parsing(self):
        self.setup_logger(Path(__file__).name)
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    def run(self):
        self.post_parsing()
        self.load_cache()

    def empty_cache(self) -> RetinotopicCache:
        return RetinotopicCache(
            exp_date=self.exp_date,
            animal=self.animal_id,
            source_version=self.source_version
        )

    def compute_cache(self, cache: RetinotopicCache) -> RetinotopicCache:
        self.riglog = self.load_riglog_data()
        self.stimlog = self.riglog.get_stimlog()
        self.prot = self.riglog.get_protocol()
        self.camlog = self.load_camlog()

        cache.prot_name = self.prot.name
        cache.trial_averaged_resp = self.get_trial_averaged_sequences()
        cache.stim_xy = None  # TODO
        cache.n_trials = self.stimlog.profile_dataframe.shape[0]

        cache.n_cycles = self.stimlog.n_cycles

        return cache

    # ======= #
    # Compute #
    # ======= #

    @property
    def trial_averaged_tiff(self) -> Path:
        return self.retinotopic_directory / 'trial_averaged_resp.tiff'

    @property
    def trial_averaged_gif(self) -> Path:
        return self.trial_averaged_tiff.with_suffix('.gif')

    def get_trial_averaged_sequences(self, force_compute: bool = False) -> np.ndarray:
        if not self.trial_averaged_tiff.exists() or force_compute:
            trial_avg = self._compute_trial_averaged_pyv()
            tifffile.imwrite(self.trial_averaged_tiff, trial_avg)
            self.logger.log(LOGGING_IO_LEVEL, f'SAVE trial averaged resp in {self.trial_averaged_tiff}')
            tif_to_gif(self.trial_averaged_tiff, self.trial_averaged_gif)
            self.logger.log(LOGGING_IO_LEVEL, f'SAVE trial averaged resp in {self.trial_averaged_gif}')

        return tifffile.imread(self.trial_averaged_tiff)

    def _compute_trial_averaged_pyv(self) -> np.ndarray:
        trial_avg = compute_wfield_trial_average(
            self.load_wfield_result().sequences,
            self.camlog.get_camera_time(self.riglog),
            self.stimlog
        )

        # combine cycle
        if self.stimlog.n_cycles != [1]:
            trial_avg = combine_cycles_within_trial(trial_avg, self.stimlog.n_cycles)

        return trial_avg


if __name__ == '__main__':
    RetinotopicCacheBuilder().main()
