import logging
from pathlib import Path
from typing import ClassVar, Iterable

from argclz import argument, str_tuple_type
from neuralib.atlas.typing import Channel, Source, HEMISPHERE_TYPE, Area, PLANE_TYPE
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.io import get_io_config

__all__ = ['HistOptions']


class HistOptions:
    GROUP_HIST: ClassVar = 'Histology Options'
    """group histology options"""

    SOURCE_ROOT: Path = argument(
        '-S', '--src-path',
        metavar='PATH',
        group=GROUP_HIST,
        default=get_io_config().source_root['histology'],
        help='histology folder source root path'
    )

    animal: str | tuple[str, ...] = argument(
        '-A', '--id', '--animal-id', '--mouse-id',
        metavar='NAME',
        required=True,
        group=GROUP_HIST,
        help='animal (mouse) ID',
    )

    glass_id: int | None = argument(
        '--gid', '--glass-id',
        group=GROUP_HIST,
        help='glass slide id, which might corresponding to column number of 48 wells whole brain section collect',
    )

    slice_id: int | None = argument(
        '--sid', '--slice-id',
        group=GROUP_HIST,
        help='slice id, which should be 1-6',
    )

    cut_plane: PLANE_TYPE = argument(
        '--plane-type', '-P',
        default='coronal',
        group=GROUP_HIST,
        help='cutting orientation',
    )

    channel: Channel = argument(
        '-C', '--channel',
        default='g',
        group=GROUP_HIST,
        help='image channel'
    )

    source: Source = argument(
        '--source',
        metavar='NAME',
        group=GROUP_HIST,
        help='tracing source. i.e., aRSC, pRSC...'
    )

    hemisphere: HEMISPHERE_TYPE = argument(
        '-H', '--hemi', '--view-hemi',
        default='both',
        group=GROUP_HIST,
        help='only looking at which hemisphere'
    )

    area: Area | None = argument(
        '--area',
        metavar='NAME,...',
        type=str_tuple_type,
        default=None,
        group=GROUP_HIST,
        help='brain region name for the histology'
    )

    debug_mode: bool = argument(
        '--debug',
        group=GROUP_HIST,
        help='enable debug mode for qt show figure'
    )

    @property
    def histology_cache_directory(self) -> Path:
        """cached directory for histology data processing"""
        ret = self.SOURCE_ROOT / 'cache'
        if not ret.exists():
            ret.mkdir(exist_ok=True)
        return ret

    def get_ccf_dir(self) -> AbstractCCFDir:
        """Got an instance of :class:`~rscvp.atlas.dir.AbstractCCFDir` initialized with the given
        ``SOURCE_ROOT``, ``animal``, ``cut_plane`` and ``hemisphere`` arguments.
        """
        return AbstractCCFDir(
            self.SOURCE_ROOT / f'{self.animal}',
            with_overlap_sources=True,
            plane_type=self.cut_plane,
            hemisphere_type=self.hemisphere
        )

    def foreach_ccf_dir(self) -> Iterable[AbstractCCFDir]:
        """Generate :class:`~rscvp.atlas.dir.AbstractCCFDir` instances for each animal."""
        animals = self.animal

        if isinstance(animals, str):
            animals = (animals,)

        if not isinstance(animals, tuple):
            raise TypeError(f'animals should be tuple type: {type(animals)}')

        try:
            for animal in animals:
                self.animal = animal
                yield AbstractCCFDir(
                    self.SOURCE_ROOT / f'{self.animal}',
                    with_overlap_sources=True,
                    plane_type=self.cut_plane,
                    hemisphere_type=self.hemisphere
                )
        finally:
            # set back for tuple type after generator loop
            self.animal = animals

    # ============= #
    # Logging Usage #
    # ============= #

    _logging_level: str | int = logging.DEBUG
    logger: logging.Logger | None = None

    def setup_logger(self, caller_name: str | None = None):
        from neuralib.util.logging import setup_clogger
        self.logger = setup_clogger(level=self._logging_level, caller_name=caller_name)
