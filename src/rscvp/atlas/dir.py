import abc
import shutil
from pathlib import Path
from typing import get_args, Final, Literal, Iterable

from neuralib.atlas.ccf.matrix import load_transform_matrix
from neuralib.atlas.typing import HEMISPHERE_TYPE, PLANE_TYPE
from neuralib.typing import PathLike
from neuralib.util.utils import joinn, uglob, ensure_dir
from neuralib.util.verbose import fprint

__all__ = ['AbstractCCFDir']

CHANNEL_SUFFIX = Literal['r', 'g', 'b', 'merge', 'overlap']
CCF_GLOB_TYPE = Literal[
    'zproj',
    'roi',
    'roi_cpose',
    'resize',
    'resize_overlap',
    'processed',
    'transformation_matrix',
    'transformation_img',
    'transformation_img_overlap'
]


class AbstractCCFDir(metaclass=abc.ABCMeta):
    """
    ::

        ANIMAL_001/ (root)
            ├── raw/ (optional)
            ├── zproj/
            │    └── ANIMAL_001_g*_s*_{channel}.tif
            ├── roi/
            │    └── ANIMAL_001_g*_s*_{channel}.roi
            ├── roi_cpose/
            │    └── ANIMAL_001_g*_s*_{channel}.roi
            ├── resize/ (src for the allenccf, if sagittal slice, could be resize_contra and resize_ipsi)
            │    ├── ANIMAL_001_g*_s*_resize.tif
            │    └── processed/
            │           ├── ANIMAL_001_g*_s*_resize_processed.tif
            │           └── transformations/
            │                 ├── ANIMAL_001_g*_s*_resize_processed_transformed.tif
            │                 ├── ANIMAL_001_g*_s*_resize_processed_transform_data.mat
            │                 └── labelled_regions/
            │                       ├── {*channel}_roitable.csv
            │                       └── parsed_data /
            │                             └── parsed_csv_merge.csv
            ├── resize_*_overlap/ (optional, same structure as **resize**, for dual channels labeling)
            │
            └── output_files/ (for generate output fig)

    """

    def __new__(
            cls,
            root: PathLike,
            with_overlap_sources: bool = True,
            plane_type: PLANE_TYPE = 'coronal',
            hemisphere_type: HEMISPHERE_TYPE | None = None,
            auto_mkdir: bool = True,
    ):

        if plane_type == 'coronal':
            if with_overlap_sources:
                return object.__new__(CoronalCCFOverlapDir)
            else:
                return object.__new__(CoronalCCFDir)

        elif plane_type == 'sagittal':

            if hemisphere_type is None or hemisphere_type not in get_args(HEMISPHERE_TYPE):
                raise ValueError(f'invalid hemisphere_type for sagittal dir: {hemisphere_type}')

            #
            if with_overlap_sources:
                return object.__new__(SagittalCCFOverlapDir)
            else:
                return object.__new__(SagittalCCFDir)

        else:
            raise ValueError(f'invalid plane type: {plane_type}')

    def __init__(
            self,
            root: PathLike,
            with_overlap_sources: bool = True,
            plane_type: PLANE_TYPE = 'coronal',
            hemisphere_type: HEMISPHERE_TYPE | None = None,
            auto_mkdir: bool = True,
    ):
        r"""

        :param root: Root path (i.e., \*/ANIMAL_001)
        :param with_overlap_sources: If there is overlap channel labeling (for dir making).
        :param plane_type: {'coronal', 'sagittal', 'transverse'}
        :param hemisphere_type: {'ipsi', 'contra', 'both'}
        :param auto_mkdir: If auto make folder structure for the pipeline.
        """
        self.root: Final[Path] = root
        self.with_overlap_sources = with_overlap_sources
        self.plane_type: Final[PLANE_TYPE] = plane_type
        self.hemisphere: HEMISPHERE_TYPE | None = hemisphere_type if plane_type == 'sagittal' else None

        if auto_mkdir:
            self._init_folder_structure()

    def __len__(self):
        """number of slices"""
        return len(list(self.resize_folder.glob('*.tif')))

    def __iter__(self):
        return self.resize_folder.glob('*.tif')

    @abc.abstractmethod
    def _init_folder_structure(self) -> None:
        pass

    slice_name: str = None  # assign when glob

    @abc.abstractmethod
    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path:
        """

        :param glass_id: Glass slide number
        :param slice_id: Slice sequencing under a glass slide
            (i.e., zigzag from upper left -> bottom left ->-> bottom right)
        :param glob_type:
        :param hemisphere:
        :param channel:
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def resize_folder(self) -> Path:
        pass

    # ============================ #
    # Default Dir for the pipeline #
    # ============================ #

    @property
    def default_iter_dir(self) -> Iterable[Path]:
        return [
            self.raw_folder,
            self.resize_folder,
            self.roi_folder,
            self.cpose_roi_folder,
            self.zproj_folder,
            self.processed_folder,
            self.transformed_folder,
            self.labelled_roi_folder,
            self.parsed_data_folder,
            self.output_folder
        ]

    @property
    def animal(self) -> str:
        return self.root.name

    @property
    def raw_folder(self) -> Path:
        return self.root / 'raw'

    @property
    def roi_folder(self) -> Path:
        return self.root / 'roi'

    @property
    def cpose_roi_folder(self) -> Path:
        return self.root / 'roi_cpose'

    @property
    def zproj_folder(self) -> Path:
        return self.root / 'zproj'

    # =============================== #
    # Dual Channel Overlap (Optional) #
    # =============================== #

    @property
    def resize_overlap_folder(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    @property
    def processed_folder_overlap(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    @property
    def transformed_folder_overlap(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    @property
    def labelled_roi_folder_overlap(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    # ========================================== #
    # CCF folder (MATLAB pipeline auto-generate) #
    # ========================================== #

    @property
    def processed_folder(self) -> Path:
        return self.resize_folder / 'processed'

    @property
    def transformed_folder(self) -> Path:
        return self.processed_folder / 'transformations'

    @property
    def labelled_roi_folder(self) -> Path:
        return self.transformed_folder / 'labelled_regions'

    @property
    def parsed_data_folder(self) -> Path:
        return self.labelled_roi_folder / 'parsed_data'

    @property
    def parse_csv(self) -> Path:
        return self.parsed_data_folder / 'parsed_roi.csv'

    # ======= #
    # Outputs #
    # ======= #

    @property
    def output_folder(self) -> Path:
        return self.root / 'output_files'

    def figure_output(self, *suffix, sep='_') -> Path:
        ret = self.output_folder / joinn(sep, *suffix)
        return ret.with_suffix('.pdf')

    def csv_output(self, *suffix, sep='_') -> Path:
        ret = self.output_folder / joinn(sep, *suffix)
        return ret.with_suffix('.csv')

    @property
    def roi_atlas_output(self) -> Path:
        return ensure_dir(self.output_folder / 'roiatlas')

    # ========= #
    # File Glob #
    # ========= #

    def get_slice_id(self, glass_id: int,
                     slice_id: int,
                     hemisphere: Literal['i', 'c'] | None = None,
                     channel: CHANNEL_SUFFIX | None = None) -> str:

        ret = f'{self.animal}_{glass_id}_{slice_id}'
        if hemisphere is not None:
            ret += f'_{hemisphere}'
        if channel is not None:
            ret += f'_{channel}'
        return ret

    def get_transformation_matrix(self, glass_id: int,
                                  slice_id: int,
                                  plane_type: PLANE_TYPE) -> 'CCFTransMatrix':
        return load_transform_matrix(
            self.glob(glass_id, slice_id, 'transformation_matrix'),
            plane_type
        )


class CoronalCCFDir(AbstractCCFDir):
    """Base folder structure for 2dccf pipeline"""

    def _init_folder_structure(self):
        for d in self.default_iter_dir:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    @property
    def resize_folder(self) -> Path:
        return self.root / 'resize'

    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE, *,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path | None:

        if glob_type in ('roi', 'roi_cpose', 'zproj'):
            self.slice_name = name = self.get_slice_id(glass_id, slice_id, hemisphere=hemisphere, channel=channel)
        else:
            self.slice_name = name = self.get_slice_id(glass_id, slice_id, hemisphere=hemisphere)

        #
        if glob_type == 'roi':
            return uglob(self.roi_folder, f'{name}*.roi')
        elif glob_type == 'roi_cpose':
            return uglob(self.cpose_roi_folder, f'{name}*cpose.roi')
        elif glob_type == 'zproj':
            return uglob(self.zproj_folder, f'{name}.*')
        elif glob_type == 'resize':
            return uglob(self.resize_folder, f'{name}_resize.*')
        elif glob_type == 'transformation_matrix':
            return uglob(self.transformed_folder, f'{name}*.mat')
        elif glob_type == 'transformation_img':
            return uglob(self.transformed_folder, f'{name}*.tif')
        else:
            return


class CoronalCCFOverlapDir(CoronalCCFDir):
    """ 2dccf Folder structure for multiple sources overlap labeling
    For example: Dual tracing with different fluorescence protein, and tend to see the overlap channel counts"""

    def _init_folder_structure(self):
        super()._init_folder_structure()

        iter_overlap = (
            self.resize_overlap_folder,
            self.processed_folder,
            self.transformed_folder_overlap
        )
        for d in iter_overlap:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    @property
    def resize_overlap_folder(self) -> Path:
        """for double labeling channel
        since maximal 2 channels for roi detection,
        then need extra folder use pseudo-color
        """
        return self.root / 'resize_overlap'

    @property
    def processed_folder_overlap(self) -> Path:
        return self.resize_overlap_folder / 'processed'

    @property
    def transformed_folder_overlap(self) -> Path:
        return self.processed_folder_overlap / 'transformations'

    @property
    def labelled_roi_folder_overlap(self) -> Path:
        return self.transformed_folder_overlap / 'labelled_regions'

    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE, *,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path | None:

        ret = super().glob(glass_id, slice_id, glob_type, hemisphere=hemisphere, channel=channel)

        if ret is None:
            if glob_type == 'resize_overlap':
                return uglob(self.resize_overlap_folder, f'{self.slice_name}_resize_overlap.*')
            elif glob_type == 'transformation_img_overlap':
                return uglob(self.transformed_folder_overlap, f'{self.slice_name}*.tif')

        return ret


class SagittalCCFDir(AbstractCCFDir):

    def _init_folder_structure(self) -> None:
        for d in self.default_iter_dir:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    def glob(self, glass_id: int, slice_id: int, glob_type: CCF_GLOB_TYPE, hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path | None:

        if glob_type in ('roi', 'roi_cpose', 'zproj'):
            self.slice_name = name = self.get_slice_id(glass_id, slice_id, hemisphere=hemisphere, channel=channel)
        else:
            self.slice_name = name = self.get_slice_id(glass_id, slice_id, hemisphere=hemisphere)

            #
        if glob_type == 'roi':
            return uglob(self.roi_folder, f'{name}*.roi')
        elif glob_type == 'roi_cpose':
            return uglob(self.cpose_roi_folder, f'{name}*cpose.roi')
        elif glob_type == 'zproj':
            return uglob(self.zproj_folder, f'{name}.*')
        elif glob_type == 'resize':
            return uglob(self.zproj_folder, f'{name}_resize.*')
        elif glob_type == 'transformation_matrix':
            return uglob(self.transformed_folder, f'{name}*.mat')
        elif glob_type == 'transformation_img':
            return uglob(self.transformed_folder, f'{name}*.tif')
        else:
            return

    @property
    def resize_folder(self) -> Path:
        if self.hemisphere == 'ipsi':
            return self.root / 'resize_ipsi'
        elif self.hemisphere == 'contra':
            return self.root / 'resize_contra'
        elif self.hemisphere == 'both':
            return self.root / 'resize'
        else:
            raise ValueError('')


class SagittalCCFOverlapDir(SagittalCCFDir):
    def _init_folder_structure(self):
        super()._init_folder_structure()

        iter_overlap = (
            self.resize_overlap_folder,
            self.processed_folder,
            self.transformed_folder_overlap
        )
        for d in iter_overlap:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    @property
    def resize_overlap_folder(self) -> Path:
        """for double labeling channel
        since maximal 2 channels for roi detection,
        then need extra folder use pseudo-color
        """
        if self.hemisphere == 'ipsi':
            return self.root / 'resize_ipsi_overlap'
        elif self.hemisphere == 'contra':
            return self.root / 'resize_contra_overlap'
        elif self.hemisphere == 'both':
            return self.root / 'resize_overlap'
        else:
            raise ValueError('')

    @property
    def processed_folder_overlap(self) -> Path:
        return self.resize_overlap_folder / 'processed'

    @property
    def transformed_folder_overlap(self) -> Path:
        return self.processed_folder_overlap / 'transformations'

    @property
    def labelled_roi_folder_overlap(self) -> Path:
        return self.transformed_folder_overlap / 'labelled_regions'

    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE, *,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path | None:

        ret = super().glob(glass_id, slice_id, glob_type, hemisphere=hemisphere, channel=channel)

        if ret is None:
            if glob_type == 'resize_overlap':
                return uglob(self.resize_overlap_folder, f'{self.slice_name}_resize_overlap.*')
            elif glob_type == 'transformation_img_overlap':
                return uglob(self.transformed_folder_overlap, f'{self.slice_name}*.tif')

        return


# ============================== #
# Before init the AbstractCCFDir #
# ============================== #


def _concat_channel(ccf_dir: AbstractCCFDir, plane: PLANE_TYPE):
    """
    Find the csv data from `labelled_roi_folder`, if multiple files are found, concat to single df.
    `channel` & `source` columns are added to the dataframe.

    If sagittal slice, auto move ipsi/contra hemispheres dataset (`resize_ipsi`, `resize_contra`)
    to new `resize` directory

    :param ccf_dir: :class:`AbstractCCFDir`
    :param plane: ``PLANE_TYPE`` {'coronal', 'sagittal', 'transverse'}
    :return:
    """
    if plane == 'sagittal':
        _auto_sagittal_combine(ccf_dir)
    elif plane == 'coronal':
        _auto_coronal_combine(ccf_dir)


def _auto_overlap_copy(ccf: CoronalCCFOverlapDir | SagittalCCFOverlapDir) -> None:
    src = uglob(ccf.labelled_roi_folder_overlap, '*.csv')
    filename = f'{ccf.animal}_overlap_roitable'
    if ccf.plane_type == 'sagittal':
        filename += f'_{ccf.hemisphere}'

    dst = (ccf.labelled_roi_folder / filename).with_suffix('.csv')
    shutil.copy(src, dst)
    print(f'copy overlap file from {src} to {dst}')


def _auto_coronal_combine(ccf_dir: CoronalCCFDir | CoronalCCFOverlapDir):
    _auto_overlap_copy(ccf_dir)


def _auto_sagittal_combine(ccf_dir: SagittalCCFDir | SagittalCCFOverlapDir) -> None:
    """copy file from overlap dir to major fluorescence (channel) folder,
    then combine different hemisphere data"""

    old_args = ccf_dir.hemisphere

    def with_hemisphere_stem(ccf: SagittalCCFDir | SagittalCCFOverlapDir) -> list[Path]:
        ls = list(ccf.labelled_roi_folder.glob('*.csv'))
        for it in ls:
            if ccf.hemisphere not in it.name:
                new_path = it.with_stem(it.stem + f'_{ccf.hemisphere}')
                it.rename(new_path)

        return list(ccf.labelled_roi_folder.glob('*.csv'))  # new glob

    mv_list = []

    ccf_dir.hemisphere = 'ipsi'
    if isinstance(ccf_dir, SagittalCCFOverlapDir):
        _auto_overlap_copy(ccf_dir)
    ext = with_hemisphere_stem(ccf_dir)
    mv_list.extend(ext)

    #
    ccf_dir.hemisphere = 'contra'
    if isinstance(ccf_dir, SagittalCCFOverlapDir):
        _auto_overlap_copy(ccf_dir)
    ext = with_hemisphere_stem(ccf_dir)
    mv_list.extend(ext)

    #
    ccf_dir.hemisphere = 'both'  # as resize
    target = ccf_dir.labelled_roi_folder
    for file in mv_list:
        shutil.copy(file, target / file.name)
        print(f'copy file from {file} to {target / file.name}')

    ccf_dir.hemisphere = old_args  # assign back
