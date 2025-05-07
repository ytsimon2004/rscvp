import math
from pathlib import Path
from typing import NamedTuple, Literal, Optional

import numpy as np
import polars as pl
from PIL import Image, ImageChops
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli.cli_hist import HistOptions
from tqdm import tqdm

from argclz import AbstractParser, argument, as_argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.ccf.matrix import load_transform_matrix
from neuralib.io import csv_header
from neuralib.plot import plot_figure
from neuralib.util.unstable import unstable
from neuralib.util.verbose import printdf


# TODO add log information for the 2dccf pipeline/repo
# TODO directly add csv/record in 2dccf matlab processor
class SliceRotation(NamedTuple):
    name: str
    mse: np.ndarray
    step: float

    @property
    def rotate(self) -> int:
        return np.argmin(self.mse) * self.step

    def plot(self):
        with plot_figure(None) as ax:
            ax.plot(self.mse)
            ax.set(xlabel='degree', ylabel='mse')


TOOL_TYPE = Literal[
    'trans-eval',
    'rename',
    'proc-eval',
    'apply-rotation',
    'revert-hemisphere',
    'concat-parsed'
]


@unstable()
class CCFPipelineFactory(AbstractParser, HistOptions, Dispatch):
    DESCRIPTION = 'General tools for 2dccf registration pipeline'

    dispatch_tool: TOOL_TYPE = argument(
        '-T', '--tool',
        required=True,
        help='which tool'
    )

    animal = as_argument(HistOptions.animal).with_options(required=False)

    # rename
    folder_path: Path = argument('--path', default=None, help='directly specify folder path')
    folder_layer: Literal['resize', 'proc', 'trans'] = argument('--layer', default='resize',
                                                                help='folder layer for rename')
    rename_suffix: Literal['.mat', '.tif', 'resize.tif'] = argument('--suffix', default='.mat',
                                                                    help='suffix for rename')
    pattern: str = argument('--pattern', default='resize', help='filename pattern under opt.directory')
    replace_pattern: str = argument('--sub', default='resize_overlap', help='pattern of filename replace to')

    # apply-rotation
    rotate_all: bool = argument('--rotate-all', help='rotate all, otherwise, specify gid, sid, hemi...')
    rotate_self: bool = argument('--rotate-self', help='rotate rg channel based on the its proc-eval, '
                                                       'otherwise apply in other channel. i.e., overlap channel')

    def run(self):
        if self.animal is not None:
            ccf_dir = self.get_ccf_dir()
        else:
            ccf_dir = None

        self.invoke_command(self.dispatch_tool, ccf_dir)

    @dispatch('trans-eval')
    def create_transformation_dataframe(self, ccf_dir: AbstractCCFDir) -> None:
        """Create dataframe about transformation information for multiple slices"""
        d = ccf_dir.transformed_folder

        ret = []
        for f in tqdm(d.glob('*.mat'), desc='load transformation matrix', unit='matrices'):
            ret.append(load_transform_matrix(f, self.cut_plane).matrix_info())

        dataframe = pl.concat(ret)
        printdf(dataframe)

        dataframe.write_csv(d / 'matrix_info.csv')

    @dispatch('rename')
    def rename_transform_matrix(self, ccf_dir: Optional[AbstractCCFDir]) -> None:
        """Rename transformation matrix.
         **Use case: for applying same transformation to overlap channel"""
        if self.folder_path is None:
            if self.folder_layer == 'resize':
                d = ccf_dir.resize_overlap_folder
            elif self.folder_layer == 'proc':
                d = ccf_dir.processed_folder_overlap
            elif self.folder_layer == 'trans':
                d = ccf_dir.transformed_folder_overlap
            else:
                raise ValueError('')
        else:
            d = self.folder_path

        for f in d.glob(f'*{self.rename_suffix}'):
            print(f'{f=}')
            s = f.name.replace(self.pattern, self.replace_pattern)
            pp = f.with_name(s)
            f.rename(pp)

    @dispatch('proc-eval')
    def get_processed_rotation(self, ccf_dir: AbstractCCFDir,
                               step: float = 0.75) -> None:
        """Use mse calculate pixel-wise diff.
        NOTE that this method assume there is no contrast changes in processed image
        TODO better/quicker way

        :param ccf_dir:
        :param step: default value followed the 2dccf.archive `SliceFlipper.m func: SliceScrollFcn`

        """

        with csv_header(ccf_dir.processed_folder / 'processed_log.csv',
                        ['slice_name', 'rotate']) as csv:

            for proc in ccf_dir.processed_folder.glob('*processed.tif'):
                p = get_slice_name(proc)
                for ori in ccf_dir.resize_folder.glob('*resize.tif'):
                    o = get_slice_name(ori)

                    if p == o:
                        mse = np.zeros(int(360 / step))
                        i_ori = Image.open(proc)
                        i_proc = Image.open(ori)

                        deg_range = tqdm(np.arange(0, 360, step), desc=f'{p} rotation eval', unit='deg')
                        for i, d in enumerate(deg_range):
                            r = i_ori.rotate(d)
                            mse[i] = _calc_rms_diff(i_proc, r)

                        rot = SliceRotation(p, mse, step).rotate
                        csv(p, rot)

    @dispatch('apply-rotation')
    def apply_rotation(self, ccf_dir: AbstractCCFDir) -> None:
        """
        ** Use case: Adapt the rotate parameter (degree) for one channel to another (rg registered -> overlap)
        :param ccf_dir:
        :return:
        """
        import cv2
        from scipy.ndimage import rotate

        rot = ccf_dir.processed_folder / 'processed_log.csv'
        df = pl.read_csv(rot)

        if self.rotate_self:
            f = ccf_dir.resize_folder
            out = ccf_dir.processed_folder
        else:
            f = ccf_dir.resize_overlap_folder
            out = ccf_dir.processed_folder_overlap

        if self.rotate_all:
            images = tqdm(f.glob('*tif'), desc='apply rotation', unit='images')
        else:
            # noinspection PyTypeChecker
            img = ccf_dir.glob(self.glass_id,
                               self.slice_id,
                               'resize' if self.rotate_self else 'resize_overlap',
                               hemisphere=self.hemi_prefix)
            images = tqdm([img], desc='apply rotation', unit='images')

        for it in images:
            name = get_slice_name(it)

            try:
                degree = df.filter(pl.col('slice_name') == name).select('rotate').item()
            except ValueError:
                raise RuntimeError(f'{name} not found in {rot}')

            raw = cv2.imread(str(it))
            rot = rotate(raw, 360 - degree, reshape=False)

            _out = str(out / f'{it.stem}_processed.tif')
            cv2.imwrite(_out, rot)

    @dispatch('revert-hemisphere')
    def revert_hemisphere(self, ccf_dir: AbstractCCFDir) -> None:
        """revert the ML coordinates from the raw csv (contra, ipsi hemisphere)
        ** Use case: wrong hemisphere during sagittal slice registration
        """
        for file in ccf_dir.labelled_roi_folder.glob('*.csv'):
            df = pl.read_csv(file)
            df = df.with_columns(pl.col('ML_location') * -1)
            df.write_csv(file)

    @dispatch('concat-parsed')
    def concat_parse_csv(self, ccf_dir: AbstractCCFDir) -> None:
        """** Use case: Concat two csv from different hemispheres (sagittal slice registration)"""
        files = list(ccf_dir.parsed_data_folder.glob('*csv'))
        if len(files) != 2:
            raise RuntimeError('')

        concat = pl.concat([pl.read_csv(file) for file in files])
        concat.write_csv(ccf_dir.parsed_data_folder / 'parsed_csv_merge.csv')


def _calc_rms_diff(x, y):
    """Calculates the root mean square error (RSME) between two images"""
    errors = np.asarray(ImageChops.difference(x, y)) / 255
    return math.sqrt(np.mean(np.square(errors)))


def get_slice_name(fpath: Path) -> str:
    n = fpath.name
    idx = n.index('_resize')
    return n[:idx]


if __name__ == '__main__':
    CCFPipelineFactory().main()
