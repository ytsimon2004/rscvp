from pathlib import Path
from typing import get_args, Literal

import numpy as np
import pandas as pd

from neuralib.atlas.ccf.matrix import load_transform_matrix
from neuralib.atlas.data import load_bg_structure_tree
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA
from neuralib.imglib.array import image_array
from neuralib.util.unstable import unstable


@unstable()
class ROIMapper:

    def __init__(self,
                 path: Path,
                 ref_path: Path,
                 roi_channel: Literal['red', 'green'],
                 plane: PLANE_TYPE):
        """

        :param path: transformation folder
        :param ref_path
        :param roi_channel
        :param plane
        """

        self.path = path
        self.ref_path = ref_path
        self.roi_channel = roi_channel
        self.plane = plane

        self.roi_annotation: dict = dict(avIndex=[], name=[], acronym=[])

        self._bregma = ALLEN_CCF_10um_BREGMA
        self._atlas_res = 0.01  # mm

        # cache
        self.__annotation_volume = None

    @property
    def data(self) -> list[Path]:  # TODO sorted?
        return list(self.path.glob('*transform_data.mat'))

    @property
    def images(self) -> list[Path]:
        return list(self.path.glob('*transformed.tif'))

    @property
    def annotation_volume(self) -> np.ndarray:
        """plane orientation"""
        if self.__annotation_volume is None:
            av = np.load(self.ref_path / 'annotation_volume_10um_by_index.npy')
            if self.plane == 'coronal':
                self.__annotation_volume = av
            elif self.plane == 'sagittal':
                x, y, z = av.shape
                self.__annotation_volume = av.reshape((z, y, x))
            elif self.plane == 'transverse':
                x, y, z = av.shape
                self.__annotation_volume = av.reshape((y, z, x))
            else:
                raise ValueError(f'{self.plane} unknown')

        return self.__annotation_volume

    @property
    def structure_tree(self) -> pd.DataFrame:
        return load_bg_structure_tree().to_pandas()

    @property
    def save_folder(self) -> Path:
        p = self.path / 'labelled_region'
        if not p.exists():
            p.mkdir(exist_ok=True, parents=True)
        return p

    def generate_roi_annotation(self,
                                rois: np.ndarray,
                                slice_angle: np.ndarray,
                                slice_num: int) -> pd.DataFrame:
        """
        annotate each roi (certain channel) based on allen map

        :param rois: binarized image array of roi selection
        :param slice_angle:
        :param slice_num:
        :return:
        """
        _, *ref_size = self.annotation_volume.shape
        if rois.shape != tuple(ref_size):
            raise RuntimeError('roi image is not the right size')

        pixel_row, pixel_col = np.where(rois > 0)

        rois_loc = np.zeros((len(pixel_row), 3))

        offset_map = self.get_offset_map(slice_angle, ref_size)
        bregma = self._bregma
        res = self._atlas_res

        for i, (pr, pc) in enumerate(zip(pixel_row, pixel_col)):
            offset = offset_map[pc, pr]

            if self.plane == 'coronal':
                ap = -(slice_num - bregma[0] + offset) * res
                dv = (pr - bregma[1]) * res
                ml = (pc - bregma[2]) * res
            elif self.plane == 'sagittal':
                ap = -(pc - bregma[0]) * res
                dv = (pr - bregma[1]) * res
                ml = -(slice_num - bregma[2] + offset) * res
            elif self.plane == 'transverse':
                ap = -(pc - bregma[0]) * res
                dv = -(slice_num - bregma[1] + offset) * res
                ml = (pr - bregma[2]) * res
            else:
                raise ValueError(f'unknown {self.plane}')

            rois_loc[i, :] = (ap, dv, ml)

            # finally, find the annotation, name, and acronym of the current ROI pixel
            annot = self.annotation_volume[int(slice_num + offset), pr, pc]
            self.roi_annotation['avIndex'].append(annot)
            self.roi_annotation['name'].append(self.structure_tree['safe_name'][annot])
            self.roi_annotation['acronym'].append(self.structure_tree['acronym'][annot])

        self.roi_annotation['AP_location'] = rois_loc[:, 0]
        self.roi_annotation['DV_location'] = rois_loc[:, 1]
        self.roi_annotation['ML_location'] = rois_loc[:, 2]

        return pd.DataFrame.from_dict(self.roi_annotation)

    @staticmethod
    def get_offset_map(slice_angle: np.ndarray, ref_size: tuple[int, int]) -> np.ndarray:
        """
        Generate offset map (for third dimension of a tilted slice)
        from allenCCF/Browsing Functions/get_offset_map.m

        ..seealso::

            :func:`matlab.img.get_offset_map.m`

        :param slice_angle: (2,)
        :param ref_size: (height, width)
        :return:

        """
        angle_ap = int(slice_angle[0])
        angle_ml = int(slice_angle[1])

        # TODO slightly difference with matlab code, do further comparison
        ap_frame = np.round(np.linspace(-angle_ap, angle_ap, ref_size[0])).astype(int)
        ml_frame = np.round(np.linspace(-angle_ml, angle_ml, ref_size[1])).astype(int)

        return ap_frame[None, :] + ml_frame[:, None]

    def run_analysis(self):
        ret = []
        for i, img in enumerate(self.images):
            print(f'{i + 1} of {len(self.images)}')
            data = self.data[i]

            # load transformed slice image and transformed matrix
            slice_img = image_array(img)  # transformed
            tran_mtx = load_transform_matrix(data, self.plane)

            # get the position within the atlas data of the transformed slice
            slice_num = tran_mtx.slice_index
            slice_angle = tran_mtx.delta_xy

            if self.roi_channel == 'red':
                rois = slice_img.local_maxima('red')
            elif self.roi_channel == 'green':
                rois = slice_img.local_maxima('green')
            else:
                raise NotImplementedError('')

            ret.append(self.generate_roi_annotation(rois, slice_angle, slice_num))

        df = pd.concat(ret).reset_index()
        df.to_csv(self.save_folder / f'{self.roi_channel}_roitabel.csv')


def main(args: list[str] = None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-P', '--path', type=Path, required=True)
    ap.add_argument('-R', '--ref', type=Path, required=True)
    ap.add_argument('-C', '--channel', required=True)
    ap.add_argument('--plane', choices=get_args(PLANE_TYPE), default='coronal')

    opt = ap.parse_args(args)

    ROIMapper(opt.path, opt.ref, opt.channel, opt.plane).run_analysis()


if __name__ == '__main__':
    main()
