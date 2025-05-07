import abc
from pathlib import Path

import numpy as np
from bokeh.models import GlyphRenderer, ColumnDataSource
from bokeh.plotting import figure

from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import get_slice_view
from neuralib.dashboard import ViewComponent
from neuralib.imglib.array import ImageArrayWrapper
from rscvp.atlas.dir import AbstractCCFDir

__all__ = ['AbstractImgView']


class AbstractImgView(ViewComponent):
    data_img: ColumnDataSource
    render_img: GlyphRenderer

    def __init__(self):
        self.data_img = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.reference: np.ndarray | None = None

        self._width: float | None = None
        self._height: float | None = None

    @property
    def width(self) -> float | None:
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = value

    @property
    def height(self) -> float | None:
        return self._height

    @height.setter
    def height(self, value: float):
        self._height = value

    def plot(self, fig: figure, **kwargs):
        self.render_img = fig.image_rgba(
            image='image',
            x='x',
            y='y',
            dw='dw',
            dh='dh',
            source=self.data_img
        )

    @property
    @abc.abstractmethod
    def brain_image(self) -> np.ndarray:
        pass

    def load_file(self, file: Path) -> None:
        if file is not None:  # prevent cv2.error
            img = ImageArrayWrapper(file, alpha=True)
            self.reference = img.view_2d()

    def load_annotation_overlay(self, ccf: AbstractCCFDir,
                                glass_id: int,
                                slice_id: int,
                                image_file: Path,
                                plane_type: PLANE_TYPE) -> None:
        """
        :param ccf: for get the corresponding transformation matrix
        :param glass_id:
        :param slice_id:
        :param image_file: transformed image
        :param plane_type
        :return:
        """
        if image_file is not None:
            img = ImageArrayWrapper(image_file)

            idx = ccf.get_transformation_matrix(glass_id, slice_id, plane_type).slice_index
            annotation = get_slice_view('annotation', plane_type).reference[idx, :, :]
            annotation = ImageArrayWrapper(annotation, 'RGB').canny_filter(0, 30).flipud()

            self.reference = img.view_2d() + annotation

    def update(self, x: float = 0, y: float = 0):
        brain = self.brain_image

        if brain is None:
            self.data_img.data = dict(
                image=[], dw=[], dh=[], x=[], y=[]
            )
            return

        h, w = brain.shape
        if self._width is None:
            self._width = w
        if self._height is None:
            self._height = h

        self.data_img.data = dict(
            image=[brain], dw=[self.width], dh=[self.height], x=[x], y=[y]
        )
