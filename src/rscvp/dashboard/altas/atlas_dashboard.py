from typing import Optional, get_args

import numpy as np
from bokeh.model import Model
from bokeh.models import ColumnDataSource, GlyphRenderer, Slider, Select
from bokeh.palettes import Greys256
from bokeh.plotting import figure

from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import AbstractSliceView, get_slice_view, VIEW_TYPE
from neuralib.dashboard import View, ViewComponent, BokehServer
from neuralib.dashboard.tool import TimeoutUpdateValue
from neuralib.dashboard.util import MsgLog


class SliceView(ViewComponent):
    data_slice: ColumnDataSource
    render_slice: GlyphRenderer

    def __init__(self, view: AbstractSliceView, plane_type: PLANE_TYPE = 'coronal'):
        self.view = view
        self.data_slice = ColumnDataSource(data=dict(x=[], y=[], intensity=[]))
        self.plane_type = plane_type
        self.idx: Optional[int] = None

    def plot(self, fig: figure, **kwargs):
        self.render_slice = fig.image(
            image='intensity',
            x=0,
            y=0,
            dw='x',
            dh='y',
            source=self.data_slice,
            palette=Greys256,
            level='image'
        )

    def update(self):
        if self.plane_type == 'coronal':
            data = self.view.reference[self.idx, :, :]  # (DV, ML)
            intensity = np.flip(data, axis=0)  # invert y
        elif self.plane_type == 'sagittal':
            data = self.view.reference[:, :, self.idx]  # (AP, DV)
            intensity = np.flip(data.T, axis=0)
        elif self.plane_type == 'transverse':
            data = self.view.reference[:, self.idx, :]  # (AP, ML)
            intensity = np.flip(data, axis=0)
        else:
            raise ValueError('')

        self.data_slice.data = dict(x=[self.view.width], y=[self.view.height], intensity=[intensity])


class AtlasView(View):
    INSTANCE = None

    select_plane_type: Select
    slider_slice_idx: Slider

    #
    view_atlas: SliceView
    #
    figure_atlas: figure

    def __init__(self, source: VIEW_TYPE = 'reference'):
        AtlasView.INSTANCE = self

        self.source = source
        self.msg_log: MsgLog | None = None
        self.slider_idx_updater: TimeoutUpdateValue | None = None
        self.view: AbstractSliceView | None = None

    @property
    def title(self) -> str:
        return 'Atlas View'

    @property
    def plane_type(self) -> PLANE_TYPE | None:
        if len(self.select_plane_type.value) == 0:
            return None
        return self.select_plane_type.value

    @property
    def slice_idx(self) -> int:
        return self.slider_slice_idx.value

    def setup(self) -> Model:
        #
        self.msg_log = MsgLog(value='wait for selection')
        #
        plane_type = list(get_args(PLANE_TYPE))
        plane_type.insert(0, '')
        self.select_plane_type = Select(
            width=500,
            title='plane type',
            value='coronal',
            options=plane_type,
        )
        self.select_plane_type.on_change('value', self.on_select_plane_type)
        self.view = get_slice_view(self.source, self.select_plane_type.value)

        #
        slice_idx = np.arange(self.view.n_planes)
        self.slider_slice_idx = Slider(
            start=np.min(slice_idx),
            end=np.max(slice_idx),
            value=slice_idx[0],
            title='Slice index',
        )

        self.slider_slice_idx.on_change('value', self.on_select_slice_idx)
        self.slider_idx_updater = TimeoutUpdateValue(self.document, self._on_select_slice_idx, delay=500)

        #
        self.figure_atlas = figure(title=f'atlas_{self.plane_type}', width=1140, height=800)
        self.view_atlas = SliceView(self.view, self.plane_type)
        self.view_atlas.plot(self.figure_atlas)

        from bokeh.layouts import column
        return column(
            self.select_plane_type,
            self.slider_slice_idx,
            self.msg_log.message_area,
            self.figure_atlas
        )

    def on_select_plane_type(self, attr: str, old: str, value: str):
        self.view = get_slice_view(self.source, self.select_plane_type.value)  # reinit
        self.view_atlas.plane_type = self.plane_type

        # slider update
        self.slider_slice_idx.value = 0
        self.slider_slice_idx.start = 0
        self.slider_slice_idx.end = self.view.n_planes - 1

        # figure update
        self.figure_atlas.title = f'atlas_{self.plane_type}'
        self.figure_atlas.width = self.view.width
        self.figure_atlas.height = self.view.height
        self.figure_atlas.x_range.update()
        self.figure_atlas.y_range.update()

    def on_select_slice_idx(self, attr: str, old: str, value: str):
        self.slider_idx_updater.update(None)

    def _on_select_slice_idx(self, ignored):
        self.view_atlas.idx = self.slice_idx

        value = self.view.plane_at(self.slice_idx).reference_value
        if self.plane_type == 'coronal':
            text = f'AP: {value} mm'
        elif self.plane_type == 'sagittal':
            text = f'ML: {value} mm'
        elif self.plane_type == 'transverse':
            text = f'DV: {value} mm'
        msg_log(text, reset=True)

        self.run_later(self.view_atlas.update)


def msg_log(*message: str, reset: bool = False):
    message = '\n'.join(message)
    print(message)
    AtlasView.INSTANCE.msg_log.on_message(message, reset)


def main():
    BokehServer(theme='caliber').start(AtlasView())


if __name__ == '__main__':
    main()
