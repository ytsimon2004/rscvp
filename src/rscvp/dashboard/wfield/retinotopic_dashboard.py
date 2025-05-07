from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import tifffile
from bokeh.layouts import row
from bokeh.model import Model
from bokeh.models import ColumnDataSource, GlyphRenderer, Slider
from bokeh.palettes import Greys256
from bokeh.plotting import figure

from neuralib.dashboard import View, ViewComponent, BokehServer
from neuralib.imaging.widefield import SequenceFFT
from neuralib.typing import PathLike


# TODO optimize performance/rendering
class ActivityView(ViewComponent):
    data_act: ColumnDataSource
    render_image: GlyphRenderer

    def __init__(self, act: np.ndarray,
                 nframe: int | None = None):
        self.act = act
        self.data_act = ColumnDataSource(data=dict(x=[], y=[], seq=[]))
        self.nframe = nframe

    def plot(self, fig: figure, **kwargs):
        self.render_image = fig.image(
            image='seq',
            x=0,
            y=0,
            dw='x',
            dh='y',
            source=self.data_act,
            level='image',
            palette=Greys256
        )

    @cached_property
    def width(self) -> int:
        return self.act.shape[2]

    @cached_property
    def height(self) -> int:
        return self.act.shape[1]

    def update(self):
        self.data_act.data = dict(x=[self.width], y=[self.height], seq=[self.act[self.nframe]])


class StimulusView(ViewComponent):
    data_stimulus: ColumnDataSource
    render_image: GlyphRenderer

    def __init__(self, stimulus: np.ndarray,
                 nframe: int | None = None):
        self.stimulus = stimulus
        self.data_stimulus = ColumnDataSource(data=dict(x=[], y=[], seq=[]))
        self.nframe = nframe

    def plot(self, fig: figure, **kwargs):
        self.render_image = fig.image(
            image='seq',
            x=0,
            y=0,
            dw='x',
            dh='y',
            source=self.data_stimulus,
            level='image',
            palette=Greys256
        )

    @cached_property
    def width(self) -> int:
        return self.stimulus.shape[2]

    @cached_property
    def height(self) -> int:
        return self.stimulus.shape[1]

    def update(self):
        self.data_stimulus.data = dict(x=[self.width], y=[self.height], seq=[self.stimulus[self.nframe]])


class FFTMap(ViewComponent):
    data_image: ColumnDataSource
    render_image: GlyphRenderer

    def __init__(self, phase_map: np.ndarray | None = None):
        self.phase_map = phase_map
        self.data_image = ColumnDataSource(data=dict(x=[], y=[], map=[]))

    def plot(self, fig: figure, **kwargs):
        self.render_image = fig.image_rgba(
            image='map',
            x=0,
            y=0,
            dw='x',
            dh='y',
            source=self.data_image,
            level='image'
        )

    def update(self):
        if (image := self.phase_map) is not None:
            self.update_image(image)

    def update_image(self, image: np.ndarray):
        self.phase_map = image
        if image is not None:
            y, x = image.shape[0], image.shape[1]
            self.data_image.data = dict(x=[x], y=[y], map=[image])


class FFTView(View):
    slider_sequence: Slider

    slider_hsv_value: Slider
    slider_hsv_hue: Slider

    view_activity: ActivityView
    view_stimulus: StimulusView
    view_map = FFTMap

    fig_activity: figure
    fig_stimulus: figure
    fig_intensity: figure
    fig_phase: figure
    fig_map: figure

    def __init__(self, seq_path: PathLike, stimulus_path: PathLike):
        """

        :param seq_path: tiff file for trial_averaged seq(mov)
        """
        self.seq_path = Path(seq_path)
        self.stimulus_path = Path(stimulus_path)

        # cache
        self._fft: SequenceFFT | None = None

    @cached_property
    def sequences(self) -> np.ndarray:
        return tifffile.imread(self.seq_path)

    @cached_property
    def stimulus(self) -> np.ndarray:
        stimulus = tifffile.imread(self.stimulus_path)
        # if stimulus.shape[0] != 3:  # RGBA
        #     weights = np.array([0.2989, 0.5870, 0.1140, 0.0])  # Ignore alpha
        #     stimulus = np.dot(stimulus, weights)
        return stimulus

    @cached_property
    def width(self) -> int:
        return self.sequences.shape[2]

    @cached_property
    def height(self) -> int:
        return self.sequences.shape[1]

    @cached_property
    def stimulus_width(self) -> int:
        return self.stimulus.shape[2]

    @cached_property
    def stimulus_height(self) -> int:
        return self.stimulus.shape[1]

    @property
    def magnification(self) -> np.ndarray:
        if self._fft is None:
            self._fft = SequenceFFT(self.sequences)

        return np.flipud(self._fft.get_intensity())

    @property
    def hue(self) -> np.ndarray:
        if self._fft is None:
            self._fft = SequenceFFT(self.sequences)

        return np.flipud(self._fft.get_phase())

    def calculate_map(self, vperc: float = 98, sperc: float = 90) -> np.ndarray:

        if self._fft is None:
            self._fft = SequenceFFT(self.sequences)

        ret = self._fft.as_colormap(value_perc=vperc, saturation_perc=sperc)
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2RGBA)
        ret = ret.view(dtype=np.uint32).reshape((self._fft.height, self._fft.width))

        return np.flipud(ret)

    def setup(self) -> Model:

        w, h = self.width, self.height

        self.slider_sequence = Slider(
            start=0,
            end=self.sequences.shape[0] - 1,
            value=0,
            width=600,
            title='hsv_value_percentile',
        )

        self.slider_sequence.on_change('value', self.on_select_activity_frame)

        self.fig_activity = figure(title='sequence', width=w, height=h)
        self.view_activity = ActivityView(np.flip(self.sequences, axis=1))
        self.view_activity.plot(self.fig_activity)

        #
        self._setup_stimulus()

        #
        self.slider_hsv_value = Slider(
            start=0,
            end=100,
            value=98,
            width=600,
            title='hsv_value_percentile'
        )

        self.slider_hsv_value.on_change('value', self.on_select_hsv_value)

        self.slider_hsv_hue = Slider(
            start=0,
            end=100,
            value=90,
            width=600,
            title='hsv_saturation_percentile'
        )

        self.slider_hsv_hue.on_change('value', self.on_select_hsv_hue)

        #
        self.fig_intensity = figure(title=f'image_intensity', width=w, height=h)
        self.fig_intensity.image(
            image=[self.magnification],
            x=0,
            y=0,
            dw=w,
            dh=h,
            level='image',
            palette=Greys256[::-1]
        )

        #
        self.fig_phase = figure(
            title=f'image_phase',
            width=w,
            height=h,
            x_range=self.fig_intensity.x_range,
            y_range=self.fig_intensity.y_range
        )
        self.fig_phase.image(
            image=[self.hue],
            x=0,
            y=0,
            dw=w,
            dh=h,
            level='image',
            palette=hsv_rgb()
        )

        #
        self.fig_map = figure(
            title=f'image_map',
            width=w,
            height=h,
            x_range=self.fig_intensity.x_range,
            y_range=self.fig_intensity.y_range
        )

        self.view_map = FFTMap()
        self.view_map.plot(self.fig_map)

        from bokeh.layouts import column
        return column(
            row(
                column(self.slider_sequence,
                       self.fig_activity,
                       self.slider_hsv_value,
                       self.slider_hsv_hue),

                column(self.fig_stimulus)
            ),
            #
            row(self.fig_intensity,
                self.fig_phase,
                self.fig_map)
        )

    def _setup_stimulus(self):
        w, h = self.stimulus_width, self.stimulus_height
        self.fig_stimulus = figure(title='stimulus', width=w, height=h)
        self.view_stimulus = StimulusView(self.stimulus)

        self.slider_sequence.on_change('value', self.on_select_stimulus_frame)
        self.view_stimulus.plot(self.fig_stimulus)

    def on_select_activity_frame(self, attr: str, old: str, value: str):
        nframe = self.slider_sequence.value
        self.view_activity.nframe = nframe
        self.run_later(self.view_activity.update)

    def on_select_stimulus_frame(self, attr: str, old: str, value: str):
        nframe = self.slider_sequence.value
        self.view_stimulus.nframe = nframe
        self.run_later(self.view_stimulus.update)

    def on_select_hsv_value(self, attr: str, old: str, value: str):
        phase_map = self.calculate_map(self.slider_hsv_value.value, self.slider_hsv_hue.value)
        self.run_later(self.view_map.update_image, phase_map)

    def on_select_hsv_hue(self, attr: str, old: str, value: str):
        phase_map = self.calculate_map(self.slider_hsv_value.value, self.slider_hsv_hue.value)
        self.run_later(self.view_map.update_image, phase_map)


def hsv_rgb():
    import matplotlib.colors as mcolors

    colormap = mcolors.hsv_to_rgb([(i / 360, 1, 1) for i in range(0, 360)])
    return [mcolors.to_hex(color) for color in colormap]


def main(args: list[str] = None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-F', '--file', metavar='FILE', type=Path, required=True, help='image file', dest='act_file')
    ap.add_argument('-S', '--stimulus', metavar='FILE', type=Path, required=True, help='stimulus file',
                    dest='stim_file')
    opt = ap.parse_args(args)
    BokehServer().start(FFTView(opt.act_file, opt.stim_file))


if __name__ == '__main__':
    main()
