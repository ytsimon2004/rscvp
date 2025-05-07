import shutil
from pathlib import Path
from typing import Literal, get_args

import brainrender
import numpy as np
import pandas as pd
from bokeh.model import Model
from bokeh.models import (
    Select,
    Div,
    ColumnDataSource,
    GlyphRenderer,
    DataTable,
    TableColumn,
    Slider,
    LabelSet,
    TabPanel,
    Tabs
)
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, cumsum
from rscvp.atlas.core import RSPRoiClassifier
from rscvp.atlas.dir import AbstractCCFDir, CHANNEL_SUFFIX, CCF_GLOB_TYPE
from rscvp.atlas.util_plot import create_allen_structure_dict
from rscvp.dashboard.altas._finder import HistPathFinder
from rscvp.dashboard.altas.view_image import AbstractImgView
from rscvp.util.io import HISTOLOGY_HOME_ROOT
from tornado.web import StaticFileHandler

from neuralib.atlas.brainrender.roi import RoiRenderCLI
from neuralib.atlas.map import NUM_MERGE_LAYER
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.dashboard import BokehServer, View, ViewComponent
from neuralib.dashboard.util import change_html_format, add_html_format, MsgLog
from neuralib.plot.colormap import DiscreteColorMapper
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import fprint


# TODO another roi viewer (BrainView) for overlap roi selection
# TODO add transformation matrix view
# TODO integrate overlap roi
# TODO norm type, plane_type to opt

class HistModel:

    def __init__(self, finder: HistPathFinder):
        self.finder = finder
        self.ref_root: Path = self.finder.data_root / 'reference-atlas-files'

    def list_animal(self) -> list[str]:
        return self.finder.find_animal()

    def list_glass_slide(self, animal: str) -> list[str]:
        return self.finder.find_glass_slide(animal)

    def list_slice_id(self, animal: str, glass_slide: str) -> list[str]:
        return self.finder.find_slice_id(animal, glass_slide)

    @staticmethod
    def list_of_plane() -> list[str]:
        return ['coronal', 'sagittal', 'transverse']

    @staticmethod
    def list_of_top_value() -> list[int]:
        """for showing the bar/pie chart cutoff"""
        return [i + 3 for i in range(38)]

    @staticmethod
    def list_merge_level() -> list[str]:
        return [str(i) for i in range(1, NUM_MERGE_LAYER + 1)]

    def get_reg_base_path(self, animal: str) -> Path:
        return self.finder.data_root / animal

    def get_resize_image_path(self, animal: str, glass_slide: str, slice_id: str) -> Path:
        return self.finder.find_resize_image_path(animal, glass_slide, slice_id)

    def get_transformation_mat(self, animal: str, glass_slide: str, slice_id: str) -> Path:
        return self.finder.find_transformation_mat(animal, glass_slide, slice_id)

    def get_parsed_csv(self, animal: str) -> Path:
        return self.finder.find_parsed_csv_path(animal)

    @staticmethod
    def get_brain_render_html() -> Path:
        return ensure_dir(HISTOLOGY_HOME_ROOT)

    @property
    def annotation_file(self) -> np.ndarray:
        return np.load(self.ref_root / 'annotation_volume_10um_by_index.npy')


# V
class BrainView(AbstractImgView):
    def __init__(self):
        super().__init__()

        self._offset: int | None = None

    @property
    def width(self) -> float | None:
        return 1140

    @width.setter
    def width(self, value: float):
        pass

    @property
    def height(self) -> float | None:
        return 800

    @height.setter
    def height(self, value: float):
        pass

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int):
        self._offset = value

    @property
    def brain_image(self) -> np.ndarray:
        if self.offset is not None:
            return self.reference[self._offset]
        else:
            return self.reference


Y_UNIT_TYPE = Literal['percent', 'n_rois']


class BarSumView(ViewComponent):
    data_sum_bar: ColumnDataSource
    render_bar: GlyphRenderer

    def __init__(self, yunit: Y_UNIT_TYPE):
        self.roi: RSPRoiClassifier | None = None
        self.data_sum_bar = ColumnDataSource(data=dict(x=[], counts=[]))
        self.fig: figure | None = None
        self.panel: TabPanel | None = None
        self.top_value: int | None = None

        if yunit not in get_args(Y_UNIT_TYPE):
            raise TypeError('')
        self.yunit = yunit

    def plot(self, fig: figure, **kwargs):
        palette = {
            'rfp': 'lightcoral',
            'gfp': 'springgreen',
            'overlap': 'orange'
        }

        self.render_bar = fig.vbar(
            x='x', top='counts', width=1.0, source=self.data_sum_bar,
            color=factor_cmap('x',
                              palette=list(palette.values()),
                              factors=list(palette.keys()),
                              start=1,
                              end=2)
        )
        self.fig = fig
        self.fig.y_range.start = 0
        self.fig.x_range.range_padding = 0.1
        self.fig.xaxis.major_label_orientation = 1
        self.fig.xgrid.grid_line_color = None

        self.panel = TabPanel(child=fig, title=self.yunit)

    def set_roi_cls(self, roi: RSPRoiClassifier):
        self.roi = roi

    def set_top_value(self, value: int):
        self.top_value = value

    def update(self):
        self.data_sum_bar.data = (
            self.roi.get_classified_data(
                top_area=self.top_value,
                add_other=False,
                supply_overlap=True).to_bokeh(yunit=self.yunit)
        )

        self.fig.x_range.factors = self.data_sum_bar.data['x']


class PieSumView(ViewComponent):
    data_sum_pie: ColumnDataSource
    render_pie: GlyphRenderer
    cmapper = DiscreteColorMapper(Category20c, 20)  # share the same mapping in all callers

    def __init__(self, channel: str):
        # self.percent_sorted_data: dict | None = None
        self.roi: RSPRoiClassifier | None = None
        self.data_sum_pie = ColumnDataSource(data=dict(regions=[], value=[], angle=[], color=[]))
        self.top_value: int | None = None
        self.channel = channel

    def plot(self, fig: figure, **kwargs):
        self.render_pie = fig.wedge(x=0, y=1, radius=0.4,
                                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                                    line_color="white", fill_color='color', legend_field='regions',
                                    source=self.data_sum_pie)
        labels = LabelSet(x=0, y=1, text='regions', angle=cumsum('angle', include_zero=True), source=self.data_sum_pie)
        fig.add_layout(labels)

        fig.axis.axis_label = None
        fig.axis.visible = False
        fig.grid.grid_line_color = None

    def set_roi_cls(self, roi: RSPRoiClassifier):
        self.roi = roi

    def set_top_value(self, value: int):
        self.top_value = value

    def _create_pie_chart(self) -> pd.DataFrame:
        from math import pi
        pie = self.roi.get_classified_data(top_area=self.top_value,
                                           add_other=True).to_bokeh()
        data = dict(zip([d[0] for d in pie['x']], pie['counts']))
        pie_data = pd.Series(data).reset_index(name='value').rename(columns={'index': 'regions'})

        region_ls = pie_data['regions']

        pie_data['angle'] = pie_data['value'] / np.sum(pie_data['value']) * 2 * pi
        pie_data['color'] = [self.cmapper[r] for r in region_ls]
        pie_data["regions"] = pie_data["regions"].str.pad(30, side="left")

        return pie_data

    def update(self):
        pie = self._create_pie_chart()
        self.data_sum_pie.data = dict(regions=pie['regions'],
                                      value=pie['value'],
                                      angle=pie['angle'],
                                      color=pie['color'])


class HistoView(View):
    INSTANCE = None
    #
    model: HistModel
    msg_log: MsgLog

    #
    view_brain_resize: BrainView
    view_brain_merge_img: BrainView
    view_brain_gfp_img: BrainView
    view_brain_rfp_img: BrainView
    view_annotation_img: BrainView

    view_summary_bar_perc: BarSumView
    view_summary_bar_ncell: BarSumView
    view_summary_pie_gfp: PieSumView
    view_summary_pie_rfp: PieSumView
    view_summary_pie_overlap: PieSumView

    #
    fig_brain_resize: figure
    fig_brain_merge: figure
    fig_brain_gfp: figure
    fig_brain_rfp: figure
    fig_brain_annotation: figure

    fig_summary_bar_perc: figure
    fig_summary_bar_ncell: figure
    abbr_table: DataTable
    fig_summary_pie_gfp: figure
    fig_summary_pie_rfp: figure
    fig_summary_pie_overlap: figure

    #
    brain_render_view: Div

    #
    select_animal: Select
    select_glass_slide: Select
    select_slice_id: Select
    select_merge_level_bar: Select
    select_merge_level_pie: Select
    slider_top_value_summary_bar: Slider
    slider_top_value_summary_pie: Slider

    def __init__(self, finder: HistPathFinder,
                 animal: str | None = None,
                 region_3d: list[str] | None = None):
        HistoView.INSTANCE = self
        self.msg_log: MsgLog | None = None
        self.model = HistModel(finder)
        self.init_animal = animal
        self.region_3d = region_3d  # will be cache somehow in server, re-create HTML for new regions view

        self.data_root: Path | None = None  # set after select animal
        self.reference_root: Path | None = None  # set after select animal
        # first set while selecting animal, update again after selection of gid, sid
        self.ccf_dir: AbstractCCFDir | None = None
        self.plane_type: PLANE_TYPE | None = None

    @property
    def title(self) -> str:
        return 'Histology Overview'

    @property
    def animal(self) -> str | None:
        if len(self.select_animal.value) == 0:
            return None
        return self.select_animal.value

    @property
    def glass_slide(self) -> str | None:
        if len(self.select_glass_slide.value) == 0:
            return None
        return self.select_glass_slide.value

    @property
    def slice_id(self) -> str | None:
        if len(self.select_slice_id.value) == 0:
            return None
        return self.select_slice_id.value

    @property
    def merge_level_bar(self) -> int:
        return int(self.select_merge_level_bar.value)

    @property
    def merge_level_pie(self) -> int:
        return int(self.select_merge_level_pie.value)

    @property
    def top_value_summary_bar(self) -> int:
        return self.slider_top_value_summary_bar.value

    @property
    def top_value_summary_pie(self) -> int:
        return self.slider_top_value_summary_pie.value

    def setup(self) -> Model:

        self.msg_log = MsgLog(value='wait for ANIMAL/GlassSlide/Slice selection')

        self.fig_brain_resize = figure(title='ROI/DAPI', width=570, height=400)
        self.view_brain_resize = BrainView()
        self.view_brain_resize.plot(self.fig_brain_resize)

        self.fig_brain_merge = figure(title='Overlap', width=570, height=400,
                                      x_range=self.fig_brain_resize.x_range,
                                      y_range=self.fig_brain_resize.y_range)
        self.view_brain_merge_img = BrainView()
        self.view_brain_merge_img.plot(self.fig_brain_merge)

        self.fig_brain_gfp = figure(title='GFP', width=570, height=400,
                                    x_range=self.fig_brain_resize.x_range,
                                    y_range=self.fig_brain_resize.y_range)
        self.view_brain_gfp_img = BrainView()
        self.view_brain_gfp_img.plot(self.fig_brain_gfp)

        self.fig_brain_rfp = figure(title='RFP', width=570, height=400,
                                    x_range=self.fig_brain_resize.x_range,
                                    y_range=self.fig_brain_resize.y_range)
        self.view_brain_rfp_img = BrainView()
        self.view_brain_rfp_img.plot(self.fig_brain_rfp)

        self.fig_brain_annotation = figure(title='Annotation', width=570, height=400)
        self.view_annotation_img = BrainView()
        self.view_annotation_img.plot(self.fig_brain_annotation)

        # bar
        self.fig_summary_bar_perc = figure(
            x_range=[],
            title="summary bar plot",
            width=1200, height=500,
            x_axis_label='area',
            y_axis_label='percent(%)'
        )
        self.view_summary_bar_perc = BarSumView(yunit='percent')
        self.view_summary_bar_perc.plot(self.fig_summary_bar_perc)

        self.fig_summary_bar_ncell = figure(
            x_range=[],
            title="summary bar plot",
            width=1200, height=500,
            x_axis_label='area',
            y_axis_label='# cells'
        )
        self.view_summary_bar_ncell = BarSumView(yunit='n_rois')
        self.view_summary_bar_ncell.plot(self.fig_summary_bar_ncell)

        # table
        abbr = dict(sorted(create_allen_structure_dict().items()))
        abbr_data = ColumnDataSource(data=dict(abbr=(list(abbr.keys())), name=(list(abbr.values()))))
        columns = [
            TableColumn(field='abbr', title='Abbreviation'),
            TableColumn(field='name', title='Full Name')
        ]
        self.abbr_table = DataTable(source=abbr_data, columns=columns, width=400, height=500, background='orange')

        # Pie
        self.fig_summary_pie_gfp = figure(
            height=400, title="Pie Chart_GFP",
            tools="hover,save", tooltips="@regions: @value", x_range=(-0.5, 1.0),
            match_aspect=True
        )
        self.view_summary_pie_gfp = PieSumView('gfp')
        self.view_summary_pie_gfp.plot(self.fig_summary_pie_gfp)

        self.fig_summary_pie_rfp = figure(
            height=400, title="Pie Chart_RFP",
            tools="hover,save", tooltips="@regions: @value", x_range=(-0.5, 1.0)
        )
        self.view_summary_pie_rfp = PieSumView('rfp')
        self.view_summary_pie_rfp.plot(self.fig_summary_pie_rfp)

        self.fig_summary_pie_overlap = figure(
            height=400, title="Pie Chart_OVERLAP",
            tools="hover,save", tooltips="@regions: @value", x_range=(-0.5, 1.0)
        )
        self.view_summary_pie_overlap = PieSumView('overlap')
        self.view_summary_pie_overlap.plot(self.fig_summary_pie_overlap)

        #
        animals = self.model.list_animal()
        animals.insert(0, '')
        self.select_animal = Select(
            title='Animal',
            value='',
            options=animals,
            background='orange'
        )
        self.select_animal.on_change('value', self.on_select_animal)

        self.select_glass_slide = Select(
            title='Glass slide',
            value='',
            options=[],
            background='orange'
        )
        self.select_glass_slide.on_change('value', self.on_select_glass_slide)

        self.select_slice_id = Select(
            title='Slice ID',
            value='',
            options=[],
            background='orange'
        )
        self.select_slice_id.on_change('value', self.on_select_slice_id)

        # fit signature
        update_summary_bar = lambda a, b, c: self.update_summary_bar()
        update_summary_pie = lambda a, b, c: self.update_summary_pie()
        merge_level = self.model.list_merge_level()
        self.select_merge_level_bar = Select(
            width=400,
            title='Bar graph merge level',
            value=merge_level[2],
            options=merge_level,
            background='orange'
        )
        self.select_merge_level_bar.on_change('value', update_summary_bar)

        self.select_merge_level_pie = Select(
            width=400,
            title='pie chart merge level',
            value=merge_level[0],
            options=merge_level,
            background='orange',
        )
        self.select_merge_level_pie.on_change('value', update_summary_pie)

        #
        top_value = self.model.list_of_top_value()
        self.slider_top_value_summary_bar = Slider(
            start=np.min(top_value),
            end=np.max(top_value),
            value=10,
            title='Top value bar display',
            background='orange',
            bar_color='black'
        )
        self.slider_top_value_summary_bar.on_change('value', update_summary_bar)

        self.slider_top_value_summary_pie = Slider(
            start=np.min(top_value),
            end=np.max(top_value),
            value=6,
            title='Top value pie display',
            background='orange',
            bar_color='black'
        )
        self.slider_top_value_summary_pie.on_change('value', update_summary_pie)

        self.brain_render_view = Div(text='',
                                     css_classes=['brain-render'],
                                     height=800,
                                     width=1200)

        from bokeh.layouts import column, row
        return column(
            row(
                self.select_animal,
                self.select_glass_slide,
                self.select_slice_id,
            ),
            self.msg_log.message_area,
            row(
                self.fig_brain_resize,
                self.fig_brain_merge
            ),
            row(
                self.fig_brain_gfp,
                self.fig_brain_rfp,
                self.fig_brain_annotation
            ),
            row(
                self.select_merge_level_bar,
                self.slider_top_value_summary_bar
            ),
            row(
                Tabs(tabs=[self.view_summary_bar_perc.panel,
                           self.view_summary_bar_ncell.panel]),
                self.abbr_table
            ),
            row(
                self.select_merge_level_pie,
                self.slider_top_value_summary_pie
            ),
            row(
                self.fig_summary_pie_gfp,
                self.fig_summary_pie_rfp,
                self.fig_summary_pie_overlap
            ),
            self.brain_render_view,
            Div(text='<style type="text/css">body {background: #000000}</style>'),
            Div(text='<style>.slick-header-column {background-color: lightblue !important;'
                     'background-image: none !important;}</style>'),
            Div(text="""
                <style>
                div.brain-render div {
                    width: 90%;
                    height: 100%;
                }
                div.brain-render div iframe.bk {
                    position: absolute;
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                </style>
                """),
            self.msg_log.set_div
        )

    _reg: AbstractCCFDir | None = None  # cache

    def _query_image_file(self, glob_type: CCF_GLOB_TYPE,
                          hemisphere: Literal['i', 'c'] = 'i',
                          channel: CHANNEL_SUFFIX = 'merge') -> Path:
        if self._reg is None:
            self.ccf_dir = AbstractCCFDir(self.data_root, plane_type=self.plane_type)
            self._reg = self.ccf_dir

        try:
            return self.ccf_dir.glob(
                int(self.glass_slide),
                int(self.slice_id),
                glob_type,
                hemisphere=hemisphere if self.plane_type == 'sagittal' else None,
                channel=channel
            )
        except FileNotFoundError as e:
            msg_log(f'ERR: image file: {glob_type} not found in {e}')

    def update(self):
        if self.init_animal is not None:
            self.select_animal.value = self.init_animal

    def update_image(self):
        resize_file = self._query_image_file('resize')
        merge_img = self._query_image_file('zproj', channel='merge')
        gfp_img = self._query_image_file('zproj', channel='g')
        rfp_img = self._query_image_file('zproj', channel='r')
        annotation_img = self._query_image_file('transformation_img')

        self.view_brain_resize.load_file(resize_file)
        self.view_brain_merge_img.load_file(merge_img)
        self.view_brain_gfp_img.load_file(gfp_img)
        self.view_brain_rfp_img.load_file(rfp_img)
        self.view_annotation_img.load_annotation_overlay(
            self.ccf_dir,
            int(self.glass_slide),
            int(self.slice_id),
            annotation_img,
            self.plane_type
        )

        self.run_later(self.view_brain_resize.update)
        self.run_later(self.view_brain_merge_img.update)
        self.run_later(self.view_brain_gfp_img.update)
        self.run_later(self.view_brain_rfp_img.update)
        self.run_later(self.view_annotation_img.update)

    def update_summary_bar(self):
        if self.data_root is not None:
            try:
                roi = RSPRoiClassifier(self.ccf_dir, self.merge_level_bar)
            except FileNotFoundError as e:
                msg_log(f'ERR: {e}')
            else:
                self.view_summary_bar_perc.set_roi_cls(roi)
                self.view_summary_bar_perc.set_top_value(self.top_value_summary_bar)
                self.view_summary_bar_ncell.set_roi_cls(roi)
                self.view_summary_bar_ncell.set_top_value(self.top_value_summary_bar)
                self.run_later(self.view_summary_bar_perc.update)
                self.run_later(self.view_summary_bar_ncell.update)

    def update_summary_pie(self):
        if self.data_root is not None:
            roi = RSPRoiClassifier(self.ccf_dir, self.merge_level_pie)

            self.view_summary_pie_gfp.set_roi_cls(roi)
            self.view_summary_pie_gfp.set_top_value(self.top_value_summary_pie)
            self.view_summary_pie_rfp.set_roi_cls(roi)
            self.view_summary_pie_rfp.set_top_value(self.top_value_summary_pie)
            self.view_summary_pie_overlap.set_roi_cls(roi)
            self.view_summary_pie_overlap.set_top_value(self.top_value_summary_pie)

            self.run_later(self.view_summary_pie_gfp.update)
            self.run_later(self.view_summary_pie_rfp.update)
            self.run_later(self.view_summary_pie_overlap.update)

    def _update_brain_render(self, animal: str, region: list[str]):
        """create html per each animal"""
        # TODO interactive show different areas, use button input?
        # TODO change sphere color
        p = self.model.get_brain_render_html()
        f = list(p.glob(f'{animal}.html'))
        try:
            return f[0].name
        except IndexError:
            html_path = (p / animal).with_suffix('.html')

            reconstruct = RoiRenderCLI
            reconstruct.csv_file = self.model.get_parsed_csv(animal)
            reconstruct.regions = region
            reconstruct.source = 'allen_mouse_10um'
            reconstruct.output = html_path
            reconstruct.scene = brainrender.Scene(inset=False, title=self.title, screenshots_folder='.')
            clz = reconstruct()
            clz.run()
            RoiRenderCLI.export(clz)

            change_html_format(html_path, {'#canvasTarget': 'background: black'})
            add_html_format(html_path, {'div.dg.main': 'height: 800px'})

            msg = f'create html for {animal} in {html_path}...'
            msg_log(msg)
            fprint(msg, vtype='io')

            return list(p.glob(f'{animal}.html'))[0].name

    def on_select_animal(self, attr: str, old: str, value: str):

        if len(value) == 0:
            self.data_root = None
            return

        glass_slide = self.model.list_glass_slide(value)
        glass_slide.insert(0, '')
        self.select_glass_slide.options = glass_slide

        src = self.model.finder.data_root
        self.data_root = src / value
        self.reference_root = src / 'reference-atlas-files'

        self.plane_type = 'coronal' if value not in RSPRoiClassifier.DEFAULT_SAGITTAL_ANIMALS else 'sagittal'
        self.ccf_dir = AbstractCCFDir(self.data_root, plane_type=self.plane_type)

        self.run_later(self.update_summary_bar)
        self.run_later(self.update_summary_pie)

        # brain render
        url = self._update_brain_render(self.animal, self.region_3d)

        if not (Path(__file__).parent / url).exists():  # copy to working dir
            src_file = (HISTOLOGY_HOME_ROOT / url).with_suffix('.html')
            dst_file = (Path(__file__).parent / url).with_suffix('.html')
            shutil.copy2(src_file, dst_file)
        self.brain_render_view.text = f'<iframe src="{url}" class="bk" ></iframe>'

    def on_select_glass_slide(self, attr: str, old: str, value: str):
        if len(value) == 0:
            return

        if self._reg is not None:
            self._reg = None  # renew img

        assert self.animal is not None

        slice_id = self.model.list_slice_id(self.animal, value)
        slice_id.insert(0, '')
        self.select_slice_id.options = slice_id

    def on_select_slice_id(self, attr: str, old: str, value: str):
        if len(value) == 0:
            return

        if self._reg is not None:
            self._reg = None  # renew img

        assert self.animal is not None

        if self.data_root is not None:
            try:
                self.update_image()
            except (AttributeError, FileNotFoundError) as e:
                print(e)
                msg_log(f'ERR: image files not found or partial loss, check {self.data_root}')


def msg_log(*message: str, reset: bool = False):
    message = '\n'.join(message)
    print(message)
    HistoView.INSTANCE.msg_log.on_message(message, reset)


def main(arg: list[str] = None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-P', '--path', type=Path, default=None)
    ap.add_argument('--disk', default=None, help='remote disk')
    ap.add_argument('--top-value', '--top', metavar='NUMBER')
    ap.add_argument('-R', '--region', type=lambda it: it.split(','), default=['RSP'],
                    help='region showed in brainrender')
    ap.add_argument('-A', '--animal', default=None)
    opt = ap.parse_args(arg)

    extra_patterns = [(r"/(.*)", StaticFileHandler,
                       {"path": str(Path(__file__).parent)})]

    BokehServer().start(
        HistoView(HistPathFinder(root=opt.path, remote_disk=opt.disk), opt.animal, region_3d=opt.region),
        extra_patterns=extra_patterns
    )


if __name__ == '__main__':
    main()
