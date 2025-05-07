from datetime import datetime
from pathlib import Path

from bokeh.model import Model
from bokeh.models import Select, ColumnDataSource, GlyphRenderer
from bokeh.plotting import figure

from neuralib.dashboard import BokehServer, View, ViewComponent
from neuralib.dashboard.util import MsgLog
from neuralib.util.verbose import fprint
from rscvp.dashboard.rig._finder import PhysPathFinder
from rscvp.dashboard.rig.util import *
from stimpyp import RiglogData


class PoolModel:

    def __init__(self, finder: PhysPathFinder):
        self.finder = finder

    def list_animal(self) -> list[str]:
        return self.finder.find_list_animal()

    def list_exp_date(self, animal: str) -> list[str]:
        return self.finder.find_list_exp_date(animal)

    def list_riglog_path(self, animal: str) -> list[Path]:
        return self.finder.find_list_riglog_path(animal)

    @staticmethod
    def list_graph_type() -> list[str]:
        return [v for _, v in PLOTTING_HEADER.items()]


class LapCountsView(ViewComponent):
    """see lap counts across days"""
    data_lap_counts: ColumnDataSource
    render_lap_count: GlyphRenderer

    def __init__(self):
        self.data_lap_counts = ColumnDataSource(data=dict(x=[], y=[]))

        self.rigs: list[RiglogData] | None = None

    def set_riglog(self, rigs: list[RiglogData]):
        self.rigs = rigs

    def plot(self, fig: figure, **kwargs):
        self.render_lap_count = fig.line(
            x='x',
            y='y',
            source=self.data_lap_counts,
        )

    def update(self):
        date = []
        nlaps = []
        for rig in self.rigs:
            date.append(self._get_date(rig.riglog_file))
            nlaps.append(rig.lap_event.value[-1] / rig.total_duration * 3600.)  # nlaps per hour

        self.data_lap_counts.data = dict(x=date, y=nlaps)

    @staticmethod
    def _get_date(p: Path) -> datetime:
        """Get date from riglog path"""
        name = p.parent.name
        part = name.split('_')
        if 'run' not in part[0]:
            ret = part[0]
        else:
            # new stimpy
            name = p.parents[1].name
            ret = name.split('_')[0]

        return datetime.strptime(ret, "%y%m%d")


class ForeachDateView(View):
    INSTANCE = None
    #
    model: PoolModel
    msg_log: MsgLog

    #
    view_lap_counts: LapCountsView
    view_pool_data: list[ViewComponent]

    #
    figure_lap_counts: figure
    figure_pool_data: list[figure]

    #
    select_animal: Select
    select_graph_type: Select

    def __init__(self, finder: PhysPathFinder,
                 animal: str | None = None,
                 graph_type: str | None = None,
                 init_fig_number: int = 10):

        ForeachDateView.INSTANCE = self
        self.model = PoolModel(finder)

        self.msg_log: MsgLog | None = None
        self.list_date: list[str] | None = None

        self._init_fig_number = init_fig_number  # pre-set

        self.figure_pool_data = [
            figure(width=350, height=350, toolbar_location='right')
            for _ in range(self._init_fig_number)
        ]

        # argp
        self._init_animal = animal
        self._init_graph_type = graph_type

    @property
    def title(self) -> str:
        return 'Behavioral Overview'

    @property
    def animal(self) -> str | None:
        if len(self.select_animal.value) == 0:
            return None
        return self.select_animal.value

    @property
    def graph_type(self) -> str | None:
        if len(self.select_graph_type.value) == 0:
            return None
        return self.select_graph_type.value

    @property
    def number_pools(self) -> int:
        if self.list_date is None:
            return self._init_fig_number
        else:
            return len(self.list_date)

    def setup(self) -> Model:

        self.msg_log = MsgLog(value='wait for selection ...')
        #
        self.figure_lap_counts = figure(title='Lap Counts across days',
                                        x_axis_location="above", x_axis_type='datetime',
                                        width=900, height=300,
                                        toolbar_location='below')
        self.view_lap_counts = LapCountsView()
        self.view_lap_counts.plot(self.figure_lap_counts)

        #
        animal_list = self.model.list_animal()
        animal_list.insert(0, '')
        self.select_animal = Select(
            title='Animal',
            value='',
            options=animal_list,
            background='orange'
        )
        self.select_animal.on_change('value', self.on_select_animal)

        #
        graph_type = self.model.list_graph_type()
        graph_type.insert(0, '')
        self.select_graph_type = Select(
            title='Graph Type',
            value='',
            options=graph_type,
            background='orange'
        )
        self.select_graph_type.on_change('value', self.on_select_graph)
        fig_up = self._init_fig_number // 2
        from bokeh.layouts import column, row
        return column(
            row(self.select_animal, self.select_graph_type),
            self.msg_log.message_area,
            self.figure_lap_counts,
            #
            row(self.figure_pool_data[:fig_up]),
            row(self.figure_pool_data[fig_up:self._init_fig_number]),

            self.msg_log.set_div
        )

    def set_pool_viewer(self):
        """set the figure content after selection of `init pool range` and `graph type`"""
        self.view_pool_data = []
        for i in range(self.number_pools):

            match self.graph_type:
                case 'encoder limit':
                    from rscvp.dashboard.rig.beh_dashboard import EncoderView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'encoder (a.u)'
                    self.figure_pool_data[i].yaxis.axis_label = 'time(s)'
                    self.view_pool_data.append(EncoderView())
                case 'lap time diff':
                    from rscvp.dashboard.rig.beh_dashboard import LapTimeDiffView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'time to reward(s)'
                    self.figure_pool_data[i].yaxis.axis_label = 'trials#'
                    self.view_pool_data.append(LapTimeDiffView())
                case 'speed heatmap':
                    from rscvp.dashboard.rig.beh_dashboard import SpeedHeatView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'position'
                    self.figure_pool_data[i].yaxis.axis_label = 'trials#'
                    self.view_pool_data.append(SpeedHeatView())
                case 'speed line':
                    from rscvp.dashboard.rig.beh_dashboard import SpeedLineView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'position'
                    self.figure_pool_data[i].yaxis.axis_label = 'cm/s'
                    self.view_pool_data.append(SpeedLineView())
                case 'peri-reward speed':
                    from rscvp.dashboard.rig.beh_dashboard import PeriRewardSpeedView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'sec'
                    self.figure_pool_data[i].yaxis.axis_label = 'cm/s'
                    self.view_pool_data.append(PeriRewardSpeedView())
                case 'peri-reward lick':
                    from rscvp.dashboard.rig.beh_dashboard import PeriRewardLickView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'sec'
                    self.figure_pool_data[i].yaxis.axis_label = 'cm/s'
                    self.view_pool_data.append(PeriRewardLickView())
                case 'interp position':
                    from rscvp.dashboard.rig.beh_dashboard import InterpPosView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'sec'
                    self.figure_pool_data[i].yaxis.axis_label = 'a.u'
                    self.view_pool_data.append(InterpPosView())
                case 'raw position':
                    from rscvp.dashboard.rig.beh_dashboard import RawPosView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'sec'
                    self.figure_pool_data[i].yaxis.axis_label = 'a.u'
                    self.view_pool_data.append(RawPosView())
                case 'interp velocity':
                    from rscvp.dashboard.rig.beh_dashboard import InterpVelView
                    self.figure_pool_data[i].title.text = self.list_date[i]
                    self.figure_pool_data[i].xaxis.axis_label = 'sec'
                    self.figure_pool_data[i].yaxis.axis_label = 'cm/s'
                    self.view_pool_data.append(InterpVelView())
                case _:
                    msg_log(f'{self.graph_type} is not supported yet!')

            self.view_pool_data[i].plot(self.figure_pool_data[i])

    def update_riglog(self, riglog_paths: list[Path]):
        rigs = [RiglogData(p) for p in riglog_paths]

        self.view_lap_counts.set_riglog(rigs)
        self.run_later(self.view_lap_counts.update)

        for i, rig in enumerate(rigs):
            self.view_pool_data[i].set_rig_attrs(rig)
            self.run_later(self.view_pool_data[i].update)

    def update(self):
        if self._init_animal is not None:
            self.select_animal.value = self._init_animal

        if self._init_graph_type is not None:
            self.select_graph_type.value = self._init_graph_type

    def on_select_animal(self, attr: str, old: str, value: str):
        self.list_date = self.model.list_exp_date(value)

        if len(self.list_date) > self.number_pools:
            msg_log('increase init fig size for more graphs')

        for i in range(self.number_pools):
            self.figure_pool_data.append(
                figure(width=350, height=350, toolbar_location='right')
            )

    def on_select_graph(self, attr: str, old: str, value: str):
        # clean and reset plots
        for fig in self.figure_pool_data:
            fig.renderers.clear()
        self.view_pool_data = []

        #
        try:
            riglog_path = self.model.list_riglog_path(animal=self.animal)
        except FileNotFoundError as e:
            fprint(e, vtype='error')
            msg_log(f'ERR: {self.animal}', reset=True)
        else:
            self.set_pool_viewer()
            self.update_riglog(riglog_path)
            msg_log(f'*{self.graph_type}* were loaded')


def msg_log(*message: str, reset: bool = False):
    message = '\n'.join(message)
    print(message)
    ForeachDateView.INSTANCE.msg_log.on_message(message, reset)


def main(args: list[str] = None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-P', '--path', type=Path, default=None)
    ap.add_argument('--disk', default=None, help='remote disk')
    ap.add_argument('-A', '--animal', default=None)
    ap.add_argument('-G', '--graph', default=None)
    ap.add_argument('-F', '--fig-number', type=int, default=10, dest='fig_number')
    opt = ap.parse_args(args)

    BokehServer().start(
        ForeachDateView(
            PhysPathFinder(root=opt.path, remote_disk=opt.disk),
            animal=opt.animal,
            graph_type=opt.graph,
            init_fig_number=opt.fig_number,
        )
    )


if __name__ == '__main__':
    main()
