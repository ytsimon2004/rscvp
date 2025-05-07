import warnings
from pathlib import Path
from typing import get_args

import numpy as np
from bokeh.model import Model
from bokeh.models import Select, Span, Div, ColumnDataSource, GlyphRenderer, Band
from bokeh.palettes import Cividis256
from bokeh.plotting import figure
from rscvp.behavioral.util import get_velocity_per_trial, peri_reward_velocity
from rscvp.dashboard.rig._finder import PhysPathFinder
from rscvp.dashboard.rig.util import PLOTTING_HEADER
from rscvp.util.cli import DAQ_TYPE
from rscvp.util.position import load_interpolated_position
from rscvp.util.util_lick import peri_reward_raster_hist
from scipy.stats import sem

from neuralib.dashboard import BokehServer, View, ViewComponent
from neuralib.dashboard.util import ColorBarView, MsgLog
from neuralib.locomotion import CircularPosition
from neuralib.util.verbose import fprint
from stimpyp import RiglogData


# M(Model)
class BehavioralModel:

    def __init__(self, finder: PhysPathFinder):
        self.finder = finder

    @staticmethod
    def list_exp_type() -> list[str]:
        return list(get_args(DAQ_TYPE))

    def list_animals(self) -> list[str]:
        return self.finder.find_list_animal()

    def list_exp_date(self, animal: str) -> list[str]:
        return self.finder.find_list_exp_date(animal)

    def get_riglog_path(self, animal: str, exp_date: str) -> Path:
        return self.finder.find_riglog_path(animal, exp_date)


class EncoderView(ViewComponent):
    """
    find peak points for the encoder limit for each lap, to avoid tape no function (use up)
    """
    data_encoder_limit: ColumnDataSource
    render_line: GlyphRenderer

    def __init__(self):
        self.data_encoder_limit = ColumnDataSource(data=dict(time=[], enc=[]))
        self.flag_text = ''
        self.flag_encoder_limit = Div(text='', styles={'color': 'orange'})

        self.rig: RiglogData | None = None

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig

    def plot(self, fig: figure, **kwargs):
        self.render_line = fig.line(x='time', y='enc', line_width=0.8, color='green',
                                    source=self.data_encoder_limit)

    def update(self):
        enc = self._find_encoder_values_per_lap(self.rig)
        self.data_encoder_limit.data = dict(time=enc[:, 0], enc=enc[:, 1])
        self.check_encoder(enc[:, 1])
        self.flag_encoder_limit.text = self.flag_text

    @staticmethod
    def _find_encoder_values_per_lap(rig: RiglogData) -> np.ndarray:
        """
        :return: array  (points, 2) with time and value
        """
        time = rig.position_event.time
        enc = rig.position_event.value
        idx = np.nonzero(np.diff(enc) < -100)[0]  # arbitrary cutoff

        return np.stack([time[idx], enc[idx]], axis=1)

    @staticmethod
    def check_encoder(enc_limit: np.ndarray):
        med_peak = np.median(enc_limit)
        err_exceed = enc_limit > (med_peak + 500)
        err_less = enc_limit < 4000

        if np.any(err_exceed):
            nlap_err = np.count_nonzero(err_exceed)
            msg_log(f'ERR: encoder value exceed in {nlap_err} lap(s)!, might be tape not sensitive')

        elif np.any(err_less):
            nlap_err = np.count_nonzero(err_less)
            msg_log(f'ERR: encoder value exceed in {nlap_err} lap(s)!, might check the screw of encoder')

        else:
            msg_log('>>> encoder readout are working functionally!')


class LapTimeDiffView(ViewComponent):
    data_lap_time_diff: ColumnDataSource
    render_line: GlyphRenderer

    def __init__(self):
        self.data_lap_time_diff = ColumnDataSource(data=dict(num=[], time=[]))
        self.flag_text = ''
        self.flag_lap_interval = Div(text='', styles={'color': 'orange'})

        self.rig: RiglogData | None = None

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig

    def plot(self, fig: figure, **kwargs):
        self.render_line = fig.line(x='num', y='time', line_width=0.8, color='green',
                                    source=self.data_lap_time_diff)

    def update(self):
        time = np.diff(self.rig.lap_event.time)
        num = np.arange(len(time))
        self.data_lap_time_diff.data = dict(num=num, time=time)

        self.check_lap_time_diff(time)
        self.flag_lap_interval.text = self.flag_text

    def check_lap_time_diff(self, time: np.ndarray, threshold: float = 3):
        """check if any lap time diff less than 3s"""
        if np.any(time < threshold):
            err_lap = np.count_nonzero(time < threshold)
            msg_log(f'ERR: lap {err_lap} found DJ issue!')
        else:
            msg_log('>>> no DJ issue observed!')


# V(View)
class PeriRewardLickView(ViewComponent):
    """Peri-reward licking behaviors"""
    data_lick_raster: ColumnDataSource
    data_lick_hist: ColumnDataSource

    render_scatter: GlyphRenderer
    render_quad: GlyphRenderer

    def __init__(self):
        self.data_lick_raster = ColumnDataSource(data=dict(x=[], y=[]))
        self.data_lick_hist = ColumnDataSource(data=dict(hist=[], left=[], right=[]))

        self.rig: RiglogData | None = None
        self.pos: CircularPosition | None = None

    def plot(self, fig: figure, **kwargs):
        self.render_scatter = fig.scatter(
            x='x',
            y='y',
            source=self.data_lick_raster,
            marker='dot',
            size=5,
            color='white'
        )

        self.render_quad = fig.quad(
            top='hist',
            bottom=0,
            left='left',
            right='right',
            fill_color="orange",
            line_color="orange",
            alpha=0.2,
            source=self.data_lick_hist
        )

        reward_time = Span(
            location=0,
            dimension='height',
            line_color='red',
            line_dash='dashed',
            line_width=2
        )
        fig.add_layout(reward_time)

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig

    def update(self):
        lick_per_trial, hist, edg = peri_reward_raster_hist(self.rig.lick_event.time,
                                                            self.rig.reward_event.time,
                                                            limit=3)
        x = []
        y = []
        for i, lick_time in enumerate(lick_per_trial):
            x.append(lick_time)
            y.append(np.full_like(lick_time, i + 1))

        x = np.concatenate(x)
        y = np.concatenate(y)

        self.data_lick_raster.data = dict(x=x, y=y)
        # norm to the same histogram height
        self.data_lick_hist.data = dict(hist=hist * len(lick_per_trial) / np.max(hist), left=edg[:-1], right=edg[1:])


class SpeedHeatView(ViewComponent):
    data_speed: ColumnDataSource
    render_heatmap: GlyphRenderer

    def __init__(self):
        self.data_speed = ColumnDataSource(data=dict(vel=[], pos=[], nlap=[]))

        self.rig: RiglogData | None = None
        self.pos: CircularPosition | None = None
        self.color_bar: ColorBarView | None = None

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig
        self.pos = load_interpolated_position(rig, sample_rate=100, force_compute=True)

    def plot(self, fig: figure, **kwargs):
        self.render_heatmap = fig.image(image='vel', x=0, y=0, dw='pos', dh='nlap', source=self.data_speed,
                                        palette=Cividis256, level='image')
        self.color_bar = ColorBarView(self.render_heatmap, key='vel', palette='Cividis256')
        self.color_bar.insert(fig)

    def update(self):
        data = get_velocity_per_trial(self.rig.lap_event.time, self.pos)
        self.data_speed.data = dict(vel=[data], pos=[data.shape[1]], nlap=[data.shape[0]])


class SpeedLineView(ViewComponent):
    """Speed as a function of position plot"""
    data_speed: ColumnDataSource
    data_speed_avg: ColumnDataSource

    render_line: GlyphRenderer
    render_avg_line: GlyphRenderer

    def __init__(self):
        self.data_speed = ColumnDataSource(data=dict(x=[], y=[]))
        self.data_speed_avg = ColumnDataSource(data=dict(x=[], y=[]))

        self.rig: RiglogData | None = None
        self.pos: CircularPosition | None = None

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig
        self.pos = load_interpolated_position(rig, sample_rate=100, force_compute=True)

    def plot(self, fig: figure, **kwargs):
        self.render_line = fig.line(x='x', y='y', source=self.data_speed, line_width=0.5, color='lightgray')
        self.render_avg_line = fig.line(x='x', y='y', source=self.data_speed_avg, line_width=2, color='orange')

    def update(self):
        data = get_velocity_per_trial(self.rig.lap_event.time, self.pos, smooth=False)
        avg_v = np.mean(data, axis=0)
        x = []
        y = []
        for lap, vel in enumerate(data):
            y.append(vel)
            x.append(np.arange(len(vel)))
            x.append([np.nan])
            y.append([np.nan])

        x = np.concatenate(x, dtype=float)
        y = np.concatenate(y, dtype=float)

        self.data_speed.data = dict(x=x, y=y)
        self.data_speed_avg.data = dict(x=list(range(0, 150)), y=avg_v)


class PeriRewardSpeedView(ViewComponent):
    data_speed: ColumnDataSource
    render_line: GlyphRenderer

    def __init__(self):
        self.data_speed = ColumnDataSource(data=dict(x=[], y=[], upper=[], lower=[]))

        self.rig: RiglogData | None = None
        self.pos: CircularPosition | None = None

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig
        self.pos = load_interpolated_position(rig, sample_rate=100, force_compute=True)

    def plot(self, fig: figure, **kwargs):
        self.render_line = fig.line(x='x', y='y', source=self.data_speed, line_width=1, color='orange')

        sem_band = Band(base='x', lower='lower', upper='upper', source=self.data_speed,
                        level='underlay', fill_alpha=0.5, line_width=1)
        reward_time = Span(location=0,
                           dimension='height', line_color='red',
                           line_dash='dashed', line_width=2)
        fig.add_layout(sem_band)
        fig.add_layout(reward_time)

    def update(self):
        vel = peri_reward_velocity(self.rig.reward_event.time, self.pos.t, self.pos.v, limit=3)
        v_mean = np.mean(vel, axis=0)
        v_sem = sem(vel)
        x = np.linspace(-3, 3, num=100)

        self.data_speed.data = dict(x=x, y=v_mean, upper=v_mean + v_sem, lower=v_mean - v_sem)


class InterpPosView(ViewComponent):
    data_interp_pos: ColumnDataSource

    data_lap: ColumnDataSource
    data_reward: ColumnDataSource
    data_lick: ColumnDataSource

    render_line_interp_pos: GlyphRenderer
    render_scatter_lap: GlyphRenderer
    render_scatter_reward: GlyphRenderer
    render_scatter_lick: GlyphRenderer

    def __init__(self):
        self.data_interp_pos = ColumnDataSource(data=dict(time=[], enc=[]))
        self.data_lap = ColumnDataSource(data=dict(time=[], lap=[]))
        self.data_reward = ColumnDataSource(data=dict(time=[], reward=[]))
        self.data_lick = ColumnDataSource(data=dict(time=[], lick=[]))

        self.rig: RiglogData | None = None
        self.pos: CircularPosition | None = None

    def plot(self, fig: figure, **kwargs):
        self.render_line_interp_pos = fig.line(x='time', y='enc', source=self.data_interp_pos,
                                               legend_label="interp_position",
                                               line_width=1,
                                               color='orange',
                                               alpha=0.5)

        self.render_scatter_lick = fig.scatter(x='time', y='lick', marker='dot', size=10, color='red',
                                               legend_label='lick', source=self.data_lick)
        self.render_scatter_lap = fig.scatter(x='time', y='lap', marker='dot', size=10, color='green',
                                              legend_label='lap', source=self.data_lap)
        self.render_scatter_reward = fig.scatter(x='time', y='reward', marker='dot', size=10, color='blue',
                                                 legend_label='reward', source=self.data_reward)

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig
        self.pos = load_interpolated_position(rig, sample_rate=60, force_compute=True, save_cache=False)

    def update(self):
        self.data_interp_pos.data = dict(time=self.pos.t, enc=self.pos.p)
        self.data_lick.data = dict(time=self.rig.lick_event.time, lick=np.full_like(self.rig.lick_event.time, 160))
        self.data_lap.data = dict(time=self.rig.lap_event.time, lap=np.full_like(self.rig.lap_event.time, 170))
        self.data_reward.data = dict(time=self.rig.reward_event.time,
                                     reward=np.full_like(self.rig.lap_event.time, 180))


class RawPosView(ViewComponent):
    data_raw_pos: ColumnDataSource
    render_line_pos: GlyphRenderer

    def __init__(self):
        self.data_raw_pos = ColumnDataSource(data=dict(time=[], enc=[]))

        self.rig: RiglogData | None = None

    def plot(self, fig: figure, **kwargs):
        self.render_line_pos = fig.line(x='time', y='enc', source=self.data_raw_pos,
                                        legend_label='position',
                                        line_width=1,
                                        color='white',
                                        alpha=0.5)

    def set_rig_attrs(self, rig: RiglogData):
        self.rig = rig

    def update(self):
        self.data_raw_pos.data = dict(time=self.rig.position_event.time, enc=self.rig.position_event.value)


class InterpVelView(ViewComponent):
    data_interp_vel: ColumnDataSource
    render_line: GlyphRenderer

    def __init__(self):
        self.data_interp_vel = ColumnDataSource(data=dict(x=[], y=[]))

        self.pos: CircularPosition | None = None

    def plot(self, fig: figure, **kwargs):
        self.render_line = fig.line(x='x', y='y', source=self.data_interp_vel, legend_label="speed", line_width=0.6,
                                    color='orange')

    def set_rig_attrs(self, rig: RiglogData):
        self.pos = load_interpolated_position(rig, sample_rate=60, force_compute=True, save_cache=False)

    def update(self):
        self.data_interp_vel.data = dict(x=self.pos.t, y=self.pos.v)


class BehavioralView(View):
    INSTANCE = None
    #
    model: BehavioralModel
    msg_log: MsgLog

    #
    view_encoder_limit: EncoderView
    view_lap_time_diff: LapTimeDiffView

    view_speed_heat: SpeedHeatView
    view_speed_line: SpeedLineView
    view_peri_reward_speed: PeriRewardSpeedView

    view_peri_reward_lick: PeriRewardLickView

    view_interp_pos: InterpPosView
    view_raw_pos: RawPosView
    view_interp_vel: InterpVelView

    #

    figure_encoder_limit: figure
    figure_lap_time_diff: figure

    figure_speed_heat: figure
    figure_speed_line: figure
    figure_peri_reward_speed: figure

    figure_peri_reward_lick: figure

    figure_interp_pos: figure
    figure_raw_pos: figure
    figure_interp_vel: figure

    #
    select_exp_type: Select
    select_animal: Select
    select_exp_date: Select

    def __init__(self, finder: PhysPathFinder, exp_type: DAQ_TYPE | None = None):
        BehavioralView.INSTANCE = self
        self.model = BehavioralModel(finder)
        self.init_exp_type = exp_type
        self.msg_log: MsgLog | None = None

    @property
    def animal(self) -> str | None:
        if len(self.select_animal.value) == 0:
            return None
        return self.select_animal.value

    @property
    def exp_type(self) -> str | None:
        if len(self.select_exp_type.value) == 0:
            return None
        return self.select_exp_type.value

    @property
    def exp_date(self) -> str | None:
        if len(self.select_exp_date.value) == 0:
            return None
        return self.select_exp_date.value

    @property
    def title(self) -> str:
        return 'Behavioral Overview'

    def setup(self) -> Model:
        #
        self.msg_log = MsgLog(value='wait for DATE/ANIMAL selection ...')
        header = PLOTTING_HEADER
        #
        self.figure_encoder_limit = figure(title=header['encoders'],
                                           width=350, height=350,
                                           x_axis_label='time(s)',
                                           y_axis_label='encoder limit(readout)',
                                           toolbar_location='right')
        self.view_encoder_limit = EncoderView()
        self.view_encoder_limit.plot(self.figure_encoder_limit)

        #
        self.figure_lap_time_diff = figure(title=header['lap_interval'],
                                           width=350, height=350,
                                           x_axis_label='lap number #',
                                           y_axis_label='\u0394t',
                                           toolbar_location='right')

        self.view_lap_time_diff = LapTimeDiffView()
        self.view_lap_time_diff.plot(self.figure_lap_time_diff)

        #
        self.figure_speed_heat = figure(title=header['speed_heatmap'],
                                        width=400, height=300,
                                        x_axis_label="position(cm)",
                                        y_axis_label="laps #",
                                        toolbar_location='right')
        self.view_speed_heat = SpeedHeatView()
        self.view_speed_heat.plot(self.figure_speed_heat)

        #
        self.figure_speed_line = figure(title=header['speed_line'],
                                        width=350, height=300,
                                        x_axis_label="position(cm)",
                                        y_axis_label="speed(cm/s)",
                                        toolbar_location='right')
        self.view_speed_line = SpeedLineView()
        self.view_speed_line.plot(self.figure_speed_line)

        #
        self.figure_peri_reward_speed = figure(title=header['peri_reward_speed'],
                                               width=350, height=300,
                                               x_axis_label="time relative to reward(s)",
                                               y_axis_label="speed(cm/s)",
                                               toolbar_location='right')
        self.view_peri_reward_speed = PeriRewardSpeedView()
        self.view_peri_reward_speed.plot(self.figure_peri_reward_speed)

        #
        self.figure_peri_reward_lick = figure(title=header['peri_reward_lick'],
                                              width=350, height=350,
                                              x_axis_label='time relative to reward(s)',
                                              y_axis_label='trials#',
                                              toolbar_location='right')
        self.view_peri_reward_lick = PeriRewardLickView()
        self.view_peri_reward_lick.plot(self.figure_peri_reward_lick)

        #
        self.figure_interp_pos = figure(title=header['interp_pos'],
                                        width=1000, height=300,
                                        x_axis_label="time(s)",
                                        y_axis_label="encoder",
                                        toolbar_location='right')

        self.view_interp_pos = InterpPosView()
        self.view_interp_pos.plot(self.figure_interp_pos)

        #
        self.figure_raw_pos = figure(title=header['raw_pos'],
                                     width=1000, height=150,
                                     x_axis_label="time(s)",
                                     y_axis_label="encoder",
                                     x_range=self.figure_interp_pos.x_range,
                                     toolbar_location='right')

        self.view_raw_pos = RawPosView()
        self.view_raw_pos.plot(self.figure_raw_pos)

        #
        self.figure_interp_vel = figure(title=header['interp_vel'],
                                        width=1000, height=150,
                                        x_axis_label="time(s)",
                                        y_axis_label="vel(cm/s)",
                                        x_range=self.figure_interp_pos.x_range,
                                        toolbar_location='right')
        self.view_interp_vel = InterpVelView()
        self.view_interp_vel.plot(self.figure_interp_vel)

        ##
        exp_type = self.model.list_exp_type()
        exp_type.insert(0, '')
        self.select_exp_type = Select(
            title='ExpType',
            value='',
            options=exp_type,
            background='orange'
        )
        self.select_exp_type.on_change('value', self.on_select_exp_type)

        self.select_animal = Select(
            title='Animal',
            value='',
            options=[],
            background='orange'
        )
        self.select_animal.on_change('value', self.on_select_animal)

        self.select_exp_date = Select(
            title='Experimental date',
            value='',
            options=[],
            background='orange'
        )
        self.select_exp_date.on_change('value', self.on_select_exp_date)

        from bokeh.layouts import column, row
        return column(
            row(
                self.select_exp_type,
                self.select_animal,
                self.select_exp_date
            ),
            self.msg_log.message_area,
            row(
                self.view_encoder_limit.flag_encoder_limit,
                self.view_lap_time_diff.flag_lap_interval,
            ),
            row(
                self.figure_encoder_limit,
                self.figure_lap_time_diff
            ),
            row(
                self.figure_speed_heat,
                self.figure_speed_line,
                self.figure_peri_reward_speed
            ),
            self.figure_peri_reward_lick,
            self.figure_interp_pos,
            self.figure_raw_pos,
            self.figure_interp_vel,
            self.msg_log.set_div
        )

    def update_riglog(self, riglog_filepath: Path):

        rig = RiglogData(riglog_filepath)

        self.view_encoder_limit.set_rig_attrs(rig)
        self.view_lap_time_diff.set_rig_attrs(rig)

        self.view_speed_heat.set_rig_attrs(rig)
        self.view_speed_line.set_rig_attrs(rig)
        self.view_peri_reward_speed.set_rig_attrs(rig)

        self.view_peri_reward_lick.set_rig_attrs(rig)
        self.view_interp_pos.set_rig_attrs(rig)
        self.view_raw_pos.set_rig_attrs(rig)
        self.view_interp_vel.set_rig_attrs(rig)

        self.run_later(self.view_encoder_limit.update)
        self.run_later(self.view_lap_time_diff.update)

        self.run_later(self.view_speed_heat.update)
        self.run_later(self.view_speed_line.update)
        self.run_later(self.view_peri_reward_speed.update)

        self.run_later(self.view_peri_reward_lick.update)

        self.run_later(self.view_interp_pos.update)
        self.run_later(self.view_raw_pos.update)
        self.run_later(self.view_interp_vel.update)

    def update(self):
        if self.init_exp_type is not None:
            self.select_exp_type.value = self.init_exp_type

    # C(Controller)
    def on_select_exp_type(self, attr: str, old: str, value: str):
        if len(value) == 0:  # empty select
            return

        self.model.finder.daq_type = value
        animals = self.model.list_animals()
        animals.insert(0, '')  # avoid loading data when re-selected exp_type
        self.select_animal.options = animals

    def on_select_animal(self, attr: str, old: str, value: str):
        if len(value) == 0:
            return
        assert self.exp_type is not None

        exp_date = self.model.list_exp_date(value)
        exp_date.insert(0, '')
        self.select_exp_date.options = exp_date

    def on_select_exp_date(self, attr: str, old: str, value: str):
        if len(value) == 0:
            return

        try:
            # noinspection PyTypeChecker
            riglog_path = self.model.get_riglog_path(self.animal, value)
        except FileNotFoundError as e:
            fprint(e, vtype='error')
            msg_log(f'ERR: Riglog not found in {value}_{self.animal}_{self.exp_type}', reset=True)
        else:
            self.update_riglog(riglog_path)
            msg_log(f'Load successful from {value}_{self.animal}_{self.exp_type}', reset=True)


def msg_log(*message: str, reset: bool = False):
    message = '\n'.join(message)
    print(message)
    try:
        BehavioralView.INSTANCE.msg_log.on_message(message, reset)
    except AttributeError:
        warnings.warn('different viewer')
        pass


def main(args: list[str] = None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-P', '--path', type=Path, default=None)
    ap.add_argument('--disk', default=None, help='remote disk')
    ap.add_argument('-E', '--exp-type', default=None)
    opt = ap.parse_args(args)

    BokehServer().start(
        BehavioralView(PhysPathFinder(root=opt.path, remote_disk=opt.disk), exp_type=opt.exp_type)
    )


if __name__ == '__main__':
    main()
