from datetime import datetime
from typing import Callable, Literal, TypeVar, Generic, ParamSpec, Any

import numpy as np
from scipy.stats import sem

from argclz import AbstractParser, as_argument, try_float_type, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure, plot_peri_onset_1d
from neuralib.util.verbose import publish_annotation
from rscvp.behavioral.util import get_velocity_per_trial
from rscvp.behavioral.util_plot import plot_velocity_line
from rscvp.util.cli.cli_camera import CameraOptions
from rscvp.util.cli.cli_treadmill import TreadmillOptions
from rscvp.util.util_lick import peri_reward_raster_hist

__all__ = ['BehaviorBatchPlotOptions']

P = ParamSpec('P')
F = TypeVar('F', bound=Callable[P, Any])


@publish_annotation('appendix', project='rscvp', as_doc=True)
class BehaviorBatchPlotOptions(AbstractParser, CameraOptions, TreadmillOptions, Dispatch, Generic[P]):
    DESCRIPTION = 'Plot multiple (batch) animals for treadmill behavioral analysis'

    dispatch_plot: Literal[
        'peri_reward_vel',
        'vel_as_position',
        'peri_reward_lick'
    ] = argument(
        '--plot',
        required=True,
        help='which dispatch analysis'
    )

    lick_thres = as_argument(CameraOptions.lick_thres).with_options(
        help='Foreach set threshold corresponding to the dataset'
    )

    fix_color: bool = argument('--gray', help='fix gray color across dataset')

    names: list[str] = []
    """list of data name. <ED>_<ID>"""

    invalid_riglog_cache = True

    def run(self):
        self.invoke_command(self.dispatch_plot)

    def plot_batch(self, dataset: list[P.args],
                   plot_func: F,
                   with_multi_args: bool = True,
                   ax_as_keyword: bool = False,
                   **kwargs):
        """
        foreach plot the batch dataset

        `Dimension parameters`:

            D = number of dataset

            ... = ``plot_func()`` return 1D value array

        :param dataset: container collect the arguments for plotting function
        :param plot_func: plotting function
        :param with_multi_args: plot_func has multiple args
        :param ax_as_keyword: if True, pass ax as keyword argument; if False, pass as first positional
        :param kwargs: pass to arg `plot_func`
        :return: values across animals, which return by the ``plot_func()``
        """
        ret = []
        with plot_figure(None) as ax:
            for i, it in enumerate(dataset):
                args = (it,) if not with_multi_args else it

                if ax_as_keyword:
                    x, arr = plot_func(*args,
                                       ax=ax,
                                       label=self.names[i],
                                       color='gray' if self.fix_color else None,
                                       alpha=0.6,
                                       **kwargs)
                else:
                    x, arr = plot_func(ax, *args,
                                       label=self.names[i],
                                       color='gray' if self.fix_color else None,
                                       alpha=0.6,
                                       **kwargs)
                ret.append(arr)
                ax.legend()

            values = np.array(ret)  # (D, B)
            sem_h = sem(values, axis=0)
            mean_values = np.mean(values, axis=0)
            ax.plot(x, mean_values, color='pink')
            ax.fill_between(x, mean_values + sem_h, mean_values - sem_h, color='pink', alpha=0.4)

    # ==================== #
    # Peri-Reward Velocity #
    # ==================== #

    VelDType = list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    """list of args from ``plot_peri_reward_velocity(ax, *arg)``"""

    @dispatch('peri_reward_vel')
    def plot_peri_reward_velocity_batch(self):
        """Plot peri-reward velocity in batch dataset"""
        dataset: BehaviorBatchPlotOptions.VelDType = []
        for i, _ in enumerate(self.foreach_dataset()):
            riglog = self.load_riglog_data()
            reward_time = riglog.reward_event.time

            pos = self.load_position()
            pt, p, v = pos.t, pos.p, pos.v

            if self.cutoff_vel is not None:
                v[(v < self.cutoff_vel)] = 0
            dataset.append((reward_time, pt, v))

            name = f'{self.exp_date}_{self.animal_id}'
            self.names.append(name)

        self.plot_batch(dataset,
                        plot_peri_onset_1d,
                        ax_as_keyword=True,
                        pre=self.psth_sec,
                        post=self.psth_sec,
                        plot_all=False,
                        with_fill_between=False)

    # ==================== #
    # Velocity As Position #
    # ==================== #

    PosDType = list[np.ndarray]

    @dispatch('vel_as_position')
    def plot_velocity_line_batch(self):
        """Plot velocity as a function of position bins in batch dataset"""
        dataset: BehaviorBatchPlotOptions.PosDType = []
        for i, _ in enumerate(self.foreach_dataset()):
            riglog = self.load_riglog_data()
            lap_time = riglog.lap_event.time

            pos = self.load_position()
            pt, p, v = pos.t, pos.p, pos.v

            if self.cutoff_vel is not None:
                v[(v < self.cutoff_vel)] = 0

            m = get_velocity_per_trial(lap_time, pos, self.track_length, self.smooth_vel)
            dataset.append(m)

            #
            name = f'{self.exp_date}_{self.animal_id}'
            self.names.append(name)

        self.plot_batch(dataset,
                        plot_velocity_line,
                        with_multi_args=False,
                        show_all_trials=False,
                        with_fill_between=False)

    # ================ #
    # Peri-Reward Lick #
    # ================ #

    LickDType = list[tuple[list[np.ndarray], np.ndarray, np.ndarray]]
    """list of args from ``peri_reward_raster_hist(*arg)``"""

    @dispatch('peri_reward_lick')
    def plot_peri_reward_lick_batch(self):
        """
        Plot the peri-reward lick histogram in batch dataset

        **Example**

        <MODULE PATH>
        -D 210315,210401 \
        -A YW006,YW008 \
        --type peri_reward_lick \
        -t 40,80

        :return:
        """
        dataset: BehaviorBatchPlotOptions.LickDType = []

        event_source = self.lick_event_source
        for _ in self.foreach_dataset(lick_thres=try_float_type):  # foreach lick

            riglog = self.load_riglog_data()
            self.offset_time = self._check_offset(self.exp_date)  # auto offset check
            tracker = self.load_lick_tracker()

            #
            if event_source == 'facecam':
                t = tracker.prob_to_event().time
            elif event_source == 'lickmeter':
                t = tracker.electrical_timestamp
            else:
                raise ValueError('')
            res = peri_reward_raster_hist(t, riglog.reward_event.time, self.psth_sec)
            dataset.append(res)

            #
            name = f'{self.exp_date}_{self.animal_id}'
            self.names.append(name)

        self._plot_lick_percentage(dataset)

    def _check_offset(self, exp_date: str) -> bool:
        date = datetime.strptime(exp_date, "%y%m%d").date()
        if date <= self.LABCAM_OFFSET_ISSUE_DATE:
            return True
        else:
            return False

    def _plot_lick_percentage(self, dataset: LickDType):
        """
        :param dataset: ``LickDType`` from ``peri_reward_raster_hist()``
        :return:
        """
        global edg

        lick_hist = []
        for it in dataset:
            lick_hist.append(it[1])
            edg = it[2]

        x = edg[1:]
        with plot_figure(None) as ax:
            hs = np.array(lick_hist)  # (nD, nbins), nD: number of data
            for i, h in enumerate(hs):
                ax.plot(x, h, label=self.names[i], color='gray', alpha=0.7)

            avg_h = np.mean(hs, axis=0)
            sem_h = sem(hs, axis=0)
            ax.plot(x, avg_h, color='pink')
            ax.fill_between(x,
                            avg_h + sem_h,
                            avg_h - sem_h,
                            color='gray' if self.fix_color else None,
                            alpha=0.4)

            ax.legend()
            ax.set_xlim(-self.psth_sec, self.psth_sec)
            ax.set(xlabel='time relative to reward time(s)', ylabel='lick percentage(%)')


if __name__ == '__main__':
    BehaviorBatchPlotOptions().main()
