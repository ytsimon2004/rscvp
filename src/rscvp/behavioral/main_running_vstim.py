import matplotlib.pyplot as plt
import numpy as np

from argclz import AbstractParser, argument
from neuralib.plot import plot_figure
from neuralib.plot.psth import peri_onset_1d, plot_peri_onset_1d
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import TreadmillOptions


@publish_annotation('appendix', project='rscvp', caption='rev')
class StimRunningOptions(AbstractParser, TreadmillOptions):
    DESCRIPTION = 'plot the running speed pre and post the visual stimulation (for multiple animal)'

    collapse: bool = argument(
        '--collapse',
        help='plot the average speed across all datasets, otherwise plot for each dataset'
    )

    pre = 1
    stim = 3
    post = 4
    invalid_riglog_cache = True

    def run(self):
        if self.collapse:
            self.plot_collapsing()
        else:
            self.plot_foreach()

    def plot_foreach(self):
        with plot_figure(None) as ax:
            for i, _ in enumerate(self.foreach_dataset()):
                rig = self.load_riglog_data()
                t = rig.get_stimlog().stimulus_segment[:, 0]
                pos = self.load_position()

                plot_peri_onset_1d(t, pos.t, pos.v, pre=self.pre, post=self.post, ax=ax,
                                   label=f'{self.exp_date}_{self.animal_id}')
                ax.axvspan(0, 3, alpha=0.3, color='gray')
                ax.set(xlabel='time to vstim', ylabel='speed (cm/s)')

            plt.legend()

    def plot_collapsing(self):
        all_data = []
        dataset_labels = []
        bins = 100

        for i, _ in enumerate(self.foreach_dataset()):
            rig = self.load_riglog_data()
            t = rig.get_stimlog().stimulus_segment[:, 0]
            pos = self.load_position()

            peri_data = peri_onset_1d(
                event_time=t,
                act_time=pos.t,
                act=pos.v,
                bins=bins,
                pre=self.pre,
                post=self.post
            )

            # average across trials within this dataset
            speed_data = np.mean(peri_data, axis=0)
            all_data.append(speed_data)
            dataset_labels.append(f'{self.exp_date}_{self.animal_id}')

        time = np.linspace(-self.pre, self.post, bins)
        with plot_figure(None) as ax:
            if all_data:
                all_data_array = np.array(all_data)
                mean_speed = np.mean(all_data_array, axis=0)  # average across all datasets
                sem_speed = np.std(all_data_array, axis=0) / np.sqrt(len(all_data_array))

                ax.plot(time, mean_speed, label=f'Average (n={len(all_data)})', linewidth=2)
                ax.fill_between(time, mean_speed - sem_speed, mean_speed + sem_speed, alpha=0.3)

                ax.axvspan(0, self.stim, alpha=0.3, color='gray')
                ax.set(xlabel='time to vstim (s)', ylabel='speed (cm/s)')
                plt.legend()


if __name__ == '__main__':
    StimRunningOptions().main()
