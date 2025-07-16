import matplotlib.pyplot as plt

from argclz import AbstractParser
from neuralib.plot import plot_figure, plot_peri_onset_1d
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli import StimpyOptions
from rscvp.util.position import load_interpolated_position


@publish_annotation('appendix', project='rscvp', caption='rev')
class StimRunningOptions(AbstractParser, StimpyOptions):
    DESCRIPTION = 'plot the running speed pre and post the visual stimulation (for multiple animal)'

    pre = 1
    stim = 3
    post = 4

    def run(self):
        with plot_figure(None) as ax:
            for i, _ in enumerate(self.foreach_dataset()):
                rig = self.load_riglog_data()
                t = rig.get_stimlog().stimulus_segment[:, 0]
                pos = load_interpolated_position(rig)

                plot_peri_onset_1d(t, pos.t, pos.v, pre=self.pre, post=self.post, ax=ax,
                                   label=f'{self.exp_date}_{self.animal_id}')
                ax.axvspan(0, 3, alpha=0.3, color='gray')
                ax.set(xlabel='time to vstim', ylabel='speed (cm/s)')

            plt.legend()


if __name__ == '__main__':
    StimRunningOptions().main()
