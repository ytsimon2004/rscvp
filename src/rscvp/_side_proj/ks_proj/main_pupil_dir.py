import numpy as np
from rscvp._side_proj.ks_proj.main_pupil_visual import ApplyVisualPupilOptions
from rscvp.visual.util_cache import VisualTuningResult

from argclz import AbstractParser
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation


@publish_annotation('appendix', project='ks project')
class VisualPatternPupilDirOptions(AbstractParser, ApplyVisualPupilOptions):
    DESCRIPTION = 'Plot the pupil size histogram across different direction'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        cache = self.apply_visual_pupil_cache()
        result = cache.load_result()
        self.plot_dir_histogram(result)

    @staticmethod
    def plot_dir_histogram(result: VisualTuningResult):
        dat = result.dat  # (F,)
        n_frames = result.n_frames

        pre = []  # (D, Fpre)
        stim = []  # (D, Fvis)
        post = []  # (D, Fpost)
        for i, st in enumerate(result.stim_index):
            v0, v1 = st[0], st[-1]

            start = i * n_frames
            end = start + n_frames

            pre.append(dat[start:v0])
            stim.append(dat[v0:v1])
            post.append(dat[v1: end])

        # avg across frames
        pre = np.mean(np.array(pre), axis=1)
        stim = np.mean(np.array(stim), axis=1)
        post = np.mean(np.array(post), axis=1)
        n_dir = len(pre)

        with plot_figure(None, 1, 3, sharey=True) as ax:
            x = np.linspace(0, 360, n_dir)
            ax[0].plot(x, pre)
            ax[1].plot(x, stim)
            ax[2].plot(x, post)

            for i in range(3):
                ax[i].set_aspect(1.0 / ax[i].get_data_ratio(), adjustable='box')
                ax[i].set_xticks([j * 90 for j in range(5)])
                ax[i].set(ylabel='pupil zscore', xlabel='direction')


if __name__ == '__main__':
    VisualPatternPupilDirOptions().main()
