import collections

import numpy as np

from argclz import AbstractParser
from neuralib.plot import plot_figure
from rscvp.visual.main_running import RunningVisualOptions
from rscvp.visual.main_tuning import ApplyVisualActOptions
from stimpyp import RiglogData


class LocomotionDirOptions(AbstractParser, ApplyVisualActOptions):
    DESCRIPTION = 'Locomotion analysis for different direction visual stimulation'

    signal_type = 'speed'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        rig = self.load_riglog_data()
        cache = self.apply_visual_tuning_cache(RunningVisualOptions, imaging=False)
        result = cache.load_result()

        dy = collections.defaultdict(list)
        for i, it in enumerate(result.stim_pattern):
            dire = it[0]
            dy[dire].append(result.signal[i])

        st = self.get_stim_time(rig)
        pre, post = result.pre_post

        with plot_figure(None, 1, len(dy), sharex=True, sharey=True, set_square=True) as ax:
            for i, (dire, sig) in enumerate(dy.items()):
                value = np.vstack(sig).mean(axis=0)  # dir-wise average
                ax[i].plot(value)
                n = len(value)
                n0 = pre / (pre + st + post) * n
                n1 = (pre + st) / (pre + st + post) * n

                ax[i].axvspan(n0, n1, alpha=0.3, color='gray')
                ax[i].set_title(dire)

    @staticmethod
    def get_stim_time(rig: RiglogData) -> int:
        """only used in const int stimulus"""
        seg = rig.get_stimlog().stimulus_segment
        return np.diff(seg, axis=1).ravel().mean().astype(int)


if __name__ == '__main__':
    LocomotionDirOptions().main()
