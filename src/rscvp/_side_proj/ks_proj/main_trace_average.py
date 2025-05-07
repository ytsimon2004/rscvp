import numpy as np

from argclz import AbstractParser
from neuralib.imaging.suite2p import Suite2PResult
from neuralib.plot import plot_figure, ax_merge
from neuralib.plot.colormap import insert_colorbar
from rscvp._side_proj.ks_proj.main_pupil_visual import ApplyVisualPupilOptions
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.visual.main_tuning import ApplyVisualActOptions
from rscvp.visual.util_cache import VisualTuningResult


class VisualTuningAvgOptions(AbstractParser, ApplyVisualActOptions, ApplyVisualPupilOptions, SelectionOptions):
    DESCRIPTION = 'Do the trial averaged for different visual stimuli pattern based on the cached result'

    s2p: Suite2PResult
    act_result: VisualTuningResult
    activity: np.ndarray  # calcium

    pupil_result: VisualTuningResult
    pupil: np.ndarray

    pre_selection = True
    not_circular_env = True

    def set_attrs(self):
        self.s2p = self.load_suite_2p()

        #
        mask = self.get_selected_neurons()
        self.act_result = (
            self.apply_visual_tuning_cache()
            .load_result()
            .with_mask(mask)
        )
        self.activity = self.act_result.dat

        #
        self.pupil_result = self.apply_visual_pupil_cache().load_result()
        self.pupil = self.pupil_result.dat

        #
        if self.act_result.pre_post != self.pupil_result.pre_post:
            raise RuntimeError('data cannot concat for plotting, due to different off duration')

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.set_attrs()
        self.plot_trial_averaged_heatmap()

    @property
    def time_range(self) -> tuple[float, float]:
        n_frames = self.activity.shape[1]
        return 0, (n_frames / self.s2p.fs)

    @property
    def n_neurons(self) -> int:
        return self.activity.shape[0]

    def plot_trial_averaged_heatmap(self, norm: bool = True,
                                    frame_to_time: bool = True,
                                    with_vspan: bool = False):
        """
        :param norm: do the normalization per neurons for the epoch in the cache.
                otherwise, would be the whole session normalization
        :param frame_to_time: x axis unit from frame to time domain
        :param with_vspan: if false, draw axvline (start with green, end with red)
        :return:
        """
        act = self.activity

        if norm:
            act /= np.max(act, axis=1, keepdims=True)
            vmax = 1
        else:
            vmax = None

        #
        with plot_figure(None, 12) as _ax:
            #
            ax0 = ax_merge(_ax)[:2]
            if frame_to_time:
                x = np.linspace(*self.time_range, num=len(self.pupil))
                ax0.plot(x, self.pupil)
            else:
                ax0.plot(self.pupil)

            ax0.set_title('pupil')
            ax0.xaxis.set_visible(False)
            ax0.spines['bottom'].set_visible(False)

            #
            ax1 = ax_merge(_ax)[2:4]
            avg = np.mean(act, axis=0)

            if frame_to_time:
                x = np.linspace(*self.time_range, num=len(avg))
                ax1.plot(x, avg)
            else:
                ax1.plot(avg)

            ax1.sharex(ax0)
            ax1.set_title(f'avg {self.act_result.signal_type}')
            ax1.xaxis.set_visible(False)
            ax1.spines['bottom'].set_visible(False)

            #
            ax2 = ax_merge(_ax)[4:]
            im = ax2.imshow(
                act,
                cmap="gray_r",
                vmin=0,
                vmax=vmax,
                aspect="auto",
                interpolation='none',
                extent=(*self.time_range, 0, self.n_neurons) if frame_to_time else None
            )
            ax2.sharex(ax1)

            cbar = insert_colorbar(ax2, im)
            cbar.ax.set_ylabel(f'{self.act_result.signal_type}')

            #
            for r in self.act_result.stim_index:
                start, end = r[0], r[-1]
                if frame_to_time:
                    start = start / self.s2p.fs
                    end = end / self.s2p.fs

                for ax in (ax0, ax1, ax2):
                    if with_vspan:
                        ax.axvspan(start, end, 0, 1, color='mistyrose', alpha=0.35)
                    else:
                        ax.axvline(start, color='g', ls='--', zorder=1, alpha=0.35)
                        ax.axvline(end, color='r', ls='--', zorder=1, alpha=0.35)


if __name__ == '__main__':
    VisualTuningAvgOptions().main()
