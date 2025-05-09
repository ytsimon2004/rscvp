import argparse
from typing import Any

import numpy as np
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.spatial.main_cache_sortidx import ApplySortIdxOptions
from rscvp.spatial.util import sort_neuron, normalized_trial_avg
from rscvp.spatial.util_plot import plot_sorted_trial_averaged_heatmap, plot_fraction_active
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.util_trials import TrialSelection

from argclz import AbstractParser, argument, int_tuple_type
from neuralib.imaging.suite2p import SIGNAL_TYPE
from neuralib.plot import plot_figure, ax_merge

__all__ = ['CPBeltSortOptions']


class CPBeltSortOptions(AbstractParser, ApplyPosBinActOptions, ApplySortIdxOptions):
    DESCRIPTION = 'plot the sorted trial-average calcium activities of population neurons along the 1d environment'
    EPILOG = """
    Example:
    python -m rscvp.spatial.main_belt_sort
    -t light                 # save idx (sort by light)
    -t dark --sort light     # use dark session activity (sort by light cache)
    """

    plot_trial: tuple[int, int] | None = argument(
        '--plot-trial',
        type=int_tuple_type,
        default=None,
        help='trial(lap) numbers for plotting, if None, do the trial average',
    )

    signal_type: SIGNAL_TYPE = 'df_f'
    pre_selection = True
    reuse_output = True

    @classmethod
    def new_parser(cls, **kwargs) -> argparse.ArgumentParser:
        return super().new_parser(
            formatter_class=argparse.RawTextHelpFormatter,
            **kwargs
        )

    def post_parsing(self):
        if self.session is not None:
            raise ValueError('opt.use_trial instead to avoid misunderstand')

        if self.use_trial is not None:
            if isinstance(self.use_trial, str):
                pass
            else:
                # (int, int)
                if len(self.use_trial) != 2:
                    raise ValueError(f'illegal use-trial, len != 2 : {self.use_trial}')
                if not all([isinstance(it, int) for it in self.use_trial]):
                    raise TypeError(f'illegal use-trial, not int : {self.use_trial}')

        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

    def run(self):
        self.post_parsing()

        output_info = self.get_data_output('sa')
        self.calactivity_belt_sorted(output_info)

    # ========================= #
    # Signal processing methods #
    # ========================= #

    def signal_all(self):
        """get binned data (N', L, B)"""
        signal_all = self._signal_all()

        # trial selection (use_trial instead of opt.session)
        if isinstance(self.use_trial, str):
            rig = self.load_riglog_data()
            indices = TrialSelection(rig, self.use_trial).get_time_profile().trial_range
            trial_range = slice(*indices)
        elif isinstance(self.use_trial, tuple):
            trial_range = slice(*self.use_trial)
        else:
            raise TypeError(f'use_trial: {type(self.use_trial)} type error')

        return signal_all[:, trial_range, :]  # (N', L', B)

    _signal_all_cache = None

    def _signal_all(self) -> np.ndarray:
        """get cached binned data (N, L, B)"""
        if self._signal_all_cache is None:
            self._signal_all_cache = self.apply_binned_act_cache().occ_activity

        # neuron selection
        cell_mask = self.get_selected_neurons()

        return self._signal_all_cache[cell_mask]

    _sorted_index = None

    @property
    def sorted_index(self) -> np.ndarray:
        if self._sorted_index is None:
            self._sorted_index = self.apply_sort_idx_cache(error_when_missing=False).sort_idx

        return self._sorted_index

    # ================ #
    # Plotting methods #
    # ================ #

    @property
    def default_fig_kwargs(self) -> dict[str, Any]:
        return dict(
            total_length=self.belt_length,
            cue_loc=self.cue_loc,
            n_selected_neurons=self.n_selected_neurons,
            n_total_neurons=self.n_total_neurons
        )

    def calactivity_belt_sorted(self, output: DataOutput):
        if self.plot_trial is None:
            index = self.sorted_index

            if index is None:
                index = slice(None, None)

            signal_all = self.signal_all()[index]
            signal = normalized_trial_avg(signal_all)

            self.plot_trial_average(signal, output)
        else:
            self.plot_selected_trials(self.signal_all(), self.plot_trial, output)

    def plot_trial_average(self, signal: np.ndarray, output: DataOutput):
        """
        Plot the sorted heatplot (trial average) in position domain (x)

        :param signal: (N, B)
        :param output:
        :return:
        """
        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            self.pc_selection if self.pc_selection is not None else None,
            self.signal_type,
            self.use_trial
        )

        with plot_figure(output_file, 3, 1, figsize=(6, 12)) as _ax:
            ax = ax_merge(_ax)[:2]
            plot_sorted_trial_averaged_heatmap(signal, ax=ax, **self.default_fig_kwargs)

            ax = ax_merge(_ax)[2:]
            plot_fraction_active(ax, signal, belt_length=self.belt_length, cue_loc=self.cue_loc)

    def plot_selected_trials(self, signal: np.ndarray, trial_range: tuple[int, int], output: DataOutput):
        """
        Plot the sorted heatplot every lap in position domain (x)

        :param signal: (N, L, B)
        :param trial_range:
        :param output
        :return:
        """
        signal = signal[:, trial_range[0]:trial_range[1], :]
        n_trials = signal.shape[1]

        # use FIRST PLOTTED TRIAL as neuron sort_idx
        sorted_index = sort_neuron(signal[:, self.plot_trial[0], :])
        signal = signal[sorted_index]

        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            self.pc_selection if self.pc_selection is not None else None,
            self.signal_type,
            self.use_trial,
            f'by{self.plot_trial}'
        )

        with plot_figure(output_file, 3, n_trials, gridspec_kw={'wspace': 0}) as _ax:
            for i in range(n_trials):

                first_panel = i == 0
                last_panel = i == n_trials - 1

                #
                ax = ax_merge(_ax)[:2, i]
                if last_panel:
                    plot_sorted_trial_averaged_heatmap(signal[:, i, :], cmap='hot', ax=ax, **self.default_fig_kwargs)
                else:
                    plot_sorted_trial_averaged_heatmap(signal[:, i, :], cmap='hot', ax=ax, **self.default_fig_kwargs)

                if first_panel:
                    ax.set_ylabel('Neurons #')
                else:
                    ax.axes.yaxis.set_visible(False)

                ax.set_xticks([150])

                #
                ax = ax_merge(_ax)[2:, i]
                plot_fraction_active(ax, signal[:, i, :], belt_length=self.belt_length, cue_loc=self.cue_loc)

                if not first_panel:
                    ax.axes.yaxis.set_visible(False)
                    ax.set_xlabel('')

                ax.set_xticks([150])


if __name__ == '__main__':
    CPBeltSortOptions().main()
