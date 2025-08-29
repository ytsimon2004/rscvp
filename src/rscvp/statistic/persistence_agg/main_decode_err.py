from typing import Final, Literal

import numpy as np
from scipy.stats import sem

from argclz import try_int_type, argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure, grid_subplots
from neuralib.typing import flatten_arraylike
from neuralib.util.verbose import publish_annotation
from rscvp.model.bayes_decoding.main_cache_bayes import ApplyBayesDecodeOptions, BayesDecodeCache
from rscvp.model.bayes_decoding.util_plot import (
    plot_decoding_err_position,
    plot_confusion_scatter,
    plot_confusion_heatmap
)
from rscvp.statistic.persistence_agg.core import AbstractPersistenceAgg, GroupInt, GroupName

__all__ = ['BayesDecodePersistenceAgg']


@publish_annotation('main', project='rscvp', figure='fig.3C (batch mode)', caption='@dispatch(confusion_matrix)', as_doc=True)
class BayesDecodePersistenceAgg(AbstractPersistenceAgg, ApplyBayesDecodeOptions, Dispatch):
    DESCRIPTION = (
        'Decoding error as a function of position bins from multiple dataset (used mainly for pair comparison,'
        'Either run in foreach or group mode'
    )

    analysis_type: Literal['confusion_matrix', 'position_bins_error'] = argument(
        '--type', '--analysis-type',
        required=True,
        help='which analysis type'
    )

    field = dict(plane_index=try_int_type)
    cache_version = 0

    GROUP_REPR: Final[dict[GroupInt, GroupName]] = AbstractPersistenceAgg.GROUP_REPR

    def run(self):
        self.post_parsing()
        caches = self.get_cache_list()
        data = self.get_cache_data(caches)
        self.invoke_command(self.analysis_type, data)

    def get_cache_list(self) -> list[BayesDecodeCache]:
        ret = []
        for i, _ in enumerate(self.foreach_dataset(**self.field)):
            ret.append(self.apply_bayes_cache(version=self.cache_version))

        return ret

    def get_cache_data(self, cache_list: list[BayesDecodeCache]) -> list[np.ndarray]:
        return self.invoke_group_command('cache2data', self.analysis_type, cache_list)

    def get_label(self, i: int) -> str:
        if self.group_mode:
            return f'{self.unique_groups[i]}'
        else:
            return f'{self.exp_list[i]}_{self.animal_list[i]}'

    def plot(self, data: list[np.ndarray]):
        raise RuntimeError('use dispatch implementation instead')

    # =================== #
    # Position Bins Error #
    # =================== #

    @dispatch('position_bins_error', group='cache2data')
    def _get_cache_data_pos_bins(self, cache_list: list[BayesDecodeCache]) -> list[np.ndarray]:
        ret = []
        if self.group_mode:
            for i, group in enumerate(self.unique_groups):
                indices = (self.data_grouping['group'] == group).arg_true().to_list()

                act_group = []  # len=G, (B,)
                for j in indices:
                    dat = np.nanmean(cache_list[j].binned_decode_error, axis=0)  # (L, B) -> (B,)
                    act_group.append(dat)

                act_mean = np.mean(act_group, axis=0)  # (B,)
                act_sem = sem(act_group, axis=0)  # (B,)

                dat = np.vstack([act_mean, act_sem])  # (2, B)
                ret.append(dat)
        else:
            for cache in cache_list:
                # trial avg (L, B) -> (B,)
                act_mean = np.nanmean(cache.binned_decode_error, axis=0)
                act_sem = sem(cache.binned_decode_error, axis=0)
                dat = np.vstack([act_mean, act_sem]).T
                ret.append(dat)  # for grid plot (B, 2)

        return ret

    @dispatch('position_bins_error')
    def position_bins_error(self, data: list[np.ndarray]):
        """Plot the position bins as a function of position bins.
        If group mode, show the diff
        """
        if self.group_mode:
            self._position_bins_error_group(data)
        else:
            self._position_bins_error_foreach(data)

    def _position_bins_error_group(self, data: list[np.ndarray]):
        with plot_figure(None, 2, 1) as ax:
            for i, dat in enumerate(data):
                plot_decoding_err_position(dat[0], dat[1],
                                           total_length=self.belt_length,
                                           window=self.pos_bins,
                                           color=None,
                                           label=self.get_label(i),
                                           ax=ax[0])

            ax[0].legend()

            x = np.linspace(0, self.belt_length, self.pos_bins)
            ax[1].plot(x, np.abs(data[0][0] - data[1][0]), 'ko', markerfacecolor='none', markersize=6)

    def _position_bins_error_foreach(self, data: list[np.ndarray]):
        grid_subplots(
            data,
            images_per_row=3,
            plot_func=plot_decoding_err_position,
            dtype='xy',
            hide_axis=False,
            figsize=(8, 8),
            total_length=self.belt_length,
            window=self.pos_bins,
            title=[f'{self.exp_list[i]}_{self.animal_list[i]}' for i in range(len(data))]
        )

    # ================ #
    # Confusion Matrix #
    # ================ #

    @dispatch('confusion_matrix', group='cache2data')
    def _get_cache_data_confusion_mtx(self, cache_list: list[BayesDecodeCache]) -> list[np.ndarray]:
        ret = []

        if self.group_mode:
            for i, group in enumerate(self.unique_groups):
                indices = (self.data_grouping['group'] == group).arg_true().to_list()

                act = []
                pred = []
                for j in indices:
                    act.append(cache_list[j].actual_position)
                    pred.append(cache_list[j].predicted_position)

                act = np.array(flatten_arraylike(act))  # (T x n, ) n: number of data in each group
                pred = np.array(flatten_arraylike(pred))  # (T x n, )

                ret.append(np.vstack([act, pred]))

        else:
            for cache in cache_list:
                act = cache.actual_position
                pred = cache.predicted_position
                ret.append(np.vstack([act, pred]).T)

        return ret

    @dispatch('confusion_matrix')
    def plot_confusion_matrix(self, data: list[np.ndarray]):
        """
        **Plot batch scatter confusion matrix in grid subplots (non group mode)**
        OR
        **Plot batch heatmap confusion matrix in grid subplots (group mode)**
        """
        if self.group_mode:
            self._plot_confusion_matrix_group(data)
        else:
            self._plot_confusion_matrix_foreach(data)

    def _plot_confusion_matrix_group(self, data: list[np.ndarray]):
        with plot_figure(None, 1, 2, tight_layout=False) as ax:
            for group, dat in enumerate(data):
                plot_confusion_heatmap(dat[0], dat[1],
                                       nbins=30,
                                       total_length=self.belt_length,
                                       landmarks=self.track_landmarks,
                                       ax=ax[group])
                ax[group].set_title(self.get_label(group))

    def _plot_confusion_matrix_foreach(self, data: list[np.ndarray]):
        grid_subplots(
            data,
            images_per_row=2,
            plot_func=plot_confusion_scatter,
            dtype='xy',
            hide_axis=False,
            figsize=(6, 6),
            total_length=self.belt_length,
            landmarks=self.track_landmarks,
            title=[f'{self.exp_list[i]}_{self.animal_list[i]}' for i in range(len(data))]
        )


if __name__ == '__main__':
    BayesDecodePersistenceAgg().main()
