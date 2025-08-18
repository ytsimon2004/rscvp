from typing import Literal, cast

import numpy as np
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

from argclz import argument, AbstractParser, float_tuple_type
from argclz.dispatch import Dispatch, dispatch
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.model.bayes_decoding.main_cache_bayes import ApplyBayesDecodeOptions, BayesDecodeData, BayesDecodeCache
from rscvp.model.bayes_decoding.util_plot import (
    plot_decoding_err_position,
    plot_confusion_scatter,
    plot_confusion_heatmap
)
from rscvp.spatial.util import sort_neuron
from rscvp.util.cli import CameraOptions, DataOutput, SQLDatabaseOptions, PlotOptions
from rscvp.util.database import BayesDecodeDB

__all__ = ['DecodeAnalysisOptions']


@publish_annotation('main', project='rscvp', figure='fig.3A-3C', caption='db usage', as_doc=True)
class DecodeAnalysisOptions(AbstractParser,
                            ApplyBayesDecodeOptions,
                            PlotOptions,
                            CameraOptions,
                            SQLDatabaseOptions,
                            Dispatch):
    DESCRIPTION = 'Decoding analysis based on the existing cache'

    analysis_type: Literal[
        'overview',
        'median_decode_error',
        'confusion_matrix',
        'position_bins_error'
    ] = argument(
        '--type', '--analysis-type',
        required=True,
        help='which dispatch analysis'
    )

    sorted_idx: np.ndarray | None = argument(
        '--sort-idx',
        metavar='FILE',
        default=None,
        help='sorting index array in certain lap',
    )

    plot_n_samples: int | None = argument(
        '--n-samples',
        type=int,
        default=None,
        help='number of samples to plot in overview',
    )

    plot_concat_time: bool = argument(
        '--plot-concat',
        help='plot concatenated time x-axis (only shows concatenated test trials)',
    )

    plot_lick: bool = argument(
        '--plot-lick',
        help='whether plot the licking rate below'
    )

    fr_norm: tuple[float, float] | None = argument(
        '--fr-norm',
        metavar='RANGE',
        type=float_tuple_type,
        default=None,
        help='fr percentile normalization'
    )

    # =================== #
    # overview class attr #
    # =================== #

    def post_parsing(self):
        if self.plot_concat_time and self.plot_lick:
            raise RuntimeError('plot lick mode cannot shown as concat time')

        if self.plot_n_samples is not None and self.plot_lick:
            raise RuntimeError('plot lick mode cannot do cut since different lengths of x in t domain')

        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.set_background()

    def run(self):
        self.post_parsing()
        cache = self.apply_bayes_cache(version=self.cache_version)
        output_info = self.get_data_output('bayes_decode')

        self.invoke_command(self.analysis_type, cache, output_info)

    def get_time(self, cache: BayesDecodeCache) -> np.ndarray:
        """Get time in sec"""
        if self.plot_concat_time:
            nbins = len(cache)  # number of temporal bins
            fs = self.load_suite_2p().fs
            return np.linspace(0, nbins / fs, num=nbins)
        else:
            return cache.fr_time

    def get_lick_rate(self) -> tuple[np.ndarray, np.ndarray]:
        """Get lick time and lick rate"""
        t, s = self.get_lick_event()
        sig_t, sig = (
            self.apply_bayes_cache(version=self.cache_version)
            .get_signal(self, time=t, value=s)
        )

        return sig_t, sig

    # ======== #
    # Overview #
    # ======== #

    @dispatch('overview')
    def plot_decoding_overview(self, cache: BayesDecodeCache,
                               output: DataOutput):
        """
        plot the decoding error and actual position in running epoch
        1. fr (T, N)
        2. posterior (T, X)
        3. decoding position (T,) and actual position (T,)
        4. decoding error  (T,)
        5. lick_rate (T,)
        """

        time = self.get_time(cache)
        result = cache.load_result()
        fr = result.fr
        rate_map = result.rate_map
        posterior = result.posterior
        actual_pos = result.actual_position
        predicted_pos = result.predicted_position
        decoding_err = result.decode_error

        if self.plot_n_samples is not None:
            end = self.plot_n_samples
            time = time[:end]
            fr = fr[:end]
            posterior = posterior[:end, :]
            predicted_pos = predicted_pos[:end]
            actual_pos = actual_pos[:end]
            decoding_err = decoding_err[:end]

        output_file = output.summary_figure_output(
            self.signal_type,
            self.session,
            f'cut{int(self.plot_n_samples)}s' if self.plot_n_samples is not None else None,
            f'{self.cv_info}',
            f'#{self.cache_version}'
        )

        nrow = 5 if self.plot_lick else 4

        with plot_figure(output_file, nrow, 1, figsize=(15, 8)) as _ax:
            ax = _ax[0]
            self.plot_firing_rate(ax, time, fr, rate_map, self.fr_norm)
            msg = self.selection_info()
            msg += f'\n{self.cv_info}'
            ax.set_title(msg, loc='right')

            ax = _ax[1]
            self.plot_posterior_probability(ax, time, posterior)
            ax.sharex(_ax[0])

            ax = _ax[2]
            self.plot_decode_actual_position(ax, time, predicted_pos, actual_pos)
            ax.sharex(_ax[0])

            ax = _ax[3]
            self.plot_decoding_err(ax, time, decoding_err)
            ax.sharex(_ax[0])

            if self.plot_lick:
                lt, lr = self.get_lick_rate()
                ax = _ax[4]
                self.plot_lick_rate(ax, lt, lr)
                ax.sharex(_ax[0])

    def plot_firing_rate(self, ax: Axes,
                         time: np.ndarray,
                         fr: np.ndarray,
                         rate_map: np.ndarray,
                         percentile_norm: tuple[float, float] | None = None):
        """heatmap for sorted firing rate of all cells"""
        sort_idx = sort_neuron(rate_map.T) if self.sorted_idx is None else self.sorted_idx
        fr = fr[:, sort_idx]  # (T, N)

        if percentile_norm is not None:
            lower, upper = percentile_norm
            lp = np.percentile(fr, lower, axis=0)  # (N,)
            up = np.percentile(fr, upper, axis=0)  # (N,)
            fr = np.clip((fr - lp) / (up - lp), 0, 1)

        ax.imshow(fr.T,
                  aspect='auto',
                  cmap=self.cmap_color,
                  # interpolation='none',
                  origin='lower',
                  extent=(0, np.max(time), 0, fr.shape[1]))
        ax.set(ylabel='# neurons')

    def plot_posterior_probability(self, ax: Axes, time: np.ndarray, posterior: np.ndarray):
        """plot posterior probability"""
        ax.imshow(posterior.T,
                  aspect='auto',
                  cmap=self.cmap_color,
                  interpolation='none',
                  origin='lower',
                  extent=(0, np.max(time), 0, posterior.shape[1]))

        ax.set(ylabel='n_spatial_bins #')

    def plot_decode_actual_position(self, ax: Axes,
                                    time: np.ndarray,
                                    predicted_pos: np.ndarray,
                                    actual_pos: np.ndarray,
                                    **kwargs):

        if self.plot_concat_time:
            ax.plot(time, predicted_pos, color='g', label='decoded', alpha=0.6, **kwargs)
            ax.plot(time, actual_pos, color=self.line_color, label='actual position', alpha=0.4, **kwargs)
        else:
            ax.plot(time, predicted_pos, 'r.', label='decoded', alpha=0.6, **kwargs)
            ax.plot(time, actual_pos, '.', color=self.line_color, label='actual position', alpha=0.2, **kwargs)

        ax.set(ylabel='cm')
        ax.legend()

    def plot_decoding_err(self, ax: Axes,
                          time: np.ndarray,
                          decode_err: np.ndarray):
        """
        plot decoding error as a function of temporal bins

        :param ax:
        :param time:
        :param decode_err
        :return:
        """
        ax.plot(time, decode_err, color=self.line_color, alpha=0.3, label='frame-wise')
        smooth_err = gaussian_filter1d(decode_err, 10)
        ax.plot(time, smooth_err, color=self.line_color, label='smooth')
        ax.set(xlabel='time(sec)', ylabel='decoding error (cm)', ylim=(0, 70))
        ax.legend()

    @staticmethod
    def plot_lick_rate(ax: Axes,
                       lick_time: np.ndarray,
                       lick_rate: np.ndarray,
                       **kwargs):
        """plot lick rate as a function of temporal bins"""
        ax.plot(lick_time, lick_rate, color='b', label='decoded', alpha=0.6, **kwargs)
        ax.set(xlabel='time(sec)', ylabel='lick rate(Hz)')

    # ===================== #
    # Median Decoding Error #
    # ===================== #

    def populate_database(self, result: BayesDecodeData):
        # noinspection PyTypeChecker
        db = BayesDecodeDB(
            date=self.exp_date,
            animal=self.animal_id,
            rec=self.daq_type,
            user=self.username,
            optic=f'{self.plane_index}' if self.plane_index is not None else 'all',
            region=(self.get_primary_key_field('region') if self.rec_region is None else self.rec_region),
            pair_wise_group=self.get_primary_key_field('pair_wise_group'),
            n_neurons=result.n_neurons,
            spatial_bins=result.spatial_bin_size,
            temporal_bins=result.temporal_bin_size,
            median_decode_error=np.median(result.decode_error),
            cross_validation=self.cross_validation,
            update_time=self.cur_time
        )

        print('NEW', db)
        if self.db_commit:
            self.add_data(db)
        else:
            print('please use --commit to add data to database')

    @dispatch('median_decode_error')
    def plot_median_decoding_err(self, cache: BayesDecodeCache, output: DataOutput):
        """plot decoding median value"""
        result = cache.load_result()

        if self.is_vop_protocol:
            self.populate_database(result)

        output_file = output.summary_figure_output(self.session, 'median_decoding_error', f'#{self.cache_version}')

        if self.plane_index is not None:
            decode_err = result.decode_error
            n = result.n_neurons
        else:
            cache = self.apply_bayes_cache()
            decode_err = cache.decode_error
            n = cache.rate_map.shape[1]

        with plot_figure(output_file) as ax:
            median_value = cast(float, np.median(decode_err))
            y = np.linspace(0, 1, num=len(decode_err))
            xmin = -5
            xmax = self.belt_length / 2

            ax.plot(np.sort(decode_err), y, color='k')
            ax.set(xlim=(xmin, xmax), xlabel='decoding error (cm)', ylabel='cumulative probability')

            ax.axvline(median_value, color='r', ls='--', ymax=0.5)
            xperc = (median_value + abs(xmin)) / (xmax - xmin)
            ax.axhline(0.5, color='r', ls='--', xmax=xperc)

            ax.set_title(f'median_error: {np.round(median_value, 2)}\n n={n}')
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    # ================ #
    # Confusion Matrix #
    # ================ #

    @dispatch('confusion_matrix')
    def plot_confusion_matrix(self, cache: BayesDecodeCache, output: DataOutput):
        """
        Plot the decoding probability confusion matrix (heatmap & scatter)

        :param cache: ``BayesDecodeCache``
        :param output: ``DataOutput``
        :return:
        """
        result = cache.load_result()
        output_file = output.summary_figure_output(self.session, 'confusion_matrix', f'#{self.cache_version}')

        with plot_figure(output_file, 1, 2, default_style=False) as ax:
            act = result.actual_position
            pred = result.predicted_position

            plot_args = {
                'actual_position': act,
                'predicted_position': pred,
                'total_length': self.belt_length,
                'cue_loc': self.cue_loc
            }
            plot_confusion_heatmap(ax=ax[0], **plot_args)
            plot_confusion_scatter(ax=ax[1], **plot_args)

    # ============================== #
    # Position-Binned Decoding Error #
    # ============================== #

    @dispatch('position_bins_error')
    def plot_decoding_err_position(self, cache: BayesDecodeCache, output: DataOutput):
        """plot decoding error as a function of position bins"""
        result = cache.load_result()
        output = output.summary_figure_output(self.session, 'line', f'#{self.cache_version}')

        with plot_figure(output) as ax:
            plot_decoding_err_position(result.binned_error_median,
                                       result.binned_error_sem,
                                       total_length=self.belt_length,
                                       window=self.pos_bins, ax=ax)


if __name__ == '__main__':
    DecodeAnalysisOptions().main()
