import attrs
import numpy as np
import rastermap.utils
from functools import cached_property
from matplotlib.axes import Axes
from rastermap import Rastermap
from scipy.interpolate import interp1d
from typing import Literal, TYPE_CHECKING

from argclz import AbstractParser, argument
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure, ax_merge
from neuralib.rastermap import RasterMapResult, RasterOptions
from neuralib.util.gpu import check_mps_available
from neuralib.util.unstable import unstable
from neuralib.util.verbose import fprint
from rscvp.model.rastermap.rastermap_2p_cache import RasterMap2PCacheBuilder, RasterInput2P
from rscvp.util.cli import CameraOptions, DataOutput, RasterMapOptions, SBXOptions, SelectionOptions

if TYPE_CHECKING:
    from neuropop.nn_prediction import PredictionNetwork
    import torch

__all__ = ['RunRasterMap2POptions']

DEFAULT_2P_RASTER_OPT: RasterOptions = {
    'n_clusters': 100,
    'n_PCs': 128,
    'locality': 0.75,
    'time_lag_window': 5,
    'grid_upsample': 10,
}


@unstable()
@attrs.frozen(repr=False)
class NeuralNetworkTrainResult:
    y_pred_all: np.ndarray
    """(T',C), T' represent number of test time-points"""
    ve_all: float
    """TODO"""
    itest: np.ndarray
    """(T',) Index array for testing dataset time"""
    test_time: np.ndarray
    """(T',)"""
    ev: np.ndarray
    """(C,)"""

    def __attrs_post_init__(self):
        fprint(f'{self}')

    def __repr__(self):
        ret = []
        # noinspection PyTypeChecker
        for k, v in attrs.asdict(self).items():
            if isinstance(v, np.ndarray):
                ret.append(f'{k}: {v.shape}')

        return '\n'.join(ret)


class RunRasterMap2POptions(AbstractParser,
                            SelectionOptions,
                            SBXOptions,
                            CameraOptions,
                            RasterMapOptions):
    DESCRIPTION = 'Run rastermap with 2P imaging data'

    dispatch_analysis: Literal['sorting', 'orofacial'] = argument(
        '--analysis',
        required=True,
        help='which analysis type'
    )

    dat: RasterInput2P
    reuse_output = True

    def post_parsing(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        if self.session is None:
            self.session = 'all'

    def run(self):
        self.post_parsing()
        self.get_rastermap_input()
        raster = self.run_rastermap()
        output_info = self.get_data_output('rastermap')

        if self.dispatch_analysis == 'sorting':
            self.plot_rastermap_sort(raster, trange=self.time_range, output=output_info)
            self.plot_raster_soma_position(raster, output=output_info)

        elif self.dispatch_analysis == 'orofacial':
            self.track_type = 'keypoints'
            fmap = self.load_facemap_result()
            behavior = fmap.get(fmap.keypoints).select('x', 'y').dataframe().to_numpy()  # TODO do some selection
            behavior = behavior[:, :self.camera_time, :]  # TODO check
            nn_result = self.train_neural_network(behavior, fmap.frame_time)
            self.plot_behavioral_prediction(raster, nn_result, output=output_info)
        else:
            raise ValueError('')

    def get_rastermap_input(self):
        self.dat = get_options_and_cache(RasterMap2PCacheBuilder, self).load_result()

    def run_rastermap(self, **kwargs) -> RasterMapResult:
        """
        :param force_compute
        :param kwargs:
        :return:
        """

        ops = DEFAULT_2P_RASTER_OPT

        if not self.rastermap_result_cache(self.plane_index).exists() or self.force_compute:
            model = Rastermap(
                n_clusters=ops['n_clusters'],
                n_PCs=ops['n_PCs'],
                locality=ops['locality'],
                time_lag_window=ops['time_lag_window'],
                grid_upsample=ops['grid_upsample'],
                **kwargs
            ).fit(self.dat.neural_activity)

            embedding = model.embedding
            isort = model.isort

            # bin over neuron axis
            sn = rastermap.utils.bin1d(self.dat.neural_activity[isort], bin_size=self.neuron_bins, axis=0)

            # For fit gui launch behavior
            filename = 'F.npy' if self.signal_type == 'df_f' else 'spks.npy'
            ret = RasterMapResult(
                filename=str(self.suite2p_directory / filename),
                save_path=str(self.rastermap_result_cache(self.plane_index)),
                isort=isort,
                embedding=embedding,
                ops=ops,
                user_clusters=[],
                super_neurons=sn
            )

            ret.save(self.rastermap_result_cache(self.plane_index))

        else:
            ret = RasterMapResult.load(self.rastermap_result_cache(self.plane_index))

        return ret

    # =========================== #
    # Run RasterMap visualization #
    # =========================== #

    def plot_rastermap_sort(self,
                            raster: RasterMapResult,
                            trange: tuple[int, int],
                            output: DataOutput):
        """

        :param raster:
        :param trange:
        :param output
        :return:
        """

        tmask = np.logical_and(trange[0] <= self.dat.image_time, self.dat.image_time <= trange[1])
        time = self.dat.image_time[tmask]

        output_file = output.summary_figure_output('2p', 'sorting', self.time_range)

        with plot_figure(output_file, 10, 20, dpi=200, gridspec_kw={'wspace': 1, 'hspace': 1}) as _ax:
            # position
            ax1: Axes = ax_merge(_ax)[0, :-1]
            pos = self.dat.position[tmask]
            ax1.plot(time, pos, color='gray')
            ax1.axis("off")
            ax1.set_title("position", color='gray')

            # running speed
            ax2: Axes = ax_merge(_ax)[1, :-1]
            ax2.plot(time, self.dat.velocity[tmask], color='gray')
            ax2.axis("off")
            ax2.set_title("running speed", color='gray')
            ax2.sharex(ax1)

            # pupil
            ax3: Axes = ax_merge(_ax)[2, :-1]
            if self.with_pupil:
                pupil = self.dat.pupil_area[tmask]
                ax3.plot(time, pupil, color='gray')
                ax3.axis("off")
                ax3.set_title("pupil", color='gray')
                ax3.sharex(ax1)
            else:
                ax3.axis('off')

            # superneuron activity
            ax4 = ax_merge(_ax)[3:, :-1]
            ax4.sharex(ax1)
            ax4.imshow(raster.super_neurons[:, tmask],
                       cmap="gray_r",
                       vmin=0,
                       vmax=0.8,
                       aspect="auto",
                       extent=(trange[0], trange[1], self.dat.n_neurons // self.neuron_bins, 0),
                       interpolation='none')
            ax4.set(xlabel="time(s)", ylabel='superneurons')

            # cue
            if self.with_cue:
                for c in self.dat.get_landmarks_index(tmask, cue_loc=self.track_landmarks):
                    ax4.axvline(time[c], color='g', linestyle='--', alpha=0.4)

            # visual stim
            if self.dat.visual_stim_start <= self.time_range[1]:
                for v in self.dat.visual_stim_trange(trange=trange):
                    ax4.axvspan(v[0], v[1], color='mistyrose', alpha=0.6)

            # disable
            ax5 = ax_merge(_ax)[:3, -1]
            ax5.axis('off')

            # color bar
            ax6 = ax_merge(_ax)[3:, -1]
            ax6.imshow(np.arange(0, raster.n_clusters)[:, np.newaxis], cmap="gist_ncar", aspect="auto")
            ax6.axis("off")

    def plot_raster_soma_position(self, raster: RasterMapResult,
                                  output: DataOutput = None):
        output_file = output.summary_figure_output('2p_soma')
        with plot_figure(output_file) as ax:
            ax.scatter(self.dat.y_pos,
                       self.dat.x_pos,
                       s=8, c=raster.embedding.flatten(), cmap="gist_ncar", alpha=0.25)
            ax.invert_yaxis()
            ax.set(xlabel='X position(mm)', ylabel='Y position(mm)')
            ax.set_aspect('equal')

    # ======================================== #
    # Neural activity prediction from behavior #
    # ======================================== #

    @cached_property
    def device(self) -> 'torch.device':
        import torch

        if torch.cuda.is_available():
            fprint('Process using cuda GPU')
            return torch.device('cuda')
        elif check_mps_available(backend='torch'):
            fprint('Process using mps GPU')
            return torch.device('mps')
        else:
            fprint(f'GPU not available, using CPU instead')
            return torch.device('cpu')

    _X_transformed = None
    _U = None

    def _decomposition(self,
                       neural_activity: np.ndarray,
                       n_components: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """

        :param neural_activity: (N, T)
        :param n_components: C, dimensionality reduction
        :return:
            X_transformed: (T, C)
            U: (N, C)
        """
        from sklearn.decomposition import TruncatedSVD

        if self._X_transformed is None or self._U is None:
            # the left singular vectors scaled by the singular values. (T, C)
            X_transformed = (TruncatedSVD(n_components=n_components)
                             .fit_transform(neural_activity.T))

            # compute the other singular vectors
            U = neural_activity @ (X_transformed / (X_transformed ** 2).sum(axis=0) ** 0.5)  # (N, C)
            U /= (U ** 2).sum(axis=0) ** 0.5

            self._X_transformed = X_transformed
            self._U = U

        return self._X_transformed, self._U

    def create_model(self,
                     behaviors: np.ndarray,
                     n_kp: int = 22) -> 'PredictionNetwork':
        """
        Create the neural network to fit

        :param behaviors: (nframes, ntypes). i.e., orofacial movement from facemap data
        :param n_kp: TODO number of keypoints
        :return:
        """
        from neuropop import nn_prediction

        X_transformed, _ = self._decomposition(self.dat.neural_activity)

        _, ntypes = behaviors.shape
        _, n_components = X_transformed.shape

        pred_model = nn_prediction.PredictionNetwork(
            n_in=ntypes,
            n_kp=n_kp,
            n_out=n_components
        )

        pred_model.to(self.device)
        print(pred_model)

        return pred_model

    def train_neural_network(self,
                             behaviors: np.ndarray,
                             behavioral_time: np.ndarray,
                             n_kp: int = 22,
                             interpolation: bool = True) -> NeuralNetworkTrainResult:
        """

        :param behaviors: (F, nType)
        :param behavioral_time: (F,) value in sec
        :param n_kp:
        :param interpolation: interpolate the behavior to neural activity time shape
                TODO not test, behavioral time cause the error in nn_prediction shape in neuropop
        :return: :class:`NeuralNetworkTrainResult`
                        itest: default test dataset with .25 fraction

        """

        X_transformed, U = self._decomposition(self.dat.neural_activity)

        # trimmed useless time
        tmask = behavioral_time <= self.dat.image_time[-1]

        model = self.create_model(behaviors[tmask], n_kp=n_kp)

        behaviors = behaviors[tmask, :]
        behavioral_time = behavioral_time[tmask]

        if interpolation:
            behaviors = interp1d(
                behavioral_time, behaviors,
                axis=0,
                bounds_error=False,
                fill_value='extrapolate'
            )(self.dat.image_time)

            behavioral_time = self.dat.image_time

        y_pred_all, ve_all, itest = model.train_model(
            X_dat=behaviors,
            Y_dat=X_transformed,  # (T, C)
            tcam_list=behavioral_time,
            tneural_list=self.dat.image_time,
            delay=-1,
            learning_rate=1e-3,
            n_iter=400,
            device=self.device,
            verbose=True
        )

        itest = itest.flatten()
        test_time = behavioral_time[itest]

        # variance explained per PC
        residual = ((y_pred_all - X_transformed[itest]) ** 2).sum(axis=0)
        ev = 1 - (residual / (X_transformed[itest] ** 2).sum(axis=0))

        return NeuralNetworkTrainResult(y_pred_all, ve_all, itest, test_time, ev)

    def plot_behavioral_prediction(self,
                                   raster: RasterMapResult,
                                   nn_result: NeuralNetworkTrainResult,
                                   output: DataOutput,
                                   xrange: tuple[int, int] = (0, 2000)):
        """
        Visualize the behavioral prediction on test data

        :param raster:
        :param nn_result:
        :param output:
        :param xrange: x data point range, unit: data point represent time (might be discrete time)
        :return:
        """
        # Compute the prediction for the superneurons:
        X_transformed, U = self._decomposition(self.dat.neural_activity)

        # principal components for superneurons
        U_sn = rastermap.utils.bin1d(U[raster.isort], bin_size=self.neuron_bins, axis=0)

        # use U_sn to project from prediction of PCs to superneurons
        sn_pred = U_sn @ nn_result.y_pred_all.T

        output_file = output.summary_figure_output(
            '2p',
            'orofacial_prediction',
        )

        with plot_figure(output_file, 13, 1,
                         gridspec_kw={'wspace': 1, 'hspace': 0.5}) as _ax:
            x1, x2 = xrange

            ax = _ax[0]
            ax.plot(self.dat.velocity[nn_result.itest][x1:x2], color='gray')
            ax.set_xlim([0, x2 - x1])
            ax.axis("off")
            ax.set_title("running speed", color='gray')

            # superneuron activity
            ax = ax_merge(_ax)[1:7]
            ax.imshow(raster.super_neurons[:, nn_result.itest][:, x1:x2],
                      cmap="gray_r",
                      vmin=0,
                      vmax=0.8,
                      aspect="auto",
                      interpolation='none')

            ax.set(ylabel='superneurons')
            ax.set_xticks([])

            # prediction
            ax = ax_merge(_ax)[7:]
            ax.imshow(sn_pred[:, x1:x2],
                      cmap="gray_r",
                      vmin=0,
                      vmax=0.85,
                      aspect="auto",
                      interpolation='none')

            ax.set(xlabel="time", ylabel='superneurons')
            ax.set_title("behavior prediction")


if __name__ == '__main__':
    RunRasterMap2POptions().main()
