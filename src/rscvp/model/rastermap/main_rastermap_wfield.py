import numpy as np
import rastermap.utils
from matplotlib.axes import Axes
from rastermap import Rastermap
from rscvp.model.rastermap.rastermap_wfield_cache import RasterInputWfield, RasterMapWfieldCacheBuilder
from rscvp.util.cli import DataOutput, RasterMapOptions, WFieldOptions
from scipy.stats import zscore

from argclz import AbstractParser
from neuralib.model.rastermap import RasterMapResult, RasterOptions
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure, ax_merge
from neuralib.util.utils import uglob
from neuralib.util.verbose import fprint

DEFAULT_WFIELD_RASTER_OPT: RasterOptions = {
    'n_clusters': 100,
    'locality': 0.5,
    'time_lag_window': 10,
    'grid_upsample': 10
}


class RunRasterMapWfieldOptions(AbstractParser, WFieldOptions, RasterMapOptions):
    DESCRIPTION = 'Run rastermap with wfield imaging data'

    dat: RasterInputWfield
    neuron_bins = 500
    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        self.load_input_from_cache()
        raster = self.run_rastermap()
        output_info = self.get_data_output('rastermap', output_type='wfield')

        self.plot_rastermap_sort(raster, self.time_range, output_info)
        self.plot_raster_voxel(raster, output_info)

    def load_input_from_cache(self) -> None:
        self.dat = get_options_and_cache(RasterMapWfieldCacheBuilder, self).load_result()

    def run_rastermap(self, **kwargs) -> RasterMapResult:
        ops = DEFAULT_WFIELD_RASTER_OPT
        ops['n_PCs'] = self.dat.n_components

        if not self.rastermap_result_cache().exists() or self.force_compute:
            model = Rastermap(
                n_clusters=ops['n_clusters'],
                n_PCs=ops['n_PCs'],
                locality=ops['locality'],
                time_lag_window=ops['time_lag_window'],
                grid_upsample=ops['grid_upsample'],
                **kwargs
            ).fit(
                Usv=self.dat.U * self.dat.sv,  # left singular vectors weighted by the singular values
                Vsv=self.dat.Vsv  # right singular vectors weighted by the singular values
            )

            embedding = model.embedding
            isort = model.isort
            Vsv_sub = model.Vsv  # these are the PCs across time with the mean across voxels subtracted

            U_sn = rastermap.utils.bin1d(self.dat.U[isort], bin_size=self.neuron_bins, axis=0)  # bin over voxel axis
            sn = U_sn @ Vsv_sub.T
            sn = zscore(sn, axis=1)

            # For fit gui launch behavior
            folder = self.processed_dir

            ret = RasterMapResult(
                filename=self._try_find_filename(),
                save_path=str(folder),
                isort=isort,
                embedding=embedding,
                ops=ops,
                user_clusters=[],
                super_neurons=sn
            )

            ret.save(self.rastermap_result_cache())
        else:
            ret = RasterMapResult.load(self.rastermap_result_cache())

        return ret

    def _try_find_filename(self) -> str | None:
        try:
            return str(uglob(self.processed_dir, '*.avi')) if self.load_avi else uglob(self.phys_dir, 'run00*')
        except FileNotFoundError as e:
            fprint(f'{e}', vtype='warning')
            return

    def plot_rastermap_sort(self,
                            raster: RasterMapResult,
                            trange: tuple[int, int],
                            output: DataOutput):
        tmask = np.logical_and(trange[0] <= self.dat.image_time, self.dat.image_time <= trange[1])
        time = self.dat.image_time[tmask]

        output_file = output.summary_figure_output(f'wfield_sorting_{self.time_range}')
        with plot_figure(output_file,
                         9, 20,
                         dpi=200,
                         gridspec_kw={'wspace': 1, 'hspace': 1}) as _ax:
            # position
            ax1: Axes = ax_merge(_ax)[0, :-1]
            ax1.plot(time, self.dat.position[tmask], color='k')
            ax1.axis("off")
            ax1.set_title("position", color='k')

            # running speed
            ax2: Axes = ax_merge(_ax)[1, :-1]
            ax2.plot(time, self.dat.velocity[tmask], color='k')
            ax2.axis("off")
            ax2.set_title("running speed", color='k')
            ax2.sharex(ax1)

            # superneuron
            ax3 = ax_merge(_ax)[2:, :-1]
            ax3.sharex(ax1)
            ax3.imshow(raster.super_neurons[:, tmask],
                       cmap="gray_r",
                       vmin=0,
                       vmax=0.8,
                       aspect="auto",
                       extent=(self.time_range[0], self.time_range[1], raster.n_clusters, 0))
            ax3.set(xlabel="time(s)", ylabel='superneurons')

            # visual stim
            if self.dat.visual_stim_start <= self.time_range[1]:
                for v in self.dat.visual_stim_trange(trange=self.time_range):
                    ax3.axvspan(v[0], v[1], color='mistyrose', alpha=0.6)

            # disable
            ax4 = ax_merge(_ax)[:2, -1]
            ax4.axis('off')

            # color bar
            ax4 = ax_merge(_ax)[2:, -1]
            ax4.imshow(np.arange(0, raster.n_clusters)[:, np.newaxis], cmap="gist_ncar", aspect="auto")
            ax4.axis("off")

    def plot_raster_voxel(self,
                          raster: RasterMapResult,
                          output: DataOutput):
        output_file = output.summary_figure_output('fov')

        with plot_figure(output_file) as ax:
            ax.scatter(self.dat.xpos,
                       self.dat.ypos,
                       s=1, c=raster.embedding, cmap="gist_ncar", alpha=0.25)
            ax.invert_yaxis()
            ax.set(xlabel='X position (um)', ylabel='Y position')
            ax.set_aspect('equal')


if __name__ == '__main__':
    RunRasterMapWfieldOptions().main()
