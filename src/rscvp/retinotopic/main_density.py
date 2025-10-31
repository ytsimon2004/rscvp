import numpy as np
import tifffile
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from argclz import AbstractParser, argument
from neuralib.imaging.widefield import SequenceFFT
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_cyclic_colorbar
from neuralib.util.unstable import unstable
from rscvp.retinotopic.cache_retinotopic import RetinotopicCacheBuilder
from rscvp.util.cli import WFieldOptions


@unstable()
class RetinotopicDensityOptions(AbstractParser, WFieldOptions):
    DESCRIPTION = ...

    bin_size: int = argument('--bin', default=50, help='bin size for the image pixel')

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        cache = get_options_and_cache(RetinotopicCacheBuilder, self)

        f = ...
        seq = tifffile.imread(f)

        fft = SequenceFFT(seq)
        phase = fft.get_phase()  # This represents azimuth position preference

        with plot_figure(None, 1, 2) as ax:
            self.plot_phase(ax[0], phase)
            self.plot_density(ax[1], phase)

    def plot_phase(self, ax: Axes, phase: np.ndarray):
        im = ax.imshow(phase, cmap='hsv')
        insert_cyclic_colorbar(ax, im, num_colors=36, width=0.2, inner_diameter=1, vmin=0, vmax=1)

    def plot_density(self, ax: Axes, phase: np.ndarray):
        height, width = phase.shape
        b = self.bin_size
        x_bins = np.arange(0, width + b, b)
        y_bins = np.arange(0, height, b)

        colors = plt.cm.viridis(np.linspace(0, 1, len(y_bins) - 1))
        for i, (y_start, color) in enumerate(zip(y_bins[:-1], colors)):
            y_end = min(y_start + b, height)

            x_centers = []
            deg_means = []

            for x_start in x_bins[:-1]:
                x_end = min(x_start + b, width)

                # Extract phase values for this y-region and x range
                phase_values = []
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        phase_values.append(phase[y, x])

                if phase_values:
                    phase_values = np.array(phase_values)
                    # Handle circular statistics for phase
                    mean_phase = np.angle(np.mean(np.exp(1j * phase_values)))
                    mean_phase_deg = np.degrees(mean_phase)

                    x_centers.append(x_start + b / 2)
                    deg_means.append(mean_phase_deg)

            if len(x_centers) > 0:
                if x_centers:
                    label = f'Y: {y_start}-{y_end - 1} pixels'
                    plt.plot(x_centers, deg_means,
                             marker='o', linewidth=2, color=color, alpha=0.8,
                             label=label)

        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('circular preference (degrees)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


if __name__ == '__main__':
    RetinotopicDensityOptions().main()
