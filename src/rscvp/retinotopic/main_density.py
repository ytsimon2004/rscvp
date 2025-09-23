import numpy as np
from matplotlib import pyplot as plt

from argclz import AbstractParser
from neuralib.imaging.widefield import SequenceFFT
from neuralib.persistence.cli_persistence import get_options_and_cache
from rscvp.retinotopic.cache_retinotopic import RetinotopicCacheBuilder
from rscvp.util.cli import WFieldOptions


class RetinotopicDensityOptions(AbstractParser, WFieldOptions):
    DESCRIPTION = ...

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        cache = get_options_and_cache(RetinotopicCacheBuilder, self)

        fft = SequenceFFT(cache.trial_averaged_resp)
        phase = fft.get_phase()  # This represents azimuth position preference

        height, width = phase.shape

        # Create x-axis bins (every 50 pixels)
        bin_size = 50
        x_bins = np.arange(0, width + bin_size, bin_size)

        # Create y-axis bins for different cortical regions
        y_bin_size = 50
        y_bins = np.arange(0, height, y_bin_size)

        plt.figure(figsize=(12, 8))

        # Plot multiple lines for different y-axis regions
        colors = plt.cm.viridis(np.linspace(0, 1, len(y_bins) - 1))

        for i, (y_start, color) in enumerate(zip(y_bins[:-1], colors)):
            y_end = min(y_start + y_bin_size, height)

            # Calculate mean azimuth preference for each x bin
            x_centers = []
            azimuth_means = []

            for x_start in x_bins[:-1]:
                x_end = min(x_start + bin_size, width)

                # Extract phase values for this y-region and x range
                phase_values = []
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        phase_values.append(phase[y, x])

                if phase_values:
                    phase_values = np.array(phase_values)
                    # Handle circular statistics for phase
                    mean_phase = np.angle(np.mean(np.exp(1j * phase_values)))
                    # Convert to degrees
                    mean_phase_deg = np.degrees(mean_phase)

                    x_centers.append(x_start + bin_size / 2)
                    azimuth_means.append(mean_phase_deg)

            # Plot line for this y-region
            if x_centers:
                label = f'Y: {y_start}-{y_end - 1} pixels'
                plt.plot(x_centers, azimuth_means,
                         marker='o', linewidth=2, color=color, alpha=0.8,
                         label=label)

        plt.xlabel('X Position (pixels)')
        plt.ylabel('Azimuth Position Preference (degrees)')
        plt.title('Azimuth Preference vs X Position\n(Different Y-axis regions)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=180, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=-180, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    RetinotopicDensityOptions().main()
