import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def plot_simulated_place_cell(smoothing_sigma: float | None = None) -> None:
    # Create a synthetic 2D firing rate map for demonstration.
    # Suppose the arena is 100 cm x 100 cm, binned into 100 x 100 bins.
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)

    # Define a 2D Gaussian place field centered at (50, 50)
    x0, y0 = 20, 50  # center of the place field (cm)
    sigma = 10  # standard deviation (cm)
    peak_rate = 5  # peak firing rate (Hz)
    baseline = 0  # baseline firing rate (Hz)

    # Generate the 2D firing rate map
    rate_map = baseline + peak_rate * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2)))

    # Add a bit of noise to simulate variability
    noise = np.random.normal(0, 0.2, rate_map.shape)
    rate_map += noise

    #
    if smoothing_sigma is not None:
        rate_map = gaussian_filter(rate_map, sigma=smoothing_sigma)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(rate_map,
                    origin='lower',
                    extent=(0, rate_map.shape[1], 0, rate_map.shape[0]),
                    cmap='magma',
                    aspect='auto')
    plt.colorbar(im, label='Firing Rate (Hz)')
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.tight_layout()
    plt.show()


def plot_simulated_grid_cell(smoothing_sigma: float | None = None) -> None:
    # Create a synthetic 2D firing rate map for a grid cell.
    # Suppose the arena is 100 cm x 100 cm, binned into 100 x 100 bins.
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)

    # Define grid cell parameters.
    spacing = 20  # distance between grid fields (cm)
    orientation = 0  # grid orientation in degrees
    peak_rate = 5  # peak firing rate (Hz)
    baseline = 0  # baseline firing rate (Hz)

    # Calculate the spatial frequency.
    k = 2 * np.pi / spacing

    # Generate the grid cell pattern by summing cosine functions along three axes.
    # The three preferred directions (0°, 60°, and 120°) are adjusted by the grid orientation.
    angles = np.deg2rad(np.array([0, 60, 120]) + orientation)
    rate = np.zeros_like(X)
    for angle in angles:
        rate += np.cos(k * (X * np.cos(angle) + Y * np.sin(angle)))

    # Normalize the summed cosines to be in the range [baseline, baseline + peak_rate].
    # The sum of three cosines ranges from -3 to 3, so we shift and scale it.
    rate_map = baseline + peak_rate * ((rate + 3) / 6)

    # Add a bit of noise to simulate variability.
    noise = np.random.normal(0, 0.2, rate_map.shape)
    rate_map += noise

    # Optionally smooth the rate map with a Gaussian filter.
    if smoothing_sigma is not None:
        rate_map = gaussian_filter(rate_map, sigma=smoothing_sigma)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(rate_map,
                    origin='lower',
                    extent=(0, rate_map.shape[1], 0, rate_map.shape[0]),
                    cmap='magma',
                    aspect='auto')
    plt.colorbar(im, label='Firing Rate (Hz)')
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.tight_layout()
    plt.show()


def plot_simulated_border_cell(smoothing_sigma: float | None = None) -> None:
    # Create a synthetic 2D firing rate map for demonstration.
    # Suppose the arena is 100 cm x 100 cm, binned into 100 x 100 bins.
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)

    # Define border cell parameters.
    # This border cell is assumed to fire along the left border.
    # Its firing field is modeled as an elongated 2D Gaussian that peaks near x = 10 cm,
    # and is centered in the y-direction.
    x0, y0 = 10, 50  # center of the border field (cm)
    sigma_x = 5  # standard deviation in x (narrow to reflect border tuning)
    sigma_y = 30  # standard deviation in y (elongated field)
    peak_rate = 5  # peak firing rate (Hz)
    baseline = 0  # baseline firing rate (Hz)

    # Generate the 2D firing rate map using a 2D Gaussian.
    rate_map = baseline + peak_rate * np.exp(
        -(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2))
    )

    # Add a bit of noise to simulate variability.
    noise = np.random.normal(0, 1, rate_map.shape)
    rate_map += noise

    # Optionally smooth the rate map with a Gaussian filter.
    if smoothing_sigma is not None:
        rate_map = gaussian_filter(rate_map, sigma=smoothing_sigma)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(rate_map,
                    origin='lower',
                    extent=(0, rate_map.shape[1], 0, rate_map.shape[0]),
                    cmap='magma',
                    aspect='auto')
    plt.colorbar(im, label='Firing Rate (Hz)')
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    plt.tight_layout()
    plt.show()


def plot_simulated_head_direction(smoothing_sigma: float | None = None) -> None:
    # Create synthetic tuning data for a head-direction cell.
    # Generate 100 equally spaced angles between 0 and 2*pi.
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)

    # Define head-direction tuning parameters.
    preferred_angle = np.pi / 4  # Preferred head direction (45° in radians).
    kappa = 4.0  # Concentration parameter (controls tuning width).
    baseline = 0  # Baseline firing rate (Hz).
    peak_rate = 10  # Peak firing rate above baseline (Hz).

    # Generate the firing rate using a von Mises tuning curve.
    # The von Mises function is analogous to a circular Gaussian.
    rate = baseline + peak_rate * np.exp(kappa * (np.cos(angles - preferred_angle) - 1))

    # Add noise to simulate trial-to-trial variability.
    noise_std = 3.0  # Standard deviation of noise.
    noise = np.random.normal(0, noise_std, size=rate.shape)
    rate += noise

    # Clip any negative firing rates to zero.
    rate = np.clip(rate, 0, None)

    # Optionally smooth the 1D tuning curve.
    # Use mode='wrap' to account for the periodic (circular) nature of the data.
    if smoothing_sigma is not None:
        rate = gaussian_filter1d(rate, sigma=smoothing_sigma, mode='wrap')

    # Close the polar plot by appending the first value to the end.
    angles = np.append(angles, angles[0])
    rate = np.append(rate, rate[0])

    # Plot the tuning curve on a polar plot.
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, rate, linestyle='-', color='r')

    # Optionally adjust the polar plot formatting.
    ax.set_theta_zero_location('N')  # Set 0° at the top (North).
    ax.set_theta_direction(-1)  # Plot angles in clockwise direction.
    plt.title("Simulated Head Direction Cell Tuning")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_simulated_head_direction(smoothing_sigma=3)
