import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson


def plot_poisson_distribution(n: int = 50,
                              lam_range: tuple[int, int] = (5, 50),
                              n_lines: int = 4):
    x = np.arange(0, n)

    for i in range(n_lines):
        lam = np.random.uniform(*lam_range)
        pmf = poisson.pmf(x, mu=lam)
        offset = i * 0.2
        plt.plot(x, pmf + offset, color='k')

    plt.show()
