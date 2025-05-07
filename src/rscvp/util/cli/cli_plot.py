from typing import Literal, ClassVar

from matplotlib import pyplot as plt

from argclz import argument

__all__ = [
    'PlotOptions',
    'FIG_MODE',
]

FIG_MODE = Literal['simplified', 'presentation']


class PlotOptions:
    """cli options for the plotting"""
    GROUP_PLOT: ClassVar = 'Plot Options'

    plot_summary: bool = argument(
        '--summary',
        group=GROUP_PLOT,
        help='for the general usage of summary plot, i.e., population data',
    )

    mode: FIG_MODE = argument(
        '--mode', '--fig-mode',
        default='simplified',
        group=GROUP_PLOT,
        help='whether plot some figure detail for batch analysis',
    )

    black_bg: bool = argument(
        '--black',
        group=GROUP_PLOT,
        help='plt black background'
    )

    line_color: str = 'k'
    cmap_color: str = 'Greys'

    def set_background(self, style='dark_background'):
        if self.black_bg:
            plt.style.use(style)
            self._set_black_default_color()

    def _set_black_default_color(self):
        self.line_color = 'white'
        self.cmap_color = 'bone'
