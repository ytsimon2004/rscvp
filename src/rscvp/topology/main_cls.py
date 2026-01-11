import matplotlib.pyplot as plt
import numpy as np

from argclz import AbstractParser, as_argument
from neuralib.imglib.color import grey2rgb
from neuralib.imglib.norm import get_percentile_value
from neuralib.plot import plot_figure
from neuralib.suite2p.plot import plot_soma_center
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_selection import SelectionOptions
from .util_plot import plot_registered_fov

__all__ = ['ClsTopoOptions']


class ClsTopoOptions(AbstractParser, SelectionOptions):
    DESCRIPTION = 'plot topographical distribution of different cell types'

    plane_index: int = as_argument(SelectionOptions.plane_index).with_options(
        required=True,
        help='topographical plot need to implement in a certain optic plane'
    )

    pc_selection = 'slb'
    vc_selection = 0.3
    pre_selection = True
    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('topo')
        self.visual_place_position(output_info)

    def visual_place_position(self, output: DataOutput):
        """
        plot identified visual, place cells position
        """

        s2p = self.load_suite_2p()

        # neuron type mask
        vc = self.select_visual_neurons()
        pc = self.select_place_neurons(classifier='slb', pf_limit=self.pf_limit)
        vP = vc & ~pc  # vc (pure)
        Vp = ~vc & pc  # pc (pure)
        vpc = vc & pc  # overlap
        uc = ~vc & ~pc  # unclass
        rc = self.select_red_neurons() if self.has_chan2 else None

        if self.pre_selection:
            ps = self.pre_select()
            vP = vP & ps
            Vp = Vp & ps
            vpc = vpc & ps
            uc = uc & ps
            rc = rc & ps if self.has_chan2 else None

        im1 = s2p.image_mean
        im2 = s2p.image_mean_ch2 if self.has_chan2 else None
        lb, up = get_percentile_value(im1)
        norm_im1 = plt.Normalize(vmin=lb, vmax=up)

        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
        )

        if not self.has_chan2:
            with plot_figure(output_file) as ax:
                ax.imshow(im1, cmap='bone', alpha=0.8, norm=norm_im1)
                plot_registered_fov(ax, s2p, vP, uni_color='Reds')
                plot_registered_fov(ax, s2p, Vp, uni_color='Greens')
                plot_registered_fov(ax, s2p, vpc, uni_color='cividis')
                plot_registered_fov(ax, s2p, uc, uni_color='Greys')
                plot_soma_center(ax, s2p, None, invert_xy=True, s=2, alpha=0.3)

        else:
            with plot_figure(output_file,
                             2, 1, figsize=(20, 10)) as _ax:

                ax = _ax[0]
                ax.imshow(im1, cmap='bone', alpha=0.8, norm=norm_im1)
                plot_registered_fov(ax, s2p, vP, uni_color='Reds')
                plot_registered_fov(ax, s2p, Vp, uni_color='Greens')
                plot_registered_fov(ax, s2p, vpc, uni_color='cividis')
                plot_registered_fov(ax, s2p, uc, uni_color='Greys')
                plot_registered_fov(ax, s2p, rc, uni_color='Blues')

                # merge of two color img, for very rough visualization
                ax = _ax[1]
                im_g = grey2rgb(im1, 1)
                im_r = grey2rgb(im2, 0)
                ax.imshow(im_g)
                ax.imshow(im_r)
                ax.set_title(f'double_label:{np.count_nonzero(rc)}')
                plot_soma_center(ax, s2p, neuron_ids=rc, invert_xy=True,
                                 color='b', marker='s', facecolors='none', edgecolors='b', alpha=0.5, s=3)


if __name__ == '__main__':
    ClsTopoOptions().main()
