from pathlib import Path

import matplotlib.pyplot as plt

from argclz import AbstractParser, as_argument, argument
from neuralib.imglib.color import grey2rgb
from neuralib.imglib.transform import affine_transform
from neuralib.suite2p import Suite2PResult, get_soma_pixel
from neuralib.util.verbose import publish_annotation
from rscvp.util.cli.cli_suite2p import Suite2pOptions

__all__ = ['FOVOptions']


@publish_annotation('main', project='rscvp', figure='fig.1C', as_doc=True)
class FOVOptions(AbstractParser, Suite2pOptions):
    DESCRIPTION = 'Plot the recording FOV in both PMT channels and suite2p registered somata'

    plot_actual_cmap: bool = argument('--actual', help='whether image plot as actual PMT color, otherwise, greyscale')
    do_affine_transform: bool = argument('--affine', help='do affine_transform to parallelogram. i.e., for etl demo')
    plane_index = as_argument(Suite2pOptions.plane_index).with_options(required=True)

    #
    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        s2p = self.load_suite_2p()
        output = self.get_data_output('topo').directory if not self.debug_mode else None
        self.plot_fov(s2p, output)

    def plot_fov(self, s2p: Suite2PResult, output: Path | None):
        self._plot_ch1_fov(s2p, output)
        if self.has_chan2:
            self._plot_ch2_fov(s2p, output)

    def _plot_ch1_fov(self, s2p, output):
        img = s2p.image_mean
        reg = get_soma_pixel(s2p, neuron_ids=None, color_diff=True)
        if self.do_affine_transform:
            img = affine_transform(img)
            reg = affine_transform(reg)
        #
        fig, ax = plt.subplots(1, 2)
        match self.plot_actual_cmap, output:
            case (True, None):
                ax[0].imshow(grey2rgb(img, 1))
                ax[1].imshow(reg, cmap='nipy_spectral')
                plt.show()
            case (False, None):
                ax[0].imshow(img, cmap='gray')
                ax[1].imshow(reg, cmap='nipy_spectral')
                plt.show()
            case (True, output) if output is not None:
                plt.imsave((output / 'ch1.png'), grey2rgb(img, 1))
                plt.imsave((output / 'reg.png'), reg, cmap='nipy_spectral')
            case (False, output) if output is not None:
                plt.imsave((output / 'ch1.png'), img, cmap='gray')
                plt.imsave((output / 'reg.png'), reg, cmap='nipy_spectral')

    def _plot_ch2_fov(self, s2p, output):
        img = s2p.image_mean_ch2
        if self.do_affine_transform:
            img = affine_transform(img)
        #
        ax = plt.gca()
        match self.plot_actual_cmap, output:
            case (True, None):
                ax.imshow(grey2rgb(img, 1))
                plt.show()
            case (False, None):
                ax.imshow(img, cmap='gray')
                plt.show()
            case (True, output) if output is not None:
                plt.imsave((output / 'ch2.png'), grey2rgb(img, 0))
            case (False, output) if output is not None:
                plt.imsave((output / 'ch1.png'), img, cmap='gray')


if __name__ == '__main__':
    FOVOptions().main()
