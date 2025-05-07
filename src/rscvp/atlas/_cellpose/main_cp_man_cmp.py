from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from roifile import ImagejRoi
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli.cli_hist import HistOptions

from argclz import AbstractParser
from neuralib.util.unstable import unstable


@unstable()
class ROIManualCposeCompareOptions(AbstractParser, HistOptions):
    DESCRIPTION = 'show the distribution of manually selection v.s. cellpose result'

    ccf_dir: AbstractCCFDir

    def run(self):
        self.ccf_dir = self.get_ccf_dir()
        df = self.load_image_roi(self.ccf_dir)
        self.plot_joint_scatter_hist(df)

    @property
    def output_file(self) -> Path:
        d = self.ccf_dir.output_folder
        f = d / self.ccf_dir.get_slice_id(self.glass_id, self.slice_id, channel=self.channel)
        return f.with_suffix('.pdf')

    def plot_joint_scatter_hist(self, df: pl.DataFrame):
        g = sns.jointplot(data=df.to_pandas(), x='x', y='y', hue='type', alpha=0.7)
        ax = g.ax_joint
        ax.invert_yaxis()
        plt.savefig(self.output_file)

    def load_image_roi(self, ccf_dir: AbstractCCFDir) -> pl.DataFrame:
        roi_man_file = ccf_dir.glob(self.glass_id, self.slice_id, 'roi', channel=self.channel)
        roi_cpose = ccf_dir.glob(self.glass_id, self.slice_id, 'roi_cpose', channel=self.channel)

        man = np.asarray(ImagejRoi.fromfile(roi_man_file).subpixel_coordinates, dtype=np.float64)
        cpose = np.asarray(ImagejRoi.fromfile(roi_cpose).integer_coordinates, dtype=np.float64)
        df_man = _construct_roi_dataframe(man[:, 0], man[:, 1], 'manual')
        df_cpose = _construct_roi_dataframe(cpose[:, 0], cpose[:, 1], 'cpose')

        return pl.concat([df_man, df_cpose])


def _construct_roi_dataframe(x: np.ndarray,
                             y: np.ndarray,
                             roi_type: str) -> pl.DataFrame:
    """

    :param x: pixel x coordinate
    :param y: pixel y coordinate
    :param roi_type: {'manual', 'cpose'}
    # :param flip_x: hemisphere reverting correction TODO
    :return:
    """
    return (
        pl.DataFrame({'x': x, 'y': y})
        .with_columns(pl.lit(roi_type).alias('type'))
    )


if __name__ == '__main__':
    ROIManualCposeCompareOptions().main()
