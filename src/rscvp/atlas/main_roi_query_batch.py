from typing import Final

import polars as pl
from rscvp.util.cli import HistOptions, ROIOptions

from argclz import AbstractParser, as_argument, str_tuple_type, argument
from neuralib.atlas.ccf import ROIS_NORM_TYPE
from neuralib.atlas.typing import Area, TreeLevel
from neuralib.plot import plot_figure, dotplot
from neuralib.util.verbose import printdf, fprint

__all__ = ['RoiQueryBatchOptions']


class RoiQueryBatchOptions(AbstractParser, ROIOptions):
    DESCRIPTION = 'Dotplot for subareas from a single area (foreach channel, animal)'

    animal = as_argument(HistOptions.animal).with_options(
        type=str_tuple_type,
        help='multiple animals. e.g. YW001,YW002'
    )

    area: Area = as_argument(HistOptions.area).with_options(
        type=str,
        help='single area query only',
    )

    force_set_show_col: TreeLevel | None = argument(
        '--show-col',
        metavar='LEVEL',
        default=None,
        help='force set show col to which level'
    )

    roi_norm: Final[ROIS_NORM_TYPE] = 'channel'

    def run(self):
        df = self.get_batch_subregion_data()
        self.plot_dot_batch(df)
        self.print_var(df)

    def get_batch_subregion_data(self) -> pl.DataFrame:
        ret = []
        for ccf_dir in self.foreach_ccf_dir(self.animal):
            subregion = (self.load_roi_dataframe(ccf_dir)
                         .to_subregion(self.area, source_order=('aRSC', 'pRSC', 'overlap'),
                                       show_col=self.force_set_show_col,
                                       animal=ccf_dir.animal))

            ret.append(subregion.dataframe())

        return pl.concat(ret, how='diagonal').fill_null(0)

    @staticmethod
    def print_var(concat_df: pl.DataFrame):
        """statistic info"""
        for k, df in concat_df.partition_by('source', as_dict=True).items():
            dfx = df.drop('source', 'animal')

            mean = dfx.mean().with_columns(pl.lit('MEAN').alias('measurement'))
            sem = dfx.select([
                (pl.std(col) / pl.count(col).cast(float).sqrt()).alias(col)
                for col in dfx.columns
            ]).with_columns(pl.lit('STDDEV').alias('measurement'))

            fprint(f'{k}')
            printdf(pl.concat([mean, sem], how="diagonal"))

    @staticmethod
    def plot_dot_batch(df: pl.DataFrame):
        with plot_figure(None, 1, 3) as ax:
            for i, (name, dat) in enumerate(df.group_by(['source'])):
                areas = dat.drop('source', 'animal').columns
                size = dat.select(areas).to_numpy()  # (Animal, Area)
                dotplot(dat['animal'], areas, size, scale='area', ax=ax[i])
                ax[i].set_title(dat['source'][0])


if __name__ == '__main__':
    RoiQueryBatchOptions().main()
