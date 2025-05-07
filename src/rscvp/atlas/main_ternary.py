import plotly.express as px

from argclz import AbstractParser, as_argument
from neuralib.atlas.ccf.dataframe import RoiNormalizedDataFrame
from neuralib.util.verbose import print_save
from rscvp.atlas.dir import AbstractCCFDir
from rscvp.util.cli.cli_roi import ROIOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

__all__ = ['TernaryPercOptions']


class TernaryPercOptions(AbstractParser, ROIOptions):
    DESCRIPTION = 'Ternary plot for each region in triangle channel space'

    roi_norm = as_argument(ROIOptions.roi_norm).with_options(choices=['cell', 'volume', 'none'])

    ccf_dir: AbstractCCFDir
    df: RoiNormalizedDataFrame

    def run(self):
        self.ccf_dir = self.get_ccf_dir()
        self.df = (
            self.load_roi_dataframe(self.ccf_dir)
            .to_normalized(self.roi_norm, self.merge_level, top_area=self.top_area, hemisphere=self.hemisphere)
        )
        df = self.df.to_winner(['aRSC', 'pRSC'])

        #
        angle = ['overlap', 'pRSC', 'aRSC']
        fig = px.scatter_ternary(
            df,
            a=angle[0],
            b=angle[1],
            c=angle[2],
            hover_name=self.df.classified_column,
            text=self.df.classified_column,
            size='total',
            color='winner',
            size_max=30,
            color_discrete_map=REGION_COLORS_HIST
        )

        fig.update_layout(
            ternary=dict(
                bgcolor='whitesmoke',
                aaxis=dict(gridcolor='silver'),
                baxis=dict(gridcolor='silver'),
                caxis=dict(gridcolor='silver'),
            )
        )

        #
        if self.debug_mode:
            fig.show()
        else:
            out = self.ccf_dir.figure_output(
                self.animal,
                f'L{self.merge_level}',
                f'T{self.top_area}',
                self.df.normalized_unit,
                'ternary'
            )
            fig.write_image(str(out))
            print_save(out)


if __name__ == '__main__':
    TernaryPercOptions().main()
