from pathlib import Path
from typing import Literal, Final

import numpy as np
import seaborn as sns
from rich.pretty import pprint

from argclz import as_argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline
from rscvp.util.cli import StatResults
from rscvp.util.database import DB_TYPE
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE
from rscvp.util.util_stat import CollectDataSet
from rscvp.visual.util import get_sftf_mesh_order

__all__ = ['SFTFPerfStat']


@publish_annotation('main', project='rscvp', figure='fig.5D', as_doc=True)
class SFTFPerfStat(StatPipeline):
    DESCRIPTION = 'See the sftf preference across animals, either in dff amplitude or preferred fraction'

    header: Literal['amp', 'frac'] = as_argument(StatPipeline.header).with_options(...)

    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))

    # load source
    sheet_name: Final[GSPREAD_SHEET_PAGE] = 'VisualSFTFDirDB'
    db_table: Final[DB_TYPE] = 'VisualSFTFDirDB'

    # statistic
    dependent = False
    parametric = False
    test_type = 'ttest'

    def run(self):
        self.load_table(primary_key='date', to_pandas=False, concat_plane=True)
        self.replace_col()
        self.run_pipeline()

    @property
    def directory(self) -> Path:
        return self.statistic_dir / self.sheet_name / f'sftf_{self.header}_{self.group_header}_{self.test_type}'

    @property
    def sftf(self) -> list[str]:
        return list(map(lambda x: x.replace(' ', '_'), get_sftf_mesh_order()))

    def replace_col(self) -> None:
        """Replace group number to actual `sf_tf`"""
        df = self.df
        for i in range(1, 7):
            df = df.rename({f'sftf_{self.header}_group{i}': f'sftf_{self.header}_{self.sftf[i - 1]}'})

        self.df = df

    def get_collect_data(self) -> CollectDataSet:
        """Categorical data {C x G: D}"""
        if self._collect_data is None:
            prev = None

            for sftf in self.sftf:
                tmp = self._get_collect_data(self.group_header,
                                             f'sftf_{self.header}_{sftf}',
                                             key_prefix=sftf,
                                             verbose=False)

                if prev is None:
                    prev = tmp
                else:
                    prev = prev.update(tmp)

            self._collect_data = prev

            # overwrite
            self._collect_data.n_groups = len(self.df[self.group_header].unique())
            self._collect_data.n_categories = len(self.sftf)
            self._collect_data.data_type = 'categorical'
            self._collect_data.test_type = self.test_type

            pprint(self._collect_data)

        return self._collect_data

    def generate_stat_result(self) -> dict[str, StatResults]:
        data = self.get_collect_data()
        ret = {}
        for sftf in self.sftf:
            x = data[('aRSC', sftf)]
            y = data[('pRSC', sftf)]

            out = self.output_statistic_json
            name = out.stem + sftf
            stat_result = self.run_ttest(dataset=np.vstack([x, y]), output=out.with_stem(name))
            ret[sftf] = stat_result

        return ret

    def plot(self):
        cols = [col for col in self.df.columns if col.startswith(f'sftf_{self.header}')]
        df = self.df.unpivot(on=cols, index='region')

        with plot_figure(self.output_figure, 5, 10) as _:
            g = sns.catplot(data=df, kind='bar', x='variable', y='value', hue='region',
                            errorbar=('se', 1), alpha=0.7)
            ax = g.ax
            sns.swarmplot(ax=ax, data=df, x='variable', y='value', hue='region', dodge=0.2)
            ax.set(ylabel=self.header, xlabel='SFTF')
            ax.tick_params(axis='x', labelrotation=45)


if __name__ == '__main__':
    SFTFPerfStat().main()
