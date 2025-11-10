from typing import Literal

import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import wilcoxon, mannwhitneyu

from argclz import as_argument, argument
from neuralib.plot import plot_figure
from neuralib.util.verbose import publish_annotation
from rscvp.statistic.core import StatPipeline, print_var
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

__all__ = ['BlankClassStat']


@publish_annotation('sup', project='rscvp', figure='fig.S2C', as_doc=True)
class BlankClassStat(StatPipeline):
    DESCRIPTION = 'Fraction of spatial cells in anterior v.s. posterior RSC in blank belt treadmill'

    analysis: Literal['anterior', 'posterior', 'overall'] = argument(
        '--analysis',
        default='overall',
        help='analysis type',
    )

    header = as_argument(StatPipeline.header).with_options(required=False)

    load_source = as_argument(StatPipeline.load_source).with_options(default='gspread', choices=('gspread', 'db'))
    sheet_list: list[GSPREAD_SHEET_PAGE] = ['BaseClassDB', 'BlankClassDB']

    df = None  # overwrite

    def run(self):
        self.get_dataframe()
        self.plot()
        self.verbose()

    def get_dataframe(self):
        concat = []
        for name in self.sheet_list:
            self.sheet_name = name
            self.db_table = name
            self.load_table(to_pandas=False)

            df = (
                self._filter_concat_optics(self.df)
                .select(['date', 'animal', 'region', 'n_selected_neurons', 'pair_wise_group', 'n_spatial_neurons'])
                .with_columns((pl.col('date') + '_' + pl.col('animal')).alias('primary_key'))
                .with_columns((pl.col('n_spatial_neurons') / pl.col('n_selected_neurons')).alias('fraction'))
                .select(['primary_key', 'region', 'fraction', 'pair_wise_group'])
                .with_columns(pl.lit(name).alias('table'))
            )

            concat.append(df)

        if self.analysis == 'overall':
            self.df = pl.concat(concat)
        else:
            self.df = self.pivot_dataframe(pl.concat(concat))

    _ignore_dataset = ('210315_YW006',)

    def pivot_dataframe(self, df):
        if self.analysis == 'anterior':
            region = 'aRSC'
        else:
            region = 'pRSC'

        animals_in_both = (
            df.filter(pl.col('region') == region)
            .with_columns((pl.col('primary_key').str.split('_').list.get(1)).alias('animal'))
            .group_by('animal')
            .agg(pl.col('table').n_unique())
            .filter(pl.col('table') == 2)
            .select('animal')
        )

        df = (
            df.filter(pl.col('region') == region)
            .filter((~pl.col('primary_key').is_in(self._ignore_dataset)))
            .with_columns((pl.col('primary_key').str.split('_').list.get(1)).alias('animal'))
            .join(animals_in_both, on='animal', how='semi')
            .with_columns(pl.col('animal').rank('dense').cast(pl.Int64).alias('pair_wise_group'))
            .sort('animal')
        )

        return df

    @property
    def x(self) -> str:
        if self.analysis == 'overall':
            return 'table'
        else:
            return 'region'

    @property
    def hue(self) -> str:
        if self.analysis == 'overall':
            return 'region'
        else:
            return 'table'

    def plot(self):
        df = self.df
        x = self.x
        hue = self.hue
        with plot_figure(None) as ax:
            sns.barplot(
                data=df,
                x=x,
                y="fraction",
                hue=hue,
                errorbar="se",
                capsize=0.2,
                ax=ax
            )

            sns.stripplot(
                data=df,
                x=x,
                y="fraction",
                hue=hue,
                dodge=True,
                jitter=False,
                alpha=0.5,
                palette='dark:black',
                ax=ax
            )

            self.plot_connect_line(ax)
            self.add_primary_key_legend(ax)

    def plot_connect_line(self, ax: Axes):
        """Connect pairs with same pair_wise_group - adapts to analysis type"""
        df_filtered = (
            self.df.cast({'pair_wise_group': pl.Int64})
            .filter(pl.col('pair_wise_group') != -1)
        )

        # Get all PathCollections (stripplot points) from the axes
        collections = [
            child
            for child in ax.get_children() if
            hasattr(child, 'get_offsets') and len(child.get_offsets()) > 0
        ]

        point_positions = {}

        if self.analysis == 'overall':
            expected_positions = {
                ('BaseClassDB', 'aRSC'): (0.2, 0.1),
                ('BaseClassDB', 'pRSC'): (-0.2, 0.1),
                ('BlankClassDB', 'aRSC'): (1.2, 0.1),
                ('BlankClassDB', 'pRSC'): (0.8, 0.1),
            }

            for collection in collections:
                positions = collection.get_offsets()

                for pos in positions:
                    x_pos, y_pos = pos
                    matching_rows = df_filtered.filter(
                        (pl.col('fraction') - y_pos).abs() < 0.0001
                    )

                    if len(matching_rows) >= 1:
                        best_match = None
                        best_distance = float('inf')

                        for row in matching_rows.iter_rows(named=True):
                            table_region = (row['table'], row['region'])
                            if table_region in expected_positions:
                                expected_x, tolerance = expected_positions[table_region]
                                distance = abs(x_pos - expected_x)

                                if distance <= tolerance and distance < best_distance:
                                    best_match = row
                                    best_distance = distance

                        if best_match:
                            key = (best_match['table'], best_match['region'], best_match['pair_wise_group'])
                            point_positions[key] = pos

            # Connect aRSC to pRSC within each table
            tables = df_filtered['table'].unique().sort()
            for table in tables:
                table_df = df_filtered.filter(pl.col('table') == table)

                for group_id in table_df['pair_wise_group'].unique():
                    pair_df = table_df.filter(pl.col('pair_wise_group') == group_id)

                    arsc_data = pair_df.filter(pl.col('region') == 'aRSC')
                    prsc_data = pair_df.filter(pl.col('region') == 'pRSC')

                    if len(arsc_data) == 1 and len(prsc_data) == 1:
                        arsc_pos = point_positions.get((table, 'aRSC', group_id))
                        prsc_pos = point_positions.get((table, 'pRSC', group_id))

                        if arsc_pos is not None and prsc_pos is not None:
                            ax.plot([arsc_pos[0], prsc_pos[0]], [arsc_pos[1], prsc_pos[1]],
                                    color='gray', alpha=0.7, linewidth=2, zorder=10)

        else:
            expected_positions = {
                'BaseClassDB': (-0.2, 0.15),
                'BlankClassDB': (0.2, 0.15),
            }

            for collection in collections:
                positions = collection.get_offsets()

                for pos in positions:
                    x_pos, y_pos = pos
                    matching_rows = df_filtered.filter(
                        (pl.col('fraction') - y_pos).abs() < 0.0001
                    )

                    if len(matching_rows) >= 1:
                        best_match = None
                        best_distance = float('inf')

                        for row in matching_rows.iter_rows(named=True):
                            table = row['table']
                            if table in expected_positions:
                                expected_x, tolerance = expected_positions[table]
                                distance = abs(x_pos - expected_x)

                                if distance <= tolerance and distance < best_distance:
                                    best_match = row
                                    best_distance = distance

                        if best_match:
                            key = (best_match['table'], best_match['pair_wise_group'])
                            point_positions[key] = pos

            # Connect same animal across tables
            for group_id in df_filtered['pair_wise_group'].unique():
                pair_df = df_filtered.filter(pl.col('pair_wise_group') == group_id)

                generic_data = pair_df.filter(pl.col('table') == 'BaseClassDB')
                blank_data = pair_df.filter(pl.col('table') == 'BlankClassDB')

                if len(generic_data) == 1 and len(blank_data) == 1:
                    generic_pos = point_positions.get(('BaseClassDB', group_id))
                    blank_pos = point_positions.get(('BlankClassDB', group_id))

                    if generic_pos is not None and blank_pos is not None:
                        ax.plot([generic_pos[0], blank_pos[0]], [generic_pos[1], blank_pos[1]],
                                color='gray', alpha=0.7, linewidth=2, zorder=10)

    def add_primary_key_legend(self, ax: Axes):
        """Add text annotations showing primary keys for each data point"""
        df_filtered = (
            self.df.cast({'pair_wise_group': pl.Int64})
            .filter(pl.col('pair_wise_group') != -1)
        )

        # Get all PathCollections (stripplot points) from the axes
        collections = [child for child in ax.get_children()
                       if hasattr(child, 'get_offsets') and len(child.get_offsets()) > 0]

        # Define expected x-ranges for each table-region combination
        expected_positions = {
            ('BaseClassDB', 'aRSC'): (0.2, 0.1),
            ('BaseClassDB', 'pRSC'): (-0.2, 0.1),
            ('BlankClassDB', 'aRSC'): (1.2, 0.1),
            ('BlankClassDB', 'pRSC'): (0.8, 0.1),
        }

        # match points and add text annotations
        for collection in collections:
            positions = collection.get_offsets()

            for pos in positions:
                x_pos, y_pos = pos

                # Find data point with matching fraction value
                matching_rows = df_filtered.filter(
                    (pl.col('fraction') - y_pos).abs() < 0.0001
                )

                if len(matching_rows) >= 1:
                    best_match = None
                    best_distance = float('inf')

                    for row in matching_rows.iter_rows(named=True):
                        table_region = (row['table'], row['region'])
                        if table_region in expected_positions:
                            expected_x, tolerance = expected_positions[table_region]
                            distance = abs(x_pos - expected_x)

                            if distance <= tolerance and distance < best_distance:
                                best_match = row
                                best_distance = distance

                    if best_match:
                        primary_key = best_match['primary_key']
                        animal_id = primary_key.split('_')[-1] if '_' in primary_key else primary_key

                        ax.annotate(animal_id,
                                    (x_pos, y_pos),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8,
                                    alpha=0.7,
                                    ha='left')

    def verbose(self):
        if self.analysis == 'overall':
            self._verbose_overall()
        else:
            self._verbose_region()

    def _verbose_region(self):
        """Run t-tests across same animal and same region (paired t-test with the same animal)"""
        ...

    def _verbose_overall(self):
        """Run t-tests across regions within same table and across tables within same region (non-pair)"""
        df_filtered = self.df.cast({'pair_wise_group': pl.Int64})

        # compare regions within same table
        print("\n1. Comparing regions within same table:")
        tables = df_filtered['table'].unique().sort()

        for table in tables:
            table_data = (
                df_filtered
                .filter(pl.col('pair_wise_group') != -1)
                .filter(pl.col('table') == table)  # pair
                .sort('pair_wise_group')
            )
            arsc_data = table_data.filter(pl.col('region') == 'aRSC')['fraction'].to_numpy()
            prsc_data = table_data.filter(pl.col('region') == 'pRSC')['fraction'].to_numpy()

            if len(arsc_data) > 0 and len(prsc_data) > 0:
                result = wilcoxon(arsc_data, prsc_data)
                p_value = result.pvalue

                print_var(arsc_data, prefix=f'aRSC in {table}')
                print_var(prsc_data, prefix=f'pRSC in {table}')
                print('=' * 50, '\n', sep='')
                print(f"   {table}: aRSC vs pRSC")
                print(f"      p-value: {p_value:.6f}")
                print(f"      n_pairs: {min(len(arsc_data), len(prsc_data))}")

        # compare tables within same region
        print("\n2. Comparing tables within same region:")
        regions = df_filtered['region'].unique().sort()

        for region in regions:
            region_data = df_filtered.filter(pl.col('region') == region)
            generic_data = region_data.filter(pl.col('table') == 'BaseClassDB')['fraction'].to_numpy()
            blank_data = region_data.filter(pl.col('table') == 'BlankClassDB')['fraction'].to_numpy()

            if len(generic_data) > 0 and len(blank_data) > 0:
                result = mannwhitneyu(generic_data, blank_data)  # non-pair
                p_value = result.pvalue

                print(f"   {region}: BaseClassDB vs BlankClassDB")
                print(f"      p-value: {p_value:.6f}")
                print(f"      n_generic: {len(generic_data)}, n_blank: {len(blank_data)}")

        print("\n" + "=" * 50)


if __name__ == '__main__':
    BlankClassStat().main()
