import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import wilcoxon, mannwhitneyu

from argclz import as_argument
from neuralib.plot import plot_figure
from rscvp.statistic.cli_gspread import GSPExtractor
from rscvp.statistic.core import StatPipeline, print_var
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

__all__ = ['SpatialFractionBlankStat']


class SpatialFractionBlankStat(StatPipeline):
    DESCRIPTION = 'Fraction of spatial cells in anterior v.s. posterior RSC in blank belt treadmill'

    header = as_argument(StatPipeline.header).with_options(required=False)

    load_source = 'gspread'
    sheet_name: list[GSPREAD_SHEET_PAGE] = ['GenericDB', 'BlankBeltGenericDB']

    df = None  # overwrite

    def run(self):
        self.get_dataframe()
        self.plot()
        self.verbose()

    def get_dataframe(self):
        concat = []
        for name in self.sheet_name:
            df = GSPExtractor(name).load_from_gspread(primary_key='date')
            df = (
                self._take_concat_planes(df)
                .select(['date', 'animal', 'region', 'n_selected_neurons', 'pair_wise_group', 'n_spatial_neurons'])
                .with_columns((pl.col('date') + '_' + pl.col('animal')).alias('primary_key'))
                .with_columns((pl.col('n_spatial_neurons') / pl.col('n_selected_neurons')).alias('fraction'))
                .select(['primary_key', 'region', 'fraction', 'pair_wise_group'])
                .with_columns(pl.lit(name).alias('table'))
            )

            concat.append(df)

        self.df = pl.concat(concat)

    def plot(self):
        df = self.df
        with plot_figure(None) as ax:
            sns.barplot(
                data=df,
                x="table",
                y="fraction",
                hue="region",
                errorbar="se",
                capsize=0.2,
                ax=ax
            )

            sns.stripplot(
                data=df,
                x="table",
                y="fraction",
                hue="region",
                dodge=True,
                jitter=False,
                alpha=0.5,
                palette='dark:black',
                ax=ax
            )

            # Add connected lines for paired observations
            self.plot_connect_line(ax)

            # Add legend showing primary keys
            self.add_primary_key_legend(ax)

    def plot_connect_line(self, ax: Axes):
        """Connect pairs with same pair_wise_group within each table"""
        df_filtered = (
            self.df.cast({'pair_wise_group': pl.Int64})
            .filter(pl.col('pair_wise_group') != -1)
        )

        # Get all PathCollections (stripplot points) from the axes
        collections = [child for child in ax.get_children()
                       if hasattr(child, 'get_offsets') and len(child.get_offsets()) > 0]

        print(f"Found {len(collections)} collections")

        # Match points by their y-values (fraction values) and x-position to positions
        point_positions = {}

        # Define expected x-ranges for each table-region combination
        # Based on the debug output:
        # GenericDB: aRSC around x=0.2, pRSC around x=-0.2
        # BlankBeltGenericDB: aRSC around x=1.2, pRSC around x=0.8
        expected_positions = {
            ('GenericDB', 'aRSC'): (0.2, 0.1),  # x=0.2 ± 0.1
            ('GenericDB', 'pRSC'): (-0.2, 0.1),  # x=-0.2 ± 0.1
            ('BlankBeltGenericDB', 'aRSC'): (1.2, 0.1),  # x=1.2 ± 0.1
            ('BlankBeltGenericDB', 'pRSC'): (0.8, 0.1),  # x=0.8 ± 0.1
        }

        for collection in collections:
            positions = collection.get_offsets()
            print(f"Collection has {len(positions)} points")

            # For each position, find the matching data point by both y-value and x-position
            for pos in positions:
                x_pos, y_pos = pos

                # Find data point with matching fraction value
                matching_rows = df_filtered.filter(
                    (pl.col('fraction') - y_pos).abs() < 0.0001  # Small tolerance for float comparison
                )

                # If multiple matches, use x-position to determine table/region
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
                        print(f"Matched point ({x_pos:.2f}, {y_pos:.4f}) to {key}")

        # Connect pairs within each table
        tables = df_filtered['table'].unique().sort()

        for table in tables:
            table_df = df_filtered.filter(pl.col('table') == table)

            for group_id in table_df['pair_wise_group'].unique():
                pair_df = table_df.filter(pl.col('pair_wise_group') == group_id)

                arsc_data = pair_df.filter(pl.col('region') == 'aRSC')
                prsc_data = pair_df.filter(pl.col('region') == 'pRSC')

                # Connect if we have both regions for this pair
                if len(arsc_data) == 1 and len(prsc_data) == 1:
                    arsc_pos = point_positions.get((table, 'aRSC', group_id))
                    prsc_pos = point_positions.get((table, 'pRSC', group_id))

                    if arsc_pos is not None and prsc_pos is not None:
                        ax.plot([arsc_pos[0], prsc_pos[0]], [arsc_pos[1], prsc_pos[1]],
                                color='gray', alpha=0.7, linewidth=2, zorder=10)
                        print(
                            f"Connected {table} group {group_id}: ({arsc_pos[0]:.2f},{arsc_pos[1]:.4f}) to ({prsc_pos[0]:.2f},{prsc_pos[1]:.4f})")

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
            ('GenericDB', 'aRSC'): (0.2, 0.1),
            ('GenericDB', 'pRSC'): (-0.2, 0.1),
            ('BlankBeltGenericDB', 'aRSC'): (1.2, 0.1),
            ('BlankBeltGenericDB', 'pRSC'): (0.8, 0.1),
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
                        print(f"Added annotation '{animal_id}' at ({x_pos:.2f}, {y_pos:.4f})")

    def verbose(self):
        """Run t-tests across regions within same table and across tables within same region"""
        df_filtered = (
            self.df.cast({'pair_wise_group': pl.Int64})
        )

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
            generic_data = region_data.filter(pl.col('table') == 'GenericDB')['fraction'].to_numpy()
            blank_data = region_data.filter(pl.col('table') == 'BlankBeltGenericDB')['fraction'].to_numpy()

            if len(generic_data) > 0 and len(blank_data) > 0:
                result = mannwhitneyu(generic_data, blank_data)  # non-pair
                p_value = result.pvalue

                print(f"   {region}: GenericDB vs BlankBeltGenericDB")
                print(f"      p-value: {p_value:.6f}")
                print(f"      n_generic: {len(generic_data)}, n_blank: {len(blank_data)}")

        print("\n" + "=" * 50)


if __name__ == '__main__':
    SpatialFractionBlankStat().main()
