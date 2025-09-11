from pathlib import Path
from typing import Literal, Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import wilcoxon

from argclz import AbstractParser, argument, as_argument, str_tuple_type, var_argument
from argclz.dispatch import Dispatch, dispatch
from neuralib.atlas.typing import Source, Area
from neuralib.plot import plot_figure, dotplot
from neuralib.typing import AxesArray
from neuralib.util.verbose import publish_annotation
from rscvp.atlas.util_plot import plot_categorical_region
from rscvp.statistic.core import print_var
from rscvp.util.cli import HistOptions, PlotOptions, ROIOptions
from rscvp.util.util_plot import REGION_COLORS_HIST

__all__ = ['RoiQuantBatchOptions']


@publish_annotation('main',
                    project='rscvp',
                    figure=['fig.6F', 'fig.6H', 'fig.7A-G bar', 'fig.7H'],
                    as_doc=True)
class RoiQuantBatchOptions(AbstractParser, ROIOptions, PlotOptions, Dispatch):
    """ROIs analysis for batch animals, see ``@dispatch(...) on method`` for different plot types"""
    DESCRIPTION = 'ROIs analysis for batch animals'

    animal = as_argument(HistOptions.animal).with_options(
        type=str_tuple_type,
        help='multiple animals. e.g. YW001,YW002'
    )

    area: Area = as_argument(HistOptions.area).with_options(type=str, help='a single brain area')

    dispatch_plot: Literal[
        'region_x',
        'region_overall',
        'expr_scatter',
        'bias_index',
        'heatmap',
        'hemi_diff',
        'family_stacked'
    ] = argument('--plot', required=True, help='analysis type')

    plot_args: list[str] = var_argument('ARGS')

    disable_overlap_in_plot = True
    _source: tuple[Source, ...] | None = None

    def post_parsing(self):
        """post argument parsing settings"""
        self.setup_logger(Path(__file__).name)
        self.set_background()

        #
        if self.disable_overlap_in_plot:
            self._source = ('aRSC', 'pRSC')
        else:
            self._source = ('aRSC', 'pRSC', 'overlap')

    def run(self):
        """Runs the main execution logic of the object"""
        self.post_parsing()
        self.invoke_command(self.dispatch_plot, *self.plot_args)

    @property
    def n_animals(self) -> int:
        """number of animals"""
        return len(self.animal)

    @property
    def output_file(self) -> Path | None:
        """output file path. If debug mode then return None"""
        if self.debug_mode:
            return None

        file = f'{self.dispatch_plot}_norm-{self.roi_norm}_{self.hemisphere}'
        if self.area is not None:
            file += f'_{self.area}'
        ret = self.SOURCE_ROOT / file

        return ret.with_suffix('.pdf')

    # set for loop
    _classified_column: str = None
    _value_col: str = None
    _normalized_unit: str = None

    def _concat_classified_dataframe(self, area: Area | None = None) -> pl.DataFrame:
        """Concat classified data for batch animals analysis"""
        ret = []
        for ccf_dir in self.foreach_ccf_dir():
            norm = (
                self.load_roi_dataframe(ccf_dir)
                .to_normalized(self.roi_norm,
                               self.merge_level,
                               top_area=self.top_area,
                               hemisphere=self.hemisphere,
                               animal=ccf_dir.animal)
            )

            if not ret:
                self._classified_column = norm.classified_column
                self._value_col = norm.value_column
                self._normalized_unit = norm.normalized_unit
            #
            if area is not None:
                norm = norm.filter_areas(area)

            if self.disable_overlap_in_plot:
                norm = norm.filter_sources(['aRSC', 'pRSC'])

            ret.append(norm.dataframe())

        return pl.concat(ret)

    # ======== #
    # Region X #
    # ======== #

    @dispatch('region_x')
    def plot_categorical_region(self, flatten: bool = True):
        """
        Plot a specific region bar/cat plot across animals (x axis)

        :param flatten: If True, plot all animals with connected line, otherwise categorical group plot foreach animal
        """
        if self.area is None:
            raise ValueError('specify region')

        df = self._concat_classified_dataframe(self.area)

        with plot_figure(self.output_file, figsize=(2.5, 6)) as ax:
            if flatten:
                order = self._source
                palette = {k: REGION_COLORS_HIST[k] for k in order}

                sns.stripplot(df, x='source', y=self._value_col, hue='animal', jitter=False, order=order, size=7.5,
                              ax=ax)
                sns.barplot(df, x='source', y=self._value_col, hue='source', palette=palette, order=order,
                            errorbar='se', ax=ax)
                self._plot_connect_points(ax, df)
                self._verbose(df)

                ax.set(ylabel=self._normalized_unit)
                ax.set_title(self.area)
            else:
                plot_categorical_region(df, x='animal', y=self._value_col, ylabel=self._normalized_unit,
                                        xlabel='animal', show_area_name=False, title=self.area)

    def _verbose(self, df: pl.DataFrame):

        arsc_data = df.filter(pl.col('source') == 'aRSC').sort('animal')[self._value_col].to_numpy()
        prsc_data = df.filter(pl.col('source') == 'pRSC').sort('animal')[self._value_col].to_numpy()

        stat = wilcoxon(arsc_data, prsc_data)
        print(f'statistic of {self.area} in {self.roi_norm} normalized:', stat)

        for src in df.partition_by('source'):
            print_var(src[self._value_col], prefix=f"{src['source'].unique().item()}")

    def _plot_connect_points(self, ax, df):
        """Connect same-animal points across source categories with a line."""
        animals = df.select('animal').unique().to_series().to_list()

        for animal in animals:
            sub_df = df.filter(pl.col('animal') == animal)

            if sub_df.select('source').n_unique() < 2:
                continue

            sub_df = sub_df.sort(by=pl.col('source').replace({s: i for i, s in enumerate(self._source)}))
            x_vals = [self._source.index(s) for s in sub_df['source']]
            y_vals = sub_df[self._value_col]

            ax.plot(x_vals, y_vals, color='gray', alpha=0.4, zorder=1)

    # =========== #
    # Overall Cat #
    # =========== #

    @dispatch('region_overall')
    def plot_categorical_overall(self):
        """Plot categorical data for multiple regions in batch animals"""
        sns.set_style("whitegrid")

        df = self._concat_classified_dataframe().to_pandas()
        g = sns.catplot(data=df, x='source', y=self._value_col, hue='animal', col=self._classified_column,
                        order=self._source, s=60, jitter=False, height=5, aspect=0.25)
        #
        g.map_dataframe(sns.barplot,
                        x='source',
                        y=self._value_col,
                        hue='source',
                        alpha=.5,
                        linewidth=2.5,
                        palette={c: REGION_COLORS_HIST[c] for c in self._source},
                        err_kws={"color": ".5", "linewidth": 2.0},
                        errorbar='se')

        g.set_axis_labels("", f'{self._normalized_unit}')
        g.set_xticklabels(self._source)
        g.tick_params(axis='x', labelrotation=45)
        g.set_titles("{col_name}")
        g.despine(left=True)

        if self.output_file is None:
            plt.show()
        else:
            plt.savefig(self.output_file)

    # ================== #
    # Expression Scatter #
    # ================== #

    @dispatch('expr_scatter')
    def plot_expression_dataset(self):
        """Plot scatter foreach source in paired animals"""
        if len(self.animal) != 2:
            raise ValueError('Only provide single pair comparison')

        df = self._concat_classified_dataframe()
        nc = 2 if self.disable_overlap_in_plot else 3
        with plot_figure(None, 1, nc, figsize=(10, 5)) as ax:
            a1 = df.filter(pl.col('animal') == self.animal[0])
            a2 = df.filter(pl.col('animal') == self.animal[1])

            for i, src in enumerate(self._source):
                x = a1.filter(pl.col('source') == src).select(self._classified_column, self._value_col)
                y = a2.filter(pl.col('source') == src).select(self._classified_column, self._value_col)
                xy = x.join(y, on=self._classified_column).to_numpy()

                self._plot_scatter(ax[i],
                                   xy[:, 1].astype(float),
                                   xy[:, 2].astype(float),
                                   a=xy[:, 0],
                                   c=REGION_COLORS_HIST[src],
                                   xlabel=f'{self.animal[0]}_{src}',
                                   ylabel=f'{self.animal[1]}_{src}')

    @staticmethod
    def _plot_scatter(ax: Axes,
                      x: np.ndarray,
                      y: np.ndarray,
                      *,
                      with_linear_fit: bool = True,
                      with_label: bool = True,
                      a: np.ndarray | None = None,
                      c: str | None = None,
                      log_axis: bool = True,
                      **kwargs):

        ax.scatter(x, y, marker='x', color=c if c is not None else 'k')
        ax.axline((0.5, 0.5), slope=1, color='k', alpha=0.5, lw=1)

        if with_linear_fit:
            from scipy.stats import linregress
            res = linregress(x, y)
            r_sqrt = res.rvalue ** 2

            if not log_axis:
                ax.plot(x, res.intercept + res.slope * x, 'r', label='linear fitting')

            ax.set_title(f'linear R-squared: {r_sqrt:.4f}')

        if with_label:
            for i, label in enumerate(a):
                ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        if log_axis:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
        ax.set(**kwargs)

    # ========== #
    # Bias_index #
    # ========== #

    @dispatch('bias_index')
    def plot_bias_index(self, common: bool = True,
                        plot_type: Literal['bar', 'heatmap'] = 'heatmap',
                        exclude: tuple[Area, ...] | None = ('DP', 'AI', 'TEa', 'RSP')) -> None:
        """
        :param common: only left common area existed in all animals
        :param plot_type: bar plot or heatmap plot
        :param exclude: area(s) not to shown in the plot (i.e., not informative)
        """

        self.roi_norm = 'channel'

        df = self._get_bias_value_dataframe()

        if common:
            df = df.drop_nulls()

        # sort based on mean bias index across animals
        df = df.with_columns(
            df.select(self.animal).mean_horizontal().alias('mean_bias_index')
        ).sort('mean_bias_index')

        df = df.unpivot(index=self._classified_column,
                        on=self.animal,
                        variable_name='animal',
                        value_name='bias_index')
        #
        if exclude is not None:
            df = df.filter(pl.col(self._classified_column).is_in(exclude).not_())

        #
        with plot_figure(None, figsize=(8, 3)) as ax:
            match plot_type:
                case 'bar':
                    self._bias_index_barplot(ax, df)
                case 'heatmap':
                    self._bias_index_heatmap(ax, df)
                case _:
                    raise ValueError(f'Unknown plot type: {plot_type}')

    def _get_bias_value_dataframe(self) -> pl.DataFrame:
        ret = []
        for i, ccf_dir in enumerate(self.foreach_ccf_dir()):
            norm = (self.load_roi_dataframe(ccf_dir)
                    .to_normalized(self.roi_norm, self.merge_level, top_area=self.top_area, hemisphere=self.hemisphere))

            if not ret:
                self._classified_column = norm.classified_column

            df = norm.to_bias_index('pRSC', 'aRSC').rename({'bias_index': f'{self.animal}'})
            ret.append(df)

        return pl.concat(ret, how='align').sort(self.animal[0])  # sort by bias index in first animal

    def _bias_index_barplot(self, ax, df):
        sns.barplot(df, ax=ax, x=self._classified_column, y='bias_index', color='k', alpha=0.6, errorbar='se')
        sns.stripplot(df, ax=ax, x=self._classified_column, y='bias_index', hue='animal', jitter=False)

    def _bias_index_heatmap(self, ax, df):
        n_areas = len(df[self._classified_column].unique())

        areas = []
        dat = np.zeros((n_areas, self.n_animals))
        for i, val in enumerate(df.partition_by(self._classified_column)):
            areas.append(val[self._classified_column].unique().item())
            dat[i] = val['bias_index'].to_numpy()

        norm = mcolors.TwoSlopeNorm(vmin=np.min(dat), vcenter=0, vmax=np.max(dat))
        im = ax.imshow(dat.T, cmap=sns.diverging_palette(230, 20, n=200, as_cmap=True), norm=norm)
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel('bias_index')

        ax.set_xticks(np.arange(n_areas), areas, rotation=45)
        ax.set_yticks(np.arange(self.n_animals), self.animal)

    # ======= #
    # HeatMap #
    # ======= #

    @dispatch('heatmap')
    def plot_heatmap_batch_animals(self, as_dot: bool = True,
                                   rotate_xy: bool = False,
                                   cbar_norm_type: Literal['log', 'power', 'none'] = 'none'):
        """plot all the areas with heatmap, rows(animal), columns(classified brain areas),
        colorbar are shared and plot with logarithmic, empty(0) indicates no count for cells

        :param as_dot: plot also as dot size, otherwise, regular heatmap
        :param rotate_xy: transpose xy in plotting
        :param cbar_norm_type: color bar normalization type
        """
        data = self._prepare_heatmap_data()

        vmin, vmax = 0.0001, float(self._vmax)  # small vmin

        match cbar_norm_type:
            case 'log':
                cbar_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            case 'power':
                cbar_norm = mcolors.PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)
            case 'none':
                cbar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            case _:
                raise ValueError(f'{cbar_norm_type} not supported')

        #
        if rotate_xy:
            nrow, ncol = 1, len(self._source)
            plot_kwargs = dict(sharey=True)
        else:
            nrow, ncol = len(self._source), 1
            plot_kwargs = dict(sharex=True)

        with plot_figure(None, nrow, ncol, default_style=False, **plot_kwargs) as ax:
            for i, df in enumerate(data):

                x = df[self._classified_column].to_list()
                val = df.select(self.animal).to_numpy()

                if as_dot:
                    val[np.isnan(val)] = 0
                    self._dotplot(ax, i, x, val, cbar_norm, rotate_xy)
                else:
                    self._imshow(ax, i, x, val, cbar_norm, rotate_xy)

    def _dotplot(self,
                 axes: AxesArray,
                 i: int,
                 x: list[str],
                 val: np.ndarray,
                 cbar_norm: mcolors.Normalize,
                 rotate_xy: bool):

        ax = axes[i]
        y = self.animal

        if rotate_xy:
            val = val.T
            y = x
            x = self.animal

        dotplot(x, y, val, with_color=True, norm=cbar_norm, scale='area', ax=ax, colorbar_title=self._normalized_unit)
        ax.set_title(self._source[i])

        if not rotate_xy:
            ax.set_xticks(np.arange(len(x)), x, rotation=45)

    def _imshow(self,
                axes: AxesArray,
                i: int,
                x: list[str],
                val: np.ndarray,
                cbar_norm: mcolors.Normalize,
                rotate_xy: bool):

        ax = axes[i]
        val = val if rotate_xy else val.T
        im = ax.imshow(val, norm=cbar_norm, cmap='cividis')
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel(f'{self._normalized_unit}')
        ax.set_title(self._source[i])

        #
        if rotate_xy:
            ax.set_xticks(np.arange(self.n_animals), self.animal, rotation=90)
            ax.set_yticks(np.arange(len(x)), x, rotation=45)
        else:
            ax.set_xticks(np.arange(len(x)), x, rotation=45)
            ax.set_yticks(np.arange(self.n_animals), self.animal)

    _vmax: float = None  # for shared color map
    _area_set = set()

    def _prepare_heatmap_data(self, sharex: bool = True) -> list[pl.DataFrame]:
        """

        :param sharex: if sharex, throughout each source, area shows the same order.
        :return:
        """
        df = self._concat_classified_dataframe()
        self._vmax = df[self._value_col].max()

        ret: list[pl.DataFrame] = []
        for i, src in enumerate(self._source):
            pvt = (
                df.filter(pl.col('source') == src)
                .select(self._classified_column, self._value_col, 'animal')
                .pivot(on='animal',
                       values=self._value_col,
                       index=self._classified_column,
                       aggregate_function='first')
                .fill_null(np.nan)
            )

            if sharex:
                area = pvt[self._classified_column].to_list()
                self._area_set = self._area_set.union(area)
                order = sorted(list(self._area_set))
                pvt = pvt.with_columns(pl.Series([order.index(a) for a in area]).alias('index')).sort('index')

            ret.append(pvt)

        return ret

    # =============== #
    # hemisphere diff #
    # =============== #

    @dispatch('hemi_diff')
    def plot_hemisphere_diff(self):
        """Plot the expression different across hemisphere, foreach animal/source"""
        with plot_figure(None, 2, self.n_animals, sharey=True) as ax:
            for i, (animal, dat) in enumerate(self._foreach_hemisphere_data()):
                ipsi = dat.filter(pl.col('hemisphere') == 'ipsi')
                contra = dat.filter(pl.col('hemisphere') == 'contra')

                for j, src in enumerate(['aRSC', 'pRSC']):
                    ipsi_src = ipsi.filter(pl.col('source') == src)
                    ipsi_val = ipsi_src[self._value_col]
                    ipsi_idx = ipsi_src[self._classified_column]

                    contra_src = contra.filter(pl.col('source') == src)
                    contra_val = contra_src[self._value_col].cast(pl.Int64)  # for later multiplication
                    contra_idx = contra_src[self._classified_column]

                    ax[j, i].barh(ipsi_idx, ipsi_val, align='center', color='k')
                    ax[j, i].barh(contra_idx, contra_val * -1, align='center', color='r')
                    ax[j, i].set_title(f'{animal}_{src}')
                    ax[j, i].set(xlabel=self._normalized_unit)

    def _foreach_hemisphere_data(self) -> Iterable[tuple[str, pl.DataFrame]]:
        self.hemisphere = 'ipsi'
        df_ipsi = self._concat_classified_dataframe()

        self.hemisphere = 'contra'
        df_contra = self._concat_classified_dataframe()

        df = pl.concat([df_ipsi, df_contra])

        for animal, dat in df.group_by(['animal']):
            yield animal[0], dat

    # ============== #
    # Family stacked #
    # ============== #

    STACKED_COLOR_LIST = ['cadetblue', 'gold', 'dimgray', 'salmon', 'peru', 'lightsteelblue']
    FAMILY_ORDER = ['ISOCORTEX', 'HPF', 'TH', 'CTXSP', 'OLF', 'CNU', 'HY', 'MB', 'HB', 'CB']

    @dispatch('family_stacked')
    def plot_family_stacked(self, foreach_animal: bool = True):
        """
        Plot the stacked bar of AllenFamily foreach animal, sources

        :param foreach_animal: if true,  plot foreach animals, otherwise, averaging all animals
        """
        self.roi_norm = 'channel'
        self.merge_level = 'family'

        df = self._concat_classified_dataframe()

        if foreach_animal:
            n_animals = df['animal'].n_unique()

            with plot_figure(None, 2, n_animals) as ax:
                for i, (animal, dat) in enumerate(df.group_by(['animal'])):

                    for j, src in enumerate(['aRSC', 'pRSC']):
                        arsc = dat.filter(pl.col('source') == src)

                        x = arsc['family']
                        y = arsc[self._value_col]

                        # follow the same order/color code
                        family = [f for f in self.FAMILY_ORDER if f in x]
                        idx = [list(x).index(f) for f in family]
                        x = x[idx]
                        y = y[idx]

                        bottom = 0
                        for k, val in enumerate(y):
                            ax[j, i].bar(f'{animal[0]}_{src}',
                                         val,
                                         bottom=bottom,
                                         color=self.STACKED_COLOR_LIST[k] if k < len(self.STACKED_COLOR_LIST) else None,
                                         label=x[k])
                            ax[j, i].legend()

                            bottom += val

        else:
            df = df.group_by(['family', 'source']).agg([
                pl.col("counts").mean(),
                pl.col(self._value_col).mean()
            ])

            with plot_figure(None, 2, 1) as ax:
                for i, (src, dat) in enumerate(df.group_by(['source'])):
                    x = dat['family']
                    y = dat[self._value_col]

                    family = [f for f in self.FAMILY_ORDER if f in x]
                    idx = [list(x).index(f) for f in family]
                    x = x[idx]
                    y = y[idx]

                    bottom = 0
                    for k, val in enumerate(y):
                        ax[i].bar(f'{src}', val, bottom=bottom,
                                  color=self.STACKED_COLOR_LIST[k] if k < len(self.STACKED_COLOR_LIST) else None,
                                  label=x[k])
                        ax[i].legend()
                        bottom += val


if __name__ == '__main__':
    RoiQuantBatchOptions().main()
