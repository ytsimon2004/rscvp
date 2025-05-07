from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from rich.pretty import pprint

from neuralib.atlas.data import load_bg_structure_tree
from neuralib.atlas.typing import Source, Area
from neuralib.plot.colormap import DiscreteColorMapper
from neuralib.typing import DataFrame
from rscvp.util.util_plot import REGION_COLORS_HIST

__all__ = ['plot_categorical_region',
           'plot_rois_bar',
           'plot_foreach_channel_pie',
           'prepare_venn_data',
           'get_acronym_fullname']


def plot_categorical_region(df: pd.DataFrame,
                            x: str | None,
                            y: str, *,
                            show_area_name: bool = True,
                            kind: str = 'bar',
                            xlabel: str = 'classified_areas',
                            **kwargs):
    """
    plot the n_cell counts / percentage in the top areas ranking by `percentage`

    :param df:
    :param x: classified based on which column in the df. if cname is None, then flatten x axis (error bar across x)
    :param y: value column
    :param show_area_name:
    :param kind:
    :param xlabel
    :return:
    """
    palette = {
        k: REGION_COLORS_HIST[k]
        for k in ['aRSC', 'pRSC', 'overlap']
    }

    g = sns.catplot(data=df, kind=kind,
                    x=x, y=y, hue='source',
                    hue_order=list(palette),
                    palette=palette)

    ax = g.ax

    if show_area_name:
        ls = get_acronym_fullname(df, x)
        ax.legend(ls, fontsize='8')

    ax.set_xlabel(xlabel)
    ax.set(**kwargs)

    ax.tick_params(axis='x', rotation=45)


def plot_rois_bar(ax: Axes,
                  x: list[str],
                  values: list[float] | pl.Series, *,
                  color: str | list[np.ndarray] = 'k',
                  alpha=1,
                  legend: bool = True,
                  fullname: bool = True,
                  **kwargs):
    """for each channel bar plot"""
    ax.bar(x=x, height=values, color=color, alpha=alpha)

    ls = get_acronym_fullname(x, sort=False) if fullname else []

    #
    if not isinstance(color, list):
        color = [color] * len(x)
    handle = [plt.Rectangle((0, 0), 1, 1, color=color[i]) for i in range(len(x))]

    #
    if legend:
        ax.legend(handle, ls, fontsize='8')
    else:
        if len(ls) != 0:
            print(ls)

    ax.set(**kwargs)
    ax.tick_params(axis='x', rotation=45)


def plot_foreach_channel_pie(ax: Axes,
                             df: DataFrame,
                             cname: str,
                             source: Source,
                             level: int,
                             cmapper: DiscreteColorMapper):
    """
    for each channel pie chart

    :param ax
    :param df:
    :param cname
    :param source: which color channel
    :param level: merge level
    :param cmapper
    :return:
    """

    ls = get_acronym_fullname(df, cname, sort=False)

    ax.pie(df['fraction'],
           labels=df[f'tree_{level}'],
           autopct='%1.1f%%', shadow=True, startangle=90, radius=2000, textprops={'fontsize': 8},
           colors=[cmapper[i] for i in df[f'tree_{level}']])

    ax.legend(ls, fontsize='8', loc='upper left')
    ax.set_title(source)
    ax.axis('equal')


def prepare_venn_data(df: pl.DataFrame,
                      region_col: str,
                      overlap_include: bool = False,
                      error_raise: bool = True) -> dict[Area, np.ndarray]:
    """
    create dict for venn plot format

    :param df:
    :param region_col: region cname. i.e., rc.classified_column
    :param overlap_include: whether overlap cells are included in rfp and gfp cell counts.
         if count overlap channel separately. then set as False
    :param error_raise: error check if negative value in n_rois counts
    :return:
        key: selected areas
        value: (3, ) corresponding cell numbers in 'aRSC', 'pRSC', 'overlap' channels, respectively
    """
    df = (
        df
        .select(region_col, 'source', 'counts')
        .sort(region_col, 'source')
        .pivot(values='counts', index=region_col, on='source', aggregate_function='first')
        .fill_null(0)
        .select([pl.col(region_col), pl.col('aRSC'), pl.col('pRSC'), pl.col('overlap')])  # reorder
    )

    if overlap_include:
        df.with_columns(
            (pl.col('aRSC') - pl.col('overlap')).alias('aRSC'),
            (pl.col('pRSC') - pl.col('overlap')).alias('pRSC')
        )

    if error_raise:
        val = df.to_numpy()[:, 1:].astype(int)  # (R, 3)
        if np.any(val < 0):
            raise RuntimeError('overlap_include arg suppose to be false')

    return {
        r[0]: np.asarray(r[1:])
        for r in df.iter_rows()
    }


def get_acronym_fullname(data: list[str] | DataFrame,
                         cname: str | None = None,
                         ret_type: Literal['list', 'dict'] = 'list',
                         sort: bool = True) -> list[str] | dict[str, str]:
    if isinstance(data, (pl.DataFrame, pd.DataFrame)) and cname is not None:
        abbr = data[cname].unique()
    elif isinstance(data, list):
        abbr = data
    else:
        raise TypeError('')

    if ret_type == 'list':
        dy = create_allen_structure_dict()
        ret = [f"{dy.get(a, 'unknown:')} ({a})" for a in abbr]
        if sort:
            ret.sort()

    elif ret_type == 'dict':

        ret = {
            create_allen_structure_dict().get(a, 'unknown:'): a
            for a in abbr
        }
        if sort:
            ret = dict(sorted(ret.items()))
    else:
        raise TypeError('')

    return ret


def create_allen_structure_dict(verbose=False) -> dict[str, str]:
    """
    Get the acronym/name pairing from structure_tree.csv

    :return: key: acronym; value: full name
    """
    tree = load_bg_structure_tree()
    tree = tree.select('name', 'acronym').sort('name')

    ret = {
        acry: name
        for name, acry in tree.iter_rows()
    }
    if verbose:
        pprint(ret)

    return ret
