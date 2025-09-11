from typing import Final, Literal

import brainrender
import numpy as np
from brainrender import Scene
from brainrender.actors import Volume

from argclz import argument
from neuralib.atlas.brainrender.core import CAMERA_ANGLE_TYPE
from neuralib.atlas.util import allen_to_brainrender_coord
from rscvp.statistic.core import StatPipeline

__all__ = ['TopoMetricVolumeOptions']


class TopoMetricVolumeOptions(StatPipeline):
    DESCRIPTION = 'Topological volumetric view of value average for the batch dataset'

    plot_type: Literal['top', 'front', 'side', '3d'] = argument(
        '--plot',
        default='3d',
        help='projection type'
    )

    camera_3d: CAMERA_ANGLE_TYPE = argument(
        '--camera-3d',
        default='three_quarters',
        help='camera angle in 3d view'
    )

    bin_size: int = argument('--bin', default=50, help='bin size for 3d histogram (um)')
    scaled: bool = argument('--scaled', help='If use scaled coordinates for ap/ml')

    load_source: Final = 'parquet'

    ap_field: str
    ml_field: str
    dv_field: str

    def post_parsing(self):
        super().post_parsing()

        self.ap_field = 'ap_cords'
        self.ml_field = 'ml_cords'
        self.dv_field = 'dv_cords'

        if self.scaled:
            self.ap_field += '_scale'
            self.ml_field += '_scale'

    def run(self):
        """overwrite"""
        self.post_parsing()
        self.load_table(to_pandas=False)
        self.plot()

    @property
    def weight(self) -> np.ndarray:
        return np.concatenate(self.df[self.header].to_numpy())

    def plot(self):
        ap, dv, ml = self._compute_coordinates()

        match self.plot_type:
            case 'top':
                dv = np.zeros_like(ap)
                camera = 'top'
            case 'front':
                ap = np.zeros_like(dv)
                camera = 'frontal'
            case 'side':
                ml = np.zeros_like(ap)
                camera = 'sagittal'
            case '3d':
                camera = self.camera_3d
            case _:
                raise ValueError(f'invalid plot type: {self.plot_type}')

        ap_edges, dv_edges, ml_edges = self._compute_edge(ap, dv, ml)
        w = self.weight

        data, edges = np.histogramdd(
            (ap, dv, ml),
            bins=(ap_edges, dv_edges, ml_edges),
            weights=w,
        )

        count, _ = np.histogramdd(
            (ap, dv, ml),
            bins=(ap_edges, dv_edges, ml_edges),
        )

        self.render_3d(data, edges, count, camera)

    def render_3d(self, data: np.ndarray,
                  edges: tuple[np.ndarray, ...],
                  count: np.ndarray,
                  camera: CAMERA_ANGLE_TYPE,
                  no_value: bool = True,
                  show_dorsal_layer_only: bool = True):
        data_mean = np.divide(data, count, where=count > 0)

        if no_value:
            data_mean = np.ones_like(data_mean)

        data_mean[count == 0] = np.nan

        brainrender.settings.SHOW_AXES = False
        scene = Scene(root=False, inset=False)
        actor = Volume(data_mean, voxel_size=self.bin_size, cmap='inferno')

        ox, oy, oz = edges[0][0], edges[1][0], edges[2][0]
        actor.shift(ox, oy, oz)
        scene.add(actor)

        a = 0.5
        if show_dorsal_layer_only:
            region = {
                'RSPd1': 'lightblue',
                'RSPd2/3': 'pink',
                # 'RSPd4': 'lightgreen',
                'RSPd5': 'turquoise',
                'RSPd6a': 'green',
                'RSPd6b': 'orange'
            }

            camera = 'sagittal2'

        else:
            region = {
                'RSPd': 'lightblue',
                'RSPv': 'pink',
                'RSPagl': 'turquoise'
            }

        for r, c in region.items():
            scene.add_brain_region(r, alpha=a, color=c, hemisphere='left')

        brainrender.settings.DEFAULT_CAMERA = camera

        scene.render(camera=camera)

    def _compute_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """convert to brainrender coordinates space"""
        df = self.df[self.header, self.ap_field, self.ml_field, self.dv_field]
        ap = np.concatenate(df[self.ap_field].to_numpy())
        ml = np.concatenate(df[self.ml_field].to_numpy())
        dv = np.concatenate(df[self.dv_field].to_numpy())

        coords = np.vstack([ap, dv, ml]).T / 1000
        coords = allen_to_brainrender_coord(coords)
        ap, dv, ml = coords[:, 0], coords[:, 1], coords[:, 2]

        return ap, dv, ml

    def _compute_edge(self, ap, dv, ml) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """compute edge for 3d histogram"""
        b = self.bin_size

        def _projection_edge(c: np.ndarray):
            return np.array([c.min(), c.min() + b]) \
                if c.max() == c.min() \
                else np.arange(c.min(), c.max() + b, b)

        ap_edges = _projection_edge(ap)
        dv_edges = _projection_edge(dv)
        ml_edges = _projection_edge(ml)

        return ap_edges, dv_edges, ml_edges


if __name__ == '__main__':
    TopoMetricVolumeOptions().main()
