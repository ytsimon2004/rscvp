import numpy as np

from neuralib.util.verbose import publish_annotation
from .util_plot import AbstractTopoPlotOptions, Metric

__all__ = ['SpatialTopoPlotOptions']


@publish_annotation('main', project='rscvp')
class SpatialTopoPlotOptions(AbstractTopoPlotOptions):
    DESCRIPTION = 'Plot topographical distribution for spatial metrics'

    pc_selection = 'slb'

    @property
    def foreach_metric(self) -> list[Metric]:
        return ['si', 'tcc', 'ev']

    def _get_value(self, metric: Metric) -> np.ndarray:
        """csv query for all cell (WITHOUT selection)"""
        match metric:
            case 'si':
                col = f'si_{self.session}'
            case 'tcc':
                col = f'trial_cc_{self.session}'
            case 'ev':
                col = f'ev_trial_avg_{self.session}'
            case _:
                raise ValueError(f'{metric}')

        return self.get_csv_data(col)


if __name__ == '__main__':
    SpatialTopoPlotOptions().main()
