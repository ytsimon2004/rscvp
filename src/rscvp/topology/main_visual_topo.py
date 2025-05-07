from typing import Iterable

import numpy as np
from neuralib.util.verbose import publish_annotation

from rscvp.visual.main_polar import BaseVisPolarOptions
from rscvp.visual.util import PrefSFTFParas
from .util_plot import AbstractTopoPlotOptions, Metric

__all__ = ['VisTopoPlotOptions']


@publish_annotation('main', project='rscvp')
class VisTopoPlotOptions(AbstractTopoPlotOptions, BaseVisPolarOptions):
    DESCRIPTION = 'Plot topographical distribution for visual metrics'

    vc_selection = 0.3

    @property
    def foreach_metric(self) -> Iterable[Metric]:
        return ['vc', 'pdir', 'pori', 'dsi', 'osi']

    def _get_value(self, metric: Metric) -> np.ndarray:
        f = self.get_csv_data

        match metric:
            case 'vc':
                return f('reliability', enable_use_session=False)
            case 'sftf_0.04_1':
                return f('sftf_amp_0.04 1', enable_use_session=False)
            case 'sftf_0.04_4':
                return f('sftf_amp_0.04 4', enable_use_session=False)
            case 'sftf_0.08_1':
                return f('sftf_amp_0.08 1', enable_use_session=False)
            case 'sftf_0.08_4':
                return f('sftf_amp_0.08 4', enable_use_session=False)
            case 'sftf_0.16_1':
                return f('sftf_amp_0.16 1', enable_use_session=False)
            case 'sftf_0.16_4':
                return f('sftf_amp_0.16 4', enable_use_session=False)
            case 'pdir' | 'pori' | 'dsi' | 'osi':
                pars = PrefSFTFParas.load_from_csv(self, use_cpx_index=self.use_cpx_selective_index)
                match metric:
                    case 'pdir':
                        return pars.pref_dir
                    case 'pori':
                        return pars.pref_ori
                    case 'osi':
                        return pars.pref_osi
                    case 'dsi':
                        return pars.pref_dsi
            case _:
                raise ValueError(f'unknown {self.metric}')


if __name__ == '__main__':
    VisTopoPlotOptions().main()
