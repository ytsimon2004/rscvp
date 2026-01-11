import numpy as np

from neuralib.spikes import CASCADE_MODEL_TYPE, cascade_predict
from neuralib.suite2p import get_neuron_signal, Suite2PResult
from neuralib.util.verbose import print_save, print_load

__all__ = [
    'DEFAULT_CASCADE_MODEL',
    'get_neuron_cascade_spks'
]

DEFAULT_CASCADE_MODEL: CASCADE_MODEL_TYPE = 'Global_EXC_30Hz_smoothing100ms'


def get_neuron_cascade_spks(s2p: Suite2PResult,
                            n: int | np.ndarray | list[int] | None = None,
                            model_type: CASCADE_MODEL_TYPE | None = None,
                            cache: bool = True,
                            force_compute: bool = False,
                            **kwargs) -> np.ndarray:
    cache_file = s2p.directory / 'cascade_spks.npy'

    if cache_file.exists() and not force_compute:
        print_load(cache_file)
        return np.load(cache_file)
    else:
        dff, _ = get_neuron_signal(s2p, n, signal_type='df_f')

        if model_type is None:
            model_type = DEFAULT_CASCADE_MODEL
        spks = cascade_predict(dff, model_type=model_type, **kwargs)

        if cache:
            np.save(cache_file, spks)
            print_save(cache_file)

        return spks
