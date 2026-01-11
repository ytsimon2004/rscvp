import numpy as np
from typing import Literal

from argclz import AbstractParser, argument, int_tuple_type, union_type
from neuralib.plot import plot_figure
from neuralib.suite2p import dff_signal, Suite2PResult, sync_s2p_rigevent
from rscvp.util.cli import StimpyOptions, Suite2pOptions


class RawTraceOptions(AbstractParser, StimpyOptions, Suite2pOptions):
    trace_type: Literal['dff', 'spks', 'position'] = argument(
        '--trace'
    )

    time_range: tuple[int, int] = argument('--time', type=int_tuple_type)

    n_neurons: int | tuple[int, ...] = argument('--n', type=union_type(int, int_tuple_type))

    randomized: bool = argument('--randomized')

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        s2p = self.load_suite_2p()
        rig = self.load_riglog_data()
        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        trace = self.get_trace(s2p)
        self.plot_trace(trace)

    def neuron_list(self, s2p: Suite2PResult):
        match self.n_neurons, self.randomized:
            case int(), True:
                return np.random.randint(s2p.n_neurons, size=self.n_neurons)
            case int(), False:
                return np.arange(self.n_neurons)
            case tuple(), False:
                return np.array(self.n_neurons)
            case _:
                raise ValueError('')

    def time_mask(self):
        return slice(*self.time_range)

    def get_trace(self, s2p: Suite2PResult) -> np.ndarray:
        match self.trace_type:
            case 'dff':
                nx = self.neuron_list(s2p)
                tx = self.time_mask()
                f = s2p.f_raw[nx, tx]
                f_neu = s2p.f_neu[nx, tx]
                return dff_signal(f, f_neu, s2p).dff
            case 'spks':
                nx = self.neuron_list(s2p)
                tx = self.time_mask()
                t = s2p.spks[nx, tx]
                return t
            case 'position':
                pass

    def plot_trace(self, trace):
        n_traces = trace.shape[0]
        y = 0
        with plot_figure(None) as ax:
            for i in range(n_traces):
                ax.plot(trace[i] + y)
                y += np.max(trace[i])


if __name__ == '__main__':
    RawTraceOptions().main()
