import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from argclz import AbstractParser
from neuralib.imaging.suite2p import get_neuron_signal, sync_s2p_rigevent, SIGNAL_TYPE
from neuralib.io import csv_header
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_selection import SelectionOptions
from rscvp.util.cli.cli_suite2p import get_neuron_list, NeuronID

__all__ = ['EVOptions']

from stimpyp import RigEvent, RiglogData


class EVOptions(AbstractParser, SelectionOptions):
    DESCRIPTION = 'Calculate explained variance of the position in a single trial'

    signal_type: SIGNAL_TYPE = 'df_f'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('ev', self.session, use_virtual_space=self.use_virtual_space)
        self.foreach_explained_variance(output_info, self.neuron_id)

    def get_lap_event(self, rig: RiglogData) -> RigEvent:
        if self.use_virtual_space:
            return rig.get_pygame_stimlog().virtual_lap_event
        else:
            return rig.lap_event

    def foreach_explained_variance(self, output: DataOutput, neuron_ids: NeuronID):
        """
        refer to Kandler & Mao, 2017 Biorxiv
        computing the fraction of variance in single trial that is explained by the average across lap

        :return:
        """
        rig = self.load_riglog_data()
        s2p = self.load_suite_2p()
        neuron_list = get_neuron_list(s2p, neuron_ids)

        image_time = rig.imaging_event.time
        image_time = sync_s2p_rigevent(image_time, s2p, self.plane_index)

        # specify session
        lap_event = self.get_lap_event(rig)
        lap_time = lap_event.time
        session = self.get_session_info(rig, self.session)
        lap_slice = session.in_slice(lap_event.time, lap_event.value.astype(int))
        lap_time = lap_time[lap_slice]

        act_mask = session.time_mask_of(image_time)

        with csv_header(output.csv_output, ['neuron_id', f'ev_trial_avg_{self.session}']) as csv:
            for n in tqdm(neuron_list, desc='ev', unit='neuron', ncols=80):
                signal, _ = get_neuron_signal(s2p, n, signal_type=self.signal_type, dff=True, normalize=False)
                sig_trial = image_time_per_trial_inp(image_time, lap_time, signal, act_mask)  # (L, S)
                n_trial = len(sig_trial)

                ev_trial = np.zeros(n_trial)
                for trial in range(n_trial):
                    y_true = sig_trial[trial]
                    y_pred = np.mean(np.delete(sig_trial, trial, axis=0), axis=0)

                    pr = np.var(y_true)
                    pe = mean_squared_error(y_true, y_pred)
                    ev = calc_ev_position(pr, pe)
                    ev_trial[trial] = ev if ev > 0 else np.nan

                # trial average, exclude error selected roi (saturated or only small calcium transients)
                ev_trial_avg = 0 if np.all(np.isnan(ev_trial)) else np.nanmean(ev_trial, axis=0)

                csv(n, ev_trial_avg)


def calc_ev_position(pr: float, pe: float) -> float:
    """
    Calculate the explained variance of the position

    :param pr: Variance of the single trial response
    :param pe: Mean square distance between single trial responses and the across trial
    :return: Explained variance of the position in a single trial
    """
    return (pr - pe) / pr * 100


def image_time_per_trial_inp(image_time: np.ndarray,
                             lap_time: np.ndarray,
                             signal: np.ndarray,
                             act_mask: np.ndarray | None = None) -> np.ndarray:
    """
    create a mask for per trial activity

    :param image_time: (S, ) sampling
    :param lap_time: (L, )  laps
    :param signal:  (S, )
    :param act_mask: (S,)
    :return:
        signal per lap (L, S')
    """
    from scipy.interpolate import interp1d
    if not (signal.ndim == 1 and image_time.ndim == 1):
        raise ValueError('signal and image_time should be 1d array')

    if act_mask is None:
        act_mask = np.ones_like(signal, dtype=bool)
    else:
        try:
            signal[act_mask]
        except:
            raise ValueError('act_mask with wrong shape')

    image_time = image_time[act_mask]
    signal = signal[act_mask]

    x = []
    for (left_t, right_t) in (zip(lap_time[:-1], lap_time[1:])):
        xx = np.logical_and(left_t < image_time, image_time < right_t)
        x.append((np.count_nonzero(xx), image_time[xx], signal[xx]))

    bins = max([it[0] for it in x])  # time bins per lap

    return np.vstack([
        interp1d(it[1], it[2])(np.linspace(it[1][0], it[1][-1], bins))
        for it in x
    ])


if __name__ == '__main__':
    EVOptions().main()
