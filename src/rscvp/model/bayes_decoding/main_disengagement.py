import numpy as np
from rscvp.model.bayes_decoding.main_cache_bayes import ApplyBayesDecodeOptions
from rscvp.model.bayes_decoding.util import calc_wrap_distance
from rscvp.util.cli import CameraOptions
from rscvp.util.position import load_interpolated_position
from rscvp.util.util_lick import calc_lick_pos_trial

from argclz import AbstractParser
from neuralib.plot.plot import scatter_histplot

__all__ = ['DecodeDisengageOptions']


class DecodeDisengageOptions(AbstractParser, ApplyBayesDecodeOptions, CameraOptions):
    DESCRIPTION = 'To see if task engagement (i.e., lick accuracy) affect the decoding error'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        cache = self.apply_bayes_cache(version=self.cache_version)
        result = cache.load_result()

        test_mask = result.get_test_trial()
        lick_std = self._get_lick_position_std()[test_mask]

        decode_err = result.binned_decode_error
        decode_err = np.mean(decode_err, axis=1)  # average across binned (L,)

        output_info = self.get_data_output('bayes_disengagement')
        output_file = output_info.summary_figure_output(self.session, f'#{self.cache_version}')

        scatter_histplot(lick_std,
                         decode_err,
                         output=output_file,
                         xlabel='lick deviation(cm)',
                         ylabel='decoding error(cm)')

    def _get_lick_position_std(self, reward_location: int = 150) -> np.ndarray:
        """Get std of lick position for each trial"""
        rig = self.load_riglog_data()
        trial_time = rig.lap_event.time
        lick_time = rig.lick_event.time
        interp_pos = load_interpolated_position(rig)

        lick_pos = []
        iter_lick_pos = (
            calc_lick_pos_trial(interp_pos, trial_time, lick_time)
            .with_session(rig, self.session)
            .lick_position
        )

        for pos in iter_lick_pos:
            y = np.full_like(pos, reward_location)
            lick_pos.append(calc_wrap_distance(pos, y, reward_location))

        return np.array(list(map(np.std, lick_pos)))


if __name__ == '__main__':
    DecodeDisengageOptions().main()
