from argclz import AbstractParser
from neuralib.imaging.widefield.plot import plot_retinotopic_maps
from neuralib.persistence.cli_persistence import get_options_and_cache
from neuralib.util.verbose import publish_annotation
from rscvp.retinotopic.cache_retinotopic import RetinotopicCacheBuilder
from rscvp.util.cli import TreadmillOptions
from rscvp.util.cli.cli_wfield import WFieldOptions

__all__ = ['RetinotopicMapOptions']


@publish_annotation('main', project='rscvp', figure='fig.S3', as_doc=True)
class RetinotopicMapOptions(AbstractParser, WFieldOptions, TreadmillOptions):
    DESCRIPTION = 'Plot the retinotopic map based on the persistence cached data'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)

        cache = get_options_and_cache(RetinotopicCacheBuilder, self)
        output = self.get_data_output('retinotopic', output_type='wfield')
        output_file = output.summary_figure_output() if not self.debug_mode else None

        plot_retinotopic_maps(cache.trial_averaged_resp, output=output_file)
        self.locomotion_verbose()

    def locomotion_verbose(self) -> None:
        rig = self.load_riglog_data()
        n_lap = rig.lap_event.value.max().astype(int)
        speed = (self.track_length * n_lap) / rig.total_duration
        print(f'run lap: {n_lap}, average speed: {speed:.2f} cm/s')


if __name__ == '__main__':
    RetinotopicMapOptions().main()
