from argclz import AbstractParser
from neuralib.plot import plot_figure
from rscvp.spatial.main_cache_align_peak import ApplyAlignPeakOptions
from rscvp.spatial.main_cache_occ import ApplyPosBinActOptions
from rscvp.spatial.util_plot import plot_alignment_map
from rscvp.util.cli.cli_output import DataOutput

__all__ = ['AlignPeakMapOptions']


class AlignPeakMapOptions(AbstractParser, ApplyPosBinActOptions, ApplyAlignPeakOptions):
    DESCRIPTION = 'Plot aligned place field location map based on spatial information'

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        output_info = self.get_data_output('am', use_virtual_space=self.use_virtual_space)
        self.plot_align_map(output_info)

    def plot_align_map(self, output: DataOutput):
        """Plot the position alignment map (trial average response for all cells), can be sorted by `si`.
        refer to Esteves et al., 2022. JN paper (McNaughton group)
        """
        sig = self.apply_align_peak_cache().trial_avg_binned_data

        output_file = output.summary_figure_output(
            'pre' if self.pre_selection else None,
            self.session,
            self.pc_selection if self.pc_selection is not None else None,
            self.signal_type,
            self.with_top if self.with_top is not None else 'all'
        )

        with plot_figure(output_file, 2, 1, tight_layout=False) as axes:
            plot_alignment_map(sig,
                               self.signal_type,
                               track_length=self.track_length,
                               select_top=self.with_top,
                               axes=axes)


if __name__ == '__main__':
    AlignPeakMapOptions().main()
