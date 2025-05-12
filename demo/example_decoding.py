from rscvp.model.bayes_decoding.main_decode_analysis import DecodeAnalysisOptions
from rscvp.selection.main_neuropil_error import NeuropilErrOptions
from rscvp.selection.main_trial_reliability import TrialReliabilityOptions
from rscvp.util.util_demo import run_demo
from rscvp.visual.main_reliability import VisualReliabilityOptions

# contact author since paper is not published yet
TOKEN = ...


class ExampleNeuropil(NeuropilErrOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    use_default = True
    reuse_output = True


class ExampleTrialReliability(TrialReliabilityOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    session = 'light'
    use_default = True
    reuse_output = True


class ExampleVisualReliability(VisualReliabilityOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    signal_type = 'df_f'
    use_default = True


class ExampleDecoding(DecodeAnalysisOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    session = 'light'
    used_session = 'light'
    random = 200
    window = 100
    spatial_bin_size = 1.5
    cross_validation = 'odd'
    pre_selection = True
    signal_type = 'df_f'
    use_default = True
    debug_mode = True
    cache_version = 0
    analysis_type = 'overview'  # try `confusion_matrix` also
    plot_concat_time = True


if __name__ == '__main__':
    # run preselection of neuron pipeline
    run_demo(ExampleNeuropil, token=TOKEN, clean_cached=False)
    run_demo(ExampleTrialReliability, token=TOKEN, clean_cached=False)
    run_demo(ExampleVisualReliability, token=TOKEN, clean_cached=False)

    # decoding pipeline
    run_demo(ExampleDecoding, token=TOKEN, clean_cached=False)
