from rscvp.util.util_demo import run_demo
from rscvp.visual.main_sftf_pref import VisualSFTFPrefOptions

# contact author since paper is not published yet
TOKEN = '...'


class ExampleRun(VisualSFTFPrefOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    neuron_id = 2
    signal_type = 'df_f'
    use_default = True
    debug_mode = True


if __name__ == '__main__':
    run_demo(ExampleRun, token=TOKEN, clean_cached=False)
