from rscvp.util.util_demo import run_demo
from rscvp.visual.main_tuning import VisualTuningOptions


class ExampleRun(VisualTuningOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    neuron_id = 2
    signal_type = 'df_f'
    invalid_cache = True
    use_default = True
    debug_mode = True


if __name__ == '__main__':
    run_demo(ExampleRun, clean_cached=False)
