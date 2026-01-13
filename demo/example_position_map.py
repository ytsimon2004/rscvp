from rscvp.spatial.main_slb import PositionLowerBoundOptions
from rscvp.util.util_demo import run_demo


class ExampleRun(PositionLowerBoundOptions):
    exp_date = '210315'
    animal_id = 'YW006'
    plane_index = 0
    neuron_id = 0
    session = 'light'
    shuffle_times = 300
    with_heatmap = True
    signal_type = 'spks'
    use_default = True
    debug_mode = True
    do_signal_smooth = True


if __name__ == '__main__':
    run_demo(ExampleRun, clean_cached=False)
