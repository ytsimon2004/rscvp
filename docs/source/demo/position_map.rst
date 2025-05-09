Position Map
===============

.. code-block:: python

    from demo.util import mkdir_test_dataset
    from rscvp.spatial.main_slb import PositionLowerBoundOptions


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


    def main():
        mkdir_test_dataset()
        ExampleRun().main()
        # clean_cache_dataset() clean all if needed


    if __name__ == '__main__':
        main()


.. seealso::

    - `demo script link <https://github.com/ytsimon2004/rscvp/blob/main/demo/example_position_map.py>`_

**Run with**

.. code-block:: bash

    $ python -m demo.example_position_map

