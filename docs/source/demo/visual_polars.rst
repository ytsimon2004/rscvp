Visual Direction Polars
========================

.. code-block:: python

    class ExampleRun(VisualPolarOptions):
        exp_date = '210315'
        animal_id = 'YW006'
        plane_index = 0
        neuron_id = 2
        signal_type = 'df_f'
        use_default = True
        debug_mode = True


    def main():
        mkdir_test_dataset()
        ExampleRun().main()
        # clean_cache_dataset() clean all if needed


    if __name__ == '__main__':
        main()



.. seealso::

    - `demo script link <https://github.com/ytsimon2004/rscvp/blob/main/demo/example_visual_polars.py>`_

