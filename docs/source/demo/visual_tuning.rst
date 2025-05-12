Visual Response Tuning
========================

.. code-block:: python

    from rscvp.util.util_demo import run_demo
    from rscvp.visual.main_tuning import VisualTuningOptions

    # contact author since paper is not published yet
    TOKEN = ...


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
        run_demo(ExampleRun, token=TOKEN, clean_cached=False)





.. seealso::

    - `demo script link <https://github.com/ytsimon2004/rscvp/blob/main/demo/example_visual_tuning.py>`_


**Run with**

.. code-block:: bash

    $ python -m demo.example_visual_tuning