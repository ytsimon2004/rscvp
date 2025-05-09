ROI Dorsal Cortex View
=======================

.. code-block:: python

    from demo.util import mkdir_test_dataset
    from neuralib.util.utils import ensure_dir
    from rscvp.atlas.main_roi_top_view import RoiTopViewOptions
    from rscvp.util.io import RSCVP_CACHE_DIRECTORY


    class ExampleRun(RoiTopViewOptions):
        SOURCE_ROOT = ensure_dir(RSCVP_CACHE_DIRECTORY) / 'rscvp_dataset' / 'analysis' / 'hist'
        animal = ('YW043',)
        area_family = 'ISOCORTEX'
        legend_number_limit = 20


    def main():
        mkdir_test_dataset()
        ExampleRun().main()
        # clean_cache_dataset() clean all if needed


    if __name__ == '__main__':
        main()



.. seealso::

    - `demo script link <https://github.com/ytsimon2004/rscvp/blob/main/demo/example_dorsal_roi.py>`_

**Run with**

.. code-block:: bash

    $ python -m demo.example_decoding

