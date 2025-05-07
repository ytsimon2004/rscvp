Figure 3
==========

figure 3A-3C
--------------------------
- Fig.3A & 3B: **Sequence and decoding error as given time**
- Fig.3C: **Decoding model confusion matrix**

.. seealso::

    `neuralib.model.bayes_decoding <https://neuralib.readthedocs.io/en/latest/api/_autosummary/neuralib.model.bayes_decoding.position.place_bayes.html#neuralib.model.bayes_decoding.position.place_bayes>`_


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Fig.3A & 3B: :meth:`~rscvp.model.bayes_decoding.main_decode_analysis.DecodeAnalysisOptions.plot_decoding_overview`. see ``@dispatch('overview')``
- Fig.3C: :meth:`~rscvp.model.bayes_decoding.main_decode_analysis.DecodeAnalysisOptions.plot_confusion_matrix`. see ``@dispatch('confusion_matrix')``

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/model/bayes_decoding/main_decode_analysis.py>`_



.. raw:: html

    <hr>



figure 3D
--------------------------
**95th percentile ΔF/F₀ amplitude in anterior versus posterior RSC**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.parq.main_generic_gsp.GenericGSP`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `rscvp source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_generic_gsp.py>`_



.. raw:: html

    <hr>




figure 3E
--------------------------
**Median decoding error in anterior versus posterior RSC**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^src/rscvp/
- Batch processing: :class:`~rscvp.statistic.sql.main_decode_err.MedianDecodeErrorStat`
- median decoding error (for each session): :meth:`~rscvp.model.bayes_decoding.main_decode_analysis.DecodeAnalysisOptions.plot_median_decoding_err`. see ``@dispatch('median_decode_error')``

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `rscvp source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/sql/main_decode_err.py>`_
- `median decoding error <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/model/bayes_decoding/main_decode_analysis.py>`_
