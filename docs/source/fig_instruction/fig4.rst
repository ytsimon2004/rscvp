Figure 4
==========


figure 4A
--------------------------
**Example ∆F/F0 time courses used for visual reliability assessment**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.visual.main_reliability.VisualReliabilityOptions`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/visual/main_reliability.py>`_



.. raw:: html

    <hr>


figure 4B
--------------------------
**Distribution of visual reliability index for all recorded neurons across anterior and posterior RSC**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.parq.main_visual_gsp.VisStatGSP`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_visual_gsp.py>`_


.. raw:: html

    <hr>



figure 4C
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



figure 4D
--------------------------
- Left: **Spatial distribution of mean visual reliability**
- Right: **Proportion of visually-evoked neurons in anterior versus posterior RSC**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Visual reliability (batch processing): :class:`~rscvp.statistic.parq.main_topo_metric_gsp.TopoMetricOptions`
- Visual reliability (for each cell): :class:`~rscvp.visual.main_reliability.VisualReliabilityOptions`
- Proportion of cell: :class:`~rscvp.statistic.sql.main_vp_fraction.VisSpaFracStat`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `batch processing source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_topo_metric_gsp.py>`_
- `visual reliability <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/visual/main_reliability.py>`_
- `proportion of cell source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/sql/main_vp_fraction.py>`_
