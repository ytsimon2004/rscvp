Figure 2
==========

figure 2A
--------------------------
**Left: Examples of normalized deconvolved calcium activity**

**Right: Proportion of position-tuned neurons in anterior versus posterior RSC**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Left: :class:`~rscvp.spatial.main_position_map.PositionMapOptions`
- Right: :class:`~rscvp.statistic.sql.main_vp_fraction.VisSpaFracStat`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Left: `example position cell source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_position_map.py>`_
- Right: `batch processing source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/sql/main_vp_fraction.py>`_



.. raw:: html

    <hr>



figure 2B
--------------------------
**Trial-averaged deconvolved ∆F/F0 for all neurons**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.persistence_agg.main_trial_avg_position.PositionBinPersistenceAgg`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `rscvp source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/persistence_agg/main_trial_avg_position.py>`_



.. raw:: html

    <hr>




figure 2C
--------------------------
**Spatial distribution of mean spatial information**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Batch processing: :class:`~rscvp.statistic.parq.main_topo_metric_gsp.TopoMetricOptions`
- Spatial information (for each cell): :class:`~rscvp.spatial.main_si.SiOptions`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `batch processing source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_topo_metric_gsp.py>`_
- `spatial information source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_si.py>`_




.. raw:: html

    <hr>



figure 2D-2F
--------------------------
**Cumulative distributions and animals' comparing**

- See ``self.animal_based_comp`` (``--animals-comp`` in argparse) for animal wise comparison


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Batch processing: :class:`~rscvp.statistic.parq.main_value_gsp.ValStatGSP`
- Position field width (for each cell): :class:`~rscvp.spatial.main_place_field.PlaceFieldsOptions`
- Spatial information (for each cell): :class:`~rscvp.spatial.main_si.SiOptions`
- Trial Correlation (for each cell): :class:`~rscvp.spatial.main_trial_corr.TrialCorrOptions`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `batch processing source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_value_gsp.py>`_
- `position field source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_place_field.py>`_
- `spatial information source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_si.py>`_
- `trial correlation source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_trial_corr.py>`_




figure 2G
--------------------------
**Fraction of neurons with number of place fields**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Batch processing: :class:`~rscvp.statistic.parq.main_pf_gsp.PFStatGSP`
- Position field numbers (for each cell): :class:`~rscvp.spatial.parq.main_place_field.PlaceFieldsOptions`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `batch processing source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_pf_gsp.py>`_
