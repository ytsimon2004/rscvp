Figure S2
==========

figure S2A
--------------------------
**Position decoding with cross-validation**

- First save cached and run with another customized package: `posdc <https://github.com/ytsimon2004/posdc>`_


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.model.bayes_decoding.main_posdc_cache.PositionDecodeCacheBuilder`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/model/bayes_decoding/main_posdc_cache.py>`_



.. raw:: html

    <hr>


figure S2B
--------------------------
**Median decoding error across recording sessions**

- Based on `posdc <https://github.com/ytsimon2004/posdc>`_ output


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.posdc.main_session_median_err.SessionMedianErr`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/posdc/main_session_median_err.py>`_




.. raw:: html

    <hr>


figure S2C
--------------------------
**Fraction of position-tuned neurons across sessions**

- Based on `posdc <https://github.com/ytsimon2004/posdc>`_ output


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.sql.main_spatial_frac_ldl.SpatialFracLDLStat`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/sql/main_spatial_frac_ldl.py>`_




.. raw:: html

    <hr>


figure S2D
--------------------------
**Pairwise scatter plots and histogram**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.parq.main_value_gsp.ValStatGSP`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_value_gsp.py>`_



.. raw:: html

    <hr>


figure S2E
--------------------------
**Trial-averaged, deconvolved ∆F/F0 for all position-tuned neurons**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- :class:`~rscvp.statistic.persistence_agg.main_trial_avg_position.PositionBinPersistenceAgg`


source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `rscvp source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/persistence_agg/main_trial_avg_position.py>`_



.. raw:: html

    <hr>


figure S2F
--------------------------
**Dorsal cortical map showing mean spatial information and mean trial-to-trial correlation across the RSC**


core API
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Batch processing: :class:`~rscvp.statistic.parq.main_topo_metric_gsp.TopoMetricOptions`
- Spatial information (for each cell): :class:`~rscvp.spatial.main_si.SiOptions`
- Trial Correlation (for each cell): :class:`~rscvp.spatial.main_trial_corr.TrialCorrOptions`

source code
^^^^^^^^^^^^^^^^^^^^^^^^^^
- `batch processing source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/statistic/parq/main_topo_metric_gsp.py>`_
- `spatial information source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_si.py>`_
- `trial correlation source code <https://github.com/ytsimon2004/rscvp/blob/main/src/rscvp/spatial/main_trial_corr.py>`_
