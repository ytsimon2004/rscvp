#! /bin/bash

set -e

spatial_agg() {
  local m=$1
  shift 1
  sleep 60

   python -m rscvp.statistic.$m \
  -D 211210,220322,211207,220325,220901,220902,221216,221215,230113 \
  -A YW022,YW033,YW032,YW036,YW045,YW045,YW048,YW048,YW049 \
  -P ,,,,0,,,, \
  --used_session light_bas \
  --page dark_parq \
  --update \
  "$@"
}


# ----- si ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H si \
  -s light_bas

spatial_agg csv_agg.main_spatial_agg \
  -H si \
  -s dark

spatial_agg csv_agg.main_spatial_agg \
  -H si \
  -s light_end

python -m rscvp.statistic.parq.main_value_gsp \
  -H si \
  --page dark_parq \
  --group-var session \
  --stat-test pairwise_ttest \
  --plot diag


# ----- tcc ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H trial_cc \
  -s light_bas

spatial_agg csv_agg.main_spatial_agg \
  -H trial_cc \
  -s dark

spatial_agg csv_agg.main_spatial_agg \
  -H trial_cc \
  -s light_end

python -m rscvp.statistic.parq.main_value_gsp \
  -H trial_cc \
  --page dark_parq \
  --group-var session \
  --stat-test pairwise_ttest \
  --plot diag

# ----- roi coordinates ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H ap_cords

spatial_agg csv_agg.main_spatial_agg \
  -H ml_cords

spatial_agg csv_agg.main_spatial_agg \
  -H dv_cords

spatial_agg csv_agg.main_spatial_agg \
  -H ap_cords_scale

spatial_agg csv_agg.main_spatial_agg \
  -H ml_cords_scale
