#! /bin/bash

set -e

run_stat() {
  local m=$1
  shift 1

  sleep 10

  python -m rscvp.statistic.$m \
  --update \
  --used_session light \
  --page ap_place \
  "$@"
}

# ============ #
# Spatial Info #
# ============ #

run_stat csv_agg.auto_cli \
  -H si \
  --trunc-session \
  -s light

run_stat parq.main_value_gsp \
  -H si \
  --stat-test ttest


# ============================= #
# Trial Correlation Coefficient #
# ============================= #

run_stat csv_agg.auto_cli \
  -H trial_cc \
  --trunc-session \
  -s light

run_stat parq.main_value_gsp \
  -H trial_cc \
  --stat-test ttest

# ========================= #
# Position Explain Variance #
# ========================= #

run_stat csv_agg.auto_cli \
  -H ev_trial_avg \
  --trunc-session \
  -s light

run_stat parq.main_value_gsp \
  -H ev_trial_avg \
  --stat-test ttest

# ================= #
# Trial Reliability #
# ================= #

run_stat csv_agg.auto_cli \
  -H trial_reliability \
  --trunc-session \
  --page ap_place

run_stat parq.main_value_gsp \
  -H trial_reliability \
  --stat-test ttest

# =============== #
# ROI Coordinates #
# =============== #

run_stat csv_agg.auto_cli \
  -H ap_cords \
  --agg main_spatial_agg

run_stat csv_agg.auto_cli \
  -H ml_cords \
  --agg main_spatial_agg

run_stat csv_agg.auto_cli \
  -H dv_cords \
  --agg main_spatial_agg

run_stat csv_agg.auto_cli \
  -H ap_cords_scale \
  --agg main_spatial_agg

run_stat csv_agg.auto_cli \
  -H ml_cords_scale \
  --agg main_spatial_agg