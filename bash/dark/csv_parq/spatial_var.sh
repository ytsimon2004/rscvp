#! /bin/bash
set -e

run_stat() {
  local m=$1
  shift 1

  sleep 30

  python -m rscvp.statistic.$m \
  --update \
  --used_session light_bas \
  "$@"
}

# ============ #
# Spatial Info #
# ============ #

run_stat csv_agg.auto_cli \
  -H si \
  -s light_bas \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H si \
  -s dark \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H si \
  -s light_end \
  --page ap_ldl

# ============================= #
# Trial Correlation Coefficient #
# ============================= #

run_stat csv_agg.auto_cli \
  -H trial_cc \
  -s light_bas \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H trial_cc \
  -s dark \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H trial_cc \
  -s light_end \
  --page ap_ldl


# =============== #
# ROI Coordinates #
# =============== #

run_stat csv_agg.auto_cli \
  -H ap_cords \
  --agg main_spatial_agg \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H ml_cords \
  --agg main_spatial_agg \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H dv_cords \
  --agg main_spatial_agg \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H ap_cords_scale \
  --agg main_spatial_agg \
  --page ap_ldl

run_stat csv_agg.auto_cli \
  -H ml_cords_scale \
  --agg main_spatial_agg \
  --page ap_ldl