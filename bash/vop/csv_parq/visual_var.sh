#! /bin/bash

set -e

run_stat() {
  local m=$1
  shift 1

  sleep 10

  python -m rscvp.statistic.$m \
  --used_session light \
  --update \
  "$@"
}

# ================== #
# Visual Reliability #
# ================== #

run_stat csv_agg.auto_cli \
  -H reliability

run_stat parq.main_visual_gsp \
  -H reliability \
  --stat-test kstest \
  --hist-norm

# ============== #
# Visual Max DFF #
# ============== #

run_stat csv_agg.auto_cli \
  -H max_vis_resp

run_stat parq.main_visual_gsp \
  -H max_vis_resp \
  --stat-test kstest

# ================= #
# Visual Perc95 DFF #
# ================= #

run_stat csv_agg.auto_cli \
  -H perc95_vis_resp

run_stat parq.main_visual_gsp \
  -H perc95_vis_resp \
  --stat-test kstest \
  --hist-norm \
  --cutoff 1000

# =============== #
# ROI Coordinates #
# =============== #

run_stat csv_agg.auto_cli \
  -H ap_cords \
  --agg main_visual_agg \
  --page ap_vz

run_stat csv_agg.auto_cli \
  -H ml_cords \
  --agg main_visual_agg \
  --page ap_vz

run_stat csv_agg.auto_cli \
  -H dv_cords \
  --agg main_visual_agg \
  --page ap_vz

run_stat csv_agg.auto_cli \
  -H ap_cords_scale \
  --agg main_visual_agg \
  --page ap_vz

run_stat csv_agg.auto_cli \
  -H ml_cords_scale \
  --agg main_visual_agg \
  --page ap_vz
