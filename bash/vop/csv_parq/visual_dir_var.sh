#! /bin/bash

set -e

run_stat() {
  local m=$1
  shift 1

  sleep 10

  python -m rscvp.statistic.$m \
  --update \
  --used_session light \
  "$@"
}

# ===================================== #
# Direction/Orientation Selective Index #
# ===================================== #

run_stat csv_agg.auto_cli \
  -H dsi

run_stat parq.main_visual_dir_gsp \
  -H dsi \
  --stat-test kstest

run_stat csv_agg.auto_cli \
  -H osi

run_stat parq.main_visual_dir_gsp \
  -H osi \
  --stat-test kstest


# =============================== #
# Preferred Direction/Orientation #
# =============================== #

run_stat csv_agg.auto_cli \
  -H pdir

run_stat parq.main_visual_dir_gsp \
  -H pdir \
  --stat-test kstest

run_stat csv_agg.auto_cli \
  -H pori

run_stat parq.main_visual_dir_gsp \
  -H pori \
  --stat-test kstest

