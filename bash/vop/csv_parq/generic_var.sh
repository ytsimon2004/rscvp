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

# ============== #
# Percentile DFF #
# ============== #

run_stat csv_agg.auto_cli \
  -H perc95_dff \
  -s light

run_stat parq.main_generic_gsp \
  -H perc95_dff \
  -s light \
  --stat-test kstest \
  --para
