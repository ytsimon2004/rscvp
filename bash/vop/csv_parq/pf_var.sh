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

# ======== #
# PF width #
# ======== #

run_stat csv_agg.auto_cli \
  -H pf_width \
   --trunc-session \
  -s light \
  --page ap_place

run_stat parq.main_pf_gsp \
  -H pf_width \
  --stat-test ttest


# ======== #
# PF Peaks #
# ======== #

run_stat csv_agg.auto_cli \
  -H pf_peak \
  --trunc-session \
  -s light \
  --page ap_place

run_stat parq.main_pf_gsp \
  -H pf_peak \
  --stat-test ttest

# ============ #
# number of PF #
# ============ #

run_stat csv_agg.auto_cli \
  -H n_pf \
  --trunc-session \
  -s light \
  --page ap_place

run_stat parq.main_pf_gsp \
  -H n_pf \
  --stat-test ttest
