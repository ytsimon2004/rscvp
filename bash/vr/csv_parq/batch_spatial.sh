#! /bin/bash

set -e

spatial_agg() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.$m \
  -D 250916,250917,250918,250919,251018,251019 \
  -A YW102,YW102,YW071,YW071,YW109,YW109 \
  -P ,,,,, \
  "$@"
}


# ----- si ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H si \
  --trunc-session \
  -s close \
  --used_session close \
  --page ap_vr \
  --update

python -m rscvp.statistic.parq.main_value_gsp \
  -H si \
  --page ap_vr \
  --stat-test ttest


# ----- tcc ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H trial_cc \
  --trunc-session \
  -s close \
  --used_session close \
  --page ap_vr \
  --update

python -m rscvp.statistic.parq.main_value_gsp \
  -H trial_cc \
  --page ap_vr \
  --stat-test ttest