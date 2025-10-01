#! /bin/bash

set -e


pf_agg() {
  local m=$1
  shift 1
  sleep 20

  python -m rscvp.statistic.$m \
  -D 250916,250917,250918,250919 \
  -A YW102,YW102,YW071,YW071 \
  -P ,,, \
  "$@"
}

# ----- width ----- #

pf_agg csv_agg.main_pf_agg \
  -H pf_width \
  --trunc-session \
  -s close \
  --used_session close \
  --page ap_vr \
  --update

python -m rscvp.statistic.parq.main_pf_gsp \
  -H pf_width \
  --page ap_vr \
  --stat-test ttest


# ----- peak ----- #

pf_agg csv_agg.main_pf_agg \
  -H pf_peak \
  --trunc-session \
  -s close \
  --used_session close \
  --page ap_vr \
  --update

python -m rscvp.statistic.parq.main_pf_gsp \
  -H pf_peak \
  --page ap_vr \
  --stat-test ttest