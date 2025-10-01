#! /bin/bash

set -e

pf_agg() {
  local m=$1
  shift 1
  sleep 10

   python -m rscvp.statistic.$m \
  -D 210315,210401,210402,210409,210402,210407,210409,210416,210604,210610,210514,210519,211202,211209,211203,211208,211202,211208,221018,221019 \
  -A YW006,YW006,YW006,YW006,YW008,YW008,YW008,YW008,YW010,YW010,YW017,YW017,YW022,YW022,YW032,YW032,YW033,YW033,YW048,YW048 \
  -P 0,0,0,0,0,0,0,0,0,0,0,0,,,,,,,, \
  --region aRSC,aRSC,pRSC,pRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC \
  --used_session light \
  --page ap_place \
  --update \
  "$@"
}

# ----- width ----- #

pf_agg csv_agg.main_pf_agg \
  -H pf_width \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_pf_gsp \
  -H pf_width \
  --page ap_place \
  --stat-test ttest


# ----- peak ----- #

pf_agg csv_agg.main_pf_agg \
  -H pf_peak \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_pf_gsp \
  -H pf_peak \
  --page ap_place \
  --stat-test ttest


# ----- number per cell ----- #

pf_agg csv_agg.main_pf_agg \
  -H n_pf \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_pf_gsp \
  -H n_pf \
  --page ap_place \
  --stat-test ttest