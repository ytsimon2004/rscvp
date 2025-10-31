#! /bin/bash

set -e

anterior_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.persistence_agg.$m\
  -D 210601,210505,210423,211126,211112,220930,220908,211208,210507 \
  -A YW006,YW010,YW017,YW032,YW033,YW048,YW049,YW036,YW018  \
  -P 0,0,0,,,,,,0 \
  "$@"
}

posterior_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.persistence_agg.$m \
  -D 210430,210427,221004,220909,211209,210512  \
  -A YW010,YW017,YW048,YW049,YW036,YW018  \
  -P 0,0,,,,0 \
  "$@"
}


# ------- Sorted Position Bins ---------- #

anterior_persistence main_trial_avg_position \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --session light \
  --used_session light \
  -T spks \
  --page apcls_blank \
  --cv

posterior_persistence main_trial_avg_position \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --session light \
  --used_session light \
  -T spks \
  --page apcls_blank \
  --cv

