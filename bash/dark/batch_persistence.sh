#! /bin/bash

set -e

anterior_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.behavioral.$m \
  -D 211210,220322,211207,220325,220901,221216 \
  -A YW022,YW033,YW032,YW036,YW045,YW048 \
  -P ,,,,0, \
  "$@"
}

posterior_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.persistence_agg.$m \
  -D 220902,221215,230113 \
  -A YW045,YW048,YW049 \
  -P ,, \
  "$@"
}



# ------- Sorted Position Bins ---------- #

anterior_persistence main_trial_avg_position \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --group \
  --as-group 0,0,0,0,0,0 \
  --session light_bas \
  --used_session light_bas \
  --sort light_bas \
  -T spks \
  --page ap_ldl


posterior_persistence main_trial_avg_position \
  --region pRSC,pRSC,pRSC \
  --group \
  --as-group 1,1,1 \
  --session light_bas \
  --used_session light_bas \
  --sort light_bas \
  -T spks \
  --page ap_ldl