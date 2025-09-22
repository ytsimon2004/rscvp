#! /bin/bash

set -e

generic_agg() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.$m \
  -D 210315,210401,210402,210409,210402,210407,210409,210416,210604,210610,210514,210519,211202,211209,211203,211208,211202,211208,221018,221019 \
  -A YW006,YW006,YW006,YW006,YW008,YW008,YW008,YW008,YW010,YW010,YW017,YW017,YW022,YW022,YW032,YW032,YW033,YW033,YW048,YW048 \
  -P 0,0,0,0,0,0,0,0,0,0,0,0,,,,,,,, \
  --region aRSC,aRSC,pRSC,pRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC \
  -s light \
  --used_session light \
  --page ap_generic \
  --update \
  "$@"
}

# ------- speed score ---------- #

generic_agg csv_agg.main_generic_agg \
 -H speed_score \
 --trunc-session


python -m rscvp.statistic.parq.main_speed_gsp \
  -H speed_score \
  --page ap_generic \
  --vis


# ------- perc95_dff ---------- #

generic_agg csv_agg.main_generic_agg \
  -H perc95_dff

python -m rscvp.statistic.parq.main_generic_gsp \
  -H perc95_dff \
  -s light \
  --stat-test kstest \
  --para
