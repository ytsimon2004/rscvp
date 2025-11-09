#! /bin/bash

set -e

spatial_agg() {
  local m=$1
  shift 1
  sleep 10

   python -m rscvp.statistic.$m \
  -D 210315,210401,210402,210409,210402,210407,210409,210416,210604,210610,210514,210519,211202,211209,211203,211208,211202,211208,221018,221019 \
  -A YW006,YW006,YW006,YW006,YW008,YW008,YW008,YW008,YW010,YW010,YW017,YW017,YW022,YW022,YW032,YW032,YW033,YW033,YW048,YW048 \
  -P 0,0,0,0,0,0,0,0,0,0,0,0,,,,,,,, \
  --region aRSC,aRSC,pRSC,pRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC \
  --used_session light \
  --page spatial_parq \
  --update \
  "$@"
}


# ----- si ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H si \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_value_gsp \
  -H si \
  --page spatial_parq \
  --stat-test ttest


# ----- speed_score ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H speed_score \
  --trunc-session \
  -s light

spatial_agg csv_agg.main_spatial_agg \
  -H speed_score_run \
  --trunc-session \
  -s light


# ----- tcc ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H trial_cc \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_value_gsp \
  -H trial_cc \
  --page spatial_parq \
  --stat-test ttest


# ----- ev ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H ev_trial_avg \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_value_gsp \
  -H ev_trial_avg \
  --page spatial_parq \
  --stat-test ttest


# ----- tr ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H trial_reliability \
  --trunc-session \
  -s light

python -m rscvp.statistic.parq.main_value_gsp \
  -H liability \
  --page spatial_parq \
  --stat-test ttest


# ----- roi coordinates ----- #

spatial_agg csv_agg.main_spatial_agg \
  -H ap_cords

spatial_agg csv_agg.main_spatial_agg \
  -H ml_cords

spatial_agg csv_agg.main_spatial_agg \
  -H dv_cords

spatial_agg csv_agg.main_spatial_agg \
  -H ap_cords_scale

spatial_agg csv_agg.main_spatial_agg \
  -H ml_cords_scale