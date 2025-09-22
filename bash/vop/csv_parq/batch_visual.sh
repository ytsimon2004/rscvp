#! /bin/bash

set -e

visual_agg() {
  local m=$1
  shift 1
  sleep 10

   python -m rscvp.statistic.$m \
  -D 210315,210401,210402,210409,210402,210407,210409,210416,210604,210610,210514,210519,211202,211209,211203,211208,211202,211208,221018,221019 \
  -A YW006,YW006,YW006,YW006,YW008,YW008,YW008,YW008,YW010,YW010,YW017,YW017,YW022,YW022,YW032,YW032,YW033,YW033,YW048,YW048 \
  -P 0,0,0,0,0,0,0,0,0,0,0,0,,,,,,,, \
  --region aRSC,aRSC,pRSC,pRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,pRSC,aRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC,aRSC,pRSC \
  --used_session light \
  --page ap_vz \
  --update \
  "$@"
}

# ------- reliability ---------- #

visual_agg csv_agg.main_visual_agg \
  -H reliability


python -m rscvp.statistic.parq.main_visual_gsp \
  -H reliability \
  --stat-test kstest \
  --hist-norm


# ------- max dF/F ---------- #

visual_agg csv_agg.main_visual_agg \
  -H max_vis_resp

python -m rscvp.statistic.parq.main_visual_gsp \
  -H max_vis_resp \
  --stat-test kstest


# ------- 95 perc dF/F ---------- #

visual_agg csv_agg.main_visual_agg \
  -H perc95_vis_resp

python -m rscvp.statistic.parq.main_visual_gsp \
  -H perc95_vis_resp \
  --stat-test kstest \
  --hist-norm \
  --cutoff 1000


# ------- coordinates ---------- #

visual_agg csv_agg.main_visual_agg \
  -H ap_cords

visual_agg csv_agg.main_visual_agg \
  -H ml_cords

visual_agg csv_agg.main_visual_agg \
  -H dv_cords

visual_agg csv_agg.main_visual_agg \
  -H ap_cords_scale

visual_agg csv_agg.main_visual_agg \
  -H ml_cords_scale