#! /bin/bash

set -e

anterior_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.behavioral.$m \
  -D 210315,210401,210402,210416,210604,210519,211202,211203,211202,221018 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  -P 0,0,0,0,0,0,,,, \
  "$@"
}

posterior_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.persistence_agg.$m \
  -D 210402,210409,210407,210409,210610,210514,211209,211208,211208,221019 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  -P 0,0,0,0,0,0,,,, \
  "$@"
}


all_persistence() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.statistic.persistence_agg.$m \
  -D 210315,210401,210402,210416,210604,210519,211202,211203,211202,221018,210402,210409,210407,210409,210610,210514,211209,211208,211208,221019 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048,YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  -P 0,0,0,0,0,0,,,,0,0,0,0,0,0,,,, \
  "$@"

}


# ------- Sorted Position Bins ---------- #

anterior_persistence main_trial_avg_position \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --group \
  --as-group 0,0,0,0,0,0,0,0,0,0 \
  --session light \
  --used_session light \
  --sort light \
  -T spks \
  --page ap_place

posterior_persistence main_trial_avg_position \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --group \
  --as-group 1,1,1,1,1,1,1,1,1,1 \
  --session light \
  --used_session light \
  --sort light \
  -T spks \
  --page ap_place

# ------- Alignment Position Bins Sort SI---------- #

anterior_persistence main_si_sorted_occ \
  --top 500

posterior_persistence main_si_sorted_occ \
  --top 500


# ------- Decoding confusion matrix ---------- #

anterior_persistence main_decode_err \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --type confusion_matrix \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place

anterior_persistence main_decode_err \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --type confusion_matrix \
  --group \
  --as-group 0,0,0,0,0,0,0,0,0,0 \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place

posterior_persistence main_decode_err \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --type confusion_matrix \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place

posterior_persistence main_decode_err \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --type confusion_matrix \
  --group \
  --as-group 0,0,0,0,0,0,0,0,0,0 \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place


# ------- Decoding position bins ---------- #

anterior_persistence main_decode_err \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --type position_bins_error \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place

anterior_persistence main_decode_err \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --type position_bins_error \
  --group \
  --as-group 0,0,0,0,0,0,0,0,0,0 \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place

posterior_persistence main_decode_err \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --type position_bins_error \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place

posterior_persistence main_decode_err \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --type position_bins_error \
  --group \
  --as-group 0,0,0,0,0,0,0,0,0,0 \
  --CV odd \
  --random 200 \
  --session light \
  --used_session light \
  -w 100 \
  --page ap_place


# ------- Topographical cell type ---------- #

all_persistence  main_topo_celltype \
  --smooth 3 \
  --ctype spatial


# ------- Motion mismatch ---------- #

anterior_persistence main_mismatch

anterior_persistence main_mismatch \
  --paired-group ctrl

posterior_persistence main_mismatch

posterior_persistence main_mismatch \
  --paired-group ctrl