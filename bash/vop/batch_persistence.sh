

# ==================== #
# Sorted Position Bins #
# ==================== #


python -m rscvp.statistic.persistence_agg.main_trial_avg_position \
  -D 210315,210401,210402,210416,210604,210519,211202,211203,211202,221018 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  -P 0,0,0,0,0,0,,,, \
  --region aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC,aRSC \
  --group \
  --as-group 0,0,0,0,0,0,0,0,0,0 \
  --session light \
  --used_session light \
  --sort light \
  -T spks \
  --page ap_place


 python -m rscvp.statistic.persistence_agg.main_trial_avg_position \
  -D 210402,210409,210407,210409,210610,210514,211209,211208,211208,221019 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  -P 0,0,0,0,0,0,,,, \
  --region pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC,pRSC \
  --group \
  --as-group 1,1,1,1,1,1,1,1,1,1 \
  --session light \
  --used_session light \
  --sort light \
  -T spks \
  --page ap_place