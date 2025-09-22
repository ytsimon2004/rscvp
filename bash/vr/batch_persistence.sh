

# ==================== #
# Sorted Position Bins #
# ==================== #


python -m rscvp.statistic.persistence_agg.main_trial_avg_position \
  -D 250916,250918 \
  -A YW102,YW071 \
  -P , \
  --region aRSC,aRSC \
  --group \
  --as-group 0,0 \
  --session close \
  --used_session close \
  --sort close \
  -T spks \
  --page ap_vr \
  --vr


python -m rscvp.statistic.persistence_agg.main_trial_avg_position \
  -D 250917,250919 \
  -A YW102,YW071 \
  -P , \
  --region pRSC,pRSC \
  --group \
  --as-group 0,0 \
  --session close \
  --used_session close \
  --sort close \
  -T spks \
  --page ap_vr \
  --vr