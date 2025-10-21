

# ==================== #
# Sorted Position Bins #
# ==================== #


python -m rscvp.statistic.persistence_agg.main_trial_avg_position \
  -D 250916,250918,251018 \
  -A YW102,YW071,YW109 \
  -P ,, \
  --region aRSC,aRSC \
  --session close \
  --used_session close \
  --sort close \
  -T spks \
  --page ap_vr \
  --vr


python -m rscvp.statistic.persistence_agg.main_trial_avg_position \
  -D 250917,250919,251019 \
  -A YW102,YW071,YW109 \
  -P ,, \
  --region pRSC,pRSC \
  --session close \
  --used_session close \
  --sort close \
  -T spks \
  --page ap_vr \
  --vr