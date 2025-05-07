#! /bin/bash

set -e

run_stat() {
  shift 1

  python -m rscvp.statistic.persistence_agg.auto_pipeline \
  "$@"
}


# ================== #
# sorted_pos_bin_ldl #
# ================== #

run_stat \
  -T sorted_pos_bin_ldl \
  --session light_bas \
  --used_seesion light_bas \
  -R aRSC

run_stat \
  -T sorted_pos_bin_ldl \
  --session dark \
  --used_seesion light_bas \
  -R aRSC

run_stat \
  -T sorted_pos_bin_ldl \
  --session light_end \
  --used_seesion light_bas \
  -R aRSC


run_stat \
  -T sorted_pos_bin_ldl \
  --session light_bas \
  --used_seesion light_bas \
  -R pRSC

run_stat \
  -T sorted_pos_bin_ldl \
  --session dark \
  --used_seesion light_bas \
  -R pRSC

run_stat \
  -T sorted_pos_bin_ldl \
  --session light_end \
  --used_seesion light_bas \
  -R pRSC