#! /bin/bash

set -e

run_stat() {
  shift 1

  python -m rscvp.statistic.persistence_agg.auto_cli \
  "$@"
}


# ================== #
# sorted_pos_bin_ldl #
# ================== #

run_stat \
  -T sorted_pos_bin_ldl \
  --session light \
  --used_seesion light \
  -R aRSC

run_stat \
  -T sorted_pos_bin_ldl \
  --session dark \
  --used_seesion light \
  -R aRSC


run_stat \
  -T sorted_pos_bin_ldl \
  --session light \
  --used_seesion light \
  -R pRSC

run_stat \
  -T sorted_pos_bin_ldl \
  --session dark \
  --used_seesion light \
  -R pRSC
