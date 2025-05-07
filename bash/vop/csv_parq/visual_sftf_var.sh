#! /bin/bash

set -e

run_stat() {
  local m=$1
  shift 1

  sleep 50

  python -m rscvp.statistic.$m \
  --used_session light \
  --update \
  "$@"
}


# ============ #
# SF/TF groups #
# ============ #

run_stat csv_agg.auto_cli \
  -H 'sftf_amp_0.04 1'

run_stat csv_agg.auto_cli \
  -H 'sftf_amp_0.04 4'

run_stat csv_agg.auto_cli \
  -H 'sftf_amp_0.08 1'

run_stat csv_agg.auto_cli \
  -H 'sftf_amp_0.08 4'

run_stat csv_agg.auto_cli \
  -H 'sftf_amp_0.16 1'

run_stat csv_agg.auto_cli \
  -H 'sftf_amp_0.16 4'


# =================== #
# DFF versus Fraction #
# =================== #

run_stat csv_agg.auto_cli \
  -H dff

run_stat csv_agg.auto_cli \
  -H fraction