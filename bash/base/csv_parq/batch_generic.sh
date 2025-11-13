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
  --used_session light \
  --page generic_parq \
  --update \
  "$@"
}

# ------- speed score ---------- #

generic_agg csv_agg.main_generic_agg \
 -H speed_score \
 -s light \
 --trunc-session


python -m rscvp.statistic.parq.main_speed_gsp \
  -H speed_score \
  --page generic_parq


# ------- perc95_dff light ---------- #

generic_agg csv_agg.main_generic_agg \
  -s light \
  -H perc95_dff

python -m rscvp.statistic.parq.main_generic_gsp \
  -H perc95_dff \
  -s light \
  --stat-test kstest \
  --para

# ------- perc95_dff visual ---------- #

generic_agg csv_agg.main_generic_agg \
  -s visual \
  -H perc95_dff

python -m rscvp.statistic.parq.main_generic_gsp \
  -H perc95_dff \
  -s visual \
  --stat-test kstest \
  --para