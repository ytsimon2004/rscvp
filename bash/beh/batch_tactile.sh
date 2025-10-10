#! /bin/bash

set -e

all_agg() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.behavioral.$m \
  -D 210315,210401,210402,210409,210402,210407,210409,210416,210604,210610,210514,210519,211202,211209,211203,211208,211202,211208,221018,221019 \
  -A YW006,YW006,YW006,YW006,YW008,YW008,YW008,YW008,YW010,YW010,YW017,YW017,YW022,YW022,YW032,YW032,YW033,YW033,YW048,YW048 \
  "$@"

}


anterior_agg() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.behavioral.$m \
  -D 210315,210401,210402,210416,210604,210519,211202,211203,211202,221018 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  "$@"
}


posterior_agg() {
  local m=$1
  shift 1
  sleep 10

  python -m rscvp.behavioral.$m \
  -D 210402,210409,210407,210409,210610,210514,211209,211208,211208,221019 \
  -A YW006,YW006,YW008,YW008,YW010,YW017,YW022,YW032,YW033,YW048 \
  "$@"
}



# ----- vstim running ------ #

all_agg main_vstim_locomotion \
  --direction 0 \
  --collapse

all_agg main_vstim_locomotion \
  --direction 180 \
  --collapse

anterior_agg main_vstim_locomotion \
  --disk BigDATA \
  --mount /Volumes \
  --collapse

posterior_agg main_vstim_locomotion \
  --disk BigDATA \
  --mount /Volumes \
  --collapse

# Peri-Reward Lick
#python -m rscvp.behavioral.main_batch \
#  -D 210315,210401,210402,210402,210407,210409,210513,210514,210519,211202,211202,211203,211208,211208,211209,221018,221019 \
#  -A YW006,YW006,YW006,YW008,YW008,YW006,YW017,YW017,YW017,YW022,YW033,YW032,YW032,YW033,YW022,YW048,YW048 \
#  --type peri_reward_lick \
#  -t 100,100,100,107,80,79,80,100,70,45,130,80,90,45,60,70,120