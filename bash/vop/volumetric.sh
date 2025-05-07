#! /bin/bash

set -e
set -x

cd ../..

if [ $# -eq 0 ]; then
  echo "$0 experimental_date animal_id ?"
  exit 1
else
  ED=$1
  ID=$2
fi

OUTPUT="e:/data/user/yu-ting/analysis/phys"
OUTPUT_FILE="$OUTPUT/${ED}_${ID}__2P_YW/concat_etl.log"

export NO_COLOR=1

# Redirect all script output (stdout and stderr) to the log file
exec > >(tee -a "$OUTPUT_FILE") 2>&1

##
run_python() {
  local m=$1
  local a=$2
  shift 2

  echo "*** now run ${ED}_${ID}__2P_YW/concat $m $a ***"
  python -m rscvp.$m $a \
    -D "$ED" \
    -A "$ID" \
    "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date

##

# ============== #
# Spatial Module #
# ============== #

run_python spatial sa \
  -t light \
  -c slb \
  --used_session light

run_python spatial sa \
  -t dark \
  --sort light \
  -c slb \
  --used_session light

run_python spatial cm \
  -x light-odd \
  -y light-even \
  -T df_f \
  -c slb \
  --pre \
  --used_session light

run_python spatial cm \
  -x light-odd \
  -y light-even \
  -T spks \
  -c slb \
  --pre \
  --re \
  --used_session light

run_python spatial cm \
  -x dark-odd \
  -y dark-even \
  -T spks \
  -c slb \
  --pre \
  --used_session light \
  --re

run_python spatial cm \
  -x light \
  -y dark \
  -T spks \
  -c slb \
  --pre \
  --used_session light \
  --re

run_python spatial pf \
  -s light \
  -c slb \
  --pre \
  --used_session light \
  --summary

run_python spatial pf \
  -s dark \
  -c slb \
  --pre \
  --used_session light \
  --summary

run_python spatial am \
  -s light \
  --pre \
  --used_session light \
  -c slb

run_python spatial am \
  -s dark \
  --pre \
  --used_session light \
  -c slb \
  --re

run_python selection cls \
  -s light \
  --us light \
  --commit

# ============ #
# Bayes Decode #
# ============ #

run_python model.bayes_decoding analysis \
  --type median_decode_error \
  -s light \
  --used_session light \
  --random 200 \
  --CV odd \
  --load 0 \
  --commit

run_python model.bayes_decoding analysis \
  --type position_bins_error \
  -s light \
  --used_session light \
  --random 200 \
  --CV odd \
  --load 0


# ============= #
# Visual Module #
# ============= #

run_python visual pa \
  --summary \
  --used_session light \
  --commit

run_python visual st \
  --summary dff \
  --pre \
  --vr 0.3 \
  --used_session light \
  --commit

run_python visual st \
  --summary fraction \
  --pre \
  --vr 0.3 \
  --used_session light \
  --commit