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
  -t light_bas \
  -c slb \
  --used_session light_bas

run_python spatial sa \
  -t dark \
  --sort light_bas \
  -c slb \
  --used_session light_bas

run_python spatial cm \
  -x light-bas \
  -y dark \
  -T spks \
  -c slb \
  --pre \
  --used_session light_bas \
  --re

run_python spatial cm \
  -x dark \
  -y light-end \
  -T spks \
  -c slb \
  --pre \
  --used_session light_bas \
  --re

run_python spatial pf \
  -s light_bas \
  -c slb \
  --pre \
  --used_session light_bas \
  --summary

run_python spatial pf \
  -s dark \
  -c slb \
  --pre \
  --used_session light_bas \
  --summary

run_python spatial am \
  -s light_bas \
  --pre \
  --used_session light_bas \
  -c slb

run_python spatial am \
  -s dark \
  --pre \
  --used_session light_bas \
  -c slb \
  --re

# ============ #
# Bayes Decode #
# ============ #

#run_python model.bayes_decoding analysis \
#  --type median_decode_error \
#  -s light_bas \
#  --used_session light_bas \
#  --random 200 \
#  --CV odd \
#  --load 0 \
#  --commit
#
#run_python model.bayes_decoding analysis \
#  --type position_bins_error \
#  -s light_bas \
#  --used_session light_bas \
#  --random 200 \
#  --CV odd \
#  --load 0

# ========= #
# Selection #
# ========= #

run_python selection cls \
  -s light_bas \
  --us light_bas \
  --commit