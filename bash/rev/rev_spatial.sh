#!/bin/bash

set -e
set -x  # Enable shell tracing

cd ../..

if [ $# -eq 0 ]; then
  echo "$0 experimental_date animal_id ?"
  exit 1
else
  ED=$1
  ID=$2
  nP=$3
fi

OUTPUT="/mnt/bigdata/data/user/yu-ting/analysis/phys"
OUTPUT_FILE="$OUTPUT/${ED}_${ID}__2P_YW/plane${nP}/cli.log"
mkdir -p "$OUTPUT/${ED}_${ID}__2P_YW/plane${nP}"


export NO_COLOR=1

# Redirect all script output (stdout and stderr) to the log file
exec > >(tee -a "$OUTPUT_FILE") 2>&1

##
run_python() {
  local m=$1
  local a=$2
  shift 2

  echo "*** now run ${ED}_${ID}__2P_YW/plane${nP} $m $a ***"
  python -m rscvp.$m $a \
    -D "$ED" \
    -A "$ID" \
    -P "${nP}" \
    --disk bigdata \
    --mount /mnt \
    "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date


run_python selection np

run_python selection tr \
  -s light \
  --stdf 3

run_python visual vc

run_python spatial si \
  --shuffle-times 100 \
  -s light

run_python spatial ba \
  -s light

run_python spatial slb \
  --shuffle-times 100 \
  --do-smooth \
  -s light

run_python spatial tcc \
  -s light


run_python spatial pf \
  -s light \
  --threshold 0.3


# ================ #
# Population Level #
# ================ #

run_python spatial sa \
  -t light \
  -c slb \
  --used_session light


run_python spatial cm \
  -x light-odd \
  -y light-even \
  -T df_f \
  -c slb \
  --pre \
  --used_session light \
  --re

run_python spatial cm \
  -x light-odd \
  -y light-even \
  -T spks \
  -c slb \
  --pre \
  --used_session light \
  --re


run_python model.bayes_decoding analysis \
  -s "light" \
  --used_session "light" \
  --random 200 \
  --pbins 100 \
  --spatial-bin 1.5 \
  --CV odd \
  --analysis overview \
  --plot-concat \
  --load 0