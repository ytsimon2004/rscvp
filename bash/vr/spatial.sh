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

OUTPUT="e:/data/user/yu-ting/analysis/phys"
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
    --vr-space \
    "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date


# ======== #
# Cellular #
# ======== #

run_python selection np

run_python selection tr \
  -s close \
  --stdf 3

run_python spatial ba \
  -T spks

run_python spatial si \
  --shuffle-times 100 \
  -s close

run_python spatial si \
  --shuffle-times 100 \
  -s open

run_python spatial pf \
  -s close \
  --threshold 0.3

run_python spatial pf \
  -s close \
  -c slb \
  --pre \
  --used_session close \
  --summary

run_python spatial slb \
  --shuffle-times 100 \
  --do-smooth \
  -s close

run_python spatial slb \
  --shuffle-times 100 \
  --do-smooth \
  -s open

run_python spatial ev \
  -s close

run_python spatial ev \
  -s open

run_python spatial tcc \
  -s close

run_python spatial tcc \
  -s open

run_python spatial ss \
  -s close \
  --run

# ================ #
# Population Level #
# ================ #

run_python spatial sa \
  -t close \
  -T df_f \
  -c slb \
  --used_session close

run_python spatial sa \
  -t close \
  -T spks \
  -c slb \
  --used_session close


run_python spatial cm \
  -x close-odd \
  -y close-even \
  -T df_f \
  -c slb \
  --pre \
  --used_session close \
  --re

run_python spatial cm \
  -x close-odd \
  -y close-even \
  -T spks \
  -c slb \
  --pre \
  --used_session close \
  --re

run_python spatial cm \
  -x open-odd \
  -y open-even \
  -T df_f \
  -c slb \
  --pre \
  --used_session close \
  --re

run_python spatial cm \
  -x open-odd \
  -y open-even \
  -T spks \
  -c slb \
  --pre \
  --used_session close \
  --re
