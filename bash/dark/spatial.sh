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
    "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date

# ========== #
# Pre-select #
# ========== #

run_python selection np

run_python selection tr \
  -s light_bas \
  --stdf 3

run_python selection tr \
  -s dark \
  --stdf 3

run_python selection tr \
  -s light_end \
  --stdf 3

# ======== #
# Cellular #
# ======== #

run_python spatial si \
  --shuffle-times 100 \
  -s light_bas

run_python spatial si \
  --shuffle-times 100 \
  -s dark

run_python spatial si \
  --shuffle-times 100 \
  -s light_end

run_python spatial spr \
  -s light_bas

run_python spatial spr \
  -s dark

run_python spatial spr \
  -s light_end

run_python spatial ba \
  -s light_bas

run_python spatial slb \
  --shuffle-times 100 \
  --do-smooth \
  -s light_bas

run_python spatial slb \
  --shuffle-times 100 \
  --do-smooth \
  -s dark

run_python spatial slb \
  --shuffle-times 100 \
  --do-smooth \
  -s light_end

run_python spatial ev \
  -s light_bas

run_python spatial ev \
  -s dark

run_python spatial ev \
  -s light_end

run_python spatial tcc \
  -s light_bas

run_python spatial tcc \
  -s dark

run_python spatial tcc \
  -s light_end

# =========== #
# Place Field #
# =========== #

run_python spatial pf \
  -s light_bas \
  --threshold 0.3

run_python spatial pf \
  -s light_bas \
  -c slb \
  --pre \
  --used_session light_bas \
  --summary

run_python spatial pf \
  -s dark \
  --threshold 0.3

run_python spatial pf \
  -s dark \
  -c slb \
  --pre \
  --used_session light_bas \
  --summary

run_python spatial pf \
  -s light_end \
  --threshold 0.3

run_python spatial pf \
  -s light_end \
  -c slb \
  --pre \
  --used_session light_bas \
  --summary

# ================ #
# Population Level #
# ================ #

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
  -x light-bas \
  -y light-end \
  -T spks \
  -c slb \
  --pre \
  --used_session light_bas \
  --re

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
