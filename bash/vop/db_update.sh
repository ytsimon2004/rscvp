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


# ============= #
# Visual Module #
# ============= #

run_python visual pa \
  --us light \
  --summary \
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

# ============= #
# Bayes Decode  #
# ============= #

run_python model.bayes_decoding analysis \
  --analysis median_decode_error \
  -s "light" \
  --used_session "light" \
  --random 200 \
  --window 100 \
  --spatial-bin 1.5 \
  --CV odd \
  --load 0 \
  --commit

# ======= #
# Generic #
# ======= #

run_python selection cls \
  -s light \
  --us light \
  --commit
