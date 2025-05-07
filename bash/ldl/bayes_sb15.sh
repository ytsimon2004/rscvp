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
    -s "light_bas" \
    --used_session "light_bas" \
    --random 200 \
    --window 100 \
    --spatial-bin 1.5 \
    --CV odd \
    "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date

run_python model.bayes_decoding decode_cache

run_python model.bayes_decoding analysis \
  --analysis overview \
  --plot-concat \
  --load 0

run_python model.bayes_decoding analysis \
  --analysis median_decode_error \
  --commit \
  --load 0

run_python model.bayes_decoding analysis \
  --analysis confusion_matrix \
  --load 0

run_python model.bayes_decoding analysis \
  --analysis position_bins_error \
  --load 0
