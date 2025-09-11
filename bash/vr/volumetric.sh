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

# which machine
if [[ $(hostname) == "bkrunch-linux" ]]; then
  OUTPUT="/scratch/data/user/yuting/analysis/phys"
elif [[ $(hostname) == "bkrunch2" ]]; then
  OUTPUT="e:/data/user/yu-ting/analysis/phys"
elif [[ $(hostname) == "Yu-Tings-MacBook-Pro.local" ]]; then
  OUTPUT="/Users/yuting/data/analysis/phys"
else
    echo "Unknown host: $(hostname)"
    OUTPUT="./analysis"
fi

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
    --vr-space \
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
  -t close \
  -c slb \
  -T spks \
  --used_session close