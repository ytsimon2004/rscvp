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
    --vr-space \
    "$@"
}

python -m rscvp.topology.main_fov_db \
    -D "$ED" \
    -A "$ID" \
    --commit

run_python selection cls \
  --session close \
  --used_session close \
  --commit