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
# DFF Signal #
# ========== #

run_python signal ds \
  -s light \
  --re

run_python signal ds \
  -s visual \
  --re

run_python signal ds \
  -s dark \
  --re

# ======================= #
# ROIs coordinates / FOVs #
# ======================= #

run_python topology fov

run_python topology cord \
  --page fov_table


# ====================== #
# Topographical Analysis #
# ====================== #

run_python topology ct

run_python topology ctopo \
  --us light \
  --page fov_table

run_python topology vtopo \
  --us light \
  --page fov_table

run_python topology stopo \
  --session light \
  --us light \
  --page fov_table


# ============== #
# Classification #
# ============== #

run_python selection cls \
  -s light \
  --us light \
  --commit
