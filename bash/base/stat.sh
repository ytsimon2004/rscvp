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

##
run_python stat nstat_normor \
  -s light \
  --used_session light \
  --plot histogram

run_python stat stat_corr \
  -s light \
  --used_session light \
  --ct spatial \
  --pt spatial \
  --annot

run_python stat stat_ses \
  --dot \
  --plot connect \
  --ct spatial \
  --pt spatial \
  --pre \
  --used_session light

run_python stat stat_ses \
  --dot \
  --plot box \
  --ct spatial \
  --pt spatial \
  --pre \
  --re \
  --used_session light

run_python stat stat_ses \
  --plot scatter \
  --ct spatial \
  --pt spatial \
  --pre \
  --used_session light
