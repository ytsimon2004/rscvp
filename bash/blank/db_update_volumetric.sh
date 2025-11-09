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


run_python selection cls \
  -s light \
  --us light \
  --blank-db \
  --commit
