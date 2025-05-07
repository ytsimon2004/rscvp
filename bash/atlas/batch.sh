#!/bin/bash

set -e
set -x  # Enable shell tracing

cd ../..

#OUTPUT="e:/data/user/yu-ting/histology"
OUTPUT="/Users/yuting/analysis/histology"
LOGFILE="$OUTPUT/hist.log"
ANIMAL="YW043,YW051,YW063,YW064"

export NO_COLOR=1

# Redirect all script output (stdout and stderr) to the log file
exec > >(tee -a "$LOGFILE") 2>&1


run_python() {
  local a=$1
  shift 1

  echo "*** now run ${ANIMAL} $a $* ***"
  python -m rscvp.atlas $a \
    -A "$ANIMAL" \
    --debug \
    "$@"

}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date

#
run_python top \
  --limit 80 \
  --level 0 \
  --family ISOCORTEX

run_python batch_quant \
  --plot family_stacked \
  --debug

run_python batch_quant \
  --plot bias_index \
  --level 2 \
  --norm volume \
  --top 60

run_python batch_quant \
  --plot bias_index \
  --level 2 \
  --norm volume \
  --top 60 \
  plot_type \
  bar

run_python batch_quant \
  --plot heatmap \
  --level 2 \
  --norm volume \
  --top 60