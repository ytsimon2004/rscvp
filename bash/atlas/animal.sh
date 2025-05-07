#! /bin/bash

set -e

cd ../..

if [ $# -eq 0 ]; then
  echo "$0 animal_id?"
  exit 1
else
  ANIMAL=$1
fi

#OUTPUT="e:/data/user/yu-ting/histology"
OUTPUT="$HOME/data/analysis/hist"
OUTPUT_FILE="$OUTPUT/${ANIMAL}/cli.log"

export NO_COLOR=1
exec > >(tee -a "$OUTPUT_FILE") 2>&1

##
run_hist() {
  local a=$1
  shift 1

  echo "*** now run ${ANIMAL} $a $* ***"
  python -m rscvp.atlas $a \
  -A "$ANIMAL" \
  --level 2 \
  --top 40 \
  "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date

#
run_hist quant \
  -g bar \

run_hist quant \
  -g bar \
  --norm volume

run_hist quant \
  -g bar \
  --norm cell

run_hist quant \
  -g bar \
  --norm channel

run_hist quant \
  -g cat

run_hist quant \
  -g cat \
  --norm volume

run_hist quant \
  -g cat \
  --norm channel

run_hist quant \
  -g cat \
  --norm cell

run_hist quant \
  -g venn

run_hist quant \
  -g pie

run_hist quant \
  -g bias

run_hist ternary \
  --norm volume

run_hist ternary \
  --norm cell
