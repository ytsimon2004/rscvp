#! /bin/bash

set -e

cd ../..

if [ $# -eq 0 ]; then
  echo "$0 animal_id?"
  exit 1
else
  ANIMAL=$1
fi

OUTPUT="e:/data/user/yu-ting/histology"
OUTPUT_FILE="$OUTPUT/${ANIMAL}/cli.log"

export NO_COLOR=1
exec > >(tee -a "$OUTPUT_FILE") 2>&1

##
run_query() {
  local a=$1
  shift 1

  echo "*** now run ${ANIMAL} $a $* ***"
  python -m rscvp.atlas $a \
    -A "$ANIMAL" \
    "$@"
}

echo '# ================================================== #'
echo 'git commit hash: ' $(git rev-parse --verify HEAD --short)
date


#
run_query query \
  --area VIS \

run_query query \
  --area PTLp \

run_query query \
  --area ATN \

run_query query \
  --area RHP \

run_query query \
  --area HIP \
