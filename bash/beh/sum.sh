#! /bin/bash

set -e

cd ../..

if [ $# -eq 0 ]; then
  echo "$0 experimental_date animal_id ?"
  exit 1
else
  ED=$1
  ID=$2
fi

##
run_python() {
  local m=$1
  local a=$2
  shift 2

  echo "*** now run $m $a ***"
  python -m rscvp.$m $a \
    -D "$ED" \
    -A "$ID" \
    "$@"
}


run_python behavioral bs \
  --vcut -20 \
  --session light,visual
