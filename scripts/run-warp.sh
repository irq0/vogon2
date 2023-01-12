#!/bin/bash

set -e
set -x

SCRIPTPATH="$(dirname "$0")"
source "$SCRIPTPATH/vogon.sh"
source "$SCRIPTPATH/s3-utils.sh"

vogon_testenv "warp-version" "$(warp --version)"

wait_http_200 "localhost:9090"

vogon_drop_caches

warp --no-color \
     get \
     --host=localhost:7480 --access-key=test --secret-key=test \
     --objects=10 \
     --concurrent=5 \
     --duration=10s \
     --benchdata="$VOGON_TESTRUN_ARCHIVE/warp.out"

"$SCRIPTPATH/warp-results-to-vogon.py" "$VOGON_TESTRUN_ARCHIVE/warp.out.csv.zst"
