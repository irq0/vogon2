#!/bin/bash

set -e
set -x

SCRIPTPATH="$(dirname "$0")"
source "$SCRIPTPATH/vogon.sh"

"$VOGON_S3GW_BIN" \
    --no-mon-config \
    --conf "$SCRIPTPATH/s3gw-bench.conf" \
    -d \
    --rgw-sfs-data-path="$VOGON_SFS_DIR"
