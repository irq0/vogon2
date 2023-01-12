#!/bin/bash

source "$(dirname "$0")/../vogon.sh"

vogon_testenv_harddisk "$(readlink -f "$VOGON_SFS_BLOCKDEV")"
vogon_testenv "s3gw-version" "$("$VOGON_S3GW_BIN" --version)"
