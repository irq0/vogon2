#!/bin/bash

source "$(dirname "$0")/../vogon.sh"

vogon_testenv_harddisk "$VOGON_BLOCKDEV"
vogon_testenv "fuse-ext2-version" "$($VOGON_TEST_BINARY | head -4 | tail -1 | cut -f 2 -d ' ')"
vogon_testenv "fuse-version" "$(fusermount -V | awk '{ print $3 }')"
