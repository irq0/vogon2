#!/bin/bash

source "$(dirname "$0")/../vogon.sh"

vogon_testenv_harddisk "$VOGON_BLOCKDEV"
vogon_testenv "ext2fuse-version" "$($VOGON_TEST_BINARY --version | colrm 1 17)"
vogon_testenv "fuse-version" "$(fusermount -V | awk '{ print $3 }')"
