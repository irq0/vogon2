#!/bin/sh

source $(dirname $0)/vogon.sh

mountpoint=$(pwd -P)/mnt
mkdir $mountpoint

vogon_drop_caches

(
    $($VOGON_TEST_MOUNT $VOGON_BLOCKDEV $mountpoint)
) &

sleep 10

(
    cd $mountpoint
    vogon_dd "dd" if=big of=/dev/null
)

(
    $($VOGON_TEST_UMOUNT $mountpoint)
)

exit 0
