#!/bin/sh -x

source $(dirname $0)/vogon.sh

mountpoint=$(pwd -P)/mnt
mkdir $mountpoint

sudo -i /sbin/mke2fs $VOGON_MKE2FS_PARAM $VOGON_BLOCKDEV

sudo mount $VOGON_BLOCKDEV $mountpoint
sudo chown vogon: $mountpoint

sudo -i umount $mountpoint
