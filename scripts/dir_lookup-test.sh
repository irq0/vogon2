#!/bin/sh

source $(dirname $0)/vogon.sh

mountpoint=$(pwd -P)/mnt
mkdir $mountpoint

## create fs; touch files
sudo -i /sbin/mke2fs $VOGON_MKE2FS_PARAM $VOGON_BLOCKDEV
sudo mount $VOGON_BLOCKDEV $mountpoint
sudo chown vogon: $mountpoint

for i in $(seq -w $VOGON_TEST_FILES); do
    echo "$i" > $mountpoint/randfile_$i
done
echo ":D" > $mountpoint/look_me_up

sudo -i umount $mountpoint

## mount and test different implementations 

# kernel +dir_hash
vogon_drop_caches
sudo mount $VOGON_BLOCKDEV $mountpoint
vogon_time_prefix "kern+hash" /bin/ls $mountpoint/look_me_up
sudo -i umount $mountpoint

# kernel -dir_hash
vogon_drop_caches
sudo tune2fs -O ^dir_index $VOGON_BLOCKDEV
sudo mount $VOGON_BLOCKDEV $mountpoint
vogon_time_prefix "kern-hash" /bin/ls -la $mountpoint/look_me_up
sudo -i umount $mountpoint

# jext2
vogon_drop_caches
$VOGON_JEXT2 $VOGON_BLOCKDEV $mountpoint
vogon_time_prefix "jext2" /bin/ls -la $mountpoint/look_me_up
sudo -i umount $mountpoint

# fuse-ext2
vogon_drop_caches
$VOGON_FUSEEXT2 $VOGON_BLOCKDEV $mountpoint
vogon_time_prefix "fuse-ext2" /bin/ls -la $mountpoint/look_me_up
sudo -i umount $mountpoint

exit 0
