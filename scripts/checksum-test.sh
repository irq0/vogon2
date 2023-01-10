#!/bin/bash -x

source $(dirname $0)/vogon.sh

mountpoint=$(pwd -P)/mnt
correct_sums=$(tempfile)

mkdir testfiles
mkdir $mountpoint

(
	cd testfiles

	for i in $(seq 1 $VOGON_TEST_RANDFILES); do
        echo $i
		dd if=/dev/urandom of=randfile_$i bs=1M count=$((2 * $i))
	done
)

sudo -i /sbin/mke2fs $VOGON_MKE2FS_PARAM $VOGON_BLOCKDEV
sudo mount $VOGON_BLOCKDEV $mountpoint
sudo chown vogon: $mountpoint

cp -Rv testfiles $mountpoint/
shasum -a 256 $mountpoint/testfiles/* > $correct_sums

sudo -i umount $mountpoint

vogon_drop_caches
sudo mount $VOGON_BLOCKDEV $mountpoint
vogon_shasum256_check "kernel" $correct_sums $mountpoint/testfiles/*
sudo -i umount $mountpoint

vogon_drop_caches
$VOGON_FUSEEXT2 $VOGON_BLOCKDEV $mountpoint
vogon_shasum256_check "fuse-ext2" $correct_sums $mountpoint/testfiles/*
sudo -i umount $mountpoint

vogon_drop_caches
$VOGON_JEXT2 $VOGON_BLOCKDEV $mountpoint
#shasum -a 256 --check $correct_sums $mountpoint/testfiles/*
vogon_shasum256_check "jext2" $correct_sums $mountpoint/testfiles/*
sudo -i umount $mountpoint
return=$?

rm $correct_sums
exit $return
