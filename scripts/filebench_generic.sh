#!/bin/sh

source $(dirname $0)/vogon.sh

ident="${VOGON_IDENT}_${VOGON_TEST_ID}_${VOGON_TESTRUN_ID}"
dumpfile="$VOGON_LOGDIR/filebench_stats_${ident}"
mountpoint=$(pwd -P)/mnt

vogon_testenv "filebench-version" "$(echo "version" | $VOGON_GO_FILEBENCH 2>&1 | tail -2 | head -1 | cut -f 6 -d ' ')"
vogon_testenv "filebench-personality" "${VOGON_TEST_PERSONALITY}"

mkdir $mountpoint

(
    $($VOGON_TEST_MOUNT $VOGON_BLOCKDEV $mountpoint $VOGON_TEST_MOUNT_POSTOPTS)
) &


sleep 10

(
    cd $mountpoint
    mkdir $mountpoint/filebench_${ident}/
)

vogon_drop_caches

(
    $VOGON_GO_FILEBENCH<<EOF
load $VOGON_TEST_PERSONALITY
debug 2
set \$dir=$mountpoint/filebench_${ident}/
create filesets
create files
create processes
stats clear
sleep $VOGON_TEST_SLEEP
stats snap
stats dump "$dumpfile"
shutdown processes
quit
EOF
)

(
    $($VOGON_TEST_UMOUNT $mountpoint)
)


sum="$(tail -1 $dumpfile | tr -d , )"

vogon_result "ops" $(echo $sum | awk ' {print $3, $4}')
vogon_result "ops/s" $(echo $sum | awk ' {print $5, $6}')
vogon_result "r/w ratio" $(echo $sum | awk ' {print $7, $8}')
vogon_result "throughput" $(echo $sum | awk ' {print $9}' | tr -d 'mb/s') "mb/s"
vogon_result "cpu-per-op" $(echo $sum | awk ' {print $10}' | tr -d 'uscpu/op') "uscpu/op"

exit $?
