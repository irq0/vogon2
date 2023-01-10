#!/bin/sh -x

source $(dirname $0)/vogon.sh

ident="${VOGON_IDENT}_${VOGON_TEST_ID}_${VOGON_TESTRUN_ID}"
logdir="${VOGON_LOGDIR}/${ident}"
mkdir -p "${logdir}"

dumpfile="${logdir}/stats"
blktracefile="${logdir}/blktrace"
mountpoint=$(pwd -P)/mnt

vogon_testenv "filebench-version" "$(echo "version" | $VOGON_GO_FILEBENCH 2>&1 | tail -2 | head -1 | awk '{ print $6 }')"
vogon_testenv "filebench-filesize" "${VOGON_TEST_FILESIZE}"
vogon_testenv "part-1kblocks" "$(/sbin/fdisk -s ${VOGON_BLOCKDEV})"
vogon_testenv "blktrace-version" "$($VOGON_BLKTRACE -v | cut -f 3 -d ' ')"

mkdir $mountpoint

for bench in $VOGON_TEST_PERSONALITIES; do
    echo "Starting filebench personality: $bench"

    echo "makefs:"
    # create new filesystem
    sudo -i /sbin/mke2fs -F $VOGON_MKE2FS_PARAM $VOGON_BLOCKDEV
    sudo mount $VOGON_BLOCKDEV $mountpoint
    sudo chown vogon: $mountpoint
    sudo -i umount $mountpoint

    echo "mount:"
    # mount
    (
	$($VOGON_TEST_MOUNT $VOGON_BLOCKDEV $mountpoint $VOGON_TEST_MOUNT_POSTOPTS)
    ) &

    sleep 10
    
    # this blocks until filesystem is accessable 
    ( 
	cd $mountpoint 
	mkdir $mountpoint/filebench_${ident}_${bench}/
    )


    echo "find filesystem process pid"
    fs_pid=$(lsof ${VOGON_BLOCKDEV} | tail -1 | awk '{ print $2 }')

    # when fuseblk is used the process does not open() the file
    if [ -z "$fs_pid" ]; then
        fs_pid=$(pgrep -f ${VOGON_BLOCKDEV})
    fi
	    
#    echo "prepare filebench:"
#    # run filebench
#    (
#	$VOGON_GO_FILEBENCH<<EOF
#$(cat ${VOGON_FILEBENCH_TESTS}/${bench})
#set \$dir=$mountpoint/filebench_${ident}_${bench}/
#create filesets
#create files
#EOF
#    )

    echo "start block trace:"
    ( sudo $VOGON_BLKTRACE -d $VOGON_BLOCKDEV -D ${logdir} -o blktrace_${bench} ) &
#    blktracetmp=$(mktemp)
#    ( sudo $VOGON_BLKTRACE -d $VOGON_BLOCKDEV -o $blktracetmp ) &

    if [ -n "$fs_pid" ]; then # watch fuse process
	echo "start pidstat:"
	( pidstat -u -p $fs_pid 1 > ${logdir}/pidstat_cpu_$bench ) &
	( pidstat -r -p $fs_pid 1 > ${logdir}/pidstat_mem_$bench ) &
    fi
    
    echo "start sar:"
    ( sar -u 1 > ${logdir}/cpu_util_$bench ) &

    echo "drop caches:"
    vogon_drop_caches    

    echo "run filebench:"
    (

	$VOGON_GO_FILEBENCH<<EOF
$(cat ${VOGON_FILEBENCH_TESTS}/${bench})
set \$dir=$mountpoint/filebench_${ident}_${bench}/
create filesets
create processes
stats clear
sleep ${VOGON_TEST_SLEEP}
stats snap
stats dump "${dumpfile}_${bench}"
shutdown processes
quit
EOF
    )
    
    echo "sync, stop blktrace, pidstat:"
    sync
    
    sudo pkill "$(basename $VOGON_BLKTRACE)"
    mv $blktracetmp $blktracefile

    pkill pidstat
    pkill sar

    echo "umount:"
    # umount
    (
	$($VOGON_TEST_UMOUNT $mountpoint)
    )
    
    echo "extract data:"
    # extract data
    sum="$(tail -1 ${dumpfile}_${bench} | tr -d , )"

    vogon_result "${bench}(ops)" $(echo $sum | awk ' {print $3, $4}')
    vogon_result "${bench}(ops/s)" $(echo $sum | awk ' {print $5, $6}')
    vogon_result "${bench}(r/w ratio)" $(echo $sum | awk ' {print $7, $8}')
    vogon_result "${bench}(throughput)" $(echo $sum | awk ' {print $9}' | tr -d 'mb/s') "mb/s"
    vogon_result "${bench}(cpu-per-op)" $(echo $sum | awk ' {print $10}' | tr -d 'uscpu/op') "uscpu/op"
    
    wait
done

exit $?
