#!/bin/sh

JARDIRS="/usr/share/java /usr/local/share/java"

for dir in $JARDIRS; do
    [ -d "$dir" ] &&  CLASSES="$(find $dir -depth -type f -iname *.jar -printf :%p)"
    export CLASSPATH=".$CLASSES"
done
export PATH="/usr/bin:/usr/lib/java/bin/:/usr/sbin:/bin:/sbin"


function vogon_time()
{
    TIME_FORMAT="VOGON_TEST_RESULT:real;%e;seconds\n\
VOGON_TEST_RESULT:user;%U;seconds\n\
VOGON_TEST_RESULT:sys;%S;seconds\n\
VOGON_TEST_RESULT:max-rss;%M;Kbytes\n"

    /usr/bin/time -f "$TIME_FORMAT" $*
} 2>&1

function vogon_time_prefix()
{
    prefix=$1

    TIME_FORMAT="VOGON_TEST_RESULT:(${prefix})real;%e;seconds\n\
VOGON_TEST_RESULT:(${prefix})user;%U;seconds\n\
VOGON_TEST_RESULT:(${prefix})sys;%S;seconds\n\
VOGON_TEST_RESULT:(${prefix})max-rss;%M;Kbytes\n"

    /usr/bin/time -f "$TIME_FORMAT" ${*:2}
} 2>&1

function vogon_result()
{
    key=$1
    value=$2
    unit=$3

    if [ -z "$key" ] || [ -z "$value" ] || [ -z "$unit" ]; then
	return
    fi

    echo "VOGON_TEST_RESULT:${key};${value};${unit}"
}

function vogon_testenv()
{
    key=$1
    value=$2

    if [ -z "$key" ] || [ -z "$value" ]; then
        echo "ERROR: key=$key value=$value"
	    return
    fi

    echo "VOGON_TEST_ENVIRONMENT:${key};${value}"
}

function vogon_testenv_harddisk()
{
    device=$1
    device_name=$(basename $device | colrm 4)
    sysfs="/sys/block/${device_name}/device"

    vogon_testenv "hdd-model" "$(cat ${sysfs}/model)"
    vogon_testenv "hdd-vendor" "$(cat ${sysfs}/vendor)"
    vogon_testenv "hdd-revision" "$(cat ${sysfs}/rev)"
    vogon_testenv "hdd-dev" "${device}"
    vogon_testenv "hdd-cachesize" "$(sudo /sbin/hdparm -I ${device} | grep "cache/buffer size" | awk '{ print $4 }')"
    vogon_testenv "hdd-rpm" "$(sudo /sbin/hdparm -I ${device} | grep "Nominal Media Rotation Rate:" | awk '{ print $5 }')"
    vogon_testenv "hdd-transport" "$(sudo /sbin/hdparm -I ${device} | grep "Transport:" | colrm 1 28)"
    vogon_testenv "hdd-size" "$(sudo /sbin/hdparm -I ${device} | grep "device size with M = 1000\*1000:" | colrm 1 45)"
    vogon_testenv "hdd-secsize" "$(sudo /sbin/hdparm -I ${device} | grep "Logical/Physical Sector size:" | colrm 1 48)"
}

function vogon_testenv_java()
{
    vogon_testenv "java-version" "$(java -version 2>&1 | head -1 | colrm 1 13 | tr -d \")"
    vogon_testenv "java-jre" "$(java -version 2>&1 | head -2 | tail -1)"
    vogon_testenv "java-vm" "$(java -version 2>&1 | tail -1)"
}

function vogon_shasum256_check() 
{
    prefix=$1
    correct=$2
    to_compare=$3

    if [ -z "$prefix" ] || [ -z "$correct" ] || [ -z "$to_compare" ]; then
       return 
    fi

    tmp=$(tempfile)
    vogon_time_prefix "$prefix" shasum -a 256 --check $correct $to_compare | tee $tmp
    
    fail=$(grep FAILED $tmp)
    result="FAILED"
    if [ -z "$fail" ]; then
        result="OK"
    fi

    vogon_result "(${prefix})sha256-checksums" "$result"  "boolean"
}

function vogon_dd()
{
	key=$1

	dd="$(LANG=c dd ${*:2} 2>&1 | grep copied | cut -f 3 -d ',')"
	result=$?
	speed="$(echo $dd | cut -f 1 -d ' ')"
	unit="$(echo $dd | cut -f 2 -d ' ')"

	vogon_result "$key" "$speed" "$unit"
}

function vogon_drop_caches()
{
    sudo sync

    # 1: free pagecache
    # 2: free dentries, inodes
    # 3: free pagecache, dentries, inodes
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
}
