#!/bin/bash

vogon_time()
{
    local time_format="VOGON_TEST_RESULT:real;%e;seconds\n\
VOGON_TEST_RESULT:user;%U;seconds\n\
VOGON_TEST_RESULT:sys;%S;seconds\n\
VOGON_TEST_RESULT:max-rss;%M;Kbytes\n"

    /usr/bin/time -f "$time_format" "$@"
} 2>&1

vogon_time_prefix()
{
    local prefix="$1"

    local time_format="VOGON_TEST_RESULT:(${prefix})real;%e;seconds\n\
VOGON_TEST_RESULT:(${prefix})user;%U;seconds\n\
VOGON_TEST_RESULT:(${prefix})sys;%S;seconds\n\
VOGON_TEST_RESULT:(${prefix})max-rss;%M;Kbytes\n"

    /usr/bin/time -f "$time_format" "${@:2}"
} 2>&1

vogon_result()
{
    local key="$1"
    local value="$2"
    local unit="$3"

    if [ -z "$key" ] || [ -z "$value" ] || [ -z "$unit" ]; then
	return
    fi

    echo "VOGON_TEST_RESULT:${key};${value};${unit}"
}

vogon_testenv()
{
    local key="$1"
    local value="$2"

    if [ -z "$key" ] || [ -z "$value" ]; then
        echo "ERROR: key=$key value=$value"
	    return
    fi

    echo "VOGON_TEST_ENVIRONMENT:${key};${value}"
}

vogon_testenv_harddisk()
{
    local device="$1"
    local device_name sysfs
    device_name="$(basename "$device" | colrm 4)"
    sysfs="/sys/block/${device_name}/device"

    vogon_testenv "hdd-model" "$(< "${sysfs}/model")"
    vogon_testenv "hdd-vendor" "$(< "${sysfs}/vendor")"
    vogon_testenv "hdd-revision" "$(< "${sysfs}/rev")"
    vogon_testenv "hdd-dev" "${device}"
    vogon_testenv "hdd-cachesize" "$(sudo /sbin/hdparm -I "${device}" | grep "cache/buffer size" | awk '{ print $4 }')"
    vogon_testenv "hdd-rpm" "$(sudo /sbin/hdparm -I "${device}" | grep "Nominal Media Rotation Rate:" | awk '{ print $5 }')"
    vogon_testenv "hdd-transport" "$(sudo /sbin/hdparm -I "${device}" | grep "Transport:" | colrm 1 28)"
    vogon_testenv "hdd-size" "$(sudo /sbin/hdparm -I "${device}" | grep "device size with M = 1000\*1000:" | colrm 1 45)"
    vogon_testenv "hdd-secsize" "$(sudo /sbin/hdparm -I "${device}" | grep "Logical/Physical Sector size:" | colrm 1 48)"
}

vogon_testenv_java()
{
    vogon_testenv "java-version" "$(java -version 2>&1 | head -1 | colrm 1 13 | tr -d \")"
    vogon_testenv "java-jre" "$(java -version 2>&1 | head -2 | tail -1)"
    vogon_testenv "java-vm" "$(java -version 2>&1 | tail -1)"
}

vogon_shasum256_check()
{
    local prefix="$1"
    local correct="$2"
    local to_compare="$3"

    if [ -z "$prefix" ] || [ -z "$correct" ] || [ -z "$to_compare" ]; then
       return
    fi

    local tmp fail result

    tmp="$(mktemp)"
    vogon_time_prefix "$prefix" shasum -a 256 --check "$correct" "$to_compare" | tee "$tmp"

    fail="$(grep FAILED "$tmp")"
    result="FAILED"
    if [ -z "$fail" ]; then
        result="OK"
    fi

    vogon_result "(${prefix})sha256-checksums" "$result"  "boolean"
}

vogon_dd()
{
    local key=$1
    local dd result speed unit

    dd="$(LANG=c dd "${@:2}" 2>&1 | grep copied | cut -f 3 -d ',')"
    result=$?
    speed="$(echo "$dd" | cut -f 1 -d ' ')"
    unit="$(echo "$dd" | cut -f 2 -d ' ')"

    vogon_result "$key" "$speed" "$unit"
}

vogon_drop_caches()
{
    sudo sync

    # 1: free pagecache
    # 2: free dentries, inodes
    # 3: free pagecache, dentries, inodes
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
}
