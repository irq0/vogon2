#!/bin/bash

wait_http_200 () {
    local url="$1"

    echo "waiting for $url to return 200:"
    while [[ $(curl -s -o/dev/null -w '%{http_code}' "$url") != "200" ]]; do
	sleep 1
    done
    echo "done"
}
