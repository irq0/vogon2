#!/bin/bash

EXT="$HOME/opt/lib/libsqlitefunctions.so"
sqlite3 "$1" "select load_extension('$EXT'); select test_id, key, real_runs, runs, mean, conf_in1, conf_in2, unit from statistics"
