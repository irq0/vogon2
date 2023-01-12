#!/usr/bin/env python3
import json
import subprocess
import sys


def vogon_result(prefix, key, value, unit):
    print(f"VOGON_TEST_RESULT:{prefix}{key};{value};{unit}")


def main():
    filename = sys.argv[1]
    jdata = subprocess.check_output(
        ["warp", "--json", "analyze", "--json", "--analyze.v", filename]
    )
    json_start = jdata.find(b"{")
    d = json.loads(jdata[json_start:])

    for op in d["operations"]:
        prefix = op["type"] + "_"
        vogon_result(prefix, "avg-ops", op["throughput"]["average_ops"], "ops/s")
        vogon_result(prefix, "avg-bps", op["throughput"]["average_bps"], "byte/s")
        vogon_result(prefix, "ops", op["throughput"]["operations"], "ops")
        vogon_result(
            prefix, "duration", op["throughput"]["measure_duration_millis"], "ms"
        )


if __name__ == "__main__":
    main()
