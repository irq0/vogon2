#!/usr/bin/env python3
import json
import logging
import pathlib
import sqlite3
import subprocess
import time
from typing import Any

import click
import docker

import results_db


class ContainerManager:
    """Wraps container runtime interface API to simplifiy
    run, terminate, retrieve logs and test environment data
    """

    def __init__(self, cri: docker.DockerClient, image: str):
        self.cri = cri
        self.image = self.cri.images.pull(image)
        logging.debug("Pulled %s", self.image)

    def run(self, **args):
        self.container = self.cri.containers.run(self.image.id, detach=True, **args)
        logging.debug(
            "Started container for image %s: %s", self.image, self.container.name
        )
        return self.container

    def run_once(self, **args):
        return self.cri.containers.run(self.image.id, **args)

    def terminate(self):
        res = self.container.stop(timeout=30)
        logging.info("Terminating container %s: %s", self.container, res)
        return res

    def logs(self, **kwargs):
        return self.container.logs(**kwargs)

    def env(self):
        return {
            "image-id": self.image.id,
            "image-tags": ";".join(self.image.tags),
        }


class S3GW(ContainerManager):
    def __init__(self, cri, image, storage):
        self.storage = storage
        super().__init__(cri, image)

    def run(self, **args):
        self.container = self.cri.containers.run(
            self.image.id,
            detach=True,
            **args,
            network_mode="host",
            volumes={
                self.storage.mountpoint: {
                    "bind": "/data",
                    "mode": "rw",
                }
            },
        )
        logging.info("S3GW container: %s", self.container.name)
        return self.container


class Storage:
    """A storage device to run the program under test on"""

    def __init__(self, device, mountpoint, format_command):
        self.mountpoint = mountpoint
        self.device = device
        self.format_command = format_command

        logging.info("Storage device: %s mountpoint %s", self.device, self.mountpoint)

        out = subprocess.check_output(
            [
                "lsblk",
                "--output",
                "NAME,TYPE,MODEL,VENDOR,FSTYPE,FSSIZE,PHY-SEC,REV,ROTA,SCHED,RQ-SIZE",
                "--json",
                self.device,
            ]
        )
        d = json.loads(out)["blockdevices"][0]
        logging.info("Device: %s", d)
        self.env_data = {
            "disk-dev-name": d["name"],
            "disk-model": d["model"],
            "disk-vendor": d["vendor"],
            "disk-rotational": d["rota"],
            "disk-scheduler": d["sched"],
            "disk-physical-sector-size": d["phy-sec"],
        }

    def reset(self):
        logging.info("Storage reset ;)")
        # unmount
        # format format command
        # mount
        pass

    def env(self):
        return self.env_data


ResultList = [(str, Any, str)]


class Test:
    def __init__(self, name: str):
        self.name = name

    def env(self, instance):
        "Return environment data. e.g test tool version"
        raise NotImplementedError

    def results(self, instance) -> ResultList:
        "Return test run results"
        raise NotImplementedError

    def run(self, instance):
        "Run test to completion"
        raise NotImplementedError


class AbortTest(Exception):
    pass


class ContainerizedTest(Test):
    def __init__(self, name: str, container: str):
        self.container = container
        super().__init__(name)

    def run(self, instance):
        container = ContainerManager(instance.cri, self.container.image)
        return container.run()

    def env(self, instance):
        return self.container.env()

    def results(self, instance):
        raise NotImplementedError


class HostTest(Test):
    def __init__(self, name):
        super().__init__(name)

    def run(self, instance):
        raise NotImplementedError

    def env(self, instance):
        raise NotImplementedError

    def results(self, instance):
        raise NotImplementedError


class FIOTest(HostTest):
    def __init__(self, name, job_file):
        self.job_file = job_file
        super().__init__(name)

    def run(self, instance):
        try:
            out = subprocess.check_output(
                ["fio", "--output-format=json", self.job_file]
            )
        except subprocess.CalledProcessError:
            logging.exception("fio crashed :()", exc_info=True)
            return -1

        self.json = out
        self.data = json.loads(out)

        return 0

    def env(self, instance):
        return {"fio-version": self.data["fio version"]}

    def results(self, instance):
        j = self.data["jobs"][0]
        return [
            ("JSON", self.json, "JSON"),
            ("read-iops", j["read"]["iops"], "iops"),
            ("read-bw", j["read"]["bw_bytes"], "byte"),
            ("write-iops", j["read"]["iops"], "iops"),
            ("write-bw", j["read"]["bw_bytes"], "byte"),
        ]


class WarpTest(ContainerizedTest):
    container_image = "minio/warp"
    default_args = [
        "--no-color",
        "--json",
    ]
    default_workload_args = [
        "--json",
        "--host=localhost:7480",
        "--access-key=test",
        "--secret-key=test",
        "--benchdata=/warp.out",
        "--duration=10s",
        "--objects=10",
        "--concurrent=1",
    ]

    def __init__(self, name, workload, args=None):
        super().__init__(name, self.container_image)
        self.workload = workload
        self.args = args

    def run(self, instance):
        self.container = ContainerManager(instance.cri, self.container_image)
        self.warp_version = self.container.run_once(command="--version")
        args = []
        args.extend(self.default_args)
        args.append(self.workload)
        args.extend(self.default_workload_args)
        if self.args:
            args.extend(self.args)

        running = self.container.run(command=args, network_mode="host")
        logging.info("Warp container: %s", running.name)

        result = running.wait()
        running.commit("localhost/vogon/warp", instance.test_id)
        self.last_run = running

        try:
            bits, stat = running.get_archive("/warp.out.csv.zst")
            with open(instance.archive / "warp.out.csv.zst", "wb") as fd:
                for chunk in bits:
                    fd.write(chunk)
        except docker.errors.NotFound:
            logging.error("warp results file not found. ignoring")

        return result["StatusCode"]

    def results(self, instance):
        results = []
        if not self.last_run:
            return None

        try:
            data = self.last_run.logs(stdout=True, stderr=False, stream=False).decode(
                "utf-8"
            )
            json_start = data.find("{")
            json_stop = data.rfind("}")
            jdata = data[json_start : json_stop + 1]
            results.append(("JSON", jdata, "JSON"))
            d = json.loads(jdata)
        except json.decoder.JSONDecodeError:
            logging.error("warp result parsing failed. %s", jdata, exc_info=True)
            return results

        for op in d["operations"]:
            prefix = op["type"] + "_"
            results.append(
                (prefix + "avg-ops", op["throughput"]["average_ops"], "ops/s")
            )
            results.append(
                (prefix + "avg-bps", op["throughput"]["average_bps"], "byte/s")
            )
            results.append((prefix + "ops", op["throughput"]["operations"], "ops"))

        return results

    def env(self, instance):
        env = self.container.env()
        env["warp-version"] = self.warp_version
        return env


class TestSuite:
    def __init__(self, name: str, tests: list[Test]):
        self.name = name
        self.tests = tests


class TestInstance:
    def __init__(
        self,
        cri: docker.DockerClient,
        db: results_db.ResultsDB,
        under_test_container: ContainerManager,
        suite: TestSuite,
        storage: Storage,
        archive: pathlib.Path,
        reps: int,
    ):
        self.cri = cri
        self.db = db
        self.under_test_container = under_test_container
        self.suite = suite
        self.storage = storage
        self.archive = archive
        self.reps = reps

        self.name = suite.name
        self.test_id = results_db.make_id()

    def run(self):
        logging.info(f"== STARTING TEST {self.name} - {self.test_id} ==")
        logging.info(f"== DOING {self.reps} RUNS ==")
        self.db.save_before_test(self.test_id, self.name)
        self.db.save_test_environment_default(self.test_id)

        for test in self.suite.tests:
            self.run_test(test)

    def run_test(self, test):
        self.db.save_test_environment(
            self.test_id, self.under_test_container.env(), "under-test-"
        )
        self.db.save_test_environment(self.test_id, self.storage.env())

        for rep in range(self.reps):
            try:
                self.run_test_rep(test, rep)
            except Exception:
                logging.exception(
                    "Test %s rep %s failed. Aborting rest run", test, rep, exc_info=True
                )
                raise AbortTest

        self.db.save_test_environment(self.test_id, test.env(self), "test-")

    def run_test_rep(self, test, rep):
        rep_id = results_db.make_id()
        self.db.save_before_rep(self.test_id, rep_id)
        self.storage.reset()

        logging.info("%d - %s %s", rep, test, rep_id)
        self.under_test_container.run()

        # TODO be smarter about the started condition
        time.sleep(10)

        returncode = test.run(self)

        if returncode != 0:
            logging.exception("Test rep failed with %d: %s", returncode, test.name)
            raise AbortTest

        self.db.save_after_rep(rep_id, returncode)
        self.db.save_test_results(rep_id, test.results(self))

        self.under_test_container.terminate()


warp_mixed_default = WarpTest("mixed-default", "mixed")
warp_get_default = WarpTest("get-default", "get")
warp_put_default = WarpTest("put-default", "put")

fio_sfs_like = FIOTest(
    "fio-sfs-like", "/home/seri/WORKSPACE/vogon/scripts/sfs-like_10M-files.fio"
)

test_suites = {
    "baseline": TestSuite(
        "Test disk baseline performance",
        tests=[
            fio_sfs_like,
        ],
    ),
    "demo": TestSuite(
        "Extensive test suite. 10 repetitions. Lots of benchmarks",
        tests=[
            warp_mixed_default,
        ],
    ),
}


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    loglevel = [logging.WARNING, logging.DEBUG][int(debug)]
    logging.basicConfig(
        level=loglevel, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ctx.obj["DEBUG"] = debug
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARN)


@cli.command()
@click.pass_context
@click.option(
    "--under-test-image",
    type=str,
    required=True,
    help="Docker image of the application under test",
)
@click.option(
    "--suite",
    type=click.Choice(test_suites),
    required=True,
    help="Test suite. See vogon2.py for configuration",
)
@click.option(
    "--archive-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory to archive test artifacts to",
)
@click.option(
    "--storage-device",
    type=click.Path(exists=True),
    required=True,
    help="Storage device to run on (e.g /dev/disk/by-id/...)",
)
@click.option(
    "--mountpoint",
    type=click.Path(exists=1),
    required=True,
    help="Mountpoint for --storage-device",
)
@click.option(
    "--docker-api",
    type=str,
    required=True,
    help="Docker API URI. e.g unix://run/podman/podman.sock",
)
@click.option(
    "--sqlite", type=str, required=True, help="Where to find the sqlite database?"
)
@click.option("--init-db/--no-init-db", default=False, help="Create tables, etc")
@click.option("--repeat", type=int, default=1, help="How many repetitions to run")
def test(
    ctx,
    under_test_image,
    suite,
    archive_dir,
    storage_device,
    mountpoint,
    docker_api,
    sqlite,
    init_db,
    repeat,
):
    cri = docker.DockerClient(base_url=docker_api)
    dbconn = sqlite3.connect(sqlite)
    if init_db:
        results_db.init_db(dbconn)
    db = results_db.ResultsDB(dbconn)
    storage = Storage(storage_device, mountpoint, None)
    s3gw = S3GW(cri, under_test_image, storage)
    test_instance = TestInstance(
        cri, db, s3gw, test_suites[suite], storage, pathlib.Path(archive_dir), repeat
    )
    try:
        test_instance.run()
    finally:
        s3gw.terminate()
        dbconn.close()


if __name__ == "__main__":
    cli(obj={})
