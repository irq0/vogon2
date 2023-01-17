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

SCRIPT_PATH = pathlib.Path(__file__).parent


DockerImageType = str


class ContainerManager:
    """Wraps container runtime interface API to simplifiy
    run, terminate, retrieve logs and test environment data
    """

    def __init__(self, cri: docker.DockerClient, image: DockerImageType):
        self.cri = cri
        self.image = self.cri.images.pull(image)
        logging.debug("Pulled %s", self.image)

    def run(self, **kwargs) -> docker.models.containers.Container:
        """wrap docker client run(). runs detached and set
        self.container to the started container. forward args"""

        self.container = self.cri.containers.run(self.image.id, detach=True, **kwargs)
        logging.debug(
            "Started container for image %s: %s", self.image, self.container.name
        )
        return self.container

    def run_once(self, **kwargs) -> str:
        "wrap docker client run(). like subprocess.check_output()"
        return self.cri.containers.run(self.image.id, **kwargs)

    def terminate(self):
        "terminate container started with run()"
        res = self.container.stop(timeout=30)
        logging.info("Terminating container %s: %s", self.container, res)

    def logs(self, **kwargs) -> bytes:
        "get logs of container started with run()"
        return self.container.logs(**kwargs)

    def env(self) -> results_db.TestEnvType:
        "get test environment data"
        return {
            "image-id": self.image.id,
            "image-tags": ";".join(self.image.tags),
        }


class S3GW(ContainerManager):
    def __init__(
        self, cri: docker.DockerClient, image: DockerImageType, storage: "Storage"
    ):
        self.storage = storage
        self.s3gw_version = "unknown"
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
        ret, version = self.container.exec_run(["radosgw", "--version"])
        if ret == 0:
            self.s3gw_version = version

        logging.info("🔎 S3GW container: %s", self.container.name)
        return self.container

    def env(self):
        e = super().env()
        e["s3gw-version"] = self.s3gw_version
        return e


class Storage:
    "A storage device to run the program under test on"

    def __init__(
        self,
        enable_reset_mkfs_mount: bool,
        device: pathlib.Path,
        mountpoint: pathlib.Path,
        mkfs_command: list[str],
    ):
        self.enable_reset_mkfs_mount = enable_reset_mkfs_mount
        self.mountpoint = mountpoint
        self.device = device
        self.mkfs_command = mkfs_command

        logging.info("Storage device: %s mountpoint %s", self.device, self.mountpoint)

        out = subprocess.check_output(
            [
                "lsblk",
                "--output",
                "NAME,TYPE,MODEL,VENDOR,FSTYPE,FSSIZE,PHY-SEC,REV,ROTA,SCHED,RQ-SIZE",
                "--json",
                str(self.device),
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
        if not self.enable_reset_mkfs_mount:
            logging.info("🛢 storage reset disabled. skipping umount, mkfs, mount")
            return
        try:
            logging.info(
                f"🛢 resetting storage: mountpoint {self.mountpoint}, "
                f"dev {self.device}, command {self.mkfs_command}"
            )
            try:
                umount_out = subprocess.check_output(
                    ["sudo", "umount", str(self.mountpoint.absolute())],
                    stderr=subprocess.STDOUT,
                )
                logging.debug("umount: %s", umount_out)
            except subprocess.CalledProcessError as e:
                # on first run mountpoint is typically not mounted
                if b"not mounted" not in e.output:
                    raise e

            mkfs_out = subprocess.check_output(
                self.mkfs_command + [str(self.device)], stderr=subprocess.STDOUT
            )
            logging.debug("mkfs out: %s", mkfs_out)

            mount_out = subprocess.check_output(
                ["sudo", "mount", str(self.device), str(self.mountpoint.absolute())],
                stderr=subprocess.STDOUT,
            )
            logging.debug("mount out: %s", mount_out)

            chown_out = subprocess.check_output(
                ["sudo", "chown", "vogon:vogon", self.mountpoint],
                stderr=subprocess.STDOUT,
            )
            logging.debug("chown out: %s", chown_out)

        except subprocess.CalledProcessError as e:
            logging.exception(
                "storage reset (umount,mkfs,mount) failed with exit %s out %s. "
                "failing benchmark",
                e.returncode,
                e.output,
            )
            raise e

    def env(self):
        return self.env_data

    def drop_caches(self):
        "Drop caches. Classic echo 3 > /proc/sys/vm/drop_caches"
        # raises exception if fails
        try:
            subprocess.check_call(
                ["sudo", "bash", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
            )
        except subprocess.CalledProcessError as e:
            logging.exception("os cached drop failed. failing benchmark")
            raise e


ResultList = [(str, Any, str)]


class Test:
    def __init__(self, name: str):
        self.name = name

    def env(self, instance: "TestRunner") -> results_db.TestEnvType:
        "Return environment data. e.g test tool version"
        raise NotImplementedError

    def results(self, instance: "TestRunner") -> results_db.TestResultsType:
        "Return test run results"
        raise NotImplementedError

    def run(self, instance: "TestRunner") -> int:
        "Run test to completion"
        raise NotImplementedError

    def kind(self) -> str:
        raise NotImplementedError


class AbortTest(Exception):
    pass


class ContainerizedTest(Test):
    def __init__(self, name: str, container_image: DockerImageType):
        self.container_image = container_image
        super().__init__(name)

    def run(self, instance: "TestRunner"):
        container_manager = ContainerManager(instance.cri, self.container_image)
        return container_manager.run()

    def env(self, instance: "TestRunner"):
        raise NotImplementedError

    def results(self, instance: "TestRunner"):
        raise NotImplementedError

    def kind(self) -> str:
        raise NotImplementedError


class HostTest(Test):
    def __init__(self, name):
        super().__init__(name)

    def run(self, instance: "TestRunner"):
        raise NotImplementedError

    def env(self, instance: "TestRunner"):
        raise NotImplementedError

    def results(self, instance: "TestRunner") -> results_db.TestResultsType:
        raise NotImplementedError

    def kind(self) -> str:
        raise NotImplementedError


class FIOTest(HostTest):
    def __init__(self, name: str, job_file: pathlib.Path):
        self.job_file = job_file
        super().__init__(name)

    def run(self, instance: "TestRunner"):
        try:
            logging.info(
                "running fio with job file %s in %s",
                self.job_file,
                instance.storage.mountpoint,
            )
            out = subprocess.check_output(
                ["fio", "--output-format=json", str(self.job_file.absolute())],
                stderr=subprocess.STDOUT,
                cwd=instance.storage.mountpoint,
            )
        except subprocess.CalledProcessError:
            logging.exception("fio crashed :()", exc_info=True)
            return -1

        self.json = out
        self.data = json.loads(out)

        return 0

    def env(self, instance: "TestRunner"):
        return {"fio-version": self.data["fio version"]}

    def results(self, instance: "TestRunner"):
        j = self.data["jobs"][0]
        return [
            ("JSON", self.json, "JSON"),
            ("read-iops", j["read"]["iops"], "iops"),
            ("read-bw", j["read"]["bw_bytes"], "byte"),
            ("write-iops", j["read"]["iops"], "iops"),
            ("write-bw", j["read"]["bw_bytes"], "byte"),
        ]

    def kind(self) -> str:
        return "FIO"

    def __str__(self) -> str:
        return f"FIO(job_file={self.job_file})"


class WarpTest(ContainerizedTest):
    default_container_image = "minio/warp"
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
    ]

    def __init__(self, name: str, workload: str, args: list[str] = []):
        super().__init__(name, self.default_container_image)
        self.workload = workload
        self.args = args

    def run(self, instance: "TestRunner"):
        self.container = ContainerManager(instance.cri, self.container_image)
        self.warp_version = self.container.run_once(command="--version")
        args = []
        args.extend(self.default_args)
        args.append(self.workload)
        args.extend(self.default_workload_args)
        args.extend(self.args)

        running = self.container.run(command=args, network_mode="host")
        logging.info("🔎 Warp container: %s", running.name)

        result = running.wait()
        self.last_run = running

        try:
            bits, stat = running.get_archive("/warp.out.csv.zst")
            with open(instance.archive / "warp.out.csv.zst", "wb") as fd:
                for chunk in bits:
                    fd.write(chunk)
        except docker.errors.NotFound:
            logging.error("warp results file not found. ignoring")

        return result["StatusCode"]

    def results(self, instance: "TestRunner"):
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

    def env(self, instance: "TestRunner"):
        env = self.container.env()
        env["warp-version"] = self.warp_version
        return env

    def kind(self) -> str:
        return "WARP"

    def __str__(self) -> str:
        args_str = ",".join(self.args)
        return f"Warp(workload={self.workload}, args={args_str})"


class TestSuite:
    def __init__(self, name: str, description: str, tests: list[Test]):
        self.name = name
        self.description = description
        self.tests = tests


class TestRunner:
    def __init__(
        self,
        cri: docker.DockerClient,
        db: results_db.ResultsDB,
        under_test_container: ContainerManager,
        storage: Storage,
        archive: pathlib.Path,
        reps: int,
    ):
        self.cri = cri
        self.db = db
        self.under_test_container = under_test_container
        self.storage = storage
        self.archive = archive
        self.reps = reps

        self.under_test_container_env_saved = False

    def __str__(self):
        return f"TestSuite(name={self.suite_name}, id={self.suite_id})"

    def run_suite(self, suite):
        suite_id = results_db.make_id()
        logging.info(f"▶️ STARTING TEST SUITE {suite.name} ID {suite_id}")
        logging.info(f"♻️ {self.reps} REPS / TEST")
        self.db.save_before_suite(suite_id, suite.name, suite.description)
        self.db.save_test_environment_default(suite_id)
        self.db.save_test_environment(suite_id, self.storage.env())

        for test in suite.tests:
            self.run_test(suite_id, test)

        self.db.save_after_suite(suite_id)

    def run_test(self, suite_id, test: Test):
        test_id = results_db.make_id()

        self.db.save_before_test(test_id, str(test), test.kind())
        logging.info(f"🔎 TEST {test} ID {test_id}")

        for rep in range(self.reps):
            logging.info("▶️ %s REP %s/%s", test.name, rep, self.reps)
            try:
                self.run_test_rep(suite_id, test_id, test, rep)
            except Exception:
                logging.exception(
                    "😥 TEST %s REP %s/%s FAILED. Continuing with next test in suite",
                    test,
                    rep,
                    self.reps,
                    exc_info=True,
                )
                return

        self.db.save_after_test(test_id)
        self.db.save_test_environment(suite_id, test.env(self), test.name + "-")

    def run_test_rep(self, suite_id, test_id, test, rep):
        rep_id = results_db.make_id()
        self.db.save_before_rep(test_id, rep_id)
        self.storage.reset()

        logging.info("🔎️ TEST %s REP ID %s", test.name, rep_id)
        self.under_test_container.run()
        self.save_under_test_container_env_once(suite_id)

        # TODO be smarter about the started condition
        time.sleep(10)

        self.storage.drop_caches()
        try:
            returncode = test.run(self)
        except Exception as e:
            logging.exception(
                "😥 TEST %s REP ID %s failed with exception: %s",
                test.name,
                rep_id,
                e,
                exc_info=True,
            )

            self.db.save_after_rep(rep_id, -10, f"failed with exception: {str(e)}")
            raise e

        if returncode != 0:
            logging.error(
                "😥 TEST %s REP ID %s finished with returncode != 0: %s",
                test.name,
                rep_id,
                returncode,
            )
            self.db.save_after_rep(
                rep_id, returncode, "failed with non-zero return code"
            )
            raise AbortTest

        try:
            results = test.results(self)
        except Exception as e:
            logging.exception(
                "😥 TEST %s REP ID %s" " result parsing failed with exception: %s",
                test.name,
                rep_id,
                e,
                exc_info=True,
            )
            self.db.save_after_rep(
                rep_id, -20, f"result parser returned exception: {str(e)}"
            )
            raise e

        self.db.save_after_rep(rep_id, returncode, "successful")
        self.db.save_test_results(rep_id, results)

        self.under_test_container.terminate()

    def save_under_test_container_env_once(self, suite_id):
        # call this after first run() - under test container usually
        # has env data avail afterwards
        if self.under_test_container_env_saved:
            return
        self.db.save_test_environment(
            suite_id, self.under_test_container.env(), "under-test-"
        )


warp_fast = WarpTest(
    "mixed-fast", "mixed", ["--duration=10s", "--objects=10", "--concurrent=1"]
)

warp_mixed_default = WarpTest("mixed-default", "mixed")
warp_get_default = WarpTest("get-default", "get")
warp_put_default = WarpTest("put-default", "put")
warp_list_default = WarpTest("list-default", "list")
warp_delete_default = WarpTest("delete-default", "delete")
warp_stat_default = WarpTest("stat-default", "stat")

fio_sfs_like = FIOTest("fio-sfs-like", SCRIPT_PATH / "fio" / "sfs-like_10M-files.fio")

test_suites = [
    TestSuite(
        "baseline",
        "Test disk baseline performance",
        tests=[
            fio_sfs_like,
        ],
    ),
    TestSuite(
        "demo",
        "Fast demo test suite",
        tests=[
            warp_fast,
        ],
    ),
    TestSuite(
        "s3-simple-micro",
        "S3 micro benchmarks. Simple operations (GET, PUT, DELETE, list, etc.).",
        tests=[
            warp_mixed_default,
            warp_get_default,
            warp_put_default,
            warp_list_default,
            warp_delete_default,
            warp_stat_default,
        ],
    ),
]

test_suites_indexed = {suite.name: suite for suite in test_suites}


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
    logging.getLogger("docker.auth").setLevel(logging.INFO)


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
    type=click.Choice(list(test_suites_indexed.keys())),
    required=True,
    help="Test suite. See vogon2.py for configuration",
)
@click.option(
    "--archive-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Directory to archive test artifacts to",
)
@click.option(
    "--storage-device",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Storage device to run on (e.g /dev/disk/by-id/...)",
)
@click.option(
    "--mountpoint",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Mountpoint for --storage-device",
)
@click.option(
    "--mkfs",
    type=str,
    required=True,
    help="mkfs command. splitted by .split(). append device on call",
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
@click.option(
    "--reset-storage/--no-reset-storage",
    default=False,
    help="Reset storage (umount,mkfs,mount) between reps",
)
@click.option("--repeat", type=int, default=1, help="How many repetitions to run")
def test(
    ctx,
    under_test_image,
    suite,
    archive_dir,
    storage_device,
    mountpoint,
    mkfs,
    docker_api,
    sqlite,
    init_db,
    reset_storage,
    repeat,
):
    cri = docker.DockerClient(base_url=docker_api)
    dbconn = sqlite3.connect(sqlite)
    if init_db:
        results_db.init_db(dbconn)
    db = results_db.ResultsDB(dbconn)
    storage = Storage(reset_storage, storage_device, mountpoint, mkfs.split())
    s3gw = S3GW(cri, under_test_image, storage)
    test_runner = TestRunner(cri, db, s3gw, storage, archive_dir, repeat)
    try:
        test_runner.run_suite(test_suites_indexed[suite])
    finally:
        s3gw.terminate()
        dbconn.close()


if __name__ == "__main__":
    cli(obj={})
