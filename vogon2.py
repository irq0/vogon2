#!/usr/bin/env python3
import collections
import json
import logging
import pathlib
import sqlite3
import subprocess
import tarfile
import time
from typing import Any

import click
import docker
import rich
from rich import print as pprint
from rich.console import Console
from rich.table import Table

import results_db

SCRIPT_PATH = pathlib.Path(__file__).parent


DockerImageType = str


class ContainerManager:
    """Wraps container runtime interface API to simplifiy
    run, terminate, retrieve logs and test environment data
    """

    def __init__(
        self, cri: docker.DockerClient, image: DockerImageType, pull_image: bool = True
    ):
        self.cri = cri
        if pull_image:
            self.image = self.cri.images.pull(image)
            logging.debug("Pulling image %s", self.image)
        else:
            self.image = self.cri.images.get(image)
            logging.debug("Using image %s", self.image)

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
        self,
        cri: docker.DockerClient,
        image: DockerImageType,
        pull_image: bool,
        storage: "Storage",
    ):
        self.storage = storage
        self.s3gw_version = "unknown"
        super().__init__(cri, image, pull_image)

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
        partition: pathlib.Path,
        mountpoint: pathlib.Path,
        mkfs_command: list[str],
    ):
        self.enable_reset_mkfs_mount = enable_reset_mkfs_mount
        self.mountpoint = mountpoint
        self.device = device
        self.partition = partition
        self.mkfs_command = mkfs_command

        logging.info(
            "Storage device: %s, mountpoint %s, partition: %s",
            self.device,
            self.mountpoint,
            self.partition,
        )

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
                f"dev {self.partition}, command {self.mkfs_command}"
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

            time.sleep(1)
            mkfs_out = subprocess.check_output(
                self.mkfs_command + [str(self.partition)], stderr=subprocess.STDOUT
            )
            logging.debug("mkfs out: %s", mkfs_out)

            mount_out = subprocess.check_output(
                ["sudo", "mount", str(self.partition), str(self.mountpoint.absolute())],
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
                [
                    "fio",
                    "--group_reporting",
                    "--output-format=json",
                    str(self.job_file.absolute()),
                ],
                stderr=subprocess.STDOUT,
                cwd=instance.storage.mountpoint,
            )
        except subprocess.CalledProcessError:
            logging.exception("fio crashed :()", exc_info=True)
            return -1

        self.data = json.loads(out)

        return 0

    def env(self, instance: "TestRunner"):
        return {"fio-version": self.data["fio version"]}

    def results(self, instance: "TestRunner"):
        result = [
            ("JSON", json.dumps(self.data), "JSON"),
        ]

        # 'group reporting' enabled -> one job entry with aggregated results
        j = self.data["jobs"][0]
        for op in ("read", "write"):
            for agg in ("min", "max", "mean"):
                result.extend(
                    [
                        (f"{op}-iops-{agg}", str(j[op][f"iops_{agg}"]), "iops"),
                        (f"{op}-bw-{agg}", str(j[op][f"bw_{agg}"] * 1024), "byte/s"),
                    ]
                )

        return result

    def kind(self) -> str:
        return "FIO"

    def __str__(self) -> str:
        return f"FIO(job_file={self.job_file})"


class WarpTest(ContainerizedTest):
    default_container_image = "minio/warp"
    default_args = ["--no-color"]
    default_workload_args = [
        "--host=localhost:7480",
        "--access-key=test",
        "--secret-key=test",
        "--noclear",  # vogon restarts and runs mkfs between runs
        "--benchdata=/warp.out",
    ]

    def __init__(self, name: str, workload: str, args: list[str] = []):
        super().__init__(name, self.default_container_image)
        self.workload = workload
        self.args = args
        self.raw_results: str = ""
        self.json_results: dict = {}

    def run(self, instance: "TestRunner"):
        self.container = ContainerManager(instance.cri, self.container_image)
        self.warp_version = self.container.run_once(command="--version").strip()
        args = []
        args.extend(self.default_args)
        args.append(self.workload)
        args.extend(self.default_workload_args)
        args.extend(self.args)

        running = self.container.run(
            command=args, network_mode="host", name=f"warp_{instance.rep_id}"
        )
        logging.info("🔎 Warp container: %s", running.name)

        result = running.wait()
        self.last_run = running

        try:
            bits, stat = self.last_run.get_archive("/warp.out.csv.zst")
            with open(
                instance.archive / (instance.rep_id + "_warp.out.csv.zst.tar"), "wb"
            ) as fd:
                for chunk in bits:
                    fd.write(chunk)
        except docker.errors.NotFound:
            logging.error("warp results file not found. ignoring")

        self.last_run_image = self.last_run.commit(
            f"localhost/{self.default_container_image}", f"test_rep_{instance.rep_id}"
        )
        logging.info("🔎 Commited warp run container to %s", self.last_run_image)

        # this looks like a lot of work and it is. unfortunately using
        # the logger was not reliable enaugh for large json
        container = instance.cri.containers.run(
            self.last_run_image,
            detach=True,
            entrypoint="/bin/sh",
            command=["-c", "/warp analyze --json /warp.out.csv.zst > /warp.json"],
        )
        logging.info("🔨 extracting warp results. container: %s", container)
        container.wait()

        results_tarball = instance.archive / (instance.rep_id + "_warp.json.tar")
        try:
            bits, stat = container.get_archive("/warp.json")
            with open(results_tarball, "wb") as fd:
                for chunk in bits:
                    fd.write(chunk)
        except docker.errors.NotFound as e:
            logging.error("warp.json not found. no results. failing")
            raise e

        with tarfile.open(results_tarball) as tf:
            for entry in tf:
                fp = tf.extractfile(entry)
                if not fp:
                    continue
                data = fp.read()
                json_start = data.find(b"{")
                json_stop = data.rfind(b"}")
                try:
                    jdata = data[json_start : json_stop + 1].decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    break

        self.raw_results = jdata
        try:
            self.json_results = json.loads(data)
        except json.decoder.JSONDecodeError:
            logging.error("warp result parsing failed.", exc_info=True)
            logging.error("data:\n %s", data, exc_info=True)
            self.json_results = {}

        return result["StatusCode"]

    def results(self, instance: "TestRunner"):
        results = []
        if not self.last_run:
            return None

        results.append(("JSON", self.raw_results, "JSON"))

        for op in self.json_results.get("operations", []):
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
        env[self.name + "-args"] = " ".join(self.args)
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

        self.suite_id = "NA"
        self.test_id = "NA"
        self.rep_id = "NA"

    def __str__(self):
        return f"TestSuite(name={self.suite_name}, suite_id={self.suite_id})"

    def run_suite(self, suite):
        self.suite_id = results_db.make_id()
        logging.info(f"▶️ STARTING TEST SUITE {suite.name} ID {self.suite_id}")
        logging.info(f"♻️ {self.reps} REPS / TEST")
        self.db.save_before_suite(self.suite_id, suite.name, suite.description)
        self.db.save_test_environment_default(self.suite_id)
        self.db.save_test_environment(self.suite_id, self.storage.env())

        for test in suite.tests:
            self.run_test(test)

        self.db.save_after_suite(self.suite_id)

    def run_test(self, test: Test):
        self.test_id = results_db.make_id()

        self.db.save_before_test(self.suite_id, self.test_id, str(test), test.kind())
        logging.info(f"🔎 TEST {test} ID {self.test_id}")

        for rep in range(self.reps):
            logging.info("▶️ %s REP %s/%s", test.name, rep, self.reps)
            try:
                self.run_test_rep(test)
            except Exception:
                logging.exception(
                    "😥 TEST %s REP %s/%s FAILED. Continuing with next test in suite",
                    test,
                    rep,
                    self.reps,
                    exc_info=True,
                )
                return

        self.db.save_after_test(self.test_id)
        self.db.save_test_environment(self.suite_id, test.env(self), "test-")

    def run_test_rep(self, test: Test):
        self.rep_id = results_db.make_id()
        self.db.save_before_rep(self.test_id, self.rep_id)
        self.storage.reset()

        logging.info("🔎️ TEST %s REP ID %s", test.name, self.rep_id)
        self.under_test_container.run(name=f"under_test_{self.rep_id}")
        self.save_under_test_container_env_once()

        # TODO be smarter about the started condition
        time.sleep(10)

        self.storage.drop_caches()
        try:
            returncode = test.run(self)
        except Exception as e:
            logging.exception(
                "😥 TEST %s REP ID %s failed with exception: %s",
                test.name,
                self.rep_id,
                e,
                exc_info=True,
            )

            self.db.save_after_rep(self.rep_id, -10, f"failed with exception: {str(e)}")
            raise e

        if returncode != 0:
            logging.error(
                "😥 TEST %s REP ID %s finished with returncode != 0: %s",
                test.name,
                self.rep_id,
                returncode,
            )
            self.db.save_after_rep(
                self.rep_id, returncode, "failed with non-zero return code"
            )
            raise AbortTest

        try:
            results = test.results(self)
        except Exception as e:
            logging.exception(
                "😥 TEST %s REP ID %s" " result parsing failed with exception: %s",
                test.name,
                self.rep_id,
                e,
                exc_info=True,
            )
            self.db.save_after_rep(
                self.rep_id, -20, f"result parser returned exception: {str(e)}"
            )
            raise e

        self.db.save_after_rep(self.rep_id, returncode, "successful")
        self.db.save_test_results(self.rep_id, results)

        self.under_test_container.terminate()

    def save_under_test_container_env_once(self):
        # call this after first run() - under test container usually
        # has env data avail afterwards
        if self.under_test_container_env_saved:
            return
        self.db.save_test_environment(
            self.suite_id, self.under_test_container.env(), "under-test-"
        )


warp_mixed_default = WarpTest("mixed-default", "mixed")
warp_get_default = WarpTest("get-default", "get")
warp_put_default = WarpTest("put-default", "put")
warp_list_default = WarpTest("list-default", "list")
warp_delete_default = WarpTest("delete-default", "delete")
warp_stat_default = WarpTest("stat-default", "stat")

test_suites = [
    TestSuite(
        "baseline",
        "Test disk baseline performance",
        tests=[
            FIOTest("fio-rand-RW", SCRIPT_PATH / "fio" / "fio-rand-RW.fio"),
            FIOTest("fio-rand-read", SCRIPT_PATH / "fio" / "fio-rand-read.fio"),
            FIOTest("fio.rand-write", SCRIPT_PATH / "fio" / "fio-rand-write.fio"),
        ],
    ),
    TestSuite(
        "demo",
        "Fast demo test suite",
        tests=[
            WarpTest(
                "mixed-fast",
                "mixed",
                ["--duration=10s", "--objects=10", "--concurrent=1"],
            ),
            WarpTest(
                "get-fast", "get", ["--duration=10s", "--objects=10", "--concurrent=1"]
            ),
            WarpTest("put-fast", "put", ["--duration=10s", "--concurrent=1"]),
            WarpTest(
                "list-fast",
                "list",
                ["--duration=10s", "--objects=10", "--concurrent=1"],
            ),
            WarpTest(
                "delete-fast",
                "delete",
                ["--duration=10s", "--objects=10", "--batch=1", "--concurrent=1"],
            ),
            WarpTest(
                "stat-fast",
                "stat",
                ["--duration=10s", "--objects=10", "--concurrent=1"],
            ),
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
    TestSuite(
        "warp-mixed-long",
        "S3 micro benchmarks. Simple operations (GET, PUT, DELETE, list, etc.).",
        tests=[
            WarpTest(
                "mixed-30m-default",
                "mixed",
                ["--duration=30m"],
            ),
            WarpTest(
                "mixed-30m-get-put",
                "mixed",
                ["--duration=30m", "--get-distrib=70", "--put-distrib=30"],
            ),
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
    logging.getLogger("docker.utils.config").setLevel(logging.INFO)


@cli.command()
@click.pass_context
@click.option(
    "--under-test-image",
    type=str,
    required=True,
    help="Docker image of the application under test",
)
@click.option(
    "--under-test-image-pull/--no-under-test-image-pull",
    default=False,
    help="Pull under test image?",
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
    help="Storage device to extract information from (e.g /dev/disk/by-id/...)",
)
@click.option(
    "--storage-partition",
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
    under_test_image_pull,
    suite,
    archive_dir,
    storage_device,
    storage_partition,
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
    cur = dbconn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.close()

    db = results_db.ResultsDB(dbconn)
    storage = Storage(
        reset_storage, storage_device, storage_partition, mountpoint, mkfs.split()
    )
    s3gw = S3GW(cri, under_test_image, under_test_image_pull, storage)
    test_runner = TestRunner(cri, db, s3gw, storage, archive_dir, repeat)
    try:
        test_runner.run_suite(test_suites_indexed[suite])
    finally:
        s3gw.terminate()
        dbconn.close()


@cli.group()
@click.pass_context
@click.option(
    "--sqlite", type=str, required=True, help="Where to find the sqlite database?"
)
def report(ctx, sqlite):
    dbconn = sqlite3.connect(sqlite)
    cur = dbconn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.close()
    ctx.obj["db"] = dbconn


def select_print_table(db, sql, params=()):
    console = Console()
    table = Table(show_header=True, box=rich.box.SIMPLE)

    cur = db.cursor()
    result = cur.execute(sql, params)
    for header in result.description:
        table.add_column(header[0])

    for row in result:
        table.add_row(*(str(col) for col in row))
    cur.close()
    console.print(table)


@report.command()
@click.pass_context
def testruns(ctx):
    "List testruns"
    select_print_table(
        ctx.obj["db"],
        """
    SELECT
      suites.suite_id,
      suites.start,
      round((julianday(suites.finished)-julianday(suites.start)) * 24 * 60)
        as runtime_min,
      suites.name,
      count(test_repetitions.test_id) as n_tests
    FROM suites JOIN tests ON (suites.suite_id = tests.suite_id)
      JOIN test_repetitions ON (tests.test_id = test_repetitions.test_id)
    GROUP BY suites.suite_id
    ORDER BY suites.start
    """,
    )


@report.command()
@click.pass_context
@click.option("--suite-id", type=str, required=True)
def environ(ctx, suite_id):
    "Test enviromnent data"
    select_print_table(
        ctx.obj["db"],
        """
        SELECT
          environment.key,
          environment.value
        FROM suites JOIN environment ON (suites.suite_id = environment.suite_id)
        WHERE suites.suite_id = ?
        ORDER BY key
        """,
        (suite_id,),
    )


@report.command()
@click.option("--suite-id", type=str, required=True)
@click.pass_context
def show(ctx, suite_id):
    "Testrun details"
    cur = ctx.obj["db"].cursor()

    console = Console()
    table = Table(show_header=False, box=rich.box.SIMPLE)
    table.add_column(style="green")
    table.add_column()

    row = cur.execute(
        """
        SELECT
        suites.suite_id,
        suites.name,
        suites.start,
        suites.finished,
        round((julianday(suites.finished)-julianday(suites.start)) * 24 * 60)
          as runtime_min,
        suites.description,
        count(tests.test_id) as n_tests,
        avg((julianday(tests.finished)-julianday(tests.start)) * 24 * 60)
          as avg_test_runtime
        FROM suites JOIN tests ON (suites.suite_id = tests.suite_id)
          JOIN test_repetitions ON (tests.test_id = test_repetitions.test_id)
        WHERE suites.suite_id = ?
        GROUP BY suites.suite_id
        """,
        (suite_id,),
    ).fetchone()
    for (k, _, _, _, _, _, _), v in zip(cur.description, row):
        table.add_row(str(k), str(v))

    row = cur.execute(
        """
        SELECT
        environment.key,
        environment.value
        FROM suites JOIN environment ON (suites.suite_id = environment.suite_id)
        WHERE suites.suite_id = ?
        AND key IN (
          'under-test-s3gw-version',
          'under-test-image-id',
          'under-test-image-tags',
          'os-release',
          'test-warp-version',
          'test-image-id',
          'test-image-tags',
          'image-tags',
          'disk-model',
          'cpu-brand'
        )
        """,
        (suite_id,),
    ).fetchall()
    for k, v in row:
        table.add_row(str(k), str(v))

    console.print(table)

    table = Table(box=rich.box.SIMPLE)
    table.add_column("Test Name", style="red")
    table.add_column("Test ID")
    table.add_column("Repetition IDs")

    tests = cur.execute(
        """
        SELECT
        tests.test_id,
        tests.name,
        group_concat(test_repetitions.rep_id) as reps
        FROM suites JOIN tests ON (suites.suite_id = tests.suite_id)
          JOIN test_repetitions ON (tests.test_id = test_repetitions.test_id)
        WHERE suites.suite_id = ?
        GROUP BY tests.test_id
        ORDER BY tests.name
        """,
        (suite_id,),
    ).fetchall()
    for test_id, name, rep_ids in tests:
        table.add_row(name, test_id, rep_ids)

    console.print(table)
    cur.close()


def get_tests(cur, suite_id):
    return cur.execute(
        """
    SELECT tests.test_id, tests.name, count(test_repetitions.rep_id) as reps, tests.kind
    FROM tests
      JOIN test_repetitions ON (tests.test_id = test_repetitions.test_id)
    WHERE tests.suite_id = ?
    GROUP BY test_repetitions.rep_id
    ORDER BY tests.name
    """,
        (suite_id,),
    ).fetchall()


def get_test_results(cur, test_id, reps):
    if reps == 1:
        rep_results = cur.execute(
            """
            SELECT results.key, results.value, results.unit
            FROM test_repetitions JOIN results
              ON (test_repetitions.rep_id = results.rep_id)
            WHERE test_repetitions.test_id = ?
              AND results.unit != "JSON"
            """,
            (test_id,),
        )
        return rep_results.fetchall()
    else:
        raise NotImplementedError(
            "Sorry printing multiple repetition runs not yet supported"
        )


def get_all_test_results(cur, tests):
    "return: test_name X ((key,value,unit), ..)"
    return [
        (test_name, get_test_results(cur, test_id, reps))
        for test_id, test_name, reps, _ in tests
    ]


def get_test_col_set(cur, results):
    cols = ["Test"] + sorted(
        {(row[0], row[2]) for _, rows in results for row in rows}
    )  # -> ("Test", row name[0], ...

    cols_pos_lookup = {
        col: i for i, col in enumerate(cols)
    }  # -> {row name: position in table}

    return cols_pos_lookup


def format_bytes(b):
    if b < 1024:
        return f"{b:1.2f}"

    for i, suffix in enumerate(("K", "M", "G"), 2):
        unit = 1024**i
        if b < unit:
            break
    return f"{(1024 * b / unit):1.2f}{suffix}"


def format_value(value, unit):
    if unit == "byte/s":
        return format_bytes(float(value))
    else:
        return value


@report.command()
@click.option("--suite-id", type=str, required=True)
@click.pass_context
def results(ctx, suite_id):
    "Results table"
    cur = ctx.obj["db"].cursor()
    tests = get_tests(cur, suite_id)
    results = get_all_test_results(cur, tests)
    cols = get_test_col_set(cur, results)

    console = Console()
    table = Table(
        show_header=True,
        box=rich.box.SIMPLE,
        highlight=True,
        caption=f"Combined test results for test suite run {suite_id}",
    )
    for col in cols.keys():
        if isinstance(col, tuple):
            headline = rf"[bold]{col[0]}[/bold] \[{col[1]}]"
        else:
            headline = str(col)
        table.add_column(headline, justify="right")

    for test, rows in results:
        out_row = [test] + [""] * (len(cols) - 1)
        for k, v, u in rows:
            out_row[cols[(k, u)]] = format_value(v, u)
        table.add_row(*out_row)

    console.print(table)
    cur.close()


@report.command()
@click.option("--rep-id", type=str, required=True)
@click.option("--key", type=str)
@click.pass_context
def result(ctx, rep_id, key):
    "Get single result value"
    cur = ctx.obj["db"].cursor()
    if key:
        data = cur.execute(
            """
            SELECT value
            FROM results
            WHERE rep_id = ? and key = ?
            """,
            (rep_id, key),
        ).fetchone()
        try:
            out = data[0].decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            out = data[0]

        console = Console()
        console.print(out)
    else:
        table = Table(show_header=True, box=rich.box.SIMPLE)
        table.add_column("Key", style="green")
        table.add_column("Unit")
        table.add_column("Value")
        rows = cur.execute(
            """
            SELECT key, value, unit
            FROM results
            WHERE rep_id = ?
            """,
            (rep_id,),
        ).fetchall()
        for k, v, u in rows:
            try:
                v = v.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                pass
            if len(v) > 42:
                v = f"[italic]Skipping {len(v)} byte value[/italic]"
            table.add_row(k, u, format_value(v, u))

        console = Console()
        console.print(table)
    cur.close()


class IncompatibleSuites(Exception):
    pass


@report.command()
@click.option("--suite-a", type=str, required=True)
@click.option("--suite-b", type=str, required=True)
@click.pass_context
def compare(ctx, suite_a, suite_b):
    "Results comparison table"
    cur = ctx.obj["db"].cursor()
    AB = collections.namedtuple("Tests", ["a", "b"])
    tests = AB(get_tests(cur, suite_a), get_tests(cur, suite_b))
    results = AB(get_all_test_results(cur, tests.a), get_all_test_results(cur, tests.b))
    cols = AB(get_test_col_set(cur, results.a), get_test_col_set(cur, results.b))

    if {test[1] for test in tests.a} != {test[1] for test in tests.b}:
        pprint(tests)
        raise IncompatibleSuites("Different set of tests")
    if cols.a != cols.b:
        pprint(cols)
        raise IncompatibleSuites("Different set of result columns")

    console = Console()
    table = Table(
        show_header=True,
        box=rich.box.SIMPLE,
        highlight=True,
        caption=f"Test result comparison. {suite_a} ￫ {suite_b}",
    )

    cols = cols.a
    for col in cols.keys():
        if isinstance(col, tuple):
            headline = rf"[bold]{col[0]}[/bold] \[{col[1]}]"
        else:
            headline = str(col)
        table.add_column(headline, justify="right")

    for (test_a, rows_a), (test_b, rows_b) in zip(results.a, results.b):
        if test_a != test_b:
            pprint(test_a, test_b)
            raise IncompatibleSuites
        test = test_a
        out_row = [test] + [""] * (len(cols) - 1)

        for a, b in zip(rows_a, rows_b):
            if a[0] != b[0]:
                pprint(a, b)
                raise IncompatibleSuites

            try:
                va, vb = float(a[1]), float(b[1])
                unit = a[2]
                if va > 0:
                    value = (
                        f"{vb/va:1.2f}x"
                        f"\n{format_value(va, unit)}￫{format_value(vb, unit)}"
                    )
                else:
                    value = "-"
            except Exception:
                logging.exception("💣 %s %s", a, b)
                value = "ERR"
            out_row[cols[(a[0], a[2])]] = value
        table.add_row(*out_row)

    console.print(table)
    cur.close()


if __name__ == "__main__":
    cli(obj={})
