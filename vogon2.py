#!/usr/bin/env python3
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
from rich.logging import RichHandler

import report
import results_db

SCRIPT_PATH = pathlib.Path(__file__).parent
LOG = logging.getLogger("vogon")


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
            LOG.debug("Pulled image %s", self.image)
        else:
            self.image = self.cri.images.get(image)
            LOG.debug("Using image %s", self.image)

    def run(self, **kwargs) -> docker.models.containers.Container:
        """wrap docker client run(). runs detached and set
        self.container to the started container. forward args"""

        self.container = self.cri.containers.run(self.image.id, detach=True, **kwargs)
        LOG.debug("Started container for image %s: %s", self.image, self.container.name)
        return self.container

    def run_once(self, **kwargs) -> str:
        "wrap docker client run(). like subprocess.check_output()"
        return self.cri.containers.run(self.image.id, **kwargs)

    def terminate(self):
        "terminate container started with run()"
        res = self.container.stop(timeout=30)
        LOG.info("Terminating container %s: %s", self.container, res)

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
            command=[
                "--rgw-backend-store",
                "sfs",
                "--rgw-enable-ops-log",
                "0",
                "--rgw-log-object-name",
                "0",
                "--debug-rgw",
                "1",
                "--rgw-frontends",
                "beast port=7480, status bind=0.0.0.0 port=9090",
            ],
        )
        ret, version = self.container.exec_run(["radosgw", "--version"])
        if ret == 0:
            self.s3gw_version = version

        LOG.info("🔎 S3GW container: %s", self.container.name)
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

        LOG.info(
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
        LOG.info("Device: %s", d)
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
            LOG.info("💾 storage reset disabled. skipping umount, mkfs, mount")
            return
        try:
            LOG.info(
                f"💾 resetting storage: mountpoint {self.mountpoint}, "
                f"dev {self.partition}, command {self.mkfs_command}"
            )
            for retry_count in range(1, 10):
                try:
                    umount_out = subprocess.check_output(
                        ["sudo", "umount", str(self.mountpoint.absolute())],
                        stderr=subprocess.STDOUT,
                    )
                    LOG.debug("umount: %s", umount_out)
                except subprocess.CalledProcessError as e:
                    if b"apparently in use by the system" in e.output:
                        LOG.debug(f"umount: {e.output}")
                        LOG.waring(
                            f"umount: still in use. retrying umount in a bit. "
                            f"retry {retry_count}"
                        )
                        time.sleep(1)
                        continue
                    # on first run mountpoint is typically not mounted
                    if b"not mounted" not in e.output:
                        raise e
                    break

            time.sleep(1)
            mkfs_out = subprocess.check_output(
                self.mkfs_command + [str(self.partition)], stderr=subprocess.STDOUT
            )
            LOG.debug("mkfs out: %s", mkfs_out)

            mount_out = subprocess.check_output(
                ["sudo", "mount", str(self.partition), str(self.mountpoint.absolute())],
                stderr=subprocess.STDOUT,
            )
            LOG.debug("mount out: %s", mount_out)

            chown_out = subprocess.check_output(
                ["sudo", "chown", "vogon:vogon", self.mountpoint],
                stderr=subprocess.STDOUT,
            )
            LOG.debug("chown out: %s", chown_out)

        except subprocess.CalledProcessError as e:
            LOG.exception(
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
        LOG.info("💾 dropping caches")
        try:
            subprocess.check_call(
                ["sudo", "bash", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
            )
        except subprocess.CalledProcessError as e:
            LOG.exception("os cached drop failed. failing benchmark")
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
            LOG.info(
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
            LOG.exception("fio crashed :()", exc_info=True)
            return -1

        self.data = json.loads(out)

        return 0

    def env(self, instance: "TestRunner"):
        with open(self.job_file) as fd:
            jobfile = fd.read()
        return {
            "fio-version": self.data["fio version"],
            "fio-jobfile": jobfile,
        }

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

    s3_op_to_fio_op = {
        "GET": "read",
        "PUT": "write",
        "DELETE": "delete",
        "LIST": "list",
    }

    def __init__(self, name: str, workload: str, args: list[str] = []):
        super().__init__(name, self.default_container_image)
        self.workload = workload
        self.args = args
        self.raw_results: str = ""
        self.warp_version: str = "unknown"
        self.json_results: dict = {}

    def make_args(self):
        args = []
        args.extend(self.default_args)
        args.append(self.workload)
        args.extend(self.default_workload_args)
        args.extend(self.args)
        return args

    def run(self, instance: "TestRunner"):
        self.container = ContainerManager(instance.cri, self.container_image)
        args = self.make_args()
        LOG.info("🔎 Warp args: %s", args)
        running = self.container.run(
            command=args, network_mode="host", name=f"warp_{instance.rep_id}"
        )
        LOG.info("🔎 Warp container: %s", running.name)

        time.sleep(2)  # ~ time it takes for usage errors to appear
        if running.status == "exited":
            LOG.error(
                "💩 Looks like warp crashed after start. "
                'State: "%s". '
                "Incorrect parameters? Suggest checking logs",
                running.status,
            )
            raise Exception(f"warp failed after start: {running.logs()}")

        ret, version = running.exec_run(cmd=["/warp", "--version"])
        if ret == 0:
            self.warp_version = version
        else:
            LOG.warning(f"warp --version fail: {ret} {version}")
        LOG.debug("Warp version string: %s", self.warp_version)

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
            LOG.error("warp results file not found. ignoring")

        self.last_run_image = self.last_run.commit(
            f"localhost/{self.default_container_image}", f"test_rep_{instance.rep_id}"
        )
        LOG.info("🔎 Commited warp run container to %s", self.last_run_image)

        # this looks like a lot of work and it is. unfortunately using
        # the logger was not reliable enaugh for large json
        container = instance.cri.containers.run(
            self.last_run_image,
            detach=True,
            entrypoint="/bin/sh",
            command=["-c", "/warp analyze --json /warp.out.csv.zst > /warp.json"],
        )
        LOG.info("🔨 extracting warp results. container: %s", container)
        container.wait()

        results_tarball = instance.archive / (instance.rep_id + "_warp.json.tar")
        try:
            bits, stat = container.get_archive("/warp.json")
            with open(results_tarball, "wb") as fd:
                for chunk in bits:
                    fd.write(chunk)
        except docker.errors.NotFound as e:
            LOG.error("warp.json not found. no results. failing")
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
            LOG.error("warp result parsing failed.", exc_info=True)
            LOG.error("data:\n %s", data, exc_info=True)
            self.json_results = {}

        return result["StatusCode"]

    def results(self, instance: "TestRunner"):
        results = []
        if not self.last_run:
            return None

        results.append(("JSON", self.raw_results, "JSON"))
        results.extend(self.normalized_results(self.json_results))
        return results

    @classmethod
    def normalized_results(cls, json_results):
        results = []
        for op in json_results.get("operations", []):
            if op["skipped"]:
                continue
            try:
                type_fio = cls.s3_op_to_fio_op[op["type"]]
            except KeyError:
                continue
            results.append(
                (f"{type_fio}-bw-mean", op["throughput"]["average_bps"], "byte/s")
            )
            results.append(
                (f"{type_fio}-iops-mean", op["throughput"]["average_ops"], "iops")
            )
        return results

    def env(self, instance: "TestRunner"):
        env = self.container.env()
        env["warp-version"] = self.warp_version
        env[self.name + "-args"] = " ".join(self.args)
        return env

    @classmethod
    def kind(cls) -> str:
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
        LOG.info(f"▶️ STARTING TEST SUITE {suite.name} ID {self.suite_id}")
        LOG.info(f"♻️  {self.reps} REPS / TEST")
        self.db.save_before_suite(self.suite_id, suite.name, suite.description)
        self.db.save_test_environment_default(self.suite_id)
        self.db.save_test_environment(self.suite_id, self.storage.env())

        for test in suite.tests:
            self.run_test(test)

        self.db.save_after_suite(self.suite_id)

    def run_test(self, test: Test):
        self.test_id = results_db.make_id()

        self.db.save_before_test(self.suite_id, self.test_id, str(test), test.kind())
        LOG.info(f"🔎 TEST {test} ID {self.test_id}")

        for rep in range(self.reps):
            LOG.info("▶️ %s REP %s/%s", test.name, rep, self.reps)
            try:
                self.run_test_rep(test)
            except Exception:
                LOG.exception(
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

        LOG.info("🔎️ TEST %s REP ID %s", test.name, self.rep_id)
        self.under_test_container.run(name=f"under_test_{self.rep_id}")
        self.save_under_test_container_env_once()

        # TODO be smarter about the started condition
        time.sleep(10)

        self.storage.drop_caches()
        try:
            returncode = test.run(self)
        except Exception as e:
            LOG.exception(
                "😥 TEST %s REP ID %s failed with exception: %s",
                test.name,
                self.rep_id,
                e,
                exc_info=True,
            )

            self.db.save_after_rep(self.rep_id, -10, f"failed with exception: {str(e)}")
            raise e

        if returncode != 0:
            LOG.error(
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
            LOG.exception(
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


test_suites = [
    TestSuite(
        "baseline",
        "Test disk baseline performance",
        tests=[
            FIOTest("fio-rand-RW", SCRIPT_PATH / "fio" / "fio-rand-RW.fio"),
        ],
    ),
    TestSuite(
        "fio-read-write-rw",
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
        "warp-all-simple-default",
        "S3 micro benchmarks. Simple operations (GET, PUT, DELETE, list, etc.).",
        tests=[
            WarpTest("mixed-default", "mixed"),
            WarpTest("get-default", "get"),
            WarpTest("put-default", "put"),
            WarpTest("list-default", "list"),
            WarpTest("delete-default", "delete"),
            WarpTest("stat-default", "stat"),
        ],
    ),
    TestSuite(
        "warp-single-op",
        "Warp: Single Operation Benchmarks",
        tests=[
            # PUT
            WarpTest(
                "put-c20-uniform32M-nomulti",
                "put",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--disable-multipart",
                    "--obj.randsize=false",
                ],
            ),
            WarpTest(
                "put-c20-uniform32M-multi",
                "put",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                ],
            ),
            WarpTest(
                "put-c20-random265-128M",
                "put",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=256,128MiB",
                    "--obj.generator=random",
                    "--obj.randsize=true",
                ],
            ),
            WarpTest(
                "put-c1-uniform32M-nomulti",
                "put",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--disable-multipart",
                    "--obj.randsize=false",
                ],
            ),
            WarpTest(
                "put-c1-uniform32M-multi",
                "put",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                ],
            ),
            # GET
            WarpTest(
                "get-c20-uniform32M",
                "get",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                    "--objects=8192",
                ],
            ),
            WarpTest(
                "get-c20-uniform32M-ranged",
                "get",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                    "--objects=8192",
                    "--range",
                ],
            ),
            WarpTest(
                "get-c20-random265-128M",
                "get",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=256,128MiB",
                    "--obj.generator=random",
                    "--obj.randsize=true",
                    "--objects=2048",
                ],
            ),
            WarpTest(
                "get-c1-uniform32M",
                "get",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                    "--objects=8192",
                ],
            ),
            WarpTest(
                "get-c1-uniform32M-ranged",
                "get",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                    "--objects=8192",
                    "--range",
                ],
            ),
            WarpTest(
                "get-c1-random265-128M",
                "get",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=256,128MiB",
                    "--obj.generator=random",
                    "--obj.randsize=true",
                    "--objects=2048",
                ],
            ),
            # LIST
            WarpTest(
                "list-c20-1k",
                "list",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=1000",
                ],
            ),
            WarpTest(
                "list-c20-10k",
                "list",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=10000",
                ],
            ),
            WarpTest(
                "list-c20-100k",
                "list",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=100000",
                ],
            ),
            WarpTest(
                "list-c1-1k",
                "list",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=1000",
                ],
            ),
            WarpTest(
                "list-c1-10k",
                "list",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=10000",
                ],
            ),
            WarpTest(
                "list-c1-100k",
                "list",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=100000",
                ],
            ),
            # DELETE
            WarpTest(
                "delete-c1-100k-small",
                "delete",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=100000",
                ],
            ),
            WarpTest(
                "delete-c20-100k-small",
                "delete",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=256",
                    "--obj.randsize=false",
                    "--objects=100000",
                ],
            ),
            WarpTest(
                "delete-c20-uniform32M",
                "delete",
                [
                    "--concurrent=20",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                    "--objects=8192",
                ],
            ),
            WarpTest(
                "delete-c1-uniform32M",
                "delete",
                [
                    "--concurrent=1",
                    "--duration=10m",
                    "--obj.size=32MiB",
                    "--obj.randsize=false",
                    "--objects=8192",
                ],
            ),
        ],
    ),
    TestSuite(
        "warp-mixed-long",
        "Warp mixed 30m",
        tests=[
            # default warp mixed with > RAM size dataset
            WarpTest(
                "mixed-30m-big",
                "mixed",
                [
                    "--duration=30m",
                    "--get-distrib=45",
                    "--stat-distrib=30",
                    "--put-distrib=15",
                    "--delete-distrib=10",
                    "--objects=20000",
                    "--obj.size=10MiB",
                ],
            ),
        ],
    ),
    TestSuite(
        "ideas",
        "?",
        tests=[
            WarpTest(
                "restric-alike",
                "put",
                ["--duration=30m", "--concurrent=5", "--obj.size=16MiB"],
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
        level=loglevel,
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)],
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
    envvar="VOGON_UNDER_TEST_IMAGE",
    required=True,
    help="Docker image of the application under test",
)
@click.option(
    "--under-test-image-pull/--no-under-test-image-pull",
    envvar="VOGON_UNDER_TEST_IMAGE_PULL",
    default=True,
    show_default=True,
    help="Pull under test image?",
)
@click.option(
    "--suite",
    type=click.Choice(list(test_suites_indexed.keys())),
    envvar="VOGON_SUITE",
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
    envvar="VOGON_ARCHIVE_DIR",
    required=True,
    help="Directory to archive test artifacts to",
)
@click.option(
    "--storage-device",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=False,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    envvar="VOGON_STORAGE_DEVICE",
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
    envvar="VOGON_STORAGE_PARTITION",
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
    envvar="VOGON_MOUNTPOINT",
    required=True,
    help="Mountpoint for --storage-device",
)
@click.option(
    "--mkfs",
    type=str,
    envvar="VOGON_MKFS",
    required=True,
    help="mkfs command. splitted by .split(). append device on call",
)
@click.option(
    "--docker-api",
    type=str,
    envvar="VOGON_DOCKER_API",
    required=True,
    help="Docker API URI. e.g unix://run/podman/podman.sock",
)
@click.option(
    "--sqlite",
    type=str,
    envvar="VOGON_SQLITE",
    show_default=True,
    required=True,
    help="Where to find the sqlite database?",
)
@click.option(
    "--reset-storage/--no-reset-storage",
    envvar="VOGON_RESET_STORAGE",
    default=False,
    show_default=True,
    help="Reset storage (umount,mkfs,mount) between reps",
)
@click.option(
    "--repeat",
    type=int,
    envvar="VOGON_REPETITIONS",
    default=1,
    help="How many repetitions to run",
)
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
    reset_storage,
    repeat,
):
    cri = docker.DockerClient(base_url=docker_api)
    dbconn = sqlite3.connect(sqlite)
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


@cli.command()
@click.option(
    "--sqlite",
    type=str,
    envvar="VOGON_SQLITE",
    show_default=True,
    required=True,
    help="Where to find the sqlite database?",
)
def init_db(sqlite):
    dbconn = sqlite3.connect(sqlite)
    results_db.init_db(dbconn)
    cur = dbconn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.close()


@cli.command()
@click.option(
    "--sqlite",
    type=str,
    required=True,
    help="Where to find the sqlite database?",
)
def normalize_result_from_json(sqlite):
    db = results_db.ResultsDB(sqlite3.connect(sqlite))
    cur = db.db.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    for test_class in [WarpTest]:
        rep_ids = cur.execute(
            """
            SELECT rep_id
            FROM tests
            INNER JOIN test_repetitions ON (tests.test_id = test_repetitions.test_id)
            WHERE kind = ?
            """,
            (test_class.kind(),),
        ).fetchall()

        for rep_id_tuple in rep_ids:
            rep_id = rep_id_tuple[0]
            row = cur.execute(
                """
                    SELECT value
                    FROM results
                    WHERE rep_id = ?
                    AND unit = 'JSON' AND key = 'JSON'
                    """,
                (rep_id,),
            ).fetchone()

            if not row:
                continue

            jresult = json.loads(row[0])
            normalized = test_class.normalized_results(jresult)
            res = cur.executemany(
                """
                INSERT INTO results (rep_id, key, value, unit)
                VALUES (?, ?, ?, ?)
                """,
                ((rep_id, k, v, u) for k, v, u in normalized),
            )
            print(f"REP ID {rep_id}: {res.rowcount} rows updated")
            db.db.commit()
    cur.close()


if __name__ == "__main__":
    cli.add_command(report.report)
    cli(obj={})
