#!/usr/bin/env python3
import json
import logging
import pathlib
import platform
import sqlite3
import subprocess
import time
from datetime import datetime

import apprise
import click
import requests
from rich.logging import RichHandler


SCRIPT_PATH = pathlib.Path(__file__).parent
LOG = logging.getLogger("vogon-sched")
SLEEP_TIME_SEC = 2
AUTOGEN_SLEEP_TIME_SEC = 600


class Job:
    required_keys = ("under_test_image", "suite")
    env_keys = ("under_test_image", "under_test_image_pull", "suite", "repetitions")

    @staticmethod
    def parse(job_path: pathlib.Path, rejected_dir: pathlib.Path):
        try:
            return Job(job_path)
        except Exception:
            rejected_fn = (
                rejected_dir / f"{job_path.name}_rejected_{datetime.now().isoformat()}"
            )
            LOG.error(
                f"failed to parse job file {job_path}. "
                f"moving to rejected pile ({rejected_fn})",
                exc_info=True,
            )
            job_path.rename(rejected_fn)

    def __init__(self, job_file: pathlib.Path):
        self.path = job_file

        with open(job_file) as fd:
            job = json.load(fd)

        if not all(key in job.keys() for key in self.required_keys):
            raise Exception(
                "not all required keys present: "
                f"keys={job.keys()} required={self.required_keys}"
            )

        self.name = job_file.stem
        self.environment = {
            f"VOGON_{env_key.upper()}": str(value)
            for env_key, value in job.items()
            if env_key in self.env_keys
        }

    def move(self, dest: pathlib.Path):
        LOG.debug(f"move: {self} -> {dest}")
        self.path = self.path.rename(dest / self.path.name)

    def __str__(self):
        return f"Job(name={self.name}, file={self.path})"

    def run(self, config):
        env = config["environment"] | self.environment
        if "virtualenv" in config.keys():
            command = [
                pathlib.Path(config["virtualenv"]) / "bin" / "python3",
                SCRIPT_PATH / "vogon2.py",
                "--debug",
                "test",
            ]
        else:
            command = [SCRIPT_PATH / "vogon2.py", "--debug", "test"]

        LOG.info(f"running {self} command {command} with env {env}")
        LOG.info(">" * 42)
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            LOG.error(f"vogon call failed with {e.returncode}")
            raise e
        LOG.info("<" * 42)


def todo_iter(todo_dir: pathlib.Path, rejected_dir: pathlib.Path):
    "Yield one task at a time. Wait if non are available"
    while True:
        jobs = sorted(todo_dir.glob("*.json"))
        if jobs:
            LOG.info(f"found {len(jobs)} job(s). processing first.")
            job = Job.parse(jobs[0], rejected_dir)
            if job:
                yield job
        else:
            time.sleep(SLEEP_TIME_SEC)


def latest_quay_tags(repo):
    quay_repo = requests.get(
        f"https://quay.io/api/v1/repository/{repo}"
        "/tag/?limit=100&page=1&onlyActiveTags=true"
    ).json()
    result = [t["name"] for t in quay_repo["tags"]]
    return result


def remember_seen_iter(tags_fn, seen_file: pathlib.Path):
    while True:
        try:
            with open(seen_file) as fd:
                seen = set(fd.read().split(";"))
        except FileNotFoundError:
            seen = set()

        seen.add("latest")
        seen.add("nightly-latest")
        tags = [t for t in tags_fn() if t not in seen]
        LOG.debug("Found unseen tags on quay: %s", tags)

        for tag in tags:
            yield tag
            seen.add(tag)

        with open(seen_file, "w") as fd:
            fd.truncate()
            fd.seek(0)
            fd.write(";".join(seen))

        time.sleep(AUTOGEN_SLEEP_TIME_SEC)


def make_notify(apprise_urls: list[str]):
    if apprise_urls:
        ap = apprise.Apprise()
        for url in apprise_urls:
            ap.add(url)

        def logplusapprise(title, body, **kwargs):
            LOG.info(f"{title}: {body}")
            ap.notify(title=title, body=body, **kwargs)

        return logplusapprise
    else:

        def logonly(title, body, **kwargs):
            LOG.info(f"{title}: {body}")

        return logonly


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    loglevel = [logging.INFO, logging.DEBUG][int(debug)]
    logging.basicConfig(
        level=loglevel,
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.pass_context
@click.option(
    "--todo-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Scheduler todo directory",
)
@click.option(
    "--seen-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Scheduler todo directory",
)
def task_creator(ctx, todo_dir: pathlib.Path, seen_dir: pathlib.Path):
    # quay s3gw/s3gw
    repo = "s3gw/s3gw"
    for tag in remember_seen_iter(
        lambda: latest_quay_tags(repo), seen_dir / "s3gw_s3gw"
    ):
        image = f"quay.io/{repo}:{tag}"
        task = {
            "under_test_image": image,
            "under_test_image_pull": "true",
            "suite": "warp-mixed-long",
        }
        task_fn = f"auto_warp-mixed-long_{tag}.json"
        with open(todo_dir / task_fn, "w") as fd:
            LOG.info(f"Creating task {task_fn}")
            json.dump(task, fd)


@cli.command()
@click.pass_context
@click.option(
    "--report-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Report output directory",
)
@click.option(
    "--sqlite",
    type=str,
    required=True,
    help="Where to find the sqlite database?",
)
@click.option(
    "--config-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Scheduler sched.json config",
)
@click.option(
    "--attach/--no-attach", default=False, help="Attach report to notification"
)
def report_creator(ctx, report_dir: pathlib.Path, sqlite, attach, config_file):
    dbconn = sqlite3.connect(sqlite)
    cur = dbconn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    def last_matching_image_tag(match):
        rows = cur.execute(
            """
        SELECT suites.suite_id, value
        FROM suites
        INNER JOIN environment
        ON (suites.suite_id = environment.suite_id)
        WHERE key = 'under-test-image-tags'
        AND value GLOB ?
        AND suites.name = 'warp-mixed-long'
        ORDER BY finished DESC
        LIMIT 5;
        """,
            (match,),
        ).fetchall()
        result = []
        for k, v in rows:
            try:
                v = v.decode("UTF-8")
            except (UnicodeDecodeError, AttributeError):
                pass
            vs = v.split(";")
            for v in vs:
                if "latest" in v:
                    continue
                tag = v.split(":")[1]
                result.append((k, tag))
        return result

    def last_nightlies():
        # return last_matching_image_tag(r"nightly-\d{4}-\d{2}-\d{2}")
        return last_matching_image_tag("*nightly-*-*-*")

    def last_releases():
        # return last_matching_image_tag(r"v\d+\.\d+\.\d+")
        return last_matching_image_tag(r"*v*.*.*")

    def latest_baseline():
        return cur.execute(
            """
        SELECT suite_id, max(finished)
        FROM suites
        WHERE suites.name = 'baseline';
        """
        ).fetchone()[0]

    def run_report_gen(config, filename, suite_ids):
        if "virtualenv" in config.keys():
            command = [
                pathlib.Path(config["virtualenv"]) / "bin" / "python3",
                SCRIPT_PATH / "vogon2.py",
            ]
        else:
            command = [SCRIPT_PATH / "vogon2.py"]

        command.extend(
            [
                "report",
                "--sqlite",
                sqlite,
                "fancy",
                "--out",
                filename,
                "--baseline-suite",
                latest_baseline(),
            ]
        )

        command.extend(suite_ids)
        LOG.info(f"running command {command}")
        LOG.info(">" * 42)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            LOG.error(f"report generator call failed with {e.returncode}")
            raise e
        LOG.info("<" * 42)

    while True:
        try:
            with open(config_file) as fd:
                config = json.load(fd)
                LOG.debug(f"read config {config_file}: {config}")
        except json.decoder.JSONDecodeError:
            LOG.error(f"JSON error in config file {config_file}. retrying load in 10s")
            time.sleep(10)
        notify = make_notify(config.get("notify", []))

        for group, bench_runs_fn, count in [
            ("nightlies", last_nightlies, 5),
            ("releases", last_releases, 3),
        ]:
            benchmark_runs = list(reversed(bench_runs_fn()[:count]))
            LOG.debug(f"Last {count} runs for {group} group: {benchmark_runs}")
            report_fn = (
                report_dir / f"{group}_{'+'.join([x[1] for x in benchmark_runs])}.html"
            )

            if report_fn.exists():
                LOG.info(f"{report_fn} already generated. skipping")
                continue

            LOG.info(
                f"Generating new report for tags "
                f"{'+'.join([x[1] for x in benchmark_runs])}: {report_fn}"
            )

            run_report_gen(config, report_fn, [x[0] for x in benchmark_runs])
            if attach:
                notify("🕵", "New report", attach=report_fn)
            else:
                notify("🕵", f"New report on {platform.node()} {report_fn}")

        time.sleep(AUTOGEN_SLEEP_TIME_SEC)


@cli.command()
@click.pass_context
@click.option(
    "--config-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Scheduler sched.json config",
)
@click.option(
    "--sched-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
    help="Scheduler directory. With todo, running, done, rejected subdirectories",
)
def run(ctx, config_file, sched_dir: pathlib.Path):
    todo = sched_dir / "todo"
    running = sched_dir / "running"
    done = sched_dir / "done"
    failed = sched_dir / "failed"
    required_dirs = (todo, running, done, failed)
    if not all(dir.exists() for dir in required_dirs):
        raise click.BadParameter(f"sched dir must contain {required_dirs} subdirs")

    for job in todo_iter(todo, failed):
        try:
            with open(config_file) as fd:
                config = json.load(fd)
                LOG.debug(f"read config {config_file}: {config}")
        except json.decoder.JSONDecodeError:
            LOG.error(f"JSON error in config file {config_file}. retrying load in 10s")
            time.sleep(10)
            continue

        notify = make_notify(config.get("notify", []))
        notify("🏃", str(job))

        job.move(running)
        try:
            job.run(config)
        except KeyboardInterrupt:
            LOG.info(f"moving interrupted job {job} to failed")
            job.move(failed)
        except subprocess.CalledProcessError:
            notify("💣", str(job))
            job.move(failed)
        except Exception as ex:
            LOG.exception(f"💣 {job}", exc_info=True)
            notify("💣", f"{job} - {ex}")
            job.move(failed)
        else:
            notify("🏁", str(job))
            job.move(done)


if __name__ == "__main__":
    cli(obj={})
