#!/usr/bin/env python3
import json
import logging
import pathlib
import subprocess
import time
from datetime import datetime

import click
from rich.logging import RichHandler


SCRIPT_PATH = pathlib.Path(__file__).parent
LOG = logging.getLogger("vogon-sched")
SLEEP_TIME_SEC = 2


class Job:
    required_keys = ("under_test_image", "suite")
    env_keys = ("under_test_image", "under_test_image_pull", "suite")

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
            f"VOGON_{env_key.upper()}": value
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
                pathlib.Path(config["virtualenv"]) / "bin" / "python",
                SCRIPT_PATH / "vogon2.py",
                "--debug",
                "test",
            ]
        else:
            command = [SCRIPT_PATH / "vogon2.py", "--debug", "test"]

        LOG.info(f"running {self} command {command} with env {env}")

        try:
            proc = subprocess.run(command, env=env, check=True)
            proc.wait()
        except subprocess.CalledProcessError as e:
            LOG.error(f"vogon call failed with {e.returncode}")
            raise e


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
            LOG.info(f"no jobs found. sleeping {SLEEP_TIME_SEC}s")
            time.sleep(SLEEP_TIME_SEC)


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
        LOG.info(f"🏃 {job}")

        try:
            with open(config_file) as fd:
                config = json.load(fd)
                LOG.debug(f"read config {config_file}: {config}")
        except json.decoder.JSONDecodeError:
            LOG.error(f"JSON error in config file {config_file}. retrying load in 10s")
            time.sleep(10)
            continue

        job.move(running)
        try:
            job.run(config)
        except KeyboardInterrupt:
            LOG.info(f"moving interrupted job {job} to failed")
            job.move(failed)
        except subprocess.CalledProcessError:
            LOG.error(f"💣 {job}")
            job.move(failed)
        except Exception:
            LOG.exception(f"💣 {job}", exc_info=True)
            job.move(failed)
        else:
            LOG.info(f"🏁 {job}")
            job.move(done)


if __name__ == "__main__":
    cli(obj={})
