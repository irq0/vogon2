#!/usr/bin/env python3
import platform
import sqlite3
import typing
import uuid

import cpuinfo

IDType = str
TestResultType = tuple[str, str, str]
TestResultsType = list[TestResultType]
TestEnvType = dict[str, typing.Any]


class ResultsDB:
    def __init__(self, sqlite_db: sqlite3.Connection):
        self.db = sqlite_db

    def parse_and_save_test_results(self, rep_id: IDType, logfile: typing.TextIO):
        """
        Parse logfile for lines starting with VOGON_TEST_RESULT and
        try to parse them for test output. This output is then written to
        sqlite database
        """
        cur = self.db.cursor()

        logfile.seek(0)
        for line in logfile:
            if not line.startswith("VOGON_TEST_RESULT:"):
                continue
            line = line[len("VOGON_TEST_RESULT:") :]

            try:
                key, value, unit = line.split(";")
                cur.execute(
                    """insert into results (rep_id, key, value, unit)
                                       values (?, ?, ?, ?);""",
                    (rep_id, key, value, unit),
                )
            except ValueError:
                pass
        self.db.commit()
        cur.close()

    def save_test_results(self, rep_id: IDType, results: TestResultsType):
        """
        Parse logfile for lines starting with VOGON_TEST_RESULT and
        try to parse them for test output. This output is then written to
        sqlite database
        """
        cur = self.db.cursor()
        for key, value, unit in results:
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            cur.execute(
                """insert into results (rep_id, key, value, unit)
                                       values (?, ?, ?, ?);""",
                (rep_id, key, value, unit),
            )
        self.db.commit()
        cur.close()

    def save_test_environment_default(self, test_id: IDType):
        """
        Save default test environment with information about the machine and test runner
        """

        env = cpuinfo.get_cpu_info()
        env["machine_type"] = platform.machine()
        env["os"] = platform.system()
        env["os-release"] = platform.release()
        env["os-version"] = platform.version()
        env["node-name"] = platform.node()

        dist = platform.freedesktop_os_release()
        env["vogon_dist_name"] = dist["NAME"]
        env["vogon_dist_version"] = dist["VERSION_ID"]

        with open("/proc/meminfo") as file:
            for line in file:
                key, value = line.split(":")
                if key in ("MemTotal", "SwapTotal"):
                    env[key.strip().lower() + "kb"] = (
                        value.strip().replace("kB", "").strip()
                    )

        self.save_test_environment(test_id, env)

    def parse_and_save_test_environment_from_file(
        self, test_id: IDType, file: typing.BinaryIO
    ):
        file.seek(0)
        env = {}
        for line in file:
            if not line.startswith(b"VOGON_TEST_ENVIRONMENT:"):
                continue
            line = line[len("VOGON_TEST_ENVIRONMENT:") :]

            try:
                key, value = line.decode("utf-8").split(";")
                env[key] = value
            except ValueError:
                pass

        self.save_test_environment(test_id, env)

    def save_test_environment(
        self, suite_id: IDType, env: TestEnvType, prefix: str = ""
    ):
        cur = self.db.cursor()
        for key, value in env.items():
            key = prefix + key
            cur.execute(
                """insert into environment (suite_id, key, value)
                values (?, ?, ?);""",
                (suite_id, key, value),
            )
        self.db.commit()
        cur.close()

    def save_before_suite(self, suite_id: IDType, suite_name: str, description: str):
        cur = self.db.cursor()
        cur.execute(
            """insert into suites (suite_id, start, name, description)
                                    values (?, strftime('%Y-%m-%d %H:%M:%f'), ?, ?);""",
            (suite_id, suite_name, description),
        )
        self.db.commit()
        cur.close()

    def save_after_suite(self, suite_id: IDType):
        cur = self.db.cursor()
        cur.execute(
            """update suites set finished = strftime('%Y-%m-%d %H:%M:%f')
                       where suite_id = ?;""",
            [suite_id],
        )
        self.db.commit()
        cur.close()

    def save_before_test(
        self, suite_id: IDType, test_id: IDType, test_name: str, kind: str
    ):
        cur = self.db.cursor()
        cur.execute(
            """insert into tests (suite_id, test_id, start, name, kind)
               values (?, ?, strftime('%Y-%m-%d %H:%M:%f'), ?, ?);""",
            (suite_id, test_id, test_name, kind),
        )
        self.db.commit()
        cur.close()

    def save_after_test(self, test_id: IDType):
        cur = self.db.cursor()
        cur.execute(
            """update tests set finished = strftime('%Y-%m-%d %H:%M:%f')
                       where test_id = ?;""",
            [test_id],
        )
        self.db.commit()
        cur.close()

    def save_before_rep(self, test_id: IDType, rep_id: IDType):
        """
        Make new entry in database with test details
        """
        cur = self.db.cursor()
        cur.execute(
            """insert into test_repetitions (rep_id, test_id, start, returncode)
            values (?, ?, strftime('%Y-%m-%d %H:%M:%f'), -1);""",
            (rep_id, test_id),
        )
        self.db.commit()
        cur.close()

    def save_after_rep(self, rep_id: IDType, returncode: int, message: str):
        """
        Make new entry in database with test details
        """
        cur = self.db.cursor()
        cur.execute(
            """update test_repetitions set finished = strftime('%Y-%m-%d %H:%M:%f')
                       where rep_id = ?;""",
            [rep_id],
        )
        cur.execute(
            """update test_repetitions set returncode = ? where rep_id = ?;""",
            (returncode, rep_id),
        )
        cur.execute(
            """update test_repetitions set message = ? where rep_id = ?;""",
            (message, rep_id),
        )
        self.db.commit()
        cur.close()


def init_db(dbconn):
    """Initialize database
    suites: an execution of a test suite from start to finish.
    has one or more tests and an environment

    tests: an execution of a test from start to finish. has one
    or more repeditions

    test_repetitions: individual executions of a test. can be one or
    many depending on the configuration. each one has individual results

    results: results of a test repetition. A result is a 3 tuple (key,
    value, unit). some results may also be JSON data.

    environment: key value data describing the test environment. once per test

    """
    cur = dbconn.cursor()
    cur.execute(
        """
        create table suites (
          suite_id text primary key,
          start timestamp,
          finished timestamp,
          name text,
          description text
        );
        """
    )
    cur.execute(
        """
        create table tests (
          test_id text primary key,
          suite_id text,
          start timestamp,
          finished timestamp,
          name text,
          kind varchar(10),
          foreign key(suite_id) references suites(suite_id)
        );
        """
    )
    cur.execute(
        """
        create table test_repetitions (
          rep_id text primary key,
          test_id text,
          start timestamp,
          finished timestamp,
          returncode integer,
          message text,
          foreign key(test_id) references tests(test_id)
        );
        """
    )
    cur.execute(
        """
        create table results (
          result_id integer primary key autoincrement,
          rep_id text,
          key text,
          value text,
          unit varchar(20),
          foreign key(rep_id) references test_repetitions(rep_id),
          unique(rep_id, key, unit) on conflict replace
        );
        """
    )
    cur.execute(
        """
        create table environment (
          env_id integer primary key autoincrement,
          suite_id text,
          key text,
          value text,
          foreign key(suite_id) references suites(suite_id),
          unique(suite_id, key) on conflict replace
        );
        """
    )
    cur.close()


def make_id() -> IDType:
    return str(uuid.uuid4())
