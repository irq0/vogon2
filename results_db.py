#!/usr/bin/env python3
import platform
import uuid

from cpuinfo import get_cpu_info


class ResultsDB:
    def __init__(self, sqlite_db):
        self.db = sqlite_db

    def parse_and_save_test_results(self, rep_id, logfile):
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

    def save_test_results(self, rep_id, results: []):
        """
        Parse logfile for lines starting with VOGON_TEST_RESULT and
        try to parse them for test output. This output is then written to
        sqlite database
        """
        cur = self.db.cursor()
        for key, value, unit in results:
            cur.execute(
                """insert into results (rep_id, key, value, unit)
                                       values (?, ?, ?, ?);""",
                (rep_id, key, value, unit),
            )
        self.db.commit()
        cur.close()

    def save_test_environment_default(self, test_id):
        """
        Save default test environment with information about the machine and test runner
        """

        env = {}
        env["machine_type"] = platform.machine()
        env["os"] = platform.system()
        env["os_release"] = platform.release()
        env["os_version"] = platform.version()
        env["node_name"] = platform.node()
        cpuinfo = get_cpu_info()
        env["cpu-brand"] = cpuinfo["brand_raw"]
        env["arch"] = cpuinfo["arch"]
        env["cpu-freq"] = cpuinfo["hz_advertised_friendly"]
        env["cpu-count"] = cpuinfo["count"]
        env["cpu-flags"] = " ".join(cpuinfo["flags"])

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

    def parse_and_save_test_environment_from_file(self, test_id, file):
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

    def save_test_environment(self, test_id, env: dict, prefix=None):
        cur = self.db.cursor()
        for key, value in env.items():
            if prefix:
                key = prefix + key
            cur.execute(
                """insert into environment (test_id, key, value)
                values (?, ?, ?);""",
                (test_id, key, value),
            )
        self.db.commit()
        cur.close()

    def save_before_test(self, test_id, test_name):
        """
        Make new entry in database with test details
        """
        cur = self.db.cursor()
        cur.execute(
            """insert into tests (test_id, start, name)
                                    values (?, strftime('%Y-%m-%d %H:%M:%f'), ?);""",
            (test_id, test_name),
        )
        self.db.commit()
        cur.close()

    def save_after_test(self, test_id):
        """
        Make new entry in database with test details
        """
        cur = self.db.cursor()
        cur.execute(
            """update tests set finished = strftime('%Y-%m-%d %H:%M:%f')
                       where test_id = ?;""",
            [self.test_id],
        )
        self.db.commit()
        cur.close()

    def save_before_rep(self, test_id, rep_id):
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

    def save_after_rep(self, rep_id, returncode):
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
        self.db.commit()
        cur.close()


def init_db(dbconn):
    """Initialize database

    tests: an execution of a test suite from start to finish. has one
    or more repeditions and an environment

    test_repetitions: individual executions of a test. can be one or
    many depending on the configuration. each one has individual results

    results: results of a test repetition. A result is a 3 tuple (key,
    value, unit). some results may also be JSON data.

    environment: key value data describing the test environment. once per test

    """
    cur = dbconn.cursor()
    cur.execute(
        """create table tests (
                    test_id text primary key,
                    start timestamp,
                    finished timestamp,
                    name text,
                    runs integer);
                """
    )
    cur.execute(
        """create table test_repetitions (
                    rep_id text primary key,
                    test_id text,
                    start timestamp,
                    finished timestamp,
                    returncode integer);
                """
    )
    cur.execute(
        """create table results (
                     result_id integer primary key autoincrement,
                     rep_id text,
                     key text,
                     value text,
                     unit varchar(20));
                """
    )
    cur.execute(
        """create table environment (
                     env_id integer primary key autoincrement,
                     test_id text,
                     key text,
                     value text);
                """
    )
    cur.close()


def make_id():
    return str(uuid.uuid4())
