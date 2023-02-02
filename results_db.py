#!/usr/bin/env python3
import platform
import sqlite3
import typing
import uuid
from contextlib import closing

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
        env["vogon-platform"] = platform.platform()

        try:
            dist = platform.freedesktop_os_release()
            env["vogon-dist-name"] = dist["NAME"]
            env["vogon-dist-version"] = dist["VERSION_ID"]
        except AttributeError:
            env["vogon-dist-name"] = "unknown"
            env["vogon-dist-version"] = "unknown"

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

    def get_tests(self, suite_id: IDType):
        with closing(self.db.cursor()) as cur:
            return cur.execute(
                """
        SELECT tests.test_id, tests.name, count(test_repetitions.rep_id) as reps,
          tests.kind
        FROM tests
          JOIN test_repetitions ON (tests.test_id = test_repetitions.test_id)
        WHERE tests.suite_id = ?
        GROUP BY test_repetitions.rep_id
        ORDER BY tests.name
        """,
                (suite_id,),
            ).fetchall()

    def get_test_results(self, test_id: IDType, reps: int):
        if reps == 1:
            with closing(self.db.cursor()) as cur:
                return cur.execute(
                    """
                SELECT results.key, results.value, results.unit
                FROM test_repetitions JOIN results
                  ON (test_repetitions.rep_id = results.rep_id)
                WHERE test_repetitions.test_id = ?
                  AND results.unit != "JSON"
                """,
                    (test_id,),
                ).fetchall()
        else:
            raise NotImplementedError(
                "Sorry printing multiple repetition runs not yet supported"
            )

    def get_all_test_results(self, tests: list[tuple[str, str, int, str]]):
        "return: test_name X ((key,value,unit), ..)"
        return [
            (test_name, self.get_test_results(test_id, reps))
            for test_id, test_name, reps, _ in tests
        ]

    @staticmethod
    def get_test_col_set(results):
        cols = ["Test"] + sorted(
            {(row[0], row[2]) for _, rows in results for row in rows}
        )  # -> ("Test", row name[0], ...

        cols_pos_lookup = {
            col: i for i, col in enumerate(cols)
        }  # -> {row name: position in table}

        return cols_pos_lookup

    def get_normalized_results(
        self, key_suffix: str, rep_ids: list[str]
    ) -> tuple[list[float], list[float]]:
        """
        Get results for rep_ids normalized to key_suffix for read and write operations.
        Output one list per operation. Each entry a result for a single rep
        """

        read = []
        write = []

        with closing(self.db.cursor()) as cur:
            for rep in rep_ids:
                data = cur.execute(
                    f"""
                    SELECT value
                    FROM results
                    WHERE rep_id = ?
                      AND key IN ('read-{key_suffix}', 'write-{key_suffix}')
                    """,
                    (rep,),
                ).fetchall()
                if not data:
                    raise Exception(f"no data: {cur} {data} {rep}")
                read.append(float(data[0][0]))
                write.append(float(data[1][0]))

        return (read, write)

    def get_testrun_details(self, suite_id: IDType) -> dict[str, str]:
        "Return testrun details (names, selected environment data) as key value dict"
        results = {}
        with closing(self.db.cursor()) as cur:
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
                try:
                    k = k.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    k = str(k)
                try:
                    v = v.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    v = str(v)
                results[k] = v

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
                  'cpu-model',
                  'cpu-count',
                  'memtotalkb',
                  'node-name'
                )
                """,
                (suite_id,),
            ).fetchall()
            for k, v in row:
                try:
                    k = k.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    k = str(k)
                try:
                    v = v.decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    v = str(v)
                results[k] = v
        return results

    def get_test_runs(self, suite_id: IDType):
        "Get test run information and rep_ids for test suite. Keyed by test name"
        results = {}
        with closing(self.db.cursor()) as cur:
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
            results[name] = {
                "name": name,
                "test_id": test_id,
                "reps": rep_ids.split(),
            }
        return results


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
