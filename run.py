#!/usr/bin/env python3
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from operator import itemgetter

from ConfigParser import ConfigParser

scriptpath = os.path.abspath(os.path.dirname(sys.argv[0]))
logdir = os.path.join(scriptpath, "logs")
logfilename = os.path.join(scriptpath, "run.log")


def die(msg):
    print("ERROR:", msg)
    sys.exit(0)


class TestCase:
    def __init__(self, filename):
        config = ConfigParser()
        config.read(filename)

        self.returncode = None

        self.identifier = config.get("meta", "ident")
        self.name = config.get("meta", "name")
        self.description = config.get("meta", "description")

        self.logfilename = os.path.join(logdir, self.identifier + ".log")

        self.requirements = map(
            itemgetter(0), filter(lambda x: x[1] == "yes", config.items("require"))
        )

        self.script = os.path.join(scriptpath, config.get("run", "script"))

        if config.has_option("run", "testenv"):
            self.testenv_script = os.path.join(scriptpath, config.get("run", "testenv"))
        else:
            self.testenv_script = None

        if config.has_option("run", "args"):
            self.args = config.get("run", "args").split(" ")
        else:
            self.args = []

        self.dbfile = os.path.join(scriptpath, "results", "%s.sqlite" % self.identifier)

        if not os.path.exists(self.dbfile):
            self.init_db()

        self.dbconn = sqlite3.connect(self.dbfile)
        self.test_id = self.fetch_test_id()

        self.environ = {
            "VOGON_TEST_ID": str(self.test_id),
            "VOGON_DATABASE": str(self.dbfile),
            "VOGON_LOGDIR": str(logdir),
            "VOGON_IDENT": str(self.identifier),
            "HOME": os.environ["HOME"],
        }

        if config.has_section("env"):
            for option in config.options("env"):
                key = f"VOGON_TEST_{option.upper()}"
                value = config.get("env", option)

                value = value.replace("$HOME", os.environ["HOME"])

                self.environ[key] = value

        global_config = ConfigParser()
        global_config.read(os.path.join(scriptpath, "global.conf"))

        if global_config.has_section("env"):
            for option in global_config.options("env"):
                key = f"VOGON_{option.upper()}"
                value = global_config.get("env", option)
                value = value.replace("$HOME", os.environ["HOME"])

                self.environ[key] = value

        if config.has_section("pass"):
            condition = config.get("pass", "condition")
            value = config.get("pass", "value")

            if condition == "returncode":
                self.passed = lambda: (self.returncode == int(value))
            elif condition == "string-match":
                # FIXME
                # self.passed = lambda : (value in self.stdout or value in self.stderr)
                self.passed = True
            else:
                die(".testcase: Condition %s unknown" % condition)
        else:
            self.passed = lambda: True

    def init_db(self):
        """
        Make inital database tables for this test. Called on first run
        of this test
        """
        dbconn = sqlite3.connect(self.dbfile)
        cur = dbconn.cursor()
        cur.execute(
            """create table test (
                        id integer primary key autoincrement,
                        start timestamp,
                        finished timestamp,
                        identifier text,
                        require text,
                        runs integer);
                    """
        )
        cur.execute(
            """create table testrun (
                        id integer primary key autoincrement,
                        test_id integer,
                        start timestamp,
                        finished timestamp,
                        returncode integer);
                    """
        )
        cur.execute(
            """create table result (
                         id integer primary key autoincrement,
                         testrun_id integer,
                         key text,
                         value text,
                         unit varchar(20));
                    """
        )
        cur.execute(
            """create table environment (
                         id integer primary key autoincrement,
                         test_id integer,
                         key text,
                         value text);
                    """
        )
        cur.close()
        dbconn.close()

    def save_test_results(self, logfile):
        """
        Parse logfile for lines starting with VOGON_TEST_RESULT and
        try to parse them for test output. This output is then written to
        sqlite database
        """
        cur = self.dbconn.cursor()

        logfile.seek(0)
        for line in logfile:
            if not line.startswith("VOGON_TEST_RESULT:"):
                continue
            line = line[len("VOGON_TEST_RESULT:") :]

            try:
                key, value, unit = line.split(";")
                cur.execute(
                    """insert into result (testrun_id, key, value, unit)
                                       values (?, ?, ?, ?);""",
                    (self.testrun_id, key, value, unit),
                )
            except ValueError:
                pass
        self.dbconn.commit()
        cur.close()

    def save_test_environment_default(self):
        """
        Parse stdout for lines starting with VOGON_TEST_ENVIRONMENT and add some generic
        information about the environmet the test ran in.
        """

        import platform
        import multiprocessing

        env = {}
        env["processor"] = platform.processor()
        env["processor_count"] = multiprocessing.cpu_count()
        env["machine_type"] = platform.machine()
        env["os"] = platform.system()
        env["os_release"] = platform.release()
        env["os_version"] = platform.version()
        env["node_name"] = platform.node()

        dist = platform.linux_distribution()
        env["dist_name"] = dist[0]
        env["dist_version"] = dist[1]
        env["dist_id"] = dist[2]

        libc = platform.libc_ver()
        env["libc_name"] = libc[0]
        env["libc_version"] = libc[1]

        with open("/proc/meminfo") as file:
            for line in file:
                key, value = line.split(":")
                if key in ("MemTotal", "SwapTotal"):
                    env[key.strip().lower()] = value.strip()

        cur = self.dbconn.cursor()

        for key, value in env.items():
            add_test_environment_value(cur, self.test_id, key, value)

        self.dbconn.commit()
        cur.close()

    def save_test_environment_from_stdout(self, file):
        env = {}
        if file is not None:
            file.seek(0)
            for line in file:
                if not line.startswith("VOGON_TEST_ENVIRONMENT:"):
                    continue
                line = line[len("VOGON_TEST_ENVIRONMENT:") :]

                try:
                    key, value = line.split(";")
                    env[key] = value
                except ValueError:
                    pass
        cur = self.dbconn.cursor()

        for key, value in env.items():
            add_test_environment_value(cur, self.test_id, key, value)
        self.dbconn.commit()
        cur.close()

    def save_before_test(self):
        """
        Make new entry in database with test details
        """
        cur = self.dbconn.cursor()
        cur.execute(
            """insert into test (start, identifier, require, runs)
                                    values (strftime('%Y-%m-%d %H:%M:%f'), ?, ?, ?);""",
            (self.identifier, ",".join(self.requirements), self.runs),
        )
        self.dbconn.commit()
        cur.close()

    def save_after_test(self):
        """
        Make new entry in database with test details
        """
        cur = self.dbconn.cursor()
        cur.execute(
            """update test set finished = strftime('%Y-%m-%d %H:%M:%f')
                       where id = ?;""",
            [str(self.test_id)],
        )
        self.dbconn.commit()
        cur.close()

    def save_before_testrun(self):
        """
        Make new entry in database with test details
        """
        cur = self.dbconn.cursor()
        cur.execute(
            """insert into testrun (start, returncode, test_id)
                                    values (strftime('%Y-%m-%d %H:%M:%f'), ?, ?);""",
            (255, self.test_id),
        )
        self.dbconn.commit()
        cur.close()

    def save_after_testrun(self):
        """
        Make new entry in database with test details
        """
        cur = self.dbconn.cursor()
        cur.execute(
            """update testrun set finished = strftime('%Y-%m-%d %H:%M:%f')
                       where id = ?;""",
            [str(self.testrun_id)],
        )

        cur.execute(
            """update testrun set returncode = ?
                       where id = ?;""",
            (self.passed(), self.testrun_id),
        )
        self.dbconn.commit()
        cur.close()

    def fetch_testrun_id(self):
        """
        Get next Test ID for following sql inserts
        """
        cur = self.dbconn.cursor()
        cur.execute("""select max(id) from testrun""")
        result = cur.fetchone()
        cur.close()

        if result == (None,):
            return 1
        else:
            return result[0] + 1

    def fetch_test_id(self):
        """
        Get next Test ID for following sql inserts
        """
        cur = self.dbconn.cursor()
        cur.execute("""select max(id) from test""")
        result = cur.fetchone()
        cur.close()

        if result == (None,):
            return 1
        else:
            return result[0] + 1

    def require(self, requirement):
        return requirement in self.requirements

    def run(self, runs):
        """
        Run the Test; Parse output; Write to database
        """

        self.runs = runs
        self.save_before_test()

        # Log the test environment to database
        self.save_test_environment_default()

        if self.testenv_script is not None:
            with tempfile.TemporaryFile() as file:
                run_command_output_to_file(self.environ, self.testenv_script, file)
                self.save_test_environment_from_stdout(file)

        # Run the tests
        for i in range(self.runs):
            with open(self.logfilename, "w+") as logfile:
                logfile.truncate(0)

                self.testrun_id = self.fetch_testrun_id()

                self.environ["VOGON_TESTRUN_ID"] = str(self.testrun_id)

                if self.require("root"):
                    command = ["sudo", "-E"] + [self.script] + self.args
                else:
                    command = [self.script] + self.args

                if self.require("tempdir"):
                    tempdir = tempfile.mkdtemp(prefix="vogon_")
                    os.chdir(tempdir)

                self.save_before_testrun()
                self.returncode = run_command_output_to_file(
                    self.environ, command, logfile
                )
                self.save_after_testrun()

                if i == 1:
                    self.save_test_environment_from_stdout(logfile)

                if self.require("tempdir"):
                    os.chdir(scriptpath)
                    shutil.rmtree(tempdir)

                self.save_test_results(logfile)

                if self.require("once"):
                    break

        self.save_after_test()
        return self.returncode


def run_command_output_to_file(environ, command, file):
    try:
        proc = subprocess.Popen(
            command,
            shell=False,
            env=environ,
            bufsize=-1,
            stdout=file,
            stderr=subprocess.STDOUT,
        )
        proc.communicate()
        return proc.returncode
    except OSError as e:
        die(f"run_command: {command} -> {e}")


def add_test_environment_value(cur, testrun_id, key, value):
    cur.execute(
        """insert into environment (test_id, key, value)
                                       values (?, ?, ?);""",
        (testrun_id, key, value),
    )


def get_tests(dir):
    test_files = []

    for entry in sorted(os.listdir(dir)):
        test_files.append(get_test(os.path.join(dir, entry)))
    return test_files


def get_test(filename):
    if os.path.isfile(filename) and filename.lower().endswith(".testcase"):
        return TestCase(filename)


def print_usage():
    print(
        """
VOGON at your service
USAGE: run.py [option]
 -d                     debug
 -test <testcase>       run specific testcase
 -testdir <directory>   run all tests in directory"
 -n <number>            number of testruns"
 -h                     help
"""
    )


def main():
    usage = True
    specific_test = False
    testdir = os.path.join(scriptpath, "tests")
    runs = 1

    if len(sys.argv) > 1:
        debug = "-d" in sys.argv
        specific_test = "-test" in sys.argv
        testdir = "-testdir" in sys.argv
        runs = "-n" in sys.argv
        usage = "-h" in sys.argv

    if usage:
        print_usage()
        sys.exit(1)

    if runs:
        runs = int(sys.argv[sys.argv.index("-n") + 1])
    else:
        runs = 1

    if not (specific_test or testdir):
        print_usage()
        die("need either -test or -testdir")

    if specific_test and testdir:
        print_usage()
        die("-test and -testdir are exclusive")

    if specific_test:
        specific_test = sys.argv[sys.argv.index("-test") + 1]
    if testdir:
        testdir = sys.argv[sys.argv.index("-testdir") + 1]

    loglevel = [logging.WARNING, logging.DEBUG][int(debug)]
    logging.basicConfig(level=loglevel)

    if specific_test:
        tests = [
            get_test(specific_test),
        ]

    if testdir:
        tests = get_tests(testdir)

    if len(tests) == 0:
        print_usage()
        die("No tests found :(")

    for test in tests:
        logging.info(f"== STARTING TEST {test.identifier} - {test.name} ==")
        logging.info(f"== DOING {runs} RUNS ==")
        logging.info(test.description)

        ret = test.run(runs)
        logging.debug("-- LAST LOG --")
        logging.debug(open(test.logfilename).read())

        logging.info("returncode: " + str(ret))
        logging.info(f"== FINISHED TEST {test.identifier} - {test.name} ==")


if __name__ == "__main__":
    main()
