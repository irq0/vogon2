#!/usr/bin/env python3
import io
import logging
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import toml

scriptpath = os.path.abspath(os.path.dirname(sys.argv[0]))
LOGDIR = os.path.join(scriptpath, "logs")
logfilename = os.path.join(scriptpath, "run.log")


def die(msg):
    print("ERROR:", msg)
    sys.exit(0)


class TestCase:
    def __init__(self, filename):
        with open(filename) as fd:
            config = toml.load(fd)
        with open(os.path.join(scriptpath, "global.toml")) as fd:
            global_config = toml.load(fd)

        self.returncode = None
        self.identifier = os.path.basename(filename)
        self.name = config["test"]["name"]

        self.requirements = config["test"]["requirements"]
        self.test_script = os.path.join(scriptpath, config["run"]["test_script"])
        self.program_under_test = global_config["run"]["program_under_test"]

        if "testenv" in config["run"]:
            self.testenv_script = os.path.join(scriptpath, config["run"]["testenv"])
        else:
            self.testenv_script = None

        self.dbfile = os.path.join(scriptpath, "results", f"{self.identifier}.sqlite")

        if not os.path.exists(self.dbfile):
            self.init_db()

        self.dbconn = sqlite3.connect(self.dbfile)
        self.test_id = self.fetch_test_id()

        self.logdir_test = pathlib.Path(LOGDIR, self.identifier, str(self.test_id))
        self.logdir_test.mkdir(parents=True)

        self.environ = {
            "VOGON_TEST_ID": str(self.test_id),
            "VOGON_DATABASE": str(self.dbfile),
            "VOGON_TEST_LOGDIR": str(self.logdir_test),
            "VOGON_IDENT": str(self.identifier),
            "HOME": os.environ["HOME"],
        }

        if "env" in config:
            for env_key, value in config["env"].items():
                env = f"VOGON_TEST_{env_key.upper()}"
                value = value.replace("$HOME", os.environ["HOME"])
                self.environ[env] = value

        if "env" in global_config:
            for env_key, value in global_config["env"].items():
                env = f"VOGON_{env_key.upper()}"
                value = value.replace("$HOME", os.environ["HOME"])
                self.environ[env] = value

        if "pass" in config:
            condition = config["pass"]["condition"]
            value = config["pass"]["value"]

            if condition == "returncode":
                self.passed = lambda: (self.returncode == int(value))
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
                        name text,
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

        dist = platform.freedesktop_os_release()
        env["dist_name"] = dist["NAME"]
        env["dist_version"] = dist["VERSION_ID"]

        libc = platform.libc_ver()
        env["libc_name"] = libc[0]
        env["libc_version"] = libc[1]

        with open("/proc/meminfo") as file:
            for line in file:
                key, value = line.split(":")
                if key in ("MemTotal", "SwapTotal"):
                    env[key.strip().lower() + "kb"] = (
                        value.strip().replace("kB", "").strip()
                    )

        cur = self.dbconn.cursor()

        for key, value in env.items():
            add_test_environment_value(cur, self.test_id, key, value)

        self.dbconn.commit()
        cur.close()

    def save_test_environment_from_file(self, file):
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
            """insert into test (start, identifier, name, require, runs)
                                    values (strftime('%Y-%m-%d %H:%M:%f'), ?, ?, ?);""",
            (self.identifier, self.name, ",".join(self.requirements), self.runs),
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
            proc = subprocess.run(
                self.testenv_script,
                check=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
            )
            sio = io.BytesIO(proc.stdout)
            self.save_test_environment_from_file(sio)

        # Run the tests
        for i in range(self.runs):
            self.testrun_id = self.fetch_testrun_id()
            self.logdir = self.logdir_test / str(self.testrun_id)
            self.logdir.mkdir()
            self.test_logfilename = self.logdir / "test.log"
            self.prog_under_test_logfilename = self.logdir / "prog_under_test.log"

            with open(self.test_logfilename, "w+") as test_logfile, open(
                self.prog_under_test_logfilename, "w+"
            ) as prog_logfile:

                self.environ["VOGON_TESTRUN_ID"] = str(self.testrun_id)
                self.environ["VOGON_TESTRUN_LOGDIR"] = str(self.logdir)
                self.environ["VOGON_TESTRUN_ARCHIVE"] = str(self.logdir)

                cwd = None
                if self.require("tempdir"):
                    cwd = tempfile.mkdtemp(prefix="vogon_")

                self.save_before_testrun()

                prog = subprocess.Popen(
                    self.program_under_test,
                    shell=False,
                    env=self.environ,
                    bufsize=1,
                    cwd=cwd,
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                    stdout=prog_logfile,
                )
                logging.debug("started prog under test: %s", prog)

                test = subprocess.Popen(
                    self.test_script,
                    shell=False,
                    env=self.environ,
                    cwd=cwd,
                    bufsize=1,
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                    stdout=test_logfile,
                )
                logging.debug("started test script: %s", test)

                logging.debug("waiting for test to finish")
                test.wait()

                logging.debug("test finished. terminating prog under test")
                prog.terminate()
                try:
                    prog.wait(10)
                except TimeoutError:
                    logging.debug("prog failed to terminate. killing")
                    prog.kill()

                self.returncode = test.returncode

                self.save_after_testrun()

                if self.require("tempdir"):
                    os.chdir(scriptpath)
                    shutil.rmtree(cwd)

                self.save_test_results(test_logfile)
                if i == 1:
                    self.save_test_environment_from_file(test_logfile)

                if self.require("once"):
                    break

        self.save_after_test()
        return self.returncode


def run_command_output_to_file(environ, command, file):
    logging.debug("Running command %s env=%s stdout_file=%s", command, environ, file)
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
    logging.basicConfig(
        level=loglevel, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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

        ret = test.run(runs)
        logging.info("-- LAST LOG --")
        logging.info("Log: %s", test.test_logfilename)
        logging.info("Prog under test log: %s", test.prog_under_test_logfilename)
        logging.info("Returncode: %d", ret)
        logging.info(f"== FINISHED TEST {test.identifier} - {test.name} ==")


if __name__ == "__main__":
    main()
