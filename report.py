#!/usr/bin/env python3
import collections
import logging
import pathlib
import sqlite3

import click
import rich
from rich import print as pprint
from rich.console import Console
from rich.table import Table

import results_db

SCRIPT_PATH = pathlib.Path(__file__).parent
LOG = logging.getLogger("vogon-report")


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


def select_print_table(db, sql, params=()):
    "Query db with sql. Print results as formatted table to console"
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


class IncompatibleSuites(Exception):
    pass


def make_comparison_table(
    db: results_db.ResultsDB,
    suite_a: results_db.IDType,
    suite_b: results_db.IDType,
    headline_fn,
    add_row_fn,
):
    """
    Create table comparing test results for suite_a and suite_b

    Output through callback functions:
    headline_fn([(headline, subtitle/tooltip), ..]), called once
    add_row_fn([(title, subtitle/tooltip), ..]), called for each row
    """

    AB = collections.namedtuple("AB", ["a", "b"])
    tests = AB(db.get_tests(suite_a), db.get_tests(suite_b))
    results = AB(db.get_all_test_results(tests.a), db.get_all_test_results(tests.b))
    cols = AB(db.get_test_col_set(results.a), db.get_test_col_set(results.b))

    if {test[1] for test in tests.a} != {test[1] for test in tests.b}:
        pprint(tests)
        raise IncompatibleSuites("Different set of tests")
    if cols.a != cols.b:
        pprint(cols)
        raise IncompatibleSuites("Different set of result columns")

    cols = cols.a
    headlines = []
    for col in cols.a.keys():
        if isinstance(col, tuple):
            headlines.append(col)
        else:
            headlines.append((str(col), ""))
    headline_fn(headlines)

    for (test_a, rows_a), (test_b, rows_b) in zip(results.a, results.b):
        if test_a != test_b:
            pprint(test_a, test_b)
            raise IncompatibleSuites
        test = test_a
        out_row = [(test, "")] + [""] * (len(cols) - 1)

        for a, b in zip(rows_a, rows_b):
            if a[0] != b[0]:
                pprint(a, b)
                raise IncompatibleSuites

            try:
                va, vb = float(a[1]), float(b[1])
                unit = a[2]
                if va > 0:
                    value = (
                        f"{vb/va:1.2f}x",
                        f"{format_value(va, unit)}￫{format_value(vb, unit)}",
                    )
                else:
                    value = ("-", "")
            except Exception:
                LOG.exception("💣 %s %s", a, b)
                value = ("ERR", "")
            out_row[cols.a[(a[0], a[2])]] = value
        add_row_fn(out_row)


@click.group()
@click.pass_context
@click.option(
    "--sqlite", type=str, required=True, help="Where to find the sqlite database?"
)
def report(ctx, sqlite):
    "Query test database"
    dbconn = sqlite3.connect(sqlite)
    cur = dbconn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.close()
    ctx.obj["db"] = results_db.ResultsDB(dbconn)


@report.command()
@click.pass_context
def testruns(ctx):
    "List testruns"
    select_print_table(
        ctx.obj["db"].db,
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
        ctx.obj["db"].db,
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
    cur = ctx.obj["db"].db.cursor()

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


@report.command()
@click.option("--suite-id", type=str, required=True)
@click.pass_context
def results(ctx, suite_id):
    "Results table"
    db = ctx.obj["db"]
    tests = db.get_tests(suite_id)
    results = db.get_all_test_results(tests)
    cols = db.get_test_col_set(results)

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


@report.command()
@click.option("--rep-id", type=str, required=True)
@click.option("--key", type=str)
@click.pass_context
def result(ctx, rep_id, key):
    "Get single result value"
    cur = ctx.obj["db"].db.cursor()
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


@report.command()
@click.option("--suite-a", type=str, required=True)
@click.option("--suite-b", type=str, required=True)
@click.pass_context
def compare(ctx, suite_a, suite_b):
    "Results comparison table"
    db = ctx.obj["db"]
    console = Console()
    table = Table(
        show_header=True,
        box=rich.box.SIMPLE,
        highlight=True,
        caption=f"Test result comparison. {suite_a} ￫ {suite_b}",
    )

    def add_headline(headlines):
        for hd in headlines:
            if hd[1]:
                headline = f"[bold]{hd[0]}[/bold]\n\\[{hd[1]}]"
            else:
                headline = hd[0]
            table.add_column(headline, justify="right")

    def add_row(row):
        out = []
        for cell in row:
            if cell[1]:
                out.append(f"[bold]{cell[0]}[/bold]\n{cell[1]}")
            else:
                out.append(cell[0])
        table.add_row(*out)

    make_comparison_table(db, suite_a, suite_b, add_headline, add_row)
    console.print(table)
