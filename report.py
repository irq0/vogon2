#!/usr/bin/env python3
import collections
import io
import itertools
import logging
import pathlib
import sqlite3

import click
import dominate
import dominate.tags as T
import matplotlib.pyplot as plt
import numpy as np
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


def make_comparison_bargraph_svg(labels, y_1, y_2, y_1_label, y_2_label, ylabel, title):
    "Make bar graph with two bars (y_1, y_2) per label"
    xaxis = np.arange(len(labels))
    bar_width = 0.35
    fig, ax = plt.subplots()
    b_1 = ax.bar(xaxis - bar_width / 2, y_1, bar_width, label=y_1_label)
    b_2 = ax.bar(xaxis + bar_width / 2, y_2, bar_width, label=y_2_label)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xaxis, labels)
    ax.legend()

    ax.bar_label(b_1, padding=3)
    ax.bar_label(b_2, padding=3)

    fig.tight_layout()

    svg_fd = io.StringIO()
    fig.savefig(svg_fd, format="svg")

    return svg_fd.getvalue()


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
        out_row = [(test, "")] + [""] * (len(cols.a) - 1)

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
@click.argument("suite-id", type=str)
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
@click.argument("suite-id", type=str)
@click.pass_context
def show(ctx, suite_id):
    "Testrun details"
    db = ctx.obj["db"]
    console = Console()

    table = Table(show_header=False, box=rich.box.SIMPLE)
    table.add_column(style="green")
    table.add_column()
    for k, v in db.get_testrun_details(suite_id).items():
        table.add_row(k, v)
    console.print(table)

    table = Table(box=rich.box.SIMPLE)
    table.add_column("Test Name", style="red")
    table.add_column("Test ID")
    table.add_column("Repetition IDs")

    for test_id, test in db.get_test_runs(suite_id).items():
        table.add_row(test["name"], test_id, ", ".join(test["reps"]))

    console.print(table)


@report.command()
@click.argument("suite-id", type=str)
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
@click.argument("suite-a", type=str)
@click.argument("suite-b", type=str)
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


@report.command()
@click.pass_context
@click.option("--baseline-suite", type=str, required=True)
@click.option(
    "--out",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        allow_dash=False,
        path_type=pathlib.Path,
    ),
    required=True,
)
@click.argument("suite-ids", nargs=-1)
def fancy(ctx, baseline_suite, out, suite_ids):
    "Results comparison table"
    db = ctx.obj["db"]
    suites = [db.get_testrun_details(suite_id) for suite_id in suite_ids]
    # TODO check that same suites were run

    baseline_tests = db.get_test_runs(baseline_suite)
    suite_tests = [db.get_test_runs(suite_id) for suite_id in suite_ids]

    all_test_names = {name for test in suite_tests for name in test.keys()}

    # Bar Graphs: Throughput MB/s and Ops/s for each test in suite
    def bar_graphs():
        div = T.div()
        baseline = baseline_tests[
            "FIO(job_file=/home/vogon/vogon2/fio/fio-rand-RW.fio)"
        ]
        for test_name in all_test_names:
            div.add(T.h3(test_name))
            try:
                rep_ids = [max(baseline["reps"])] + [
                    max(t[test_name]["reps"]) for t in suite_tests
                ]
            except KeyError:
                div.add(T.p("At least one test suite does not have this test :("))
                continue
            labels = ["FIO Baseline"] + [f"{suite['suite_id']:9.9}" for suite in suites]

            # note: convert to MB
            bw_read, bw_write = db.get_normalized_results("bw-mean", rep_ids)
            bw_read, bw_write = [x / 1024**2 for x in bw_read], [
                x / 1024**2 for x in bw_write
            ]
            fig = div.add(T.figure(style="display: inline-block"))
            fig.add_raw_string(
                make_comparison_bargraph_svg(
                    labels,
                    bw_read,
                    bw_write,
                    "read/GET",
                    "write/PUT",
                    "Throughput [MB/s]",
                    "Throughput",
                )
            )

            # note: skips baseline, fio iops not comparable with S3 ops
            ops_read, ops_write = db.get_normalized_results("iops-mean", rep_ids[1:])
            fig = div.add(T.figure(style="display: inline-block"))
            fig.add_raw_string(
                make_comparison_bargraph_svg(
                    labels[1:],
                    ops_read,
                    ops_write,
                    "GET",
                    "PUT",
                    "Ops [1/s]",
                    "Operations",
                )
            )
        return div

    # Comparison Tables
    def comparision_table(a, b):
        div = T.div()
        div.add(T.h3(f"{a['suite_id']:9.9} ➙ {b['suite_id']:9.9}"))
        div.add(T.p(f"{a['under-test-s3gw-version']} ➙ {b['under-test-s3gw-version']}"))
        table = div.add(T.table())
        thead = table.add(T.thead())

        def add_headline(headlines):
            row = thead.add(T.tr())
            for hd in headlines:
                if isinstance(hd, tuple):
                    row.add(T.th(T.strong(hd[0]), T.br(), hd[1]))
                else:
                    row.add(T.th(str(hd)))

        tbody = table.add(T.tbody())

        def add_row(row):
            tbody.add(T.tr((T.td(T.span(val, title=detail)) for val, detail in row)))

        try:
            make_comparison_table(
                db, a["suite_id"], b["suite_id"], add_headline, add_row
            )
        except IncompatibleSuites:
            div.add(T.p("Tests suites incompatible"))

        return div

    # Test environment data
    def env_table():
        div = T.div()

        # list of suite dicts -> dict of rows
        combined = collections.defaultdict(list)
        all_keys = {k for suite in suites for k in suite.keys()}
        for suite in suites:
            for k in all_keys:
                combined[k].append(suite.get(k, "-"))

        def sort_key_fn(thing):
            if thing[0] in ("suite_id", "name"):
                return "AAAAAAA"
            elif thing[0].startswith("under-test-"):
                return "BBBBBBB"
            else:
                return str(thing[0])

        table = div.add(T.table())
        tbody = table.add(T.body())
        for k, vs in sorted(combined.items(), key=sort_key_fn):
            with tbody.add(T.tr()):
                T.th(str(k))
                for v in vs:
                    try:
                        v = v.decode("utf-8")
                    except (UnicodeDecodeError, AttributeError):
                        pass
                    if k == "suite_id":
                        T.td(T.strong(v[:9]), v[9:])
                    else:
                        T.td(v)
        return div

    # Assemble report
    doc = dominate.document(title="PR Performance Report")
    with doc:
        T.div(T.h1("PR Performance Report"), bar_graphs())

        T.div(
            T.h2("Comparision Tables"),
            T.p("> 1 faster, = 1 no change, < 1 slower, > 1.3x 😎"),
            (comparision_table(a, b) for a, b in itertools.combinations(suites, 2)),
        )

        T.div(T.h2("Test Environment"), env_table())

    with open(out, "wb") as fd:
        fd.write(doc.render().encode("utf-8"))
