#!/usr/bin/env python3
import collections
import io
import itertools
import json
import logging
import pathlib
import sqlite3
import contextlib
from contextlib import closing

import docker
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

    for i, _suffix in enumerate(("K", "M", "G"), 2):
        unit = 1024**i
        if b < unit:
            break
    return f"{(1024 * b / unit):1.2f}{_suffix}"


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


def make_comparison_bargraph_svg(labels, ys, ys_label, ylabel):
    "Make bar graph with bars (y_i, y_i+i, .. in ys) per label"
    assert len(ys) == len(ys_label)
    xaxis = np.arange(len(labels))
    fig, ax = plt.subplots()
    bar_width = 1 / len(ys) - 0.02

    for i, y in enumerate(ys):
        bar = ax.bar(xaxis + i * bar_width, y, bar_width, label=ys_label[i])
        ax.bar_label(bar, padding=3)

    ax.set_ylabel(ylabel)
    ax.set_xticks(xaxis, labels)
    ax.legend()

    fig.tight_layout()

    svg_fd = io.StringIO()
    fig.savefig(svg_fd, format="svg", transparent=True)
    plt.close()

    svg = svg_fd.getvalue()
    # hack: remove width and height to make it scalable with CSS
    svg = svg.replace("width", "width_inactive", 1)
    svg = svg.replace("height", "height_inactive", 1)
    return svg


def make_percentiles_svg(ys, ylabel, ymax):
    xaxis = np.arange(101)
    fig, ax = plt.subplots()
    ax.bar(xaxis, ys, width=2, linewidth=0.7, edgecolor="white")
    ax.set_axisbelow(True)
    ax.grid()
    ax.set(xlim=(0, 100), ylim=(0, ymax))
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    svg_fd = io.StringIO()
    fig.savefig(svg_fd, format="svg", transparent=True)
    plt.close()

    svg = svg_fd.getvalue()
    # hack: remove width and height to make it scalable with CSS
    svg = svg.replace("width", "width_inactive", 1)
    svg = svg.replace("height", "height_inactive", 1)
    return svg


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
        out_row = [(test, "")] + [("-", "")] * (len(cols.a) - 1)

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
    db = ctx.obj["db"]
    console = Console()
    table = Table(show_header=True, box=rich.box.SIMPLE)

    cur = db.db.cursor()
    result = cur.execute(
        """
    SELECT
      suites.suite_id
    FROM suites
    ORDER BY suites.start
    """,
        [],
    )

    table.add_column("")
    table.add_column("ID")
    table.add_column("Started")
    table.add_column("Name")
    table.add_column("Runtime [min]")
    table.add_column("#Tests")
    table.add_column("Image Tags")

    for row in result:
        suite_id = row[0]
        try:
            details = db.get_testrun_details(suite_id)
        except Exception:
            continue
        table.add_row(
            details["human-id"],
            suite_id,
            details["start"],
            details["name"],
            details["runtime_min"],
            details["n_tests"],
            details.get("under-test-image-tags", "?"),
        )
    cur.close()
    console.print(table)


@report.command()
@click.pass_context
@click.argument("suite-id", type=str)
def environ(ctx, suite_id):
    "Test environment data"
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
    table.add_column("Test", style="red")
    table.add_column("Description", style="green")
    table.add_column("Testrun ID")
    table.add_column("Success")
    table.add_column("Repetition IDs")

    for test in db.get_test_runs(suite_id).values():
        table.add_row(
            test["name"],
            test["description"],
            test["test_id"],
            ["😫", "✅"][int(test["success"])],
            ", ".join(test["reps"]),
        )

    console.print(table)


@report.command()
@click.argument("suite-id", type=str)
@click.argument("desc", type=str, default="")
@click.pass_context
def rename(ctx, suite_id, desc):
    """
    Set test suite run description. Will become the human-id and
    used in reports afterwards
    """
    db = ctx.obj["db"]
    console = Console()
    details = db.get_testrun_details(suite_id)
    assert details

    table = Table(show_header=False, box=rich.box.SIMPLE)
    table.add_column(style="green")
    table.add_column()
    table.add_row("Before", "-" * 10)
    table.add_row("suite_id", details["suite_id"])
    table.add_row("human-id", details["human-id"])
    table.add_row("description", details["description"])

    with closing(db.db.cursor()) as cur:
        cur.execute(
            "UPDATE suites SET description = ? WHERE suite_id = ?", [desc, suite_id]
        )
        db.db.commit()
    details = db.get_testrun_details(suite_id)
    table.add_row("Now", "-" * 10)
    table.add_row("human-id", details["human-id"])
    table.add_row("description", details["description"])
    console.print(table)


@report.command()
@click.option(
    "--docker-api",
    type=str,
    envvar="VOGON_DOCKER_API",
    default="unix://var/run/docker.sock",
    help="Docker API URI. e.g unix://run/podman/podman.sock",
)
@click.argument("suite-id", type=str)
@click.pass_context
def containers(ctx, docker_api, suite_id):
    "List local containers available still available that ran during test"
    cri = docker.DockerClient(base_url=docker_api)
    db = ctx.obj["db"]
    console = Console()
    all_containers = {c.name: c for c in cri.containers.list(all=True)}

    table = Table(box=rich.box.SIMPLE, title="Containers during test reps")
    table.add_column("Test", style="red")
    table.add_column("Success")
    table.add_column("Rep")
    table.add_column("Containers", no_wrap=True)
    for test in db.get_test_runs(suite_id).values():
        for i, rep in enumerate(test["reps"]):
            rep_containers = [c for name, c in all_containers.items() if rep in name]
            for c in rep_containers:
                del all_containers[c.name]

            table.add_row(
                [test["name"], ""][int(i > 0)],
                ["😫", "✅"][int(test["success"])],
                rep,
                ", ".join(c.name for c in rep_containers),
            )

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
        title=f"Combined test results for test suite run {suite_id}",
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
            with contextlib.suppress(UnicodeDecodeError, AttributeError):
                v = v.decode("utf-8")
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
        title=f"Test result comparison. {suite_a} ￫ {suite_b}",
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

    all_test_names = list({name for test in suite_tests for name in test.keys()})
    all_test_names.sort()

    def fail_table():
        failed_tests = [
            (t[test_name], suites[i])
            for i, t in enumerate(suite_tests)
            for test_name in all_test_names
            if test_name in t and not t[test_name]["success"]
        ]
        div = T.div()
        if failed_tests:
            table = div.add(T.table())
            thead = table.add(T.thead())
            with thead.add(T.tr()):
                T.th("Testrun")
                T.th("Failed test")
            tbody = table.add(T.body())
            for test, suite in failed_tests:
                with tbody.add(T.tr()):
                    T.td(str(suite["human-id"]))
                    T.td(str(test["name"]))
                    # TODO add why?
        else:
            div.add(T.p("All good"))
        return div

    # Bar Graphs: Throughput MB/s and Ops/s for each test in suite
    def bar_graphs():
        all_div = T.div()
        baseline = baseline_tests[
            "FIO(job_file=/home/vogon/vogon2/fio/fio-rand-RW.fio)"
        ]
        for test_name in all_test_names:
            all_div.add(T.h3(test_name))
            try:
                rep_ids = [max(baseline["reps"])] + [
                    max(t[test_name]["reps"]) for t in suite_tests
                ]
            except KeyError:
                all_div.add(T.p("At least one test suite does not have this test :("))
                continue
            if any(not t[test_name]["success"] for t in suite_tests):
                failed_in_suites = [
                    suites[i]
                    for i, t in enumerate(suite_tests)
                    if not t[test_name]["success"]
                ]
                all_div.add(
                    T.p(
                        T.i(
                            "Test failed in suite "
                            + ", ".join(s["human-id"] for s in failed_in_suites)
                        )
                    )
                )

            labels = ["FIO Baseline"] + [f"{suite['human-id']}" for suite in suites]

            bw = db.get_normalized_results("bw-mean", rep_ids)
            bw_read, bw_write = [x / 1024**2 for x in bw["read-bw-mean"]], [
                x / 1024**2 for x in bw["write-bw-mean"]
            ]

            div = all_div.add(T.div(style="display: flex; flex-wrap: wrap;"))
            fig = div.add(T.figure(style="width: 25rem;"))
            fig.add(T.figcaption("Throughput"))
            fig.add_raw_string(
                make_comparison_bargraph_svg(
                    labels,
                    [bw_read, bw_write],
                    ["read/GET", "write/PUT"],
                    "Throughput [MB/s]",
                )
            )

            # note: skips baseline, fio iops not comparable with S3 ops
            ops = db.get_normalized_results("iops-mean", rep_ids[1:])

            ops_ordered = [
                ops.get("read-iops-mean", float("nan")),
                ops.get("write-iops-mean", float("nan")),
                ops.get("delete-iops-mean", float("nan")),
                ops.get("list-iops-mean", float("nan")),
                ops.get("stat-iops-mean", float("nan")),
            ]
            ops_labels = ["GET", "PUT", "DELETE", "LIST", "STAT"]

            fig = div.add(T.figure(style="width: 25rem;"))
            fig.add(T.figcaption("Operations"))
            fig.add_raw_string(
                make_comparison_bargraph_svg(
                    labels[1:], ops_ordered, ops_labels, "Ops [1/s]"
                )
            )
        return all_div

    # Comparison Tables
    def comparision_table(a, b):
        div = T.div()
        div.add(T.h3(f"{a['human-id']} ➙ {b['human-id']}"))
        div.add(T.p(f"{a['description']} ➙ {b['description']}"))
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

    # Op latency
    def warp_latency_graph(rep_id):
        op_ymax = {
            "PUT": 2000,
        }
        all_div = T.div(style="display: flex; flex-wrap: wrap;")
        with closing(db.db.cursor()) as cur:
            data = json.loads(
                cur.execute(
                    """
            SELECT value
            FROM results
            WHERE rep_id = ? and key = 'JSON'
            """,
                    (rep_id,),
                ).fetchone()[0]
            )
            if not data or "operations" not in data:
                all_div.add(T.p(f"No JSON results found for {rep_id}"))
                return all_div
            for op in data["operations"]:
                if op["skipped"]:
                    continue
                fig = all_div.add(
                    T.figure(
                        style="width: 20rem;",
                        title=(
                            "Request duration percentiles in milliseconds,"
                            f" test rep {rep_id}"
                        ),
                    )
                )
                if "single_sized_requests" in op:
                    stats = op["single_sized_requests"]
                    fig.add_raw_string(
                        make_percentiles_svg(
                            stats["dur_percentiles_millis"],
                            "Request latency [ms]",
                            op_ymax.get(op["type"], 1000),
                        )
                    )
                    fig.add(
                        T.figcaption(
                            T.table(
                                T.tr(
                                    T.td("op"),
                                    T.td("fastest [ms]"),
                                    T.td("slowest [ms]"),
                                    T.td("avg [ms]"),
                                    T.td("median [ms]"),
                                ),
                                T.tr(
                                    T.th(op["type"]),
                                    T.td(stats["fastest_millis"]),
                                    T.td(stats["slowest_millis"]),
                                    T.td(stats["dur_avg_millis"]),
                                    T.td(stats["dur_median_millis"]),
                                ),
                            )
                        )
                    )
                else:
                    fig.add(T.figcaption("No data"))
        return all_div

    # Tabulate latency graphs by testname sections with row per suite
    def latency_table():
        div = T.div()
        for row_test_name in all_test_names:
            div.add(T.h3(row_test_name))
            table = div.add(T.table())
            table.add(T.tr(T.th("Test"), T.th("")))
            for suite, tests in zip(suites, suite_tests):
                for test_name, test in tests.items():
                    if row_test_name == test_name:
                        if test["success"]:
                            table.add(
                                T.tr(
                                    T.th(f"{suite['human-id']}"),
                                    T.td(warp_latency_graph(max(test["reps"]))),
                                )
                            )
                        else:
                            table.add(
                                T.tr(
                                    T.th(f"{suite['human-id']}"),
                                    T.td("failed"),
                                )
                            )

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
            if thing[0] == "human-id":
                return "AAAAAA"
            elif thing[0] in ("suite_id", "name", "description", "url"):
                return "BBBBBBB"
            elif thing[0].startswith("under-test-"):
                return "CCCCCCC"
            else:
                return str(thing[0])

        table = div.add(T.table())
        tbody = table.add(T.body())
        for k, vs in sorted(combined.items(), key=sort_key_fn):
            with tbody.add(T.tr()):
                T.th(str(k))
                for v in vs:
                    with contextlib.suppress(UnicodeDecodeError, AttributeError):
                        v = v.decode("utf-8")
                    if k == "suite_id":
                        T.td(T.strong(v[:9]), v[9:])
                    elif k == "human-id":
                        T.th(v)
                    elif k == "url":
                        T.td(T.a("URL", href=v))
                    else:
                        T.td(v)
        return div

    # Assemble report
    doc = dominate.document(title="S3GW Performance Report", lang="en")
    with doc.head:
        T.meta(http_equiv="Content-Type", content="text/html; charset=utf-8")
    with doc:
        T.div(T.h1("S3GW Performance Report"), bar_graphs())

        T.div(T.h2("Test Failures"), fail_table())

        T.div(
            T.h2("Comparison Tables"),
            T.p("> 1 faster, = 1 no change, < 1 slower, > 1.3x 😎"),
            (comparision_table(a, b) for a, b in itertools.combinations(suites, 2)),
        )

        T.div(T.h2("Test Environment"), env_table())

        T.div(T.h2("Latency Graphs"), latency_table())

    with open(out, "wb") as fd:
        fd.write(doc.render().encode("utf-8"))

    print("done")
