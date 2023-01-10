#!/usr/bin/env python3
# äüö
import os.path
import tempfile
from operator import itemgetter
from pprint import pprint

import Gnuplot
from pysqlite2 import dbapi2 as sqlite3


# dbs = ["filebench_micro_ext2fuse.sqlite",
#       "filebench_micro_fuse-ext2.sqlite",
#       "filebench_micro_jext2-client.sqlite",
#       "filebench_micro_jext2-server.sqlite",
#       "filebench_micro_kernel-ext2.sqlite"]

dbs = [
    "filebench_macro_ext2fuse.sqlite",
    "filebench_macro_fuse-ext2.sqlite",
    "filebench_macro_jext2-client.sqlite",
    "filebench_macro_jext2-server.sqlite",
    "filebench_macro_kernel-ext2.sqlite",
]


def get_results(databases):
    results = {}
    for db in databases:
        con = sqlite3.connect(db)
        con.enable_load_extension(True)
        con.load_extension(os.path.expanduser("~/opt/lib/libsqlitefunctions.dylib"))
        con.row_factory = sqlite3.Row

        cur = con.cursor()
        try:
            cur.execute("""select * from statistics""")
        except sqlite3.OperationalError as e:
            print(e)
            print(db)

        for row in cur:
            key = row["key"]

            if key not in results:
                results[key] = {}
            else:
                results[key][db] = (
                    key,
                    row["unit"],
                    row["mean"],
                    row["st_dev"],
                    row["st_err"],
                    row["conf_in1"],
                    row["conf_in2"],
                    db,
                )
    return results


for key, value in get_results(dbs).items():
    print(key)

    items = value.items()
    values = map(itemgetter(1), items)
    keys = map(itemgetter(0), items)

    means = map(itemgetter(2), values)
    confidence_low = map(itemgetter(5), values)
    confidence_high = map(itemgetter(6), values)
    unit = values[0][1].strip()
    measurement = values[0][1]
    names = map(lambda x: x.split("_")[2].split(".")[0], keys)
    title = key.replace("_", " ").split("(")[0]

    pprint(values)
    pprint(map(itemgetter(6), values))
    pprint(map(itemgetter(5), values))

    print(unit)
    pprint(means)

    g = Gnuplot.Gnuplot(debug=1)
    g.title(title)
    g.ylabel(unit)
    g("set xrange [0:%s]" % (len(means) + 1))
    g('set fontpath "Library/Fonts"')
    g("set terminal aqua enhanced")
    g("set boxwidth 0.5")
    g("set style fill transparent solid 0.5 noborder")
    g("set nokey")
    #    g('set mytics 2')
    #    g('set grid mytics ytics')
    g("set grid ytics")

    filename = None
    with tempfile.NamedTemporaryFile(delete=False) as file:

        lines = map(
            lambda x: """%s %s %s %s "%s"\n""" % x,
            zip(
                range(1, len(means) + 1), means, confidence_low, confidence_high, names
            ),
        )

        file.writelines(lines)
        filename = file.name

    d = Gnuplot.File(
        filename,
        using="1:2:3:4:(0.5):xticlabels(5)",
        with_="""boxerrorbars lc rgb "#444444" lw 1.5""",
    )

    g.plot(d)
