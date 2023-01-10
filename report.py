#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import run
import logging
import sqlite3

mail_from = "vogon@irq0.org"
mail_to = ["ml@irq0.org"]


def mail_report(text):
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(text)
    msg["Subject"] = "Test results"
    msg["From"] = "Prostetnik Vogon Jeltz <%s>" % mail_from
    msg["To"] = mail_to[0]

    s = smtplib.SMTP("localhost")
    s.sendmail(mail_from, mail_to, msg.as_string())
    s.quit()


def main():
    debug = len(sys.argv) > 1 and "-d" in sys.argv
    loglevel = [logging.WARNING, logging.DEBUG][int(debug)]
    logging.basicConfig(level=loglevel)

    tests = run.get_tests()
    text = []
    overview = []

    for test in tests:
        logging.info("== STARTING REPORT %s - %s ==" % (test.identifier, test.name))
        logging.info(test.description)

        dbconn = sqlite3.connect(test.dbfile)
        cur = dbconn.cursor()
        cur.execute(
            """select testrun_id, timestamp, identifier, returncode, key, value, unit
                       from result, testrun
                       where result.testrun_id = testrun.id
                       and testrun.id = (select max(id) from testrun)"""
        )
        result = cur.fetchall()
        if result == []:
            continue

        overview.append(
            "{0:30} \t [{1}]".format(test.name, ["FAILED", "PASSED"][result[0][3]])
        )

        text.append("\n")
        text.append(
            "===== {0:60} =====".format(
                test.name,
            )
        )
        text.append("Timestamp:       " + result[0][1])
        text.append("Testrun#:        " + str(result[0][0]))
        text.append("Identifier:      " + result[0][2])
        text.append("Test result:     " + ["FAILED", "PASSED"][result[0][3]])
        text.append("Description:     ")
        text.append(test.description)

        text.append("")
        text.append("{0:20} {1:20} {2:20}".format("Key", "Value", "Unit"))
        for row in result:
            text.append("{0:20} {1:20} {2:20}".format(row[4], row[5], row[6]))

        dbconn.close()

        logging.debug(test.dbfile)
        logging.info("== FINISHED REPORT %s - %s ==" % (test.identifier, test.name))

    mail_report(
        "\n".join(
            overview
            + [
                "",
                "",
                "",
            ]
            + text
        )
    )


if __name__ == "__main__":
    main()
