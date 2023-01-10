#!/usr/bin/env python3
import logging
import sqlite3
import sys

import run

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
        logging.info(f"== STARTING REPORT {test.identifier} - {test.name} ==")
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
            "{:30} \t [{}]".format(test.name, ["FAILED", "PASSED"][result[0][3]])
        )

        text.append("\n")
        text.append(
            "===== {:60} =====".format(
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
        text.append("{:20} {:20} {:20}".format("Key", "Value", "Unit"))
        for row in result:
            text.append(f"{row[4]:20} {row[5]:20} {row[6]:20}")

        dbconn.close()

        logging.debug(test.dbfile)
        logging.info(f"== FINISHED REPORT {test.identifier} - {test.name} ==")

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
