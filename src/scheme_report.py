"""
scheme_report.py — Which laundering does a run catch?

    python src/scheme_report.py --data data/raw/HI-Small_Trans.csv \\
        --patterns data/raw/HI-Small_Patterns.txt --run outputs

The IBM AML files come with a *_Patterns.txt file that lists the laundering
transactions belonging to a scheme (fan-out, cycle, ...). The rest of the
laundering belongs to no listed scheme. This report breaks the test-split
results of a run (made with --save_scores) down by scheme, by position inside
the scheme, and by payment format.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from data import COLUMNS

NO_SCHEME = "(no listed scheme)"


def read_schemes(patterns_path: str) -> dict:
    """CSV line of every listed laundering transaction → (scheme type, scheme number, position, scheme size)."""
    membership, scheme, number = {}, None, 0
    with open(patterns_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                scheme = (line.split(" - ")[1].split(":")[0].strip(), [])
            elif line.startswith("END LAUNDERING ATTEMPT"):
                kind, lines = scheme
                for position, member in enumerate(lines):
                    membership[member] = (kind, number, position, len(lines))
                scheme, number = None, number + 1
            elif line and scheme is not None:
                scheme[1].append(line)
    return membership


def laundering_rows(csv_path: str) -> dict:
    """Row number (0 = first data row) → CSV line, for the laundering rows only."""
    rows = {}
    with open(csv_path) as f:
        next(f)
        for number, line in enumerate(f):
            if line.endswith((",1\n", ",1")):
                rows[number] = line.rstrip("\n")
    return rows


def table(title: str, rows: list) -> None:
    print(f"\n{title:<34}{'laundering':>11}{'caught':>8}{'recall':>8}")
    for name, total, caught in rows:
        if total:
            print(f"{name:<34}{total:>11,}{caught:>8,}{caught / total:>8.2f}")


def main(argv: list | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", required=True, help="The *_Trans.csv file the run was trained on")
    p.add_argument("--patterns", required=True, help="The matching *_Patterns.txt file")
    p.add_argument("--run", required=True, help="Output directory of a run made with --save_scores")
    args = p.parse_args(argv)

    scores = np.load(f"{args.run}/tgn_test_scores.npz")
    label = scores["label"].astype(bool)
    flagged = scores["score"] >= float(scores["threshold"])
    active = scores["active_day"]

    # The test split is the tail of the time-ordered stream; `row` maps it back to the CSV.
    row = np.load(args.data + ".events.npz")["row"][-label.size:]
    lines = laundering_rows(args.data)
    membership = read_schemes(args.patterns)
    if not all(int(r) in lines for r in row[label]):
        raise SystemExit("The scores do not line up with the CSV; was the run made on this file?")

    by_scheme = defaultdict(lambda: [0, 0])
    by_position = defaultdict(lambda: [0, 0])
    by_format = defaultdict(lambda: [0, 0])
    by_period = defaultdict(lambda: [0, 0])
    schemes_seen = defaultdict(list)
    format_column = COLUMNS.index("payment_format")

    for index in np.flatnonzero(label):
        line = lines[int(row[index])]
        caught = bool(flagged[index])
        kind, number, position, _ = membership.get(line, (NO_SCHEME, None, None, None))
        for counter, key in (
            (by_scheme, kind),
            (by_format, line.split(",")[format_column]),
            (by_period, ("active days, " if active[index] else "quiet days, ")
                        + ("in a scheme" if number is not None else "no scheme")),
        ):
            counter[key][0] += 1
            counter[key][1] += caught
        if number is not None:
            schemes_seen[number].append(caught)
            key = "first" if position == 0 else "2nd to 5th" if position < 5 else "6th or later"
            by_position[key][0] += 1
            by_position[key][1] += caught

    def rows_of(counter, order=None):
        keys = order or sorted(counter, key=lambda k: -counter[k][0])
        return [(key, *counter[key]) for key in keys if key in counter]

    false_alarms = int(np.sum(flagged & ~label))
    print(f"Test split: {label.size:,} transactions, {int(label.sum()):,} laundering; "
          f"{int(flagged.sum()):,} flagged, {false_alarms:,} of them wrongly")
    table("By scheme", rows_of(by_scheme))
    table("By position in the scheme", rows_of(by_position, ["first", "2nd to 5th", "6th or later"]))
    table("By period", rows_of(by_period))
    table("By payment format", rows_of(by_format))
    some = sum(any(hits) for hits in schemes_seen.values())
    print(f"\nSchemes with transactions in the test split: {len(schemes_seen)}; "
          f"at least one transaction caught in {some}")

    return {
        "by_scheme": dict(by_scheme), "by_position": dict(by_position),
        "by_period": dict(by_period), "by_format": dict(by_format),
        "schemes": len(schemes_seen), "schemes_caught": some, "false_alarms": false_alarms,
    }


if __name__ == "__main__":
    main()
