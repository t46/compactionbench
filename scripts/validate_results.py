"""Validate reproduce.sh outputs against the verified CompactionBench metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", default="EXPECTED_RESULTS.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip panels that are not present, useful after ./reproduce.sh 3b or 7b",
    )
    args = ap.parse_args()

    expected_path = Path(args.expected)
    results_dir = Path(args.results_dir)
    expected = json.loads(expected_path.read_text())
    default_tol = float(expected.get("tolerance", 0.0))
    failures: list[str] = []
    checked = 0
    panels_checked = 0

    for panel, spec in expected["panels"].items():
        tol = float(spec.get("tolerance", default_tol))
        result_path = results_dir / f"{panel}.json"
        if not result_path.exists():
            if args.allow_missing:
                continue
            failures.append(f"{panel}: missing {result_path}")
            continue

        got = json.loads(result_path.read_text())
        panels_checked += 1
        for metric, want in spec["metrics"].items():
            if metric not in got:
                failures.append(f"{panel}: missing metric {metric}")
                continue
            value = got[metric]
            if abs(float(value) - float(want)) > tol:
                failures.append(f"{panel}.{metric}: got {value}, expected {want} +/- {tol}")
            checked += 1

    if panels_checked == 0:
        failures.append(f"no result panels found in {results_dir}")

    if failures:
        print("FAILED: reproduced metrics differ from EXPECTED_RESULTS.json", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"OK: {checked} reproduced metrics match EXPECTED_RESULTS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
