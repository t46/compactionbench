"""Validate a CompactionBench leaderboard submission JSON file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("policy_name", "task", "model", "metrics", "commit", "command")
HEX_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _has_metric(metrics: dict[str, Any], prefix: str, suffix: str = "") -> bool:
    return any(k.startswith(prefix) and k.endswith(suffix) for k in metrics)


def validate_submission(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["submission must be a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing required field: {field}")

    for field in ("policy_name", "task", "model"):
        if field in payload and not isinstance(payload[field], str):
            errors.append(f"{field} must be a string")
        elif field in payload and not payload[field].strip():
            errors.append(f"{field} must not be empty")

    commit = payload.get("commit")
    if commit is not None:
        if not isinstance(commit, str):
            errors.append("commit must be a string")
        elif not HEX_RE.fullmatch(commit):
            errors.append("commit must be a 7-40 character lowercase git hex SHA")

    command = payload.get("command")
    if command is not None:
        if not isinstance(command, str):
            errors.append("command must be the exact shell command as a string")
        elif not command.strip():
            errors.append("command must not be empty")

    metrics = payload.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, dict):
            errors.append("metrics must be an object")
        else:
            if not _has_metric(metrics, "acc_", "_balanced"):
                errors.append("metrics must include at least one acc_*_balanced metric")
            if not _has_metric(metrics, "budgetfrac_"):
                errors.append("metrics must include at least one budgetfrac_* metric")
            if "usage_calls" not in metrics:
                errors.append("metrics must include usage_calls")
            for name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"metric {name} must be numeric")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    args = ap.parse_args()

    try:
        payload = json.loads(args.submission.read_text())
    except json.JSONDecodeError as exc:
        print(f"FAILED: invalid JSON: {exc}", file=sys.stderr)
        return 1

    errors = validate_submission(payload)
    if errors:
        print("FAILED: invalid CompactionBench submission", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"OK: {args.submission} is a valid CompactionBench submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
