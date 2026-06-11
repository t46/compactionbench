from __future__ import annotations

import json
import subprocess
import sys

from scripts.validate_submission import validate_submission


def _valid_submission() -> dict:
    return {
        "policy_name": "ledger+refetch",
        "task": "stateful",
        "model": "qwen2.5:3b-instruct",
        "commit": "e070567",
        "command": (
            "uv run --frozen python eval.py --task stateful --policies "
            "full,keep_last_k:8,ledger+refetch"
        ),
        "metrics": {
            "acc_ledger_refetch_balanced": 0.725,
            "budgetfrac_ledger_refetch": 0.257,
            "usage_calls": 300,
        },
    }


def test_valid_submission_passes_schema() -> None:
    assert validate_submission(_valid_submission()) == []


def test_submission_requires_reproducibility_fields() -> None:
    submission = _valid_submission()
    del submission["commit"]
    del submission["command"]

    errors = validate_submission(submission)
    assert "missing required field: commit" in errors
    assert "missing required field: command" in errors


def test_submission_requires_metric_families() -> None:
    submission = _valid_submission()
    submission["metrics"] = {"acc_ledger_refetch_z0": 0.7, "usage_calls": 300}

    errors = validate_submission(submission)
    assert "metrics must include at least one acc_*_balanced metric" in errors
    assert "metrics must include at least one budgetfrac_* metric" in errors


def test_submission_validator_cli(tmp_path) -> None:
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(_valid_submission()))

    subprocess.run(
        [sys.executable, "scripts/validate_submission.py", str(path)],
        check=True,
    )
