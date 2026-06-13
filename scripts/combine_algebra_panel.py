#!/usr/bin/env python
"""Combine the v0/v1 algebra-aware refetch panels into one metrics file."""

from __future__ import annotations

import argparse
import json


def _load(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v0 = _load(args.v0)
    v1 = _load(args.v1)
    metrics = {}
    metrics.update(_prefixed("stateful", v0))
    metrics.update(_prefixed("accumulate", v1))

    # Primary policy-family headline: matched-budget, algebra-aware refetch
    # improvement over keep_last_k:8, averaged over the two state algebras.
    v0_gain = v0["recoverable_gain_refetch_algebra_8"]
    v1_gain = v1["recoverable_gain_refetch_algebra_8"]
    metrics["algebra_refetch_mean_gain_8"] = round((v0_gain + v1_gain) / 2, 4)
    metrics["algebra_refetch_min_gain_8"] = round(min(v0_gain, v1_gain), 4)
    metrics["algebra_refetch_budgetfrac_mean"] = round(
        (
            v0["budgetfrac_ledger_refetch_algebra"]
            + v1["budgetfrac_ledger_refetch_algebra"]
        )
        / 2,
        4,
    )
    metrics["usage_calls"] = v0.get("usage_calls", 0) + v1.get("usage_calls", 0)

    with open(args.out, "w") as f:
        json.dump(metrics, f)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote combined metrics to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
