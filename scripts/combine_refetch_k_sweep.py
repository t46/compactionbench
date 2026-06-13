#!/usr/bin/env python
"""Combine v0/v1 algebra-aware refetch budget-sweep panels."""

from __future__ import annotations

import argparse
import json
import re


def _load(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def _ptag(policy: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", policy.lower()).strip("_")


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="4,6,8,10,12")
    args = ap.parse_args()

    v0 = _load(args.v0)
    v1 = _load(args.v1)
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    metrics = {}
    metrics.update(_prefixed("stateful", v0))
    metrics.update(_prefixed("accumulate", v1))

    gains = []
    budgetfracs = []
    for k in ks:
        refetch = _ptag(f"ledger+refetch_algebra:{k}")
        trunc = _ptag(f"keep_last_k:{k}")

        v0_gain = v0[f"acc_{refetch}_balanced"] - v0[f"acc_{trunc}_balanced"]
        v1_gain = v1[f"acc_{refetch}_balanced"] - v1[f"acc_{trunc}_balanced"]
        mean_gain = round((v0_gain + v1_gain) / 2, 4)
        mean_budget = round(
            (v0[f"budgetfrac_{refetch}"] + v1[f"budgetfrac_{refetch}"]) / 2,
            4,
        )

        metrics[f"k{k}_algebra_refetch_mean_gain_vs_trunc"] = mean_gain
        metrics[f"k{k}_algebra_refetch_min_gain_vs_trunc"] = round(
            min(v0_gain, v1_gain), 4
        )
        metrics[f"k{k}_algebra_refetch_budgetfrac_mean"] = mean_budget
        gains.append(mean_gain)
        budgetfracs.append(mean_budget)

    best_i = max(range(len(ks)), key=lambda i: gains[i])
    metrics["sweep_best_k"] = ks[best_i]
    metrics["sweep_best_mean_gain"] = gains[best_i]
    metrics["sweep_best_budgetfrac_mean"] = budgetfracs[best_i]
    metrics["sweep_min_budgetfrac_mean"] = min(budgetfracs)
    metrics["usage_calls"] = v0.get("usage_calls", 0) + v1.get("usage_calls", 0)

    with open(args.out, "w") as f:
        json.dump(metrics, f)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote combined metrics to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
