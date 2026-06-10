"""CompactionBench eval harness (bet-0003).

Runs qwen2.5:3b-instruct (default, local ollama) over StatefulQA items under one
or more context policies, scores exact-match, and writes numeric metrics to
$AAD_METRICS_PATH. Bootstraps its own environment via uv; needs only a running
ollama with the model pulled.

Usage:
  uv run --frozen python eval.py --n-items 50 --policies full,drop_distractors,drop_relevant
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from compactbench.engine import Engine, strip_think
from compactbench.policies import apply_policy
from compactbench.tasks.stateful import make_dataset

INT_RE = re.compile(r"-?\d+")


def extract_answer(text: str) -> set[str]:
    return set(INT_RE.findall(strip_think(text)))


def score(gold: str, response: str) -> bool:
    nums = extract_answer(response)
    # Correct iff the model commits to exactly the gold number (no decoys volunteered).
    return nums == {gold}


def run_policy(engine: Engine, items, policy: str) -> dict:
    correct = 0
    resp_chars = 0
    for it in items:
        messages, _info = apply_policy(it, policy)
        resp = engine.chat(messages)
        resp_chars += len(resp)
        if score(it.answer, resp):
            correct += 1
    n = len(items)
    return {
        "policy": policy,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "mean_resp_chars": resp_chars / n if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--engine-seed", type=int, default=0)
    ap.add_argument("--n-distractors", type=int, default=40)
    ap.add_argument("--n-reassign", type=int, default=4)
    ap.add_argument(
        "--policies",
        default="full,drop_distractors,drop_relevant",
        help="comma-separated policy names",
    )
    args = ap.parse_args()

    items = make_dataset(
        base_seed=args.base_seed,
        n_items=args.n_items,
        n_distractors=args.n_distractors,
        n_reassign=args.n_reassign,
    )
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    engine = Engine(model=args.model, seed=args.engine_seed)
    per_policy = {}
    for p in policies:
        res = run_policy(engine, items, p)
        per_policy[p] = res
        print(
            f"[{p}] acc={res['accuracy']:.3f} ({res['correct']}/{res['n']}) "
            f"mean_resp_chars={res['mean_resp_chars']:.1f}",
            flush=True,
        )

    # Flat numeric metrics (harness reads these). Sanitize policy names for keys.
    def key(p: str) -> str:
        return "acc_" + re.sub(r"[^a-z0-9]+", "_", p.lower())

    metrics: dict = {key(p): per_policy[p]["accuracy"] for p in policies}
    if "full" in per_policy and "drop_relevant" in per_policy:
        # Headroom: how much accuracy depends on the answer-bearing context.
        metrics["headroom_full_minus_droprelevant"] = (
            per_policy["full"]["accuracy"] - per_policy["drop_relevant"]["accuracy"]
        )
    if "drop_distractors" in per_policy and "full" in per_policy:
        # Ideal-compaction gap: positive => compaction HELPS (less distraction).
        metrics["ideal_compaction_gain"] = (
            per_policy["drop_distractors"]["accuracy"] - per_policy["full"]["accuracy"]
        )
    metrics["usage_calls"] = engine.usage.calls
    metrics["n_items"] = args.n_items
    metrics["n_distractors"] = args.n_distractors

    out = {
        "metrics": metrics,
        "per_policy": per_policy,
        "config": vars(args),
        "usage": engine.usage.as_dict(),
    }
    print(json.dumps(out["metrics"], indent=2), flush=True)

    path = os.environ.get("AAD_METRICS_PATH")
    if path:
        with open(path, "w") as f:
            json.dump(metrics, f)
        print(f"wrote metrics to {path}", flush=True)
    else:
        print("WARN: AAD_METRICS_PATH unset; metrics not persisted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
