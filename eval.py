"""CompactionBench eval harness (bet-0003).

Runs qwen2.5:3b-instruct (default, local ollama) over StatefulQA items under one
or more context policies, scores exact-match, and writes numeric metrics to
$AAD_METRICS_PATH. Bootstraps its own environment via uv; needs only a running
ollama with the model pulled.

Two modes:
  * default single-config: --policies over one (n_distractors, needle_depth).
  * depth-stratified panel (--needle-depths set): for each policy, sweeps the
    CONTROLLED answer depth and averages over --base-seeds, emitting per-depth
    means (acc_<p>_z<NN>), a depth-balanced headline (acc_<p>_balanced), and a
    cross-seed stability std (seedstd_<p>_balanced). This is the protocol that
    de-confounds the recency/truncation baseline (fair across depths) and holds
    the instruction constant (wording isolated to the verbose_instruction policy).

Usage:
  uv run --frozen python eval.py --n-items 25 \
    --policies full,drop_distractors,drop_relevant,keep_last_k:8,keep_last_k:16,verbose_instruction \
    --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000 --n-distractors 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from compactbench.engine import Engine, strip_think
from compactbench.policies import apply_policy
from compactbench.tasks import accumulate as accumulate_task
from compactbench.tasks import stateful as stateful_task

INT_RE = re.compile(r"-?\d+")


def extract_answer(text: str) -> set[str]:
    return set(INT_RE.findall(strip_think(text)))


def score(gold: str, response: str) -> bool:
    nums = extract_answer(response)
    # Correct iff the model commits to exactly the gold number (no decoys volunteered).
    return nums == {gold}


def run_policy(engine: Engine, items, policy: str, task: str = "stateful") -> tuple[float, float, float]:
    """Return (accuracy, mean_kept_lines, mean_prompt_chars) over the items.

    The two budget terms make the compaction cost explicit: a policy that "wins"
    only by keeping more context is not a compaction win.
    """
    correct = 0
    kept_lines = 0
    prompt_chars = 0
    for it in items:
        messages, info = apply_policy(it, policy, task=task)
        kept_lines += info["kept_lines"]
        prompt_chars += info["prompt_chars"]
        resp = engine.chat(messages)
        if score(it.answer, resp):
            correct += 1
    n = len(items) or 1
    return correct / n, kept_lines / n, prompt_chars / n


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def key(p: str) -> str:
    return "acc_" + re.sub(r"[^a-z0-9]+", "_", p.lower())


def depth_tag(z: float) -> str:
    return f"z{round(z * 100):d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=25)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--base-seeds", default="", help="comma-separated; overrides --base-seed when set")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--engine-seed", type=int, default=0)
    ap.add_argument("--n-distractors", type=int, default=40)
    ap.add_argument("--n-reassign", type=int, default=4, help="stateful: target reassignments")
    ap.add_argument("--n-ops", type=int, default=5, help="accumulate: target ops (1 set + deltas)")
    ap.add_argument(
        "--task",
        default="stateful",
        choices=["stateful", "accumulate"],
        help="stateful (v0, most-recent answer) or accumulate (v1, cumulative-sum answer)",
    )
    ap.add_argument(
        "--policies",
        default="full,drop_distractors,drop_relevant,keep_last_k:8,keep_last_k:16,verbose_instruction",
        help="comma-separated policy names",
    )
    ap.add_argument(
        "--needle-depths",
        default="",
        help="comma-separated controlled answer depths in [0,1]; enables the "
        "depth-stratified multi-seed panel mode",
    )
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    seeds = (
        [int(s) for s in args.base_seeds.split(",") if s.strip()]
        if args.base_seeds.strip()
        else [args.base_seed]
    )
    engine = Engine(model=args.model, seed=args.engine_seed)
    metrics: dict = {}

    # Route to the task family. Both expose make_dataset(base_seed, n_items,
    # n_distractors, needle_depth, ...); only the task-specific op-count kwarg
    # differs (n_reassign for stateful, n_ops for accumulate).
    if args.task == "accumulate":
        make_dataset = accumulate_task.make_dataset
        op_kwargs = {"n_ops": args.n_ops}
    else:
        make_dataset = stateful_task.make_dataset
        op_kwargs = {"n_reassign": args.n_reassign}

    if not args.needle_depths.strip():
        # Single-config mode (legacy): one (n_distractors, random depth) cell.
        items = make_dataset(
            base_seed=seeds[0],
            n_items=args.n_items,
            n_distractors=args.n_distractors,
            **op_kwargs,
        )
        accs = {}
        for p in policies:
            accs[p], kl, pc = run_policy(engine, items, p, task=args.task)
            metrics[key(p)] = accs[p]
            metrics[f"keptlines_{re.sub(r'[^a-z0-9]+', '_', p.lower())}"] = round(kl, 2)
            metrics[f"promptchars_{re.sub(r'[^a-z0-9]+', '_', p.lower())}"] = round(pc, 1)
            print(f"[{p}] acc={accs[p]:.3f} keptlines={kl:.1f} chars={pc:.0f}", flush=True)
        if "drop_distractors" in accs and "full" in accs:
            metrics["ideal_compaction_gain"] = accs["drop_distractors"] - accs["full"]
        metrics["n_distractors"] = args.n_distractors
    else:
        depths = [float(x) for x in args.needle_depths.split(",") if x.strip()]
        # acc[p][depth] = list of per-seed accuracies
        acc: dict = {p: {z: [] for z in depths} for p in policies}
        # Budget axis (depth/seed-invariant by construction for most policies):
        # accumulate mean kept-lines and prompt-chars per policy across all cells.
        budget: dict = {p: {"kl": [], "pc": []} for p in policies}
        for sd in seeds:
            for z in depths:
                items = make_dataset(
                    base_seed=sd,
                    n_items=args.n_items,
                    n_distractors=args.n_distractors,
                    needle_depth=z,
                    **op_kwargs,
                )
                for p in policies:
                    a, kl, pc = run_policy(engine, items, p, task=args.task)
                    acc[p][z].append(a)
                    budget[p]["kl"].append(kl)
                    budget[p]["pc"].append(pc)
                    print(f"[seed={sd}][z={z}][{p}] acc={a:.3f} keptlines={kl:.1f}", flush=True)

        # Aggregate: per-depth mean over seeds, depth-balanced headline, seed-std.
        balanced = {}
        for p in policies:
            for z in depths:
                metrics[f"{key(p)}_{depth_tag(z)}"] = round(_mean(acc[p][z]), 4)
            # Per-seed balanced acc = mean over depths within that seed.
            per_seed_balanced = [
                _mean([acc[p][z][i] for z in depths]) for i in range(len(seeds))
            ]
            balanced[p] = _mean(per_seed_balanced)
            metrics[f"{key(p)}_balanced"] = round(balanced[p], 4)
            metrics[f"seedstd_{re.sub(r'[^a-z0-9]+', '_', p.lower())}_balanced"] = round(
                _std(per_seed_balanced), 4
            )

        # Budget axis: mean kept-lines and prompt-chars per policy, plus the
        # fraction of full-context budget retained (the charter's "≤50% budget"
        # bar is read directly off budgetfrac_<p>).
        full_chars = _mean(budget["full"]["pc"]) if "full" in budget else 0.0
        for p in policies:
            ptag = re.sub(r"[^a-z0-9]+", "_", p.lower())
            mean_pc = _mean(budget[p]["pc"])
            metrics[f"keptlines_{ptag}"] = round(_mean(budget[p]["kl"]), 2)
            metrics[f"promptchars_{ptag}"] = round(mean_pc, 1)
            if full_chars > 0:
                metrics[f"budgetfrac_{ptag}"] = round(mean_pc / full_chars, 4)

        # Headline comparisons (depth-balanced).
        if "drop_distractors" in balanced and "full" in balanced:
            metrics["ideal_compaction_gain_balanced"] = round(
                balanced["drop_distractors"] - balanced["full"], 4
            )
        # Does recency-truncation still beat ideal compaction once depth is fair?
        # Positive => truncation still wins (confound persists); <=0 => fixed.
        for kk in ("keep_last_k:8", "keep_last_k:16"):
            if kk in balanced and "drop_distractors" in balanced:
                metrics[f"truncation_minus_ideal_{kk.split(':')[1]}"] = round(
                    balanced[kk] - balanced["drop_distractors"], 4
                )
        if "verbose_instruction" in balanced and "full" in balanced:
            metrics["verbose_penalty"] = round(
                balanced["verbose_instruction"] - balanced["full"], 4
            )

        # HEADLINE (charter deliverable #2): does a HONEST recoverable-compaction
        # policy beat the fair truncation incumbent? +delta at near-equal-or-lower
        # budget = a real win. Baseline = keep_last_k:8 (fair, depth-balanced).
        base = "keep_last_k:8"
        for pol, mname in (
            ("ledger+refetch", "recoverable_gain_refetch_8"),
            ("ledger+refetch_inplace", "recoverable_gain_refetch_inplace_8"),
            ("ledger_state", "recoverable_gain_ledger_state_8"),
            ("ledger", "recoverable_gain_ledger_8"),
            ("ledger_accumulate", "recoverable_gain_ledger_accumulate_8"),
        ):
            if pol in balanced and base in balanced:
                metrics[mname] = round(balanced[pol] - balanced[base], 4)
        # AccumulatorQA (v1) named headlines. The CONSERVATIVE headline is refetch
        # (recovers all target ops; model does the arithmetic). dedup_penalty
        # quantifies the v1 point: most-recent-wins dedup, the v0 near-solver, now
        # FAILS — it should be <= 0 (no better than truncation, often worse).
        if args.task == "accumulate":
            if "ledger+refetch" in balanced and base in balanced:
                metrics["accum_recoverable_gain_refetch_8"] = round(
                    balanced["ledger+refetch"] - balanced[base], 4
                )
            if "ledger_state" in balanced and base in balanced:
                metrics["accum_dedup_penalty_vs_truncation"] = round(
                    balanced["ledger_state"] - balanced[base], 4
                )
            if "ledger_accumulate" in balanced and "ledger+refetch" in balanced:
                metrics["accum_fold_minus_refetch"] = round(
                    balanced["ledger_accumulate"] - balanced["ledger+refetch"], 4
                )
        # RETRIEVAL-POSITION effect (anomaly test): same content + budget, only
        # the re-insertion slot differs. >0 => placing refetched lines in the
        # most-recent slot (adjacent to query) helps => retrieval ORDER is a
        # first-class compaction lever. ~0 => the refetch win is pure content
        # recovery, position-independent.
        if "ledger+refetch" in balanced and "ledger+refetch_inplace" in balanced:
            metrics["refetch_position_effect"] = round(
                balanced["ledger+refetch"] - balanced["ledger+refetch_inplace"], 4
            )
        metrics["n_distractors"] = args.n_distractors
        metrics["n_seeds"] = len(seeds)
        metrics["n_depths"] = len(depths)

    metrics["usage_calls"] = engine.usage.calls
    metrics["n_items"] = args.n_items

    print(json.dumps(metrics, indent=2), flush=True)

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
