# CompactionBench (bet-0003)

Measuring **context rot** and **recoverable compaction** on cheap local models.
Two task families: **v0 StatefulQA** (answer = most-recent value) and **v1
AccumulatorQA** (answer = cumulative sum). v1 exists because a most-recent-wins
ledger dedup near-solves v0 (1.000); v1 makes the answer a total no single line
holds, so the benchmark discriminates compaction operators by the **state algebra**
they respect. Verified results across two models (3b protocols p-d3c4bf50,
p-d816ff49, p-8e96fc78, p-9fd9858e; 7b cross-model protocols p-3ad438f7,
p-b7a62692, p-8a283782, p-66c7e90a).

**Start here:** [`FINDINGS.md`](FINDINGS.md) — paper-style write-up of the four
mechanistic findings. **Publish packet:** [`RELEASE.md`](RELEASE.md). **Reproduce
everything:** `./reproduce.sh` (both models; `./reproduce.sh 3b` for the cheap
half). Expected verified metrics live in [`EXPECTED_RESULTS.json`](EXPECTED_RESULTS.json)
and are checked by `scripts/validate_results.py`. **Frozen splits:**
`frozen/splits.json` pins SHA256 of every eval cell (the benchmark is
procedurally generated, so generator+seed+config IS the split);
`scripts/freeze_splits.py --check` detects any drift.

## 2026-06-13 3B replication and algebra-aware refetch update

Independent clean-checkout verification on this checkout reproduced the 3B
CompactionBench ordering at Qwen2.5-3B-Instruct: recoverable compaction still
dominates both full context and truncation on StatefulQA, and the corrected
AccumulatorQA instruction surface preserves the algebra result.

| panel | policy / metric | acc or gain | budgetfrac | protocol / run |
|---|---|---:|---:|---|
| v0 StatefulQA | full | 0.2708 | 1.000 | p-6abf2f5e / r-04e3816f2b |
| v0 StatefulQA | keep_last_k:8 | 0.1750 | 0.2135 | p-6abf2f5e / r-04e3816f2b |
| v0 StatefulQA | ledger+refetch | 0.7250 | 0.2570 | p-6abf2f5e / r-04e3816f2b |
| v0 StatefulQA | recoverable_gain_refetch_8 | +0.5500 |  | verified exact |
| v1 AccumulatorQA | keep_last_k:8 | 0.0958 | 0.1973 | p-b1c6166f / r-8a779662f0 |
| v1 AccumulatorQA | ledger+refetch | 0.2792 | 0.2233 | p-b1c6166f / r-8a779662f0 |
| v1 AccumulatorQA | ledger+refetch_inplace | 0.3417 | 0.2233 | p-b1c6166f / r-8a779662f0 |
| v1 AccumulatorQA | ledger_accumulate | 0.9542 | 0.1339 | p-b1c6166f / r-8a779662f0 |
| v1 AccumulatorQA | recoverable_gain_refetch_inplace_8 | +0.2458 |  | verified within 0.01 |
| paired v0/v1 | ledger+refetch_algebra mean gain | +0.3917 | 0.2402 | p-f5a44444 / r-8b372b694c |
| paired v0/v1 | ledger+refetch_algebra:4 mean gain vs k=4 truncation | +0.4292 | 0.1527 | p-b37160f8 / r-6633ecbea2 |

`ledger+refetch_algebra` is a small policy-family wrapper: it appends refetched
lines adjacent to the query for select-latest state, and preserves chronological
in-place order for accumulative state. The verified primary metric is
`algebra_refetch_mean_gain_8`; clean-checkout verification reproduced 0.3896
within tolerance 0.03 vs the claimed 0.3917.

The low-budget frontier point is `ledger+refetch_algebra:4`: at mean budget
fraction 0.1527, it beats matched `keep_last_k:4` by +0.4292 mean gain
(StatefulQA +0.5500; AccumulatorQA +0.3083). Clean-checkout verification
reproduced the primary exactly within tolerance 0.03.

License: MIT.

## Task: StatefulQA (v0, `--task stateful`)
A long "session log" of typed segments (instruction / relevant state / distractors),
then a query. The answer is the **most recent** value assigned to a target register,
which is reassigned several times amid many distractor turns. Exact-match ground truth.
Two difficulty knobs: context length (`--n-distractors`) and confusability (`--n-reassign`).

Typed segments let an ablation **policy** drop context BY TYPE — the mechanistic
question of the bet (which context elements' loss degrades success).

## Task: AccumulatorQA (v1, `--task accumulate`)
Same shape, but a target register is `set` to a base then `increased`/`decreased`
by a sequence of deltas; the answer is **base + Σ signed deltas** — a value NO
SINGLE LINE contains. Arithmetic is kept small (`--n-ops 3`, deltas 1 digit) so the
bottleneck is finding all relevant lines under distraction, not the addition (3b
clean ceiling ≈ 0.54). This kills the v0 dedup near-solver and tests whether a
compaction operator respects the state's algebra (aggregate vs select-latest).

## Policies (compactbench/policies.py)

**Baselines / oracles** (may read ground-truth segment `kind`):
- `full` — keep everything (lossy-by-distraction baseline)
- `drop_distractors` — keep only answer-bearing state (CHEATING ideal compaction; upper bound)
- `drop_relevant` — drop answer-bearing state (necessity check; floors at 0)
- `keep_last_k:K` — recency truncation (threshold-summarization analog; the fair incumbent)
- `verbose_instruction` — wording ablation (full log + verbose system)

**Honest recoverable-compaction policies** (decide from surface regex + the query
ONLY — never read `kind`/`is_answer`):
- `ledger` — keep assignment-shaped lines, drop filler (achievable analog of `drop_distractors`)
- `ledger_state` — typed LEDGER OF FACTS: latest-value-per-register dedup (most-recent-wins)
- `ledger+refetch` — `keep_last_k:8` + lazy re-fetch of the queried target's lines
  from the dropped tail (recoverable compaction; conservative — model still picks recency).
  On v1 it recovers ALL target ops (completeness), not just one.
- `ledger+refetch_inplace` — same recovered content as `ledger+refetch`, but
  restored before the retained recency window, preserving chronological order.
- `ledger+refetch_algebra` — task-aware wrapper: appended refetch for v0
  select-latest state, chronological in-place refetch for v1 fold state.
  It accepts an optional suffix such as `ledger+refetch_algebra:4` to set the
  retained recency window.
- `ledger_accumulate` (v1) — fold every op per register (set re-bases, inc/dec adjust)
  into a running total. Correct operator for accumulative state; computes the answer
  in-compactor, so reported as the OPERATOR-correctness ceiling, not a model win.

## v0 headline (VERIFIED, protocol p-d816ff49, qwen2.5:3b, depth-balanced)
| policy | acc | budgetfrac |
|---|---|---|
| keep_last_k:8 (incumbent) | 0.175 | 0.214 |
| drop_distractors (cheating ideal) | 0.604 | 0.082 |
| **ledger+refetch** | **0.725** | **0.257** |
| **ledger_state** | **1.000** | **0.143** |

`recoverable_gain_refetch_8 = +0.55`. Plus `refetch_position_effect = +0.30`
(protocol p-8e96fc78): re-injecting recovered facts adjacent to the query beats
original-position by 30pp at identical content/budget.

## v1 headline (VERIFIED, protocol p-b1c6166f, qwen2.5:3b, depth-balanced)
| policy | acc | budgetfrac | note |
|---|---|---|---|
| keep_last_k:8 (incumbent) | 0.096 | 0.197 | truncation |
| **ledger_state (select-latest)** | **0.008** | 0.152 | v0 near-solver — DEAD on v1 |
| ledger+refetch (recover all ops, appended) | 0.279 | 0.223 | conservative recovery, beats truncation |
| ledger+refetch_inplace (chronological) | 0.342 | 0.223 | best conservative v1 placement |
| **ledger_accumulate (fold)** | **0.954** | 0.134 | correct operator ceiling |

`accum_fold_minus_dedup = +0.9458` (verified): two honest typed ledgers, identical
parsing, fold vs select-latest — on accumulative state fold is correct and
select-latest (the v0 near-solver) collapses. **The compaction operator must match
the state's algebra.** budgetfrac = mean prompt-chars / full-context prompt-chars.

## Prior cross-model robustness (VERIFIED, qwen2.5:7b-instruct, same panels, n_ops=3)

These 7B numbers are prior verified evidence from the original release. The
current `EXPECTED_RESULTS.json` fixture is scoped to the 3B bet-0008 update
above, because the v1 task instruction was corrected in this checkout.

| metric | 3b | 7b | protocol (7b) |
|---|---|---|---|
| v0 recoverable_gain_refetch_8 | +0.55 | **+0.78** | p-3ad438f7 |
| v0 refetch_position_effect | +0.30 | **+0.12** | p-b7a62692 |
| v1 accum_fold_minus_dedup | +0.996 | **+0.992** | p-8a283782 |
| v1 accum_recoverable_gain_refetch_8 | +0.125 | +0.275 | (secondary) |
| v1 refetch_position_effect | −0.03 | −0.146 | p-66c7e90a |

All signs survive scale. Two mechanistic refinements:
1. **Recovery gains GROW with scale** — truncation's loss is positional/
   information-theoretic (scale can't fix absent lines: keep_last_k:8 goes
   0.175 → 0.20), while recovery's loss is selection-error (scale fixes it:
   ledger+refetch goes 0.725 → 0.983). On 7b, refetch at 26% budget (0.983)
   **beats full context** (0.763) — compaction + recovery > not compacting.
2. **The position effect flips sign with task algebra** (+0.12 recency task,
   −0.146 fold task, both |Δ| ≥ 0.10 on the same model): adjacent-to-query
   placement helps select-latest queries, hurts aggregate queries that need
   chronological order. Placement, like the operator, must match the algebra.

## Run
```
# v0 de-confounded depth panel (protocol p-6abf2f5e):
uv run --frozen python eval.py --n-items 16 \
  --policies full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace \
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000 --n-distractors 40

# v1 accumulative panel (protocol p-b1c6166f):
uv run --frozen python eval.py --task accumulate --n-items 16 --n-ops 3 \
  --policies full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace,ledger_accumulate \
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000 --n-distractors 40

# algebra-aware paired 3B panel (protocol p-f5a44444):
./scripts/run_algebra_refetch_3b.sh

# low-budget k=4 algebra-aware paired 3B panel (protocol p-b37160f8):
./scripts/run_refetch_k4_3b.sh
```
Requires local ollama with `qwen2.5:3b-instruct`. Metrics →
`$AAD_METRICS_PATH`.
