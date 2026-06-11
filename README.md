# CompactionBench (bet-0003)

Measuring **context rot** and **recoverable compaction** on cheap local models.
Two task families: **v0 StatefulQA** (answer = most-recent value) and **v1
AccumulatorQA** (answer = cumulative sum). v1 exists because a most-recent-wins
ledger dedup near-solves v0 (1.000); v1 makes the answer a total no single line
holds, so the benchmark discriminates compaction operators by the **state algebra**
they respect. Verified results across two models (3b protocols p-d3c4bf50,
p-d816ff49, p-8e96fc78, p-9fd9858e; 7b cross-model protocols p-3ad438f7,
p-b7a62692, p-8a283782, p-6a21adb3).

**Start here:** [`FINDINGS.md`](FINDINGS.md) — paper-style write-up of the four
mechanistic findings. **Reproduce everything:** `./reproduce.sh` (both models;
`./reproduce.sh 3b` for the cheap half). **Frozen splits:**
`frozen/splits.json` pins SHA256 of every eval cell (the benchmark is
procedurally generated, so generator+seed+config IS the split);
`scripts/freeze_splits.py --check` detects any drift.

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

## v1 headline (VERIFIED, protocol p-9fd9858e, qwen2.5:3b, depth-balanced)
| policy | acc | budgetfrac | note |
|---|---|---|---|
| keep_last_k:8 (incumbent) | 0.067 | 0.197 | truncation |
| **ledger_state (select-latest)** | **0.004** | 0.152 | v0 near-solver — DEAD on v1 |
| ledger+refetch (recover all ops) | 0.192 | 0.223 | conservative recovery, beats truncation |
| drop_distractors (cheating ideal) | 0.542 | 0.071 | model-does-math ceiling |
| **ledger_accumulate (fold)** | **1.000** | 0.134 | correct operator |

`accum_fold_minus_dedup = +0.9958` (verified): two honest typed ledgers, identical
parsing, fold vs select-latest — on accumulative state fold is correct and
select-latest (the v0 near-solver) collapses. **The compaction operator must match
the state's algebra.** budgetfrac = mean prompt-chars / full-context prompt-chars.

## Cross-model robustness (VERIFIED, qwen2.5:7b-instruct, same panels, n_ops=3)

| metric | 3b | 7b | protocol (7b) |
|---|---|---|---|
| v0 recoverable_gain_refetch_8 | +0.55 | **+0.78** | p-3ad438f7 |
| v0 refetch_position_effect | +0.30 | **+0.12** | p-b7a62692 |
| v1 accum_fold_minus_dedup | +0.996 | **+0.992** | p-8a283782 |
| v1 accum_recoverable_gain_refetch_8 | +0.125 | +0.275 | (secondary) |
| v1 refetch_position_effect | −0.03 | −0.146 | p-6a21adb3 |

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
# v0 de-confounded depth panel (protocol p-d816ff49 / p-8e96fc78):
uv run --frozen python eval.py --n-items 16 \
  --policies full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace \
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000 --n-distractors 40

# v1 accumulative panel (protocol p-9fd9858e):
uv run --frozen python eval.py --task accumulate --n-items 16 --n-ops 3 \
  --policies full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace,ledger_accumulate \
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000 --n-distractors 40
```
Cross-model (protocols p-3ad438f7 / p-b7a62692 / p-8a283782): same two commands
with `--model qwen2.5:7b-instruct`. Requires local ollama with
`qwen2.5:3b-instruct` (and `qwen2.5:7b-instruct` for the cross-model panels).
Metrics → `$AAD_METRICS_PATH`.
