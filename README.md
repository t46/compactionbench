# CompactionBench (bet-0003)

Measuring **context rot** and **recoverable compaction** on cheap local models.

## Task: StatefulQA
A long "session log" of typed segments (instruction / relevant state / distractors),
then a query. The answer is the **most recent** value assigned to a target register,
which is reassigned several times amid many distractor turns. Exact-match ground truth.
Two difficulty knobs: context length (`--n-distractors`) and confusability (`--n-reassign`).

Typed segments let an ablation **policy** drop context BY TYPE — the mechanistic
question of the bet (which context elements' loss degrades success).

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
  from the dropped tail (recoverable compaction; conservative — model still picks recency)

## Headline result (VERIFIED, protocol p-d816ff49, qwen2.5:3b, depth-balanced)
| policy | acc | budgetfrac |
|---|---|---|
| keep_last_k:8 (incumbent) | 0.175 | 0.214 |
| drop_distractors (cheating ideal) | 0.604 | 0.082 |
| **ledger+refetch** | **0.725** | **0.257** |
| **ledger_state** | **1.000** | **0.143** |

`recoverable_gain_refetch_8 = +0.55` (verified from clean checkout). budgetfrac =
mean prompt-chars / full-context prompt-chars (the ≤50% budget axis).

## Run
```
# de-confounded depth panel (the registered protocol):
uv run --frozen python eval.py --n-items 16 \
  --policies full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch \
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000 --n-distractors 40
```
Requires local ollama with `qwen2.5:3b-instruct`. Metrics → `$AAD_METRICS_PATH`.
