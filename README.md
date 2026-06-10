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
- `full` — keep everything (baseline)
- `drop_distractors` — keep only answer-bearing state (ideal lossless compaction)
- `drop_relevant` — drop answer-bearing state (necessity check; should floor)
- `drop_instruction` — drop the task instruction
- `keep_last_k:K` — recency truncation (threshold-summarization analog)

## Run
```
uv run --frozen python eval.py --n-items 50 --policies full,drop_distractors,drop_relevant
# sweep context length:
uv run --frozen python eval.py --n-items 40 --sweep-distractors 10,40,100
```
Requires local ollama with `qwen2.5:3b-instruct`. Metrics → `$AAD_METRICS_PATH`.
