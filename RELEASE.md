# CompactionBench v0.1 Release Packet

CompactionBench is ready to publish as a small benchmark artifact: two task
families, frozen generated splits, eight independently verified results, and a
one-command reproduction script.

## Release Channel

Recommended first channel: a public Git repo plus a short preprint-style note.
The benchmark is synthetic and cheap enough for external users to rerun locally,
so the adoption target should be "clone, run, compare a compaction policy" rather
than a closed leaderboard.

## What To Publish

- `README.md` as the repo landing page.
- `FINDINGS.md` as the short paper-style note.
- `EXPECTED_RESULTS.json` plus `scripts/validate_results.py` as the reproducibility
  contract for external runs.
- `frozen/splits.json` and `scripts/freeze_splits.py --check` as the split
  stability contract.
- `reproduce.sh` as the executable entry point.

## Suggested Announcement

CompactionBench v0.1: a seed-frozen benchmark for context rot and recoverable
compaction in long agent logs.

Headline result: on qwen2.5:7b, a typed ledger plus lazy re-fetch reaches 0.983
accuracy at 26% context budget, beating both truncation (0.200) and full context
(0.763). The same benchmark shows that compaction operators are not algebra-free:
latest-value dedup near-solves select-latest state but collapses on accumulative
state, where a fold ledger is required.

Run:

```bash
./reproduce.sh 3b
```

Full cross-model reproduction:

```bash
./reproduce.sh
```

## Minimum External Acceptance Bar

An external policy comparison should report:

- task family: `stateful` or `accumulate`
- model and decoding settings
- accuracy and `budgetfrac`
- comparison against `keep_last_k:8`
- whether the policy merges state or deletes and re-fetches it

Do not compare wall-clock timings from local Macs; this artifact only makes
accuracy and context-budget claims.
