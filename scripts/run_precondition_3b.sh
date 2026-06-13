#!/usr/bin/env bash
# Hard-precondition triage for bet-0008.
#
# Reproduces the required Qwen2.5-3B StatefulQA ordering:
#   full context vs naive recency truncation vs refetch-8 recoverable compaction.
# Metrics are written by eval.py to $AAD_METRICS_PATH for the AAD harness.

set -euo pipefail
cd "$(dirname "$0")/.."

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}"

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/freeze_splits.py --check

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python eval.py \
  --model qwen2.5:3b-instruct \
  --task stateful \
  --n-items 16 \
  --n-distractors 40 \
  --policies full,keep_last_k:8,ledger+refetch \
  --needle-depths 0,0.25,0.5,0.75,1.0 \
  --base-seeds 1000,2000,3000
