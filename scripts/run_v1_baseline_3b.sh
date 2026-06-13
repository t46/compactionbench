#!/usr/bin/env bash
# Matched 3B AccumulatorQA (v1) baseline panel for bet-0008.
#
# Primary conservative metric:
#   accum_recoverable_gain_refetch_8 =
#     acc_ledger_refetch_balanced - acc_keep_last_k_8_balanced
#
# The panel also records the state-algebra contrast:
#   accum_fold_minus_dedup =
#     acc_ledger_accumulate_balanced - acc_ledger_state_balanced

set -euo pipefail
cd "$(dirname "$0")/.."

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}"

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/freeze_splits.py --check

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python eval.py \
  --model qwen2.5:3b-instruct \
  --task accumulate \
  --n-ops 3 \
  --n-items 16 \
  --n-distractors 40 \
  --policies full,keep_last_k:8,ledger+refetch,ledger+refetch_inplace,ledger_state,ledger_accumulate \
  --needle-depths 0,0.25,0.5,0.75,1.0 \
  --base-seeds 1000,2000,3000
