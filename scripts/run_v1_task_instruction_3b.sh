#!/usr/bin/env bash
# Instruction-corrected 3B AccumulatorQA (v1) panel for bet-0008.
#
# This is matched to run_v1_baseline_3b.sh, but after compaction policies use
# the accumulate task's terse system instruction instead of the v0 most-recent
# instruction. Primary metric:
#   recoverable_gain_refetch_inplace_8 =
#     acc_ledger_refetch_inplace_balanced - acc_keep_last_k_8_balanced

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
