#!/usr/bin/env bash
# Full paired 3B panel for low-budget algebra-aware lazy refetch at k=4.
#
# Primary metric:
#   k4_algebra_refetch_mean_gain_vs_trunc =
#     mean over v0/v1 of (acc_ledger_refetch_algebra_4 - acc_keep_last_k_4)

set -euo pipefail
cd "$(dirname "$0")/.."

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}"
OUT="${AAD_METRICS_PATH:?AAD_METRICS_PATH must be set by aad exp run}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

POLICIES="full,keep_last_k:4,ledger+refetch_algebra:4"

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/freeze_splits.py --check

env UV_CACHE_DIR="$UV_CACHE_DIR" AAD_METRICS_PATH="$TMP_DIR/stateful.json" \
  uv run --frozen python eval.py \
    --model qwen2.5:3b-instruct \
    --task stateful \
    --n-items 16 \
    --n-distractors 40 \
    --policies "$POLICIES" \
    --needle-depths 0,0.25,0.5,0.75,1.0 \
    --base-seeds 1000,2000,3000

env UV_CACHE_DIR="$UV_CACHE_DIR" AAD_METRICS_PATH="$TMP_DIR/accumulate.json" \
  uv run --frozen python eval.py \
    --model qwen2.5:3b-instruct \
    --task accumulate \
    --n-ops 3 \
    --n-items 16 \
    --n-distractors 40 \
    --policies "$POLICIES" \
    --needle-depths 0,0.25,0.5,0.75,1.0 \
    --base-seeds 1000,2000,3000

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/combine_refetch_k_sweep.py \
  --v0 "$TMP_DIR/stateful.json" \
  --v1 "$TMP_DIR/accumulate.json" \
  --ks "4" \
  --out "$OUT"
