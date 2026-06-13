#!/usr/bin/env bash
# Small paired 3B smoke sweep for algebra-aware lazy-refetch budgets.
#
# Primary metric:
#   sweep_best_mean_gain =
#     best over k in {4,6,8,10,12} of mean(v0/v1 algebra-refetch gain vs
#     keep_last_k at the same k).

set -euo pipefail
cd "$(dirname "$0")/.."

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}"
OUT="${AAD_METRICS_PATH:?AAD_METRICS_PATH must be set by aad exp run}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

KS="4,6,8,10,12"
POLICIES="full,keep_last_k:4,keep_last_k:6,keep_last_k:8,keep_last_k:10,keep_last_k:12,ledger+refetch_algebra:4,ledger+refetch_algebra:6,ledger+refetch_algebra:8,ledger+refetch_algebra:10,ledger+refetch_algebra:12"

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/freeze_splits.py --check

env UV_CACHE_DIR="$UV_CACHE_DIR" AAD_METRICS_PATH="$TMP_DIR/stateful.json" \
  uv run --frozen python eval.py \
    --model qwen2.5:3b-instruct \
    --task stateful \
    --n-items 4 \
    --n-distractors 40 \
    --policies "$POLICIES" \
    --needle-depths 0,0.5,1.0 \
    --base-seeds 1000

env UV_CACHE_DIR="$UV_CACHE_DIR" AAD_METRICS_PATH="$TMP_DIR/accumulate.json" \
  uv run --frozen python eval.py \
    --model qwen2.5:3b-instruct \
    --task accumulate \
    --n-ops 3 \
    --n-items 4 \
    --n-distractors 40 \
    --policies "$POLICIES" \
    --needle-depths 0,0.5,1.0 \
    --base-seeds 1000

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/combine_refetch_k_sweep.py \
  --v0 "$TMP_DIR/stateful.json" \
  --v1 "$TMP_DIR/accumulate.json" \
  --ks "$KS" \
  --out "$OUT"
