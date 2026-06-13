#!/usr/bin/env bash
# Paired 3B panel for the algebra-aware lazy-refetch policy family.
#
# ledger+refetch_algebra uses appended refetch for StatefulQA (v0,
# select-latest state) and chronological in-place refetch for AccumulatorQA
# (v1, fold state). Primary metric:
#   algebra_refetch_mean_gain_8 =
#     mean over v0/v1 of (acc_ledger_refetch_algebra - acc_keep_last_k_8)

set -euo pipefail
cd "$(dirname "$0")/.."

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}"
OUT="${AAD_METRICS_PATH:?AAD_METRICS_PATH must be set by aad exp run}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/freeze_splits.py --check

env UV_CACHE_DIR="$UV_CACHE_DIR" AAD_METRICS_PATH="$TMP_DIR/stateful.json" \
  uv run --frozen python eval.py \
    --model qwen2.5:3b-instruct \
    --task stateful \
    --n-items 16 \
    --n-distractors 40 \
    --policies full,keep_last_k:8,ledger+refetch,ledger+refetch_inplace,ledger+refetch_algebra \
    --needle-depths 0,0.25,0.5,0.75,1.0 \
    --base-seeds 1000,2000,3000

env UV_CACHE_DIR="$UV_CACHE_DIR" AAD_METRICS_PATH="$TMP_DIR/accumulate.json" \
  uv run --frozen python eval.py \
    --model qwen2.5:3b-instruct \
    --task accumulate \
    --n-ops 3 \
    --n-items 16 \
    --n-distractors 40 \
    --policies full,keep_last_k:8,ledger+refetch,ledger+refetch_inplace,ledger+refetch_algebra \
    --needle-depths 0,0.25,0.5,0.75,1.0 \
    --base-seeds 1000,2000,3000

env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/combine_algebra_panel.py \
  --v0 "$TMP_DIR/stateful.json" \
  --v1 "$TMP_DIR/accumulate.json" \
  --out "$OUT"
