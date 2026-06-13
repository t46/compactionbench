#!/usr/bin/env bash
# CompactionBench — one-command reproduction of current 3B verified results.
#
# Requirements: uv (https://docs.astral.sh/uv/), local ollama serving
#   qwen2.5:3b-instruct (`ollama pull qwen2.5:3b-instruct`).
# All runs are greedy + seed-pinned. Verification used tolerance-aware replay
# because local model binaries/backends can differ slightly.
#
# Usage:
#   ./reproduce.sh            # qwen2.5:3b current panels
#   ./reproduce.sh 3b         # same as above
#
# Per-panel metrics land in results/<panel>.json. Expected verified values are
# listed in EXPECTED_RESULTS.json and checked at the end.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}"

WHICH="${1:-3b}"
if [[ "$WHICH" != "3b" && "$WHICH" != "all" ]]; then
  echo "usage: ./reproduce.sh [3b]" >&2
  exit 2
fi

echo "== freezing check: generated splits must match frozen/splits.json =="
env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/freeze_splits.py --check

V0_ARGS=(--task stateful --n-items 16 --n-distractors 40
  --policies "full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)
V1_ARGS=(--task accumulate --n-items 16 --n-ops 3 --n-distractors 40
  --policies "full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace,ledger_accumulate"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)
ALGEBRA_V0_ARGS=(--task stateful --n-items 16 --n-distractors 40
  --policies "full,keep_last_k:8,ledger+refetch,ledger+refetch_inplace,ledger+refetch_algebra"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)
ALGEBRA_V1_ARGS=(--task accumulate --n-items 16 --n-ops 3 --n-distractors 40
  --policies "full,keep_last_k:8,ledger+refetch,ledger+refetch_inplace,ledger+refetch_algebra"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)
K4_V0_ARGS=(--task stateful --n-items 16 --n-distractors 40
  --policies "full,keep_last_k:4,ledger+refetch_algebra:4"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)
K4_V1_ARGS=(--task accumulate --n-items 16 --n-ops 3 --n-distractors 40
  --policies "full,keep_last_k:4,ledger+refetch_algebra:4"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)

run_panel() { # name model args...
  local name="$1" model="$2"; shift 2
  echo "== panel: $name (model $model) =="
  AAD_METRICS_PATH="results/${name}.json" \
    env UV_CACHE_DIR="$UV_CACHE_DIR" \
    uv run --frozen python eval.py --model "$model" "$@"
}

if [[ "$WHICH" == "all" || "$WHICH" == "3b" ]]; then
  run_panel v0_3b qwen2.5:3b-instruct "${V0_ARGS[@]}"   # protocol p-6abf2f5e
  run_panel v1_3b qwen2.5:3b-instruct "${V1_ARGS[@]}"   # protocol p-b1c6166f
  run_panel algebra_v0_3b qwen2.5:3b-instruct "${ALGEBRA_V0_ARGS[@]}"
  run_panel algebra_v1_3b qwen2.5:3b-instruct "${ALGEBRA_V1_ARGS[@]}"
  env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/combine_algebra_panel.py \
    --v0 results/algebra_v0_3b.json \
    --v1 results/algebra_v1_3b.json \
    --out results/algebra_refetch_3b.json
  run_panel refetch_k4_v0_3b qwen2.5:3b-instruct "${K4_V0_ARGS[@]}"
  run_panel refetch_k4_v1_3b qwen2.5:3b-instruct "${K4_V1_ARGS[@]}"
  env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/combine_refetch_k_sweep.py \
    --v0 results/refetch_k4_v0_3b.json \
    --v1 results/refetch_k4_v1_3b.json \
    --ks "4" \
    --out results/refetch_k4_3b.json
fi

echo "== headline metrics =="
env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python - <<'PY'
import json, pathlib
KEYS = {
    "v0_3b": ["recoverable_gain_refetch_8"],
    "v1_3b": ["recoverable_gain_refetch_inplace_8", "accum_fold_minus_dedup"],
    "algebra_refetch_3b": ["algebra_refetch_mean_gain_8", "algebra_refetch_min_gain_8",
                           "algebra_refetch_budgetfrac_mean", "stateful_refetch_position_effect"],
    "refetch_k4_3b": ["k4_algebra_refetch_mean_gain_vs_trunc",
                      "k4_algebra_refetch_min_gain_vs_trunc",
                      "k4_algebra_refetch_budgetfrac_mean"],
}
EXPECTED = {
    ("v0_3b", "recoverable_gain_refetch_8"): 0.55,
    ("v1_3b", "recoverable_gain_refetch_inplace_8"): 0.2458,
    ("v1_3b", "accum_fold_minus_dedup"): 0.9458,
    ("algebra_refetch_3b", "algebra_refetch_mean_gain_8"): 0.3917,
    ("algebra_refetch_3b", "algebra_refetch_min_gain_8"): 0.2333,
    ("algebra_refetch_3b", "algebra_refetch_budgetfrac_mean"): 0.2402,
    ("algebra_refetch_3b", "stateful_refetch_position_effect"): 0.30,
    ("refetch_k4_3b", "k4_algebra_refetch_mean_gain_vs_trunc"): 0.4292,
    ("refetch_k4_3b", "k4_algebra_refetch_min_gain_vs_trunc"): 0.3083,
    ("refetch_k4_3b", "k4_algebra_refetch_budgetfrac_mean"): 0.1527,
}
for panel, keys in KEYS.items():
    p = pathlib.Path(f"results/{panel}.json")
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    for k in keys:
        exp = EXPECTED.get((panel, k))
        tail = f"  (verified: {exp})" if exp is not None else ""
        print(f"{panel:6s} {k:34s} {m.get(k)}{tail}")
PY

echo "== validating reproduced metrics =="
env UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen python scripts/validate_results.py --allow-missing
