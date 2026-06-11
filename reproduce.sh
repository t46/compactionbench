#!/usr/bin/env bash
# CompactionBench — one-command reproduction of all verified results.
#
# Requirements: uv (https://docs.astral.sh/uv/), local ollama serving
#   qwen2.5:3b-instruct and qwen2.5:7b-instruct (`ollama pull <model>`).
# All runs are greedy + seed-pinned: every number is bit-deterministic for a
# given model binary/backend. Wall-clock on an M-class Mac: ~10 min (3b panels)
# + ~36 min (7b panels).
#
# Usage:
#   ./reproduce.sh            # both models, all four panels
#   ./reproduce.sh 3b         # only the qwen2.5:3b panels
#   ./reproduce.sh 7b         # only the qwen2.5:7b panels
#
# Per-panel metrics land in results/<panel>.json. Expected headline values
# (depth-balanced) are listed in FINDINGS.md; key ones echoed at the end.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

WHICH="${1:-all}"

echo "== freezing check: generated splits must match frozen/splits.json =="
uv run --frozen python scripts/freeze_splits.py --check

V0_ARGS=(--task stateful --n-items 16 --n-distractors 40
  --policies "full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)
V1_ARGS=(--task accumulate --n-items 16 --n-ops 3 --n-distractors 40
  --policies "full,drop_distractors,keep_last_k:8,keep_last_k:16,ledger,ledger_state,ledger+refetch,ledger+refetch_inplace,ledger_accumulate"
  --needle-depths 0,0.25,0.5,0.75,1.0 --base-seeds 1000,2000,3000)

run_panel() { # name model args...
  local name="$1" model="$2"; shift 2
  echo "== panel: $name (model $model) =="
  AAD_METRICS_PATH="results/${name}.json" \
    env UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/aad-uv-cache}" \
    uv run --frozen python eval.py --model "$model" "$@"
}

if [[ "$WHICH" == "all" || "$WHICH" == "3b" ]]; then
  run_panel v0_3b qwen2.5:3b-instruct "${V0_ARGS[@]}"   # protocols p-d816ff49, p-8e96fc78
  run_panel v1_3b qwen2.5:3b-instruct "${V1_ARGS[@]}"   # protocol  p-9fd9858e
fi
if [[ "$WHICH" == "all" || "$WHICH" == "7b" ]]; then
  run_panel v0_7b qwen2.5:7b-instruct "${V0_ARGS[@]}"   # protocols p-3ad438f7, p-b7a62692
  run_panel v1_7b qwen2.5:7b-instruct "${V1_ARGS[@]}"   # protocols p-8a283782, p-66c7e90a
fi

echo "== headline metrics =="
uv run --frozen python - <<'PY'
import json, pathlib
KEYS = {
    "v0_3b": ["recoverable_gain_refetch_8", "refetch_position_effect"],
    "v1_3b": ["accum_fold_minus_dedup"],
    "v0_7b": ["recoverable_gain_refetch_8", "refetch_position_effect"],
    "v1_7b": ["accum_fold_minus_dedup", "refetch_position_effect",
              "accum_recoverable_gain_refetch_8"],
}
EXPECTED = {
    ("v0_3b", "recoverable_gain_refetch_8"): 0.55,
    ("v0_3b", "refetch_position_effect"): 0.30,
    ("v1_3b", "accum_fold_minus_dedup"): 0.9958,
    ("v0_7b", "recoverable_gain_refetch_8"): 0.7833,
    ("v0_7b", "refetch_position_effect"): 0.1167,
    ("v1_7b", "accum_fold_minus_dedup"): 0.9917,
    ("v1_7b", "refetch_position_effect"): -0.1458,
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
