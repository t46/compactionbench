"""Freeze CompactionBench splits: hash every registered eval cell.

The benchmark is procedurally generated, so the (generator code, seed, config)
triple IS the split. This script materializes every cell used by the registered
protocols and writes canonical SHA256 hashes to frozen/splits.json, so any
future generator change that silently alters items is detectable, and external
users can confirm they are evaluating on the same data.

Cells frozen (both panels, model-independent):
  v0 StatefulQA   : n_items=16, n_distractors=40, n_reassign=4,
                    depths {0,0.25,0.5,0.75,1.0}, seeds {1000,2000,3000}
                    (protocols p-d3c4bf50, p-d816ff49, p-8e96fc78, p-3ad438f7,
                     p-b7a62692)
  v1 AccumulatorQA: n_items=16, n_distractors=40, n_ops=3, same depths/seeds
                    (protocols p-9fd9858e, p-8a283782, p-6a21adb3)

Usage:
  uv run --frozen python scripts/freeze_splits.py            # write frozen/splits.json
  uv run --frozen python scripts/freeze_splits.py --check    # verify against frozen file
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compactbench.tasks import accumulate as accumulate_task
from compactbench.tasks import stateful as stateful_task

SEEDS = [1000, 2000, 3000]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_ITEMS = 16
N_DISTRACTORS = 40

PANELS = {
    "v0_stateful": (stateful_task.make_dataset, {"n_reassign": 4}),
    "v1_accumulate": (accumulate_task.make_dataset, {"n_ops": 3}),
}


def cell_hash(items) -> str:
    payload = json.dumps(
        [dataclasses.asdict(it) for it in items],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze() -> dict:
    out: dict = {"n_items": N_ITEMS, "n_distractors": N_DISTRACTORS, "panels": {}}
    for name, (make_dataset, op_kwargs) in PANELS.items():
        cells = {}
        for sd in SEEDS:
            for z in DEPTHS:
                items = make_dataset(
                    base_seed=sd,
                    n_items=N_ITEMS,
                    n_distractors=N_DISTRACTORS,
                    needle_depth=z,
                    **op_kwargs,
                )
                cells[f"seed{sd}_z{z}"] = cell_hash(items)
        panel_hash = hashlib.sha256(
            json.dumps(cells, sort_keys=True).encode("utf-8")
        ).hexdigest()
        out["panels"][name] = {"op_kwargs": op_kwargs, "cells": cells, "panel_sha256": panel_hash}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify against frozen/splits.json")
    args = ap.parse_args()

    frozen_path = Path(__file__).resolve().parents[1] / "frozen" / "splits.json"
    current = freeze()

    if args.check:
        recorded = json.loads(frozen_path.read_text())
        if recorded == current:
            print("OK: generated splits match frozen/splits.json")
            return 0
        print("MISMATCH: generated splits differ from frozen/splits.json", file=sys.stderr)
        for name, panel in current["panels"].items():
            rec = recorded.get("panels", {}).get(name, {})
            if panel["panel_sha256"] != rec.get("panel_sha256"):
                print(f"  panel {name}: {rec.get('panel_sha256')} -> {panel['panel_sha256']}", file=sys.stderr)
        return 1

    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    print(f"wrote {frozen_path}")
    for name, panel in current["panels"].items():
        print(f"  {name}: {panel['panel_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
