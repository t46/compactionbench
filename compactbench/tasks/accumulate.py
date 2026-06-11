"""AccumulatorQA: a long-horizon ACCUMULATIVE state-tracking task (CompactionBench v1).

Why this exists. v0 (StatefulQA) answers are the MOST-RECENT assignment to a
register, so a most-recent-wins ledger dedup (`ledger_state`) is a near-complete
solver (it scored 1.000) — the benchmark stops discriminating compaction quality.
v1 fixes that: the answer is a CUMULATIVE total that NO SINGLE LINE contains. A
target register is `set` to a base, then receives a sequence of `increased by` /
`decreased by` deltas amid distractor ops to other registers and filler. The
current value = base + sum(signed deltas). Exact-match ground truth.

Consequences for policies (the point of the family):
  * most-recent-wins dedup (`ledger_state`) now FAILS: keeping only the latest
    line per register discards every earlier delta -> wrong total.
  * recency truncation (`keep_last_k`) fails whenever the relevant span spills
    out of the window (depth-controlled, as in v0) -> a fair, beatable incumbent.
  * honest recoverable compaction must recover ALL target lines (lazy re-fetch),
    not just one -> a real test of recovery completeness.
  * the correct stateful-memory analog is FOLD-the-deltas (`ledger_accumulate`),
    which is the ceiling but computes the answer in-compactor (reported as such).

Arithmetic is kept small (base ~2 digits, deltas single digit) so the BOTTLENECK
is finding the relevant lines under distraction (context rot), not the addition.
Segment kinds mirror StatefulQA so the type-ablation baselines are unchanged.
"""

from __future__ import annotations

import random

from .stateful import (
    DISTRACTOR,
    FILLER_TEMPLATES,
    INSTRUCTION,
    QUERY,
    RELEVANT,
    VAR_NAMES,
    Item,
    Segment,
)

# Instruction text specialised for accumulative state (used only by the
# verbose_instruction wording ablation; compaction policies use TERSE_SYSTEM).
INSTRUCTION_TEXT = (
    "You are tracking the running value of named registers across a session log. "
    "A line may SET a register to a value, or INCREASE / DECREASE it by an amount "
    "(e.g. 'register X is set to 20', 'register X increased by 5'). The current "
    "value is the base it was last set to plus every later increase minus every "
    "later decrease. When asked for a register's value, reply with ONLY that "
    "number and nothing else."
)


def _base(rng: random.Random) -> int:
    return rng.randint(10, 40)


def _delta(rng: random.Random) -> int:
    return rng.randint(1, 9)


def _op_line(rng: random.Random, reg: str, is_first: bool) -> tuple[str, int]:
    """Return (line_text, signed_contribution). The first op for a register is a
    `set` (establishes the base); later ops are increase/decrease deltas."""
    if is_first:
        v = _base(rng)
        return f"register {reg} is set to {v}", v
    d = _delta(rng)
    if rng.random() < 0.5:
        return f"register {reg} increased by {d}", d
    return f"register {reg} decreased by {d}", -d


def make_item(
    seed: int,
    n_ops: int = 5,
    n_distractors: int = 40,
    n_distractor_vars: int = 8,
    filler_ratio: float = 0.5,
    needle_depth: float | None = None,
) -> Item:
    """Generate one AccumulatorQA item.

    n_ops: number of target-register ops (1 `set` + n_ops-1 deltas). The answer
        requires ALL of them, so completeness — not recency — is the hard part.
    n_distractors: number of distractor turns (sets context length).
    n_distractor_vars: pool of non-target registers for distractor ops.
    filler_ratio: fraction of distractor turns that are filler vs ops to others.
    needle_depth: CONTROLLED position in [0,1] of the BASE `set` line. The deltas
        are placed at random positions strictly AFTER the set (so the set is
        always the earliest target op). depth=1 crams the whole relevant span
        near the end (recency truncation captures it -> wins); depth=0 spreads it
        from the front (truncation drops early deltas -> loses). A depth-balanced
        score makes truncation a fair incumbent, exactly as in v0. If None, the
        base is placed at the front and deltas spread across the whole log
        (legacy/uncontrolled).
    """
    rng = random.Random(seed)
    names = VAR_NAMES[:]
    rng.shuffle(names)
    target = names[0]
    distractor_names = names[1 : 1 + n_distractor_vars]

    # Target ops: first is the base `set`, the rest are signed deltas.
    target_segs: list[Segment] = []
    total = 0
    for i in range(n_ops):
        text, contrib = _op_line(rng, target, is_first=(i == 0))
        total += contrib
        # Mark the base set as the "answer anchor" for the answer_retained info
        # field; the true answer needs every target op, scored by exact match.
        target_segs.append(Segment(RELEVANT, text, is_answer=(i == 0)))
    answer = str(total)

    # Distractor turns: filler or ops to OTHER registers (never the target).
    distractors: list[Segment] = []
    for _ in range(n_distractors):
        if rng.random() < filler_ratio or not distractor_names:
            distractors.append(Segment(DISTRACTOR, rng.choice(FILLER_TEMPLATES)))
        else:
            dn = rng.choice(distractor_names)
            # is_first is irrelevant for distractors (not scored); use a delta or
            # set at random so distractor surface forms match target forms.
            text, _ = _op_line(rng, dn, is_first=(rng.random() < 0.3))
            distractors.append(Segment(DISTRACTOR, text))

    n_total = n_distractors + n_ops
    set_seg = target_segs[0]
    delta_segs = target_segs[1:]

    if needle_depth is None:
        # Legacy: base at front, deltas spread across the whole log.
        log = distractors[:]
        positions = sorted(rng.randint(0, len(log)) for _ in range(len(delta_segs)))
        for offset, (pos, seg) in enumerate(zip(positions, delta_segs)):
            log.insert(pos + offset, seg)
        log.insert(0, set_seg)
        set_index = 0
    else:
        d = min(max(needle_depth, 0.0), 1.0)
        set_index = round(d * (n_total - 1))
        # Need room for all deltas strictly after the set, plus a trailing line.
        set_index = min(set_index, n_total - len(delta_segs) - 2)
        set_index = max(set_index, 0)
        # Distinct positions for the deltas in (set_index, n_total).
        delta_positions = sorted(
            rng.sample(range(set_index + 1, n_total), len(delta_segs))
        )
        log = []
        di = 0  # distractor index
        ci = 0  # delta-position cursor
        for idx in range(n_total):
            if idx == set_index:
                log.append(set_seg)
            elif ci < len(delta_segs) and idx == delta_positions[ci]:
                log.append(delta_segs[ci])
                ci += 1
            else:
                log.append(distractors[di])
                di += 1

    segments = (
        [Segment(INSTRUCTION, INSTRUCTION_TEXT)]
        + log
        + [Segment(QUERY, f"What is the current value of register {target}?")]
    )
    return Item(
        segments=segments,
        target=target,
        answer=answer,
        meta={
            "seed": seed,
            "n_ops": n_ops,
            "n_distractors": n_distractors,
            "n_log_lines": len(log),
            "needle_depth": needle_depth,
            "set_index": set_index,
            "set_depth_frac": set_index / (len(log) - 1) if len(log) > 1 else 0.0,
        },
    )


def make_dataset(base_seed: int, n_items: int, **kwargs) -> list[Item]:
    return [make_item(base_seed + i, **kwargs) for i in range(n_items)]
