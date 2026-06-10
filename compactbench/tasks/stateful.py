"""StatefulQA: a controllable long-horizon state-tracking task with typed context.

An item is a "session log": a sequence of typed segments the agent must read,
then a query. The answer is the MOST RECENT value assigned to a target
variable, which is reassigned several times (so the model must find the latest
assignment) amid many distractor turns (filler / assignments to other
variables). This isolates the "tool-output / decision recall under distraction"
axis of context rot, with exact-match ground truth and two orthogonal difficulty
knobs: context length (n_distractors) and confusability (n_reassign).

Why typed segments: each segment carries a `kind` so an ablation policy can drop
context BY TYPE (instruction vs relevant-state vs distractor), which is exactly
the mechanistic question of the bet — which context elements' loss degrades
success.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Segment kinds.
INSTRUCTION = "instruction"
RELEVANT = "relevant"      # an assignment to the target variable
DISTRACTOR = "distractor"  # filler or assignment to a non-target variable
QUERY = "query"

VAR_NAMES = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]

FILLER_TEMPLATES = [
    "The team reviewed the logs but found nothing actionable in this period.",
    "A routine status check was performed; all systems nominal.",
    "Note: the meeting was rescheduled and no decisions were recorded here.",
    "The report was filed under miscellaneous and requires no follow-up.",
    "An automated job ran successfully with no parameters changed.",
    "Background: this section contains commentary unrelated to any register.",
    "The auditor skimmed the section and moved on without changes.",
    "A reminder was posted about the upcoming maintenance window.",
]

INSTRUCTION_TEXT = (
    "You are tracking the values of named registers across a session log. "
    "Each line may assign a register a value (e.g. 'register X is now 42'). "
    "A register may be assigned several times; only the MOST RECENT assignment "
    "is current. When asked for a register's value, reply with ONLY that number "
    "and nothing else."
)


@dataclass
class Segment:
    kind: str
    text: str
    # True for the single relevant segment whose value is the answer.
    is_answer: bool = False


@dataclass
class Item:
    segments: list[Segment]
    target: str
    answer: str
    meta: dict = field(default_factory=dict)


def _val(rng: random.Random) -> int:
    return rng.randint(100, 999)


def make_item(
    seed: int,
    n_reassign: int = 4,
    n_distractors: int = 40,
    n_distractor_vars: int = 8,
    filler_ratio: float = 0.5,
) -> Item:
    """Generate one StatefulQA item.

    n_reassign: how many times the target register is (re)assigned. The last one
        (in log order) is the answer; earlier ones are confusable decoys.
    n_distractors: number of distractor turns (sets context length).
    n_distractor_vars: pool of non-target registers used by distractor assigns.
    filler_ratio: fraction of distractor turns that are pure filler vs decoy
        assignments to other registers.
    """
    rng = random.Random(seed)
    names = VAR_NAMES[:]
    rng.shuffle(names)
    target = names[0]
    distractor_names = names[1 : 1 + n_distractor_vars]

    # Build the relevant assignments to the target; the LAST is the answer.
    target_vals = [_val(rng) for _ in range(n_reassign)]
    answer = str(target_vals[-1])
    relevant = [
        Segment(RELEVANT, f"register {target} is now {v}", is_answer=(i == n_reassign - 1))
        for i, v in enumerate(target_vals)
    ]

    # Build distractor turns.
    distractors: list[Segment] = []
    for _ in range(n_distractors):
        if rng.random() < filler_ratio or not distractor_names:
            distractors.append(Segment(DISTRACTOR, rng.choice(FILLER_TEMPLATES)))
        else:
            dn = rng.choice(distractor_names)
            distractors.append(Segment(DISTRACTOR, f"register {dn} is now {_val(rng)}"))

    # Interleave relevant assignments among distractors, preserving relevant
    # ORDER (so the answer-bearing one stays last among the relevant set), and
    # ensure the answer-bearing assignment is NOT the very last line overall
    # (otherwise pure recency without reading would solve it).
    log = distractors[:]
    # choose n_reassign insertion positions in [0, len(log)] in increasing order
    # (repeats allowed so this works even when distractors are few/zero).
    positions = sorted(rng.randint(0, len(log)) for _ in range(n_reassign))
    for offset, (pos, seg) in enumerate(zip(positions, relevant)):
        log.insert(pos + offset, seg)
    # If the answer-bearing segment ended up last, append one filler after it.
    if log[-1].is_answer:
        log.append(Segment(DISTRACTOR, rng.choice(FILLER_TEMPLATES)))

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
            "n_reassign": n_reassign,
            "n_distractors": n_distractors,
            "n_log_lines": len(log),
        },
    )


def make_dataset(
    base_seed: int,
    n_items: int,
    **kwargs,
) -> list[Item]:
    return [make_item(base_seed + i, **kwargs) for i in range(n_items)]
