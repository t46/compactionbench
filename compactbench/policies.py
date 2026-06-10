"""Context policies: turn an Item's typed segments into chat messages.

A policy decides which context survives into the prompt. This is where
"compaction" lives: the baseline keeps everything; ablations drop a context
TYPE to measure its marginal value; recency/threshold baselines mimic the
lossy threshold-summarization that production harnesses use today.

Every policy returns (messages, info). To isolate COMPACTION quality from prompt
WORDING, all compaction policies use ONE fixed terse system instruction (session
1 found the 3B is highly sensitive to instruction phrasing — a verbose task
instruction depressed accuracy ~0.30 absolute, which would otherwise contaminate
every policy that carried it). The verbose-vs-terse wording effect is measured
separately by the explicit `verbose_instruction` policy.
"""

from __future__ import annotations

from .tasks.stateful import DISTRACTOR, INSTRUCTION, QUERY, RELEVANT, Item

# Fixed, terse, COMPLETE instruction used by every compaction policy. Terse but
# states the rule (most-recent-wins) so the task is well-posed; held constant so
# we measure compaction, not wording.
TERSE_SYSTEM = (
    "Track register values from the log. A register's current value is its most "
    "recent assignment. Answer with only the number."
)


def _assemble(system: str, log_texts: list[str], query_text: str) -> list[dict]:
    log_block = "\n".join(f"- {t}" for t in log_texts)
    user = f"Session log:\n{log_block}\n\n{query_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def apply_policy(item: Item, policy: str) -> tuple[list[dict], dict]:
    """Return (messages, info). `policy` is a name, optionally 'keep_last_k:K'."""
    verbose = next((s.text for s in item.segments if s.kind == INSTRUCTION), None)
    query = next(s.text for s in item.segments if s.kind == QUERY)
    log = [s for s in item.segments if s.kind in (RELEVANT, DISTRACTOR)]

    system = TERSE_SYSTEM
    if policy == "full":
        kept = log
    elif policy == "drop_distractors":  # lossless ideal compaction
        kept = [s for s in log if s.kind == RELEVANT]
    elif policy == "drop_relevant":  # ablate answer-bearing info (necessity check)
        kept = [s for s in log if s.kind == DISTRACTOR]
    elif policy == "verbose_instruction":  # wording ablation: full log, verbose system
        kept = log
        system = verbose if verbose is not None else TERSE_SYSTEM
    elif policy.startswith("keep_last_k:"):
        k = int(policy.split(":", 1)[1])
        kept = log[-k:] if k > 0 else []
    else:
        raise ValueError(f"unknown policy: {policy}")

    messages = _assemble(system, [s.text for s in kept], query)
    info = {
        "policy": policy,
        "kept_lines": len(kept),
        "answer_retained": any(s.kind == RELEVANT and s.is_answer for s in kept),
        "system_is_verbose": system != TERSE_SYSTEM,
    }
    return messages, info
