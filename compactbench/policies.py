"""Context policies: turn an Item's typed segments into chat messages.

A policy decides which context survives into the prompt. This is where
"compaction" lives: the baseline keeps everything; ablations drop a context
TYPE to measure its marginal value; recency/threshold baselines mimic the
lossy threshold-summarization that production harnesses use today.

Every policy returns (messages, kept_kinds) where messages is an
OpenAI-style [{role, content}] list. A generic fallback system prompt is used
when the task instruction is dropped, so the model is still told to answer —
isolating the value of the SPECIFIC instruction rather than of "any system
prompt at all".
"""

from __future__ import annotations

from .tasks.stateful import DISTRACTOR, INSTRUCTION, QUERY, RELEVANT, Item

GENERIC_SYSTEM = "Answer the question using the session log below. Reply with only the answer."


def _assemble(instruction_text: str | None, log_texts: list[str], query_text: str) -> list[dict]:
    system = instruction_text if instruction_text is not None else GENERIC_SYSTEM
    log_block = "\n".join(f"- {t}" for t in log_texts)
    user = f"Session log:\n{log_block}\n\n{query_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def apply_policy(item: Item, policy: str) -> tuple[list[dict], dict]:
    """Return (messages, info). `policy` is a name, optionally 'keep_last_k:K'."""
    instr = next((s.text for s in item.segments if s.kind == INSTRUCTION), None)
    query = next(s.text for s in item.segments if s.kind == QUERY)
    log = [s for s in item.segments if s.kind in (RELEVANT, DISTRACTOR)]

    keep_instr = instr
    if policy == "full":
        kept = log
    elif policy == "drop_distractors":  # lossless ideal compaction
        kept = [s for s in log if s.kind == RELEVANT]
    elif policy == "drop_relevant":  # ablate answer-bearing info (necessity check)
        kept = [s for s in log if s.kind == DISTRACTOR]
    elif policy == "drop_instruction":
        keep_instr = None
        kept = log
    elif policy.startswith("keep_last_k:"):
        k = int(policy.split(":", 1)[1])
        kept = log[-k:] if k > 0 else []
    else:
        raise ValueError(f"unknown policy: {policy}")

    messages = _assemble(keep_instr, [s.text for s in kept], query)
    info = {
        "policy": policy,
        "kept_lines": len(kept),
        "answer_retained": any(s.kind == RELEVANT and s.is_answer for s in kept),
        "instruction_retained": keep_instr is not None,
    }
    return messages, info
