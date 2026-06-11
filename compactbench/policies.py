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

import re

from .tasks.stateful import DISTRACTOR, INSTRUCTION, QUERY, RELEVANT, Item

# Fixed, terse, COMPLETE instruction used by every compaction policy. Terse but
# states the rule (most-recent-wins) so the task is well-posed; held constant so
# we measure compaction, not wording.
TERSE_SYSTEM = (
    "Track register values from the log. A register's current value is its most "
    "recent assignment. Answer with only the number."
)

# Surface-form matcher for an assignment line ("register X is now 42"). The
# recoverable-compaction policies use ONLY this text pattern and the query text
# to decide what to keep — never the ground-truth segment `kind`/`is_answer`.
# That is the honest line: drop_distractors cheats by reading `kind`; a real
# compactor must classify the surface text itself, which is what this does.
ASSIGN_RE = re.compile(r"register\s+(\w+)\s+is\s+now\s+(-?\d+)")


def _target_of(query_text: str) -> str | None:
    m = re.search(r"register\s+(\w+)", query_text)
    return m.group(1) if m else None


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
    elif policy == "ledger":
        # HONEST analog of drop_distractors: keep lines that LOOK like register
        # assignments (surface regex), drop pure filler. Does NOT read `kind`, so
        # it cannot tell target-assignments from decoy-register assignments — it
        # keeps both. An achievable approximation of ideal type-based compaction.
        kept = [s for s in log if ASSIGN_RE.search(s.text)]
    elif policy == "ledger_state":
        # The charter's TYPED LEDGER OF FACTS: parse every assignment line and
        # keep only the LATEST value per register (most-recent-wins applied
        # generically, exactly as a stateful memory would). Drops filler AND
        # superseded decoy assignments. Emits one synthesized line per register.
        latest: dict[str, str] = {}
        order: list[str] = []
        for s in log:
            m = ASSIGN_RE.search(s.text)
            if not m:
                continue
            reg, val = m.group(1), m.group(2)
            if reg not in latest:
                order.append(reg)
            latest[reg] = val  # later assignment overwrites = current value
        kept_texts = [f"register {reg} is now {latest[reg]}" for reg in order]
        messages = _assemble(system, kept_texts, query)
        info = _info(policy, kept_texts, messages, system)
        return messages, info
    elif policy in ("ledger+refetch", "ledger+refetch_inplace"):
        # keep_last_k recency window + LAZY RE-FETCH: if the queried target has
        # no assignment in the kept window, pull its assignment lines back from
        # the dropped tail (retrieval keyed on the query, in original order so
        # most-recent-wins is preserved). Measures recoverable compaction vs
        # plain truncation directly.
        #
        # Two re-insertion ORDERS, isolating the retrieval-POSITION effect
        # (anomaly bet-0003-refetch-exceeds-prediction-positional):
        #   * ledger+refetch         -> refetched lines APPENDED after the window,
        #     i.e. adjacent to the query (most-recent slot).
        #   * ledger+refetch_inplace -> refetched lines PREPENDED before the
        #     window, i.e. at their ORIGINAL relative position (they came from
        #     the dropped prefix log[:-k], which precedes the window). Same
        #     content, same budget; only position changes.
        k = 8
        window = log[-k:] if k > 0 else []
        target_name = _target_of(query)
        in_window = any(
            (m := ASSIGN_RE.search(s.text)) and m.group(1) == target_name
            for s in window
        )
        refetched: list = []
        if target_name is not None and not in_window:
            dropped = log[:-k] if k > 0 else log
            refetched = [
                s
                for s in dropped
                if (m := ASSIGN_RE.search(s.text)) and m.group(1) == target_name
            ]
        if policy == "ledger+refetch_inplace":
            kept = refetched + list(window)
        else:
            kept = list(window) + refetched
    else:
        raise ValueError(f"unknown policy: {policy}")

    messages = _assemble(system, [s.text for s in kept], query)
    info = _info(policy, [s.text for s in kept], messages, system)
    info["answer_retained"] = any(s.kind == RELEVANT and s.is_answer for s in kept)
    return messages, info


def _info(policy: str, kept_texts: list[str], messages: list[dict], system: str) -> dict:
    # Budget axis: lines kept and the user-message char count (deterministic,
    # backend-independent — no M1-banned wall-clock). Lets every "win" be read
    # against how much context it actually keeps.
    user_chars = sum(len(m["content"]) for m in messages if m["role"] == "user")
    return {
        "policy": policy,
        "kept_lines": len(kept_texts),
        "prompt_chars": user_chars,
        "system_is_verbose": system != TERSE_SYSTEM,
    }
