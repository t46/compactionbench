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

ORACLE_POLICIES = frozenset({"drop_distractors", "drop_relevant"})
LEADERBOARD_POLICIES = frozenset(
    {
        "full",
        "ledger",
        "ledger_state",
        "ledger+refetch",
        "ledger+refetch_inplace",
        "ledger+refetch_algebra",
        "ledger_accumulate",
        "verbose_instruction",
    }
)

# Fixed, terse, COMPLETE instructions used by every compaction policy. Terse but
# task-correct; held constant within a task so we measure compaction, not
# wording.
TERSE_STATEFUL_SYSTEM = (
    "Track register values from the log. A register's current value is its most "
    "recent assignment. Answer with only the number."
)
TERSE_ACCUMULATE_SYSTEM = (
    "Track register values from the log. A register's current value is its last "
    "set value plus later increases minus later decreases. Answer with only the "
    "number."
)

# Surface-form matcher for an assignment line ("register X is now 42"). The
# recoverable-compaction policies use ONLY this text pattern and the query text
# to decide what to keep — never the ground-truth segment `kind`/`is_answer`.
# That is the honest line: drop_distractors cheats by reading `kind`; a real
# compactor must classify the surface text itself, which is what this does.
ASSIGN_RE = re.compile(r"register\s+(\w+)\s+is\s+now\s+(-?\d+)")

# AccumulatorQA (v1) surface grammar: a register is SET to a base, or INCREASED /
# DECREASED by a delta. Honest accumulate policies use ONLY this pattern and the
# query text — never the ground-truth `kind`. `set` lines carry a base value; the
# current value of a register = its last base + every later signed delta.
OP_RE = re.compile(
    r"register\s+(\w+)\s+(?:is\s+set\s+to|(increased)|(decreased))\s+(?:by\s+)?(-?\d+)"
)

# Any line that MENTIONS a register (the honest re-fetch key: pull every line
# referencing the queried target, regardless of op form). Matches both the v1 op
# grammar and the v0 "is now" grammar via the shared "register <name>" prefix.
REG_MENTION_RE = re.compile(r"register\s+(\w+)\b")


def _parse_op(text: str) -> tuple[str, int] | None:
    """Parse an AccumulatorQA op line into (register, signed_contribution).
    `set` -> (+base); `increased by d` -> (+d); `decreased by d` -> (-d)."""
    m = OP_RE.search(text)
    if not m:
        return None
    reg, inc, dec, num = m.group(1), m.group(2), m.group(3), int(m.group(4))
    if dec:
        return reg, -num
    return reg, num  # set or increase both contribute positively


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


def _split_refetch_policy(policy: str) -> tuple[str, int] | None:
    """Parse ledger+refetch policy names, with optional ':K' budget suffix."""
    for base in ("ledger+refetch_algebra", "ledger+refetch_inplace", "ledger+refetch"):
        if policy == base:
            return base, 8
        prefix = f"{base}:"
        if policy.startswith(prefix):
            return base, int(policy.split(":", 1)[1])
    return None


def apply_policy(item: Item, policy: str, task: str = "stateful") -> tuple[list[dict], dict]:
    """Return (messages, info). `policy` is a name, optionally 'keep_last_k:K'.

    `task` selects the surface grammar the HONEST recoverable-compaction policies
    parse: "stateful" (v0, "register X is now N", answer = most-recent) or
    "accumulate" (v1, set/increase/decrease, answer = base + sum of deltas). The
    type-ablation baselines (full / drop_distractors / drop_relevant /
    keep_last_k / verbose_instruction) are grammar-independent and behave
    identically for both tasks.
    """
    verbose = next((s.text for s in item.segments if s.kind == INSTRUCTION), None)
    query = next(s.text for s in item.segments if s.kind == QUERY)
    log = [s for s in item.segments if s.kind in (RELEVANT, DISTRACTOR)]

    system = TERSE_ACCUMULATE_SYSTEM if task == "accumulate" else TERSE_STATEFUL_SYSTEM
    if policy == "full":
        kept = log
    elif policy == "drop_distractors":  # lossless ideal compaction
        kept = [s for s in log if s.kind == RELEVANT]
    elif policy == "drop_relevant":  # ablate answer-bearing info (necessity check)
        kept = [s for s in log if s.kind == DISTRACTOR]
    elif policy == "verbose_instruction":  # wording ablation: full log, verbose system
        kept = log
        system = verbose if verbose is not None else system
    elif policy.startswith("keep_last_k:"):
        k = int(policy.split(":", 1)[1])
        kept = log[-k:] if k > 0 else []
    elif policy == "ledger":
        # HONEST analog of drop_distractors: keep lines that LOOK like register
        # operations (surface regex), drop pure filler. Does NOT read `kind`, so
        # it cannot tell target-ops from decoy-register ops — it keeps both. An
        # achievable approximation of ideal type-based compaction. The op grammar
        # differs by task (assignments vs set/inc/dec).
        op_re = OP_RE if task == "accumulate" else ASSIGN_RE
        kept = [s for s in log if op_re.search(s.text)]
    elif policy == "ledger_accumulate":
        # The accumulate-task TYPED LEDGER OF FACTS (v1 analog of ledger_state):
        # fold every op per register — `set` resets the base, increase/decrease
        # adjust it — and emit one synthesized current-value line per register.
        # Correct stateful-memory operator for ACCUMULATIVE state; it computes the
        # total in-compactor, so report it as a CEILING, not the conservative
        # headline (refetch leaves the arithmetic to the model). Honest: parses
        # only surface op text, never `kind`.
        totals: dict[str, int] = {}
        order: list[str] = []
        for s in log:
            parsed = _parse_op(s.text)
            if parsed is None:
                continue
            reg, contrib = parsed
            if reg not in totals:
                order.append(reg)
                totals[reg] = 0
            # A `set` line re-bases; for simplicity (and since AccumulatorQA emits
            # exactly one set per register, first) we additively fold — set's +base
            # plus subsequent signed deltas yields the current value.
            totals[reg] += contrib
        kept_texts = [f"register {reg} is now {totals[reg]}" for reg in order]
        messages = _assemble(system, kept_texts, query)
        info = _info(policy, kept_texts, messages, system)
        return messages, info
    elif policy == "ledger_state":
        # MOST-RECENT-WINS dedup: keep only the latest line per register. On v0
        # (assignments) this is the correct current value and near-solves the
        # task; on v1 (accumulate) it is a DELIBERATE failure demonstration —
        # keeping only the last op per register discards every earlier delta, so
        # the synthesized "current value" is wrong. Reporting it on v1 shows the
        # benchmark now discriminates: dedup is no longer a free solver.
        if task == "accumulate":
            last_line: dict[str, str] = {}
            order: list[str] = []
            for s in log:
                m = REG_MENTION_RE.search(s.text)
                if not m or _parse_op(s.text) is None:
                    continue
                reg = m.group(1)
                if reg not in last_line:
                    order.append(reg)
                last_line[reg] = s.text  # later op overwrites = keeps only latest
            kept_texts = [last_line[reg] for reg in order]
        else:
            latest: dict[str, str] = {}
            order = []
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
    elif refetch_cfg := _split_refetch_policy(policy):
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
        #   * ledger+refetch_algebra -> task-aware wrapper: append for v0
        #     select-latest state, chronological in-place for v1 fold state.
        #     This is a named deployable policy family over the two benchmark
        #     state algebras, while preserving the primitive policies for
        #     ablation.
        refetch_policy, k = refetch_cfg
        window = log[-k:] if k > 0 else []
        target_name = _target_of(query)
        dropped = log[:-k] if k > 0 else log
        refetched: list = []
        if task == "accumulate":
            # COMPLETENESS recovery: the answer needs EVERY target op, so always
            # pull all dropped lines mentioning the target (the window holds only
            # the ops that happen to fall in the recency tail). window and dropped
            # are disjoint, so each target op is kept exactly once -> no double
            # count. Honest: match on the queried register name only.
            if target_name is not None:
                refetched = [
                    s
                    for s in dropped
                    if (m := REG_MENTION_RE.search(s.text)) and m.group(1) == target_name
                ]
        else:
            # v0: one assignment = the answer; re-fetch only if the window lacks
            # any target assignment (frozen behavior).
            in_window = any(
                (m := ASSIGN_RE.search(s.text)) and m.group(1) == target_name
                for s in window
            )
            if target_name is not None and not in_window:
                refetched = [
                    s
                    for s in dropped
                    if (m := ASSIGN_RE.search(s.text)) and m.group(1) == target_name
                ]
        if refetch_policy == "ledger+refetch_inplace" or (
            refetch_policy == "ledger+refetch_algebra" and task == "accumulate"
        ):
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
        "system_is_verbose": system not in (TERSE_STATEFUL_SYSTEM, TERSE_ACCUMULATE_SYSTEM),
    }
