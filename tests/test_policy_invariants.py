from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

from compactbench.policies import LEADERBOARD_POLICIES, ORACLE_POLICIES, apply_policy
from compactbench.tasks import accumulate as accumulate_task
from compactbench.tasks import stateful as stateful_task
from compactbench.tasks.stateful import DISTRACTOR, INSTRUCTION, QUERY, RELEVANT, Item, Segment


def _relabel_log_kinds(item: Item) -> Item:
    relabeled: list[Segment] = []
    for segment in item.segments:
        if segment.kind == RELEVANT:
            relabeled.append(replace(segment, kind=DISTRACTOR))
        elif segment.kind == DISTRACTOR:
            relabeled.append(replace(segment, kind=RELEVANT))
        else:
            relabeled.append(segment)
    return replace(item, segments=relabeled)


def _budget_info(info: dict) -> dict:
    return {
        "kept_lines": info["kept_lines"],
        "prompt_chars": info["prompt_chars"],
        "system_is_verbose": info["system_is_verbose"],
    }


def _assert_kind_blind(item: Item, policy: str, task: str) -> None:
    relabeled = _relabel_log_kinds(item)
    messages_a, info_a = apply_policy(item, policy, task=task)
    messages_b, info_b = apply_policy(relabeled, policy, task=task)

    assert messages_a == messages_b
    assert _budget_info(info_a) == _budget_info(info_b)


def test_frozen_splits_are_current() -> None:
    subprocess.run(
        [sys.executable, "scripts/freeze_splits.py", "--check"],
        check=True,
    )


def test_policy_metadata_marks_oracles_outside_leaderboard() -> None:
    assert ORACLE_POLICIES == {"drop_distractors", "drop_relevant"}
    assert ORACLE_POLICIES.isdisjoint(LEADERBOARD_POLICIES)
    assert {"ledger", "ledger_state", "ledger+refetch", "ledger_accumulate"}.issubset(
        LEADERBOARD_POLICIES
    )


def test_honest_stateful_policies_are_blind_to_relevant_labels() -> None:
    item = stateful_task.make_item(
        seed=123,
        n_reassign=4,
        n_distractors=24,
        needle_depth=0.4,
    )
    for policy in (
        "full",
        "verbose_instruction",
        "keep_last_k:8",
        "ledger",
        "ledger_state",
        "ledger+refetch",
        "ledger+refetch_inplace",
    ):
        _assert_kind_blind(item, policy, task="stateful")


def test_honest_accumulate_policies_are_blind_to_relevant_labels() -> None:
    item = accumulate_task.make_item(
        seed=456,
        n_ops=5,
        n_distractors=24,
        needle_depth=0.25,
    )
    for policy in (
        "full",
        "keep_last_k:8",
        "ledger",
        "ledger_state",
        "ledger+refetch",
        "ledger+refetch_inplace",
        "ledger_accumulate",
    ):
        _assert_kind_blind(item, policy, task="accumulate")


def test_oracle_ablations_are_not_kind_blind() -> None:
    item = stateful_task.make_item(seed=789, n_reassign=4, n_distractors=16)
    relabeled = _relabel_log_kinds(item)

    for policy in ORACLE_POLICIES:
        messages_a, _ = apply_policy(item, policy, task="stateful")
        messages_b, _ = apply_policy(relabeled, policy, task="stateful")
        assert messages_a != messages_b


def test_budget_info_matches_assembled_prompt() -> None:
    item = stateful_task.make_item(seed=321, n_reassign=4, n_distractors=20)
    for policy in (
        "full",
        "drop_distractors",
        "drop_relevant",
        "verbose_instruction",
        "keep_last_k:8",
        "ledger",
        "ledger_state",
        "ledger+refetch",
        "ledger+refetch_inplace",
    ):
        messages, info = apply_policy(item, policy, task="stateful")
        assert {"kept_lines", "prompt_chars"}.issubset(info)
        assert info["prompt_chars"] == sum(
            len(message["content"]) for message in messages if message["role"] == "user"
        )

    accum_item = accumulate_task.make_item(seed=654, n_ops=5, n_distractors=20)
    messages, info = apply_policy(accum_item, "ledger_accumulate", task="accumulate")
    assert info["prompt_chars"] == sum(
        len(message["content"]) for message in messages if message["role"] == "user"
    )
    assert len([s for s in accum_item.segments if s.kind in (INSTRUCTION, QUERY)]) == 2
