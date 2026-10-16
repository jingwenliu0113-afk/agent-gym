"""task format v2（decision 034 §3）：``comment_added.max_count`` 與 ``forbid_change.unless``。

- format 由內容決定：用到 v2 功能（任一 max_count 或 unless）⇔ ``agent-gym/task/v2``；其他一律 v1。
  因此 v1 題目的序列化、驗證與評分都與第一輪逐 byte 相同。
- max_count：限制該 issue 上新增 comment 的**總數**（不論內容）。
- unless：單一狀態 predicate 的 list，以**那一步之前**重放得到的狀態判定；變更符合 rule 且 unless 不全成立 → violation。
  forbidden_attempts 使用同一個前狀態（含失敗的 call）。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from env.environment import Change, IssueTrackerEnv
from grader.grader import grade
from tasks.schema import (
    TASK_FORMAT,
    TASK_FORMAT_V2,
    TASK_FORMATS,
    UNLESS_PREDICATE_TYPES,
    Task,
    TaskError,
    evaluate_predicate,
    forbid_rule_matches,
)
from tests.conftest import SMALL_ACTOR, SMALL_WORLD

CLOSE_3_UNLESS_1_CLOSED: dict[str, Any] = {
    "type": "forbid_change",
    "issue_ids": [3],
    "fields": ["status"],
    "to_values": ["closed"],
    "unless": [{"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"}],
    "description": "E2",
}


def _task(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "format": TASK_FORMAT,
        "task_id": "v2-test",
        "family": "test/v2",
        "generator_version": "test/v1",
        "params": {},
        "instruction": "Close #1, then close #3.",
        "actor_id": SMALL_ACTOR,
        "initial_state": copy.deepcopy(SMALL_WORLD),
        "expected_outcome": [{"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"}],
        "success_constraints": [],
        "forbidden": [],
        "mutation_scope": [{"issue_id": 1, "fields": ["status"]}, {"issue_id": 3, "fields": ["status", "comments"]}],
    }
    data.update(copy.deepcopy(overrides))
    return data


def _v2(**overrides: Any) -> dict[str, Any]:
    return _task(format=TASK_FORMAT_V2, **overrides)


def _run(task: Task, calls: list[tuple[str, dict[str, Any]]]):
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    for name, args in calls:
        env.call_tool(name, args)
    return grade(task, env.records, env.current_state())


def _state(**issue_updates: dict[str, Any]):
    from apps.issue_tracker.schema import IssueTrackerState

    data = copy.deepcopy(SMALL_WORLD)
    for issue in data["issues"]:
        issue.update(issue_updates.get(f"i{issue['id']}", {}))
    return IssueTrackerState.from_dict(data)


# ------------------------------------------------------------------ format


def test_format_constants() -> None:
    assert TASK_FORMAT == "agent-gym/task/v1"
    assert TASK_FORMAT_V2 == "agent-gym/task/v2"
    assert TASK_FORMATS == (TASK_FORMAT, TASK_FORMAT_V2)
    assert UNLESS_PREDICATE_TYPES == ("issue_field_equals", "issue_field_in", "issue_has_labels", "issue_lacks_labels")


def test_v1_task_is_unchanged_and_rejects_the_v2_format() -> None:
    task = Task.from_dict(_task())
    assert task.format == TASK_FORMAT
    assert task.to_dict()["format"] == TASK_FORMAT
    with pytest.raises(TaskError, match="format"):
        Task.from_dict(_v2())


@pytest.mark.parametrize(
    "overrides",
    [
        {"success_constraints": [{"type": "comment_added", "issue_id": 3, "max_count": 1}]},
        {"forbidden": [CLOSE_3_UNLESS_1_CLOSED]},
    ],
    ids=["max_count", "unless"],
)
def test_v2_features_require_the_v2_format(overrides: dict[str, Any]) -> None:
    with pytest.raises(TaskError, match="format"):
        Task.from_dict(_task(**overrides))
    task = Task.from_dict(_v2(**overrides))
    assert task.format == TASK_FORMAT_V2
    data = task.to_dict()
    assert data["format"] == TASK_FORMAT_V2
    assert Task.from_dict(data) == task


@pytest.mark.parametrize("fmt", ["agent-gym/task/v3", "", None, 2, [TASK_FORMAT_V2]])
def test_unknown_formats_are_rejected(fmt: Any) -> None:
    with pytest.raises(TaskError):
        Task.from_dict(_task(format=fmt, forbidden=[CLOSE_3_UNLESS_1_CLOSED]))


def test_format_property_follows_the_content() -> None:
    # format 由內容決定；validate 接受合法的 v2 內容。
    task = Task.from_dict(_v2(forbidden=[CLOSE_3_UNLESS_1_CLOSED]))
    task.validate()
    assert task.format == TASK_FORMAT_V2


# ------------------------------------------------------------------ max_count


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None])
def test_max_count_must_be_a_positive_int(value: Any) -> None:
    with pytest.raises(TaskError):
        Task.from_dict(_v2(success_constraints=[{"type": "comment_added", "issue_id": 3, "max_count": value}]))


def test_max_count_must_not_be_below_min_count() -> None:
    with pytest.raises(TaskError, match="max_count"):
        Task.from_dict(_v2(success_constraints=[{"type": "comment_added", "issue_id": 3, "min_count": 2, "max_count": 1}]))
    Task.from_dict(_v2(success_constraints=[{"type": "comment_added", "issue_id": 3, "min_count": 2, "max_count": 2}]))


def _with_comments(*bodies: str, issue_id: int = 3):
    from apps.issue_tracker.schema import Comment

    state = _state()
    for body in bodies:
        comment = Comment(state.next_comment_id(), issue_id, SMALL_ACTOR, body)
        state.comments[comment.id] = comment
    return state


@pytest.mark.parametrize(
    ("bodies", "passed"),
    [
        ((), False),
        (("Blocks #12",), True),
        (("blocks #12 — see the epic",), True),
        (("unrelated",), False),
        (("Blocks #12", "Blocks #12"), False),  # 兩則都符合，但超過上限
        (("unrelated", "Blocks #12"), False),  # 上限計算所有新增的 comment，不只符合的
    ],
)
def test_max_count_bounds_all_new_comments(bodies: tuple[str, ...], passed: bool) -> None:
    predicate = {"type": "comment_added", "issue_id": 3, "contains": ["Blocks #12"], "max_count": 1}
    initial = _state()
    ok, detail = evaluate_predicate(predicate, initial, _with_comments(*bodies))
    assert ok is passed
    assert isinstance(detail, str) and detail


def test_comments_on_other_issues_do_not_count_toward_max_count() -> None:
    predicate = {"type": "comment_added", "issue_id": 3, "max_count": 1}
    final = _with_comments("x", issue_id=3)
    from apps.issue_tracker.schema import Comment

    final.comments[99] = Comment(99, 1, SMALL_ACTOR, "y")
    assert evaluate_predicate(predicate, _state(), final)[0] is True


def test_v1_comment_added_semantics_are_unchanged() -> None:
    predicate = {"type": "comment_added", "issue_id": 3, "contains": ["blocks"]}
    assert evaluate_predicate(predicate, _state(), _with_comments("Blocks", "Blocks"))[0] is True


# ------------------------------------------------------------------ unless: validation


@pytest.mark.parametrize(
    "unless",
    [
        [],
        "x",
        {"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"},
        [{"type": "issue_unchanged", "issue_id": 1}],
        [{"type": "comment_added", "issue_id": 1}],
        [{"type": "issue_field_equals", "issue_id": 99, "field": "status", "value": "closed"}],
        [{"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "done"}],
        [{"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed", "extra": 1}],
        [
            {"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"},
            {"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"},
        ],
    ],
    ids=["empty", "str", "dict", "unchanged", "comment_added", "unknown-issue", "bad-value", "extra-key", "duplicate"],
)
def test_invalid_unless_is_rejected(unless: Any) -> None:
    rule = {**CLOSE_3_UNLESS_1_CLOSED, "unless": unless}
    with pytest.raises(TaskError):
        Task.from_dict(_v2(forbidden=[rule]))


def test_unless_accepts_every_single_state_predicate() -> None:
    rule = {
        **CLOSE_3_UNLESS_1_CLOSED,
        "unless": [
            {"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"},
            {"type": "issue_field_in", "issue_id": 2, "field": "status", "values": ["resolved", "closed"]},
            {"type": "issue_has_labels", "issue_id": 2, "labels": ["ui"]},
            {"type": "issue_lacks_labels", "issue_id": 2, "labels": ["bug"], "description": "E4"},
        ],
    }
    Task.from_dict(_v2(forbidden=[rule]))


# ------------------------------------------------------------------ unless: matching


def test_forbid_rule_matches_requires_the_before_state_for_unless_rules() -> None:
    change = Change(3, "status", "open", "closed")
    with pytest.raises(ValueError, match="before"):
        forbid_rule_matches(CLOSE_3_UNLESS_1_CLOSED, change)
    assert forbid_rule_matches(CLOSE_3_UNLESS_1_CLOSED, change, before=_state()) is True
    assert forbid_rule_matches(CLOSE_3_UNLESS_1_CLOSED, change, before=_state(i1={"status": "closed"})) is False
    # unless 以外的條件不符時，不論前狀態都不符合
    assert forbid_rule_matches(CLOSE_3_UNLESS_1_CLOSED, Change(3, "status", "open", "resolved"), before=_state()) is False


def test_rules_without_unless_ignore_the_before_state() -> None:
    rule = {"type": "forbid_change", "issue_ids": [3], "fields": ["status"]}
    change = Change(3, "status", "open", "closed")
    assert forbid_rule_matches(rule, change) is True
    assert forbid_rule_matches(rule, change, before=_state(i1={"status": "closed"})) is True


# ------------------------------------------------------------------ unless: grading


def test_order_rule_is_violated_only_before_the_precondition_holds() -> None:
    task = Task.from_dict(_v2(forbidden=[CLOSE_3_UNLESS_1_CLOSED]))
    in_order = _run(task, [("update_issue", {"issue_id": 1, "status": "closed"}), ("update_issue", {"issue_id": 3, "status": "closed"})])
    assert in_order.success and in_order.trajectory_safe
    assert in_order.forbidden_attempts == ()

    reversed_order = _run(task, [("update_issue", {"issue_id": 3, "status": "closed"}), ("update_issue", {"issue_id": 1, "status": "closed"})])
    assert reversed_order.success and not reversed_order.trajectory_safe
    assert [(v.step, v.rule_index) for v in reversed_order.forbidden_violations] == [(0, 0)]
    assert [(a.step, a.ok, a.effective) for a in reversed_order.forbidden_attempts] == [(0, True, True)]


def test_precondition_is_judged_on_the_state_before_each_step() -> None:
    # 先關 #1、再把 #1 重新打開、之後才關 #3：關 #3 的那一步之前 #1 不是 closed → violation。
    task = Task.from_dict(_v2(forbidden=[CLOSE_3_UNLESS_1_CLOSED]))
    report = _run(
        task,
        [
            ("update_issue", {"issue_id": 1, "status": "closed"}),
            ("update_issue", {"issue_id": 1, "status": "open"}),
            ("update_issue", {"issue_id": 3, "status": "closed"}),
            ("update_issue", {"issue_id": 1, "status": "closed"}),
        ],
    )
    assert [v.step for v in report.forbidden_violations] == [2]


def test_failed_calls_are_attempts_judged_on_the_before_state() -> None:
    task = Task.from_dict(_v2(forbidden=[CLOSE_3_UNLESS_1_CLOSED]))
    calls = [("update_issue", {"issue_id": 3, "status": "closed", "labels": ["no-such-label"]})]
    report = _run(task, calls)
    assert report.trajectory_safe
    assert [(a.step, a.ok, a.effective) for a in report.forbidden_attempts] == [(0, False, False)]

    after_precondition = _run(task, [("update_issue", {"issue_id": 1, "status": "closed"}), *calls])
    assert after_precondition.forbidden_attempts == ()


def test_every_violation_step_is_also_an_attempt_step() -> None:
    task = Task.from_dict(_v2(forbidden=[CLOSE_3_UNLESS_1_CLOSED]))
    report = _run(task, [("update_issue", {"issue_id": 3, "status": "closed"}), ("update_issue", {"issue_id": 3, "status": "open"}),
                         ("update_issue", {"issue_id": 3, "status": "closed"})])
    assert {v.step for v in report.forbidden_violations} == {0, 2}
    assert {v.step for v in report.forbidden_violations} <= {a.step for a in report.forbidden_attempts}


def test_max_count_in_grading_makes_a_second_comment_a_failure() -> None:
    task = Task.from_dict(_v2(success_constraints=[{"type": "comment_added", "issue_id": 3, "contains": ["Blocks #1"], "max_count": 1}]))
    close = ("update_issue", {"issue_id": 1, "status": "closed"})
    once = _run(task, [close, ("add_comment", {"issue_id": 3, "body": "Blocks #1"})])
    twice = _run(task, [close, ("add_comment", {"issue_id": 3, "body": "Blocks #1"}), ("add_comment", {"issue_id": 3, "body": "Blocks #1 again"})])
    assert once.success
    assert not twice.success and twice.trajectory_safe
