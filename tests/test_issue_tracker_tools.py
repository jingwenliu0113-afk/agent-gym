"""apps/issue_tracker/tools.py 的契約測試（decision log 001）。"""

from __future__ import annotations

import copy
import json
import random
from typing import Any

import pytest

from apps.issue_tracker.schema import IssueTrackerState
from apps.issue_tracker.tools import (
    ERROR_CODES,
    MUTATING_TOOLS,
    SEARCH_DEFAULT_LIMIT,
    SEARCH_MAX_LIMIT,
    TOOL_NAMES,
    TOOL_SPECS,
    ToolError,
    execute_tool,
    get_tool_spec,
    validate_args,
)
from tests.conftest import SMALL_ACTOR, SMALL_WORLD

FULL_ISSUE_KEYS = {
    "id",
    "title",
    "body",
    "status",
    "priority",
    "assignee_id",
    "reporter_id",
    "labels",
    "created_at",
}
SUMMARY_KEYS = {"id", "title", "status", "priority", "assignee_id", "labels"}
COMMENT_KEYS = {"id", "issue_id", "author_id", "body"}

STATUS_ENUM = "must be one of: open, in_progress, resolved, closed"
PRIORITY_ENUM = "must be one of: p0, p1, p2, p3"
AT_LEAST_ONE = "update_issue requires at least one of: title, body, status, priority, labels"
LIMIT_RANGE = "argument 'limit' must be between 1 and 50"


# ---------------------------------------------------------------- helpers


def _run(state: IssueTrackerState, name: str, args: Any, actor: str = SMALL_ACTOR) -> dict[str, Any]:
    return execute_tool(state, actor, name, args)


def _assert_tool_error(
    state: IssueTrackerState,
    name: str,
    args: Any,
    code: str,
    message: str,
    actor: str = SMALL_ACTOR,
) -> None:
    """execute_tool 必須 raise 指定錯誤，且 state 完全不變（原子性）。"""
    before = state.to_dict()
    with pytest.raises(ToolError) as excinfo:
        execute_tool(state, actor, name, copy.deepcopy(args))
    assert excinfo.value.code == code
    assert excinfo.value.message == message
    assert state.to_dict() == before


def _assert_validate_error(name: str, args: Any, code: str, message: str) -> None:
    with pytest.raises(ToolError) as excinfo:
        validate_args(name, copy.deepcopy(args))
    assert excinfo.value.code == code
    assert excinfo.value.message == message


def _assert_json_native(value: Any) -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                assert type(key) is str
                walk(child)
        elif isinstance(node, list):
            assert type(node) is list
            for child in node:
                walk(child)
        else:
            assert node is None or type(node) in (str, int, bool), repr(node)

    walk(value)
    assert json.loads(json.dumps(value)) == value


def _world_issue(issue_id: int) -> dict[str, Any]:
    return copy.deepcopy(next(i for i in SMALL_WORLD["issues"] if i["id"] == issue_id))


def _world_comment(comment_id: int) -> dict[str, Any]:
    return copy.deepcopy(next(c for c in SMALL_WORLD["comments"] if c["id"] == comment_id))


def _summary(issue: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(issue[key]) for key in SUMMARY_KEYS}


def _issue_of(state_dict: dict[str, Any], issue_id: int) -> dict[str, Any]:
    return next(i for i in state_dict["issues"] if i["id"] == issue_id)


def _ids(result: dict[str, Any]) -> list[int]:
    return [summary["id"] for summary in result["issues"]]


def _new_issue(issue_id: int, title: str, body: str, **overrides: Any) -> dict[str, Any]:
    issue = {
        "id": issue_id,
        "title": title,
        "body": body,
        "status": "open",
        "priority": "p2",
        "assignee_id": None,
        "reporter_id": "u1",
        "labels": [],
        "created_at": "2026-09-10",
    }
    issue.update(overrides)
    return issue


def _add_generated_issues(world: dict[str, Any], issue_ids: range) -> None:
    for issue_id in issue_ids:
        world["issues"].append(
            _new_issue(issue_id, f"Generated issue {issue_id}", "Generated for limit tests.")
        )


def _outcome(state: IssueTrackerState, name: str, args: Any) -> tuple[Any, ...]:
    try:
        return ("ok", execute_tool(state, SMALL_ACTOR, name, args))
    except ToolError as err:
        return ("error", err.code, err.message)


# ---------------------------------------------------------------- constants / specs


def test_tool_spec_names_match_tool_names_in_order():
    assert tuple(spec["name"] for spec in TOOL_SPECS) == TOOL_NAMES


def test_tool_names_are_the_five_spec_tools():
    assert set(TOOL_NAMES) == {
        "search_issues",
        "get_issue",
        "update_issue",
        "add_comment",
        "assign_issue",
    }
    assert len(TOOL_NAMES) == 5


def test_mutating_tools_are_exactly_the_three_writers():
    assert MUTATING_TOOLS == frozenset({"update_issue", "add_comment", "assign_issue"})
    assert MUTATING_TOOLS <= set(TOOL_NAMES)


def test_error_codes_are_the_three_documented_codes():
    assert ERROR_CODES == ("unknown_tool", "invalid_arguments", "not_found")


def test_search_limit_constants():
    assert SEARCH_DEFAULT_LIMIT == 20
    assert SEARCH_MAX_LIMIT == 50


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s["name"])
def test_tool_spec_has_name_description_and_strict_object_schema(spec):
    assert set(spec) == {"name", "description", "input_schema"}
    assert isinstance(spec["description"], str) and spec["description"]
    schema = spec["input_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert isinstance(schema["properties"], dict)
    assert set(schema["required"]) <= set(schema["properties"])


def test_tool_error_exposes_code_and_message():
    err = ToolError("not_found", "issue 9 not found")
    assert isinstance(err, Exception)
    assert err.code == "not_found"
    assert err.message == "issue 9 not found"
    assert str(err) == "not_found: issue 9 not found"


@pytest.fixture
def restore_tool_specs():
    """若 get_tool_spec 沒有深拷貝，測試會污染 TOOL_SPECS；結束後原地復原以免波及其他測試。"""
    saved = copy.deepcopy(TOOL_SPECS)
    yield saved
    for spec, original in zip(TOOL_SPECS, saved):
        spec.clear()
        spec.update(copy.deepcopy(original))


def _mutate_every_container(node: Any) -> None:
    if isinstance(node, dict):
        for child in list(node.values()):
            _mutate_every_container(child)
        node["__mutated__"] = True
    elif isinstance(node, list):
        for child in list(node):
            _mutate_every_container(child)
        node.append("__mutated__")


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_get_tool_spec_returns_equal_spec(name):
    expected = next(spec for spec in TOOL_SPECS if spec["name"] == name)
    spec = get_tool_spec(name)
    assert spec == expected
    assert spec is not expected


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_get_tool_spec_returns_deep_copy(name, restore_tool_specs):
    spec = get_tool_spec(name)
    _mutate_every_container(spec)
    assert list(TOOL_SPECS) == list(restore_tool_specs)
    assert get_tool_spec(name) == next(s for s in restore_tool_specs if s["name"] == name)


@pytest.mark.parametrize("name", ["nope", "", "Search_Issues", "list_users", "get_issue "])
def test_get_tool_spec_unknown_name_raises_unknown_tool(name):
    with pytest.raises(ToolError) as excinfo:
        get_tool_spec(name)
    assert excinfo.value.code == "unknown_tool"
    assert excinfo.value.message == f"unknown tool '{name}'"


# ---------------------------------------------------------------- validation: 1. unknown tool

UNKNOWN_TOOL_CASES = [
    pytest.param("nope", {}, id="unknown-empty-args"),
    pytest.param("nope", "not a dict", id="unknown-beats-non-dict-args"),
    pytest.param("nope", None, id="unknown-beats-none-args"),
    pytest.param("Search_Issues", {"query": "x"}, id="case-sensitive-name"),
    pytest.param("list_users", {"bogus": 1}, id="unknown-beats-unexpected-args"),
    pytest.param("", {}, id="empty-name"),
]


@pytest.mark.parametrize("name, args", UNKNOWN_TOOL_CASES)
def test_validate_args_unknown_tool(name, args):
    _assert_validate_error(name, args, "unknown_tool", f"unknown tool '{name}'")


@pytest.mark.parametrize("name, args", UNKNOWN_TOOL_CASES)
def test_execute_tool_unknown_tool_keeps_state(small_world, name, args):
    state = IssueTrackerState.from_dict(small_world)
    _assert_tool_error(state, name, args, "unknown_tool", f"unknown tool '{name}'")


# ---------------------------------------------------------------- validation: 2..6 (invalid_arguments)

INVALID_ARGUMENT_CASES = [
    # 2. args 不是 dict
    pytest.param("get_issue", None, "arguments must be an object", id="args-none"),
    pytest.param("get_issue", [], "arguments must be an object", id="args-empty-list"),
    pytest.param("search_issues", "query=login", "arguments must be an object", id="args-str"),
    pytest.param("update_issue", 1, "arguments must be an object", id="args-int"),
    pytest.param("assign_issue", [["issue_id", 1]], "arguments must be an object", id="args-pairs"),
    pytest.param("add_comment", True, "arguments must be an object", id="args-bool"),
    # 3. 多餘參數（依名稱排序取第一個）
    pytest.param(
        "get_issue",
        {"issue_id": 1, "zeta": 1, "alpha": 2},
        "unexpected argument 'alpha'",
        id="unexpected-first-sorted",
    ),
    pytest.param(
        "search_issues",
        {"query": "x", "sort": "id", "offset": 1},
        "unexpected argument 'offset'",
        id="unexpected-search-sorted",
    ),
    pytest.param("search_issues", {"q": "login"}, "unexpected argument 'q'", id="unexpected-q"),
    # 字串排序 = Python str 比較（code point）：sorted(['b', 'B', '_x']) == ['B', '_x', 'b']
    pytest.param(
        "get_issue",
        {"issue_id": 1, "b": 1, "B": 1, "_x": 1},
        "unexpected argument 'B'",
        id="unexpected-sorted-case-sensitive",
    ),
    pytest.param(
        "get_issue",
        {"issue_id": 1, "é": 1, "z": 1},
        "unexpected argument 'z'",
        id="unexpected-sorted-non-ascii",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "assignee_id": "u1"},
        "unexpected argument 'assignee_id'",
        id="update-cannot-set-assignee",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "reporter_id": "u3"},
        "unexpected argument 'reporter_id'",
        id="update-cannot-set-reporter",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "title": "x", "created_at": "2026-01-01"},
        "unexpected argument 'created_at'",
        id="update-cannot-set-created-at",
    ),
    pytest.param(
        "add_comment",
        {"issue_id": 1, "body": "x", "author_id": "u1"},
        "unexpected argument 'author_id'",
        id="comment-cannot-set-author",
    ),
    pytest.param("get_issue", {"id": 1}, "unexpected argument 'id'", id="unexpected-beats-missing"),
    pytest.param(
        "assign_issue",
        {"issue_id": "1", "assignee": "u1"},
        "unexpected argument 'assignee'",
        id="unexpected-beats-missing-and-type",
    ),
    # 4. 缺 required（依 spec required 順序）
    pytest.param("get_issue", {}, "missing required argument 'issue_id'", id="get-missing-issue"),
    pytest.param("update_issue", {}, "missing required argument 'issue_id'", id="update-missing-issue"),
    pytest.param(
        "update_issue",
        {"title": "x"},
        "missing required argument 'issue_id'",
        id="update-missing-issue-with-field",
    ),
    pytest.param("add_comment", {}, "missing required argument 'issue_id'", id="comment-missing-both"),
    pytest.param(
        "add_comment", {"body": "x"}, "missing required argument 'issue_id'", id="comment-missing-issue"
    ),
    pytest.param(
        "add_comment", {"issue_id": 1}, "missing required argument 'body'", id="comment-missing-body"
    ),
    pytest.param("assign_issue", {}, "missing required argument 'issue_id'", id="assign-missing-both"),
    pytest.param(
        "assign_issue",
        {"issue_id": 1},
        "missing required argument 'assignee_id'",
        id="assign-missing-assignee-even-though-nullable",
    ),
    pytest.param(
        "add_comment", {"body": ""}, "missing required argument 'issue_id'", id="missing-beats-min-length"
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": "x"},
        "missing required argument 'assignee_id'",
        id="missing-beats-issue-type",
    ),
    pytest.param(
        "assign_issue",
        {"assignee_id": 5},
        "missing required argument 'issue_id'",
        id="missing-beats-assignee-type",
    ),
    # 5. 依 properties 宣告順序逐欄位檢查
    pytest.param(
        "update_issue",
        {"issue_id": "1", "title": ""},
        "argument 'issue_id' must be of type integer",
        id="update-order-issue-before-title",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "title": "", "body": 5},
        "argument 'title' must not be empty",
        id="update-order-title-before-body",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "body": 5, "status": "bad"},
        "argument 'body' must be of type string",
        id="update-order-body-before-status",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "status": "bad", "priority": "bad"},
        f"argument 'status' {STATUS_ENUM}",
        id="update-order-status-before-priority",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": "bug", "priority": "bad"},
        f"argument 'priority' {PRIORITY_ENUM}",
        id="update-order-priority-before-labels",
    ),
    pytest.param(
        "search_issues",
        {"status": "bad", "query": ""},
        "argument 'query' must not be empty",
        id="search-order-query-before-status",
    ),
    pytest.param(
        "search_issues",
        {"priority": "bad", "status": "bad"},
        f"argument 'status' {STATUS_ENUM}",
        id="search-order-status-before-priority",
    ),
    pytest.param(
        "search_issues",
        {"label": "", "priority": "bad"},
        f"argument 'priority' {PRIORITY_ENUM}",
        id="search-order-priority-before-label",
    ),
    pytest.param(
        "search_issues",
        {"assignee_id": "", "label": ""},
        "argument 'label' must not be empty",
        id="search-order-label-before-assignee",
    ),
    pytest.param(
        "search_issues",
        {"limit": 0, "assignee_id": ""},
        "argument 'assignee_id' must not be empty",
        id="search-order-assignee-before-limit",
    ),
    pytest.param(
        "add_comment",
        {"issue_id": "1", "body": ""},
        "argument 'issue_id' must be of type integer",
        id="comment-order-issue-before-body",
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": True, "assignee_id": ""},
        "argument 'issue_id' must be of type integer",
        id="assign-order-issue-before-assignee",
    ),
    # 單一欄位內：type → enum → minLength → min/max → items
    pytest.param(
        "update_issue",
        {"issue_id": 1, "status": 5},
        "argument 'status' must be of type string",
        id="type-before-enum",
    ),
    pytest.param(
        "search_issues",
        {"priority": None},
        "argument 'priority' must be of type string",
        id="type-before-enum-none",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": "bug"},
        "argument 'labels' must be of type array",
        id="labels-str-not-array",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ("bug",)},
        "argument 'labels' must be of type array",
        id="labels-tuple-not-array",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": None},
        "argument 'labels' must be of type array",
        id="labels-none-not-array",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": {"bug": True}},
        "argument 'labels' must be of type array",
        id="labels-dict-not-array",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": [5]},
        "argument 'labels[0]' must be of type string",
        id="labels-item-type",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["bug", None]},
        "argument 'labels[1]' must be of type string",
        id="labels-item-none",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["bug", "ui", ["x"]]},
        "argument 'labels[2]' must be of type string",
        id="labels-item-nested-list",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["bug", ""]},
        "argument 'labels[1]' must not be empty",
        id="labels-item-empty",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["", 5]},
        "argument 'labels[0]' must not be empty",
        id="labels-items-by-index-empty-first",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": [5, ""]},
        "argument 'labels[0]' must be of type string",
        id="labels-items-by-index-type-first",
    ),
    # enum（依宣告順序列出）
    pytest.param(
        "update_issue", {"issue_id": 1, "status": "Open"}, f"argument 'status' {STATUS_ENUM}", id="enum-status-case"
    ),
    pytest.param(
        "update_issue", {"issue_id": 1, "status": "done"}, f"argument 'status' {STATUS_ENUM}", id="enum-status"
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "priority": "P1"},
        f"argument 'priority' {PRIORITY_ENUM}",
        id="enum-priority-case",
    ),
    pytest.param("search_issues", {"status": "OPEN"}, f"argument 'status' {STATUS_ENUM}", id="enum-search-status"),
    pytest.param(
        "search_issues", {"priority": "high"}, f"argument 'priority' {PRIORITY_ENUM}", id="enum-search-priority"
    ),
    # minLength
    pytest.param(
        "update_issue", {"issue_id": 1, "title": ""}, "argument 'title' must not be empty", id="empty-title"
    ),
    pytest.param("search_issues", {"query": ""}, "argument 'query' must not be empty", id="empty-query"),
    pytest.param("search_issues", {"label": ""}, "argument 'label' must not be empty", id="empty-label"),
    pytest.param(
        "search_issues", {"assignee_id": ""}, "argument 'assignee_id' must not be empty", id="empty-search-assignee"
    ),
    pytest.param(
        "add_comment", {"issue_id": 1, "body": ""}, "argument 'body' must not be empty", id="empty-comment-body"
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": 1, "assignee_id": ""},
        "argument 'assignee_id' must not be empty",
        id="empty-assign-assignee",
    ),
    # assignee_id 的型別在兩個 tool 不同
    pytest.param(
        "search_issues",
        {"assignee_id": None},
        "argument 'assignee_id' must be of type string",
        id="search-assignee-none-is-not-string",
    ),
    # 6. update_issue 至少一個欄位
    pytest.param("update_issue", {"issue_id": 1}, AT_LEAST_ONE, id="update-no-field"),
    pytest.param("update_issue", {"issue_id": 999}, AT_LEAST_ONE, id="update-no-field-missing-issue"),
    pytest.param(
        "update_issue",
        {"issue_id": "1"},
        "argument 'issue_id' must be of type integer",
        id="field-type-before-at-least-one",
    ),
]


@pytest.mark.parametrize("name, args, message", INVALID_ARGUMENT_CASES)
def test_validate_args_reports_first_error(name, args, message):
    _assert_validate_error(name, args, "invalid_arguments", message)


@pytest.mark.parametrize("name, args, message", INVALID_ARGUMENT_CASES)
def test_execute_tool_reports_first_validation_error_and_keeps_state(small_world, name, args, message):
    state = IssueTrackerState.from_dict(small_world)
    _assert_tool_error(state, name, args, "invalid_arguments", message)


_ISSUE_ID_EXTRA_ARGS = {
    "get_issue": {},
    "update_issue": {"status": "open"},
    "add_comment": {"body": "hello"},
    "assign_issue": {"assignee_id": None},
}


@pytest.mark.parametrize("name", list(_ISSUE_ID_EXTRA_ARGS))
@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, [1]], ids=repr)
def test_issue_id_must_be_strict_integer(small_world, name, value):
    state = IssueTrackerState.from_dict(small_world)
    args = {"issue_id": value, **_ISSUE_ID_EXTRA_ARGS[name]}
    message = "argument 'issue_id' must be of type integer"
    _assert_validate_error(name, args, "invalid_arguments", message)
    _assert_tool_error(state, name, args, "invalid_arguments", message)


_STRING_FIELDS = [
    ("search_issues", {}, "query"),
    ("search_issues", {}, "status"),
    ("search_issues", {}, "priority"),
    ("search_issues", {}, "label"),
    ("search_issues", {}, "assignee_id"),
    ("update_issue", {"issue_id": 1}, "title"),
    ("update_issue", {"issue_id": 1}, "body"),
    ("update_issue", {"issue_id": 1}, "status"),
    ("update_issue", {"issue_id": 1}, "priority"),
    ("add_comment", {"issue_id": 1}, "body"),
]


@pytest.mark.parametrize(
    "name, base, field", _STRING_FIELDS, ids=[f"{n}.{f}" for n, _, f in _STRING_FIELDS]
)
@pytest.mark.parametrize("value", [1, 1.5, True, ["x"]], ids=repr)
def test_string_fields_reject_non_string(name, base, field, value):
    args = {**base, field: value}
    _assert_validate_error(name, args, "invalid_arguments", f"argument '{field}' must be of type string")


@pytest.mark.parametrize("value", [5, 1.5, True, ["u1"], {"id": "u1"}], ids=repr)
def test_assign_issue_assignee_must_be_string_or_null(small_world, value):
    state = IssueTrackerState.from_dict(small_world)
    args = {"issue_id": 1, "assignee_id": value}
    message = "argument 'assignee_id' must be of type string or null"
    _assert_validate_error("assign_issue", args, "invalid_arguments", message)
    _assert_tool_error(state, "assign_issue", args, "invalid_arguments", message)


@pytest.mark.parametrize("value", [True, False, 1.0, 20.0, "5", None], ids=repr)
def test_search_limit_must_be_strict_integer(small_world, value):
    state = IssueTrackerState.from_dict(small_world)
    args = {"limit": value}
    message = "argument 'limit' must be of type integer"
    _assert_validate_error("search_issues", args, "invalid_arguments", message)
    _assert_tool_error(state, "search_issues", args, "invalid_arguments", message)


@pytest.mark.parametrize("value", [0, -1, 51, 1000])
def test_search_limit_out_of_bounds(small_world, value):
    state = IssueTrackerState.from_dict(small_world)
    _assert_validate_error("search_issues", {"limit": value}, "invalid_arguments", LIMIT_RANGE)
    _assert_tool_error(state, "search_issues", {"limit": value}, "invalid_arguments", LIMIT_RANGE)


VALID_ARGS_CASES = [
    pytest.param("search_issues", {}, id="search-no-filters"),
    pytest.param(
        "search_issues",
        {"query": "x", "status": "open", "priority": "p0", "label": "nope", "assignee_id": "ghost", "limit": 50},
        id="search-all-filters-unknown-refs",
    ),
    pytest.param("search_issues", {"limit": 1}, id="search-limit-min"),
    pytest.param("search_issues", {"assignee_id": "unassigned"}, id="search-unassigned"),
    pytest.param("get_issue", {"issue_id": 999}, id="get-missing-issue"),
    pytest.param("get_issue", {"issue_id": 0}, id="get-zero"),
    pytest.param("get_issue", {"issue_id": -5}, id="get-negative"),
    pytest.param("update_issue", {"issue_id": 999, "labels": ["nope"]}, id="update-unknown-refs"),
    pytest.param("update_issue", {"issue_id": 1, "labels": []}, id="update-empty-labels"),
    pytest.param("update_issue", {"issue_id": 1, "body": ""}, id="update-empty-body"),
    pytest.param("update_issue", {"issue_id": 1, "title": " "}, id="update-whitespace-title"),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "title": "t", "body": "b", "status": "closed", "priority": "p3", "labels": ["bug", "bug"]},
        id="update-all-fields",
    ),
    pytest.param("add_comment", {"issue_id": 999, "body": " "}, id="comment-whitespace-body"),
    pytest.param("assign_issue", {"issue_id": 999, "assignee_id": "ghost"}, id="assign-unknown-refs"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": None}, id="assign-null"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": "unassigned"}, id="assign-reserved-word"),
]


@pytest.mark.parametrize("name, args", VALID_ARGS_CASES)
def test_validate_args_accepts_valid_args_without_state(name, args):
    assert validate_args(name, copy.deepcopy(args)) is None


# ---------------------------------------------------------------- 7. not_found

NOT_FOUND_CASES = [
    pytest.param("get_issue", {"issue_id": 999}, "issue 999 not found", id="get"),
    pytest.param("get_issue", {"issue_id": 0}, "issue 0 not found", id="get-zero"),
    pytest.param("get_issue", {"issue_id": -1}, "issue -1 not found", id="get-negative"),
    pytest.param("add_comment", {"issue_id": 999, "body": "x"}, "issue 999 not found", id="comment"),
    pytest.param("update_issue", {"issue_id": 999, "status": "open"}, "issue 999 not found", id="update"),
    pytest.param(
        "update_issue", {"issue_id": 999, "labels": ["nope"]}, "issue 999 not found", id="update-issue-before-label"
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["zeta", "bug", "alpha", "zeta"]},
        "label 'alpha' not found",
        id="update-first-missing-label-sorted",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["zeta", "bug"]},
        "label 'zeta' not found",
        id="update-missing-label",
    ),
    pytest.param(
        "update_issue", {"issue_id": 1, "labels": ["Bug"]}, "label 'Bug' not found", id="update-label-case-sensitive"
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "status": "closed", "labels": ["bug", "nope"]},
        "label 'nope' not found",
        id="update-partial-valid",
    ),
    pytest.param(
        "assign_issue", {"issue_id": 999, "assignee_id": "ghost"}, "issue 999 not found", id="assign-issue-before-user"
    ),
    pytest.param("assign_issue", {"issue_id": 999, "assignee_id": None}, "issue 999 not found", id="assign-null"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": "ghost"}, "user 'ghost' not found", id="assign-user"),
    pytest.param(
        "assign_issue",
        {"issue_id": 1, "assignee_id": "unassigned"},
        "user 'unassigned' not found",
        id="assign-reserved-word-is-not-a-user",
    ),
    pytest.param(
        "search_issues", {"label": "nope", "assignee_id": "ghost"}, "label 'nope' not found", id="search-label-first"
    ),
    pytest.param("search_issues", {"label": "BUG"}, "label 'BUG' not found", id="search-label-case-sensitive"),
    pytest.param("search_issues", {"assignee_id": "ghost"}, "user 'ghost' not found", id="search-user"),
    pytest.param(
        "search_issues",
        {"assignee_id": "ghost", "status": "open"},
        "user 'ghost' not found",
        id="search-user-with-filter",
    ),
]


@pytest.mark.parametrize("name, args, message", NOT_FOUND_CASES)
def test_execute_tool_not_found_and_keeps_state(small_world, name, args, message):
    state = IssueTrackerState.from_dict(small_world)
    _assert_tool_error(state, name, args, "not_found", message)


VALIDATION_BEATS_NOT_FOUND = [
    pytest.param("update_issue", {"issue_id": 999}, AT_LEAST_ONE, id="update-at-least-one"),
    pytest.param(
        "update_issue", {"issue_id": 999, "status": "bad"}, f"argument 'status' {STATUS_ENUM}", id="update-enum"
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["nope", ""]},
        "argument 'labels[1]' must not be empty",
        id="update-items-before-label-lookup",
    ),
    pytest.param(
        "add_comment", {"issue_id": 999, "body": ""}, "argument 'body' must not be empty", id="comment-empty-body"
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": 999, "assignee_id": ""},
        "argument 'assignee_id' must not be empty",
        id="assign-empty-assignee",
    ),
    pytest.param("search_issues", {"label": "nope", "limit": 0}, LIMIT_RANGE, id="search-limit"),
    pytest.param("get_issue", {"issue_id": 999, "extra": 1}, "unexpected argument 'extra'", id="get-unexpected"),
]


@pytest.mark.parametrize("name, args, message", VALIDATION_BEATS_NOT_FOUND)
def test_validation_errors_take_precedence_over_not_found(small_world, name, args, message):
    state = IssueTrackerState.from_dict(small_world)
    _assert_tool_error(state, name, args, "invalid_arguments", message)


def test_failed_partial_update_does_not_apply_valid_fields(small_world):
    state = IssueTrackerState.from_dict(small_world)
    _assert_tool_error(
        state,
        "update_issue",
        {"issue_id": 1, "title": "Renamed", "status": "closed", "labels": ["bug", "nope"]},
        "not_found",
        "label 'nope' not found",
    )
    assert _issue_of(state.to_dict(), 1) == _world_issue(1)


# ---------------------------------------------------------------- search_issues

SEARCH_CASES = [
    pytest.param({}, [1, 2, 3, 4], id="no-filter"),
    pytest.param({"status": "open"}, [1, 3], id="status-open"),
    pytest.param({"status": "in_progress"}, [2], id="status-in-progress"),
    pytest.param({"status": "resolved"}, [], id="status-resolved"),
    pytest.param({"status": "closed"}, [4], id="status-closed"),
    pytest.param({"priority": "p0"}, [3], id="priority-p0"),
    pytest.param({"priority": "p1"}, [1], id="priority-p1"),
    pytest.param({"priority": "p2"}, [2], id="priority-p2"),
    pytest.param({"priority": "p3"}, [4], id="priority-p3"),
    pytest.param({"label": "bug"}, [1, 3], id="label-bug"),
    pytest.param({"label": "ui"}, [1, 2], id="label-ui"),
    pytest.param({"label": "security"}, [3], id="label-security"),
    pytest.param({"assignee_id": "u1"}, [2], id="assignee-u1"),
    pytest.param({"assignee_id": "u2"}, [3, 4], id="assignee-inactive-u2"),
    pytest.param({"assignee_id": "u3"}, [], id="assignee-no-issues"),
    pytest.param({"assignee_id": "bot"}, [], id="assignee-actor"),
    pytest.param({"assignee_id": "unassigned"}, [1], id="assignee-unassigned"),
    pytest.param({"query": "login"}, [1, 2], id="query-title-or-body"),
    pytest.param({"query": "LOGIN"}, [1, 2], id="query-upper"),
    pytest.param({"query": "LoGiN"}, [1, 2], id="query-mixed-case"),
    pytest.param({"query": "登入"}, [2], id="query-non-ascii-title"),
    pytest.param({"query": "字型錯誤"}, [2], id="query-non-ascii-title-suffix"),
    pytest.param({"query": "cleanup"}, [4], id="query-title-only"),
    pytest.param({"query": "debug"}, [3], id="query-body-only"),
    pytest.param({"query": "Access tokens"}, [3], id="query-with-space"),
    pytest.param({"query": "lo"}, [1, 2, 3], id="query-substring"),
    pytest.param({"query": "rotate"}, [], id="query-ignores-comments"),
    pytest.param({"query": "build 42"}, [], id="query-ignores-comments-2"),
    pytest.param({"query": "zzz"}, [], id="query-no-match"),
    pytest.param({"status": "open", "label": "bug"}, [1, 3], id="and-status-label"),
    pytest.param({"status": "open", "label": "bug", "priority": "p0"}, [3], id="and-three"),
    pytest.param({"status": "open", "assignee_id": "unassigned"}, [1], id="and-status-unassigned"),
    pytest.param({"query": "login", "label": "bug"}, [1], id="and-query-label"),
    pytest.param({"query": "login", "status": "in_progress"}, [2], id="and-query-status"),
    pytest.param({"label": "ui", "assignee_id": "u1"}, [2], id="and-label-assignee"),
    pytest.param({"assignee_id": "u2", "status": "open"}, [3], id="and-assignee-status"),
    pytest.param({"query": "log", "priority": "p0"}, [3], id="and-query-priority"),
    pytest.param({"status": "closed", "label": "bug"}, [], id="and-empty"),
]


@pytest.mark.parametrize("args, expected_ids", SEARCH_CASES)
def test_search_issues_filters(small_world, args, expected_ids):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "search_issues", args)
    assert set(result) == {"issues", "total"}
    assert _ids(result) == expected_ids
    assert result["total"] == len(expected_ids)


def test_search_issues_returns_exact_summaries(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "search_issues", {})
    assert result["issues"] == [_summary(issue) for issue in SMALL_WORLD["issues"]]
    for summary in result["issues"]:
        assert set(summary) == SUMMARY_KEYS


def test_search_issues_summary_reflects_current_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    _run(state, "update_issue", {"issue_id": 4, "title": "Reopened", "status": "open", "labels": ["ui", "bug"]})
    _run(state, "assign_issue", {"issue_id": 4, "assignee_id": None})
    result = _run(state, "search_issues", {"query": "reopened"})
    assert result == {
        "issues": [
            {
                "id": 4,
                "title": "Reopened",
                "status": "open",
                "priority": "p3",
                "assignee_id": None,
                "labels": ["bug", "ui"],
            }
        ],
        "total": 1,
    }


def test_search_query_uses_casefold(small_world):
    small_world["issues"].append(_new_issue(5, "Straße repaving", "Resurface the road."))
    state = IssueTrackerState.from_dict(small_world)
    assert _ids(_run(state, "search_issues", {"query": "STRASSE"})) == [5]
    assert _ids(_run(state, "search_issues", {"query": "straße"})) == [5]


def test_search_query_does_not_normalize_unicode(small_world):
    small_world["issues"].append(_new_issue(5, "Café menu broken", "Menu page is broken."))
    state = IssueTrackerState.from_dict(small_world)
    assert _ids(_run(state, "search_issues", {"query": "café"})) == [5]
    assert _ids(_run(state, "search_issues", {"query": "café"})) == []


def test_search_label_membership_is_case_sensitive(small_world):
    small_world["labels"].append({"name": "Bug", "description": "Capitalised variant"})
    small_world["issues"][3]["labels"] = ["Bug"]
    state = IssueTrackerState.from_dict(small_world)
    assert _ids(_run(state, "search_issues", {"label": "bug"})) == [1, 3]
    assert _ids(_run(state, "search_issues", {"label": "Bug"})) == [4]


def test_search_orders_by_id_regardless_of_input_order(small_world):
    small_world["issues"].reverse()
    small_world["comments"].reverse()
    state = IssueTrackerState.from_dict(small_world)
    assert _ids(_run(state, "search_issues", {})) == [1, 2, 3, 4]


def test_search_default_limit_truncates_to_lowest_ids(small_world):
    _add_generated_issues(small_world, range(5, 26))
    small_world["issues"].reverse()
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "search_issues", {})
    assert _ids(result) == list(range(1, 21))
    assert result["total"] == 25


def test_search_total_counts_matches_before_default_limit(small_world):
    _add_generated_issues(small_world, range(5, 26))
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "search_issues", {"status": "open"})
    assert _ids(result) == [1, 3, *range(5, 23)]
    assert result["total"] == 23


def test_search_max_limit_returns_fifty(small_world):
    _add_generated_issues(small_world, range(5, 61))
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "search_issues", {"limit": 50})
    assert _ids(result) == list(range(1, 51))
    assert result["total"] == 60


@pytest.mark.parametrize(
    "args, expected_ids, expected_total",
    [
        pytest.param({"limit": 3}, [1, 2, 3], 4, id="limit-3"),
        pytest.param({"limit": 1}, [1], 4, id="limit-1"),
        pytest.param({"status": "open", "limit": 1}, [1], 2, id="limit-with-filter"),
        pytest.param({"limit": 50}, [1, 2, 3, 4], 4, id="limit-above-matches"),
        pytest.param({"label": "security", "limit": 1}, [3], 1, id="limit-equals-matches"),
        pytest.param({"status": "resolved", "limit": 5}, [], 0, id="limit-no-matches"),
    ],
)
def test_search_limit_truncates_but_total_does_not(small_world, args, expected_ids, expected_total):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "search_issues", args)
    assert _ids(result) == expected_ids
    assert result["total"] == expected_total


# ---------------------------------------------------------------- get_issue


def test_get_issue_returns_full_issue_and_comments(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "get_issue", {"issue_id": 1})
    assert result == {"issue": _world_issue(1), "comments": [_world_comment(1)]}
    assert set(result["issue"]) == FULL_ISSUE_KEYS
    assert set(result["comments"][0]) == COMMENT_KEYS


def test_get_issue_without_comments_returns_empty_list(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert _run(state, "get_issue", {"issue_id": 2}) == {"issue": _world_issue(2), "comments": []}


def test_get_issue_non_ascii_and_inactive_assignee(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert _run(state, "get_issue", {"issue_id": 2})["issue"]["title"] == "登入頁面字型錯誤"
    assert _run(state, "get_issue", {"issue_id": 3})["issue"]["assignee_id"] == "u2"


def test_get_issue_comments_ordered_by_id_and_filtered_by_issue(small_world):
    small_world["comments"] += [
        {"id": 7, "issue_id": 3, "author_id": "u3", "body": "seven"},
        {"id": 6, "issue_id": 1, "author_id": "u2", "body": "six"},
        {"id": 5, "issue_id": 3, "author_id": "u2", "body": "five"},
    ]
    small_world["comments"].reverse()
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "get_issue", {"issue_id": 3})
    assert [c["id"] for c in result["comments"]] == [2, 5, 7]
    assert result["comments"][1] == {"id": 5, "issue_id": 3, "author_id": "u2", "body": "five"}
    assert [c["id"] for c in _run(state, "get_issue", {"issue_id": 1})["comments"]] == [1, 6]


# ---------------------------------------------------------------- update_issue


def test_update_issue_labels_replace_whole_list(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 1, "labels": ["security"]})
    expected = {**_world_issue(1), "labels": ["security"]}
    assert result == {"issue": expected}
    assert _issue_of(state.to_dict(), 1) == expected


def test_update_issue_labels_are_sorted_and_deduplicated(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 4, "labels": ["ui", "bug", "ui", "security"]})
    assert result["issue"]["labels"] == ["bug", "security", "ui"]
    assert _issue_of(state.to_dict(), 4)["labels"] == ["bug", "security", "ui"]


def test_update_issue_empty_labels_clears_all(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 3, "labels": []})
    assert result["issue"]["labels"] == []
    assert _issue_of(state.to_dict(), 3)["labels"] == []


def test_update_issue_empty_body_is_allowed(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 1, "body": ""})
    assert result["issue"] == {**_world_issue(1), "body": ""}
    assert _issue_of(state.to_dict(), 1)["body"] == ""


def test_update_issue_empty_title_is_rejected(small_world):
    state = IssueTrackerState.from_dict(small_world)
    _assert_tool_error(
        state,
        "update_issue",
        {"issue_id": 1, "title": ""},
        "invalid_arguments",
        "argument 'title' must not be empty",
    )


def test_update_issue_multiple_fields_keeps_other_fields(small_world):
    state = IssueTrackerState.from_dict(small_world)
    args = {"issue_id": 2, "title": "Font fix", "status": "resolved", "priority": "p0"}
    result = _run(state, "update_issue", args)
    expected = {**_world_issue(2), "title": "Font fix", "status": "resolved", "priority": "p0"}
    assert result == {"issue": expected}
    assert set(result["issue"]) == FULL_ISSUE_KEYS


def test_update_issue_does_not_touch_other_issues_or_comments(small_world):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    _run(state, "update_issue", {"issue_id": 2, "status": "closed", "labels": ["bug"]})
    after = state.to_dict()
    assert [i for i in after["issues"] if i["id"] != 2] == [i for i in before["issues"] if i["id"] != 2]
    for key in ("users", "labels", "comments"):
        assert after[key] == before[key]


def test_update_issue_allows_reopening_closed_issue(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 4, "status": "open"})
    assert result["issue"]["status"] == "open"


def test_update_issue_result_matches_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 3, "title": "Renamed", "labels": ["security"]})
    assert result["issue"] == _issue_of(state.to_dict(), 3)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"issue_id": 1, "status": "open"}, id="same-status"),
        pytest.param({"issue_id": 1, "labels": ["ui", "bug"]}, id="same-labels-reordered"),
        pytest.param({"issue_id": 1, "labels": ["bug", "ui", "bug"]}, id="same-labels-duplicated"),
        pytest.param(
            {
                "issue_id": 1,
                "title": "Crash on login",
                "body": "App crashes when pressing the Login button.",
                "priority": "p1",
            },
            id="same-everything",
        ),
        pytest.param({"issue_id": 4, "labels": []}, id="same-empty-labels"),
    ],
)
def test_update_issue_same_value_succeeds_without_state_change(small_world, args):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    result = _run(state, "update_issue", args)
    assert result == {"issue": _world_issue(args["issue_id"])}
    assert state.to_dict() == before


def test_update_issue_state_does_not_alias_labels_argument(small_world):
    state = IssueTrackerState.from_dict(small_world)
    labels = ["security"]
    _run(state, "update_issue", {"issue_id": 1, "labels": labels})
    after = state.to_dict()
    labels.append("bug")
    assert state.to_dict() == after


# ---------------------------------------------------------------- add_comment


def test_add_comment_creates_comment_with_actor_and_next_id(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "add_comment", {"issue_id": 2, "body": "Looking into it."})
    expected = {"id": 3, "issue_id": 2, "author_id": SMALL_ACTOR, "body": "Looking into it."}
    assert result == {"comment": expected}
    assert set(result["comment"]) == COMMENT_KEYS
    assert state.to_dict()["comments"][-1] == expected


def test_add_comment_id_equals_next_comment_id(small_world):
    state = IssueTrackerState.from_dict(small_world)
    expected_id = state.next_comment_id()
    result = _run(state, "add_comment", {"issue_id": 1, "body": "x"})
    assert result["comment"]["id"] == expected_id
    assert state.next_comment_id() == expected_id + 1


def test_add_comment_author_is_the_actor(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "add_comment", {"issue_id": 1, "body": "From u3."}, actor="u3")
    assert result["comment"]["author_id"] == "u3"


def test_add_comment_sequential_ids(small_world):
    state = IssueTrackerState.from_dict(small_world)
    first = _run(state, "add_comment", {"issue_id": 1, "body": "first"})
    second = _run(state, "add_comment", {"issue_id": 3, "body": "second"})
    assert (first["comment"]["id"], second["comment"]["id"]) == (3, 4)


def test_add_comment_first_comment_gets_id_one(small_world):
    small_world["comments"] = []
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "add_comment", {"issue_id": 1, "body": "first"})
    assert result["comment"]["id"] == 1


def test_add_comment_id_is_max_plus_one_with_gaps(small_world):
    small_world["comments"][1]["id"] = 10
    state = IssueTrackerState.from_dict(small_world)
    assert _run(state, "add_comment", {"issue_id": 1, "body": "x"})["comment"]["id"] == 11


def test_add_comment_visible_in_get_issue(small_world):
    state = IssueTrackerState.from_dict(small_world)
    _run(state, "add_comment", {"issue_id": 1, "body": "已確認"})
    comments = _run(state, "get_issue", {"issue_id": 1})["comments"]
    assert comments == [
        _world_comment(1),
        {"id": 3, "issue_id": 1, "author_id": SMALL_ACTOR, "body": "已確認"},
    ]


def test_add_comment_does_not_change_issues(small_world):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    _run(state, "add_comment", {"issue_id": 4, "body": "Commenting on a closed issue."})
    after = state.to_dict()
    assert after["issues"] == before["issues"]
    assert after["comments"][: len(before["comments"])] == before["comments"]
    assert len(after["comments"]) == len(before["comments"]) + 1


# ---------------------------------------------------------------- assign_issue


def test_assign_issue_sets_assignee(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "assign_issue", {"issue_id": 1, "assignee_id": "u3"})
    expected = {**_world_issue(1), "assignee_id": "u3"}
    assert result == {"issue": expected}
    assert _issue_of(state.to_dict(), 1) == expected


def test_assign_issue_allows_inactive_user(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "assign_issue", {"issue_id": 1, "assignee_id": "u2"})
    assert result["issue"]["assignee_id"] == "u2"
    assert _issue_of(state.to_dict(), 1)["assignee_id"] == "u2"


def test_assign_issue_allows_actor(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert _run(state, "assign_issue", {"issue_id": 1, "assignee_id": SMALL_ACTOR})["issue"]["assignee_id"] == "bot"


def test_assign_issue_none_unassigns(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "assign_issue", {"issue_id": 2, "assignee_id": None})
    assert result == {"issue": {**_world_issue(2), "assignee_id": None}}
    assert _ids(_run(state, "search_issues", {"assignee_id": "unassigned"})) == [1, 2]


@pytest.mark.parametrize(
    "issue_id, assignee_id",
    [pytest.param(1, None, id="none-to-none"), pytest.param(2, "u1", id="same-user")],
)
def test_assign_issue_same_value_succeeds_without_state_change(small_world, issue_id, assignee_id):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    result = _run(state, "assign_issue", {"issue_id": issue_id, "assignee_id": assignee_id})
    assert result == {"issue": _world_issue(issue_id)}
    assert state.to_dict() == before


def test_assign_issue_only_changes_assignee(small_world):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    _run(state, "assign_issue", {"issue_id": 3, "assignee_id": "u1"})
    after = state.to_dict()
    assert _issue_of(after, 3) == {**_issue_of(before, 3), "assignee_id": "u1"}
    assert [i for i in after["issues"] if i["id"] != 3] == [i for i in before["issues"] if i["id"] != 3]
    assert after["comments"] == before["comments"]


# ---------------------------------------------------------------- read-only / JSON-native / aliasing

READ_CALLS = [
    pytest.param("search_issues", {}, id="search-all"),
    pytest.param("search_issues", {"query": "login", "limit": 1}, id="search-filtered"),
    pytest.param("get_issue", {"issue_id": 1}, id="get-1"),
    pytest.param("get_issue", {"issue_id": 3}, id="get-3"),
]


@pytest.mark.parametrize("name, args", READ_CALLS)
def test_read_only_tools_do_not_change_state(small_world, name, args):
    state = IssueTrackerState.from_dict(small_world)
    assert name not in MUTATING_TOOLS
    before = state.to_dict()
    _run(state, name, args)
    assert state.to_dict() == before


SUCCESS_CALLS = READ_CALLS + [
    pytest.param("update_issue", {"issue_id": 1, "labels": ["security", "bug"]}, id="update-labels"),
    pytest.param("update_issue", {"issue_id": 2, "body": ""}, id="update-body"),
    pytest.param("add_comment", {"issue_id": 2, "body": "看過了"}, id="comment"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": "u3"}, id="assign"),
    pytest.param("assign_issue", {"issue_id": 2, "assignee_id": None}, id="unassign"),
]


@pytest.mark.parametrize("name, args", SUCCESS_CALLS)
def test_results_are_json_native(small_world, name, args):
    state = IssueTrackerState.from_dict(small_world)
    _assert_json_native(_run(state, name, args))


def test_mutating_search_result_does_not_affect_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    result = _run(state, "search_issues", {})
    result["issues"][0]["labels"].append("security")
    result["issues"][0]["title"] = "tampered"
    result["issues"].clear()
    assert state.to_dict() == before
    assert _run(state, "search_issues", {})["issues"][0] == _summary(_world_issue(1))


def test_mutating_get_issue_result_does_not_affect_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    before = state.to_dict()
    result = _run(state, "get_issue", {"issue_id": 1})
    result["issue"]["labels"].clear()
    result["issue"]["status"] = "closed"
    result["comments"][0]["body"] = "tampered"
    result["comments"].append({"id": 99})
    assert state.to_dict() == before
    assert _run(state, "get_issue", {"issue_id": 1}) == {
        "issue": _world_issue(1),
        "comments": [_world_comment(1)],
    }


def test_mutating_update_result_does_not_affect_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "update_issue", {"issue_id": 1, "labels": ["bug"]})
    after = state.to_dict()
    result["issue"]["labels"].append("ui")
    result["issue"]["title"] = "tampered"
    assert state.to_dict() == after


def test_mutating_add_comment_result_does_not_affect_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "add_comment", {"issue_id": 1, "body": "original"})
    after = state.to_dict()
    result["comment"]["body"] = "tampered"
    result["comment"]["id"] = 99
    assert state.to_dict() == after


def test_mutating_assign_result_does_not_affect_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    result = _run(state, "assign_issue", {"issue_id": 1, "assignee_id": "u1"})
    after = state.to_dict()
    result["issue"]["assignee_id"] = "u3"
    result["issue"]["labels"].clear()
    assert state.to_dict() == after


# ---------------------------------------------------------------- determinism

_DETERMINISM_SEQUENCE = [
    ("search_issues", {"query": "login"}),
    ("get_issue", {"issue_id": 3}),
    ("update_issue", {"issue_id": 1, "labels": ["ui", "security", "ui"], "status": "in_progress"}),
    ("add_comment", {"issue_id": 1, "body": "Second look."}),
    ("assign_issue", {"issue_id": 1, "assignee_id": "u2"}),
    ("update_issue", {"issue_id": 2, "labels": ["nope", "zz"]}),
    ("update_issue", {"issue_id": 2, "status": "bad", "priority": "bad"}),
    ("assign_issue", {"issue_id": 3, "assignee_id": None}),
    ("search_issues", {"assignee_id": "unassigned"}),
    ("add_comment", {"issue_id": 3, "body": "Third."}),
    ("get_issue", {"issue_id": 1}),
]


def test_same_call_sequence_on_equal_states_gives_identical_outcomes(small_world):
    state_a = IssueTrackerState.from_dict(copy.deepcopy(small_world))
    state_b = IssueTrackerState.from_dict(copy.deepcopy(small_world))
    outcomes_a = [_outcome(state_a, name, copy.deepcopy(args)) for name, args in _DETERMINISM_SEQUENCE]
    outcomes_b = [_outcome(state_b, name, copy.deepcopy(args)) for name, args in _DETERMINISM_SEQUENCE]
    assert outcomes_a == outcomes_b
    assert state_a.to_dict() == state_b.to_dict()


@pytest.mark.parametrize("name, args", READ_CALLS)
def test_repeated_read_call_gives_identical_result(small_world, name, args):
    state = IssueTrackerState.from_dict(small_world)
    assert _run(state, name, copy.deepcopy(args)) == _run(state, name, copy.deepcopy(args))


@pytest.mark.parametrize(
    "name, args",
    [
        ("update_issue", {"issue_id": 1, "status": "bad", "priority": "bad"}),
        ("assign_issue", {"issue_id": 1, "assignee_id": "ghost"}),
        ("get_issue", {"issue_id": 1, "zeta": 1, "alpha": 1}),
    ],
)
def test_repeated_error_call_gives_identical_error(small_world, name, args):
    state = IssueTrackerState.from_dict(small_world)
    first = _outcome(state, name, copy.deepcopy(args))
    second = _outcome(state, name, copy.deepcopy(args))
    assert first[0] == "error"
    assert first == second


KEY_ORDER_CASES = [
    pytest.param(
        "update_issue",
        {"issue_id": 1, "status": "bad", "priority": "bad"},
        "invalid_arguments",
        f"argument 'status' {STATUS_ENUM}",
        id="field-order",
    ),
    pytest.param(
        "get_issue",
        {"issue_id": 1, "zeta": 1, "alpha": 1, "mid": 1},
        "invalid_arguments",
        "unexpected argument 'alpha'",
        id="unexpected-order",
    ),
    pytest.param(
        "get_issue",
        {"issue_id": 1, "b": 1, "B": 1, "_x": 1},
        "invalid_arguments",
        "unexpected argument 'B'",
        id="unexpected-order-case-sensitive",
    ),
    pytest.param(
        "get_issue",
        {"issue_id": 1, "é": 1, "z": 1},
        "invalid_arguments",
        "unexpected argument 'z'",
        id="unexpected-order-non-ascii",
    ),
    pytest.param(
        "search_issues",
        {"query": "", "limit": 0, "status": "bad"},
        "invalid_arguments",
        "argument 'query' must not be empty",
        id="search-order",
    ),
    pytest.param(
        "add_comment",
        {"issue_id": "x", "body": ""},
        "invalid_arguments",
        "argument 'issue_id' must be of type integer",
        id="comment-order",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["zz", "bug", "aa"], "status": "open"},
        "not_found",
        "label 'aa' not found",
        id="label-lookup-order",
    ),
    pytest.param(
        "search_issues",
        {"assignee_id": "ghost", "label": "nope"},
        "not_found",
        "label 'nope' not found",
        id="search-lookup-order",
    ),
]


@pytest.mark.parametrize("name, args, code, message", KEY_ORDER_CASES)
def test_error_does_not_depend_on_args_key_order(small_world, name, args, code, message):
    state = IssueTrackerState.from_dict(small_world)
    keys = list(args)
    orders = [keys, list(reversed(keys))]
    rng = random.Random(7)
    for _ in range(5):
        shuffled = keys[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    for order in orders:
        reordered = {key: copy.deepcopy(args[key]) for key in order}
        if code == "invalid_arguments":
            _assert_validate_error(name, reordered, code, message)
        _assert_tool_error(state, name, reordered, code, message)


# ---------------------------------------------------------------- randomized (fixed seed)

_RANDOM_SEED = 20261008
_RANDOM_STEPS = 300

# (大多合法的值, 不合法的值)；「合法」只用來調整分布，斷言不依賴這個分類。
_VALUE_POOLS: dict[str, tuple[list[Any], list[Any]]] = {
    "issue_id": ([1, 2, 3, 4, 999], ["1", True, 1.0]),
    "query": (["login", "LOGIN", "登入", "token", "zzz"], ["", 3]),
    "status": (["open", "in_progress", "resolved", "closed"], ["OPEN", None]),
    "priority": (["p0", "p1", "p2", "p3"], ["P1"]),
    "label": (["bug", "ui", "security", "nope"], [""]),
    "assignee_id": (["u1", "u2", "u3", "bot", "ghost", "unassigned", None], ["", 7]),
    "limit": ([1, 2, 20, 50], [0, 51, True, "3"]),
    "title": (["Renamed", "登入修正"], ["", 7]),
    "body": (["Some text", ""], [None]),
    "labels": ([["bug"], ["ui", "bug", "ui"], [], ["nope", "bug"]], ["bug", ["bug", ""], [5]]),
    "zz_extra": ([], [1]),
}

_TOOL_KEYS: dict[str, tuple[str, ...]] = {
    "search_issues": ("query", "status", "priority", "label", "assignee_id", "limit"),
    "get_issue": ("issue_id",),
    "update_issue": ("issue_id", "title", "body", "status", "priority", "labels"),
    "add_comment": ("issue_id", "body"),
    "assign_issue": ("issue_id", "assignee_id"),
}

_LIKELY_KEYS = {
    ("get_issue", "issue_id"),
    ("update_issue", "issue_id"),
    ("add_comment", "issue_id"),
    ("add_comment", "body"),
    ("assign_issue", "issue_id"),
    ("assign_issue", "assignee_id"),
}


def _pick(rng: random.Random, key: str) -> Any:
    valid, invalid = _VALUE_POOLS[key]
    if invalid and (not valid or rng.random() < 0.2):
        return copy.deepcopy(rng.choice(invalid))
    return copy.deepcopy(rng.choice(valid))


def _random_call(rng: random.Random) -> tuple[str, Any]:
    name = rng.choice(TOOL_NAMES + ("bogus_tool",))
    if rng.random() < 0.03:
        return name, rng.choice([None, [], "issue_id=1", 5])
    args: dict[str, Any] = {}
    for key in _TOOL_KEYS.get(name, ("issue_id",)):
        threshold = 0.9 if (name, key) in _LIKELY_KEYS else 0.35
        if rng.random() < threshold:
            args[key] = _pick(rng, key)
    if rng.random() < 0.05:
        stray = rng.choice(sorted(_VALUE_POOLS))
        args[stray] = _pick(rng, stray)
    return name, args


def _shuffle_keys(args: Any, rng: random.Random) -> Any:
    if not isinstance(args, dict):
        return copy.deepcopy(args)
    items = list(args.items())
    rng.shuffle(items)
    return copy.deepcopy(dict(items))


def test_random_calls_are_deterministic_and_key_order_independent(small_world):
    rng = random.Random(_RANDOM_SEED)
    state_a = IssueTrackerState.from_dict(copy.deepcopy(small_world))
    state_b = IssueTrackerState.from_dict(copy.deepcopy(small_world))
    for step in range(_RANDOM_STEPS):
        name, args = _random_call(rng)
        outcome_a = _outcome(state_a, name, copy.deepcopy(args))
        outcome_b = _outcome(state_b, name, _shuffle_keys(args, rng))
        assert outcome_a == outcome_b, (step, name, args)
    assert state_a.to_dict() == state_b.to_dict()


def test_random_calls_keep_state_on_error_and_reads_never_mutate(small_world):
    rng = random.Random(_RANDOM_SEED + 1)
    state = IssueTrackerState.from_dict(small_world)
    seen = {"ok": 0, "error": 0}
    for step in range(_RANDOM_STEPS):
        name, args = _random_call(rng)
        before = state.to_dict()
        outcome = _outcome(state, name, args)
        after = state.to_dict()
        seen[outcome[0]] += 1
        if outcome[0] == "error":
            assert outcome[1] in ERROR_CODES, (step, name, args)
            assert after == before, (step, name, args)
        else:
            _assert_json_native(outcome[1])
            if name not in MUTATING_TOOLS:
                assert after == before, (step, name, args)
    assert seen["ok"] > 0 and seen["error"] > 0


def test_random_calls_validate_args_agrees_with_execute_tool(small_world):
    rng = random.Random(_RANDOM_SEED + 2)
    state = IssueTrackerState.from_dict(small_world)
    for step in range(_RANDOM_STEPS):
        name, args = _random_call(rng)
        try:
            validate_args(name, copy.deepcopy(args))
            validation = None
        except ToolError as err:
            validation = ("error", err.code, err.message)
        outcome = _outcome(state, name, copy.deepcopy(args))
        if validation is not None:
            assert validation[1] in ("unknown_tool", "invalid_arguments")
            assert outcome == validation, (step, name, args)
        else:
            assert outcome[0] == "ok" or outcome[1] == "not_found", (step, name, args, outcome)
