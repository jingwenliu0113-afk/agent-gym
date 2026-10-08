"""env.environment 的契約測試（decision log 002）：Change、diff_states、state_hash_of、
ToolCallRecord、IssueTrackerEnv（reset / call_tool / records / current_state）。

snapshot / restore 由 tests/test_snapshot_restore.py 負責，這裡不測。
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from dataclasses import fields, replace
from typing import Any

import pytest

from apps.issue_tracker.schema import (
    MUTABLE_ISSUE_FIELDS,
    PRIORITIES,
    SCHEMA_VERSION,
    STATUSES,
    Comment,
    Issue,
    IssueTrackerState,
    Label,
    SchemaError,
    User,
)
from apps.issue_tracker.tools import TOOL_SPECS, ToolError, execute_tool
from env import environment as env_module
from env.environment import (
    COMMENTS_FIELD,
    Change,
    IssueTrackerEnv,
    ToolCallRecord,
    diff_states,
    state_hash_of,
)
from env.hashing import HASH_PREFIX, canonical_json, sha256_of
from tests.conftest import SMALL_ACTOR, SMALL_WORLD

RECORD_KEYS = frozenset(
    {
        "step",
        "tool",
        "args",
        "ok",
        "result",
        "error",
        "state_hash_before",
        "state_hash_after",
        "changes",
    }
)
CHANGE_KEYS = frozenset({"issue_id", "field", "before", "after"})
COMMENT_KEYS = frozenset({"id", "issue_id", "author_id", "body"})
FAKE_HASH_A = "sha256:" + "a" * 64
FAKE_HASH_B = "sha256:" + "b" * 64
NEW_COMMENT = {"id": 3, "issue_id": 2, "author_id": SMALL_ACTOR, "body": "On it."}


# ---------------------------------------------------------------- helpers


def _issue(issue_id: int) -> dict[str, Any]:
    return copy.deepcopy(next(i for i in SMALL_WORLD["issues"] if i["id"] == issue_id))


def _comment(comment_id: int) -> dict[str, Any]:
    return copy.deepcopy(next(c for c in SMALL_WORLD["comments"] if c["id"] == comment_id))


def _summary(issue_id: int) -> dict[str, Any]:
    issue = _issue(issue_id)
    return {k: issue[k] for k in ("id", "title", "status", "priority", "assignee_id", "labels")}


def _world(
    issue_updates: dict[int, dict[str, Any]] | None = None,
    new_comments: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> dict[str, Any]:
    """SMALL_WORLD 的深拷貝，套用 issue 欄位更新並附加新 comments（呼叫端負責維持 canonical）。"""
    w = copy.deepcopy(SMALL_WORLD)
    for issue in w["issues"]:
        issue.update(copy.deepcopy((issue_updates or {}).get(issue["id"], {})))
    w["comments"].extend(copy.deepcopy(list(new_comments)))
    return w


def _build_state(world: dict[str, Any], *, reverse: bool = False) -> IssueTrackerState:
    """直接以 dataclass 建立 state（不經 from_dict）；reverse=True 時以反向順序插入各 dict。"""

    def ordered(items: list[Any]) -> list[Any]:
        return list(reversed(items)) if reverse else list(items)

    return IssueTrackerState(
        users={u["id"]: User(**u) for u in ordered(world["users"])},
        labels={lb["name"]: Label(**lb) for lb in ordered(world["labels"])},
        issues={
            i["id"]: Issue(**{**i, "labels": tuple(i["labels"])}) for i in ordered(world["issues"])
        },
        comments={c["id"]: Comment(**c) for c in ordered(world["comments"])},
    )


def _expected_hash(world: dict[str, Any], actor: str = SMALL_ACTOR) -> str:
    """world 必須是 canonical dict（即 state.to_dict() 的形式）。"""
    return sha256_of({"schema_version": SCHEMA_VERSION, "actor_id": actor, "state": world})


def _new_env(world: dict[str, Any] | None = None, actor: str = SMALL_ACTOR) -> IssueTrackerEnv:
    env = IssueTrackerEnv()
    env.reset(copy.deepcopy(SMALL_WORLD if world is None else world), actor)
    return env


def _env_after_one_mutation() -> IssueTrackerEnv:
    env = _new_env()
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    return env


def _fingerprint(env: IssueTrackerEnv) -> tuple[Any, ...]:
    return (env.step, env.actor_id, env.records, env.state_hash(), env.current_state().to_dict())


def _tool_error(
    name: str, args: Any, world: dict[str, Any] = SMALL_WORLD, actor: str = SMALL_ACTOR
) -> dict[str, str]:
    """直接向 tool 層取得預期的錯誤；錯誤訊息模板只在 test_issue_tracker_tools.py 精確斷言。"""
    with pytest.raises(ToolError) as excinfo:
        execute_tool(IssueTrackerState.from_dict(copy.deepcopy(world)), actor, name, copy.deepcopy(args))
    return {"code": excinfo.value.code, "message": excinfo.value.message}


class _InjectedFailure(Exception):
    """測試注入的非 ToolError 例外。"""


def _raise_injected(*args: Any, **kwargs: Any) -> Any:
    raise _InjectedFailure("boom")


# ---------------------------------------------------------------- Change


def test_comments_field_constant():
    assert COMMENTS_FIELD == "comments"


def test_change_to_dict_has_exact_keys_and_values():
    assert Change(1, "priority", "p1", "p0").to_dict() == {
        "issue_id": 1,
        "field": "priority",
        "before": "p1",
        "after": "p0",
    }


def test_change_to_dict_for_comment_creation():
    d = Change(2, COMMENTS_FIELD, None, copy.deepcopy(NEW_COMMENT)).to_dict()
    assert set(d) == CHANGE_KEYS
    assert d == {"issue_id": 2, "field": "comments", "before": None, "after": NEW_COMMENT}


def test_change_to_dict_deep_copies_labels():
    change = Change(1, "labels", ["bug", "ui"], ["bug"])
    d = change.to_dict()
    assert d["before"] is not change.before
    assert d["after"] is not change.after
    d["before"].append("security")
    d["after"].clear()
    assert change.before == ["bug", "ui"]
    assert change.after == ["bug"]


def test_change_to_dict_deep_copies_comment_dict():
    change = Change(2, COMMENTS_FIELD, None, copy.deepcopy(NEW_COMMENT))
    d = change.to_dict()
    d["after"]["body"] = "tampered"
    assert change.after == NEW_COMMENT


ROUND_TRIP_CHANGES = [
    pytest.param(Change(1, "title", "Crash on login", "Login crash"), id="title"),
    pytest.param(Change(2, "body", "Font fallback", ""), id="body_to_empty"),
    pytest.param(Change(3, "status", "open", "resolved"), id="status"),
    pytest.param(Change(4, "priority", "p3", "p0"), id="priority"),
    pytest.param(Change(1, "assignee_id", None, "u3"), id="assign"),
    pytest.param(Change(3, "assignee_id", "u2", None), id="unassign"),
    pytest.param(Change(1, "labels", ["bug", "ui"], []), id="labels"),
    pytest.param(Change(2, "comments", None, dict(NEW_COMMENT)), id="comment"),
]


@pytest.mark.parametrize("change", ROUND_TRIP_CHANGES)
def test_change_dict_round_trip(change):
    assert Change.from_dict(change.to_dict()) == change


@pytest.mark.parametrize("change", ROUND_TRIP_CHANGES)
def test_change_canonical_json_round_trip(change):
    text = canonical_json(change.to_dict())
    restored = Change.from_dict(json.loads(text))
    assert restored == change
    assert canonical_json(restored.to_dict()) == text


FIELD_SAMPLE_VALUES: dict[str, tuple[Any, Any]] = {
    "title": ("Old", "New"),
    "body": ("", "text"),
    "status": ("open", "closed"),
    "priority": ("p1", "p0"),
    "assignee_id": (None, "u1"),
    "labels": (["bug"], []),
}


@pytest.mark.parametrize("field", MUTABLE_ISSUE_FIELDS)
def test_change_from_dict_accepts_every_mutable_field(field):
    before, after = FIELD_SAMPLE_VALUES[field]
    data = {"issue_id": 1, "field": field, "before": before, "after": after}
    assert Change.from_dict(data) == Change(1, field, before, after)


def test_change_from_dict_accepts_comment_creation():
    data = {"issue_id": 2, "field": "comments", "before": None, "after": dict(NEW_COMMENT)}
    assert Change.from_dict(data) == Change(2, "comments", None, NEW_COMMENT)


def test_change_from_dict_deep_copies_input():
    data = {"issue_id": 1, "field": "labels", "before": ["bug"], "after": ["bug", "ui"]}
    change = Change.from_dict(data)
    data["before"].append("security")
    data["after"].clear()
    assert change.before == ["bug"]
    assert change.after == ["bug", "ui"]


def test_change_from_dict_deep_copies_comment_dict():
    data = {"issue_id": 2, "field": "comments", "before": None, "after": dict(NEW_COMMENT)}
    change = Change.from_dict(data)
    data["after"]["body"] = "tampered"
    assert change.after == NEW_COMMENT


_VALID_CHANGE = {"issue_id": 1, "field": "priority", "before": "p1", "after": "p0"}
_VALID_COMMENT_CHANGE = {
    "issue_id": 2,
    "field": "comments",
    "before": None,
    "after": dict(NEW_COMMENT),
}

INVALID_CHANGE_DICTS = [
    pytest.param({**_VALID_CHANGE, "step": 0}, id="extra_key"),
    *[
        pytest.param({k: v for k, v in _VALID_CHANGE.items() if k != key}, id=f"missing_{key}")
        for key in _VALID_CHANGE
    ],
    pytest.param({}, id="empty_dict"),
    pytest.param({**_VALID_CHANGE, "issue_id": True}, id="issue_id_true"),
    pytest.param({**_VALID_CHANGE, "issue_id": False}, id="issue_id_false"),
    pytest.param({**_VALID_CHANGE, "issue_id": "1"}, id="issue_id_str"),
    pytest.param({**_VALID_CHANGE, "issue_id": 1.0}, id="issue_id_float"),
    pytest.param({**_VALID_CHANGE, "issue_id": None}, id="issue_id_none"),
    pytest.param({**_VALID_CHANGE, "field": "reporter_id"}, id="field_reporter_id"),
    pytest.param({**_VALID_CHANGE, "field": "created_at"}, id="field_created_at"),
    pytest.param({**_VALID_CHANGE, "field": "id"}, id="field_id"),
    pytest.param({**_VALID_CHANGE, "field": "Priority"}, id="field_wrong_case"),
    pytest.param({**_VALID_CHANGE, "field": "comment"}, id="field_comment_singular"),
    pytest.param({**_VALID_CHANGE, "field": ""}, id="field_empty"),
    pytest.param({**_VALID_CHANGE, "field": None}, id="field_none"),
    pytest.param({**_VALID_CHANGE, "field": 1}, id="field_int"),
    pytest.param({**_VALID_COMMENT_CHANGE, "before": dict(NEW_COMMENT)}, id="comments_before_dict"),
    pytest.param({**_VALID_COMMENT_CHANGE, "before": ""}, id="comments_before_empty_str"),
    pytest.param({**_VALID_COMMENT_CHANGE, "before": False}, id="comments_before_false"),
    pytest.param({**_VALID_COMMENT_CHANGE, "before": []}, id="comments_before_empty_list"),
]


@pytest.mark.parametrize("data", INVALID_CHANGE_DICTS)
def test_change_from_dict_rejects_invalid_data(data):
    with pytest.raises(ValueError):
        Change.from_dict(copy.deepcopy(data))


# ---------------------------------------------------------------- diff_states


def _diff(after_world: dict[str, Any], *, reverse: bool = False) -> tuple[Change, ...]:
    return diff_states(_build_state(SMALL_WORLD, reverse=reverse), _build_state(after_world, reverse=reverse))


def test_diff_identical_states_is_empty_tuple():
    assert diff_states(_build_state(SMALL_WORLD), _build_state(SMALL_WORLD)) == ()


def test_diff_same_object_is_empty_tuple():
    state = _build_state(SMALL_WORLD)
    assert diff_states(state, state) == ()


def test_diff_ignores_dict_insertion_order():
    assert diff_states(_build_state(SMALL_WORLD), _build_state(SMALL_WORLD, reverse=True)) == ()
    assert diff_states(_build_state(SMALL_WORLD, reverse=True), _build_state(SMALL_WORLD)) == ()


def test_diff_returns_tuple_of_changes():
    result = _diff(_world({1: {"priority": "p0"}}))
    assert isinstance(result, tuple)
    assert all(isinstance(c, Change) for c in result)


SINGLE_FIELD_CHANGES = [
    pytest.param(1, "title", "Crash on login", "Login crash", id="title"),
    pytest.param(2, "body", "Font fallback is wrong on the LOGIN page.", "", id="body_to_empty"),
    pytest.param(3, "status", "open", "resolved", id="status"),
    pytest.param(4, "priority", "p3", "p0", id="priority"),
    pytest.param(1, "assignee_id", None, "u3", id="assign"),
    pytest.param(3, "assignee_id", "u2", None, id="unassign"),
    pytest.param(2, "assignee_id", "u1", "u2", id="reassign"),
    pytest.param(1, "labels", ["bug", "ui"], ["security"], id="labels_replaced"),
    pytest.param(4, "labels", [], ["bug", "security"], id="labels_added"),
    pytest.param(3, "labels", ["bug", "security"], [], id="labels_cleared"),
]


@pytest.mark.parametrize("issue_id, field, before, after", SINGLE_FIELD_CHANGES)
def test_diff_single_field_change(issue_id, field, before, after):
    assert _diff(_world({issue_id: {field: after}})) == (Change(issue_id, field, before, after),)


@pytest.mark.parametrize(
    "issue_id, after",
    [
        pytest.param(1, ["security"], id="replaced"),
        pytest.param(4, ["bug", "security", "ui"], id="added"),
        pytest.param(3, [], id="cleared"),
    ],
)
def test_diff_labels_are_sorted_lists(issue_id, after):
    (change,) = _diff(_world({issue_id: {"labels": after}}))
    assert type(change.before) is list
    assert type(change.after) is list
    assert change.before == sorted(change.before)
    assert change.after == after


def test_diff_orders_fields_by_mutable_issue_fields():
    updates = {
        "labels": ["bug", "ui"],
        "assignee_id": "u3",
        "priority": "p0",
        "status": "resolved",
        "body": "Fixed.",
        "title": "Font issue",
    }
    before = _issue(2)
    expected = tuple(Change(2, f, before[f], updates[f]) for f in MUTABLE_ISSUE_FIELDS)
    assert _diff(_world({2: updates})) == expected


def test_diff_skips_unchanged_fields():
    result = _diff(_world({1: {"title": "Crash on login", "status": "resolved", "labels": ["bug", "ui"]}}))
    assert result == (Change(1, "status", "open", "resolved"),)


@pytest.mark.parametrize("reverse", [False, True], ids=["sorted_insertion", "reversed_insertion"])
def test_diff_orders_by_issue_id(reverse):
    after = _world({4: {"priority": "p0"}, 2: {"priority": "p0"}, 1: {"priority": "p0"}})
    assert _diff(after, reverse=reverse) == (
        Change(1, "priority", "p1", "p0"),
        Change(2, "priority", "p2", "p0"),
        Change(4, "priority", "p3", "p0"),
    )


def test_diff_comment_creation_change_shape():
    result = _diff(_world(new_comments=[NEW_COMMENT]))
    assert result == (Change(2, "comments", None, NEW_COMMENT),)
    (change,) = result
    assert change.issue_id == NEW_COMMENT["issue_id"]
    assert change.before is None
    assert set(change.after) == COMMENT_KEYS


@pytest.mark.parametrize("reverse", [False, True], ids=["sorted_insertion", "reversed_insertion"])
def test_diff_orders_fields_before_comments_and_comments_by_id(reverse):
    c3 = {"id": 3, "issue_id": 3, "author_id": "bot", "body": "c3"}
    c4 = {"id": 4, "issue_id": 1, "author_id": "bot", "body": "c4"}
    c5 = {"id": 5, "issue_id": 1, "author_id": "u1", "body": "c5"}
    after = _world({1: {"priority": "p0"}, 3: {"status": "resolved"}}, new_comments=[c3, c4, c5])
    assert _diff(after, reverse=reverse) == (
        Change(1, "priority", "p1", "p0"),
        Change(1, "comments", None, c4),
        Change(1, "comments", None, c5),
        Change(3, "status", "open", "resolved"),
        Change(3, "comments", None, c3),
    )


def _strip_label(w: dict[str, Any], name: str) -> None:
    w["labels"] = [lb for lb in w["labels"] if lb["name"] != name]
    for issue in w["issues"]:
        issue["labels"] = [x for x in issue["labels"] if x != name]


CORRUPTING_MUTATIONS = [
    pytest.param(lambda w: w["users"][1].update(name="Ann Renamed"), id="user_renamed"),
    pytest.param(lambda w: w["users"][3].update(active=False), id="user_deactivated"),
    pytest.param(lambda w: w["users"][1].update(team="frontend"), id="user_team_changed"),
    pytest.param(
        lambda w: w["users"].append({"id": "u4", "name": "New", "team": "qa", "active": True}),
        id="user_added",
    ),
    pytest.param(lambda w: w["users"].pop(0), id="user_removed"),
    pytest.param(lambda w: w["labels"][0].update(description="changed"), id="label_description_changed"),
    pytest.param(lambda w: w["labels"].append({"name": "docs", "description": "Docs"}), id="label_added"),
    pytest.param(lambda w: _strip_label(w, "security"), id="label_removed"),
    pytest.param(
        lambda w: w["issues"].append({**_issue(4), "id": 5, "title": "New issue"}), id="issue_added"
    ),
    pytest.param(lambda w: w["issues"].pop(3), id="issue_removed"),
    pytest.param(lambda w: w["issues"][0].update(reporter_id="u3"), id="reporter_changed"),
    pytest.param(lambda w: w["issues"][0].update(created_at="2026-09-30"), id="created_at_changed"),
    pytest.param(
        lambda w: w["issues"][0].update(reporter_id="u3", priority="p0"),
        id="reporter_changed_with_legit_change",
    ),
    pytest.param(lambda w: w["comments"][0].update(body="edited"), id="comment_body_modified"),
    pytest.param(lambda w: w["comments"][0].update(author_id="u3"), id="comment_author_modified"),
    pytest.param(lambda w: w["comments"][0].update(issue_id=2), id="comment_moved_to_other_issue"),
    pytest.param(lambda w: w["comments"].pop(0), id="comment_deleted"),
    pytest.param(
        lambda w: (w["comments"].pop(0), w["comments"].append(dict(NEW_COMMENT))),
        id="comment_deleted_and_new_added",
    ),
]


@pytest.mark.parametrize("mutate", CORRUPTING_MUTATIONS)
def test_diff_rejects_changes_tools_cannot_produce(mutate):
    after = copy.deepcopy(SMALL_WORLD)
    mutate(after)
    with pytest.raises(ValueError):
        diff_states(_build_state(SMALL_WORLD), _build_state(after))


# comment_deleted 反向會變成「新增 id=1 的 comment」，契約未把它列為損毀，故排除。
REVERSIBLE_CORRUPTIONS = [p for p in CORRUPTING_MUTATIONS if p.id != "comment_deleted"]


@pytest.mark.parametrize("mutate", REVERSIBLE_CORRUPTIONS)
def test_diff_rejects_reverse_direction_of_corruption(mutate):
    corrupted = copy.deepcopy(SMALL_WORLD)
    mutate(corrupted)
    with pytest.raises(ValueError):
        diff_states(_build_state(corrupted), _build_state(SMALL_WORLD))


# ---------------------------------------------------------------- state_hash_of


def test_state_hash_of_equals_sha256_of_documented_payload():
    expected = sha256_of({"schema_version": SCHEMA_VERSION, "actor_id": SMALL_ACTOR, "state": SMALL_WORLD})
    assert state_hash_of(_build_state(SMALL_WORLD), SMALL_ACTOR) == expected


def test_state_hash_of_matches_decision_002_canonical_json_definition():
    payload = {"schema_version": SCHEMA_VERSION, "actor_id": SMALL_ACTOR, "state": SMALL_WORLD}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert state_hash_of(_build_state(SMALL_WORLD), SMALL_ACTOR) == expected


def test_state_hash_of_format():
    h = state_hash_of(_build_state(SMALL_WORLD), SMALL_ACTOR)
    assert h.startswith(HASH_PREFIX)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", h)


def test_state_hash_of_is_deterministic():
    assert state_hash_of(_build_state(SMALL_WORLD), SMALL_ACTOR) == state_hash_of(
        _build_state(SMALL_WORLD), SMALL_ACTOR
    )


def test_state_hash_of_ignores_dict_insertion_order():
    assert state_hash_of(_build_state(SMALL_WORLD, reverse=True), SMALL_ACTOR) == _expected_hash(SMALL_WORLD)


@pytest.mark.parametrize("actor", ["u1", "u2", "u3"])
def test_state_hash_of_depends_on_actor_id(actor):
    state = _build_state(SMALL_WORLD)
    assert state_hash_of(state, actor) != state_hash_of(state, SMALL_ACTOR)
    assert state_hash_of(state, actor) == _expected_hash(SMALL_WORLD, actor)


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(_world({1: {"priority": "p0"}}), id="priority"),
        pytest.param(_world({4: {"labels": ["bug"]}}), id="labels"),
        pytest.param(_world(new_comments=[NEW_COMMENT]), id="new_comment"),
    ],
)
def test_state_hash_of_depends_on_state(world):
    h = state_hash_of(_build_state(world), SMALL_ACTOR)
    assert h != _expected_hash(SMALL_WORLD)
    assert h == _expected_hash(world)


# ---------------------------------------------------------------- ToolCallRecord（不經 env）


def test_tool_call_record_has_nine_documented_fields():
    assert {f.name for f in fields(ToolCallRecord)} == RECORD_KEYS


def _ok_record_dict() -> dict[str, Any]:
    return {
        "step": 0,
        "tool": "get_issue",
        "args": {"issue_id": 1},
        "ok": True,
        "result": {"issue": _issue(1), "comments": [_comment(1)]},
        "error": None,
        "state_hash_before": FAKE_HASH_A,
        "state_hash_after": FAKE_HASH_A,
        "changes": [],
    }


def _error_record_dict() -> dict[str, Any]:
    return {
        "step": 1,
        "tool": "get_issue",
        "args": {"issue_id": 99},
        "ok": False,
        "result": None,
        "error": {"code": "not_found", "message": "issue 99 not found"},
        "state_hash_before": FAKE_HASH_A,
        "state_hash_after": FAKE_HASH_A,
        "changes": [],
    }


def _mutating_record_dict() -> dict[str, Any]:
    return {
        "step": 2,
        "tool": "update_issue",
        "args": {"issue_id": 1, "priority": "p0"},
        "ok": True,
        "result": {"issue": {**_issue(1), "priority": "p0"}},
        "error": None,
        "state_hash_before": FAKE_HASH_A,
        "state_hash_after": FAKE_HASH_B,
        "changes": [{"issue_id": 1, "field": "priority", "before": "p1", "after": "p0"}],
    }


RECORD_DICT_FACTORIES = {
    "ok": _ok_record_dict,
    "error": _error_record_dict,
    "mutating": _mutating_record_dict,
}


@pytest.mark.parametrize("kind", sorted(RECORD_DICT_FACTORIES))
def test_record_from_dict_to_dict_round_trip(kind):
    data = RECORD_DICT_FACTORIES[kind]()
    record = ToolCallRecord.from_dict(copy.deepcopy(data))
    assert record.to_dict() == data


def test_record_from_dict_converts_changes_to_tuple_of_change():
    record = ToolCallRecord.from_dict(_mutating_record_dict())
    assert type(record.changes) is tuple
    assert record.changes == (Change(1, "priority", "p1", "p0"),)


def test_record_to_dict_has_exact_keys_and_changes_as_list_of_dicts():
    change = Change(2, "comments", None, dict(NEW_COMMENT))
    record = ToolCallRecord(
        step=0,
        tool="add_comment",
        args={"issue_id": 2, "body": "On it."},
        ok=True,
        result={"comment": dict(NEW_COMMENT)},
        error=None,
        state_hash_before=FAKE_HASH_A,
        state_hash_after=FAKE_HASH_B,
        changes=(change,),
    )
    d = record.to_dict()
    assert set(d) == RECORD_KEYS
    assert type(d["changes"]) is list
    assert d["changes"] == [change.to_dict()]
    assert d["args"] == {"issue_id": 2, "body": "On it."}


def test_record_to_dict_is_deep_copy():
    record = ToolCallRecord.from_dict(_mutating_record_dict())
    d = record.to_dict()
    d["args"]["issue_id"] = 999
    d["result"]["issue"]["labels"].append("security")
    d["changes"][0]["after"] = "p3"
    assert record.to_dict() == _mutating_record_dict()


def test_record_observation_for_success_is_deep_copy():
    record = ToolCallRecord.from_dict(_ok_record_dict())
    obs = record.observation()
    assert obs == {"ok": True, "data": {"issue": _issue(1), "comments": [_comment(1)]}}
    obs["data"]["issue"]["labels"].append("security")
    obs["data"]["comments"].clear()
    assert record.result == {"issue": _issue(1), "comments": [_comment(1)]}


def test_record_observation_for_failure_is_deep_copy():
    record = ToolCallRecord.from_dict(_error_record_dict())
    obs = record.observation()
    assert obs == {"ok": False, "error": {"code": "not_found", "message": "issue 99 not found"}}
    obs["error"]["code"] = "tampered"
    assert record.error == {"code": "not_found", "message": "issue 99 not found"}


INVALID_RECORD_MUTATIONS = [
    pytest.param("ok", lambda d: d.update(observation={"ok": True}), id="extra_key"),
    *[
        pytest.param("ok", lambda d, k=k: d.pop(k), id=f"missing_{k}")
        for k in sorted(RECORD_KEYS)
    ],
    pytest.param("ok", lambda d: d.update(result=None), id="ok_true_result_none"),
    pytest.param("ok", lambda d: d.update(result=[1]), id="ok_true_result_list"),
    pytest.param(
        "ok",
        lambda d: d.update(error={"code": "not_found", "message": "x"}),
        id="ok_true_with_error",
    ),
    pytest.param("error", lambda d: d.update(result={"issue": {}}), id="ok_false_with_result"),
    pytest.param("error", lambda d: d.update(error=None), id="ok_false_error_none"),
    pytest.param("error", lambda d: d["error"].pop("message"), id="ok_false_error_missing_message"),
    pytest.param("error", lambda d: d["error"].pop("code"), id="ok_false_error_missing_code"),
    pytest.param("error", lambda d: d["error"].update(detail="x"), id="ok_false_error_extra_key"),
    pytest.param("error", lambda d: d.update(error="not_found"), id="ok_false_error_not_dict"),
    pytest.param("ok", lambda d: d.update(step=True), id="step_true"),
    pytest.param("ok", lambda d: d.update(step=False), id="step_false"),
    pytest.param("ok", lambda d: d.update(step="0"), id="step_str"),
    pytest.param("ok", lambda d: d.update(step=0.0), id="step_float"),
    pytest.param("ok", lambda d: d.update(step=None), id="step_none"),
    pytest.param(
        "mutating", lambda d: d["changes"][0].update(field="reporter_id"), id="change_bad_field"
    ),
    pytest.param("mutating", lambda d: d["changes"][0].update(step=0), id="change_extra_key"),
    pytest.param("mutating", lambda d: d["changes"][0].update(issue_id=True), id="change_bool_issue_id"),
]


@pytest.mark.parametrize("kind, mutate", INVALID_RECORD_MUTATIONS)
def test_record_from_dict_rejects_invalid_data(kind, mutate):
    data = RECORD_DICT_FACTORIES[kind]()
    mutate(data)
    with pytest.raises(ValueError):
        ToolCallRecord.from_dict(data)


# ---------------------------------------------------------------- IssueTrackerEnv：未初始化 / tool_specs


UNINITIALIZED_ACCESSORS = [
    pytest.param(lambda e: e.call_tool("get_issue", {"issue_id": 1}), id="call_tool"),
    pytest.param(lambda e: e.snapshot(), id="snapshot"),
    pytest.param(lambda e: e.state_hash(), id="state_hash"),
    pytest.param(lambda e: e.current_state(), id="current_state"),
    pytest.param(lambda e: e.step, id="step"),
    pytest.param(lambda e: e.actor_id, id="actor_id"),
    pytest.param(lambda e: e.records, id="records"),
]


@pytest.mark.parametrize("access", UNINITIALIZED_ACCESSORS)
def test_uninitialized_env_raises_runtime_error(access):
    env = IssueTrackerEnv()
    with pytest.raises(RuntimeError):
        access(env)


def test_tool_specs_is_static_and_equals_tool_specs():
    specs = IssueTrackerEnv.tool_specs()
    assert isinstance(specs, tuple)
    assert specs == TOOL_SPECS


def test_tool_specs_works_on_uninitialized_instance():
    assert IssueTrackerEnv().tool_specs() == TOOL_SPECS


def test_tool_specs_returns_deep_copy():
    pristine = copy.deepcopy(TOOL_SPECS)
    specs = IssueTrackerEnv.tool_specs()
    assert all(a is not b for a, b in zip(specs, TOOL_SPECS))
    specs[0]["name"] = "tampered"
    specs[0]["input_schema"]["properties"]["query"]["minLength"] = 99
    specs[2]["input_schema"]["properties"]["labels"]["items"]["minLength"] = 0
    specs[2]["input_schema"]["required"].append("title")
    assert TOOL_SPECS == pristine
    assert IssueTrackerEnv.tool_specs() == pristine


# ---------------------------------------------------------------- reset


def test_reset_with_dict_initializes_env(small_world):
    env = IssueTrackerEnv()
    env.reset(small_world, SMALL_ACTOR)
    assert env.step == 0
    assert env.actor_id == SMALL_ACTOR
    assert env.records == ()
    assert isinstance(env.records, tuple)
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD)


def test_reset_with_state_object_initializes_env():
    state = _build_state(SMALL_WORLD)
    env = IssueTrackerEnv()
    env.reset(state, SMALL_ACTOR)
    current = env.current_state()
    assert isinstance(current, IssueTrackerState)
    assert current == state
    assert current is not state
    assert env.state_hash() == _expected_hash(SMALL_WORLD)
    assert env.step == 0
    assert env.records == ()


def test_reset_with_state_object_normalizes_insertion_order():
    env = IssueTrackerEnv()
    env.reset(_build_state(SMALL_WORLD, reverse=True), SMALL_ACTOR)
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD)


def test_reset_with_dict_normalizes_non_canonical_labels(small_world):
    small_world["issues"][0]["labels"] = ["ui", "bug", "ui"]
    env = IssueTrackerEnv()
    env.reset(small_world, SMALL_ACTOR)
    assert env.current_state().issues[1].labels == ("bug", "ui")
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD)
    assert small_world["issues"][0]["labels"] == ["ui", "bug", "ui"]


def test_reset_with_state_object_normalizes_non_canonical_labels():
    state = _build_state(SMALL_WORLD)
    state.issues[1] = replace(state.issues[1], labels=("ui", "bug", "ui"))
    env = IssueTrackerEnv()
    env.reset(state, SMALL_ACTOR)
    assert env.current_state().issues[1].labels == ("bug", "ui")
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD)
    assert state.issues[1].labels == ("ui", "bug", "ui")


def test_reset_with_other_actor():
    env = _new_env(actor="u3")
    assert env.actor_id == "u3"
    assert env.state_hash() == _expected_hash(SMALL_WORLD, "u3")


INVALID_WORLDS = [
    pytest.param(lambda w: w.pop("comments"), id="missing_top_level_key"),
    pytest.param(lambda w: w.update(extra=[]), id="extra_top_level_key"),
    pytest.param(lambda w: w["issues"][0].update(status="done"), id="bad_status"),
    pytest.param(lambda w: w["issues"][0].update(priority="P1"), id="bad_priority"),
    pytest.param(lambda w: w["issues"][3].update(id=True), id="bool_issue_id"),
    pytest.param(lambda w: w["issues"][0].update(assignee_id="ghost"), id="dangling_assignee"),
    pytest.param(lambda w: w["issues"][0].update(labels=["nope"]), id="unknown_label"),
    pytest.param(lambda w: w["issues"][1].update(id=1), id="duplicate_issue_id"),
    pytest.param(lambda w: w["issues"][0].pop("created_at"), id="issue_missing_field"),
    pytest.param(
        lambda w: w["users"].append({"id": "unassigned", "name": "X", "team": "x", "active": True}),
        id="reserved_user_id",
    ),
    pytest.param(lambda w: w["issues"].append("not an issue"), id="issue_not_dict"),
]


@pytest.mark.parametrize("mutate", INVALID_WORLDS)
def test_reset_rejects_invalid_dict_with_schema_error(small_world, mutate):
    mutate(small_world)
    with pytest.raises(SchemaError):
        IssueTrackerEnv().reset(small_world, SMALL_ACTOR)


def test_reset_rejects_invalid_state_object_with_schema_error():
    state = _build_state(SMALL_WORLD)
    state.issues[1] = replace(state.issues[1], status="done")
    with pytest.raises(SchemaError):
        IssueTrackerEnv().reset(state, SMALL_ACTOR)


# 契約：「actor_id 必須是既有 user，否則 raise ValueError」——非 str / unhashable 也不得漏出 TypeError。
NON_STR_ACTORS = [
    ("none", None),
    ("int", 1),
    ("bool", True),
    ("list", ["bot"]),
    ("dict", {"id": "bot"}),
]


@pytest.mark.parametrize("actor", ["ghost", "unassigned", "", "BOT"])
def test_reset_rejects_unknown_actor(small_world, actor):
    with pytest.raises(ValueError):
        IssueTrackerEnv().reset(small_world, actor)


@pytest.mark.parametrize(
    "actor", [actor for _, actor in NON_STR_ACTORS], ids=[label for label, _ in NON_STR_ACTORS]
)
def test_reset_rejects_non_str_actor_with_value_error(small_world, actor):
    env = IssueTrackerEnv()
    with pytest.raises(ValueError):
        env.reset(small_world, actor)
    with pytest.raises(RuntimeError):
        env.step


BAD_RESETS = [
    pytest.param(lambda w: w.pop("comments"), SMALL_ACTOR, SchemaError, id="missing_key"),
    pytest.param(lambda w: w["issues"][0].update(status="done"), SMALL_ACTOR, SchemaError, id="bad_status"),
    pytest.param(lambda w: None, "ghost", ValueError, id="unknown_actor"),
    *[
        pytest.param(lambda w: None, actor, ValueError, id=f"actor_{label}")
        for label, actor in NON_STR_ACTORS
    ],
]


@pytest.mark.parametrize("mutate, actor, exc", BAD_RESETS)
def test_failed_reset_leaves_initialized_env_unchanged(mutate, actor, exc):
    env = _env_after_one_mutation()
    before = _fingerprint(env)
    world = _world({2: {"title": "Other world"}})
    mutate(world)
    with pytest.raises(exc):
        env.reset(world, actor)
    assert _fingerprint(env) == before
    env.call_tool("get_issue", {"issue_id": 1})
    assert env.records[-1].step == 1


@pytest.mark.parametrize("mutate, actor, exc", BAD_RESETS)
def test_failed_reset_keeps_uninitialized_env_uninitialized(mutate, actor, exc):
    env = IssueTrackerEnv()
    world = copy.deepcopy(SMALL_WORLD)
    mutate(world)
    with pytest.raises(exc):
        env.reset(world, actor)
    with pytest.raises(RuntimeError):
        env.step
    with pytest.raises(RuntimeError):
        env.state_hash()


def test_reset_clears_records_and_step():
    env = _env_after_one_mutation()
    env.call_tool("get_issue", {"issue_id": 99})
    env.call_tool("add_comment", {"issue_id": 1, "body": "note"})
    assert env.step == 3
    env.reset(copy.deepcopy(SMALL_WORLD), "u1")
    assert env.step == 0
    assert env.records == ()
    assert env.actor_id == "u1"
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD, "u1")
    env.call_tool("get_issue", {"issue_id": 1})
    assert env.records[0].step == 0
    assert env.step == 1


def test_reset_loads_given_world():
    world = _world({1: {"priority": "p0", "labels": ["security"]}})
    env = _new_env(world)
    assert env.current_state().to_dict() == world
    assert env.state_hash() == _expected_hash(world)


def test_reset_does_not_modify_or_alias_dict_input(small_world):
    env = IssueTrackerEnv()
    env.reset(small_world, SMALL_ACTOR)
    assert small_world == SMALL_WORLD
    small_world["issues"][0]["title"] = "tampered"
    small_world["issues"][0]["labels"].append("security")
    small_world["users"].clear()
    small_world["comments"].pop()
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD)


def test_env_mutation_does_not_touch_dict_input(small_world):
    env = IssueTrackerEnv()
    env.reset(small_world, SMALL_ACTOR)
    env.call_tool("update_issue", {"issue_id": 1, "labels": ["security"], "priority": "p0"})
    env.call_tool("add_comment", {"issue_id": 1, "body": "note"})
    assert small_world == SMALL_WORLD


def test_reset_does_not_modify_or_alias_state_input():
    state = _build_state(SMALL_WORLD)
    env = IssueTrackerEnv()
    env.reset(state, SMALL_ACTOR)
    assert state == _build_state(SMALL_WORLD)
    state.issues[1] = replace(state.issues[1], title="tampered")
    state.issues.pop(4)
    state.users.clear()
    assert env.current_state().to_dict() == SMALL_WORLD


def test_env_mutation_does_not_touch_state_input():
    state = _build_state(SMALL_WORLD)
    env = IssueTrackerEnv()
    env.reset(state, SMALL_ACTOR)
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    env.call_tool("add_comment", {"issue_id": 1, "body": "note"})
    assert state == _build_state(SMALL_WORLD)


# ---------------------------------------------------------------- call_tool：observation 與 record


def test_get_issue_observation_and_record():
    env = _new_env()
    h0 = _expected_hash(SMALL_WORLD)
    obs = env.call_tool("get_issue", {"issue_id": 1})
    data = {"issue": _issue(1), "comments": [_comment(1)]}
    assert obs == {"ok": True, "data": data}
    assert env.step == 1
    assert env.records == (
        ToolCallRecord(
            step=0,
            tool="get_issue",
            args={"issue_id": 1},
            ok=True,
            result=data,
            error=None,
            state_hash_before=h0,
            state_hash_after=h0,
            changes=(),
        ),
    )


def test_search_issues_observation():
    env = _new_env()
    obs = env.call_tool("search_issues", {"status": "open"})
    assert obs == {"ok": True, "data": {"issues": [_summary(1), _summary(3)], "total": 2}}


def test_success_observation_has_exactly_ok_and_data():
    obs = _new_env().call_tool("get_issue", {"issue_id": 2})
    assert set(obs) == {"ok", "data"}
    assert obs["ok"] is True


def test_failure_observation_has_exactly_ok_and_error():
    obs = _new_env().call_tool("get_issue", {"issue_id": 99})
    assert set(obs) == {"ok", "error"}
    assert obs["ok"] is False
    assert set(obs["error"]) == {"code", "message"}


# 預期錯誤由 tool 層產生（_tool_error）；這裡只固定 code，驗證 env 原樣轉交 ToolError。
FAILING_CALLS = [
    pytest.param("get_issue", {"issue_id": 99}, "not_found", id="get_missing_issue"),
    pytest.param("update_issue", {"issue_id": 99, "priority": "p0"}, "not_found", id="update_missing_issue"),
    pytest.param("update_issue", {"issue_id": 1, "labels": ["zzz", "aaa"]}, "not_found", id="update_unknown_label"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": "ghost"}, "not_found", id="assign_unknown_user"),
    pytest.param("add_comment", {"issue_id": 99, "body": "hi"}, "not_found", id="comment_missing_issue"),
    pytest.param("search_issues", {"label": "nope"}, "not_found", id="search_unknown_label"),
    pytest.param("update_issue", {"issue_id": 1}, "invalid_arguments", id="update_without_fields"),
    pytest.param(
        "update_issue", {"issue_id": 1, "priority": "p0", "extra": 1}, "invalid_arguments", id="unexpected_argument"
    ),
    pytest.param("assign_issue", {"issue_id": 1}, "invalid_arguments", id="missing_argument"),
    pytest.param("get_issue", {"issue_id": "1"}, "invalid_arguments", id="str_issue_id"),
    pytest.param("get_issue", {"issue_id": True}, "invalid_arguments", id="bool_issue_id"),
    pytest.param("get_issue", {"issue_id": 1.0}, "invalid_arguments", id="float_issue_id"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": 5}, "invalid_arguments", id="int_assignee"),
    pytest.param("search_issues", {"status": "done"}, "invalid_arguments", id="bad_enum"),
    pytest.param("add_comment", {"issue_id": 1, "body": ""}, "invalid_arguments", id="empty_comment_body"),
    pytest.param("update_issue", {"issue_id": 1, "labels": ["bug", ""]}, "invalid_arguments", id="empty_label_item"),
    pytest.param("search_issues", {"limit": 0}, "invalid_arguments", id="limit_out_of_range"),
    pytest.param("get_issue", [1], "invalid_arguments", id="non_dict_args"),
    pytest.param("close_issue", {"issue_id": 1}, "unknown_tool", id="unknown_tool"),
]


@pytest.mark.parametrize("name, args, code", FAILING_CALLS)
def test_failing_call_is_recorded_without_state_change(name, args, code):
    error = _tool_error(name, args)
    assert error["code"] == code
    env = _new_env()
    h0 = env.state_hash()
    obs = env.call_tool(name, copy.deepcopy(args))
    assert obs == {"ok": False, "error": error}
    assert env.step == 1
    assert env.records == (
        ToolCallRecord(
            step=0,
            tool=name,
            args=args,
            ok=False,
            result=None,
            error=error,
            state_hash_before=h0,
            state_hash_after=h0,
            changes=(),
        ),
    )
    assert env.state_hash() == h0
    assert env.current_state().to_dict() == SMALL_WORLD


READ_ONLY_CALLS = [
    pytest.param("get_issue", {"issue_id": 1}, id="get_1"),
    pytest.param("get_issue", {"issue_id": 4}, id="get_4"),
    pytest.param("search_issues", {}, id="search_all"),
    pytest.param("search_issues", {"query": "LOGIN"}, id="search_query"),
    pytest.param("search_issues", {"assignee_id": "unassigned"}, id="search_unassigned"),
    pytest.param("search_issues", {"label": "bug", "limit": 1}, id="search_label_limit"),
]


@pytest.mark.parametrize("name, args", READ_ONLY_CALLS)
def test_read_only_call_has_equal_hashes_and_no_changes(name, args):
    env = _new_env()
    h0 = env.state_hash()
    obs = env.call_tool(name, copy.deepcopy(args))
    assert obs["ok"] is True
    (rec,) = env.records
    assert rec.ok is True
    assert rec.error is None
    assert rec.result == obs["data"]
    assert rec.state_hash_before == rec.state_hash_after == h0
    assert rec.changes == ()
    assert env.state_hash() == h0
    assert env.current_state().to_dict() == SMALL_WORLD


NOOP_WRITES = [
    pytest.param("update_issue", {"issue_id": 1, "priority": "p1"}, id="same_priority"),
    pytest.param("update_issue", {"issue_id": 1, "labels": ["ui", "bug"]}, id="same_labels_unsorted"),
    pytest.param("update_issue", {"issue_id": 1, "labels": ["bug", "ui", "bug"]}, id="same_labels_duplicated"),
    pytest.param("update_issue", {"issue_id": 4, "labels": []}, id="clear_empty_labels"),
    pytest.param(
        "update_issue", {"issue_id": 2, "title": "登入頁面字型錯誤", "status": "in_progress"}, id="same_title_status"
    ),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": None}, id="unassign_unassigned"),
    pytest.param("assign_issue", {"issue_id": 2, "assignee_id": "u1"}, id="assign_same_user"),
]


@pytest.mark.parametrize("name, args", NOOP_WRITES)
def test_noop_write_succeeds_without_changes(name, args):
    env = _new_env()
    h0 = env.state_hash()
    obs = env.call_tool(name, copy.deepcopy(args))
    assert obs["ok"] is True
    assert env.step == 1
    (rec,) = env.records
    assert rec.ok is True
    assert rec.changes == ()
    assert rec.state_hash_before == rec.state_hash_after == h0
    assert env.current_state().to_dict() == SMALL_WORLD


MUTATING_CALLS = [
    pytest.param(
        "update_issue",
        {"issue_id": 1, "priority": "p0"},
        (Change(1, "priority", "p1", "p0"),),
        {1: {"priority": "p0"}},
        (),
        id="update_priority",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["security", "bug", "security"]},
        (Change(1, "labels", ["bug", "ui"], ["bug", "security"]),),
        {1: {"labels": ["bug", "security"]}},
        (),
        id="update_labels_replace",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 3, "labels": []},
        (Change(3, "labels", ["bug", "security"], []),),
        {3: {"labels": []}},
        (),
        id="update_labels_clear",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 2, "body": ""},
        (Change(2, "body", "Font fallback is wrong on the LOGIN page.", ""),),
        {2: {"body": ""}},
        (),
        id="update_body_empty",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "labels": ["security"], "priority": "p0", "status": "resolved", "body": "Fixed", "title": "Login crash"},
        (
            Change(1, "title", "Crash on login", "Login crash"),
            Change(1, "body", "App crashes when pressing the Login button.", "Fixed"),
            Change(1, "status", "open", "resolved"),
            Change(1, "priority", "p1", "p0"),
            Change(1, "labels", ["bug", "ui"], ["security"]),
        ),
        {1: {"labels": ["security"], "priority": "p0", "status": "resolved", "body": "Fixed", "title": "Login crash"}},
        (),
        id="update_many_fields_ordered",
    ),
    pytest.param(
        "update_issue",
        {"issue_id": 1, "priority": "p1", "status": "closed"},
        (Change(1, "status", "open", "closed"),),
        {1: {"status": "closed"}},
        (),
        id="update_partially_same",
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": 1, "assignee_id": "u3"},
        (Change(1, "assignee_id", None, "u3"),),
        {1: {"assignee_id": "u3"}},
        (),
        id="assign",
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": 1, "assignee_id": "u2"},
        (Change(1, "assignee_id", None, "u2"),),
        {1: {"assignee_id": "u2"}},
        (),
        id="assign_inactive_user",
    ),
    pytest.param(
        "assign_issue",
        {"issue_id": 3, "assignee_id": None},
        (Change(3, "assignee_id", "u2", None),),
        {3: {"assignee_id": None}},
        (),
        id="unassign",
    ),
    pytest.param(
        "add_comment",
        {"issue_id": 2, "body": "On it."},
        (Change(2, "comments", None, NEW_COMMENT),),
        {},
        (NEW_COMMENT,),
        id="add_comment",
    ),
]


@pytest.mark.parametrize("name, args, changes, issue_updates, new_comments", MUTATING_CALLS)
def test_mutating_call_record(name, args, changes, issue_updates, new_comments):
    env = _new_env()
    obs = env.call_tool(name, copy.deepcopy(args))
    expected_world = _world(issue_updates, new_comments)
    if name == "add_comment":
        assert obs == {"ok": True, "data": {"comment": new_comments[0]}}
    else:
        issue = next(i for i in expected_world["issues"] if i["id"] == args["issue_id"])
        assert obs == {"ok": True, "data": {"issue": issue}}
    assert env.step == 1
    (rec,) = env.records
    assert rec.step == 0
    assert rec.tool == name
    assert rec.args == args
    assert rec.ok is True
    assert rec.result == obs["data"]
    assert rec.error is None
    assert rec.changes == changes
    assert type(rec.changes) is tuple
    assert rec.state_hash_before == _expected_hash(SMALL_WORLD)
    assert rec.state_hash_after == _expected_hash(expected_world)
    assert rec.state_hash_before != rec.state_hash_after
    assert env.state_hash() == rec.state_hash_after
    assert env.current_state().to_dict() == expected_world


def test_add_comment_author_is_env_actor():
    env = _new_env(actor="u3")
    obs = env.call_tool("add_comment", {"issue_id": 1, "body": "Looking."})
    comment = {"id": 3, "issue_id": 1, "author_id": "u3", "body": "Looking."}
    assert obs == {"ok": True, "data": {"comment": comment}}
    assert env.records[0].changes == (Change(1, "comments", None, comment),)
    assert env.state_hash() == _expected_hash(_world(new_comments=[comment]), "u3")


def test_step_increments_for_success_and_tool_error():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    assert env.step == 1
    env.call_tool("get_issue", {"issue_id": 99})
    assert env.step == 2
    env.call_tool("close_issue", {})
    assert env.step == 3
    env.call_tool("get_issue", "1")
    assert env.step == 4
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    assert env.step == 5
    assert [r.step for r in env.records] == [0, 1, 2, 3, 4]


def test_record_hashes_form_a_chain():
    env = _new_env()
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    env.call_tool("get_issue", {"issue_id": 1})
    env.call_tool("add_comment", {"issue_id": 3, "body": "Rotated."})
    env.call_tool("assign_issue", {"issue_id": 3, "assignee_id": "ghost"})
    env.call_tool("assign_issue", {"issue_id": 3, "assignee_id": "u1"})
    recs = env.records
    assert recs[0].state_hash_before == _expected_hash(SMALL_WORLD)
    for prev, nxt in zip(recs, recs[1:]):
        assert nxt.state_hash_before == prev.state_hash_after
    assert recs[-1].state_hash_after == env.state_hash()
    assert env.state_hash() == state_hash_of(env.current_state(), env.actor_id)


def test_state_hash_independent_of_step():
    env = _new_env()
    h0 = env.state_hash()
    for _ in range(3):
        env.call_tool("get_issue", {"issue_id": 1})
    env.call_tool("get_issue", {"issue_id": 99})
    assert env.step == 4
    assert env.state_hash() == h0 == _expected_hash(SMALL_WORLD)


def test_reverting_a_change_restores_initial_hash():
    env = _new_env()
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p1"})
    first, second = env.records
    assert first.changes == (Change(1, "priority", "p1", "p0"),)
    assert second.changes == (Change(1, "priority", "p0", "p1"),)
    assert second.state_hash_after == first.state_hash_before == _expected_hash(SMALL_WORLD)
    assert env.state_hash() == _expected_hash(SMALL_WORLD)
    assert env.step == 2


SAMPLE_CALLS = [
    pytest.param("get_issue", {"issue_id": 1}, id="read_ok"),
    pytest.param("search_issues", {"status": "open"}, id="search_ok"),
    pytest.param("get_issue", {"issue_id": 99}, id="not_found"),
    pytest.param("update_issue", {"issue_id": 1, "priority": "p0", "labels": ["security"]}, id="update"),
    pytest.param("update_issue", {"issue_id": 1, "priority": "p1"}, id="update_noop"),
    pytest.param("add_comment", {"issue_id": 2, "body": "On it."}, id="add_comment"),
    pytest.param("assign_issue", {"issue_id": 3, "assignee_id": None}, id="unassign"),
    pytest.param("close_issue", {"issue_id": 1}, id="unknown_tool"),
    pytest.param("get_issue", [1, 2], id="list_args"),
    pytest.param("get_issue", None, id="null_args"),
    pytest.param("get_issue", {"issue_id": 1.0}, id="float_issue_id"),
]


@pytest.mark.parametrize("name, args", SAMPLE_CALLS)
def test_record_observation_equals_call_tool_return_value(name, args):
    env = _new_env()
    obs = env.call_tool(name, copy.deepcopy(args))
    assert env.records[0].observation() == obs


@pytest.mark.parametrize("name, args", SAMPLE_CALLS)
def test_record_ok_result_error_are_consistent(name, args):
    env = _new_env()
    obs = env.call_tool(name, copy.deepcopy(args))
    (rec,) = env.records
    assert rec.step == 0
    assert rec.tool == name
    assert rec.args == args
    assert rec.ok is obs["ok"]
    if rec.ok:
        assert isinstance(rec.result, dict)
        assert rec.error is None
        assert rec.result == obs["data"]
    else:
        assert rec.result is None
        assert set(rec.error) == {"code", "message"}
        assert rec.error == obs["error"]
        assert rec.changes == ()
        assert rec.state_hash_before == rec.state_hash_after


@pytest.mark.parametrize("name, args", SAMPLE_CALLS)
def test_record_to_dict_from_env_has_exact_keys(name, args):
    env = _new_env()
    env.call_tool(name, copy.deepcopy(args))
    (rec,) = env.records
    d = rec.to_dict()
    assert set(d) == RECORD_KEYS
    assert type(d["changes"]) is list
    assert d["changes"] == [c.to_dict() for c in rec.changes]
    assert d["step"] == rec.step
    assert d["args"] == rec.args


@pytest.mark.parametrize("name, args", SAMPLE_CALLS)
def test_record_json_round_trip(name, args):
    env = _new_env()
    env.call_tool(name, copy.deepcopy(args))
    (rec,) = env.records
    text = canonical_json(rec.to_dict())
    restored = ToolCallRecord.from_dict(json.loads(text))
    assert restored == rec
    assert canonical_json(restored.to_dict()) == text
    assert ToolCallRecord.from_dict(rec.to_dict()) == rec


# ---------------------------------------------------------------- call_tool：非 dict 的 JSON args / unknown tool


NON_DICT_JSON_ARGS = [
    pytest.param([1], id="list"),
    pytest.param([], id="empty_list"),
    pytest.param([{"issue_id": 1}], id="list_of_dict"),
    pytest.param("issue_id=1", id="str"),
    pytest.param("", id="empty_str"),
    pytest.param(1, id="int"),
    pytest.param(0, id="zero"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(None, id="none"),
]


@pytest.mark.parametrize("args", NON_DICT_JSON_ARGS)
def test_non_dict_json_args_recorded_as_invalid_arguments(args):
    env = _new_env()
    obs = env.call_tool("get_issue", copy.deepcopy(args))
    error = _tool_error("get_issue", args)
    assert error["code"] == "invalid_arguments"
    assert obs == {"ok": False, "error": error}
    assert env.step == 1
    (rec,) = env.records
    assert rec.args == args
    assert type(rec.args) is type(args)
    assert rec.ok is False
    assert rec.result is None
    assert rec.error == error
    assert rec.changes == ()
    assert rec.state_hash_before == rec.state_hash_after == _expected_hash(SMALL_WORLD)


@pytest.mark.parametrize("name", ["close_issue", "", "GET_ISSUE", "get_issue ", "delete_issue"])
def test_unknown_tool_is_recorded(name):
    env = _new_env()
    h0 = env.state_hash()
    obs = env.call_tool(name, {"issue_id": 1})
    error = _tool_error(name, {"issue_id": 1})
    assert error["code"] == "unknown_tool"
    assert obs == {"ok": False, "error": error}
    assert env.step == 1
    assert env.records == (
        ToolCallRecord(
            step=0,
            tool=name,
            args={"issue_id": 1},
            ok=False,
            result=None,
            error=error,
            state_hash_before=h0,
            state_hash_after=h0,
            changes=(),
        ),
    )


@pytest.mark.parametrize("args", [None, [1], "x", 3])
def test_unknown_tool_takes_priority_over_non_dict_args(args):
    env = _new_env()
    obs = env.call_tool("close_issue", copy.deepcopy(args))
    assert obs["error"]["code"] == "unknown_tool"
    assert obs == {"ok": False, "error": _tool_error("close_issue", args)}
    assert env.records[0].args == args


# ---------------------------------------------------------------- call_tool：runner 前置條件


NON_STR_NAMES = [
    pytest.param(123, id="int"),
    pytest.param(None, id="none"),
    pytest.param(b"get_issue", id="bytes"),
    pytest.param(("get_issue",), id="tuple"),
    pytest.param(["get_issue"], id="list"),
]


@pytest.mark.parametrize("name", NON_STR_NAMES)
def test_non_str_tool_name_raises_type_error_without_record(name):
    env = _env_after_one_mutation()
    before = _fingerprint(env)
    with pytest.raises(TypeError):
        env.call_tool(name, {"issue_id": 1})
    assert _fingerprint(env) == before


NON_JSON_ARGS = [
    pytest.param("update_issue", {"issue_id": 1, "labels": ("security",)}, id="nested_tuple"),
    pytest.param("search_issues", ("status", "open"), id="top_level_tuple"),
    pytest.param("update_issue", {"issue_id": 1, "labels": {"security"}}, id="nested_set"),
    pytest.param("search_issues", {"status", "open"}, id="top_level_set"),
    pytest.param("get_issue", {"issue_id": float("nan")}, id="nan"),
    pytest.param("get_issue", {"issue_id": float("inf")}, id="inf"),
    pytest.param("get_issue", {"issue_id": float("-inf")}, id="neg_inf"),
    pytest.param("update_issue", {"issue_id": 1, "priority": "p0", 1: "x"}, id="int_key"),
    pytest.param("update_issue", {"issue_id": 1, "priority": "p0", None: "x"}, id="none_key"),
    pytest.param("update_issue", {"issue_id": 1, "priority": "p0", True: "x"}, id="bool_key"),
    pytest.param("search_issues", {"extra": [{2: "x"}]}, id="nested_int_key"),
    pytest.param("add_comment", {"issue_id": 1, "body": "bad \ud800"}, id="lone_surrogate_value"),
    pytest.param("add_comment", {"issue_id": 1, "body": "ok", "\udc00": 1}, id="lone_surrogate_key"),
    pytest.param("add_comment", {"issue_id": 1, "body": b"bytes"}, id="bytes_value"),
    pytest.param("get_issue", {"issue_id": object()}, id="object_value"),
    pytest.param("close_issue", {"issue_id": (1,)}, id="unknown_tool_with_tuple"),
]


@pytest.mark.parametrize("name, args", NON_JSON_ARGS)
def test_non_json_args_raise_without_record(name, args):
    env = _env_after_one_mutation()
    before = _fingerprint(env)
    with pytest.raises((TypeError, ValueError)):
        env.call_tool(name, args)
    assert _fingerprint(env) == before
    env.call_tool("get_issue", {"issue_id": 1})
    assert env.records[-1].step == 1
    assert env.step == 2


# ---------------------------------------------------------------- call_tool：非 ToolError 例外的原子性


def _assert_env_continues_like_untouched_twin(env: IssueTrackerEnv, name: str, args: Any) -> None:
    """失敗後再做同一個 call，結果必須與從未失敗過的 env 完全相同。"""
    twin = _env_after_one_mutation()
    assert env.call_tool(name, copy.deepcopy(args)) == twin.call_tool(name, copy.deepcopy(args))
    assert env.records == twin.records
    assert env.records[-1].step == 1
    assert env.step == 2
    assert env.state_hash() == twin.state_hash()


def test_non_tool_error_inside_tool_propagates_and_env_is_unchanged(monkeypatch):
    # tools.py 契約：add_comment 的 id = state.next_comment_id()
    env = _env_after_one_mutation()
    before = _fingerprint(env)
    args = {"issue_id": 1, "body": "x"}
    with monkeypatch.context() as m:
        m.setattr(IssueTrackerState, "next_comment_id", _raise_injected)
        with pytest.raises(_InjectedFailure):
            env.call_tool("add_comment", copy.deepcopy(args))
    assert _fingerprint(env) == before
    assert env.step == 1
    assert len(env.records) == 1
    assert env.current_state().issues[1].priority == "p0"
    assert set(env.current_state().comments) == {1, 2}
    _assert_env_continues_like_untouched_twin(env, "add_comment", args)
    assert env.records[-1].result["comment"]["id"] == 3


POST_TOOL_FAILURE_CALLS = [
    pytest.param("update_issue", {"issue_id": 1, "priority": "p2"}, id="update_issue"),
    pytest.param("assign_issue", {"issue_id": 1, "assignee_id": "u3"}, id="assign_issue"),
    pytest.param("add_comment", {"issue_id": 1, "body": "x"}, id="add_comment"),
]


@pytest.mark.parametrize("name, args", POST_TOOL_FAILURE_CALLS)
def test_diff_failure_after_tool_ran_propagates_and_env_is_unchanged(monkeypatch, name, args):
    # 契約：tool 在拷貝上執行，hash 與 diff_states 都成功後才提交；此時拷貝已被 tool 修改。
    env = _env_after_one_mutation()
    before = _fingerprint(env)
    with monkeypatch.context() as m:
        m.setattr(env_module, "diff_states", _raise_injected)
        with pytest.raises(_InjectedFailure):
            env.call_tool(name, copy.deepcopy(args))
    assert _fingerprint(env) == before
    assert env.step == 1
    assert len(env.records) == 1
    state = env.current_state()
    assert state.issues[1].priority == "p0"
    assert state.issues[1].assignee_id is None
    assert set(state.comments) == {1, 2}
    _assert_env_continues_like_untouched_twin(env, name, args)
    assert env.records[-1].changes != ()


# ---------------------------------------------------------------- 深拷貝 / aliasing


def test_call_tool_does_not_modify_args_and_records_deep_copy():
    env = _new_env()
    args = {"issue_id": 1, "labels": ["security", "bug", "security"]}
    original = copy.deepcopy(args)
    env.call_tool("update_issue", args)
    assert args == original
    assert env.records[0].args == original
    args["labels"].append("ui")
    args["issue_id"] = 2
    assert env.records[0].args == original
    assert env.current_state().issues[1].labels == ("bug", "security")


def test_record_args_are_deep_copied_for_nested_values():
    env = _new_env()
    args = {"query": "x", "extra": [{"a": [1]}]}
    env.call_tool("search_issues", args)
    assert env.records[0].error == _tool_error("search_issues", {"query": "x", "extra": [{"a": [1]}]})
    assert env.records[0].error["code"] == "invalid_arguments"
    args["extra"][0]["a"].append(2)
    assert env.records[0].args == {"query": "x", "extra": [{"a": [1]}]}


def test_mutating_success_observation_does_not_affect_env():
    env = _new_env()
    obs = env.call_tool("get_issue", {"issue_id": 1})
    expected = copy.deepcopy(obs)
    obs["data"]["issue"]["labels"].append("security")
    obs["data"]["issue"]["title"] = "tampered"
    obs["data"]["comments"].clear()
    assert env.records[0].observation() == expected
    assert env.records[0].result == expected["data"]
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.call_tool("get_issue", {"issue_id": 1}) == expected


def test_mutating_write_observation_does_not_affect_env():
    env = _new_env()
    obs = env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    expected = copy.deepcopy(obs)
    obs["data"]["issue"]["title"] = "tampered"
    obs["data"]["issue"]["labels"].clear()
    assert env.records[0].observation() == expected
    assert env.current_state().to_dict() == _world({1: {"priority": "p0"}})


def test_mutating_error_observation_does_not_affect_records():
    env = _new_env()
    obs = env.call_tool("get_issue", {"issue_id": 99})
    expected = _tool_error("get_issue", {"issue_id": 99})
    obs["error"]["code"] = "tampered"
    obs["error"]["message"] = "tampered"
    assert env.records[0].error == expected
    assert env.records[0].observation() == {"ok": False, "error": expected}


def test_records_are_deep_copies():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    env.call_tool("add_comment", {"issue_id": 2, "body": "On it."})
    recs = env.records
    recs[0].result["issue"]["title"] = "tampered"
    recs[0].result["comments"].clear()
    recs[0].args["issue_id"] = 2
    recs[1].changes[0].after["body"] = "tampered"
    recs[1].result["comment"]["body"] = "tampered"
    first, second = env.records
    assert first.args == {"issue_id": 1}
    assert first.result == {"issue": _issue(1), "comments": [_comment(1)]}
    assert second.result == {"comment": NEW_COMMENT}
    assert second.changes == (Change(2, "comments", None, NEW_COMMENT),)
    assert env.current_state().comments[3].body == "On it."


def test_records_property_returns_tuple_of_tool_call_records():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    env.call_tool("get_issue", {"issue_id": 99})
    recs = env.records
    assert isinstance(recs, tuple)
    assert len(recs) == 2
    assert all(isinstance(r, ToolCallRecord) for r in recs)


def test_record_observation_returns_deep_copy_from_env_record():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    rec = env.records[0]
    obs = rec.observation()
    obs["data"]["issue"]["labels"].append("security")
    assert rec.result["issue"]["labels"] == ["bug", "ui"]


def test_current_state_returns_independent_copy():
    env = _new_env()
    s = env.current_state()
    s.issues.pop(1)
    s.comments.clear()
    s.users["u1"] = replace(s.users["u1"], active=False)
    s.issues[2] = replace(s.issues[2], priority="p0")
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD)


def test_current_state_returns_new_object_each_call():
    env = _new_env()
    a, b = env.current_state(), env.current_state()
    assert a is not b
    assert a == b
    assert a.issues is not b.issues


def test_current_state_is_not_affected_by_later_calls():
    env = _new_env()
    s = env.current_state()
    env.call_tool("update_issue", {"issue_id": 1, "priority": "p0"})
    env.call_tool("add_comment", {"issue_id": 1, "body": "note"})
    assert s.issues[1].priority == "p1"
    assert set(s.comments) == {1, 2}
    assert env.current_state().issues[1].priority == "p0"


# ---------------------------------------------------------------- 隨機序列（固定 seed）


def _random_call(rng: random.Random) -> tuple[str, Any]:
    kind = rng.choice(["get", "search", "update", "comment", "assign", "unknown", "non_dict"])
    issue_id = rng.choice([1, 2, 3, 4, 99])
    if kind == "get":
        return "get_issue", {"issue_id": issue_id}
    if kind == "search":
        args: dict[str, Any] = {}
        if rng.random() < 0.5:
            args["status"] = rng.choice(STATUSES)
        if rng.random() < 0.3:
            args["label"] = rng.choice(["bug", "ui", "security", "nope"])
        return "search_issues", args
    if kind == "update":
        args = {"issue_id": issue_id}
        if rng.random() < 0.5:
            args["priority"] = rng.choice(PRIORITIES)
        if rng.random() < 0.5:
            args["status"] = rng.choice(STATUSES)
        if rng.random() < 0.4:
            args["labels"] = rng.sample(["bug", "ui", "security"], rng.randint(0, 3))
        return "update_issue", args
    if kind == "comment":
        return "add_comment", {"issue_id": issue_id, "body": f"note {rng.randint(0, 9)}"}
    if kind == "assign":
        return "assign_issue", {"issue_id": issue_id, "assignee_id": rng.choice([None, "u1", "u2", "u3", "ghost"])}
    if kind == "unknown":
        return "close_issue", {"issue_id": issue_id}
    return "get_issue", rng.choice([[issue_id], str(issue_id), issue_id, None])


def _random_calls(seed: int, n: int = 60) -> list[tuple[str, Any]]:
    rng = random.Random(seed)
    return [_random_call(rng) for _ in range(n)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_call_sequence_invariants(seed):
    env = _new_env()
    prev_hash = env.state_hash()
    for i, (name, args) in enumerate(_random_calls(seed)):
        obs = env.call_tool(name, copy.deepcopy(args))
        assert env.step == i + 1
        rec = env.records[-1]
        assert len(env.records) == i + 1
        assert rec.step == i
        assert rec.tool == name
        assert rec.args == args
        assert rec.observation() == obs
        assert rec.state_hash_before == prev_hash
        assert rec.state_hash_after == env.state_hash()
        if not rec.ok or name in ("get_issue", "search_issues"):
            assert rec.changes == ()
        if rec.changes:
            assert rec.state_hash_before != rec.state_hash_after
        else:
            assert rec.state_hash_before == rec.state_hash_after
        assert ToolCallRecord.from_dict(json.loads(canonical_json(rec.to_dict()))) == rec
        prev_hash = rec.state_hash_after
    assert env.state_hash() == state_hash_of(env.current_state(), SMALL_ACTOR)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_call_sequence_is_deterministic(seed):
    calls = _random_calls(seed)
    a, b = _new_env(), _new_env()
    for name, args in calls:
        assert a.call_tool(name, copy.deepcopy(args)) == b.call_tool(name, copy.deepcopy(args))
    assert a.records == b.records
    assert [canonical_json(r.to_dict()) for r in a.records] == [canonical_json(r.to_dict()) for r in b.records]
    assert a.state_hash() == b.state_hash()


@pytest.mark.parametrize("data", [None, [], "change", 1], ids=["none", "list", "str", "int"])
def test_change_from_dict_rejects_non_dict(data):
    with pytest.raises(ValueError):
        Change.from_dict(data)


@pytest.mark.parametrize("data", [None, [], "record", 1], ids=["none", "list", "str", "int"])
def test_record_from_dict_rejects_non_dict(data):
    with pytest.raises(ValueError):
        ToolCallRecord.from_dict(data)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.update(changes=None), id="changes_none"),
        pytest.param(lambda d: d.update(changes={}), id="changes_dict"),
        pytest.param(lambda d: d.update(changes="x"), id="changes_str"),
        pytest.param(lambda d: d["changes"].__setitem__(0, "not-a-dict"), id="change_element_str"),
        pytest.param(lambda d: d["changes"].__setitem__(0, None), id="change_element_none"),
    ],
)
def test_record_from_dict_rejects_malformed_changes(mutate):
    data = RECORD_DICT_FACTORIES["mutating"]()
    mutate(data)
    with pytest.raises(ValueError):
        ToolCallRecord.from_dict(data)


# ---------------------------------------------------------------- last_records（主 session 補充，供 runner 使用）


def test_last_records_returns_last_n_in_order():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    env.call_tool("add_comment", {"issue_id": 1, "body": "x"})
    env.call_tool("get_issue", {"issue_id": 99})
    all_records = env.records
    assert env.last_records(0) == ()
    assert env.last_records(2) == all_records[1:]
    assert env.last_records(3) == all_records
    assert [r.step for r in env.last_records(2)] == [1, 2]


def test_last_records_are_deep_copies():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    (record,) = env.last_records(1)
    record.result["issue"]["title"] = "tampered"
    record.args["issue_id"] = 2
    assert env.last_records(1)[0].result["issue"]["title"] == "Crash on login"
    assert env.last_records(1)[0].args == {"issue_id": 1}


@pytest.mark.parametrize("count", [-1, 2, True, 1.0, "1", None])
def test_last_records_rejects_invalid_count(count):
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    with pytest.raises(ValueError):
        env.last_records(count)


def test_last_records_counts_since_restore():
    env = _new_env()
    env.call_tool("get_issue", {"issue_id": 1})
    snap = env.snapshot()
    env.restore(snap)
    assert env.last_records(0) == ()
    with pytest.raises(ValueError):
        env.last_records(1)
    env.call_tool("get_issue", {"issue_id": 2})
    (record,) = env.last_records(1)
    assert record.step == 1


def test_last_records_requires_initialized_env():
    with pytest.raises(RuntimeError):
        IssueTrackerEnv().last_records(0)
