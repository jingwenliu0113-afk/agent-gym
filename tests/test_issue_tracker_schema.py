"""apps/issue_tracker/schema.py 的契約測試（decision log 001）。"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from typing import Any, Callable

import pytest

from apps.issue_tracker.schema import (
    MUTABLE_ISSUE_FIELDS,
    PRIORITIES,
    SCHEMA_VERSION,
    STATUSES,
    UNASSIGNED,
    Comment,
    Issue,
    IssueTrackerState,
    Label,
    SchemaError,
    User,
)
from tests.conftest import SMALL_WORLD

USER_FIELDS = ("id", "name", "team", "active")
LABEL_FIELDS = ("name", "description")
ISSUE_FIELDS = (
    "id",
    "title",
    "body",
    "status",
    "priority",
    "assignee_id",
    "reporter_id",
    "labels",
    "created_at",
)
COMMENT_FIELDS = ("id", "issue_id", "author_id", "body")
TOP_LEVEL_KEYS = ("users", "labels", "issues", "comments")

# canonical 形式；id 刻意讓字典序與數值序不同，用來檢查排序方式。
RICH_WORLD: dict[str, Any] = {
    "users": [
        {"id": "u10", "name": "Ten", "team": "backend", "active": True},
        {"id": "u2", "name": "Two", "team": "frontend", "active": False},
        {"id": "u9", "name": "Nine", "team": "infra", "active": True},
    ],
    "labels": [
        {"name": "alpha", "description": "A"},
        {"name": "beta-10", "description": "B10"},
        {"name": "beta-2", "description": "B2"},
        {"name": "zeta", "description": "Z"},
    ],
    "issues": [
        {
            "id": 2,
            "title": "Two",
            "body": "second",
            "status": "resolved",
            "priority": "p0",
            "assignee_id": "u9",
            "reporter_id": "u10",
            "labels": ["alpha", "beta-10", "beta-2", "zeta"],
            "created_at": "2026-01-02",
        },
        {
            "id": 9,
            "title": "Nine",
            "body": "",
            "status": "open",
            "priority": "p3",
            "assignee_id": None,
            "reporter_id": "u2",
            "labels": ["zeta"],
            "created_at": "2026-01-09",
        },
        {
            "id": 10,
            "title": "十",
            "body": "ten",
            "status": "closed",
            "priority": "p2",
            "assignee_id": "u2",
            "reporter_id": "u2",
            "labels": [],
            "created_at": "2026-01-10",
        },
    ],
    "comments": [
        {"id": 3, "issue_id": 10, "author_id": "u9", "body": "c3"},
        {"id": 11, "issue_id": 2, "author_id": "u10", "body": "c11"},
        {"id": 20, "issue_id": 2, "author_id": "u2", "body": "c20"},
    ],
}


def _assert_json_native(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _assert_json_native(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"non-str key at {path}: {k!r}"
            _assert_json_native(v, f"{path}.{k}")
        return
    pytest.fail(f"non JSON-native value at {path}: {type(value).__name__}")


def _shuffled_world(world: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """打亂 collection 順序、issue labels 順序與每個 dict 的 key 插入順序。"""
    out: dict[str, Any] = {}
    keys = list(world)
    rng.shuffle(keys)
    for key in keys:
        entities = []
        for entity in world[key]:
            fields = list(entity)
            rng.shuffle(fields)
            new_entity = {}
            for f in fields:
                v = entity[f]
                if f == "labels" and key == "issues":
                    v = list(v)
                    rng.shuffle(v)
                new_entity[f] = v
            entities.append(new_entity)
        rng.shuffle(entities)
        out[key] = entities
    return out


# ---------------------------------------------------------------------------
# 常數與 dataclass
# ---------------------------------------------------------------------------


def test_schema_version_constant():
    assert SCHEMA_VERSION == "issue_tracker/v1"


def test_unassigned_constant():
    assert UNASSIGNED == "unassigned"


def test_statuses_constant():
    assert STATUSES == ("open", "in_progress", "resolved", "closed")


def test_priorities_constant():
    assert PRIORITIES == ("p0", "p1", "p2", "p3")


def test_mutable_issue_fields_constant():
    assert MUTABLE_ISSUE_FIELDS == (
        "title",
        "body",
        "status",
        "priority",
        "assignee_id",
        "labels",
    )


def test_mutable_issue_fields_exclude_reporter_and_created_at():
    assert "reporter_id" not in MUTABLE_ISSUE_FIELDS
    assert "created_at" not in MUTABLE_ISSUE_FIELDS
    assert "id" not in MUTABLE_ISSUE_FIELDS


def test_schema_error_is_value_error():
    assert issubclass(SchemaError, ValueError)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (User, USER_FIELDS),
        (Label, LABEL_FIELDS),
        (Issue, ISSUE_FIELDS),
        (Comment, COMMENT_FIELDS),
    ],
)
def test_entity_dataclass_fields(cls, expected):
    assert tuple(f.name for f in dataclasses.fields(cls)) == expected


@pytest.mark.parametrize(
    ("instance", "field"),
    [
        (User(id="u1", name="n", team="t", active=True), "name"),
        (Label(name="bug", description="d"), "description"),
        (
            Issue(
                id=1,
                title="t",
                body="b",
                status="open",
                priority="p1",
                assignee_id=None,
                reporter_id="u1",
                labels=("bug",),
                created_at="2026-01-01",
            ),
            "status",
        ),
        (Comment(id=1, issue_id=1, author_id="u1", body="b"), "body"),
    ],
)
def test_entities_are_frozen(instance, field):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, "changed")


# ---------------------------------------------------------------------------
# from_dict：合法輸入
# ---------------------------------------------------------------------------


def test_from_dict_returns_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert isinstance(state, IssueTrackerState)


def test_from_dict_keys_users_by_id(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert set(state.users) == {"bot", "u1", "u2", "u3"}
    assert state.users["u2"] == User(id="u2", name="Ian Inactive", team="backend", active=False)


def test_from_dict_keys_labels_by_name(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert set(state.labels) == {"bug", "security", "ui"}
    assert state.labels["security"] == Label(name="security", description="Security relevant")


def test_from_dict_keys_issues_by_int_id(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert set(state.issues) == {1, 2, 3, 4}
    assert state.issues[1] == Issue(
        id=1,
        title="Crash on login",
        body="App crashes when pressing the Login button.",
        status="open",
        priority="p1",
        assignee_id=None,
        reporter_id="u1",
        labels=("bug", "ui"),
        created_at="2026-09-01",
    )


def test_from_dict_keys_comments_by_int_id(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert set(state.comments) == {1, 2}
    assert state.comments[2] == Comment(
        id=2, issue_id=3, author_id="u1", body="Please rotate the tokens."
    )


def test_from_dict_entity_values_have_entity_types(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert all(type(u) is User for u in state.users.values())
    assert all(type(lb) is Label for lb in state.labels.values())
    assert all(type(i) is Issue for i in state.issues.values())
    assert all(type(c) is Comment for c in state.comments.values())


def test_from_dict_issue_labels_are_tuples(small_world):
    state = IssueTrackerState.from_dict(small_world)
    for issue in state.issues.values():
        assert type(issue.labels) is tuple
    assert state.issues[4].labels == ()


def test_from_dict_keeps_non_ascii_text(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert state.issues[2].title == "登入頁面字型錯誤"


def test_from_dict_allows_inactive_assignee(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert state.issues[3].assignee_id == "u2"
    assert state.users["u2"].active is False


def test_from_dict_accepts_empty_world():
    data = {"users": [], "labels": [], "issues": [], "comments": []}
    state = IssueTrackerState.from_dict(data)
    assert state.users == {}
    assert state.labels == {}
    assert state.issues == {}
    assert state.comments == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["ui", "bug"], ("bug", "ui")),
        (["ui", "bug", "ui"], ("bug", "ui")),
        (["security", "security"], ("security",)),
        (["ui", "security", "bug", "bug", "ui"], ("bug", "security", "ui")),
        ([], ()),
    ],
)
def test_from_dict_normalizes_issue_labels(small_world, raw, expected):
    small_world["issues"][0]["labels"] = raw
    state = IssueTrackerState.from_dict(small_world)
    assert state.issues[1].labels == expected
    assert state.to_dict()["issues"][0]["labels"] == list(expected)


def test_from_dict_order_of_input_collections_is_irrelevant(small_world):
    expected = IssueTrackerState.from_dict(copy.deepcopy(SMALL_WORLD))
    rng = random.Random(20261008)
    for _ in range(10):
        shuffled = _shuffled_world(small_world, rng)
        assert IssueTrackerState.from_dict(shuffled) == expected


def test_from_dict_does_not_mutate_input(small_world):
    small_world["issues"][0]["labels"] = ["ui", "bug", "ui"]
    small_world["users"].reverse()
    before = copy.deepcopy(small_world)
    IssueTrackerState.from_dict(small_world)
    assert small_world == before
    assert small_world["issues"][0]["labels"] == ["ui", "bug", "ui"]


def test_from_dict_state_does_not_share_labels_list(small_world):
    state = IssueTrackerState.from_dict(small_world)
    raw_labels = small_world["issues"][0]["labels"]
    assert state.issues[1].labels is not raw_labels
    raw_labels.append("security")
    assert state.issues[1].labels == ("bug", "ui")


def test_from_dict_state_unaffected_by_later_input_mutation(small_world):
    state = IssueTrackerState.from_dict(small_world)
    expected = copy.deepcopy(SMALL_WORLD)
    small_world["issues"][0]["labels"].append("security")
    small_world["issues"][1]["title"] = "changed"
    small_world["users"].append({"id": "u9", "name": "N", "team": "t", "active": True})
    small_world["comments"].clear()
    small_world["labels"][0]["description"] = "changed"
    assert state.to_dict() == expected


@pytest.mark.parametrize("bad_input", [False, True], ids=["valid", "invalid"])
def test_from_dict_on_error_does_not_mutate_input(small_world, bad_input):
    small_world["issues"][0]["labels"] = ["ui", "bug", "ui"]
    if bad_input:
        small_world["comments"][0]["issue_id"] = 99
    before = copy.deepcopy(small_world)
    if bad_input:
        with pytest.raises(SchemaError):
            IssueTrackerState.from_dict(small_world)
    else:
        IssueTrackerState.from_dict(small_world)
    assert small_world == before


# ---------------------------------------------------------------------------
# from_dict：SchemaError
# ---------------------------------------------------------------------------


def _set(collection: str, index: int, field: str, value: Any) -> Callable[[dict], dict]:
    def mutate(data: dict) -> dict:
        data[collection][index][field] = value
        return data

    return mutate


def _delete(collection: str, index: int, field: str) -> Callable[[dict], dict]:
    def mutate(data: dict) -> dict:
        del data[collection][index][field]
        return data

    return mutate


def _append(collection: str, entity: Any) -> Callable[[dict], dict]:
    def mutate(data: dict) -> dict:
        data[collection].append(copy.deepcopy(entity))
        return data

    return mutate


def _replace_entity(collection: str, index: int, value: Any) -> Callable[[dict], dict]:
    def mutate(data: dict) -> dict:
        data[collection][index] = value
        return data

    return mutate


def _drop_top(key: str) -> Callable[[dict], dict]:
    def mutate(data: dict) -> dict:
        del data[key]
        return data

    return mutate


def _add_top(key: str, value: Any) -> Callable[[dict], dict]:
    def mutate(data: dict) -> dict:
        data[key] = value
        return data

    return mutate


def _new_label(name: Any = "extra", description: Any = "extra label") -> dict:
    return {"name": name, "description": description}


# 會被修改的 entity：users[0]=bot、issues[3]=#4、comments[1]=#2 都不被其他 entity 參照，
# 型別錯誤不會順帶造成參照錯誤；label 型別錯誤則新增一個未被參照的 label。
_ENTITY_INDEX = {"users": 0, "labels": 0, "issues": 3, "comments": 1}
_ENTITY_FIELDS = {
    "users": USER_FIELDS,
    "labels": LABEL_FIELDS,
    "issues": ISSUE_FIELDS,
    "comments": COMMENT_FIELDS,
}

TOP_LEVEL_CASES = [
    *(pytest.param(_drop_top(k), id=f"missing_top_level_{k}") for k in TOP_LEVEL_KEYS),
    pytest.param(_add_top("schema_version", SCHEMA_VERSION), id="extra_top_level_schema_version"),
    pytest.param(_add_top("next_comment_id", 3), id="extra_top_level_next_comment_id"),
    pytest.param(_add_top("extra", []), id="extra_top_level_extra"),
]

MISSING_FIELD_CASES = [
    pytest.param(_delete(coll, _ENTITY_INDEX[coll], f), id=f"missing_{coll}_{f}")
    for coll, fields in _ENTITY_FIELDS.items()
    for f in fields
]

EXTRA_FIELD_CASES = [
    pytest.param(_set("users", 0, "email", "bot@example.com"), id="extra_user_field"),
    pytest.param(_set("labels", 0, "color", "red"), id="extra_label_field"),
    pytest.param(_set("issues", 3, "updated_at", "2026-09-05"), id="extra_issue_updated_at"),
    pytest.param(_set("issues", 3, "comments", []), id="extra_issue_comments"),
    pytest.param(_set("comments", 1, "created_at", "2026-09-05"), id="extra_comment_field"),
]

ENTITY_NOT_DICT_CASES = [
    pytest.param(_replace_entity("users", 0, "bot"), id="user_is_str"),
    pytest.param(_replace_entity("users", 0, ["bot", "Test Bot"]), id="user_is_list"),
    pytest.param(_replace_entity("labels", 0, "bug"), id="label_is_str"),
    pytest.param(_replace_entity("issues", 3, None), id="issue_is_none"),
    pytest.param(_replace_entity("issues", 3, [4]), id="issue_is_list"),
    pytest.param(_replace_entity("comments", 1, 2), id="comment_is_int"),
    pytest.param(_replace_entity("comments", 1, ("id", 2)), id="comment_is_tuple"),
]

USER_TYPE_CASES = [
    pytest.param(_set("users", 0, "id", 7), id="user_id_int"),
    pytest.param(_set("users", 0, "id", None), id="user_id_none"),
    pytest.param(_set("users", 0, "id", ["bot"]), id="user_id_unhashable_list"),
    pytest.param(_set("users", 0, "name", None), id="user_name_none"),
    pytest.param(_set("users", 0, "name", 1), id="user_name_int"),
    pytest.param(_set("users", 0, "team", None), id="user_team_none"),
    pytest.param(_set("users", 0, "team", 1), id="user_team_int"),
    pytest.param(_set("users", 0, "active", 1), id="user_active_int_1"),
    pytest.param(_set("users", 0, "active", 0), id="user_active_int_0"),
    pytest.param(_set("users", 0, "active", "true"), id="user_active_str"),
    pytest.param(_set("users", 0, "active", None), id="user_active_none"),
]

LABEL_TYPE_CASES = [
    pytest.param(_append("labels", _new_label(name=5)), id="label_name_int"),
    pytest.param(_append("labels", _new_label(name=None)), id="label_name_none"),
    pytest.param(_append("labels", _new_label(name=["x"])), id="label_name_unhashable_list"),
    pytest.param(_append("labels", _new_label(description=None)), id="label_description_none"),
    pytest.param(_append("labels", _new_label(description=1)), id="label_description_int"),
]

ISSUE_TYPE_CASES = [
    pytest.param(_set("issues", 3, "id", "4"), id="issue_id_str"),
    pytest.param(_set("issues", 3, "id", 4.0), id="issue_id_float"),
    pytest.param(_set("issues", 3, "id", None), id="issue_id_none"),
    pytest.param(_set("issues", 3, "id", [4]), id="issue_id_unhashable_list"),
    # issue #1 改成 True：True == 1，comment 仍能「找到」它，只有 bool 檢查能擋下。
    pytest.param(_set("issues", 0, "id", True), id="issue_id_bool"),
    pytest.param(_set("issues", 3, "title", None), id="issue_title_none"),
    pytest.param(_set("issues", 3, "title", 1), id="issue_title_int"),
    pytest.param(_set("issues", 3, "body", None), id="issue_body_none"),
    pytest.param(_set("issues", 3, "body", 1), id="issue_body_int"),
    pytest.param(_set("issues", 3, "status", None), id="issue_status_none"),
    pytest.param(_set("issues", 3, "status", 1), id="issue_status_int"),
    pytest.param(_set("issues", 3, "status", ["closed"]), id="issue_status_list"),
    pytest.param(_set("issues", 3, "priority", None), id="issue_priority_none"),
    pytest.param(_set("issues", 3, "priority", 3), id="issue_priority_int"),
    pytest.param(_set("issues", 3, "assignee_id", 1), id="issue_assignee_int"),
    pytest.param(_set("issues", 3, "assignee_id", False), id="issue_assignee_bool"),
    pytest.param(_set("issues", 3, "assignee_id", ["u2"]), id="issue_assignee_unhashable_list"),
    pytest.param(_set("issues", 3, "reporter_id", None), id="issue_reporter_none"),
    pytest.param(_set("issues", 3, "reporter_id", 1), id="issue_reporter_int"),
    pytest.param(_set("issues", 3, "reporter_id", ["u2"]), id="issue_reporter_unhashable_list"),
    pytest.param(_set("issues", 3, "labels", None), id="issue_labels_none"),
    pytest.param(_set("issues", 3, "labels", "bug"), id="issue_labels_str"),
    pytest.param(_set("issues", 3, "labels", {"bug": 1}), id="issue_labels_dict"),
    pytest.param(_set("issues", 3, "labels", [1]), id="issue_labels_item_int"),
    pytest.param(_set("issues", 3, "labels", ["bug", None]), id="issue_labels_item_none"),
    pytest.param(_set("issues", 3, "labels", ["bug", ["ui"]]), id="issue_labels_item_list"),
    pytest.param(_set("issues", 3, "created_at", None), id="issue_created_at_none"),
    pytest.param(_set("issues", 3, "created_at", 20260801), id="issue_created_at_int"),
]

COMMENT_TYPE_CASES = [
    pytest.param(_set("comments", 1, "id", "2"), id="comment_id_str"),
    pytest.param(_set("comments", 1, "id", 2.0), id="comment_id_float"),
    pytest.param(_set("comments", 1, "id", None), id="comment_id_none"),
    pytest.param(_set("comments", 1, "id", [2]), id="comment_id_unhashable_list"),
    pytest.param(_set("comments", 0, "id", True), id="comment_id_bool"),
    pytest.param(_set("comments", 1, "issue_id", "3"), id="comment_issue_id_str"),
    pytest.param(_set("comments", 1, "issue_id", 3.0), id="comment_issue_id_float"),
    pytest.param(_set("comments", 1, "issue_id", None), id="comment_issue_id_none"),
    pytest.param(_set("comments", 1, "issue_id", [3]), id="comment_issue_id_unhashable_list"),
    # comment #1 的 issue_id 1 → True：True == 1 仍能對應 issue #1，只有 bool 檢查能擋下。
    pytest.param(_set("comments", 0, "issue_id", True), id="comment_issue_id_bool"),
    pytest.param(_set("comments", 1, "author_id", None), id="comment_author_none"),
    pytest.param(_set("comments", 1, "author_id", 1), id="comment_author_int"),
    pytest.param(_set("comments", 1, "author_id", ["u1"]), id="comment_author_unhashable_list"),
    pytest.param(_set("comments", 1, "body", None), id="comment_body_none"),
    pytest.param(_set("comments", 1, "body", 1), id="comment_body_int"),
]

ENUM_CASES = [
    pytest.param(_set("issues", 3, "status", "Open"), id="status_capitalized"),
    pytest.param(_set("issues", 3, "status", "CLOSED"), id="status_upper"),
    pytest.param(_set("issues", 3, "status", "done"), id="status_unknown"),
    pytest.param(_set("issues", 3, "status", ""), id="status_empty"),
    pytest.param(_set("issues", 3, "status", " open"), id="status_whitespace"),
    pytest.param(_set("issues", 3, "status", "in-progress"), id="status_dash"),
    pytest.param(_set("issues", 3, "priority", "P1"), id="priority_upper"),
    pytest.param(_set("issues", 3, "priority", "p4"), id="priority_out_of_range"),
    pytest.param(_set("issues", 3, "priority", "high"), id="priority_unknown"),
    pytest.param(_set("issues", 3, "priority", ""), id="priority_empty"),
]

DUPLICATE_CASES = [
    pytest.param(
        _append("users", {"id": "u1", "name": "Other Ann", "team": "x", "active": True}),
        id="duplicate_user_id",
    ),
    pytest.param(
        _append("labels", {"name": "bug", "description": "duplicate"}), id="duplicate_label_name"
    ),
    pytest.param(
        _append(
            "issues",
            {
                "id": 2,
                "title": "Duplicate",
                "body": "",
                "status": "open",
                "priority": "p1",
                "assignee_id": None,
                "reporter_id": "u1",
                "labels": [],
                "created_at": "2026-09-09",
            },
        ),
        id="duplicate_issue_id",
    ),
    pytest.param(
        _append("comments", {"id": 1, "issue_id": 3, "author_id": "u3", "body": "dup"}),
        id="duplicate_comment_id",
    ),
    pytest.param(
        _append("users", copy.deepcopy(SMALL_WORLD["users"][1])), id="duplicate_user_identical"
    ),
]

DANGLING_CASES = [
    pytest.param(_set("issues", 3, "assignee_id", "ghost"), id="dangling_assignee"),
    pytest.param(_set("issues", 3, "assignee_id", UNASSIGNED), id="dangling_assignee_unassigned"),
    pytest.param(_set("issues", 3, "reporter_id", "ghost"), id="dangling_reporter"),
    pytest.param(_set("comments", 1, "author_id", "ghost"), id="dangling_comment_author"),
    pytest.param(_set("issues", 3, "labels", ["ghost"]), id="dangling_label"),
    pytest.param(_set("issues", 0, "labels", ["bug", "ui", "ghost"]), id="dangling_label_among_valid"),
    pytest.param(_set("issues", 3, "labels", ["Bug"]), id="dangling_label_case_mismatch"),
    pytest.param(_set("comments", 1, "issue_id", 99), id="dangling_comment_issue"),
]

RESERVED_CASES = [
    pytest.param(
        _append("users", {"id": UNASSIGNED, "name": "Nobody", "team": "none", "active": True}),
        id="reserved_user_id_unassigned",
    ),
]

ALL_INVALID_CASES = (
    TOP_LEVEL_CASES
    + MISSING_FIELD_CASES
    + EXTRA_FIELD_CASES
    + ENTITY_NOT_DICT_CASES
    + USER_TYPE_CASES
    + LABEL_TYPE_CASES
    + ISSUE_TYPE_CASES
    + COMMENT_TYPE_CASES
    + ENUM_CASES
    + DUPLICATE_CASES
    + DANGLING_CASES
    + RESERVED_CASES
)


@pytest.mark.parametrize("mutate", ALL_INVALID_CASES)
def test_from_dict_rejects_invalid_data(small_world, mutate):
    data = mutate(small_world)
    with pytest.raises(SchemaError):
        IssueTrackerState.from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty_list"),
        pytest.param("users", id="str"),
        pytest.param(42, id="int"),
        pytest.param([("users", []), ("labels", []), ("issues", []), ("comments", [])], id="pairs"),
    ],
)
def test_from_dict_rejects_non_dict_data(data):
    with pytest.raises(SchemaError):
        IssueTrackerState.from_dict(data)


# collection 依 canonical 格式必須是 list；非 list 時天真的迭代會漏出 TypeError。
_BAD_COLLECTIONS = [
    pytest.param(None, id="none"),
    pytest.param({}, id="empty_dict"),
    pytest.param("x", id="str"),
    pytest.param(5, id="int"),
    pytest.param({"1": SMALL_WORLD["issues"][0]}, id="dict_of_entity"),
]


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
@pytest.mark.parametrize("bad", _BAD_COLLECTIONS)
def test_from_dict_rejects_non_list_collection(small_world, key, bad):
    small_world[key] = copy.deepcopy(bad)
    with pytest.raises(SchemaError):
        IssueTrackerState.from_dict(small_world)


_FUZZ_JUNK: tuple[Any, ...] = (None, True, 0, 1.5, "", "x", [], {}, [None], {"k": 1})


def _paths(obj: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        return out
    for k, v in items:
        out.append(prefix + (k,))
        out.extend(_paths(v, prefix + (k,)))
    return out


def _fuzz_mutate(rng: random.Random, data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    d = copy.deepcopy(data)
    path = rng.choice(_paths(d))
    parent: Any = d
    for key in path[:-1]:
        parent = parent[key]
    last = path[-1]
    action = rng.choice(("replace", "delete", "extra"))
    if action == "replace":
        parent[last] = copy.deepcopy(rng.choice(_FUZZ_JUNK))
    elif action == "delete":
        del parent[last]
    else:
        target = parent[last]
        if isinstance(target, dict):
            target["unexpected"] = 1
        elif isinstance(target, list):
            target.append(copy.deepcopy(rng.choice(_FUZZ_JUNK)))
        else:
            parent[last] = [target]
    return d, f"{action} {path!r}"


def test_from_dict_only_raises_schema_error_on_random_corruption():
    """隨機破壞合法 state：from_dict 只能成功或 raise SchemaError，不得漏出其他例外。"""
    rng = random.Random(20261008)
    for _ in range(400):
        mutated, what = _fuzz_mutate(rng, SMALL_WORLD)
        before = copy.deepcopy(mutated)
        try:
            state = IssueTrackerState.from_dict(mutated)
        except SchemaError:
            assert mutated == before, what
            continue
        except Exception as e:  # pragma: no cover - 失敗時給出可重現資訊
            raise AssertionError(f"{what}: leaked {type(e).__name__}: {e}") from e
        assert mutated == before, what
        assert IssueTrackerState.from_dict(state.to_dict()) == state, what


def test_from_dict_valid_mutation_baselines_are_accepted(small_world):
    """確認上面挑選的 entity 本身合法，錯誤只來自各案例的修改。"""
    small_world["labels"].append(_new_label())
    state = IssueTrackerState.from_dict(small_world)
    assert "extra" in state.labels


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_canonical_small_world(small_world):
    assert IssueTrackerState.from_dict(small_world).to_dict() == SMALL_WORLD


def test_to_dict_round_trips_canonical_rich_world():
    data = copy.deepcopy(RICH_WORLD)
    assert IssueTrackerState.from_dict(data).to_dict() == RICH_WORLD


def test_to_dict_top_level_keys(small_world):
    d = IssueTrackerState.from_dict(small_world).to_dict()
    assert set(d) == set(TOP_LEVEL_KEYS)
    for key in TOP_LEVEL_KEYS:
        assert type(d[key]) is list


@pytest.mark.parametrize(
    ("collection", "fields"),
    [
        ("users", USER_FIELDS),
        ("labels", LABEL_FIELDS),
        ("issues", ISSUE_FIELDS),
        ("comments", COMMENT_FIELDS),
    ],
)
def test_to_dict_entity_keys(small_world, collection, fields):
    d = IssueTrackerState.from_dict(small_world).to_dict()
    assert d[collection]
    for entity in d[collection]:
        assert set(entity) == set(fields)


def test_to_dict_users_sorted_by_id():
    d = IssueTrackerState.from_dict(copy.deepcopy(RICH_WORLD)).to_dict()
    assert [u["id"] for u in d["users"]] == ["u10", "u2", "u9"]


def test_to_dict_labels_sorted_by_name():
    d = IssueTrackerState.from_dict(copy.deepcopy(RICH_WORLD)).to_dict()
    assert [lb["name"] for lb in d["labels"]] == ["alpha", "beta-10", "beta-2", "zeta"]


def test_to_dict_issues_sorted_by_numeric_id():
    d = IssueTrackerState.from_dict(copy.deepcopy(RICH_WORLD)).to_dict()
    assert [i["id"] for i in d["issues"]] == [2, 9, 10]


def test_to_dict_comments_sorted_by_numeric_id():
    d = IssueTrackerState.from_dict(copy.deepcopy(RICH_WORLD)).to_dict()
    assert [c["id"] for c in d["comments"]] == [3, 11, 20]


@pytest.mark.parametrize("world", [SMALL_WORLD, RICH_WORLD], ids=["small", "rich"])
def test_to_dict_is_canonical_for_shuffled_input(world):
    rng = random.Random(1008)
    for _ in range(20):
        shuffled = _shuffled_world(copy.deepcopy(world), rng)
        assert IssueTrackerState.from_dict(shuffled).to_dict() == world


def test_to_dict_issue_labels_sorted_unique_lists(small_world):
    small_world["issues"][2]["labels"] = ["security", "bug", "security"]
    d = IssueTrackerState.from_dict(small_world).to_dict()
    issue3 = next(i for i in d["issues"] if i["id"] == 3)
    assert issue3["labels"] == ["bug", "security"]
    for issue in d["issues"]:
        assert type(issue["labels"]) is list
        assert issue["labels"] == sorted(set(issue["labels"]))


def test_to_dict_preserves_none_assignee_and_bools(small_world):
    d = IssueTrackerState.from_dict(small_world).to_dict()
    assert d["issues"][0]["assignee_id"] is None
    assert d["users"][2]["active"] is False
    assert d["users"][1]["active"] is True


def test_to_dict_orders_directly_constructed_state():
    users = {
        "u9": User(id="u9", name="Nine", team="t", active=True),
        "u10": User(id="u10", name="Ten", team="t", active=True),
        "u2": User(id="u2", name="Two", team="t", active=False),
    }
    labels = {
        "zeta": Label(name="zeta", description="z"),
        "alpha": Label(name="alpha", description="a"),
    }
    issues = {
        10: Issue(10, "ten", "", "open", "p1", None, "u2", ("alpha", "zeta"), "2026-01-10"),
        2: Issue(2, "two", "", "closed", "p3", "u9", "u10", (), "2026-01-02"),
        9: Issue(9, "nine", "", "resolved", "p0", "u2", "u9", ("zeta",), "2026-01-09"),
    }
    comments = {
        20: Comment(20, 10, "u2", "c20"),
        3: Comment(3, 2, "u9", "c3"),
        11: Comment(11, 9, "u10", "c11"),
    }
    d = IssueTrackerState(users=users, labels=labels, issues=issues, comments=comments).to_dict()
    assert [u["id"] for u in d["users"]] == ["u10", "u2", "u9"]
    assert [lb["name"] for lb in d["labels"]] == ["alpha", "zeta"]
    assert [i["id"] for i in d["issues"]] == [2, 9, 10]
    assert [c["id"] for c in d["comments"]] == [3, 11, 20]
    assert d["issues"][2]["labels"] == ["alpha", "zeta"]


@pytest.mark.parametrize("world", [SMALL_WORLD, RICH_WORLD], ids=["small", "rich"])
def test_to_dict_is_json_native(world):
    d = IssueTrackerState.from_dict(copy.deepcopy(world)).to_dict()
    _assert_json_native(d)


@pytest.mark.parametrize("world", [SMALL_WORLD, RICH_WORLD], ids=["small", "rich"])
def test_to_dict_json_round_trip_equality(world):
    d = IssueTrackerState.from_dict(copy.deepcopy(world)).to_dict()
    assert json.loads(json.dumps(d)) == d


def test_to_dict_returns_fresh_objects_each_call(small_world):
    state = IssueTrackerState.from_dict(small_world)
    a = state.to_dict()
    b = state.to_dict()
    assert a == b
    assert a is not b
    assert a["issues"] is not b["issues"]
    assert a["issues"][0] is not b["issues"][0]
    assert a["issues"][0]["labels"] is not b["issues"][0]["labels"]


def test_to_dict_mutation_does_not_affect_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    d = state.to_dict()
    d["issues"][0]["labels"].append("security")
    d["issues"][0]["title"] = "changed"
    d["users"].clear()
    d["comments"].append({"id": 9, "issue_id": 1, "author_id": "u1", "body": "x"})
    d["labels"][0]["description"] = "changed"
    del d["issues"][1]
    assert state.to_dict() == SMALL_WORLD
    assert state.issues[1].labels == ("bug", "ui")


def test_to_dict_does_not_mutate_state(small_world):
    state = IssueTrackerState.from_dict(small_world)
    before = state.copy()
    state.to_dict()
    assert state == before


def test_to_dict_of_empty_world():
    data = {"users": [], "labels": [], "issues": [], "comments": []}
    assert IssueTrackerState.from_dict(data).to_dict() == data


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", [SMALL_WORLD, RICH_WORLD], ids=["small", "rich"])
def test_from_dict_of_to_dict_round_trip(world):
    state = IssueTrackerState.from_dict(copy.deepcopy(world))
    assert IssueTrackerState.from_dict(state.to_dict()) == state


@pytest.mark.parametrize("world", [SMALL_WORLD, RICH_WORLD], ids=["small", "rich"])
def test_round_trip_through_json_text(world):
    state = IssueTrackerState.from_dict(copy.deepcopy(world))
    text = json.dumps(state.to_dict(), ensure_ascii=False)
    assert IssueTrackerState.from_dict(json.loads(text)) == state


def test_round_trip_of_directly_constructed_state():
    state = IssueTrackerState(
        users={"u1": User("u1", "Ann", "backend", True)},
        labels={"bug": Label("bug", "broken")},
        issues={7: Issue(7, "t", "b", "open", "p2", "u1", "u1", ("bug",), "2026-01-01")},
        comments={5: Comment(5, 7, "u1", "hi")},
    )
    assert IssueTrackerState.from_dict(state.to_dict()) == state


# ---------------------------------------------------------------------------
# next_comment_id
# ---------------------------------------------------------------------------


def test_next_comment_id_small_world(small_world):
    assert IssueTrackerState.from_dict(small_world).next_comment_id() == 3


def test_next_comment_id_is_one_when_no_comments(small_world):
    small_world["comments"] = []
    assert IssueTrackerState.from_dict(small_world).next_comment_id() == 1


def test_next_comment_id_empty_world():
    data = {"users": [], "labels": [], "issues": [], "comments": []}
    assert IssueTrackerState.from_dict(data).next_comment_id() == 1


def test_next_comment_id_non_contiguous_ids():
    assert IssueTrackerState.from_dict(copy.deepcopy(RICH_WORLD)).next_comment_id() == 21


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([1], 2),
        ([10], 11),
        ([2, 7, 5], 8),
        ([7, 2, 5], 8),
        ([100, 1], 101),
    ],
)
def test_next_comment_id_is_max_plus_one(small_world, ids, expected):
    small_world["comments"] = [
        {"id": cid, "issue_id": 1, "author_id": "u1", "body": f"c{cid}"} for cid in ids
    ]
    assert IssueTrackerState.from_dict(small_world).next_comment_id() == expected


def test_next_comment_id_is_derived_from_current_comments(small_world):
    state = IssueTrackerState.from_dict(small_world)
    state.comments[10] = Comment(id=10, issue_id=1, author_id="u1", body="x")
    assert state.next_comment_id() == 11
    del state.comments[10]
    assert state.next_comment_id() == 3
    state.comments.clear()
    assert state.next_comment_id() == 1


def test_next_comment_id_is_pure(small_world):
    state = IssueTrackerState.from_dict(small_world)
    assert state.next_comment_id() == state.next_comment_id() == 3
    assert state.to_dict() == SMALL_WORLD


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_equals_original(small_world):
    state = IssueTrackerState.from_dict(small_world)
    clone = state.copy()
    assert isinstance(clone, IssueTrackerState)
    assert clone == state
    assert clone is not state
    assert clone.to_dict() == state.to_dict()


def test_copy_does_not_share_containers(small_world):
    state = IssueTrackerState.from_dict(small_world)
    clone = state.copy()
    assert clone.users is not state.users
    assert clone.labels is not state.labels
    assert clone.issues is not state.issues
    assert clone.comments is not state.comments


def test_mutating_copy_does_not_affect_original(small_world):
    state = IssueTrackerState.from_dict(small_world)
    clone = state.copy()
    clone.issues[1] = dataclasses.replace(clone.issues[1], title="changed", labels=())
    clone.comments[3] = Comment(id=3, issue_id=1, author_id="bot", body="new")
    del clone.users["bot"]
    clone.labels.clear()
    assert state.to_dict() == SMALL_WORLD
    assert state.next_comment_id() == 3


def test_mutating_original_does_not_affect_copy(small_world):
    state = IssueTrackerState.from_dict(small_world)
    clone = state.copy()
    state.issues[2] = dataclasses.replace(state.issues[2], status="closed")
    state.comments.clear()
    state.users["u9"] = User(id="u9", name="N", team="t", active=True)
    assert clone.to_dict() == SMALL_WORLD
    assert clone.next_comment_id() == 3


def test_copy_of_empty_world():
    state = IssueTrackerState.from_dict({"users": [], "labels": [], "issues": [], "comments": []})
    clone = state.copy()
    assert clone == state
    clone.users["u1"] = User(id="u1", name="n", team="t", active=True)
    assert state.users == {}
