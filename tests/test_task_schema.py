"""tasks/schema.py 的契約測試（decision log 003）。"""

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
    STATUSES,
    Comment,
    Issue,
    IssueTrackerState,
    Label,
    User,
)
from env.environment import COMMENTS_FIELD, Change
from env.hashing import canonical_json
from tasks.schema import (
    FORBID_RULE_TYPES,
    PREDICATE_TYPES,
    SCALAR_PREDICATE_FIELDS,
    TASK_FORMAT,
    TASK_KEYS,
    ScopeEntry,
    Task,
    TaskError,
    change_in_scope,
    evaluate_predicate,
    forbid_rule_matches,
)
from tests.conftest import SMALL_ACTOR, SMALL_WORLD

SECTIONS = ("expected_outcome", "success_constraints", "forbidden", "mutation_scope")
F3_FIELDS = (*MUTABLE_ISSUE_FIELDS, COMMENTS_FIELD)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _task_dict() -> dict[str, Any]:
    """最小合法 task（Task.to_dict 形式），各測試自行修改。"""
    return {
        "format": TASK_FORMAT,
        "task_id": "test-assign",
        "family": "test/assign",
        "generator_version": "test/v1",
        "params": {},
        "instruction": "Assign issue #1 to Ann Active (u1).",
        "actor_id": SMALL_ACTOR,
        "initial_state": copy.deepcopy(SMALL_WORLD),
        "expected_outcome": [
            {"type": "issue_field_equals", "issue_id": 1, "field": "assignee_id", "value": "u1"},
        ],
        "success_constraints": [{"type": "issue_unchanged", "issue_id": 2}],
        "forbidden": [
            {"type": "forbid_change", "issue_ids": "*", "fields": ["status"], "to_values": ["closed"]},
        ],
        "mutation_scope": [{"issue_id": 1, "fields": ["assignee_id"]}],
    }


def _full_task_dict() -> dict[str, Any]:
    """涵蓋所有 predicate 型別與選填 key 的合法 task。"""
    d = _task_dict()
    d["params"] = {
        "seed": 7,
        "ratio": 0.5,
        "flag": True,
        "missing": None,
        "names": ["a", "b"],
        "nested": {"k": [1, {"x": "y"}]},
    }
    d["expected_outcome"] = [
        {"type": "issue_field_equals", "issue_id": 1, "field": "assignee_id", "value": "u1",
         "description": "assign #1"},
        {"type": "issue_field_in", "issue_id": 1, "field": "status", "values": ["in_progress", "resolved"]},
        {"type": "issue_has_labels", "issue_id": 1, "labels": ["bug", "ui"]},
        {"type": "issue_lacks_labels", "issue_id": 1, "labels": ["security"]},
        {"type": "comment_added", "issue_id": 1, "min_count": 1, "contains": ["Ann", "登入"]},
    ]
    d["success_constraints"] = [
        {"type": "issue_unchanged", "issue_id": 1, "fields": ["title", "body", "priority", "labels"]},
        {"type": "issue_unchanged", "issue_id": 2},
        {"type": "issue_unchanged", "issue_id": 3, "fields": ["comments"], "description": "no comments on #3"},
        {"type": "issue_unchanged", "issue_id": 4},
    ]
    d["forbidden"] = [
        {"type": "forbid_change", "issue_ids": "*", "fields": ["status"], "to_values": ["closed"]},
        {"type": "forbid_change", "issue_ids": [3, 4], "fields": "*", "description": "do not touch #3 or #4"},
        {"type": "forbid_change", "issue_ids": [1], "fields": ["assignee_id"], "to_values": ["u2", None]},
    ]
    d["mutation_scope"] = [
        {"issue_id": 1, "fields": ["assignee_id", "status", "comments"]},
        {"issue_id": 2, "fields": ["labels"]},
    ]
    return d


def _empty_sections_task_dict() -> dict[str, Any]:
    d = _task_dict()
    d["success_constraints"] = []
    d["forbidden"] = []
    d["mutation_scope"] = []
    return d


TASK_DICT_BUILDERS = {
    "minimal": _task_dict,
    "full": _full_task_dict,
    "empty-sections": _empty_sections_task_dict,
}


def _build(data: dict[str, Any]) -> Task:
    """不經 from_dict，直接以建構子建立 Task（頂層 list 轉 tuple、scope entry 轉 ScopeEntry）。"""
    d = copy.deepcopy(data)

    def scope_entry(e: dict[str, Any]) -> ScopeEntry:
        fields = e["fields"]
        return ScopeEntry(e["issue_id"], tuple(fields) if isinstance(fields, list) else fields)

    return Task(
        task_id=d["task_id"],
        family=d["family"],
        generator_version=d["generator_version"],
        params=d["params"],
        instruction=d["instruction"],
        actor_id=d["actor_id"],
        initial_state=d["initial_state"],
        expected_outcome=tuple(d["expected_outcome"]),
        success_constraints=tuple(d["success_constraints"]),
        forbidden=tuple(d["forbidden"]),
        mutation_scope=tuple(scope_entry(e) for e in d["mutation_scope"]),
    )


def _with_item(section: str, item: Any) -> dict[str, Any]:
    d = _task_dict()
    d[section] = [item]
    return d


def _mod(base: dict[str, Any], drop: tuple[str, ...] = (), **over: Any) -> dict[str, Any]:
    d = copy.deepcopy(base)
    for k in drop:
        del d[k]
    d.update(over)
    return d


def _contains_tuple(obj: Any) -> bool:
    if isinstance(obj, tuple):
        return True
    if isinstance(obj, dict):
        return any(_contains_tuple(k) or _contains_tuple(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_contains_tuple(v) for v in obj)
    return False


@pytest.fixture(params=["validate", "from_dict"])
def assert_invalid(request: pytest.FixtureRequest) -> Callable[[dict[str, Any]], None]:
    """同一份不合法 task 同時檢查 Task.validate() 與 Task.from_dict()。"""

    def check(d: dict[str, Any]) -> None:
        if request.param == "validate":
            task = _build(d)
            with pytest.raises(TaskError):
                task.validate()
        else:
            with pytest.raises(TaskError):
                Task.from_dict(d)

    return check


@pytest.fixture(params=["validate", "from_dict"])
def assert_valid(request: pytest.FixtureRequest) -> Callable[[dict[str, Any]], None]:
    def check(d: dict[str, Any]) -> None:
        if request.param == "validate":
            assert _build(d).validate() is None
        else:
            assert Task.from_dict(d) == _build(d)

    return check


@pytest.fixture(params=["expected_outcome", "success_constraints"])
def predicate_section(request: pytest.FixtureRequest) -> str:
    return request.param


VALID_PREDICATES: dict[str, dict[str, Any]] = {
    "issue_field_equals": {"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "resolved"},
    "issue_field_in": {"type": "issue_field_in", "issue_id": 1, "field": "status", "values": ["resolved", "closed"]},
    "issue_has_labels": {"type": "issue_has_labels", "issue_id": 1, "labels": ["security"]},
    "issue_lacks_labels": {"type": "issue_lacks_labels", "issue_id": 1, "labels": ["ui"]},
    "issue_unchanged": {"type": "issue_unchanged", "issue_id": 2, "fields": ["status", "comments"]},
    "comment_added": {"type": "comment_added", "issue_id": 1, "min_count": 2, "contains": ["fixed"]},
}

VALID_RULE: dict[str, Any] = {
    "type": "forbid_change",
    "issue_ids": [1, 3],
    "fields": ["status"],
    "to_values": ["closed"],
}

GOOD_VALUE: dict[str, Any] = {
    "title": "New title",
    "body": "",
    "status": "closed",
    "priority": "p0",
    "assignee_id": "u3",
}

BAD_VALUES = [
    ("title", ""),
    ("title", 1),
    ("title", None),
    ("title", ["t"]),
    ("body", None),
    ("body", 1),
    ("body", ["b"]),
    ("status", "done"),
    ("status", "Open"),
    ("status", None),
    ("status", 1),
    ("status", ["open"]),
    ("priority", "p4"),
    ("priority", "P1"),
    ("priority", None),
    ("priority", 0),
    ("assignee_id", "ghost"),
    ("assignee_id", "unassigned"),
    ("assignee_id", ""),
    ("assignee_id", 1),
    ("assignee_id", True),
    ("assignee_id", ["u1"]),
]

VALID_VALUES = [
    ("title", "New title"),
    ("title", "登入頁面"),
    ("body", ""),
    ("body", "Some text."),
    *[("status", s) for s in STATUSES],
    *[("priority", p) for p in PRIORITIES],
    ("assignee_id", None),
    ("assignee_id", "u1"),
    ("assignee_id", "u2"),
    ("assignee_id", "bot"),
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_task_format_constant() -> None:
    assert TASK_FORMAT == "agent-gym/task/v1"


def test_task_keys_are_format_plus_dataclass_fields() -> None:
    assert len(TASK_KEYS) == 12
    assert len(set(TASK_KEYS)) == 12
    assert set(TASK_KEYS) == {"format"} | {f.name for f in dataclasses.fields(Task)}


def test_predicate_and_rule_type_constants() -> None:
    assert set(PREDICATE_TYPES) == {
        "issue_field_equals",
        "issue_field_in",
        "issue_has_labels",
        "issue_lacks_labels",
        "issue_unchanged",
        "comment_added",
    }
    assert FORBID_RULE_TYPES == ("forbid_change",)


def test_scalar_predicate_fields_are_mutable_fields_without_labels() -> None:
    assert set(SCALAR_PREDICATE_FIELDS) == set(MUTABLE_ISSUE_FIELDS) - {"labels"}


def test_task_error_is_value_error() -> None:
    assert issubclass(TaskError, ValueError)


def test_task_and_scope_entry_are_frozen() -> None:
    entry = ScopeEntry(1, ("status",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.issue_id = 2  # type: ignore[misc]
    task = _build(_task_dict())
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.task_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", TASK_DICT_BUILDERS.values(), ids=TASK_DICT_BUILDERS.keys())
def test_to_dict_has_exactly_task_keys(builder: Callable[[], dict[str, Any]]) -> None:
    out = _build(builder()).to_dict()
    assert set(out) == set(TASK_KEYS)
    assert out["format"] == TASK_FORMAT


@pytest.mark.parametrize("builder", TASK_DICT_BUILDERS.values(), ids=TASK_DICT_BUILDERS.keys())
def test_to_dict_equals_source_dict(builder: Callable[[], dict[str, Any]]) -> None:
    d = builder()
    assert _build(d).to_dict() == d


def test_to_dict_converts_tuples_to_lists() -> None:
    out = _build(_full_task_dict()).to_dict()
    for section in SECTIONS:
        assert isinstance(out[section], list)
    for entry in out["mutation_scope"]:
        assert isinstance(entry["fields"], list)
    assert not _contains_tuple(out)


def test_to_dict_mutation_scope_shape_and_order() -> None:
    d = _task_dict()
    d["mutation_scope"] = [
        {"issue_id": 3, "fields": ["status", "comments"]},
        {"issue_id": 1, "fields": ["labels"]},
    ]
    out = _build(d).to_dict()
    assert out["mutation_scope"] == [
        {"issue_id": 3, "fields": ["status", "comments"]},
        {"issue_id": 1, "fields": ["labels"]},
    ]
    for entry in out["mutation_scope"]:
        assert set(entry) == {"issue_id", "fields"}


def test_to_dict_preserves_predicate_order() -> None:
    d = _full_task_dict()
    out = _build(d).to_dict()
    assert out["expected_outcome"] == d["expected_outcome"]
    assert out["success_constraints"] == d["success_constraints"]
    assert out["forbidden"] == d["forbidden"]


def test_to_dict_from_scope_entries_built_with_tuples() -> None:
    task = dataclasses.replace(
        _build(_task_dict()),
        mutation_scope=(ScopeEntry(2, ("labels", "comments")), ScopeEntry(1, ("assignee_id",))),
    )
    assert task.to_dict()["mutation_scope"] == [
        {"issue_id": 2, "fields": ["labels", "comments"]},
        {"issue_id": 1, "fields": ["assignee_id"]},
    ]


def test_to_dict_is_json_native() -> None:
    out = _build(_full_task_dict()).to_dict()
    assert json.loads(json.dumps(out)) == out


def test_to_dict_passes_canonical_json() -> None:
    out = _build(_full_task_dict()).to_dict()
    assert json.loads(canonical_json(out)) == out


def test_to_dict_returns_deep_copy() -> None:
    d = _full_task_dict()
    task = _build(d)
    out = task.to_dict()
    assert out["initial_state"] is not task.initial_state
    assert out["params"] is not task.params
    assert out["expected_outcome"][0] is not task.expected_outcome[0]
    out["initial_state"]["issues"].clear()
    out["params"]["nested"]["k"].append(2)
    out["expected_outcome"][0]["value"] = "u3"
    out["forbidden"][2]["to_values"].append("u1")
    out["mutation_scope"][0]["fields"].append("title")
    assert task.to_dict() == d
    assert task == _build(d)


def test_to_dict_returns_new_object_each_call() -> None:
    task = _build(_full_task_dict())
    a = task.to_dict()
    b = task.to_dict()
    assert a == b
    assert a is not b
    assert a["initial_state"] is not b["initial_state"]


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", TASK_DICT_BUILDERS.values(), ids=TASK_DICT_BUILDERS.keys())
def test_from_dict_matches_directly_built_task(builder: Callable[[], dict[str, Any]]) -> None:
    d = builder()
    assert Task.from_dict(d) == _build(d)


def test_from_dict_converts_lists_to_tuples_and_scope_entries() -> None:
    d = _full_task_dict()
    task = Task.from_dict(d)
    assert task.task_id == d["task_id"]
    assert task.family == d["family"]
    assert task.generator_version == d["generator_version"]
    assert task.params == d["params"]
    assert task.instruction == d["instruction"]
    assert task.actor_id == d["actor_id"]
    assert task.initial_state == d["initial_state"]
    for section in ("expected_outcome", "success_constraints", "forbidden"):
        assert isinstance(getattr(task, section), tuple)
        assert list(getattr(task, section)) == d[section]
    assert task.mutation_scope == (
        ScopeEntry(1, ("assignee_id", "status", "comments")),
        ScopeEntry(2, ("labels",)),
    )
    for entry in task.mutation_scope:
        assert isinstance(entry, ScopeEntry)
        assert isinstance(entry.fields, tuple)


@pytest.mark.parametrize("builder", TASK_DICT_BUILDERS.values(), ids=TASK_DICT_BUILDERS.keys())
def test_round_trip_through_to_dict(builder: Callable[[], dict[str, Any]]) -> None:
    task = Task.from_dict(builder())
    assert Task.from_dict(task.to_dict()) == task


@pytest.mark.parametrize("builder", TASK_DICT_BUILDERS.values(), ids=TASK_DICT_BUILDERS.keys())
@pytest.mark.parametrize("dumps", [canonical_json, json.dumps], ids=["canonical_json", "json.dumps"])
def test_round_trip_through_json(builder: Callable[[], dict[str, Any]], dumps: Callable[[Any], str]) -> None:
    task = Task.from_dict(builder())
    assert Task.from_dict(json.loads(dumps(task.to_dict()))) == task


def test_round_trip_of_directly_built_task() -> None:
    task = _build(_full_task_dict())
    assert Task.from_dict(task.to_dict()) == task
    assert Task.from_dict(json.loads(canonical_json(task.to_dict()))) == task


def test_from_dict_does_not_mutate_input() -> None:
    d = _full_task_dict()
    before = copy.deepcopy(d)
    Task.from_dict(d)
    assert d == before


def test_from_dict_deep_copies_input() -> None:
    d = _full_task_dict()
    task = Task.from_dict(d)
    expected = _build(_full_task_dict())
    d["initial_state"]["issues"].clear()
    d["params"]["nested"]["k"].append(2)
    d["expected_outcome"][0]["value"] = "u3"
    d["success_constraints"][0]["fields"].append("status")
    d["forbidden"][2]["to_values"].append("u1")
    d["mutation_scope"][0]["fields"].append("title")
    assert task == expected


@pytest.mark.parametrize(
    "data",
    [None, [], "task", 1, list(_task_dict().items())],
    ids=["none", "empty-list", "str", "int", "list-of-pairs"],
)
def test_from_dict_rejects_non_dict(data: Any) -> None:
    with pytest.raises(TaskError):
        Task.from_dict(data)


@pytest.mark.parametrize("key", TASK_KEYS)
def test_from_dict_rejects_missing_key(key: str) -> None:
    d = _task_dict()
    del d[key]
    with pytest.raises(TaskError):
        Task.from_dict(d)


@pytest.mark.parametrize("key", ["notes", "description", "Format", "reference_solution"])
def test_from_dict_rejects_extra_key(key: str) -> None:
    d = _task_dict()
    d[key] = "x"
    with pytest.raises(TaskError):
        Task.from_dict(d)


@pytest.mark.parametrize(
    "fmt",
    ["agent-gym/task/v2", "agent-gym/task/v0", "", None, 1, "agent-gym/env-snapshot/v1", ["agent-gym/task/v1"]],
)
def test_from_dict_rejects_wrong_format(fmt: Any) -> None:
    d = _task_dict()
    d["format"] = fmt
    with pytest.raises(TaskError):
        Task.from_dict(d)


@pytest.mark.parametrize("section", SECTIONS)
@pytest.mark.parametrize("kind", ["none", "str", "dict", "tuple", "int"])
def test_from_dict_rejects_section_that_is_not_a_list(section: str, kind: str) -> None:
    d = _task_dict()
    d[section] = {
        "none": None,
        "str": "x",
        "dict": {},
        "tuple": tuple(d[section]),
        "int": 1,
    }[kind]
    with pytest.raises(TaskError):
        Task.from_dict(d)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(None, id="none"),
        pytest.param([1, ["status"]], id="list"),
        pytest.param("1:status", id="str"),
        pytest.param({"issue_id": 1}, id="missing-fields"),
        pytest.param({"fields": ["status"]}, id="missing-issue-id"),
        pytest.param({}, id="empty-dict"),
        pytest.param({"issue_id": 1, "fields": ["status"], "description": "x"}, id="extra-key"),
        pytest.param({"issue": 1, "fields": ["status"]}, id="renamed-key"),
        pytest.param({"issue_id": 1, "fields": "status"}, id="fields-str"),
        pytest.param({"issue_id": 1, "fields": ("status",)}, id="fields-tuple"),
        pytest.param({"issue_id": 1, "fields": None}, id="fields-none"),
    ],
)
def test_from_dict_rejects_bad_scope_entry(entry: Any) -> None:
    with pytest.raises(TaskError):
        Task.from_dict(_with_item("mutation_scope", entry))


_FUZZ_JUNK: tuple[Any, ...] = (None, True, False, 0, -1, 7, 1.5, "", "x", "*", [], {}, [None], {"k": 1}, [1, 1])


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


def test_from_dict_only_raises_task_error_on_random_corruption() -> None:
    """隨機破壞合法 task：from_dict 只能成功或 raise TaskError，不得漏出其他例外。"""
    rng = random.Random(20261008)
    base = _full_task_dict()
    for _ in range(400):
        mutated, what = _fuzz_mutate(rng, base)
        try:
            task = Task.from_dict(mutated)
        except TaskError:
            continue
        except Exception as e:  # pragma: no cover - 失敗時給出可重現資訊
            raise AssertionError(f"{what}: leaked {type(e).__name__}: {e}") from e
        assert Task.from_dict(task.to_dict()) == task, what


# ---------------------------------------------------------------------------
# validate：positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", TASK_DICT_BUILDERS.values(), ids=TASK_DICT_BUILDERS.keys())
def test_valid_task_dicts_are_valid(assert_valid: Callable, builder: Callable[[], dict[str, Any]]) -> None:
    assert_valid(builder())


@pytest.mark.parametrize("section", ["success_constraints", "forbidden", "mutation_scope"])
def test_optional_sections_may_be_empty(assert_valid: Callable, section: str) -> None:
    d = _full_task_dict()
    d[section] = []
    assert_valid(d)


@pytest.mark.parametrize("ptype", PREDICATE_TYPES)
def test_every_predicate_type_is_valid_in_both_predicate_sections(
    assert_valid: Callable, predicate_section: str, ptype: str
) -> None:
    assert_valid(_with_item(predicate_section, VALID_PREDICATES[ptype]))


@pytest.mark.parametrize("field, value", VALID_VALUES)
def test_field_equals_accepts_valid_values(
    assert_valid: Callable, predicate_section: str, field: str, value: Any
) -> None:
    pred = {"type": "issue_field_equals", "issue_id": 3, "field": field, "value": value}
    assert_valid(_with_item(predicate_section, pred))


@pytest.mark.parametrize("field, value", VALID_VALUES)
def test_field_in_accepts_valid_values(assert_valid: Callable, predicate_section: str, field: str, value: Any) -> None:
    pred = {"type": "issue_field_in", "issue_id": 3, "field": field, "values": [value]}
    assert_valid(_with_item(predicate_section, pred))


@pytest.mark.parametrize("field, value", VALID_VALUES)
def test_to_values_accepts_valid_values_for_single_scalar_field(assert_valid: Callable, field: str, value: Any) -> None:
    rule = {"type": "forbid_change", "issue_ids": "*", "fields": [field], "to_values": [value]}
    assert_valid(_with_item("forbidden", rule))


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param({"type": "issue_field_in", "issue_id": 1, "field": "assignee_id", "values": [None, "u2"]},
                     id="field-in-none-and-inactive"),
        pytest.param({"type": "issue_field_in", "issue_id": 1, "field": "status", "values": list(STATUSES)},
                     id="field-in-all-statuses"),
        pytest.param({"type": "issue_has_labels", "issue_id": 4, "labels": ["bug", "security", "ui"]},
                     id="has-labels-not-on-issue"),
        pytest.param({"type": "issue_lacks_labels", "issue_id": 1, "labels": ["bug"]}, id="lacks-label-on-issue"),
        pytest.param({"type": "issue_unchanged", "issue_id": 1}, id="unchanged-fields-omitted"),
        pytest.param({"type": "issue_unchanged", "issue_id": 1, "fields": ["comments"]}, id="unchanged-comments"),
        pytest.param({"type": "issue_unchanged", "issue_id": 1, "fields": list(F3_FIELDS)}, id="unchanged-all"),
        pytest.param({"type": "comment_added", "issue_id": 2}, id="comment-added-minimal"),
        pytest.param({"type": "comment_added", "issue_id": 2, "min_count": 1}, id="comment-added-min-count-1"),
        pytest.param({"type": "comment_added", "issue_id": 2, "min_count": 5}, id="comment-added-min-count-5"),
        pytest.param({"type": "comment_added", "issue_id": 2, "contains": ["登入", "#2"]}, id="comment-added-contains"),
        pytest.param({"type": "comment_added", "issue_id": 2, "contains": ["fix", "FIX"]},
                     id="comment-added-contains-differ-by-case"),
    ],
)
def test_valid_predicate_variants(assert_valid: Callable, predicate_section: str, pred: dict[str, Any]) -> None:
    assert_valid(_with_item(predicate_section, pred))


@pytest.mark.parametrize("ptype", PREDICATE_TYPES)
def test_predicate_description_str_is_accepted(assert_valid: Callable, predicate_section: str, ptype: str) -> None:
    pred = _mod(VALID_PREDICATES[ptype], description="why this matters")
    assert_valid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param({"type": "forbid_change", "issue_ids": "*", "fields": "*"}, id="star-star"),
        pytest.param({"type": "forbid_change", "issue_ids": [2], "fields": "*"}, id="ids-star-fields"),
        pytest.param({"type": "forbid_change", "issue_ids": "*", "fields": ["comments"]}, id="comments"),
        pytest.param({"type": "forbid_change", "issue_ids": [1, 2, 3, 4], "fields": list(F3_FIELDS)},
                     id="all-ids-all-fields"),
        pytest.param({"type": "forbid_change", "issue_ids": "*", "fields": ["labels", "comments"]},
                     id="labels-and-comments"),
        pytest.param({"type": "forbid_change", "issue_ids": "*", "fields": ["assignee_id"], "to_values": [None]},
                     id="to-values-none"),
        pytest.param({"type": "forbid_change", "issue_ids": "*", "fields": ["assignee_id"],
                      "to_values": ["u2", None]}, id="to-values-inactive-and-none"),
        pytest.param({"type": "forbid_change", "issue_ids": [3], "fields": ["status"],
                      "to_values": ["resolved", "closed"], "description": "never close #3"}, id="with-description"),
    ],
)
def test_valid_forbid_rule_variants(assert_valid: Callable, rule: dict[str, Any]) -> None:
    assert_valid(_with_item("forbidden", rule))


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param([{"issue_id": 1, "fields": ["comments"]}], id="comments"),
        pytest.param([{"issue_id": 1, "fields": list(F3_FIELDS)}], id="all-fields"),
        pytest.param([{"issue_id": i, "fields": ["status"]} for i in (4, 3, 2, 1)], id="every-issue"),
    ],
)
def test_valid_mutation_scope_variants(assert_valid: Callable, scope: list[dict[str, Any]]) -> None:
    d = _task_dict()
    d["mutation_scope"] = scope
    assert_valid(d)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"seed": 1},
        {"a": [1, 2.5, "x", True, None, {"b": []}]},
        {"nested": {"deep": {"deeper": [[], {}]}}},
        {"選項": "值"},
    ],
)
def test_json_native_params_are_valid(assert_valid: Callable, params: dict[str, Any]) -> None:
    d = _task_dict()
    d["params"] = params
    assert_valid(d)


@pytest.mark.parametrize("actor", ["bot", "u1", "u2", "u3"])
def test_any_existing_user_may_be_actor(assert_valid: Callable, actor: str) -> None:
    d = _task_dict()
    d["actor_id"] = actor
    assert_valid(d)


# ---------------------------------------------------------------------------
# validate：top-level fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["task_id", "family", "generator_version", "instruction"])
@pytest.mark.parametrize("value", ["", None, 1, ["x"]], ids=["empty", "none", "int", "list"])
def test_required_str_fields_must_be_non_empty_str(assert_invalid: Callable, key: str, value: Any) -> None:
    d = _task_dict()
    d[key] = value
    assert_invalid(d)


def test_expected_outcome_must_not_be_empty(assert_invalid: Callable) -> None:
    d = _task_dict()
    d["expected_outcome"] = []
    assert_invalid(d)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"a": (1, 2)}, id="tuple-value"),
        pytest.param({"a": [1, (2,)]}, id="tuple-in-list"),
        pytest.param({"a": {"b": (1,)}}, id="tuple-in-dict"),
        pytest.param({1: "a"}, id="int-key"),
        pytest.param({None: "a"}, id="none-key"),
        pytest.param({"a": {2: "b"}}, id="nested-int-key"),
        pytest.param({"a": {1, 2}}, id="set"),
        pytest.param({"a": b"x"}, id="bytes"),
        pytest.param({"a": object()}, id="object"),
    ],
)
def test_params_must_be_json_native_without_tuples(assert_invalid: Callable, params: dict[Any, Any]) -> None:
    d = _task_dict()
    d["params"] = params
    assert_invalid(d)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"x": float("nan")}, id="nan"),
        pytest.param({"x": [float("inf")]}, id="inf-in-list"),
        pytest.param({"x": {"y": float("-inf")}}, id="neg-inf-in-dict"),
    ],
)
def test_params_floats_must_be_finite(assert_invalid: Callable, params: dict[str, Any]) -> None:
    """to_dict 承諾可通過 canonical_json（NaN / ±inf 會 ValueError），canonical round-trip 也必須成立。"""
    d = _task_dict()
    d["params"] = params
    assert_invalid(d)


@pytest.mark.parametrize("actor", ["ghost", "", None, 1, "unassigned", "BOT"])
def test_actor_must_be_existing_user(assert_invalid: Callable, actor: Any) -> None:
    d = _task_dict()
    d["actor_id"] = actor
    assert_invalid(d)


def _break_state(kind: str) -> Any:
    world = copy.deepcopy(SMALL_WORLD)
    if kind == "unsorted-issue-labels":
        world["issues"][0]["labels"] = ["ui", "bug"]
    elif kind == "duplicate-issue-labels":
        world["issues"][0]["labels"] = ["bug", "bug", "ui"]
    elif kind == "tuple-issue-labels":
        world["issues"][0]["labels"] = ("bug", "ui")
    elif kind == "users-out-of-order":
        world["users"].reverse()
    elif kind == "labels-out-of-order":
        world["labels"].reverse()
    elif kind == "issues-out-of-order":
        world["issues"].reverse()
    elif kind == "comments-out-of-order":
        world["comments"].reverse()
    elif kind == "missing-comments-key":
        del world["comments"]
    elif kind == "extra-top-level-key":
        world["milestones"] = []
    elif kind == "invalid-status":
        world["issues"][0]["status"] = "done"
    elif kind == "unknown-assignee":
        world["issues"][0]["assignee_id"] = "ghost"
    elif kind == "comment-on-unknown-issue":
        world["comments"][0]["issue_id"] = 99
    elif kind == "not-dict-str":
        return "world"
    elif kind == "not-dict-list":
        return []
    elif kind == "not-dict-none":
        return None
    else:  # pragma: no cover
        raise AssertionError(kind)
    return world


@pytest.mark.parametrize(
    "kind",
    [
        "unsorted-issue-labels",
        "duplicate-issue-labels",
        "tuple-issue-labels",
        "users-out-of-order",
        "labels-out-of-order",
        "issues-out-of-order",
        "comments-out-of-order",
        "missing-comments-key",
        "extra-top-level-key",
        "invalid-status",
        "unknown-assignee",
        "comment-on-unknown-issue",
        "not-dict-str",
        "not-dict-list",
        "not-dict-none",
    ],
)
def test_initial_state_must_be_valid_and_canonical(assert_invalid: Callable, kind: str) -> None:
    d = _task_dict()
    d["initial_state"] = _break_state(kind)
    assert_invalid(d)


# ---------------------------------------------------------------------------
# validate：predicates（rules 1–7, 11）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param("issue_unchanged", id="str"),
        pytest.param(None, id="none"),
        pytest.param(1, id="int"),
        pytest.param(["type", "issue_unchanged"], id="list"),
        pytest.param(("issue_unchanged", 1), id="tuple"),
    ],
)
def test_predicate_must_be_dict(assert_invalid: Callable, predicate_section: str, pred: Any) -> None:
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param({"issue_id": 1}, id="missing-type"),
        pytest.param({"type": "issue_field_not_equals", "issue_id": 1}, id="unknown-type"),
        pytest.param({"type": "Issue_Unchanged", "issue_id": 1}, id="wrong-case-type"),
        pytest.param({"type": "forbid_change", "issue_ids": "*", "fields": "*"}, id="forbid-rule-as-predicate"),
        pytest.param({"type": 1, "issue_id": 1}, id="int-type"),
        pytest.param({"type": None, "issue_id": 1}, id="none-type"),
        pytest.param({"type": ["issue_unchanged"], "issue_id": 1}, id="unhashable-type"),
    ],
)
def test_predicate_type_must_be_known(assert_invalid: Callable, predicate_section: str, pred: dict[str, Any]) -> None:
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "pred",
    [
        *[pytest.param(_mod(VALID_PREDICATES[t], foo=1), id=f"{t}-foo") for t in PREDICATE_TYPES],
        pytest.param(_mod(VALID_PREDICATES["issue_field_equals"], values=["closed"]), id="equals-with-values"),
        pytest.param(_mod(VALID_PREDICATES["issue_field_in"], value="closed"), id="in-with-value"),
        pytest.param(_mod(VALID_PREDICATES["issue_has_labels"], field="labels"), id="has-labels-with-field"),
        pytest.param(_mod(VALID_PREDICATES["issue_unchanged"], value="open"), id="unchanged-with-value"),
        pytest.param(_mod(VALID_PREDICATES["comment_added"], author_id="bot"), id="comment-added-with-author"),
        pytest.param(_mod(VALID_PREDICATES["issue_unchanged"], issue_ids=[2]), id="unchanged-with-issue-ids"),
        pytest.param({**VALID_PREDICATES["issue_unchanged"], 7: "x"}, id="non-str-extra-key"),
    ],
)
def test_predicate_extra_key_is_invalid(assert_invalid: Callable, predicate_section: str, pred: dict[Any, Any]) -> None:
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "ptype, key",
    [
        ("issue_field_equals", "issue_id"),
        ("issue_field_equals", "field"),
        ("issue_field_equals", "value"),
        ("issue_field_in", "issue_id"),
        ("issue_field_in", "field"),
        ("issue_field_in", "values"),
        ("issue_has_labels", "issue_id"),
        ("issue_has_labels", "labels"),
        ("issue_lacks_labels", "issue_id"),
        ("issue_lacks_labels", "labels"),
        ("issue_unchanged", "issue_id"),
        ("comment_added", "issue_id"),
    ],
)
def test_predicate_missing_required_key_is_invalid(
    assert_invalid: Callable, predicate_section: str, ptype: str, key: str
) -> None:
    assert_invalid(_with_item(predicate_section, _mod(VALID_PREDICATES[ptype], drop=(key,))))


@pytest.mark.parametrize("ptype", PREDICATE_TYPES)
@pytest.mark.parametrize("description", [1, None, ["d"], {"d": 1}], ids=["int", "none", "list", "dict"])
def test_predicate_description_must_be_str(
    assert_invalid: Callable, predicate_section: str, ptype: str, description: Any
) -> None:
    assert_invalid(_with_item(predicate_section, _mod(VALID_PREDICATES[ptype], description=description)))


@pytest.mark.parametrize("ptype", PREDICATE_TYPES)
@pytest.mark.parametrize(
    "issue_id",
    [True, "1", 1.0, [1], None, 0, 99],
    ids=["bool", "str", "float", "list", "none", "zero", "unknown"],
)
def test_predicate_issue_id_must_be_existing_int(
    assert_invalid: Callable, predicate_section: str, ptype: str, issue_id: Any
) -> None:
    assert_invalid(_with_item(predicate_section, _mod(VALID_PREDICATES[ptype], issue_id=issue_id)))


@pytest.mark.parametrize("ptype", ["issue_field_equals", "issue_field_in"])
@pytest.mark.parametrize(
    "field",
    ["labels", "comments", "reporter_id", "created_at", "id", "Status", "", None, ["status"]],
)
def test_scalar_predicate_field_must_be_scalar_field(
    assert_invalid: Callable, predicate_section: str, ptype: str, field: Any
) -> None:
    pred = _mod(VALID_PREDICATES[ptype], field=field)
    if field == "labels":
        pred = _mod(pred, **({"value": ["bug"]} if ptype == "issue_field_equals" else {"values": [["bug"]]}))
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize("field, value", BAD_VALUES)
def test_field_equals_rejects_invalid_value(
    assert_invalid: Callable, predicate_section: str, field: str, value: Any
) -> None:
    pred = {"type": "issue_field_equals", "issue_id": 1, "field": field, "value": value}
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize("field, value", BAD_VALUES)
def test_field_in_rejects_invalid_value(assert_invalid: Callable, predicate_section: str, field: str, value: Any) -> None:
    pred = {"type": "issue_field_in", "issue_id": 1, "field": field, "values": [GOOD_VALUE[field], value]}
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([], id="empty"),
        pytest.param(["closed", "closed"], id="duplicate"),
        pytest.param("closed", id="str"),
        pytest.param(("closed",), id="tuple"),
        pytest.param(None, id="none"),
        pytest.param({"closed": 1}, id="dict"),
    ],
)
def test_field_in_values_must_be_non_empty_unique_list(
    assert_invalid: Callable, predicate_section: str, values: Any
) -> None:
    pred = _mod(VALID_PREDICATES["issue_field_in"], values=values)
    assert_invalid(_with_item(predicate_section, pred))


def test_field_in_values_duplicate_none_is_invalid(assert_invalid: Callable, predicate_section: str) -> None:
    pred = {"type": "issue_field_in", "issue_id": 1, "field": "assignee_id", "values": [None, None]}
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize("ptype", ["issue_has_labels", "issue_lacks_labels"])
@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([], id="empty"),
        pytest.param(["bug", "bug"], id="duplicate"),
        pytest.param(["nope"], id="unknown"),
        pytest.param(["bug", "nope"], id="partly-unknown"),
        pytest.param(["Bug"], id="wrong-case"),
        pytest.param([1], id="int-element"),
        pytest.param([["bug"]], id="unhashable-element"),
        pytest.param("bug", id="str"),
        pytest.param(("bug",), id="tuple"),
        pytest.param(None, id="none"),
    ],
)
def test_label_predicate_labels_must_be_unique_existing_labels(
    assert_invalid: Callable, predicate_section: str, ptype: str, labels: Any
) -> None:
    pred = _mod(VALID_PREDICATES[ptype], labels=labels)
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param([], id="empty"),
        pytest.param(["status", "status"], id="duplicate"),
        pytest.param(["reporter_id"], id="reporter-id"),
        pytest.param(["created_at"], id="created-at"),
        pytest.param(["id"], id="id"),
        pytest.param(["status", "nope"], id="partly-unknown"),
        pytest.param([["status"]], id="unhashable-element"),
        pytest.param("*", id="star"),
        pytest.param("status", id="str"),
        pytest.param(("status",), id="tuple"),
        pytest.param(None, id="none"),
    ],
)
def test_unchanged_fields_must_be_unique_list_of_mutable_fields_or_comments(
    assert_invalid: Callable, predicate_section: str, fields: Any
) -> None:
    pred = _mod(VALID_PREDICATES["issue_unchanged"], fields=fields)
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "min_count",
    [0, -1, True, False, 1.0, "1", None, [1]],
    ids=["zero", "negative", "true", "false", "float", "str", "none", "list"],
)
def test_comment_added_min_count_must_be_positive_int(
    assert_invalid: Callable, predicate_section: str, min_count: Any
) -> None:
    pred = _mod(VALID_PREDICATES["comment_added"], min_count=min_count)
    assert_invalid(_with_item(predicate_section, pred))


@pytest.mark.parametrize(
    "contains",
    [
        pytest.param([], id="empty"),
        pytest.param([""], id="empty-element"),
        pytest.param(["ok", ""], id="partly-empty-element"),
        pytest.param(["fix", "fix"], id="duplicate"),
        pytest.param([1], id="int-element"),
        pytest.param([None], id="none-element"),
        pytest.param([["fix"]], id="unhashable-element"),
        pytest.param("fix", id="str"),
        pytest.param(("fix",), id="tuple"),
        pytest.param(None, id="none"),
    ],
)
def test_comment_added_contains_must_be_unique_non_empty_strs(
    assert_invalid: Callable, predicate_section: str, contains: Any
) -> None:
    pred = _mod(VALID_PREDICATES["comment_added"], contains=contains)
    assert_invalid(_with_item(predicate_section, pred))


# ---------------------------------------------------------------------------
# validate：forbid rules（rules 1–7, 11）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param("forbid_change", id="str"),
        pytest.param(None, id="none"),
        pytest.param(["forbid_change"], id="list"),
        pytest.param({"issue_ids": "*", "fields": "*"}, id="missing-type"),
        pytest.param({"type": "forbid_comment", "issue_ids": "*", "fields": "*"}, id="unknown-type"),
        pytest.param({"type": ["forbid_change"], "issue_ids": "*", "fields": "*"}, id="unhashable-type"),
        pytest.param(VALID_PREDICATES["issue_unchanged"], id="predicate-as-rule"),
        pytest.param(_mod(VALID_RULE, drop=("issue_ids",)), id="missing-issue-ids"),
        pytest.param(_mod(VALID_RULE, drop=("fields",)), id="missing-fields"),
        pytest.param(_mod(VALID_RULE, foo=1), id="extra-key"),
        pytest.param(_mod(VALID_RULE, to_value="closed"), id="extra-to-value"),
        pytest.param(_mod(VALID_RULE, issue_id=1), id="extra-issue-id"),
        pytest.param({**VALID_RULE, 3: "x"}, id="non-str-extra-key"),
        pytest.param(_mod(VALID_RULE, description=1), id="int-description"),
        pytest.param(_mod(VALID_RULE, description=None), id="none-description"),
    ],
)
def test_forbid_rule_structure(assert_invalid: Callable, rule: Any) -> None:
    assert_invalid(_with_item("forbidden", rule))


@pytest.mark.parametrize(
    "issue_ids",
    [
        pytest.param([], id="empty"),
        pytest.param([1, 1], id="duplicate"),
        pytest.param([True], id="bool"),
        pytest.param(["1"], id="str-element"),
        pytest.param([1.0], id="float-element"),
        pytest.param([99], id="unknown"),
        pytest.param([1, 99], id="partly-unknown"),
        pytest.param([[1]], id="unhashable-element"),
        pytest.param("all", id="str-not-star"),
        pytest.param("**", id="double-star"),
        pytest.param(1, id="int"),
        pytest.param((1,), id="tuple"),
        pytest.param(None, id="none"),
    ],
)
def test_forbid_issue_ids_must_be_star_or_unique_existing_ints(assert_invalid: Callable, issue_ids: Any) -> None:
    rule = {"type": "forbid_change", "issue_ids": issue_ids, "fields": ["status"]}
    assert_invalid(_with_item("forbidden", rule))


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param([], id="empty"),
        pytest.param(["status", "status"], id="duplicate"),
        pytest.param(["reporter_id"], id="reporter-id"),
        pytest.param(["created_at"], id="created-at"),
        pytest.param(["status", "nope"], id="partly-unknown"),
        pytest.param([["status"]], id="unhashable-element"),
        pytest.param("all", id="str-not-star"),
        pytest.param("status", id="str"),
        pytest.param(("status",), id="tuple"),
        pytest.param(None, id="none"),
    ],
)
def test_forbid_fields_must_be_star_or_unique_f3_list(assert_invalid: Callable, fields: Any) -> None:
    rule = {"type": "forbid_change", "issue_ids": "*", "fields": fields}
    assert_invalid(_with_item("forbidden", rule))


@pytest.mark.parametrize(
    "to_values",
    [
        pytest.param([], id="empty"),
        pytest.param(["closed", "closed"], id="duplicate"),
        pytest.param("closed", id="str"),
        pytest.param(("closed",), id="tuple"),
        pytest.param(None, id="none"),
        pytest.param([["closed"]], id="unhashable-element"),
    ],
)
def test_forbid_to_values_must_be_non_empty_unique_list(assert_invalid: Callable, to_values: Any) -> None:
    rule = {"type": "forbid_change", "issue_ids": "*", "fields": ["status"], "to_values": to_values}
    assert_invalid(_with_item("forbidden", rule))


@pytest.mark.parametrize(
    "fields, to_values",
    [
        pytest.param(["status", "priority"], ["closed"], id="two-scalar-fields"),
        pytest.param(["status", "comments"], ["closed"], id="scalar-and-comments"),
        pytest.param("*", ["closed"], id="star-fields"),
        pytest.param(["labels"], [["bug"]], id="labels-field"),
        pytest.param(["labels"], ["bug"], id="labels-field-str-value"),
        pytest.param(["comments"], ["hello"], id="comments-field"),
    ],
)
def test_to_values_requires_exactly_one_scalar_field(assert_invalid: Callable, fields: Any, to_values: Any) -> None:
    rule = {"type": "forbid_change", "issue_ids": "*", "fields": fields, "to_values": to_values}
    assert_invalid(_with_item("forbidden", rule))


@pytest.mark.parametrize("field, value", BAD_VALUES)
def test_to_values_rejects_invalid_value_for_field(assert_invalid: Callable, field: str, value: Any) -> None:
    rule = {"type": "forbid_change", "issue_ids": "*", "fields": [field], "to_values": [GOOD_VALUE[field], value]}
    assert_invalid(_with_item("forbidden", rule))


# ---------------------------------------------------------------------------
# validate：mutation_scope（rules 1, 2, 9, 11）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"issue_id": True, "fields": ["status"]}, id="bool-issue-id"),
        pytest.param({"issue_id": "1", "fields": ["status"]}, id="str-issue-id"),
        pytest.param({"issue_id": 1.0, "fields": ["status"]}, id="float-issue-id"),
        pytest.param({"issue_id": None, "fields": ["status"]}, id="none-issue-id"),
        pytest.param({"issue_id": 99, "fields": ["status"]}, id="unknown-issue"),
        pytest.param({"issue_id": [1], "fields": ["status"]}, id="unhashable-issue-id"),
        pytest.param({"issue_id": 1, "fields": []}, id="empty-fields"),
        pytest.param({"issue_id": 1, "fields": ["status", "status"]}, id="duplicate-fields"),
        pytest.param({"issue_id": 1, "fields": ["reporter_id"]}, id="reporter-id"),
        pytest.param({"issue_id": 1, "fields": ["created_at"]}, id="created-at"),
        pytest.param({"issue_id": 1, "fields": ["status", "nope"]}, id="partly-unknown-field"),
        pytest.param({"issue_id": 1, "fields": [["status"]]}, id="unhashable-field"),
        pytest.param({"issue_id": 1, "fields": "*"}, id="star-fields"),
        pytest.param({"issue_id": 1, "fields": "status"}, id="str-fields"),
        pytest.param({"issue_id": 1, "fields": None}, id="none-fields"),
    ],
)
def test_scope_entry_must_reference_existing_issue_and_f3_fields(assert_invalid: Callable, entry: dict[str, Any]) -> None:
    assert_invalid(_with_item("mutation_scope", entry))


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param([{"issue_id": 1, "fields": ["status"]}, {"issue_id": 1, "fields": ["priority"]}],
                     id="same-issue-different-fields"),
        pytest.param([{"issue_id": 1, "fields": ["status"]}, {"issue_id": 1, "fields": ["status"]}],
                     id="identical-entries"),
        pytest.param([{"issue_id": 2, "fields": ["status"]}, {"issue_id": 3, "fields": ["status"]},
                      {"issue_id": 2, "fields": ["labels"]}], id="non-adjacent"),
    ],
)
def test_scope_issue_must_not_repeat(assert_invalid: Callable, scope: list[dict[str, Any]]) -> None:
    d = _task_dict()
    d["mutation_scope"] = scope
    assert_invalid(d)


def test_validate_rejects_scope_entry_with_non_str_field() -> None:
    task = dataclasses.replace(_build(_task_dict()), mutation_scope=(ScopeEntry(1, ("status", 1)),))  # type: ignore[arg-type]
    with pytest.raises(TaskError):
        task.validate()


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param(({"issue_id": 1, "fields": ["assignee_id"]},), id="dict"),
        pytest.param((None,), id="none"),
        pytest.param(((1, ("assignee_id",)),), id="tuple"),
        pytest.param(("1:assignee_id",), id="str"),
    ],
)
def test_validate_rejects_non_scope_entry_items(scope: tuple[Any, ...]) -> None:
    """rule 11：scope entry 不是 ScopeEntry 時 raise TaskError，不得漏出 AttributeError / TypeError。"""
    task = dataclasses.replace(_build(_task_dict()), mutation_scope=scope)
    with pytest.raises(TaskError):
        task.validate()


# ---------------------------------------------------------------------------
# initial_state_obj
# ---------------------------------------------------------------------------


def _state(world: dict[str, Any] = SMALL_WORLD) -> IssueTrackerState:
    """不經 IssueTrackerState.from_dict，直接由 dict 建出 state。"""
    return IssueTrackerState(
        users={u["id"]: User(**u) for u in world["users"]},
        labels={lb["name"]: Label(**lb) for lb in world["labels"]},
        issues={i["id"]: Issue(**{**i, "labels": tuple(sorted(set(i["labels"])))}) for i in world["issues"]},
        comments={c["id"]: Comment(**c) for c in world["comments"]},
    )


def test_initial_state_obj_matches_initial_state() -> None:
    task = _build(_task_dict())
    state = task.initial_state_obj()
    assert isinstance(state, IssueTrackerState)
    assert state == _state()
    assert state.to_dict() == task.initial_state


def test_initial_state_obj_returns_new_object_each_call() -> None:
    task = _build(_task_dict())
    a = task.initial_state_obj()
    b = task.initial_state_obj()
    assert a is not b
    assert a == b
    for attr in ("users", "labels", "issues", "comments"):
        assert getattr(a, attr) is not getattr(b, attr)
    a.issues.clear()
    a.comments.clear()
    assert b == _state()
    assert task.initial_state_obj() == _state()
    assert task.initial_state == SMALL_WORLD


# ---------------------------------------------------------------------------
# evaluate_predicate
# ---------------------------------------------------------------------------


def _clone(state: IssueTrackerState) -> IssueTrackerState:
    return IssueTrackerState(
        users=dict(state.users),
        labels=dict(state.labels),
        issues=dict(state.issues),
        comments=dict(state.comments),
    )


def _edit(state: IssueTrackerState, issue_id: int, **fields: Any) -> IssueTrackerState:
    new = _clone(state)
    if "labels" in fields:
        fields["labels"] = tuple(sorted(fields["labels"]))
    new.issues[issue_id] = dataclasses.replace(new.issues[issue_id], **fields)
    return new


def _comment(state: IssueTrackerState, issue_id: int, body: str, author_id: str = SMALL_ACTOR) -> IssueTrackerState:
    new = _clone(state)
    cid = max(new.comments, default=0) + 1
    new.comments[cid] = Comment(id=cid, issue_id=issue_id, author_id=author_id, body=body)
    return new


def _passed(pred: dict[str, Any], initial: IssueTrackerState, final: IssueTrackerState) -> bool:
    return evaluate_predicate(pred, initial, final)[0]


@pytest.fixture
def initial() -> IssueTrackerState:
    return _state()


CHANGED_VALUE: dict[str, Any] = {
    "title": "Crash on logout",
    "body": "Updated body.",
    "status": "closed",
    "priority": "p0",
    "assignee_id": "u3",
    "labels": ("bug",),
}


# --- issue_field_equals / issue_field_in ---


@pytest.mark.parametrize("field", SCALAR_PREDICATE_FIELDS)
def test_field_equals_passes_when_final_value_matches(initial: IssueTrackerState, field: str) -> None:
    value = getattr(initial.issues[1], field)
    pred = {"type": "issue_field_equals", "issue_id": 1, "field": field, "value": value}
    assert _passed(pred, initial, _clone(initial)) is True


@pytest.mark.parametrize("field", SCALAR_PREDICATE_FIELDS)
def test_field_equals_compares_final_state_only(initial: IssueTrackerState, field: str) -> None:
    old = getattr(initial.issues[1], field)
    new = CHANGED_VALUE[field]
    final = _edit(initial, 1, **{field: new})
    assert _passed({"type": "issue_field_equals", "issue_id": 1, "field": field, "value": new}, initial, final) is True
    assert _passed({"type": "issue_field_equals", "issue_id": 1, "field": field, "value": old}, initial, final) is False


@pytest.mark.parametrize(
    "issue_id, value, expected",
    [(1, None, True), (1, "u1", False), (2, None, False), (2, "u1", True), (3, "u2", True)],
)
def test_field_equals_assignee_with_none(initial: IssueTrackerState, issue_id: int, value: Any, expected: bool) -> None:
    pred = {"type": "issue_field_equals", "issue_id": issue_id, "field": "assignee_id", "value": value}
    assert _passed(pred, initial, _clone(initial)) is expected


def test_field_equals_after_unassign(initial: IssueTrackerState) -> None:
    final = _edit(initial, 2, assignee_id=None)
    assert _passed({"type": "issue_field_equals", "issue_id": 2, "field": "assignee_id", "value": None},
                   initial, final) is True
    assert _passed({"type": "issue_field_equals", "issue_id": 2, "field": "assignee_id", "value": "u1"},
                   initial, final) is False


@pytest.mark.parametrize(
    "final_status, expected",
    [("closed", True), ("resolved", True), ("open", False), ("in_progress", False)],
)
def test_field_in_checks_final_membership(initial: IssueTrackerState, final_status: str, expected: bool) -> None:
    final = _edit(initial, 1, status=final_status)
    pred = {"type": "issue_field_in", "issue_id": 1, "field": "status", "values": ["resolved", "closed"]}
    assert _passed(pred, initial, final) is expected


@pytest.mark.parametrize(
    "issue_id, final_assignee, expected",
    [(1, None, True), (1, "u3", True), (1, "u1", False), (2, "u1", False), (2, None, True), (3, "u2", False)],
)
def test_field_in_with_none_in_values(
    initial: IssueTrackerState, issue_id: int, final_assignee: Any, expected: bool
) -> None:
    final = _edit(initial, issue_id, assignee_id=final_assignee)
    pred = {"type": "issue_field_in", "issue_id": issue_id, "field": "assignee_id", "values": [None, "u3"]}
    assert _passed(pred, initial, final) is expected


# --- issue_has_labels / issue_lacks_labels ---


@pytest.mark.parametrize(
    "final_labels, labels, expected",
    [
        (("bug", "ui"), ["bug"], True),
        (("bug", "ui"), ["ui", "bug"], True),
        (("bug", "ui"), ["bug", "security"], False),
        ((), ["bug"], False),
        (("ui",), ["bug"], False),
        (("bug", "security", "ui"), ["security"], True),
        (("bug", "security", "ui"), ["security", "bug", "ui"], True),
    ],
)
def test_has_labels_is_subset_of_final_labels(
    initial: IssueTrackerState, final_labels: tuple[str, ...], labels: list[str], expected: bool
) -> None:
    final = _edit(initial, 1, labels=final_labels)
    pred = {"type": "issue_has_labels", "issue_id": 1, "labels": labels}
    assert _passed(pred, initial, final) is expected


@pytest.mark.parametrize(
    "final_labels, labels, expected",
    [
        (("bug", "ui"), ["security"], True),
        (("bug", "ui"), ["security", "bug"], False),
        (("bug", "ui"), ["ui"], False),
        (("bug",), ["ui"], True),
        ((), ["bug", "security", "ui"], True),
        (("bug", "security", "ui"), ["security"], False),
    ],
)
def test_lacks_labels_is_disjoint_from_final_labels(
    initial: IssueTrackerState, final_labels: tuple[str, ...], labels: list[str], expected: bool
) -> None:
    final = _edit(initial, 1, labels=final_labels)
    pred = {"type": "issue_lacks_labels", "issue_id": 1, "labels": labels}
    assert _passed(pred, initial, final) is expected


# --- issue_unchanged ---


@pytest.mark.parametrize("issue_id", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "fields",
    [None, ["status"], ["comments"], list(F3_FIELDS)],
    ids=["omitted", "status", "comments", "all"],
)
def test_unchanged_passes_for_identical_states(initial: IssueTrackerState, issue_id: int, fields: Any) -> None:
    pred: dict[str, Any] = {"type": "issue_unchanged", "issue_id": issue_id}
    if fields is not None:
        pred["fields"] = fields
    assert _passed(pred, initial, _clone(initial)) is True


@pytest.mark.parametrize("field", MUTABLE_ISSUE_FIELDS)
def test_unchanged_explicit_field_detects_change(initial: IssueTrackerState, field: str) -> None:
    final = _edit(initial, 1, **{field: CHANGED_VALUE[field]})
    assert _passed({"type": "issue_unchanged", "issue_id": 1, "fields": [field]}, initial, final) is False


@pytest.mark.parametrize("field", MUTABLE_ISSUE_FIELDS)
def test_unchanged_explicit_fields_ignore_other_fields(initial: IssueTrackerState, field: str) -> None:
    others = [f for f in MUTABLE_ISSUE_FIELDS if f != field]
    final = _edit(initial, 1, **{field: CHANGED_VALUE[field]})
    assert _passed({"type": "issue_unchanged", "issue_id": 1, "fields": others}, initial, final) is True


def test_unchanged_compares_values_not_identity(initial: IssueTrackerState) -> None:
    final = _edit(initial, 1, labels=["ui", "bug"], title="Crash on " + "login")
    assert _passed({"type": "issue_unchanged", "issue_id": 1}, initial, final) is True


def test_unchanged_is_final_state_only(initial: IssueTrackerState) -> None:
    final = _edit(_edit(initial, 1, status="closed"), 1, status="open")
    assert _passed({"type": "issue_unchanged", "issue_id": 1}, initial, final) is True


def test_unchanged_comments_fails_on_new_comment_on_same_issue(initial: IssueTrackerState) -> None:
    final = _comment(initial, 1, "Looking into it.")
    assert _passed({"type": "issue_unchanged", "issue_id": 1, "fields": ["comments"]}, initial, final) is False


def test_unchanged_comments_ignores_new_comment_on_other_issue(initial: IssueTrackerState) -> None:
    final = _comment(initial, 2, "Looking into it.")
    assert _passed({"type": "issue_unchanged", "issue_id": 1, "fields": ["comments"]}, initial, final) is True


def test_unchanged_comments_on_issue_without_comments(initial: IssueTrackerState) -> None:
    pred = {"type": "issue_unchanged", "issue_id": 4, "fields": ["comments"]}
    assert _passed(pred, initial, _comment(initial, 3, "x")) is True
    assert _passed(pred, initial, _comment(initial, 4, "x")) is False


def test_unchanged_comments_ignores_field_changes(initial: IssueTrackerState) -> None:
    final = _edit(initial, 1, status="closed", labels=())
    assert _passed({"type": "issue_unchanged", "issue_id": 1, "fields": ["comments"]}, initial, final) is True


def test_unchanged_explicit_fields_without_comments_ignore_new_comment(initial: IssueTrackerState) -> None:
    final = _comment(initial, 1, "Looking into it.")
    pred = {"type": "issue_unchanged", "issue_id": 1, "fields": list(MUTABLE_ISSUE_FIELDS)}
    assert _passed(pred, initial, final) is True


@pytest.mark.parametrize("field", MUTABLE_ISSUE_FIELDS)
def test_unchanged_without_fields_detects_each_mutable_field(initial: IssueTrackerState, field: str) -> None:
    final = _edit(initial, 1, **{field: CHANGED_VALUE[field]})
    assert _passed({"type": "issue_unchanged", "issue_id": 1}, initial, final) is False


def test_unchanged_without_fields_detects_new_comment(initial: IssueTrackerState) -> None:
    final = _comment(initial, 1, "Looking into it.")
    assert _passed({"type": "issue_unchanged", "issue_id": 1}, initial, final) is False


def test_unchanged_without_fields_ignores_other_issues(initial: IssueTrackerState) -> None:
    final = _comment(_edit(initial, 2, status="closed", assignee_id=None), 2, "Done.")
    assert _passed({"type": "issue_unchanged", "issue_id": 1}, initial, final) is True


def _final_variants(initial: IssueTrackerState) -> list[IssueTrackerState]:
    return [
        _clone(initial),
        *[_edit(initial, 1, **{f: CHANGED_VALUE[f]}) for f in MUTABLE_ISSUE_FIELDS],
        _comment(initial, 1, "new"),
        _comment(initial, 2, "new"),
        _edit(initial, 3, status="closed"),
    ]


def test_unchanged_without_fields_equals_all_fields_plus_comments(initial: IssueTrackerState) -> None:
    omitted = {"type": "issue_unchanged", "issue_id": 1}
    explicit = {"type": "issue_unchanged", "issue_id": 1, "fields": list(F3_FIELDS)}
    for final in _final_variants(initial):
        assert _passed(omitted, initial, final) is _passed(explicit, initial, final)


# --- comment_added ---


def test_comment_added_fails_without_new_comments(initial: IssueTrackerState) -> None:
    assert _passed({"type": "comment_added", "issue_id": 1}, initial, _clone(initial)) is False


def test_comment_added_passes_with_one_new_comment(initial: IssueTrackerState) -> None:
    final = _comment(initial, 1, "Working on it.")
    assert _passed({"type": "comment_added", "issue_id": 1}, initial, final) is True


def test_comment_added_on_issue_without_prior_comments(initial: IssueTrackerState) -> None:
    final = _comment(initial, 4, "Reopening discussion.")
    assert _passed({"type": "comment_added", "issue_id": 4}, initial, final) is True


def test_comment_added_ignores_existing_comments(initial: IssueTrackerState) -> None:
    pred = {"type": "comment_added", "issue_id": 1, "contains": ["build 42"]}
    assert _passed(pred, initial, _clone(initial)) is False
    assert _passed(pred, initial, _comment(initial, 1, "Unrelated note.")) is False
    assert _passed(pred, initial, _comment(initial, 1, "Still fails on build 42.")) is True


def test_comment_added_ignores_new_comments_on_other_issues(initial: IssueTrackerState) -> None:
    final = _comment(_comment(initial, 2, "fixed"), 3, "fixed")
    assert _passed({"type": "comment_added", "issue_id": 1}, initial, final) is False
    assert _passed({"type": "comment_added", "issue_id": 1, "contains": ["fixed"]}, initial, final) is False


@pytest.mark.parametrize("author", ["bot", "u1", "u2", "u3"])
def test_comment_added_counts_comments_by_any_author(initial: IssueTrackerState, author: str) -> None:
    final = _comment(initial, 1, "fixed", author_id=author)
    assert _passed({"type": "comment_added", "issue_id": 1, "contains": ["fixed"]}, initial, final) is True


def test_comment_added_requires_all_contains_in_same_comment(initial: IssueTrackerState) -> None:
    pred = {"type": "comment_added", "issue_id": 1, "contains": ["fixed", "#2"]}
    split = _comment(_comment(initial, 1, "fixed it"), 1, "see #2")
    assert _passed(pred, initial, split) is False
    assert _passed(pred, initial, _comment(split, 1, "Fixed, duplicate of #2.")) is True


@pytest.mark.parametrize(
    "body, contains, expected",
    [
        ("Fixed in BUILD 43", ["build 43"], True),
        ("fixed in build 43", ["BUILD 43"], True),
        ("Closing as duplicate of #2.", ["#2"], True),
        ("Grüße aus der Straße", ["STRASSE"], True),
        ("Moved to STRASSE", ["straße"], True),
        ("ΣΊΣΥΦΟΣ", ["σίσυφος"], True),
        ("登入頁面已修復", ["登入"], True),
        ("Fixed", ["fixed", "#2"], False),
        ("partial", ["partially"], False),
        ("Bob", ["Bob Li"], False),
    ],
)
def test_comment_added_contains_uses_casefold_substring(
    initial: IssueTrackerState, body: str, contains: list[str], expected: bool
) -> None:
    final = _comment(initial, 1, body)
    pred = {"type": "comment_added", "issue_id": 1, "contains": contains}
    assert _passed(pred, initial, final) is expected


@pytest.mark.parametrize(
    "bodies, contains, min_count, expected",
    [
        (["x"], None, None, True),
        ([], None, None, False),
        (["x"], None, 2, False),
        (["x", "y"], None, 2, True),
        (["fix one"], ["fix"], 2, False),
        (["fix one", "fix two"], ["fix"], 2, True),
        (["fix one", "no match"], ["fix"], 2, False),
        (["fix one", "fix two", "other"], ["fix"], 3, False),
        (["fix one", "fix two", "fix three"], ["fix"], 3, True),
        (["FIX one", "fix two", "Fix three", "other"], ["fix"], 3, True),
    ],
)
def test_comment_added_counts_matching_new_comments(
    initial: IssueTrackerState, bodies: list[str], contains: Any, min_count: Any, expected: bool
) -> None:
    final = initial
    for body in bodies:
        final = _comment(final, 1, body)
    final = _comment(final, 2, "fix elsewhere")
    pred: dict[str, Any] = {"type": "comment_added", "issue_id": 1}
    if contains is not None:
        pred["contains"] = contains
    if min_count is not None:
        pred["min_count"] = min_count
    assert _passed(pred, initial, final) is expected


@pytest.mark.parametrize("new_comments", [0, 1, 2])
def test_comment_added_default_min_count_is_one(initial: IssueTrackerState, new_comments: int) -> None:
    final = initial
    for i in range(new_comments):
        final = _comment(final, 1, f"note {i}")
    default = _passed({"type": "comment_added", "issue_id": 1}, initial, final)
    explicit = _passed({"type": "comment_added", "issue_id": 1, "min_count": 1}, initial, final)
    assert default is explicit is (new_comments >= 1)


# --- 共通性質 ---


SAMPLE_PREDICATES: list[dict[str, Any]] = [
    {"type": "issue_field_equals", "issue_id": 1, "field": "status", "value": "closed"},
    {"type": "issue_field_equals", "issue_id": 1, "field": "priority", "value": "p0"},
    {"type": "issue_field_in", "issue_id": 1, "field": "assignee_id", "values": [None, "u2"]},
    {"type": "issue_has_labels", "issue_id": 1, "labels": ["security"]},
    {"type": "issue_lacks_labels", "issue_id": 1, "labels": ["ui"]},
    {"type": "issue_unchanged", "issue_id": 1},
    {"type": "issue_unchanged", "issue_id": 2, "fields": ["status", "comments"]},
    {"type": "comment_added", "issue_id": 1, "min_count": 1, "contains": ["fixed"]},
    {"type": "comment_added", "issue_id": 3},
]


def _sample_final(initial: IssueTrackerState) -> IssueTrackerState:
    return _comment(_edit(initial, 1, status="closed", labels=("bug", "security")), 1, "Fixed in build 43.")


@pytest.mark.parametrize("pred", SAMPLE_PREDICATES, ids=lambda p: p["type"])
def test_evaluate_returns_bool_and_non_empty_str(initial: IssueTrackerState, pred: dict[str, Any]) -> None:
    for final in (_clone(initial), _sample_final(initial)):
        result = evaluate_predicate(pred, initial, final)
        assert isinstance(result, tuple)
        assert len(result) == 2
        passed, detail = result
        assert isinstance(passed, bool)
        assert isinstance(detail, str)
        assert detail != ""


@pytest.mark.parametrize("pred", SAMPLE_PREDICATES, ids=lambda p: p["type"])
def test_evaluate_is_deterministic(initial: IssueTrackerState, pred: dict[str, Any]) -> None:
    final = _sample_final(initial)
    first = evaluate_predicate(pred, initial, final)
    assert evaluate_predicate(pred, initial, final) == first
    assert evaluate_predicate(copy.deepcopy(pred), _clone(initial), _clone(final)) == first


@pytest.mark.parametrize("pred", SAMPLE_PREDICATES, ids=lambda p: p["type"])
def test_evaluate_does_not_mutate_inputs(initial: IssueTrackerState, pred: dict[str, Any]) -> None:
    final = _sample_final(initial)
    pred_before = copy.deepcopy(pred)
    initial_before = copy.deepcopy(initial)
    final_before = copy.deepcopy(final)
    evaluate_predicate(pred, initial, final)
    assert pred == pred_before
    assert initial == initial_before
    assert final == final_before


@pytest.mark.parametrize("pred", SAMPLE_PREDICATES, ids=lambda p: p["type"])
def test_evaluate_ignores_description(initial: IssueTrackerState, pred: dict[str, Any]) -> None:
    final = _sample_final(initial)
    with_description = _mod(pred, description="only for humans")
    assert _passed(with_description, initial, final) is _passed(pred, initial, final)


def test_sample_predicates_expected_results(initial: IssueTrackerState) -> None:
    final = _sample_final(initial)
    results = [_passed(p, initial, final) for p in SAMPLE_PREDICATES]
    assert results == [True, False, True, True, True, False, True, True, False]


@pytest.mark.parametrize("ptype", PREDICATE_TYPES)
def test_evaluate_unknown_issue_raises_task_error(initial: IssueTrackerState, ptype: str) -> None:
    pred = _mod(VALID_PREDICATES[ptype], issue_id=99)
    with pytest.raises(TaskError):
        evaluate_predicate(pred, initial, _clone(initial))


@pytest.mark.parametrize("ptype", PREDICATE_TYPES)
@pytest.mark.parametrize("missing_in", ["initial", "final"])
def test_evaluate_issue_missing_in_one_state_raises_task_error(
    initial: IssueTrackerState, ptype: str, missing_in: str
) -> None:
    pred = _mod(VALID_PREDICATES[ptype], issue_id=4)
    without = _clone(initial)
    del without.issues[4]
    a, b = (without, initial) if missing_in == "initial" else (initial, without)
    with pytest.raises(TaskError):
        evaluate_predicate(pred, a, b)


@pytest.mark.parametrize(
    "pred",
    [
        {"type": "issue_field_not_equals", "issue_id": 1, "field": "status", "value": "open"},
        {"type": "forbid_change", "issue_ids": "*", "fields": "*"},
        {"type": "", "issue_id": 1},
    ],
    ids=["unknown", "forbid-rule", "empty"],
)
def test_evaluate_unknown_type_raises_task_error(initial: IssueTrackerState, pred: dict[str, Any]) -> None:
    with pytest.raises(TaskError):
        evaluate_predicate(pred, initial, _clone(initial))


# ---------------------------------------------------------------------------
# forbid_rule_matches
# ---------------------------------------------------------------------------


def _field_change(issue_id: int, field: str, before: Any, after: Any) -> Change:
    return Change(issue_id=issue_id, field=field, before=before, after=after)


def _comment_change(issue_id: int, cid: int = 3, body: str = "hello") -> Change:
    return Change(
        issue_id=issue_id,
        field=COMMENTS_FIELD,
        before=None,
        after={"id": cid, "issue_id": issue_id, "author_id": SMALL_ACTOR, "body": body},
    )


SAMPLE_CHANGES: list[Change] = [
    _field_change(1, "title", "Crash on login", "Crash on logout"),
    _field_change(1, "body", "old", ""),
    _field_change(1, "status", "open", "closed"),
    _field_change(2, "priority", "p2", "p0"),
    _field_change(3, "assignee_id", "u2", None),
    _field_change(1, "labels", ["bug", "ui"], ["bug"]),
    _comment_change(4),
]


def _rule(issue_ids: Any, fields: Any, to_values: Any = None) -> dict[str, Any]:
    rule: dict[str, Any] = {"type": "forbid_change", "issue_ids": issue_ids, "fields": fields}
    if to_values is not None:
        rule["to_values"] = to_values
    return rule


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_star_star_rule_matches_every_change_including_comments(change: Change) -> None:
    assert forbid_rule_matches(_rule("*", "*"), change) is True


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_star_fields_rule_matches_only_listed_issues(change: Change) -> None:
    assert forbid_rule_matches(_rule([1], "*"), change) is (change.issue_id == 1)
    assert forbid_rule_matches(_rule([2, 4], "*"), change) is (change.issue_id in (2, 4))


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_star_issue_ids_rule_matches_only_listed_fields(change: Change) -> None:
    assert forbid_rule_matches(_rule("*", ["status"]), change) is (change.field == "status")
    assert forbid_rule_matches(_rule("*", ["comments"]), change) is (change.field == COMMENTS_FIELD)
    assert forbid_rule_matches(_rule("*", ["labels", "assignee_id"]), change) is (
        change.field in ("labels", "assignee_id")
    )


@pytest.mark.parametrize(
    "rule, change, expected",
    [
        (_rule([1], ["status"]), _field_change(1, "status", "open", "closed"), True),
        (_rule([2], ["status"]), _field_change(1, "status", "open", "closed"), False),
        (_rule([1], ["priority"]), _field_change(1, "status", "open", "closed"), False),
        (_rule([1, 3], ["status", "priority"]), _field_change(3, "priority", "p0", "p3"), True),
        (_rule([1, 3], ["status", "priority"]), _field_change(2, "priority", "p2", "p3"), False),
        (_rule([4], ["comments"]), _comment_change(4), True),
        (_rule([4], ["status"]), _comment_change(4), False),
        (_rule([1], ["comments"]), _comment_change(4), False),
    ],
)
def test_forbid_rule_with_specific_lists(rule: dict[str, Any], change: Change, expected: bool) -> None:
    assert forbid_rule_matches(rule, change) is expected


@pytest.mark.parametrize(
    "rule, change, expected",
    [
        (_rule("*", ["status"], ["closed"]), _field_change(1, "status", "open", "closed"), True),
        (_rule("*", ["status"], ["closed"]), _field_change(1, "status", "open", "resolved"), False),
        (_rule("*", ["status"], ["closed"]), _field_change(1, "status", "closed", "open"), False),
        (_rule("*", ["status"], ["resolved", "closed"]), _field_change(4, "status", "open", "resolved"), True),
        (_rule("*", ["assignee_id"], [None]), _field_change(2, "assignee_id", "u1", None), True),
        (_rule("*", ["assignee_id"], [None]), _field_change(2, "assignee_id", "u1", "u3"), False),
        (_rule("*", ["assignee_id"], ["u2"]), _field_change(1, "assignee_id", None, "u2"), True),
        (_rule("*", ["assignee_id"], ["u2"]), _field_change(3, "assignee_id", "u2", "u1"), False),
        (_rule("*", ["assignee_id"], ["u2"]), _field_change(1, "assignee_id", "u2", None), False),
        (_rule("*", ["assignee_id"], ["u2", None]), _field_change(1, "assignee_id", "u1", None), True),
        (_rule([1], ["status"], ["closed"]), _field_change(2, "status", "open", "closed"), False),
        (_rule("*", ["status"], ["closed"]), _field_change(1, "priority", "p1", "p0"), False),
        (_rule("*", ["body"], [""]), _field_change(1, "body", "old", ""), True),
        (_rule("*", ["status"], ["closed"]), _comment_change(1, body="closed"), False),
    ],
)
def test_forbid_rule_to_values_match_change_after(rule: dict[str, Any], change: Change, expected: bool) -> None:
    assert forbid_rule_matches(rule, change) is expected


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_forbid_rule_without_to_values_matches_any_after(change: Change) -> None:
    assert forbid_rule_matches(_rule([change.issue_id], [change.field]), change) is True


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_forbid_rule_ignores_description(change: Change) -> None:
    for rule in (_rule("*", "*"), _rule([2], ["priority"]), _rule("*", ["status"], ["closed"])):
        described = _mod(rule, description="never do this")
        assert forbid_rule_matches(described, change) is forbid_rule_matches(rule, change)


def test_forbid_rule_matches_does_not_mutate_inputs() -> None:
    rule = _rule([1, 2], ["status", "labels"])
    change = _field_change(1, "labels", ["bug", "ui"], ["bug"])
    rule_before = copy.deepcopy(rule)
    change_before = copy.deepcopy(change)
    forbid_rule_matches(rule, change)
    assert rule == rule_before
    assert change == change_before


# ---------------------------------------------------------------------------
# change_in_scope
# ---------------------------------------------------------------------------


SCOPE = (
    ScopeEntry(1, ("status", "assignee_id")),
    ScopeEntry(3, ("comments",)),
    ScopeEntry(4, ("priority", "labels", "comments")),
)


@pytest.mark.parametrize(
    "change, expected",
    [
        (_field_change(1, "status", "open", "closed"), True),
        (_field_change(1, "assignee_id", None, "u1"), True),
        (_field_change(1, "priority", "p1", "p0"), False),
        (_field_change(1, "labels", ["bug", "ui"], ["bug"]), False),
        (_comment_change(1), False),
        (_field_change(2, "status", "in_progress", "closed"), False),
        (_comment_change(2), False),
        (_comment_change(3), True),
        (_field_change(3, "status", "open", "closed"), False),
        (_field_change(4, "priority", "p3", "p0"), True),
        (_field_change(4, "labels", [], ["bug"]), True),
        (_comment_change(4), True),
        (_field_change(4, "status", "closed", "open"), False),
    ],
)
def test_change_in_scope_is_field_level(change: Change, expected: bool) -> None:
    assert change_in_scope(SCOPE, change) is expected
    assert change_in_scope(list(SCOPE), change) is expected


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_empty_scope_contains_no_change(change: Change) -> None:
    assert change_in_scope((), change) is False


@pytest.mark.parametrize("change", SAMPLE_CHANGES, ids=lambda c: f"{c.issue_id}-{c.field}")
def test_full_scope_contains_every_change(change: Change) -> None:
    scope = tuple(ScopeEntry(i, F3_FIELDS) for i in (1, 2, 3, 4))
    assert change_in_scope(scope, change) is True


@pytest.mark.parametrize(
    "params",
    [
        pytest.param([], id="list"),
        pytest.param("x", id="str"),
        pytest.param(1, id="int"),
        pytest.param(None, id="none"),
    ],
)
def test_params_top_level_must_be_dict(assert_invalid: Callable, params: Any) -> None:
    d = _task_dict()
    d["params"] = params
    assert_invalid(d)
