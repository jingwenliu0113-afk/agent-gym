"""snapshot / restore / state_hash 與核心 replay invariant（decision log 002、env/environment.py）。"""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Callable

import pytest

from apps.issue_tracker.schema import (
    PRIORITIES,
    SCHEMA_VERSION,
    STATUSES,
    IssueTrackerState,
    SchemaError,
)
from apps.issue_tracker.tools import ERROR_CODES
from env.environment import (
    SNAPSHOT_FORMAT,
    SNAPSHOT_KEYS,
    IssueTrackerEnv,
    SnapshotError,
    state_hash_of,
)
from env.hashing import HASH_PREFIX, canonical_json, sha256_of
from tests.conftest import SMALL_ACTOR, SMALL_WORLD

Call = tuple[str, Any]

_ISSUE_IDS = [issue["id"] for issue in SMALL_WORLD["issues"]]
_USER_IDS = [user["id"] for user in SMALL_WORLD["users"]]
_LABEL_NAMES = [label["name"] for label in SMALL_WORLD["labels"]]
_MISSING_ISSUE_ID = 99
_SEEDS = [0, 1, 2, 3, 17, 42, 2026, 31337]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _new_env(world: dict[str, Any] | None = None, actor: str = SMALL_ACTOR) -> IssueTrackerEnv:
    env = IssueTrackerEnv()
    env.reset(copy.deepcopy(SMALL_WORLD if world is None else world), actor)
    return env


def _run(env: IssueTrackerEnv, calls: list[Call]) -> list[dict[str, Any]]:
    return [env.call_tool(name, copy.deepcopy(args)) for name, args in calls]


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _canonical_roundtrip(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _expected_hash(state: dict[str, Any], actor_id: Any) -> str:
    """state_hash_of 的契約公式，直接對 canonical state dict 計算。"""
    return sha256_of({"schema_version": SCHEMA_VERSION, "actor_id": actor_id, "state": state})


def _assert_json_native(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, f"{path}: non-str key {key!r}"
            _assert_json_native(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_native(item, f"{path}[{index}]")
    else:
        assert value is None or type(value) in (str, int, bool), (
            f"{path}: non JSON-native type {type(value).__name__}"
        )


def _assert_uninitialized(env: IssueTrackerEnv) -> None:
    for attr in ("step", "actor_id", "records"):
        with pytest.raises(RuntimeError):
            getattr(env, attr)
    for method in ("snapshot", "state_hash", "current_state"):
        with pytest.raises(RuntimeError):
            getattr(env, method)()
    with pytest.raises(RuntimeError):
        env.call_tool("get_issue", {"issue_id": 1})


def _env_view(env: IssueTrackerEnv) -> tuple[Any, ...]:
    return (
        env.snapshot(),
        env.records,
        env.step,
        env.actor_id,
        env.state_hash(),
        env.current_state().to_dict(),
    )


def _check_replay_invariant(
    prefix: list[Call], suffix: list[Call], target: IssueTrackerEnv | None = None
) -> tuple[IssueTrackerEnv, IssueTrackerEnv, dict[str, Any]]:
    """environment.py 模組 docstring 的三步檢驗程序。"""
    env_a = _new_env()
    _run(env_a, prefix)
    snap = env_a.snapshot()
    snap_copy = copy.deepcopy(snap)

    env_b = IssueTrackerEnv() if target is None else target
    env_b.restore(_json_roundtrip(snap))
    assert env_b.snapshot() == snap
    assert env_b.records == ()

    obs_a = _run(env_a, suffix)
    obs_b = _run(env_b, suffix)
    assert len(obs_a) == len(obs_b) == len(suffix)
    for index, (seen_a, seen_b) in enumerate(zip(obs_a, obs_b)):
        assert seen_a == seen_b, f"observation differs at suffix[{index}] = {suffix[index]!r}"

    assert env_a.records[len(prefix):] == env_b.records
    assert [r.step for r in env_b.records] == list(
        range(snap["step"], snap["step"] + len(suffix))
    )
    assert env_a.records[: snap["step"]] + env_b.records == env_a.records
    assert env_a.snapshot() == env_b.snapshot()
    assert env_a.step == env_b.step == len(prefix) + len(suffix)
    assert snap == snap_copy
    return env_a, env_b, snap


# ---------------------------------------------------------------------------
# 固定的 call 序列
# ---------------------------------------------------------------------------

_MUTATIONS: list[Call] = [
    ("update_issue", {"issue_id": 1, "status": "in_progress", "priority": "p0"}),
    ("update_issue", {"issue_id": 3, "labels": ["security"]}),
    ("assign_issue", {"issue_id": 1, "assignee_id": "u3"}),
    ("assign_issue", {"issue_id": 2, "assignee_id": None}),
    ("add_comment", {"issue_id": 4, "body": "Reopening soon."}),
    (
        "update_issue",
        {"issue_id": 4, "title": "清理舊功能旗標", "body": "", "status": "open", "labels": ["ui", "bug", "ui"]},
    ),
    ("assign_issue", {"issue_id": 4, "assignee_id": "u2"}),
]

_ERRORS: list[Call] = [
    ("delete_issue", {"issue_id": 1}),
    ("get_issue", {"issue_id": _MISSING_ISSUE_ID}),
    ("get_issue", {"issue_id": True}),
    ("update_issue", {"issue_id": 1}),
    ("update_issue", {"issue_id": 2, "labels": ["wontfix"]}),
    ("assign_issue", {"issue_id": 3, "assignee_id": "ghost"}),
    ("add_comment", {"issue_id": 1, "body": ""}),
    ("search_issues", {"label": "wontfix"}),
    ("search_issues", ["not", "an", "object"]),
    ("add_comment", None),
]

_SEARCH_FILTERS: list[dict[str, Any]] = [
    {},
    {"status": "open"},
    {"status": "in_progress"},
    {"priority": "p0"},
    {"label": "bug"},
    {"label": "ui", "status": "open"},
    {"assignee_id": "unassigned"},
    {"assignee_id": "u2"},
    {"assignee_id": "u3"},
    {"query": "login"},
    {"query": "LOGIN", "limit": 1},
    {"query": "登入"},
    {"limit": 2},
    {"status": "closed", "assignee_id": "u2", "priority": "p3"},
    {"query": "zzz-no-match"},
]

_READ_CALLS: list[Call] = [("search_issues", f) for f in _SEARCH_FILTERS] + [
    ("get_issue", {"issue_id": issue_id}) for issue_id in _ISSUE_IDS
]
_READ_CALL_IDS = [f"{i:02d}-{name}" for i, (name, _) in enumerate(_READ_CALLS)]

_MIXED: list[Call] = [
    _MUTATIONS[0],
    _ERRORS[0],
    ("get_issue", {"issue_id": 1}),
    _MUTATIONS[1],
    _ERRORS[4],
    ("search_issues", {"label": "security"}),
    _MUTATIONS[2],
    _ERRORS[8],
    _MUTATIONS[4],
    ("add_comment", {"issue_id": 1, "body": "已確認，處理中"}),
    _ERRORS[5],
    _MUTATIONS[5],
    ("search_issues", {"assignee_id": "unassigned"}),
    _MUTATIONS[3],
]

_SOURCE_PREFIX: list[Call] = [
    ("update_issue", {"issue_id": 1, "status": "in_progress", "labels": ["bug"]}),
    ("add_comment", {"issue_id": 2, "body": "Checking the font stack."}),
    ("assign_issue", {"issue_id": 1, "assignee_id": "u3"}),
    ("get_issue", {"issue_id": _MISSING_ISSUE_ID}),
]


def _source_snapshot() -> dict[str, Any]:
    """合法 snapshot：actor=bot、step=4、state 已被修改。"""
    env = _new_env()
    _run(env, _SOURCE_PREFIX)
    return env.snapshot()


def _initialized_target() -> IssueTrackerEnv:
    """與 source 不同 actor / step / state 的已初始化 env。"""
    env = _new_env(actor="u3")
    _run(
        env,
        [
            ("assign_issue", {"issue_id": 4, "assignee_id": None}),
            ("update_issue", {"issue_id": 2, "priority": "p0"}),
            ("add_comment", {"issue_id": 3, "body": "Target env comment."}),
            ("search_issues", {"status": "open"}),
        ],
    )
    return env


# ---------------------------------------------------------------------------
# snapshot 結構
# ---------------------------------------------------------------------------


def test_snapshot_has_exactly_snapshot_keys():
    assert set(_new_env().snapshot()) == SNAPSHOT_KEYS


def test_snapshot_after_reset_has_expected_values():
    snap = _new_env().snapshot()
    assert snap["format"] == SNAPSHOT_FORMAT
    assert snap["schema_version"] == SCHEMA_VERSION
    assert snap["actor_id"] == SMALL_ACTOR
    assert snap["step"] == 0
    assert snap["state"] == SMALL_WORLD
    assert snap["state_hash"] == _expected_hash(SMALL_WORLD, SMALL_ACTOR)


@pytest.mark.parametrize("prefix_len", [0, 1, 3, len(_MIXED)])
def test_snapshot_fields_match_env_after_calls(prefix_len):
    env = _new_env(actor="u3")
    _run(env, _MIXED[:prefix_len])
    snap = env.snapshot()
    assert snap["format"] == SNAPSHOT_FORMAT
    assert snap["schema_version"] == SCHEMA_VERSION
    assert snap["actor_id"] == env.actor_id == "u3"
    assert snap["step"] == env.step == prefix_len
    assert snap["state"] == env.current_state().to_dict()
    assert snap["state_hash"] == env.state_hash()
    assert snap["state_hash"] == state_hash_of(env.current_state(), "u3")
    assert snap["state_hash"] == _expected_hash(snap["state"], "u3")


def test_snapshot_state_hash_is_prefixed_sha256_hex():
    state_hash = _new_env().snapshot()["state_hash"]
    assert state_hash.startswith(HASH_PREFIX)
    digest = state_hash[len(HASH_PREFIX):]
    assert len(digest) == 64
    int(digest, 16)


def test_snapshot_step_is_step_of_next_record():
    env = _new_env()
    _run(env, _MIXED[:3])
    snap = env.snapshot()
    env.call_tool("search_issues", {})
    assert env.records[-1].step == snap["step"]
    assert env.step == snap["step"] + 1


def test_snapshot_of_env_reset_from_non_canonical_input_is_canonical(small_world):
    small_world["issues"].reverse()
    small_world["users"].reverse()
    small_world["issues"][-1]["labels"] = ["ui", "bug", "ui"]
    snap = _new_env(small_world).snapshot()
    assert snap["state"] == SMALL_WORLD
    restored = IssueTrackerEnv()
    restored.restore(snap)
    assert restored.snapshot() == snap


@pytest.mark.parametrize("prefix_len", [0, len(_MIXED)])
def test_snapshot_is_json_native(prefix_len):
    env = _new_env()
    _run(env, _MIXED[:prefix_len])
    snap = env.snapshot()
    _assert_json_native(snap)
    assert json.loads(json.dumps(snap)) == snap
    assert json.loads(canonical_json(snap)) == snap


def test_mutating_returned_snapshot_does_not_change_env():
    env = _new_env()
    _run(env, _MUTATIONS[:2])
    expected = copy.deepcopy(env.snapshot())
    snap = env.snapshot()
    snap["step"] = 99
    snap["actor_id"] = "u1"
    snap["state_hash"] = HASH_PREFIX + "0" * 64
    snap["state"]["issues"][0]["title"] = "hacked"
    snap["state"]["issues"][0]["labels"].append("security")
    snap["state"]["users"][0]["active"] = False
    snap["state"]["comments"].clear()
    snap["state"]["labels"].pop()
    assert env.snapshot() == expected
    assert env.state_hash() == expected["state_hash"]
    assert env.current_state().to_dict() == expected["state"]
    assert env.step == expected["step"]
    assert env.actor_id == expected["actor_id"]


def test_later_env_mutations_do_not_change_earlier_snapshot():
    env = _new_env()
    snap = env.snapshot()
    frozen = copy.deepcopy(snap)
    _run(env, _MUTATIONS)
    assert snap == frozen


def test_consecutive_snapshots_are_equal_but_independent():
    env = _new_env()
    first = env.snapshot()
    second = env.snapshot()
    assert first == second
    first["state"]["issues"][1]["labels"].append("bug")
    first["state"]["comments"].append({"id": 9, "issue_id": 1, "author_id": "u1", "body": "x"})
    assert second["state"]["issues"][1]["labels"] == ["ui"]
    assert len(second["state"]["comments"]) == 2


def test_snapshot_does_not_advance_step_or_records():
    env = _new_env()
    _run(env, _MIXED[:2])
    records, step, state_hash = env.records, env.step, env.state_hash()
    env.snapshot()
    env.snapshot()
    assert env.step == step
    assert env.records == records
    assert env.state_hash() == state_hash


@pytest.mark.parametrize(
    "call",
    [
        ("get_issue", {"issue_id": 1}),
        ("search_issues", {"label": "bug"}),
        ("update_issue", {"issue_id": 1, "status": "open"}),
        ("assign_issue", {"issue_id": 1, "assignee_id": None}),
        ("get_issue", {"issue_id": _MISSING_ISSUE_ID}),
        ("delete_issue", {}),
        ("update_issue", {"issue_id": 1, "title": ""}),
        ("add_comment", None),
    ],
    ids=[
        "read_get",
        "read_search",
        "noop_update",
        "noop_assign",
        "not_found",
        "unknown_tool",
        "invalid_arguments",
        "non_dict_args",
    ],
)
def test_state_preserving_call_only_advances_snapshot_step(call):
    env = _new_env()
    before = env.snapshot()
    _run(env, [call])
    after = env.snapshot()
    assert after["step"] == before["step"] + 1
    assert {k: v for k, v in after.items() if k != "step"} == {
        k: v for k, v in before.items() if k != "step"
    }


def test_failed_calls_advance_step_but_keep_initial_state():
    env = _new_env()
    _run(env, _ERRORS)
    snap = env.snapshot()
    assert snap["step"] == len(_ERRORS)
    assert snap["state"] == SMALL_WORLD
    assert snap["state_hash"] == _expected_hash(SMALL_WORLD, SMALL_ACTOR)


def test_uninitialized_env_rejects_snapshot_and_state_access():
    _assert_uninitialized(IssueTrackerEnv())


# ---------------------------------------------------------------------------
# restore 基本行為
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "serialize",
    [copy.deepcopy, _json_roundtrip, _canonical_roundtrip],
    ids=["deepcopy", "json_roundtrip", "canonical_json_roundtrip"],
)
def test_restore_on_fresh_env_reproduces_snapshot(serialize):
    source = _new_env()
    _run(source, _MIXED)
    snap = source.snapshot()
    env = IssueTrackerEnv()
    env.restore(serialize(snap))
    assert env.snapshot() == snap
    assert env.records == ()
    assert env.step == snap["step"] == len(_MIXED)
    assert env.actor_id == snap["actor_id"]
    assert env.state_hash() == snap["state_hash"] == source.state_hash()
    assert env.current_state().to_dict() == snap["state"]


def test_restore_into_initialized_env_replaces_everything_and_clears_records():
    snap = _source_snapshot()
    env = _initialized_target()
    assert env.records != ()
    env.restore(snap)
    assert env.snapshot() == snap
    assert env.records == ()
    assert env.actor_id == SMALL_ACTOR
    assert env.step == len(_SOURCE_PREFIX)


def test_env_can_restore_its_own_snapshot():
    env = _new_env()
    _run(env, _MIXED[:5])
    snap = env.snapshot()
    env.restore(snap)
    assert env.snapshot() == snap
    assert env.records == ()
    assert env.step == 5


def test_first_call_after_restore_continues_step_numbering():
    snap = _source_snapshot()
    env = IssueTrackerEnv()
    env.restore(snap)
    env.call_tool("get_issue", {"issue_id": 1})
    assert len(env.records) == 1
    assert env.records[0].step == snap["step"]
    assert env.step == snap["step"] + 1


def test_restore_and_later_calls_do_not_mutate_snapshot():
    snap = _source_snapshot()
    frozen = copy.deepcopy(snap)
    env = IssueTrackerEnv()
    env.restore(snap)
    _run(env, _MUTATIONS + _ERRORS)
    env.restore(snap)
    _run(env, _MUTATIONS)
    assert snap == frozen


def test_restore_accepts_hand_built_consistent_snapshot(small_world):
    small_world["issues"][3]["status"] = "open"
    snap = {
        "format": SNAPSHOT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "actor_id": "u1",
        "step": 7,
        "state": small_world,
        "state_hash": _expected_hash(small_world, "u1"),
    }
    env = IssueTrackerEnv()
    env.restore(copy.deepcopy(snap))
    assert env.snapshot() == snap
    assert env.actor_id == "u1"
    assert env.step == 7
    assert env.current_state().to_dict() == small_world


@pytest.mark.parametrize("step", [0, 1, 1000])
def test_restore_accepts_any_non_negative_step_because_step_is_not_hashed(step):
    snap = _source_snapshot()
    snap["step"] = step
    env = IssueTrackerEnv()
    env.restore(snap)
    assert env.step == step
    assert env.snapshot() == snap
    env.call_tool("search_issues", {})
    assert env.records[0].step == step
    assert env.step == step + 1


def test_restore_restores_actor_used_by_add_comment():
    snap = _new_env(actor="u3").snapshot()
    env = _new_env(actor=SMALL_ACTOR)
    env.restore(snap)
    assert env.actor_id == "u3"
    observation = env.call_tool("add_comment", {"issue_id": 1, "body": "hi"})
    assert observation["data"]["comment"]["author_id"] == "u3"


def test_comment_ids_after_rollback_are_derived_from_restored_comments():
    env = _new_env()
    snap0 = env.snapshot()
    first = env.call_tool("add_comment", {"issue_id": 1, "body": "first"})
    second = env.call_tool("add_comment", {"issue_id": 1, "body": "second"})
    assert first["data"]["comment"]["id"] == 3
    assert second["data"]["comment"]["id"] == 4
    env.restore(snap0)
    again = env.call_tool("add_comment", {"issue_id": 2, "body": "after rollback"})
    assert again["data"]["comment"]["id"] == 3


def test_restoring_earlier_snapshot_after_more_mutations_rolls_back_exactly():
    env = _new_env()
    _run(env, _MUTATIONS[:2])
    snap = env.snapshot()
    old_hash = env.state_hash()
    _run(env, _MUTATIONS[2:] + _ERRORS)
    assert env.state_hash() != old_hash
    env.restore(snap)
    assert env.state_hash() == old_hash
    assert env.snapshot() == snap
    assert env.current_state().to_dict() == snap["state"]
    assert env.step == snap["step"]
    assert env.records == ()


def test_restoring_reset_snapshot_returns_to_initial_world():
    env = _new_env()
    snap0 = env.snapshot()
    _run(env, _MIXED)
    env.restore(snap0)
    assert env.current_state().to_dict() == SMALL_WORLD
    assert env.state_hash() == _expected_hash(SMALL_WORLD, SMALL_ACTOR)
    assert env.step == 0


def test_restoring_same_snapshot_repeatedly_yields_identical_records():
    env_a = _new_env()
    _run(env_a, _MUTATIONS[:3])
    snap = env_a.snapshot()
    frozen = copy.deepcopy(snap)
    suffix = _MUTATIONS[3:] + _ERRORS + _READ_CALLS[:4]
    shared = IssueTrackerEnv()
    results = []
    for env in (shared, shared, IssueTrackerEnv(), _new_env(actor="u3"), env_a):
        env.restore(snap)
        observations = _run(env, suffix)
        results.append((observations, env.records, env.snapshot()))
    for result in results[1:]:
        assert result == results[0]
    assert snap == frozen


# ---------------------------------------------------------------------------
# 核心 invariant：固定序列
# ---------------------------------------------------------------------------

_INVARIANT_CASES: dict[str, tuple[list[Call], list[Call]]] = {
    "both_empty": ([], []),
    "empty_prefix": ([], _MIXED),
    "empty_suffix": (_MIXED, []),
    "mutations_then_reads": (_MUTATIONS, _READ_CALLS),
    "reads_then_mutations": (_READ_CALLS, _MUTATIONS),
    "errors_then_mutations": (_ERRORS, _MUTATIONS),
    "mutations_then_errors": (_MUTATIONS, _ERRORS),
    "mixed_split_mid": (_MIXED[:7], _MIXED[7:]),
    "comments_on_both_sides": (
        [("add_comment", {"issue_id": 1, "body": "a"}), ("add_comment", {"issue_id": 3, "body": "b"})],
        [("add_comment", {"issue_id": 1, "body": "c"}), ("get_issue", {"issue_id": 1})],
    ),
}


@pytest.mark.parametrize(
    ("prefix", "suffix"), list(_INVARIANT_CASES.values()), ids=list(_INVARIANT_CASES)
)
def test_replay_invariant_on_fixed_sequences_with_fresh_env(prefix, suffix):
    _check_replay_invariant(prefix, suffix)


@pytest.mark.parametrize(
    ("prefix", "suffix"), list(_INVARIANT_CASES.values()), ids=list(_INVARIANT_CASES)
)
def test_replay_invariant_on_fixed_sequences_with_initialized_env(prefix, suffix):
    _check_replay_invariant(prefix, suffix, _initialized_target())


@pytest.mark.parametrize("call", _READ_CALLS, ids=_READ_CALL_IDS)
def test_read_call_observation_identical_after_restore(call):
    env_a = _new_env()
    _run(env_a, _MUTATIONS)
    snap = env_a.snapshot()
    env_b = IssueTrackerEnv()
    env_b.restore(_json_roundtrip(snap))
    obs_a = _run(env_a, [call])[0]
    obs_b = _run(env_b, [call])[0]
    assert obs_a["ok"] is True
    assert obs_b == obs_a
    assert env_b.records == env_a.records[-1:]
    assert env_b.state_hash() == env_a.state_hash() == snap["state_hash"]


def test_read_calls_cover_every_issue_in_snapshot():
    snap = _source_snapshot()
    issue_ids = {args["issue_id"] for name, args in _READ_CALLS if name == "get_issue"}
    assert issue_ids == {issue["id"] for issue in snap["state"]["issues"]}


def test_read_observations_identical_after_rolling_back_same_env():
    env = _new_env()
    _run(env, _MUTATIONS[:3])
    snap = env.snapshot()
    before = _run(env, _READ_CALLS)
    _run(env, _MUTATIONS[3:])
    assert _run(env, _READ_CALLS) != before
    env.restore(snap)
    assert _run(env, _READ_CALLS) == before


# ---------------------------------------------------------------------------
# 核心 invariant：隨機序列
# ---------------------------------------------------------------------------


def _rand_issue_id(rng: random.Random) -> Any:
    roll = rng.random()
    if roll < 0.8:
        return rng.choice(_ISSUE_IDS)
    if roll < 0.9:
        return _MISSING_ISSUE_ID
    return rng.choice(["1", True, 2.0, None])


def _rand_search(rng: random.Random) -> Call:
    args: dict[str, Any] = {}
    if rng.random() < 0.3:
        args["query"] = rng.choice(["login", "LOGIN", "token", "登入", "crash", "zzz-no-match", ""])
    if rng.random() < 0.3:
        args["status"] = rng.choice(STATUSES + ("Open",))
    if rng.random() < 0.3:
        args["priority"] = rng.choice(PRIORITIES)
    if rng.random() < 0.3:
        args["label"] = rng.choice(_LABEL_NAMES + ["missing-label"])
    if rng.random() < 0.3:
        args["assignee_id"] = rng.choice(_USER_IDS + ["unassigned", "ghost"])
    if rng.random() < 0.3:
        args["limit"] = rng.choice([1, 2, 3, 50, 0, 51, "5"])
    return "search_issues", args


def _rand_get(rng: random.Random) -> Call:
    return "get_issue", {"issue_id": _rand_issue_id(rng)}


def _rand_update(rng: random.Random) -> Call:
    args: dict[str, Any] = {"issue_id": _rand_issue_id(rng)}
    if rng.random() < 0.4:
        args["title"] = rng.choice(["Renamed", "Crash on login", "標題", ""])
    if rng.random() < 0.3:
        args["body"] = rng.choice(["", "New body", "Body mentioning LOGIN"])
    if rng.random() < 0.4:
        args["status"] = rng.choice(STATUSES + ("done",))
    if rng.random() < 0.4:
        args["priority"] = rng.choice(PRIORITIES)
    if rng.random() < 0.4:
        labels = rng.sample(_LABEL_NAMES, rng.randint(0, len(_LABEL_NAMES)))
        roll = rng.random()
        if roll < 0.1:
            labels.append("missing-label")
        elif roll < 0.2:
            labels = labels + labels
        elif roll < 0.25:
            labels.append("")
        args["labels"] = labels
    if rng.random() < 0.05:
        args["assignee_id"] = "u1"
    return "update_issue", args


def _rand_comment(rng: random.Random) -> Call:
    args: dict[str, Any] = {"issue_id": _rand_issue_id(rng)}
    if rng.random() < 0.9:
        args["body"] = rng.choice(["Looking into it.", "已確認，處理中", "x", ""])
    return "add_comment", args


def _rand_assign(rng: random.Random) -> Call:
    args: dict[str, Any] = {"issue_id": _rand_issue_id(rng)}
    if rng.random() < 0.9:
        args["assignee_id"] = rng.choice(_USER_IDS + [None, None, "ghost", "", "unassigned"])
    return "assign_issue", args


def _rand_malformed(rng: random.Random) -> Call:
    return copy.deepcopy(
        rng.choice(
            [
                ("delete_issue", {"issue_id": 1}),
                ("Search_Issues", {}),
                ("get_issue", [1]),
                ("search_issues", None),
                ("add_comment", "please comment"),
                ("get_issue", {"issue_id": 1, "verbose": True}),
                ("assign_issue", {"issue_id": 1, "assignee_id": "u1", "notify": False}),
            ]
        )
    )


_GENERATORS: list[Callable[[random.Random], Call]] = [
    _rand_search,
    _rand_get,
    _rand_update,
    _rand_update,
    _rand_comment,
    _rand_assign,
    _rand_assign,
    _rand_malformed,
]

# 保證每條隨機序列都涵蓋三種錯誤碼、成功的 mutation 與成功的 read。
_COVERAGE_CALLS: list[Call] = [
    ("delete_issue", {"issue_id": 1}),
    ("get_issue", {"issue_id": "1"}),
    ("get_issue", {"issue_id": _MISSING_ISSUE_ID}),
    ("add_comment", {"issue_id": 2, "body": "coverage"}),
    ("search_issues", {}),
]


def _random_calls(seed: int, n: int = 30) -> tuple[random.Random, list[Call]]:
    rng = random.Random(seed)
    calls = [rng.choice(_GENERATORS)(rng) for _ in range(n)] + copy.deepcopy(_COVERAGE_CALLS)
    rng.shuffle(calls)
    return rng, calls


def _assert_mixed(records: tuple[Any, ...]) -> None:
    assert {r.error["code"] for r in records if not r.ok} == set(ERROR_CODES)
    assert any(r.ok and r.changes for r in records)
    assert any(r.ok and not r.changes for r in records)


@pytest.mark.parametrize("target", ["fresh", "initialized"])
@pytest.mark.parametrize("seed", _SEEDS)
def test_random_trajectory_satisfies_replay_invariant(seed, target):
    rng, calls = _random_calls(seed)
    split = rng.randint(0, len(calls))
    env_b = IssueTrackerEnv() if target == "fresh" else _initialized_target()
    env_a, _, _ = _check_replay_invariant(calls[:split], calls[split:], env_b)
    _assert_mixed(env_a.records)


@pytest.mark.parametrize("seed", _SEEDS[:4])
def test_random_trajectory_every_checkpoint_is_restorable(seed):
    _, calls = _random_calls(seed, n=20)
    env = _new_env()
    snaps = [env.snapshot()]
    for name, args in calls:
        env.call_tool(name, copy.deepcopy(args))
        snaps.append(env.snapshot())
    full = env.records
    for index, record in enumerate(full):
        assert record.step == snaps[index]["step"] == index
        assert record.state_hash_before == snaps[index]["state_hash"]
        assert record.state_hash_after == snaps[index + 1]["state_hash"]
    for index, snap in enumerate(snaps):
        replay = IssueTrackerEnv()
        replay.restore(_json_roundtrip(snap))
        assert replay.snapshot() == snap
        _run(replay, calls[index:])
        assert full[:index] + replay.records == full
        assert replay.snapshot() == snaps[-1]


@pytest.mark.parametrize("seed", _SEEDS[:4])
def test_random_rollback_on_same_env_restores_exact_hash(seed):
    rng, calls = _random_calls(seed, n=20)
    env = _new_env()
    snaps = []
    for name, args in calls:
        snaps.append(env.snapshot())
        env.call_tool(name, copy.deepcopy(args))
    snaps.append(env.snapshot())
    frozen = copy.deepcopy(snaps)
    for index in rng.sample(range(len(snaps)), 6):
        env.restore(snaps[index])
        assert env.state_hash() == snaps[index]["state_hash"]
        assert env.snapshot() == snaps[index]
        assert env.records == ()
        _run(env, calls[index : index + 3])
    assert snaps == frozen


@pytest.mark.parametrize("seed", _SEEDS)
def test_random_repeated_restore_is_reproducible(seed):
    rng, calls = _random_calls(seed)
    split = rng.randint(0, len(calls))
    env_a = _new_env()
    _run(env_a, calls[:split])
    snap = env_a.snapshot()
    frozen = copy.deepcopy(snap)
    results = []
    for env in (IssueTrackerEnv(), _initialized_target(), env_a):
        for _ in range(2):
            env.restore(snap)
            observations = _run(env, calls[split:])
            results.append((observations, env.records, env.snapshot()))
    for result in results[1:]:
        assert result == results[0]
    assert snap == frozen


# ---------------------------------------------------------------------------
# restore 驗證與原子性
# ---------------------------------------------------------------------------

Mutator = Callable[[dict[str, Any]], Any]


def _set_top(key: Any, value: Any) -> Mutator:
    def mutate(snap: dict[str, Any]) -> Any:
        snap[key] = copy.deepcopy(value)
        return snap

    return mutate


def _drop_top(key: str) -> Mutator:
    def mutate(snap: dict[str, Any]) -> Any:
        del snap[key]
        return snap

    return mutate


def _edit_state(edit: Callable[[dict[str, Any]], Any]) -> Mutator:
    def mutate(snap: dict[str, Any]) -> Any:
        edit(snap["state"])
        return snap

    return mutate


def _issue(state: dict[str, Any], issue_id: int) -> dict[str, Any]:
    return next(issue for issue in state["issues"] if issue["id"] == issue_id)


def _actor_with_recomputed_hash(actor: Any) -> Mutator:
    def mutate(snap: dict[str, Any]) -> Any:
        snap["actor_id"] = actor
        snap["state_hash"] = _expected_hash(snap["state"], actor)
        return snap

    return mutate


def _flip_last_hash_char(snap: dict[str, Any]) -> Any:
    state_hash = snap["state_hash"]
    snap["state_hash"] = state_hash[:-1] + ("0" if state_hash[-1] != "0" else "1")
    return snap


def _strip_hash_prefix(snap: dict[str, Any]) -> Any:
    snap["state_hash"] = snap["state_hash"][len(HASH_PREFIX):]
    return snap


def _uppercase_hash_hex(snap: dict[str, Any]) -> Any:
    snap["state_hash"] = HASH_PREFIX + snap["state_hash"][len(HASH_PREFIX):].upper()
    return snap


def _hash_of_other_actor(snap: dict[str, Any]) -> Any:
    snap["state_hash"] = _expected_hash(snap["state"], "u1")
    return snap


def _append_duplicate_issue(state: dict[str, Any]) -> None:
    state["issues"].append(copy.deepcopy(state["issues"][-1]))


def _append_reserved_user(state: dict[str, Any]) -> None:
    state["users"].append({"id": "unassigned", "name": "Nobody", "team": "none", "active": True})


_NON_DICT_CASES = [
    pytest.param(lambda s: None, id="none"),
    pytest.param(lambda s: [], id="empty_list"),
    pytest.param(lambda s: list(s.items()), id="list_of_items"),
    pytest.param(lambda s: json.dumps(s), id="json_string"),
    pytest.param(lambda s: 0, id="int"),
]

_KEY_CASES = [pytest.param(_drop_top(key), id=f"missing_{key}") for key in sorted(SNAPSHOT_KEYS)] + [
    pytest.param(_set_top("records", []), id="extra_records_key"),
    pytest.param(_set_top("note", "x"), id="extra_note_key"),
    pytest.param(_set_top(1, "x"), id="extra_non_str_key"),
]

_VERSION_CASES = [
    pytest.param(_set_top("format", "agent-gym/env-snapshot/v0"), id="format_old"),
    pytest.param(_set_top("format", "agent-gym/task/v1"), id="format_other_artifact"),
    pytest.param(_set_top("format", None), id="format_none"),
    pytest.param(_set_top("schema_version", "issue_tracker/v2"), id="schema_version_v2"),
    pytest.param(_set_top("schema_version", ""), id="schema_version_empty"),
    pytest.param(_set_top("schema_version", None), id="schema_version_none"),
]

_STEP_CASES = [
    pytest.param(_set_top("step", True), id="step_true"),
    pytest.param(_set_top("step", False), id="step_false"),
    pytest.param(_set_top("step", -1), id="step_negative"),
    pytest.param(_set_top("step", float(len(_SOURCE_PREFIX))), id="step_float_equal_value"),
    pytest.param(_set_top("step", 1.5), id="step_float"),
    pytest.param(_set_top("step", str(len(_SOURCE_PREFIX))), id="step_str"),
    pytest.param(_set_top("step", None), id="step_none"),
    pytest.param(_set_top("step", [len(_SOURCE_PREFIX)]), id="step_list"),
]

# state 不是 dict：契約只要求 SnapshotError；實作可先做型別檢查而不經 from_dict，
# 因此不要求 __cause__ 是 SchemaError。
_NON_DICT_STATE_CASES = [
    pytest.param(_set_top("state", "state"), id="state_str"),
    pytest.param(_set_top("state", None), id="state_none"),
    pytest.param(_set_top("state", []), id="state_list"),
]

# state 是 dict 但 from_dict 必定 raise SchemaError（hash 以 parse 後的 state 計算，必須先 parse）。
_UNPARSABLE_STATE_CASES = [
    pytest.param(_edit_state(lambda st: st.pop("comments")), id="state_missing_comments"),
    pytest.param(_edit_state(lambda st: st.__setitem__("milestones", [])), id="state_extra_top_key"),
    pytest.param(_edit_state(lambda st: _issue(st, 2).update(status="done")), id="issue_bad_status"),
    pytest.param(_edit_state(lambda st: _issue(st, 2).update(priority="P1")), id="issue_bad_priority"),
    pytest.param(
        _edit_state(lambda st: _issue(st, 2).update(updated_at="2026-10-08")), id="issue_extra_field"
    ),
    pytest.param(_edit_state(lambda st: _issue(st, 2).pop("created_at")), id="issue_missing_field"),
    pytest.param(_edit_state(lambda st: _issue(st, 1).update(id=True)), id="issue_bool_id"),
    pytest.param(_edit_state(lambda st: _issue(st, 3).update(labels="bug")), id="issue_labels_not_list"),
    pytest.param(_edit_state(lambda st: _issue(st, 4).update(labels=["wontfix"])), id="issue_unknown_label"),
    pytest.param(
        _edit_state(lambda st: _issue(st, 2).update(assignee_id="ghost")), id="issue_unknown_assignee"
    ),
    pytest.param(
        _edit_state(lambda st: st["comments"][0].update(issue_id=_MISSING_ISSUE_ID)),
        id="comment_unknown_issue",
    ),
    pytest.param(
        _edit_state(lambda st: st["comments"][0].update(author_id="ghost")), id="comment_unknown_author"
    ),
    pytest.param(_edit_state(_append_duplicate_issue), id="duplicate_issue_id"),
    pytest.param(_edit_state(_append_reserved_user), id="reserved_user_id"),
    pytest.param(_edit_state(lambda st: st["users"][1].update(active=1)), id="user_active_not_bool"),
    pytest.param(_edit_state(lambda st: st["users"].__setitem__(2, "u2")), id="user_entity_not_dict"),
]

# state 可解析、parse 後與原 state 相同（hash 仍相符），但不是 canonical 形式。
_NON_CANONICAL_STATE_CASES = [
    pytest.param(
        _edit_state(lambda st: _issue(st, 3).update(labels=["security", "bug"])), id="issue_labels_unsorted"
    ),
    pytest.param(
        _edit_state(lambda st: _issue(st, 3).update(labels=["bug", "bug", "security"])),
        id="issue_labels_duplicated",
    ),
    pytest.param(_edit_state(lambda st: st["issues"].reverse()), id="issues_unsorted"),
    pytest.param(_edit_state(lambda st: st["users"].reverse()), id="users_unsorted"),
    pytest.param(_edit_state(lambda st: st["labels"].reverse()), id="labels_unsorted"),
    pytest.param(_edit_state(lambda st: st["comments"].reverse()), id="comments_unsorted"),
]

_ACTOR_CASES = [
    pytest.param(_set_top("actor_id", "ghost"), id="actor_unknown_stale_hash"),
    pytest.param(_actor_with_recomputed_hash("ghost"), id="actor_unknown_recomputed_hash"),
    pytest.param(_actor_with_recomputed_hash("unassigned"), id="actor_reserved_recomputed_hash"),
    pytest.param(_set_top("actor_id", None), id="actor_none"),
    pytest.param(_set_top("actor_id", 7), id="actor_int"),
    pytest.param(_set_top("actor_id", [SMALL_ACTOR]), id="actor_list"),
]

# state 可解析、canonical、actor 存在，只有 hash 對不上。
_HASH_MISMATCH_CASES = [
    pytest.param(_edit_state(lambda st: _issue(st, 2).update(title="Tampered")), id="tampered_title"),
    pytest.param(_edit_state(lambda st: _issue(st, 1).update(status="closed")), id="tampered_status"),
    pytest.param(
        _edit_state(lambda st: st["comments"][-1].update(body="tampered")), id="tampered_comment_body"
    ),
    pytest.param(_edit_state(lambda st: _issue(st, 4).update(labels=["ui"])), id="tampered_added_label"),
    pytest.param(_edit_state(lambda st: st["comments"].pop()), id="tampered_removed_comment"),
    pytest.param(_set_top("state", copy.deepcopy(SMALL_WORLD)), id="state_from_other_snapshot"),
    pytest.param(_set_top("actor_id", "u1"), id="actor_other_user_stale_hash"),
    pytest.param(_flip_last_hash_char, id="hash_last_char_flipped"),
    pytest.param(_set_top("state_hash", HASH_PREFIX + "0" * 64), id="hash_all_zeros"),
    pytest.param(_strip_hash_prefix, id="hash_without_prefix"),
    pytest.param(_uppercase_hash_hex, id="hash_uppercase_hex"),
    pytest.param(_hash_of_other_actor, id="hash_of_other_actor"),
    pytest.param(_set_top("state_hash", ""), id="hash_empty"),
    pytest.param(_set_top("state_hash", None), id="hash_none"),
    pytest.param(_set_top("state_hash", 0), id="hash_int"),
]

_ALL_INVALID_CASES = (
    _NON_DICT_CASES
    + _KEY_CASES
    + _VERSION_CASES
    + _STEP_CASES
    + _NON_DICT_STATE_CASES
    + _UNPARSABLE_STATE_CASES
    + _NON_CANONICAL_STATE_CASES
    + _ACTOR_CASES
    + _HASH_MISMATCH_CASES
)

_PROBE_CALLS: list[Call] = [
    ("add_comment", {"issue_id": 1, "body": "still alive"}),
    ("get_issue", {"issue_id": 1}),
]


def _bad_snapshot(mutate: Mutator) -> Any:
    return mutate(copy.deepcopy(_source_snapshot()))


def test_source_snapshot_used_by_invalid_cases_is_valid():
    snap = _source_snapshot()
    for env in (IssueTrackerEnv(), _initialized_target()):
        env.restore(copy.deepcopy(snap))
        assert env.snapshot() == snap


@pytest.mark.parametrize("mutate", _ALL_INVALID_CASES)
def test_invalid_snapshot_leaves_initialized_env_unchanged(mutate):
    bad = _bad_snapshot(mutate)
    env = _initialized_target()
    twin = _initialized_target()
    before = _env_view(env)
    with pytest.raises(SnapshotError):
        env.restore(bad)
    assert _env_view(env) == before
    assert _run(env, _PROBE_CALLS) == _run(twin, _PROBE_CALLS)
    assert env.records == twin.records
    assert env.snapshot() == twin.snapshot()


@pytest.mark.parametrize("mutate", _ALL_INVALID_CASES)
def test_invalid_snapshot_keeps_fresh_env_uninitialized(mutate):
    bad = _bad_snapshot(mutate)
    env = IssueTrackerEnv()
    with pytest.raises(SnapshotError):
        env.restore(bad)
    _assert_uninitialized(env)
    valid = _source_snapshot()
    env.restore(valid)
    assert env.snapshot() == valid


@pytest.mark.parametrize("mutate", _ALL_INVALID_CASES)
def test_failed_restore_does_not_mutate_input(mutate):
    bad = _bad_snapshot(mutate)
    bad_copy = copy.deepcopy(bad)
    with pytest.raises(SnapshotError):
        IssueTrackerEnv().restore(bad)
    assert bad == bad_copy


@pytest.mark.parametrize("mutate", _UNPARSABLE_STATE_CASES)
def test_unparsable_state_raises_snapshot_error_wrapping_schema_error(mutate):
    bad = _bad_snapshot(mutate)
    with pytest.raises(SchemaError):
        IssueTrackerState.from_dict(copy.deepcopy(bad["state"]))
    with pytest.raises(SnapshotError) as excinfo:
        IssueTrackerEnv().restore(bad)
    assert isinstance(excinfo.value.__cause__, SchemaError)


@pytest.mark.parametrize("mutate", _NON_CANONICAL_STATE_CASES)
def test_non_canonical_state_is_rejected_even_with_matching_hash(mutate):
    bad = _bad_snapshot(mutate)
    parsed = IssueTrackerState.from_dict(copy.deepcopy(bad["state"]))
    assert parsed.to_dict() != bad["state"]
    assert state_hash_of(parsed, bad["actor_id"]) == bad["state_hash"]
    with pytest.raises(SnapshotError):
        IssueTrackerEnv().restore(bad)


@pytest.mark.parametrize("actor", ["ghost", "unassigned"])
def test_actor_not_in_users_is_rejected_even_with_matching_hash(actor):
    bad = _bad_snapshot(_actor_with_recomputed_hash(actor))
    parsed = IssueTrackerState.from_dict(copy.deepcopy(bad["state"]))
    assert parsed.to_dict() == bad["state"]
    assert state_hash_of(parsed, actor) == bad["state_hash"]
    with pytest.raises(SnapshotError):
        IssueTrackerEnv().restore(bad)


@pytest.mark.parametrize("mutate", _HASH_MISMATCH_CASES)
def test_hash_mismatch_is_rejected(mutate):
    bad = _bad_snapshot(mutate)
    parsed = IssueTrackerState.from_dict(copy.deepcopy(bad["state"]))
    assert parsed.to_dict() == bad["state"]
    assert bad["actor_id"] in {user["id"] for user in bad["state"]["users"]}
    assert state_hash_of(parsed, bad["actor_id"]) != bad["state_hash"]
    with pytest.raises(SnapshotError):
        IssueTrackerEnv().restore(bad)


# ---------------------------------------------------------------------------
# restore：隨機損毀（只能成功或 raise SnapshotError）
# ---------------------------------------------------------------------------

_CORRUPTION_SEED = 2026
_CORRUPTION_ROUNDS = 300
_JUNK_VALUES: list[Any] = [None, True, 0, -1, 1.5, "", "x", [], {}, [None], {"k": 1}]

Path = tuple[Any, ...]


def _all_paths(value: Any, path: Path = ()) -> list[Path]:
    """含根節點的所有路徑；dict 依 key 排序，避免依賴實作的插入順序。"""
    paths = [path]
    if isinstance(value, dict):
        for key in sorted(value):
            paths += _all_paths(value[key], path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths += _all_paths(item, path + (index,))
    return paths


def _at(value: Any, path: Path) -> Any:
    for key in path:
        value = value[key]
    return value


def _random_corruption(snap: dict[str, Any], rng: random.Random) -> tuple[Any, str]:
    """對 snap 的深拷貝做一次 replace / delete / extra，回傳（損毀後的值, 描述）。"""
    corrupted = copy.deepcopy(snap)
    op = rng.choice(["replace", "delete", "extra"])
    paths = _all_paths(corrupted)
    if op == "delete":
        paths = [p for p in paths if p]
    elif op == "extra":
        paths = [p for p in paths if isinstance(_at(corrupted, p), (dict, list))]
    path = rng.choice(paths)
    junk = copy.deepcopy(rng.choice(_JUNK_VALUES))
    if op == "delete":
        del _at(corrupted, path[:-1])[path[-1]]
        return corrupted, f"delete at {list(path)!r}"
    description = f"{op} at {list(path)!r} with {junk!r}"
    if op == "replace":
        if not path:
            return junk, description
        _at(corrupted, path[:-1])[path[-1]] = junk
    else:
        target = _at(corrupted, path)
        if isinstance(target, dict):
            target["extra"] = junk
        else:
            target.insert(rng.randint(0, len(target)), junk)
    return corrupted, description


@pytest.mark.parametrize("target", ["initialized", "fresh"])
def test_restore_only_raises_snapshot_error_on_random_corruption(target):
    rng = random.Random(_CORRUPTION_SEED)
    source = _source_snapshot()
    rejected = 0
    for round_index in range(_CORRUPTION_ROUNDS):
        corrupted, description = _random_corruption(source, rng)
        label = f"round {round_index}: {description}"
        frozen = copy.deepcopy(corrupted)
        env = _initialized_target() if target == "initialized" else IssueTrackerEnv()
        before = _env_view(env) if target == "initialized" else None
        try:
            env.restore(corrupted)
        except SnapshotError:
            rejected += 1
            if before is None:
                _assert_uninitialized(env)
            else:
                assert _env_view(env) == before, f"env changed after rejecting {label}"
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{label}: raised {type(exc).__name__} instead of SnapshotError: {exc!r}")
        else:
            assert env.snapshot() == corrupted, f"accepted {label} but snapshot differs"
            assert env.records == (), label
        assert corrupted == frozen, f"restore mutated input for {label}"
    assert rejected > 0


def test_env_does_not_alias_snapshot_after_restore():
    snap = _source_snapshot()
    env = IssueTrackerEnv()
    env.restore(snap)
    before = env.snapshot()
    snap["state"]["issues"][0]["title"] = "tampered"
    snap["state"]["issues"][0]["labels"].append("security")
    snap["state"]["comments"].clear()
    snap["step"] = 999
    assert env.snapshot() == before


def test_state_with_tuple_labels_is_non_canonical():
    snap = _source_snapshot()
    issue = snap["state"]["issues"][0]
    issue["labels"] = tuple(issue["labels"])
    env = IssueTrackerEnv()
    with pytest.raises(SnapshotError):
        env.restore(snap)
