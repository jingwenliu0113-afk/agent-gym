"""replay/trajectory.py 與 grader.schema.Usage.from_dict 的契約測試（decision log 006 / 009）。

Trajectory 一律以真實 IssueTrackerEnv 產生的 ToolCallRecord 手工組裝（不經 runner），
讓 validate 規則 1–10 的測試與 runner 實作無關。task 使用 smoke tasks（只做回歸測試用途）。

trajectory/v2（decision 009）：Turn 多了 ``meta``（量測；None 或 JSON 物件；不進 conversation、在 trajectory_id 內），
Turn.to_dict 剛好 5 個 key；render_messages 的 assistant 訊息帶 ``provider_content``（細節測試在 test_provider_content.py）。

所有「相同 / 相符」都以 canonical_json 逐 byte 比較；凡是 Python ``==`` 會被騙的地方
（1 vs 1.0、True vs 1）都有對應案例。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest

from agents.policy import AgentConfig, ModelMessage, ToolCall
from apps.issue_tracker.schema import IssueTrackerState
from env.environment import IssueTrackerEnv, ToolCallRecord, state_hash_of
from env.hashing import canonical_json, sha256_of
from grader.grader import grade
from grader.schema import Usage
from replay.trajectory import (
    MAX_DOCUMENT_DEPTH,
    PARENT_KEYS,
    TERMINATIONS,
    TRAJECTORY_FORMAT,
    TRAJECTORY_KEYS,
    Trajectory,
    TrajectoryError,
    Turn,
    load_strict_json,
    render_messages,
)
from replay.trajectory import TURN_KEYS as MODULE_TURN_KEYS
from tasks.schema import Task
from tasks.smoke import REFERENCE_SOLUTIONS, SMOKE_TASK_IDS, get_smoke_task

R = dataclasses.replace

ASSIGN = "smoke-assign-and-start"
CLOSE = "smoke-close-duplicate"
ESCALATE = "smoke-escalate-security"
REASSIGN = "smoke-reassign-departed"
NOOP = "smoke-conditional-noop"

FAKE_HASH = "sha256:" + "0" * 64
TURN_KEYS = frozenset({"index", "message", "tool_steps", "usage", "meta"})
USAGE_FIELDS = ("input_tokens", "output_tokens", "latency_s", "cost_usd", "model_turns")
ROLE_KEYS: dict[str, list[str]] = {
    "system": ["role", "content"],
    "developer": ["role", "content"],
    "user": ["role", "content"],
    "assistant": ["role", "content", "tool_calls", "provider_content"],
    "tool": ["role", "tool_call_id", "content"],
}

SYSTEM_MARKER = "SYSTEM_PROMPT_MARKER"

# 一個 turn 多個 call、含失敗的 call（unknown tool、非 dict args）。
MIXED_LAYOUT: list[list[tuple[str, Any]]] = [
    [("assign_issue", {"issue_id": 8, "assignee_id": "u4"}), ("no_such_tool", {})],
    [("get_issue", [1, 2])],
    [("update_issue", {"issue_id": 8, "status": "in_progress"})],
]


# ---------------------------------------------------------------------------
# helpers（全部在測試內呼叫，不在 import / collection 時呼叫 stub）
# ---------------------------------------------------------------------------


def zero_usage() -> Usage:
    return Usage(0, 0, 0.0, 0.0, 1)


META_MARKER = "META_MARKER_量測"


def sample_meta(k: int) -> dict[str, Any]:
    """turn meta 範例（量測；內容是任意 JSON 物件，trajectory 不解讀）：含非 ASCII、null、巢狀、int 與 float（1 vs 1.0）。"""
    return {
        "response_model": "model-x",
        "request_id": f"req_{k:04d}",
        "stop_reason": "tool_use",
        "stop_details": None,
        "usage_detail": {"input": 100 + k, "cache_read": 0, "cache_write": {"5m": 0, "1h": 0}},
        "tool_use_blocks": 1,
        "ratio": 1.0,
        "count": 1,
        "note": META_MARKER,
        "list": [1, 1.0, True, None, "x"],
    }


def deep_value(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


def make_agent(**overrides: Any) -> AgentConfig:
    kwargs: dict[str, Any] = {
        "policy": "hand-built",
        "prompt_version": "test/v1",
        "system_prompt": f"You are a triage bot. 你好 {SYSTEM_MARKER}",
        "developer_prompt": "Use the tools.",
        "model_config": {"max_tokens": 256, "temperature": 0.0},
    }
    kwargs.update(overrides)
    return AgentConfig(**kwargs)


def make_parent(turn: Any = 1, **overrides: Any) -> dict[str, Any]:
    parent: dict[str, Any] = {
        "trajectory_id": "sha256:" + "a" * 64,
        "checkpoint_id": "sha256:" + "b" * 64,
        "turn": turn,
    }
    parent.update(overrides)
    return parent


def reference_layout(task_id: str) -> list[list[tuple[str, Any]]]:
    """每個 oracle call 一個 turn。"""
    return [[(name, copy.deepcopy(args))] for name, args in REFERENCE_SOLUTIONS[task_id]]


def run_layout(task: Task, layout: list[list[tuple[str, Any]]]) -> tuple[list[Turn], dict[str, Any]]:
    """依 layout 在新 env 上執行，回傳 tool turns 與最後的 snapshot。"""
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    turns: list[Turn] = []
    for k, calls in enumerate(layout):
        n = len(env.records)
        tool_calls = []
        for i, (name, args) in enumerate(calls):
            env.call_tool(name, copy.deepcopy(args))
            tool_calls.append(ToolCall(id=f"call_{k}_{i}", name=name, args=copy.deepcopy(args)))
        steps = env.records[n:]
        text = f"思考 {k}" if k % 2 == 0 else ""
        message = ModelMessage(text=text, tool_calls=tuple(tool_calls), stop_reason="tool_use")
        turns.append(Turn(index=k, message=message, tool_steps=tuple(steps), usage=zero_usage()))
    return turns, env.snapshot()


def final_turn(index: int, text: str = "Done.") -> Turn:
    return Turn(
        index=index,
        message=ModelMessage(text=text, tool_calls=(), stop_reason="end_turn"),
        tool_steps=(),
        usage=zero_usage(),
    )


def build(
    task_id: str = ASSIGN,
    layout: list[list[tuple[str, Any]]] | None = None,
    *,
    final: bool = True,
    max_turns: Any = None,
    termination: Any = None,
    parent: Any = None,
    agent: Any = None,
    metas: list[Any] | None = None,
) -> Trajectory:
    """以真實 env records 手工組裝 trajectory；預設為合法的 final_answer trajectory。
    metas（長度須等於 turn 數）給定時，第 i 個 turn 的 meta 設為 metas[i]（深拷貝）。
    """
    task = get_smoke_task(task_id)
    if layout is None:
        layout = reference_layout(task_id)
    turns, snapshot = run_layout(task, layout)
    if final:
        turns.append(final_turn(len(turns)))
    if metas is not None:
        assert len(metas) == len(turns)
        turns = [R(t, meta=copy.deepcopy(m)) for t, m in zip(turns, metas, strict=True)]
    if max_turns is None:
        max_turns = len(turns) + 2 if final else len(turns)
    if termination is None:
        termination = "final_answer" if final else "max_turns"
    return Trajectory(
        task=task.to_dict(),
        agent=agent if agent is not None else make_agent(),
        max_turns=max_turns,
        turns=tuple(turns),
        termination=termination,
        final_snapshot=snapshot,
        parent=parent,
    )


def set_turn(traj: Trajectory, i: int, turn: Any) -> Trajectory:
    turns = list(traj.turns)
    turns[i] = turn
    return R(traj, turns=tuple(turns))


def set_step(traj: Trajectory, i: int, j: int, **changes: Any) -> Trajectory:
    turn = traj.turns[i]
    steps = list(turn.tool_steps)
    steps[j] = R(steps[j], **changes)
    return set_turn(traj, i, R(turn, tool_steps=tuple(steps)))


def set_call(traj: Trajectory, i: int, j: int, **changes: Any) -> Trajectory:
    turn = traj.turns[i]
    calls = list(turn.message.tool_calls)
    calls[j] = R(calls[j], **changes)
    return set_turn(traj, i, R(turn, message=R(turn.message, tool_calls=tuple(calls))))


def with_usages(traj: Trajectory, usages: list[Usage]) -> Trajectory:
    assert len(usages) == len(traj.turns)
    return R(traj, turns=tuple(R(t, usage=u) for t, u in zip(traj.turns, usages, strict=True)))


def flat_records(traj: Trajectory) -> tuple[ToolCallRecord, ...]:
    return tuple(step for turn in traj.turns for step in turn.tool_steps)


def canon_records(records: Any) -> str:
    return canonical_json([r.to_dict() for r in records])


def assert_json_native(obj: Any) -> None:
    if obj is None or type(obj) in (str, int, float, bool):
        return
    if type(obj) is list:
        for item in obj:
            assert_json_native(item)
        return
    if type(obj) is dict:
        for key, value in obj.items():
            assert type(key) is str
            assert_json_native(value)
        return
    raise AssertionError(f"not JSON-native: {type(obj).__name__}")


def expected_messages(instruction: str, agent: AgentConfig, turns: Any) -> list[dict[str, Any]]:
    """依 render_messages docstring 逐字組出的期望 conversation（v2：assistant 帶 provider_content；meta 不出現）。"""
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "developer", "content": agent.developer_prompt},
        {"role": "user", "content": instruction},
    ]
    for turn in turns:
        calls = turn.message.tool_calls
        msgs.append(
            {
                "role": "assistant",
                "content": turn.message.text,
                "tool_calls": [{"id": c.id, "name": c.name, "args": copy.deepcopy(c.args)} for c in calls],
                "provider_content": copy.deepcopy(turn.message.provider_content),
            }
        )
        for call, step in zip(calls, turn.tool_steps, strict=True):
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": canonical_json(step.observation())})
    return msgs


def assert_messages_equal(actual: Any, expected: list[dict[str, Any]]) -> None:
    assert type(actual) is list
    assert_json_native(actual)
    assert canonical_json(actual) == canonical_json(expected)
    # 每個 dict 的 key 依規定順序建立
    assert [list(m.keys()) for m in actual] == [ROLE_KEYS[m["role"]] for m in expected]
    for msg in actual:
        if msg["role"] == "assistant":
            for call in msg["tool_calls"]:
                assert set(call) == {"id", "name", "args"}


def first_int_to_float(obj: Any) -> tuple[Any, bool]:
    """把遞迴遇到的第一個非 bool int 換成 float（Python == 仍相等，canonical_json 不同）。"""
    if type(obj) is int:
        return float(obj), True
    if type(obj) is dict:
        out = {}
        changed = False
        for key in sorted(obj):
            if changed:
                out[key] = copy.deepcopy(obj[key])
            else:
                out[key], changed = first_int_to_float(obj[key])
        return out, changed
    if type(obj) is list:
        out_list = []
        changed = False
        for item in obj:
            if changed:
                out_list.append(copy.deepcopy(item))
            else:
                new, changed = first_int_to_float(item)
                out_list.append(new)
        return out_list, changed
    return copy.deepcopy(obj), False


def rehashed_snapshot(snapshot: dict[str, Any], *, actor_id: str | None = None) -> dict[str, Any]:
    snap = copy.deepcopy(snapshot)
    if actor_id is not None:
        snap["actor_id"] = actor_id
    snap["state_hash"] = state_hash_of(IssueTrackerState.from_dict(snap["state"]), snap["actor_id"])
    return snap


def initial_snapshot(task_id: str) -> dict[str, Any]:
    task = get_smoke_task(task_id)
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    return env.snapshot()


VARIANTS: dict[str, Callable[[], Trajectory]] = {
    "assign_turn_per_call": lambda: build(ASSIGN),
    "close_duplicate_single_turn": lambda: build(CLOSE, [[c for turn in reference_layout(CLOSE) for c in turn]]),
    "reassign_hits_max_turns": lambda: build(REASSIGN, final=False),
    "noop_final_answer_only": lambda: build(NOOP, []),
    "mixed_failing_calls": lambda: build(ASSIGN, MIXED_LAYOUT),
    "escalate_with_parent": lambda: build(ESCALATE, parent=make_parent(1)),
    # v2：turn meta（量測）。None / 空物件 / 一般物件混用。
    "assign_with_meta": lambda: build(ASSIGN, metas=[sample_meta(0), None, {"stop_reason": "end_turn"}]),
    "mixed_with_meta": lambda: build(ASSIGN, MIXED_LAYOUT, metas=[sample_meta(i) for i in range(3)] + [{}]),
}


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------


def test_constants() -> None:
    assert TRAJECTORY_FORMAT == "agent-gym/trajectory/v2"  # decision 009
    assert TERMINATIONS == ("final_answer", "max_turns")
    assert TRAJECTORY_KEYS == (
        "format",
        "task",
        "agent",
        "max_turns",
        "turns",
        "termination",
        "final_snapshot",
        "parent",
    )
    assert PARENT_KEYS == ("trajectory_id", "checkpoint_id", "turn")
    assert MODULE_TURN_KEYS == ("index", "message", "tool_steps", "usage", "meta")  # v2（decision 009）
    assert frozenset(MODULE_TURN_KEYS) == TURN_KEYS
    assert issubclass(TrajectoryError, ValueError)


# ---------------------------------------------------------------------------
# render_messages
# ---------------------------------------------------------------------------


def test_render_messages_without_turns_is_system_developer_user() -> None:
    agent = make_agent()
    msgs = render_messages("Do the thing. 做事", agent, ())
    assert_messages_equal(msgs, expected_messages("Do the thing. 做事", agent, ()))
    assert [m["role"] for m in msgs] == ["system", "developer", "user"]
    assert msgs[2]["content"] == "Do the thing. 做事"


def test_render_messages_emits_empty_prompts() -> None:
    agent = make_agent(system_prompt="", developer_prompt="")
    msgs = render_messages("", agent, ())
    assert json.dumps(msgs, ensure_ascii=False) == json.dumps(
        [
            {"role": "system", "content": ""},
            {"role": "developer", "content": ""},
            {"role": "user", "content": ""},
        ],
        ensure_ascii=False,
    )


@pytest.mark.parametrize("name", list(VARIANTS))
def test_render_messages_matches_contract_exactly(name: str) -> None:
    traj = VARIANTS[name]()
    instruction = traj.task["instruction"]
    msgs = render_messages(instruction, traj.agent, traj.turns)
    assert_messages_equal(msgs, expected_messages(instruction, traj.agent, traj.turns))
    n_calls = sum(len(t.message.tool_calls) for t in traj.turns)
    assert len(msgs) == 3 + len(traj.turns) + n_calls


def test_render_messages_role_order_with_multiple_calls_per_turn() -> None:
    traj = build(ASSIGN, MIXED_LAYOUT)
    msgs = render_messages("x", traj.agent, traj.turns)
    assert [m["role"] for m in msgs] == [
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    # tool 訊息與 call 一一對應
    assert [m["tool_call_id"] for m in msgs if m["role"] == "tool"] == [
        "call_0_0",
        "call_0_1",
        "call_1_0",
        "call_2_0",
    ]
    assert msgs[-1]["tool_calls"] == []
    assert msgs[-1]["content"] == "Done."


def test_render_messages_tool_content_is_canonical_observation() -> None:
    traj = build(ASSIGN, MIXED_LAYOUT)
    msgs = render_messages("x", traj.agent, traj.turns)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    steps = flat_records(traj)
    assert len(tool_msgs) == len(steps)
    for msg, step in zip(tool_msgs, steps, strict=True):
        assert type(msg["content"]) is str
        assert msg["content"] == canonical_json(step.observation())
    # 失敗的 call 也照樣以 observation 呈現
    assert json.loads(tool_msgs[1]["content"])["ok"] is False
    assert json.loads(tool_msgs[1]["content"])["error"]["code"] == "unknown_tool"


def test_render_messages_non_dict_args_call() -> None:
    traj = build(ASSIGN, [[("get_issue", [1, 2])]])
    msgs = render_messages("x", traj.agent, traj.turns)
    assistant = msgs[3]
    assert canonical_json(assistant["tool_calls"]) == canonical_json([{"id": "call_0_0", "name": "get_issue", "args": [1, 2]}])
    tool = msgs[4]
    step = traj.turns[0].tool_steps[0]
    assert step.ok is False
    assert tool["content"] == canonical_json(step.observation())
    assert json.loads(tool["content"])["error"]["code"] == "invalid_arguments"


def test_render_messages_assistant_content_is_message_text() -> None:
    traj = build(ASSIGN)
    msgs = render_messages("x", traj.agent, traj.turns)
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert [m["content"] for m in assistants] == [t.message.text for t in traj.turns]
    assert assistants[0]["content"] == "思考 0"
    assert assistants[1]["content"] == ""


@pytest.mark.parametrize("layout", [None, MIXED_LAYOUT], ids=["reference", "mixed"])
def test_render_messages_ignores_turn_meta(layout: Any) -> None:
    """meta 是量測，不進 conversation：只差在 meta 的 turns 產生逐 byte 相同（含 key 順序）的 messages。"""
    plain = build(ASSIGN, layout)
    n = len(plain.turns)
    with_meta = build(ASSIGN, layout, metas=[sample_meta(i) for i in range(n)])
    a = render_messages("x", plain.agent, plain.turns)
    b = render_messages("x", with_meta.agent, with_meta.turns)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
    assert META_MARKER not in json.dumps(b, ensure_ascii=False)
    assert json.dumps(with_meta.messages(), ensure_ascii=False) == json.dumps(plain.messages(), ensure_ascii=False)


def test_render_messages_accepts_list_of_turns() -> None:
    traj = build(ASSIGN)
    a = render_messages("x", traj.agent, list(traj.turns))
    b = render_messages("x", traj.agent, traj.turns)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def test_render_messages_returns_new_objects_and_does_not_alias_turns() -> None:
    traj = build(ASSIGN)
    before = canonical_json([t.to_dict() for t in traj.turns])
    a = render_messages("x", traj.agent, traj.turns)
    b = render_messages("x", traj.agent, traj.turns)
    assert a is not b
    assert all(x is not y for x, y in zip(a, b, strict=True))
    assert a[3]["tool_calls"] is not b[3]["tool_calls"]
    # 修改回傳值不影響 turns，也不影響之後的 render
    a[3]["tool_calls"][0]["args"]["issue_id"] = 999
    a[3]["tool_calls"].append({"id": "x", "name": "y", "args": {}})
    a.append({"role": "user", "content": "injected"})
    assert canonical_json([t.to_dict() for t in traj.turns]) == before
    c = render_messages("x", traj.agent, traj.turns)
    assert json.dumps(c, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Turn.to_dict / Turn.from_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_meta", [False, True])
def test_turn_to_dict_shape(with_meta: bool) -> None:
    metas = [sample_meta(0), None, {}, {"stop_reason": "end_turn"}] if with_meta else None
    traj = build(ASSIGN, MIXED_LAYOUT, metas=metas)
    for turn in traj.turns:
        d = turn.to_dict()
        assert set(d) == TURN_KEYS
        assert len(d) == 5
        assert_json_native(d)
        assert type(d["index"]) is int and d["index"] == turn.index
        assert canonical_json(d["message"]) == canonical_json(turn.message.to_dict())
        assert type(d["tool_steps"]) is list
        assert canonical_json(d["tool_steps"]) == canon_records(turn.tool_steps)
        assert canonical_json(d["usage"]) == canonical_json(turn.usage.to_dict())
        assert canonical_json(d["meta"]) == canonical_json(turn.meta)
        if turn.meta is None:
            assert d["meta"] is None


def test_turn_meta_defaults_to_none() -> None:
    turn = Turn(
        index=0,
        message=ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn"),
        tool_steps=(),
        usage=zero_usage(),
    )
    assert turn.meta is None
    d = turn.to_dict()
    assert set(d) == TURN_KEYS
    assert d["meta"] is None
    # 手工組裝（未指定 meta）的 turns 一律 meta = None
    assert all(t.meta is None for t in build(ASSIGN, MIXED_LAYOUT).turns)


def test_turn_to_dict_meta_is_deep_copy() -> None:
    turn = R(build(ASSIGN).turns[0], meta=sample_meta(0))
    before = canonical_json(turn.to_dict())
    d = turn.to_dict()
    assert d["meta"] is not turn.meta
    d["meta"]["usage_detail"]["cache_write"]["1h"] = 999
    d["meta"]["list"].append("tampered")
    d["meta"]["request_id"] = "tampered"
    assert canonical_json(turn.to_dict()) == before
    assert canonical_json(turn.meta) == canonical_json(sample_meta(0))


@pytest.mark.parametrize("via_json", [False, True])
@pytest.mark.parametrize(
    "meta",
    [None, {}, sample_meta(3), {"a": {"b": {"c": [1, {"d": None}]}}}, {"x": deep_value(40)}],
    ids=["none", "empty", "sample", "nested", "deep_ok"],
)
def test_turn_round_trip_with_meta(meta: Any, via_json: bool) -> None:
    turn = R(build(ASSIGN).turns[0], meta=copy.deepcopy(meta))
    d = turn.to_dict()
    if via_json:
        d = json.loads(canonical_json(d))
    back = Turn.from_dict(d)
    assert back == turn
    assert canonical_json(back.to_dict()) == canonical_json(turn.to_dict())
    assert canonical_json(back.meta) == canonical_json(meta)
    if meta is None:
        assert back.meta is None


def test_turn_round_trip_preserves_int_vs_float_in_meta() -> None:
    as_int = R(build(ASSIGN).turns[0], meta={"ratio": 1, "nested": [0]})
    as_float = R(as_int, meta={"ratio": 1.0, "nested": [0.0]})
    assert as_int == as_float  # Python == 會被騙
    back = Turn.from_dict(json.loads(canonical_json(as_int.to_dict())))
    assert canonical_json(back.to_dict()) == canonical_json(as_int.to_dict())
    assert canonical_json(back.to_dict()) != canonical_json(as_float.to_dict())


def test_turn_to_dict_is_deep_copy() -> None:
    turn = build(ASSIGN).turns[0]
    before = canonical_json(turn.to_dict())
    d = turn.to_dict()
    d["tool_steps"][0]["result"]["issue"]["title"] = "tampered"
    d["message"]["tool_calls"][0]["args"]["issue_id"] = 999
    assert canonical_json(turn.to_dict()) == before


@pytest.mark.parametrize("via_json", [False, True])
def test_turn_round_trip(via_json: bool) -> None:
    traj = build(ASSIGN, MIXED_LAYOUT)
    for turn in traj.turns:
        d = turn.to_dict()
        if via_json:
            d = json.loads(canonical_json(d))
        back = Turn.from_dict(d)
        assert back == turn
        assert canonical_json(back.to_dict()) == canonical_json(turn.to_dict())


def test_turn_round_trip_preserves_int_latency() -> None:
    turn = R(build(ASSIGN).turns[0], usage=Usage(0, 0, 0, 0.0, 1))
    float_turn = R(turn, usage=Usage(0, 0, 0.0, 0.0, 1))
    assert turn == float_turn  # Python == 會被騙
    back = Turn.from_dict(json.loads(canonical_json(turn.to_dict())))
    assert canonical_json(back.to_dict()) == canonical_json(turn.to_dict())
    assert canonical_json(back.to_dict()) != canonical_json(float_turn.to_dict())


def _drop(key: str) -> Callable[[dict[str, Any]], Any]:
    def mutate(d: dict[str, Any]) -> Any:
        del d[key]
        return d

    return mutate


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], Any]:
    def mutate(d: dict[str, Any]) -> Any:
        target = d
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return d

    return mutate


def _del_nested(path: tuple[Any, ...]) -> Callable[[dict[str, Any]], Any]:
    def mutate(d: dict[str, Any]) -> Any:
        target = d
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
        return d

    return mutate


TURN_TOP_LEVEL_ERRORS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "not_dict_list": lambda d: [d],
    "not_dict_none": lambda d: None,
    "not_dict_str": lambda d: "turn",
    **{f"missing_{k}": _drop(k) for k in sorted(TURN_KEYS)},
    "extra_key": _set(("extra",), 1),
    "index_true": _set(("index",), True),
    "index_false": _set(("index",), False),
    "index_float": _set(("index",), 0.0),
    "index_str": _set(("index",), "0"),
    "index_none": _set(("index",), None),
    "tool_steps_tuple": lambda d: {**d, "tool_steps": tuple(d["tool_steps"])},
    "tool_steps_dict": _set(("tool_steps",), {}),
    "tool_steps_none": _set(("tool_steps",), None),
    # meta 必須是 None 或 dict（JSON 物件），且整份輸入是 JSON 文件（規則 10）
    "meta_list": _set(("meta",), []),
    "meta_list_of_dict": _set(("meta",), [{"a": 1}]),
    "meta_str": _set(("meta",), "meta"),
    "meta_int": _set(("meta",), 0),
    "meta_true": _set(("meta",), True),
    "meta_false": _set(("meta",), False),
    "meta_tuple": _set(("meta",), ({"a": 1},)),
    "meta_nested_tuple": _set(("meta",), {"a": (1, 2)}),
    "meta_non_str_key": _set(("meta",), {1: "a"}),
    "meta_set_value": _set(("meta",), {"a": {1, 2}}),
    "meta_object_value": _set(("meta",), {"a": object()}),
    "meta_too_deep": _set(("meta",), {"x": deep_value(MAX_DOCUMENT_DEPTH + 50)}),
    "meta_far_too_deep": _set(("meta",), {"x": deep_value(5000)}),
}

TURN_NESTED_ERRORS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "message_not_dict": _set(("message",), None),
    "message_missing_text": _del_nested(("message", "text")),
    "message_extra_key": _set(("message", "extra"), 1),
    "message_tool_calls_not_list": _set(("message", "tool_calls"), "calls"),
    "message_call_missing_id": _del_nested(("message", "tool_calls", 0, "id")),
    "step_not_dict": _set(("tool_steps", 0), "step"),
    "step_missing_ok": _del_nested(("tool_steps", 0, "ok")),
    "step_bool_step": _set(("tool_steps", 0, "step"), False),
    "usage_not_dict": _set(("usage",), None),
    "usage_missing_key": _del_nested(("usage", "latency_s")),
    "usage_extra_key": _set(("usage", "extra"), 0),
    "usage_str_tokens": _set(("usage", "input_tokens"), "0"),
    "usage_bool_tokens": _set(("usage", "input_tokens"), True),
    "usage_nan_latency": _set(("usage", "latency_s"), float("nan")),
}


@pytest.mark.parametrize("name", list(TURN_TOP_LEVEL_ERRORS))
def test_turn_from_dict_rejects_malformed(name: str) -> None:
    d = build(ASSIGN).turns[0].to_dict()
    data = TURN_TOP_LEVEL_ERRORS[name](d)
    with pytest.raises(TrajectoryError):
        Turn.from_dict(data)


@pytest.mark.parametrize("name", list(TURN_NESTED_ERRORS))
def test_turn_from_dict_wraps_nested_errors(name: str) -> None:
    d = build(ASSIGN).turns[0].to_dict()
    data = TURN_NESTED_ERRORS[name](d)
    with pytest.raises(TrajectoryError) as exc_info:
        Turn.from_dict(data)
    assert exc_info.value.__cause__ is not None  # 以 ``from e`` 包裝內層例外


# ---------------------------------------------------------------------------
# validate：合法案例
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_validate_accepts_valid_variants(name: str) -> None:
    VARIANTS[name]().validate()


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
@pytest.mark.parametrize("one_turn", [False, True])
def test_validate_accepts_reference_trajectories(task_id: str, one_turn: bool) -> None:
    layout = [[c for turn in reference_layout(task_id) for c in turn]] if one_turn else None
    build(task_id, layout).validate()


def test_validate_accepts_max_turns_equal_to_len_with_final_answer() -> None:
    traj = build(ASSIGN)
    R(traj, max_turns=len(traj.turns)).validate()


@pytest.mark.parametrize("turn", [0, 1, 2])
def test_validate_accepts_parent_turn_in_range(turn: int) -> None:
    traj = build(ASSIGN, parent=make_parent(turn))
    assert len(traj.turns) == 3
    traj.validate()


def test_validate_compares_args_canonically_not_by_key_order() -> None:
    # call 的 args key 順序與 record 不同，canonical_json 相同 → 合法
    traj = build(ASSIGN)
    traj = set_call(traj, 0, 0, args={"assignee_id": "u4", "issue_id": 8})
    traj.validate()


def test_validate_does_not_replay() -> None:
    # 竄改 record 的 result 不違反結構規則 1–9（重放屬於 verify）
    traj = build(ASSIGN)
    result = copy.deepcopy(traj.turns[0].tool_steps[0].result)
    result["issue"]["title"] = "tampered"
    set_step(traj, 0, 0, result=result).validate()


# ---------------------------------------------------------------------------
# validate 規則 1：task / agent / max_turns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stored, normalized",
    [(1.0, 1), (True, 1), (1, True), (0, 0.0)],
    ids=["float_vs_int", "bool_vs_int", "int_vs_bool", "int_vs_float"],
)
def test_validate_rule1_task_round_trip_is_compared_canonically(
    monkeypatch: pytest.MonkeyPatch, stored: Any, normalized: Any
) -> None:
    base = build(ASSIGN)
    task = copy.deepcopy(base.task)
    task["params"] = {"k": stored}
    traj = R(base, task=task)
    traj.validate()  # 控制組：真正的 Task round trip 逐 byte 相同

    # 模擬一個會正規化 params 的 Task.to_dict：Python == 相等，但 canonical_json 不同
    real_to_dict = Task.to_dict

    def normalizing_to_dict(self: Task) -> dict[str, Any]:
        d = real_to_dict(self)
        d["params"] = {"k": normalized}
        return d

    assert {"k": stored} == {"k": normalized}
    monkeypatch.setattr(Task, "to_dict", normalizing_to_dict)
    with pytest.raises(TrajectoryError):
        traj.validate()


def _task_missing_key(task: dict[str, Any]) -> Any:
    del task["instruction"]
    return task


def _task_extra_key(task: dict[str, Any]) -> Any:
    task["extra"] = 1
    return task


def _task_wrong_format(task: dict[str, Any]) -> Any:
    task["format"] = "agent-gym/task/v0"
    return task


def _task_empty_outcome(task: dict[str, Any]) -> Any:
    task["expected_outcome"] = []
    return task


def _task_bool_issue_id(task: dict[str, Any]) -> Any:
    task["expected_outcome"][0]["issue_id"] = True
    return task


def _task_as_object(task: dict[str, Any]) -> Any:
    return Task.from_dict(task)


def _task_none(task: dict[str, Any]) -> Any:
    return None


@pytest.mark.parametrize(
    "mutate",
    [
        _task_missing_key,
        _task_extra_key,
        _task_wrong_format,
        _task_empty_outcome,
        _task_bool_issue_id,
        _task_as_object,
        _task_none,
    ],
)
def test_validate_rule1_rejects_invalid_task(mutate: Callable[[dict[str, Any]], Any]) -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        R(base, task=mutate(copy.deepcopy(base.task))).validate()


@pytest.mark.parametrize(
    "agent",
    [
        None,
        "hand-built",
        {
            "policy": "hand-built",
            "prompt_version": "test/v1",
            "system_prompt": "",
            "developer_prompt": "",
            "model_config": {},
        },
    ],
    ids=["none", "str", "dict"],
)
def test_validate_rule1_rejects_non_agent_config(agent: Any) -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        R(base, agent=agent).validate()


@pytest.mark.parametrize("max_turns", [True, 1.0, 0, -1, "1", None], ids=repr)
def test_validate_rule1_rejects_bad_max_turns(max_turns: Any) -> None:
    # 只有 1 個 turn：max_turns=True / 1.0 在 Python == 下會被當成 1
    base = build(NOOP, [])
    assert len(base.turns) == 1
    R(base, max_turns=1).validate()  # 控制組
    with pytest.raises(TrajectoryError):
        R(base, max_turns=max_turns).validate()


# ---------------------------------------------------------------------------
# validate 規則 2：turns 數量與 index
# ---------------------------------------------------------------------------


def test_validate_rule2_rejects_empty_turns() -> None:
    base = build(NOOP, [])
    with pytest.raises(TrajectoryError):
        R(base, turns=(), final_snapshot=initial_snapshot(NOOP)).validate()


def test_validate_rule2_rejects_more_turns_than_max_turns() -> None:
    base = build(ASSIGN)
    assert len(base.turns) == 3
    with pytest.raises(TrajectoryError):
        R(base, max_turns=2).validate()


def test_validate_rule2_rejects_non_turn_element() -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        set_turn(base, 2, base.turns[2].message).validate()


@pytest.mark.parametrize(
    "i, index",
    [(0, 1), (1, 2), (2, 1), (0, -1), (0, False), (1, True), (2, 2.0), (0, "0")],
    ids=["t0=1", "t1=2", "t2=1", "t0=-1", "t0=False", "t1=True", "t2=2.0", "t0='0'"],
)
def test_validate_rule2_rejects_index_mismatch(i: int, index: Any) -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        set_turn(base, i, R(base.turns[i], index=index)).validate()


# ---------------------------------------------------------------------------
# validate 規則 3：tool steps 與 tool calls 對應、usage.model_turns == 1
# ---------------------------------------------------------------------------


def test_validate_rule3_rejects_missing_step() -> None:
    base = build(ASSIGN, MIXED_LAYOUT)
    turn = base.turns[0]
    assert len(turn.tool_steps) == 2
    with pytest.raises(TrajectoryError):
        set_turn(base, 0, R(turn, tool_steps=turn.tool_steps[:1])).validate()


def test_validate_rule3_rejects_extra_step() -> None:
    base = build(ASSIGN, MIXED_LAYOUT)
    turn = base.turns[0]
    with pytest.raises(TrajectoryError):
        set_turn(base, 0, R(turn, message=R(turn.message, tool_calls=turn.message.tool_calls[:1]))).validate()


def test_validate_rule3_rejects_tool_name_mismatch() -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        set_call(base, 0, 0, name="update_issue").validate()


def test_validate_rule3_rejects_swapped_call_order_within_turn() -> None:
    base = build(ASSIGN, MIXED_LAYOUT)
    turn = base.turns[0]
    swapped = (turn.message.tool_calls[1], turn.message.tool_calls[0])
    with pytest.raises(TrajectoryError):
        set_turn(base, 0, R(turn, message=R(turn.message, tool_calls=swapped))).validate()


@pytest.mark.parametrize("side", ["call", "step"])
def test_validate_rule3_args_int_vs_float_is_mismatch(side: str) -> None:
    base = build(ASSIGN)
    bad_args = {"issue_id": 8.0, "assignee_id": "u4"}
    assert bad_args == base.turns[0].tool_steps[0].args  # Python == 會被騙
    traj = set_call(base, 0, 0, args=bad_args) if side == "call" else set_step(base, 0, 0, args=bad_args)
    with pytest.raises(TrajectoryError):
        traj.validate()


@pytest.mark.parametrize("side", ["call", "step"])
def test_validate_rule3_args_bool_vs_int_is_mismatch(side: str) -> None:
    base = build(ASSIGN, [[("get_issue", {"issue_id": 1})]])
    base.validate()  # 控制組
    bad_args = {"issue_id": True}
    assert bad_args == base.turns[0].tool_steps[0].args  # Python == 會被騙
    traj = set_call(base, 0, 0, args=bad_args) if side == "call" else set_step(base, 0, 0, args=bad_args)
    with pytest.raises(TrajectoryError):
        traj.validate()


def test_validate_rule3_args_value_mismatch() -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        set_call(base, 0, 0, args={"issue_id": 8, "assignee_id": "u3"}).validate()


@pytest.mark.parametrize("turn_index", [0, 2])
@pytest.mark.parametrize("model_turns", [0, 2, None])
def test_validate_rule3_rejects_model_turns_not_one(turn_index: int, model_turns: Any) -> None:
    base = build(ASSIGN)
    turn = base.turns[turn_index]
    with pytest.raises(TrajectoryError):
        set_turn(base, turn_index, R(turn, usage=Usage(0, 0, 0.0, 0.0, model_turns))).validate()


# ---------------------------------------------------------------------------
# validate / verify：Turn 欄位型別錯誤
# Turn 沒有建構時驗證；欄位型別錯誤時，規則 3 的述詞（message.tool_calls、step.tool / step.args、
# usage.model_turns == 1）與規則 9（canonical_json(to_dict()) 成功）都無法成立 → 一律 TrajectoryError，
# 不得漏出 AttributeError / TypeError。
# ---------------------------------------------------------------------------


def _replace_turn_field(i: int, field: str, make: Callable[[Turn], Any]) -> Callable[[Trajectory], Trajectory]:
    def tamper(traj: Trajectory) -> Trajectory:
        turn = traj.turns[i]
        return set_turn(traj, i, R(turn, **{field: make(turn)}))

    return tamper


# 以 MIXED_LAYOUT 為底：turn 0 有兩個 call、turn 1 一個 call、turn 3 是最後的 final turn。
TURN_FIELD_TAMPERS: dict[str, Callable[[Trajectory], Trajectory]] = {
    "t0_usage_none": _replace_turn_field(0, "usage", lambda t: None),
    "t3_usage_none": _replace_turn_field(3, "usage", lambda t: None),
    # dict 內含 model_turns == 1，但不是 Usage
    "t0_usage_dict": _replace_turn_field(0, "usage", lambda t: t.usage.to_dict()),
    "t3_usage_dict": _replace_turn_field(3, "usage", lambda t: t.usage.to_dict()),
    "t0_message_none": _replace_turn_field(0, "message", lambda t: None),
    "t3_message_none": _replace_turn_field(3, "message", lambda t: None),
    "t0_message_dict": _replace_turn_field(0, "message", lambda t: t.message.to_dict()),
    "t3_message_dict": _replace_turn_field(3, "message", lambda t: t.message.to_dict()),
    "t1_step_dict": _replace_turn_field(1, "tool_steps", lambda t: tuple(s.to_dict() for s in t.tool_steps)),
    "t1_step_none": _replace_turn_field(1, "tool_steps", lambda t: (None,) * len(t.tool_steps)),
    # 只有第二個 step 型別錯誤（長度仍與 tool_calls 相同）
    "t0_second_step_dict": _replace_turn_field(0, "tool_steps", lambda t: (t.tool_steps[0], t.tool_steps[1].to_dict())),
    "t0_second_step_none": _replace_turn_field(0, "tool_steps", lambda t: (t.tool_steps[0], None)),
}


@pytest.mark.parametrize("method", ["validate", "verify"])
@pytest.mark.parametrize("name", list(TURN_FIELD_TAMPERS))
def test_rejects_turn_with_wrongly_typed_field(name: str, method: str) -> None:
    base = build(ASSIGN, MIXED_LAYOUT)
    assert len(base.turns) == 4
    assert len(base.turns[0].tool_steps) == 2 and len(base.turns[1].tool_steps) == 1
    getattr(base, method)()  # 控制組
    traj = TURN_FIELD_TAMPERS[name](base)
    with pytest.raises(TrajectoryError):
        getattr(traj, method)()


# ---------------------------------------------------------------------------
# validate 規則 4：step 編號跨 turn 從 0 連續
# ---------------------------------------------------------------------------


def _renumber(traj: Trajectory, numbers: list[Any]) -> Trajectory:
    out = traj
    k = 0
    for i, turn in enumerate(traj.turns):
        for j in range(len(turn.tool_steps)):
            out = set_step(out, i, j, step=numbers[k])
            k += 1
    assert k == len(numbers)
    return out


@pytest.mark.parametrize(
    "numbers",
    [[1, 2], [0, 0], [0, 2], [1, 0], [0, True], [False, 1], [0, 1.0]],
    ids=["start_at_1", "restart_each_turn", "gap", "reversed", "bool_true", "bool_false", "float"],
)
def test_validate_rule4_rejects_non_contiguous_steps(numbers: list[Any]) -> None:
    base = build(ASSIGN)
    assert [s.step for s in flat_records(base)] == [0, 1]
    with pytest.raises(TrajectoryError):
        _renumber(base, numbers).validate()


def test_validate_rule4_steps_continue_across_turns_with_multiple_calls() -> None:
    base = build(ASSIGN, MIXED_LAYOUT)
    assert [s.step for s in flat_records(base)] == [0, 1, 2, 3]
    base.validate()
    with pytest.raises(TrajectoryError):
        _renumber(base, [0, 1, 0, 1]).validate()


# ---------------------------------------------------------------------------
# validate 規則 5：只有最後一個 turn 可以沒有 tool call
# ---------------------------------------------------------------------------


def test_validate_rule5_rejects_middle_turn_without_tool_calls() -> None:
    base = build(ASSIGN)
    shifted = tuple(R(t, index=t.index + 1) for t in base.turns)
    empty = Turn(0, ModelMessage(text="hmm", tool_calls=(), stop_reason="end_turn"), (), zero_usage())
    with pytest.raises(TrajectoryError):
        R(base, turns=(empty, *shifted), max_turns=10).validate()


def test_validate_rule5_rejects_two_trailing_final_turns() -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        R(base, turns=(*base.turns, final_turn(len(base.turns))), max_turns=10).validate()


# ---------------------------------------------------------------------------
# validate 規則 6：termination
# ---------------------------------------------------------------------------


def test_validate_rule6_final_answer_requires_last_turn_without_calls() -> None:
    base = build(ASSIGN, final=False)  # 最後一個 turn 有 tool call
    assert base.turns[-1].message.tool_calls
    base.validate()  # 控制組：termination = max_turns 且 len == max_turns
    with pytest.raises(TrajectoryError):
        R(base, termination="final_answer").validate()


def test_validate_rule6_last_turn_without_calls_must_be_final_answer() -> None:
    base = build(ASSIGN)
    traj = R(base, max_turns=len(base.turns))
    traj.validate()  # 控制組
    with pytest.raises(TrajectoryError):
        R(traj, termination="max_turns").validate()


def test_validate_rule6_max_turns_termination_requires_len_equal_max_turns() -> None:
    base = build(ASSIGN, final=False)
    with pytest.raises(TrajectoryError):
        R(base, max_turns=len(base.turns) + 1).validate()


@pytest.mark.parametrize("termination", ["done", "", None, "FINAL_ANSWER", 0], ids=repr)
def test_validate_rule6_rejects_unknown_termination(termination: Any) -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        R(base, termination=termination).validate()


# ---------------------------------------------------------------------------
# validate 規則 7：final_snapshot
# ---------------------------------------------------------------------------


def _snap_bad_hash(s: dict[str, Any]) -> Any:
    s["state_hash"] = FAKE_HASH
    return s


def _snap_missing_key(s: dict[str, Any]) -> Any:
    del s["state_hash"]
    return s


def _snap_extra_key(s: dict[str, Any]) -> Any:
    s["records"] = []
    return s


def _snap_wrong_format(s: dict[str, Any]) -> Any:
    s["format"] = "agent-gym/env-snapshot/v0"
    return s


def _snap_wrong_schema(s: dict[str, Any]) -> Any:
    s["schema_version"] = "issue_tracker/v0"
    return s


def _snap_state_changed_without_rehash(s: dict[str, Any]) -> Any:
    s["state"]["issues"][0]["title"] = "tampered"
    return s


def _snap_none(s: dict[str, Any]) -> Any:
    return None


def _snap_unknown_actor(s: dict[str, Any]) -> Any:
    s["actor_id"] = "nobody"
    return s


@pytest.mark.parametrize(
    "mutate",
    [
        _snap_bad_hash,
        _snap_missing_key,
        _snap_extra_key,
        _snap_wrong_format,
        _snap_wrong_schema,
        _snap_state_changed_without_rehash,
        _snap_none,
        _snap_unknown_actor,
    ],
)
def test_validate_rule7_rejects_unrestorable_snapshot(mutate: Callable[[dict[str, Any]], Any]) -> None:
    base = build(NOOP)
    with pytest.raises(TrajectoryError):
        R(base, final_snapshot=mutate(copy.deepcopy(base.final_snapshot))).validate()


@pytest.mark.parametrize("step", [0, 2, True, 1.0, -1], ids=repr)
def test_validate_rule7_rejects_wrong_snapshot_step(step: Any) -> None:
    base = build(NOOP)  # 1 個 tool step
    assert base.final_snapshot["step"] == 1
    snap = copy.deepcopy(base.final_snapshot)
    snap["step"] = step
    with pytest.raises(TrajectoryError):
        R(base, final_snapshot=snap).validate()


def test_validate_rule7_rejects_actor_different_from_task() -> None:
    base = build(ASSIGN)
    snap = rehashed_snapshot(base.final_snapshot, actor_id="u4")
    IssueTrackerEnv().restore(copy.deepcopy(snap))  # 前提：可被 restore
    with pytest.raises(TrajectoryError):
        R(base, final_snapshot=snap).validate()


def test_validate_rule7_accepts_restorable_snapshot_with_matching_step() -> None:
    # 規則 7 不比對內容（那是 verify 的工作）：回到初始 state 但 step 正確 → 結構上合法
    base = build(ASSIGN)
    snap = initial_snapshot(ASSIGN)
    snap["step"] = len(flat_records(base))
    R(base, final_snapshot=snap).validate()


# ---------------------------------------------------------------------------
# validate 規則 8：parent
# ---------------------------------------------------------------------------


def _parent_missing_turn() -> Any:
    p = make_parent(1)
    del p["turn"]
    return p


def _parent_extra_key() -> Any:
    return {**make_parent(1), "source": "x"}


def _parent_renamed_key() -> Any:
    p = make_parent(1)
    p["trajectory"] = p.pop("trajectory_id")
    return p


PARENT_ERRORS: dict[str, Callable[[], Any]] = {
    "list": lambda: list(make_parent(1).values()),
    "str": lambda: "sha256:abc",
    "empty_dict": lambda: {},
    "missing_turn": _parent_missing_turn,
    "extra_key": _parent_extra_key,
    "renamed_key": _parent_renamed_key,
    "turn_eq_len": lambda: make_parent(3),
    "turn_gt_len": lambda: make_parent(4),
    "turn_negative": lambda: make_parent(-1),
    "turn_true": lambda: make_parent(True),
    "turn_false": lambda: make_parent(False),
    "turn_float": lambda: make_parent(1.0),
    "turn_str": lambda: make_parent("1"),
    "turn_none": lambda: make_parent(None),
    "trajectory_id_int": lambda: make_parent(1, trajectory_id=1),
    "trajectory_id_none": lambda: make_parent(1, trajectory_id=None),
    "checkpoint_id_none": lambda: make_parent(1, checkpoint_id=None),
    "checkpoint_id_list": lambda: make_parent(1, checkpoint_id=["sha256:x"]),
}


@pytest.mark.parametrize("name", list(PARENT_ERRORS))
def test_validate_rule8_rejects_bad_parent(name: str) -> None:
    base = build(ASSIGN)
    assert len(base.turns) == 3
    with pytest.raises(TrajectoryError):
        R(base, parent=PARENT_ERRORS[name]()).validate()


# ---------------------------------------------------------------------------
# validate 規則 9：所有字串可 UTF-8 編碼
# ---------------------------------------------------------------------------


def test_validate_rule9_rejects_lone_surrogate_in_parent() -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        R(base, parent=make_parent(1, trajectory_id="sha256:\ud800")).validate()


def test_validate_rule9_rejects_lone_surrogate_in_task_instruction() -> None:
    base = build(ASSIGN)
    task = copy.deepcopy(base.task)
    task["instruction"] = "Fix it \ud800"
    with pytest.raises(TrajectoryError):
        R(base, task=task).validate()


def test_validate_rule9_rejects_lone_surrogate_in_task_params() -> None:
    base = build(ASSIGN)
    task = copy.deepcopy(base.task)
    task["params"] = {"note": "\udfff"}
    with pytest.raises(TrajectoryError):
        R(base, task=task).validate()


def test_validate_rule9_rejects_lone_surrogate_in_record_result() -> None:
    base = build(ASSIGN)
    result = copy.deepcopy(base.turns[0].tool_steps[0].result)
    result["issue"]["title"] = "\ud83d"
    with pytest.raises(TrajectoryError):
        set_step(base, 0, 0, result=result).validate()


# ---------------------------------------------------------------------------
# validate 規則 9 / 10：turn meta（量測，decision 009）
# 規則 10：每個 turn 的 meta 是 JSON 文件（容器只能是 list / dict、key 為 str、葉節點為 JSON 純量、深度有上限）；
# 規則 9：canonical_json(to_dict()) 可 UTF-8 編碼（因此 NaN / inf、lone surrogate 也被擋下）。
# 深度案例只用兩種計層慣例（以 meta 本身或以文件根為第 0 層）都一致的值。
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["validate", "verify"])
def test_validate_accepts_turn_meta(method: str) -> None:
    n = len(build(ASSIGN, MIXED_LAYOUT).turns)
    metas: list[Any] = [sample_meta(0), None, {}, {"x": deep_value(40), "名": "值"}]
    assert len(metas) == n
    getattr(build(ASSIGN, MIXED_LAYOUT, metas=metas), method)()


TURN_META_INVALID: dict[str, Callable[[], Any]] = {
    "tuple_value": lambda: {"a": (1, 2)},
    "tuple_in_list": lambda: {"a": [1, (2,)]},
    "set_value": lambda: {"a": {1, 2}},
    "bytes_value": lambda: {"a": b"x"},
    "object_value": lambda: {"a": object()},
    "non_str_key": lambda: {1: "a"},
    "nested_non_str_key": lambda: {"a": {None: 1}},
    "nan": lambda: {"a": float("nan")},
    "inf": lambda: {"a": [float("inf")]},
    "lone_surrogate_value": lambda: {"a": "\ud800"},
    "lone_surrogate_key": lambda: {"\udfff": 1},
    "too_deep": lambda: {"x": deep_value(MAX_DOCUMENT_DEPTH + 50)},
    "far_too_deep": lambda: {"x": deep_value(5000)},
}


@pytest.mark.parametrize("method", ["validate", "verify"])
@pytest.mark.parametrize("turn_index", [0, 3])
@pytest.mark.parametrize("name", list(TURN_META_INVALID))
def test_validate_rejects_invalid_turn_meta(name: str, turn_index: int, method: str) -> None:
    """不合法的 meta 一律 TrajectoryError（不得漏出 TypeError / RecursionError 等）。turn 3 是最後的 final turn。"""
    base = build(ASSIGN, MIXED_LAYOUT)
    assert len(base.turns) == 4
    getattr(base, method)()  # 控制組
    traj = set_turn(base, turn_index, R(base.turns[turn_index], meta=TURN_META_INVALID[name]()))
    with pytest.raises(TrajectoryError):
        getattr(traj, method)()


def test_verify_and_grade_ignore_turn_meta() -> None:
    """meta 是量測：不影響重放與評分。"""
    plain = build(ASSIGN, MIXED_LAYOUT)
    with_meta = build(ASSIGN, MIXED_LAYOUT, metas=[sample_meta(i) for i in range(len(plain.turns))])
    with_meta.verify()
    assert canon_records(with_meta.records()) == canon_records(plain.records())
    assert canonical_json(with_meta.grade().to_dict()) == canonical_json(plain.grade().to_dict())
    assert canonical_json(with_meta.usage().to_dict()) == canonical_json(plain.usage().to_dict())


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
@pytest.mark.parametrize("one_turn", [False, True])
def test_verify_accepts_reference_trajectories(task_id: str, one_turn: bool) -> None:
    layout = [[c for turn in reference_layout(task_id) for c in turn]] if one_turn else None
    build(task_id, layout).verify()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_verify_accepts_valid_variants(name: str) -> None:
    VARIANTS[name]().verify()


def test_verify_does_not_modify_trajectory() -> None:
    traj = build(ASSIGN, MIXED_LAYOUT)
    before = canonical_json(traj.to_dict())
    traj.verify()
    assert canonical_json(traj.to_dict()) == before


def test_verify_runs_validate_first() -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        R(base, max_turns=2).verify()


def test_verify_rejects_tampered_record_result() -> None:
    base = build(ASSIGN)
    result = copy.deepcopy(base.turns[0].tool_steps[0].result)
    result["issue"]["title"] = "tampered"
    traj = set_step(base, 0, 0, result=result)
    traj.validate()  # 結構上合法，只有重放擋得下
    with pytest.raises(TrajectoryError):
        traj.verify()


def test_verify_rejects_record_result_int_vs_float() -> None:
    base = build(ASSIGN)
    original = base.turns[1].tool_steps[0].result
    tampered, changed = first_int_to_float(original)
    assert changed and tampered == original  # Python == 會被騙
    assert canonical_json(tampered) != canonical_json(original)
    with pytest.raises(TrajectoryError):
        set_step(base, 1, 0, result=tampered).verify()


def test_verify_rejects_record_ok_as_int() -> None:
    base = build(ASSIGN)
    assert base.turns[0].tool_steps[0].ok is True
    traj = set_step(base, 0, 0, ok=1)
    assert traj.turns[0].tool_steps[0] == base.turns[0].tool_steps[0]  # Python == 會被騙
    with pytest.raises(TrajectoryError):
        traj.verify()


def test_verify_rejects_change_issue_id_as_float() -> None:
    base = build(ASSIGN)
    step = base.turns[0].tool_steps[0]
    change = step.changes[0]
    tampered = (R(change, issue_id=float(change.issue_id)), *step.changes[1:])
    assert tampered == step.changes  # Python == 會被騙
    with pytest.raises(TrajectoryError):
        set_step(base, 0, 0, changes=tampered).verify()


@pytest.mark.parametrize(
    "field, value",
    [("state_hash_after", FAKE_HASH), ("state_hash_before", FAKE_HASH), ("changes", ())],
)
def test_verify_rejects_tampered_record_fields(field: str, value: Any) -> None:
    base = build(ASSIGN)
    with pytest.raises(TrajectoryError):
        set_step(base, 0, 0, **{field: value}).verify()


def test_verify_rejects_tampered_failed_record_error() -> None:
    base = build(ASSIGN, MIXED_LAYOUT)
    step = base.turns[0].tool_steps[1]
    assert step.ok is False
    with pytest.raises(TrajectoryError):
        set_step(base, 0, 1, error={"code": step.error["code"], "message": "tampered"}).verify()


def test_verify_rejects_consistent_but_unreplayable_args() -> None:
    # call 與 step 的 args 一致（validate 通過），但 record 的結果是另一組 args 產生的
    base = build(ASSIGN)
    new_args = {"issue_id": 8, "assignee_id": "u3"}
    traj = set_step(set_call(base, 0, 0, args=new_args), 0, 0, args=new_args)
    traj.validate()
    with pytest.raises(TrajectoryError):
        traj.verify()


def test_verify_rejects_reordered_turns() -> None:
    base = build(ASSIGN)
    t0, t1, t2 = base.turns
    s0, s1 = t0.tool_steps[0], t1.tool_steps[0]
    new0 = R(t1, index=0, tool_steps=(R(s1, step=0),))
    new1 = R(t0, index=1, tool_steps=(R(s0, step=1),))
    traj = R(base, turns=(new0, new1, t2))
    traj.validate()  # 結構上合法（final state 也相同，兩個 call 可交換）
    with pytest.raises(TrajectoryError):
        traj.verify()


def test_verify_rejects_final_snapshot_rolled_back_to_initial_state() -> None:
    base = build(ASSIGN)
    snap = initial_snapshot(ASSIGN)
    snap["step"] = len(flat_records(base))
    traj = R(base, final_snapshot=snap)
    traj.validate()
    with pytest.raises(TrajectoryError):
        traj.verify()


def test_verify_rejects_rehashed_tampered_final_state() -> None:
    base = build(ASSIGN)
    snap = copy.deepcopy(base.final_snapshot)
    snap["state"]["issues"][0]["title"] = "tampered"
    snap = rehashed_snapshot(snap)
    traj = R(base, final_snapshot=snap)
    traj.validate()
    with pytest.raises(TrajectoryError):
        traj.verify()


@pytest.mark.parametrize("path", [("step",), ("state", "issues", 0, "id")], ids=["step", "issue_id"])
def test_verify_rejects_final_snapshot_int_vs_float(path: tuple[Any, ...]) -> None:
    base = build(ASSIGN)
    snap = copy.deepcopy(base.final_snapshot)
    target = snap
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = float(target[path[-1]])
    assert snap == base.final_snapshot  # Python == 會被騙
    with pytest.raises(TrajectoryError):
        R(base, final_snapshot=snap).verify()


# ---------------------------------------------------------------------------
# records / messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_records_flattens_in_order(name: str) -> None:
    traj = VARIANTS[name]()
    recs = traj.records()
    assert type(recs) is tuple
    assert all(isinstance(r, ToolCallRecord) for r in recs)
    assert canon_records(recs) == canon_records(flat_records(traj))
    assert [r.step for r in recs] == list(range(len(recs)))


def test_records_empty_for_final_answer_only() -> None:
    assert build(NOOP, []).records() == ()


def test_records_is_deep_copy() -> None:
    traj = build(ASSIGN)
    before = canonical_json(traj.to_dict())
    recs = traj.records()
    recs[0].result["issue"]["title"] = "tampered"
    recs[0].args["issue_id"] = 999
    assert canonical_json(traj.to_dict()) == before
    assert traj.records()[0].result["issue"]["title"] != "tampered"
    assert traj.turns[0].tool_steps[0].result["issue"]["title"] != "tampered"


@pytest.mark.parametrize("name", list(VARIANTS))
def test_messages_is_render_of_task_agent_turns(name: str) -> None:
    traj = VARIANTS[name]()
    msgs = traj.messages()
    expected = render_messages(traj.task["instruction"], traj.agent, traj.turns)
    assert json.dumps(msgs, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
    assert_messages_equal(msgs, expected_messages(traj.task["instruction"], traj.agent, traj.turns))


def test_messages_returns_new_objects() -> None:
    traj = build(ASSIGN)
    a = traj.messages()
    a[2]["content"] = "injected"
    a.append({"role": "user", "content": "more"})
    b = traj.messages()
    assert b[2]["content"] == traj.task["instruction"]
    assert len(b) == len(a) - 1


def test_messages_of_max_turns_trajectory_ends_with_tool_message() -> None:
    traj = build(REASSIGN, final=False)
    msgs = traj.messages()
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["content"] == canonical_json(traj.turns[-1].tool_steps[-1].observation())


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


def canon_usage(u: Usage) -> str:
    return canonical_json(u.to_dict())


@pytest.mark.parametrize("name", list(VARIANTS))
def test_usage_docstring_example(name: str) -> None:
    traj = VARIANTS[name]()
    u = traj.usage()
    assert isinstance(u, Usage)
    assert canon_usage(u) == canon_usage(Usage(0, 0, 0.0, 0.0, len(traj.turns)))
    assert type(u.input_tokens) is int and type(u.latency_s) is float and type(u.cost_usd) is float


def test_usage_sums_ints_as_ints() -> None:
    traj = with_usages(build(ASSIGN), [Usage(3, 0, 1, 2, 1), Usage(4, 0, 2, 0, 1), Usage(5, 7, 3, 1, 1)])
    u = traj.usage()
    expected = Usage(12, 7, 6, 3, 3)
    assert u == Usage(12, 7, 6.0, 3.0, 3)  # Python == 分不出 int / float
    assert canon_usage(u) == canon_usage(expected)
    assert type(u.latency_s) is int and type(u.cost_usd) is int


def test_usage_mixed_int_float() -> None:
    traj = with_usages(build(ASSIGN), [Usage(0, 0, 1, 0, 1), Usage(0, 0, 0.5, 0.25, 1), Usage(0, 0, 0, 0, 1)])
    u = traj.usage()
    assert canon_usage(u) == canon_usage(Usage(0, 0, 1.5, 0.25, 3))


def _naive_fold(values: list[Any]) -> Any:
    total: Any = 0
    for v in values:
        total = total + v
    return total


def test_usage_uses_builtin_sum_in_turn_order() -> None:
    latencies: list[Any] = [1, 2**53, 1.0]
    costs = [0.1, 0.2, 0.3]
    # 前提（Python >= 3.12）：內建 sum 的結果與 math.fsum / 逐項 += 不同，因此只有照契約用內建 sum 才會相符
    assert sum(latencies) != math.fsum(latencies)
    assert sum(costs) != _naive_fold(costs)
    traj = with_usages(build(ASSIGN), [Usage(0, 0, lat, c, 1) for lat, c in zip(latencies, costs, strict=True)])
    u = traj.usage()
    assert canon_usage(u) == canon_usage(Usage(0, 0, sum(latencies), sum(costs), 3))


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "latency_s", "cost_usd"])
def test_usage_none_in_any_turn_propagates(field: str) -> None:
    base = Usage(2, 3, 0.5, 0.25, 1)
    usages = [base, R(base, **{field: None}), base]
    u = with_usages(build(ASSIGN), usages).usage()
    expected = {"input_tokens": 6, "output_tokens": 9, "latency_s": 1.5, "cost_usd": 0.75, "model_turns": 3}
    expected[field] = None
    assert canonical_json(u.to_dict()) == canonical_json(expected)


def test_usage_all_unknown() -> None:
    traj = build(ASSIGN)
    unknown = Usage(None, None, None, None, 1)
    u = with_usages(traj, [unknown] * len(traj.turns)).usage()
    assert canon_usage(u) == canon_usage(Usage(None, None, None, None, len(traj.turns)))


def test_usage_model_turns_is_len_turns() -> None:
    traj = build(REASSIGN, final=False)
    assert len(traj.turns) == 7
    assert traj.usage().model_turns == 7


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_grade_matches_grader_on_reference(task_id: str) -> None:
    traj = build(task_id)
    report = traj.grade()
    expected = grade(
        get_smoke_task(task_id),
        flat_records(traj),
        traj.final_snapshot["state"],
        Usage(0, 0, 0.0, 0.0, len(traj.turns)),
    )
    assert canonical_json(report.to_dict()) == canonical_json(expected.to_dict())
    assert report.success is True and report.trajectory_safe is True


@pytest.mark.parametrize(
    "layout",
    [[], [[("get_issue", {"issue_id": 8})]], MIXED_LAYOUT, [[("update_issue", {"issue_id": 8, "status": "closed"})]]],
    ids=["empty", "read_only", "mixed", "forbidden_close"],
)
def test_grade_matches_grader_on_other_trajectories(layout: list[list[tuple[str, Any]]]) -> None:
    traj = build(ASSIGN, layout)
    usages = [Usage(i, 2 * i, 0.5 * i, 0.125, 1) for i in range(len(traj.turns))]
    traj = with_usages(traj, usages)
    expected = grade(
        get_smoke_task(ASSIGN),
        flat_records(traj),
        traj.final_snapshot["state"],
        Usage(
            sum(u.input_tokens for u in usages),
            sum(u.output_tokens for u in usages),
            sum(u.latency_s for u in usages),
            sum(u.cost_usd for u in usages),
            len(usages),
        ),
    )
    report = traj.grade()
    assert canonical_json(report.to_dict()) == canonical_json(expected.to_dict())


def test_grade_reports_failure_for_unsafe_trajectory() -> None:
    traj = build(ASSIGN, [[("update_issue", {"issue_id": 8, "status": "closed"})]])
    report = traj.grade()
    assert report.trajectory_safe is False
    assert report.success is False


# ---------------------------------------------------------------------------
# trajectory_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_trajectory_id_is_sha256_of_to_dict(name: str) -> None:
    traj = VARIANTS[name]()
    tid = traj.trajectory_id
    assert tid == sha256_of(traj.to_dict())
    assert tid.startswith("sha256:") and len(tid) == len("sha256:") + 64


def test_trajectory_id_is_deterministic() -> None:
    assert build(ASSIGN, MIXED_LAYOUT).trajectory_id == build(ASSIGN, MIXED_LAYOUT).trajectory_id


@pytest.mark.parametrize(
    "change",
    [
        lambda t: with_usages(t, [R(t.turns[0].usage, latency_s=0.5), *[x.usage for x in t.turns[1:]]]),
        lambda t: with_usages(t, [R(t.turns[0].usage, input_tokens=1), *[x.usage for x in t.turns[1:]]]),
        lambda t: with_usages(t, [R(t.turns[0].usage, cost_usd=None), *[x.usage for x in t.turns[1:]]]),
        lambda t: R(t, parent=make_parent(0)),
        lambda t: R(t, max_turns=t.max_turns + 1),
        lambda t: R(t, agent=make_agent(system_prompt="other")),
        lambda t: set_turn(t, 0, R(t.turns[0], meta=sample_meta(0))),
        lambda t: set_turn(t, 2, R(t.turns[2], meta={})),
    ],
    ids=["latency", "tokens", "cost_unknown", "parent", "max_turns", "agent", "meta_set", "meta_empty"],
)
def test_trajectory_id_changes_with_content(change: Callable[[Trajectory], Trajectory]) -> None:
    base = build(ASSIGN)
    changed = change(base)
    changed.validate()
    assert changed.trajectory_id != base.trajectory_id


def test_trajectory_id_distinguishes_meta_values() -> None:
    """trajectory_id 是紀錄 id（含量測）：meta 的任何差異（含 1 vs 1.0、None vs {}）都改變它。"""
    base = build(ASSIGN)
    as_int = set_turn(base, 0, R(base.turns[0], meta={"ratio": 1}))
    as_float = set_turn(base, 0, R(base.turns[0], meta={"ratio": 1.0}))
    empty = set_turn(base, 0, R(base.turns[0], meta={}))
    assert as_int == as_float  # Python == 會被騙
    ids = {base.trajectory_id, as_int.trajectory_id, as_float.trajectory_id, empty.trajectory_id}
    assert len(ids) == 4


def test_trajectory_id_distinguishes_int_and_float_latency() -> None:
    base = build(ASSIGN)
    as_int = with_usages(base, [R(t.usage, latency_s=0) for t in base.turns])
    as_float = with_usages(base, [R(t.usage, latency_s=0.0) for t in base.turns])
    assert as_int == as_float  # Python == 會被騙
    assert as_int.trajectory_id != as_float.trajectory_id


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_to_dict_shape(name: str) -> None:
    traj = VARIANTS[name]()
    d = traj.to_dict()
    assert set(d) == set(TRAJECTORY_KEYS)
    assert len(d) == len(TRAJECTORY_KEYS)
    assert_json_native(d)
    assert d["format"] == TRAJECTORY_FORMAT
    assert canonical_json(d["task"]) == canonical_json(traj.task)
    assert canonical_json(d["agent"]) == canonical_json(traj.agent.to_dict())
    assert type(d["max_turns"]) is int and d["max_turns"] == traj.max_turns
    assert type(d["turns"]) is list
    assert canonical_json(d["turns"]) == canonical_json([t.to_dict() for t in traj.turns])
    assert d["termination"] == traj.termination
    assert canonical_json(d["final_snapshot"]) == canonical_json(traj.final_snapshot)
    assert canonical_json(d["parent"]) == canonical_json(traj.parent)


def test_to_dict_meta_is_deep_copy() -> None:
    traj = build(ASSIGN, metas=[sample_meta(0), None, sample_meta(2)])
    before = canonical_json(traj.to_dict())
    d = traj.to_dict()
    assert d["turns"][1]["meta"] is None
    assert d["turns"][0]["meta"] is not traj.turns[0].meta
    d["turns"][0]["meta"]["usage_detail"]["input"] = -1
    d["turns"][2]["meta"]["list"].clear()
    d["turns"][1]["meta"] = {"injected": True}
    assert canonical_json(traj.to_dict()) == before


def test_to_dict_parent_none() -> None:
    assert build(ASSIGN).to_dict()["parent"] is None


def test_to_dict_is_deep_copy() -> None:
    traj = build(ESCALATE, parent=make_parent(1))
    before = canonical_json(traj.to_dict())
    d = traj.to_dict()
    assert d["task"] is not traj.task
    assert d["final_snapshot"] is not traj.final_snapshot
    assert d["parent"] is not traj.parent
    d["task"]["instruction"] = "tampered"
    d["task"]["initial_state"]["issues"].clear()
    d["final_snapshot"]["state"]["issues"][0]["title"] = "tampered"
    d["agent"]["model_config"]["temperature"] = 1.0
    d["turns"][0]["message"]["tool_calls"][0]["args"]["label"] = "tampered"
    d["turns"][0]["tool_steps"][0]["result"]["tampered"] = True
    d["parent"]["turn"] = 0
    assert canonical_json(traj.to_dict()) == before


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_from_dict_round_trip(name: str) -> None:
    traj = VARIANTS[name]()
    back = Trajectory.from_dict(traj.to_dict())
    assert back == traj
    assert canonical_json(back.to_dict()) == canonical_json(traj.to_dict())


@pytest.mark.parametrize("name", list(VARIANTS))
def test_from_dict_round_trip_via_canonical_json(name: str) -> None:
    traj = VARIANTS[name]()
    text = canonical_json(traj.to_dict())
    back = Trajectory.from_dict(json.loads(text))
    assert back == traj
    assert canonical_json(back.to_dict()) == text
    assert back.trajectory_id == traj.trajectory_id


def test_from_dict_round_trip_preserves_int_latency() -> None:
    base = build(ASSIGN)
    traj = with_usages(base, [R(t.usage, latency_s=0, cost_usd=0) for t in base.turns])
    text = canonical_json(traj.to_dict())
    back = Trajectory.from_dict(json.loads(text))
    assert canonical_json(back.to_dict()) == text
    assert all(type(t.usage.latency_s) is int for t in back.turns)


TRAJ_DICT_ERRORS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "not_dict_list": lambda d: [d],
    "not_dict_none": lambda d: None,
    "not_dict_str": lambda d: canonical_json(d),
    **{f"missing_{k}": _drop(k) for k in TRAJECTORY_KEYS},
    "extra_key": _set(("extra",), None),
    "format_v1": _set(("format",), "agent-gym/trajectory/v1"),  # v1 已被 v2 取代（decision 009）
    "format_v3": _set(("format",), "agent-gym/trajectory/v3"),
    "format_none": _set(("format",), None),
    "format_checkpoint": _set(("format",), "agent-gym/checkpoint/v2"),
    "task_missing_key": _del_nested(("task", "instruction")),
    "task_not_dict": _set(("task",), "task"),
    "agent_none": _set(("agent",), None),
    "agent_missing_key": _del_nested(("agent", "policy")),
    "agent_extra_key": _set(("agent", "extra"), 1),
    "agent_model_config_list": _set(("agent", "model_config"), []),
    "max_turns_true": _set(("max_turns",), True),
    "max_turns_float": _set(("max_turns",), 5.0),
    "max_turns_str": _set(("max_turns",), "5"),
    "max_turns_too_small": _set(("max_turns",), 2),
    "turns_none": _set(("turns",), None),
    "turns_dict": _set(("turns",), {}),
    "turns_str": _set(("turns",), "turns"),
    "turns_empty": _set(("turns",), []),
    "turn_not_dict": _set(("turns", 0), None),
    "turn_missing_key": _del_nested(("turns", 0, "usage")),
    "turn_missing_meta": _del_nested(("turns", 0, "meta")),
    "turn_final_missing_meta": _del_nested(("turns", 2, "meta")),
    "turn_meta_list": _set(("turns", 0, "meta"), []),
    "turn_meta_str": _set(("turns", 0, "meta"), "meta"),
    "turn_meta_int": _set(("turns", 1, "meta"), 1),
    "turn_meta_nan": _set(("turns", 0, "meta"), {"x": float("nan")}),
    "turn_meta_lone_surrogate": _set(("turns", 0, "meta"), {"x": "\ud800"}),
    "turn_meta_tuple": _set(("turns", 0, "meta"), {"x": (1,)}),
    "turn_meta_too_deep": _set(("turns", 0, "meta"), {"x": deep_value(MAX_DOCUMENT_DEPTH + 50)}),
    "turn_bad_index": _set(("turns", 1, "index"), 5),
    "turn_bool_index": _set(("turns", 1, "index"), True),
    "turn_bad_usage": _set(("turns", 0, "usage", "model_turns"), "1"),
    "turn_bad_message": _del_nested(("turns", 0, "message", "stop_reason")),
    "turn_bad_step": _set(("turns", 0, "tool_steps", 0, "ok"), "yes"),
    "turn_step_args_float": _set(("turns", 0, "tool_steps", 0, "args", "issue_id"), 8.0),
    "termination_unknown": _set(("termination",), "done"),
    "termination_mismatch": _set(("termination",), "max_turns"),
    "final_snapshot_bad_hash": _set(("final_snapshot", "state_hash"), FAKE_HASH),
    "final_snapshot_missing_key": _del_nested(("final_snapshot", "step")),
    "final_snapshot_step_true": _set(("final_snapshot", "step"), True),
    "final_snapshot_none": _set(("final_snapshot",), None),
    "parent_bad_keys": _set(("parent",), {"trajectory_id": "x", "turn": 0}),
    "parent_turn_out_of_range": _set(("parent",), {"trajectory_id": "x", "checkpoint_id": "y", "turn": 3}),
    "parent_list": _set(("parent",), []),
}


@pytest.mark.parametrize("name", list(TRAJ_DICT_ERRORS))
def test_from_dict_rejects_malformed(name: str) -> None:
    traj = build(ASSIGN)
    assert len(traj.turns) == 3 and traj.max_turns == 5
    data = TRAJ_DICT_ERRORS[name](traj.to_dict())
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(data)


def test_from_dict_accepts_valid_parent() -> None:
    traj = build(ASSIGN)
    d = traj.to_dict()
    d["parent"] = make_parent(2)
    back = Trajectory.from_dict(d)
    assert canonical_json(back.parent) == canonical_json(make_parent(2))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_save_load_round_trip(tmp_path: Path, name: str) -> None:
    traj = VARIANTS[name]()
    path = tmp_path / "t.json"
    traj.save(path)
    loaded = Trajectory.load(path)
    assert loaded == traj
    assert canonical_json(loaded.to_dict()) == canonical_json(traj.to_dict())
    assert loaded.trajectory_id == traj.trajectory_id


def test_save_writes_canonical_json_plus_newline(tmp_path: Path) -> None:
    traj = build(ASSIGN)
    path = tmp_path / "t.json"
    traj.save(path)
    raw = path.read_bytes()
    assert raw == (canonical_json(traj.to_dict()) + "\n").encode("utf-8")
    assert "你好".encode("utf-8") in raw  # UTF-8，不轉成 \\u 跳脫


def test_save_and_load_accept_str_paths(tmp_path: Path) -> None:
    traj = build(NOOP)
    path = str(tmp_path / "t.json")
    traj.save(path)
    assert Trajectory.load(path) == traj


def test_load_accepts_file_without_trailing_newline(tmp_path: Path) -> None:
    traj = build(ASSIGN, MIXED_LAYOUT)
    path = tmp_path / "t.json"
    path.write_bytes(canonical_json(traj.to_dict()).encode("utf-8"))
    loaded = Trajectory.load(path)
    assert loaded == traj
    assert canonical_json(loaded.to_dict()) == canonical_json(traj.to_dict())


def test_load_preserves_int_vs_float(tmp_path: Path) -> None:
    base = build(ASSIGN)
    traj = with_usages(base, [R(t.usage, latency_s=0) for t in base.turns])
    path = tmp_path / "t.json"
    traj.save(path)
    loaded = Trajectory.load(path)
    assert canonical_json(loaded.to_dict()) == canonical_json(traj.to_dict())
    assert loaded.trajectory_id == traj.trajectory_id


def _saved_bytes(tmp_path: Path) -> tuple[Trajectory, bytes]:
    traj = build(ASSIGN)
    path = tmp_path / "src.json"
    traj.save(path)
    return traj, path.read_bytes()


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda b: b"\xff" + b,
        lambda b: b.replace(SYSTEM_MARKER.encode(), b"\xed\xa0\x80"),  # UTF-8 編碼的 lone surrogate
        lambda b: b.replace(SYSTEM_MARKER.encode(), b"\xc0\xaf"),  # overlong encoding
        lambda b: b.replace("你好".encode("utf-8"), "你好".encode("utf-8")[:-1]),  # 截斷的多 byte 字元
        lambda b: b.replace(SYSTEM_MARKER.encode(), b"\x80abc"),  # 孤立的 continuation byte
    ],
    ids=["leading_ff", "encoded_surrogate", "overlong", "truncated_multibyte", "lone_continuation"],
)
def test_load_rejects_invalid_utf8(tmp_path: Path, corrupt: Callable[[bytes], bytes]) -> None:
    _, raw = _saved_bytes(tmp_path)
    bad = corrupt(raw)
    assert bad != raw
    with pytest.raises(UnicodeDecodeError):
        bad.decode("utf-8")
    path = tmp_path / "bad.json"
    path.write_bytes(bad)
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_non_finite_literals(tmp_path: Path, literal: str) -> None:
    _, raw = _saved_bytes(tmp_path)
    text = raw.decode("utf-8")
    assert '"latency_s":0.0' in text
    path = tmp_path / "bad.json"
    path.write_text(text.replace('"latency_s":0.0', f'"latency_s":{literal}', 1), encoding="utf-8")
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_load_rejects_duplicate_top_level_key_even_with_identical_value(tmp_path: Path) -> None:
    traj, raw = _saved_bytes(tmp_path)
    text = raw.decode("utf-8")
    assert text.startswith("{")
    dup = '{"parent":null,' + text[1:]
    # 前提：一般 json.loads 會默默接受，且得到相同的 dict
    assert canonical_json(json.loads(dup)) == canonical_json(traj.to_dict())
    path = tmp_path / "dup.json"
    path.write_text(dup, encoding="utf-8")
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_load_rejects_duplicate_nested_key(tmp_path: Path) -> None:
    traj, raw = _saved_bytes(tmp_path)
    text = raw.decode("utf-8")
    needle = '"agent":{'
    assert needle in text
    dup = text.replace(needle, needle + f'"policy":{json.dumps(traj.agent.policy)},', 1)
    assert canonical_json(json.loads(dup)) == canonical_json(traj.to_dict())
    path = tmp_path / "dup.json"
    path.write_text(dup, encoding="utf-8")
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


@pytest.mark.parametrize(
    "text",
    ["", "{", "[]", "null", '"trajectory"', "1", '{"format": "agent-gym/trajectory/v1"}', "{'a': 1}"],
    ids=["empty", "open_brace", "list", "null", "string", "number", "partial", "single_quotes"],
)
def test_load_rejects_malformed_or_non_trajectory_json(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_load_rejects_truncated_file(tmp_path: Path) -> None:
    _, raw = _saved_bytes(tmp_path)
    path = tmp_path / "bad.json"
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_load_rejects_structurally_invalid_trajectory(tmp_path: Path) -> None:
    traj = build(ASSIGN)
    d = traj.to_dict()
    d["max_turns"] = 2
    path = tmp_path / "bad.json"
    path.write_text(canonical_json(d) + "\n", encoding="utf-8")
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_load_rejects_json_escaped_lone_surrogate(tmp_path: Path) -> None:
    # bytes 是合法 UTF-8（跳脫序列本身是 ASCII），但解析出的字串無法 UTF-8 編碼 → 規則 9
    traj = build(ASSIGN, parent=make_parent(1))
    text = canonical_json(traj.to_dict())
    assert '"trajectory_id":"sha256:' in text
    text = text.replace('"trajectory_id":"sha256:', '"trajectory_id":"\\ud800sha256:', 1)
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_load_missing_file_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Trajectory.load(tmp_path / "missing.json")


def test_load_directory_raises_os_error_not_trajectory_error(tmp_path: Path) -> None:
    with pytest.raises(OSError) as exc_info:
        Trajectory.load(tmp_path)
    assert not isinstance(exc_info.value, TrajectoryError)


# ---------------------------------------------------------------------------
# load_strict_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, canonical",
    [
        (b'{"a": [1, 2.5, true, null, "x"]}', '{"a":[1,2.5,true,null,"x"]}'),
        (b"[1, 1.0, true, 0, false]", "[1,1.0,true,0,false]"),
        (b'{"a":1}\n', '{"a":1}'),
        ('{"名":"值"}'.encode("utf-8"), '{"名":"值"}'),
        (b'"\\u00e9"', '"é"'),
        (b"  [ ]  ", "[]"),
        (b"-0.5", "-0.5"),
    ],
    ids=["object", "int_float_bool", "trailing_newline", "non_ascii", "escape", "whitespace", "number"],
)
def test_load_strict_json_parses_valid(raw: bytes, canonical: str) -> None:
    value = load_strict_json(raw)
    assert canonical_json(value) == canonical


def test_load_strict_json_keeps_int_float_bool_distinct() -> None:
    value = load_strict_json(b"[1, 1.0, true]")
    assert [type(v) for v in value] == [int, float, bool]


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b'"\xed\xa0\x80"',
        b'"\xc0\xaf"',
        b'{"a": "\xe4\xbd"}',
        b"NaN",
        b"[Infinity]",
        b'{"a": -Infinity}',
        b'[1, {"b": NaN}]',
        b'{"a": 1, "a": 1}',
        b'{"a": 1, "a": 2}',
        b'{"a": {"b": null, "b": null}}',
        b'[{"x": 0, "x": 0}]',
        b"",
        b"{",
        b"[1,]",
        b"{'a': 1}",
        b'{"a": 1} {"b": 2}',
    ],
    ids=[
        "invalid_byte",
        "encoded_surrogate",
        "overlong",
        "truncated_multibyte",
        "nan",
        "infinity",
        "neg_infinity",
        "nested_nan",
        "dup_same_value",
        "dup_different_value",
        "dup_nested",
        "dup_in_list",
        "empty",
        "unclosed",
        "trailing_comma",
        "single_quotes",
        "extra_data",
    ],
)
def test_load_strict_json_rejects(raw: bytes) -> None:
    with pytest.raises(ValueError):
        load_strict_json(raw)


# ---------------------------------------------------------------------------
# Usage.from_dict
# ---------------------------------------------------------------------------


USAGE_SAMPLES = [
    Usage(),
    Usage(1, 2, 0.5, 0.01, 3),
    Usage(0, 0, 0, 0, 1),
    Usage(0, 0, 0.0, 0.0, 1),
    Usage(10, None, 1.25, None, None),
    Usage(latency_s=-0.5, cost_usd=2),
]


@pytest.mark.parametrize("usage", USAGE_SAMPLES, ids=repr)
def test_usage_from_dict_round_trip(usage: Usage) -> None:
    back = Usage.from_dict(usage.to_dict())
    assert back == usage
    assert canonical_json(back.to_dict()) == canonical_json(usage.to_dict())
    via_json = Usage.from_dict(json.loads(canonical_json(usage.to_dict())))
    assert canonical_json(via_json.to_dict()) == canonical_json(usage.to_dict())


def test_usage_from_dict_preserves_int_vs_float() -> None:
    as_int = Usage.from_dict({"input_tokens": 0, "output_tokens": 0, "latency_s": 0, "cost_usd": 0, "model_turns": 1})
    as_float = Usage.from_dict(
        {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0, "model_turns": 1}
    )
    assert as_int == as_float  # Python == 會被騙
    assert type(as_int.latency_s) is int and type(as_float.latency_s) is float
    assert canonical_json(as_int.to_dict()) != canonical_json(as_float.to_dict())


def _usage_dict(**overrides: Any) -> dict[str, Any]:
    d = Usage(1, 2, 0.5, 0.25, 1).to_dict()
    d.update(overrides)
    return d


def _usage_without(key: str) -> dict[str, Any]:
    d = _usage_dict()
    del d[key]
    return d


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        "usage",
        list(_usage_dict().items()),
        {},
        *[_usage_without(k) for k in USAGE_FIELDS],
        _usage_dict(extra=0),
        _usage_dict(model_turn=1),
    ],
    ids=[
        "none",
        "list",
        "str",
        "pairs",
        "empty",
        *[f"missing_{k}" for k in USAGE_FIELDS],
        "extra",
        "typo_key",
    ],
)
def test_usage_from_dict_rejects_bad_shape(data: Any) -> None:
    with pytest.raises(ValueError):
        Usage.from_dict(data)


def test_usage_from_dict_rejects_usage_instance() -> None:
    with pytest.raises(ValueError):
        Usage.from_dict(Usage())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field, value",
    [
        ("input_tokens", "1"),
        ("input_tokens", True),
        ("input_tokens", 1.0),
        ("output_tokens", False),
        ("output_tokens", [1]),
        ("model_turns", True),
        ("model_turns", 1.0),
        ("latency_s", "0.5"),
        ("latency_s", True),
        ("cost_usd", [0.1]),
        ("cost_usd", {"usd": 0.1}),
    ],
    ids=repr,
)
def test_usage_from_dict_rejects_wrong_value_types(field: str, value: Any) -> None:
    # grader/schema.py：值的驗證同建構子（TypeError / ValueError）
    with pytest.raises((TypeError, ValueError)):
        Usage.from_dict(_usage_dict(**{field: value}))


@pytest.mark.parametrize(
    "field, value",
    [("latency_s", float("nan")), ("latency_s", float("inf")), ("cost_usd", float("-inf"))],
    ids=repr,
)
def test_usage_from_dict_rejects_non_finite(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        Usage.from_dict(_usage_dict(**{field: value}))
