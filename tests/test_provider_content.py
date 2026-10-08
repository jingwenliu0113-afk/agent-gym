"""ModelMessage.provider_content、PolicyStep.meta / Turn.meta 與 trajectory / checkpoint v2 的契約測試（decision log 009 第 7 點）。

依據：
- agents/policy.py：模組 docstring「統一規則」（正規化、例外型別、UTF-8、from_dict 一律 ValueError）；
  ModelMessage docstring（provider_content 為 None 或元素皆為 dict 的 list；JSON 值、深度上限、正規化深拷貝；
  to_dict 剛好 4 個 key；from_dict 要求 4 個 key；本類別不解讀 provider_content）；
  PolicyStep docstring（meta 為 None 或 dict，內容規則同 model_config：JSON 值、深度不超過 MAX_MODEL_CONFIG_DEPTH、
  字串可 UTF-8 編碼；以正規化深拷貝保存；meta 是量測，不進 conversation）。
- replay/trajectory.py：``render_messages``（assistant 訊息 key 順序 role, content, tool_calls, provider_content；不含 meta）、
  ``TRAJECTORY_FORMAT``、``TURN_KEYS``（5 個 key，含 meta）、``Turn.to_dict`` / ``Turn.from_dict``（meta 為 None 或 dict、
  必須有 meta key）、Trajectory 規則 9 / 10（每個 turn 的 meta 必須是 JSON 文件）。
- replay/runner.py：迴圈第 3 步 ``Turn(..., meta=step.meta)``。
- replay/checkpoint.py：``CHECKPOINT_FORMAT``、turns 與 Trajectory 相同編碼、核心 invariant（restore 後的 conversation 與
  原 episode 中 policy 實際收到的逐 byte 相同）。

Trajectory 以真實 IssueTrackerEnv 的 records 手工組裝（smoke tasks 只做回歸用途）。provider block 的類型取自 decision 009
第 4 點的白名單（text / thinking / redacted_thinking / tool_use），保存哪些欄位由本檔選定（ModelMessage 不解讀內容），
由 tests/fake_anthropic.py 的 fake block 轉成 dict（key 順序刻意不排序，用來驗證正規化）。
所有「逐 byte 相同」都以 canonical_json 或對順序敏感的 ``json.dumps`` 比較，並放入 Python ``==`` 會被騙過的案例（1 vs 1.0）。
不呼叫任何真正的 API、不建立任何 SDK client、不需要 credential；模組層級不建構任何 ModelMessage / PolicyStep（只在測試內建構）。

深度慣例（契約模糊處）：「巢狀深度不超過 MAX_MODEL_CONFIG_DEPTH」對 provider_content 沒有說明以 list 本身或每個 block
為第 0 層。本檔只用兩種慣例都一致的案例：最內層值位於 provider_content 根下第 MAX_MODEL_CONFIG_DEPTH 層 → 接受；
第 MAX_MODEL_CONFIG_DEPTH + 2 層以上 → ValueError（不測第 MAX + 1 層的邊界）。
PolicyStep.meta 的規則明文「同 model_config」，因此以「與 AgentConfig.model_config 的接受 / 拒絕結果逐一相同」測邊界。
Turn.meta 在 Trajectory 文件中的深度邊界（MAX_DOCUMENT_DEPTH 以 meta 或文件根起算）未明定，只測遠超過上限（5000 層）的案例。
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from agents.policy import MAX_MODEL_CONFIG_DEPTH, AgentConfig, ModelMessage, PolicyStep, ToolCall, oracle_policy
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json, sha256_of
from grader.schema import Usage
from replay.checkpoint import (
    CHECKPOINT_FORMAT,
    Checkpoint,
    CheckpointError,
    make_checkpoint,
    rerun,
    restore,
    verify_lineage,
)
from replay.runner import run_episode
from replay.trajectory import TRAJECTORY_FORMAT, TURN_KEYS, Trajectory, TrajectoryError, Turn, render_messages
from tasks.smoke import REFERENCE_SOLUTIONS, SMOKE_TASK_IDS, get_smoke_task
from tests.fake_anthropic import FakeBlock, redacted_thinking_block, text_block, thinking_block, tool_use_block

R = dataclasses.replace

ASSIGN = "smoke-assign-and-start"
REASSIGN = "smoke-reassign-departed"

NAN = float("nan")
INF = float("inf")

MESSAGE_KEYS = {"text", "tool_calls", "stop_reason", "provider_content"}
ROLE_KEYS: dict[str, list[str]] = {
    "system": ["role", "content"],
    "developer": ["role", "content"],
    "user": ["role", "content"],
    "assistant": ["role", "content", "tool_calls", "provider_content"],
    "tool": ["role", "tool_call_id", "content"],
}

# provider block 的白名單欄位（decision 009 第 7 點）；dict 依此順序建立 → key 刻意不是排序後的順序。
BLOCK_FIELDS: dict[str, tuple[str, ...]] = {
    "text": ("type", "text"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
    "tool_use": ("type", "id", "name", "input"),
}

PROVIDER_MODES = ("all", "mixed", "none")


# ---------------------------------------------------------------------------
# helpers（純資料；所有 ModelMessage / Trajectory / Checkpoint 都在測試內建構）
# ---------------------------------------------------------------------------


def _nested(depth: int) -> Any:
    """depth 層 list 包住 1。"""
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


def _deep_pc(n: int) -> list[dict[str, Any]]:
    """``[{"type": "deep", "v": _nested(n)}]``：最內層值位於 provider_content 根（第 0 層）下第 n + 2 層。"""
    return [{"type": "deep", "v": _nested(n)}]


DEPTH_AT_LIMIT = MAX_MODEL_CONFIG_DEPTH - 2  # 最內層值剛好在第 MAX_MODEL_CONFIG_DEPTH 層
DEPTH_PAST_LIMIT = MAX_MODEL_CONFIG_DEPTH  # 最內層值在第 MAX_MODEL_CONFIG_DEPTH + 2 層


def _ordered_json(value: Any) -> str:
    """對 key 順序敏感的 JSON（不排序），用來驗證「key 已遞迴排序」。"""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _assert_keys_sorted_recursively(value: Any) -> None:
    if isinstance(value, dict):
        assert list(value) == sorted(value)
        for item in value.values():
            _assert_keys_sorted_recursively(item)
    elif isinstance(value, list):
        for item in value:
            _assert_keys_sorted_recursively(item)


def _assert_json_native(value: Any) -> None:
    if value is None or type(value) in (str, int, float, bool):
        return
    if type(value) is list:
        for item in value:
            _assert_json_native(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_json_native(item)
        return
    raise AssertionError(f"not JSON-native: {type(value).__name__}")


def block_dict(block: FakeBlock) -> dict[str, Any]:
    """依白名單把 fake block 轉成 dict（不含 extra 等其他屬性）。"""
    return {field: copy.deepcopy(getattr(block, field)) for field in BLOCK_FIELDS[block.type]}


def sample_blocks() -> list[dict[str, Any]]:
    """四種白名單 block 各一個（含非 ASCII 字串）。"""
    return [
        block_dict(thinking_block("sig-α", thinking="考慮 #8 的指派")),
        block_dict(redacted_thinking_block("cmVkYWN0ZWQ=")),
        block_dict(text_block("I'll assign it. 好")),
        block_dict(tool_use_block("toolu_01", "assign_issue", {"issue_id": 8, "assignee_id": "u4"})),
    ]


def rich_block() -> dict[str, Any]:
    """涵蓋所有 JSON 葉節點型別與巢狀容器；key 刻意以反序建立。"""
    return {
        "type": "custom",
        "z": None,
        "y": True,
        "x": 1,
        "w": 1.5,
        "v": "登入 ✓",
        "u": [1, 1.0, False, None, {"b": 2, "a": [{"d": 0, "c": -0.25}]}],
        "t": {},
        "s": [],
    }


def make_msg(provider_content: Any, *, text: str = "", calls: tuple[ToolCall, ...] = (), stop_reason: Any = "end_turn"
             ) -> ModelMessage:
    return ModelMessage(text=text, tool_calls=calls, stop_reason=stop_reason, provider_content=provider_content)


def tool_turn_content(k: int, name: str, args: Any, call_id: str, text: str) -> list[dict[str, Any]]:
    """一個 tool turn 的 provider blocks：thinking（+ text）+ tool_use。"""
    blocks = [block_dict(thinking_block(f"sig-{k}", thinking=f"思考 {k}"))]
    if text:
        blocks.append(block_dict(text_block(text)))
    blocks.append(block_dict(tool_use_block(call_id, name, copy.deepcopy(args))))
    return blocks


def final_turn_content() -> list[dict[str, Any]]:
    return [
        block_dict(thinking_block("sig-final", thinking="wrap up")),
        block_dict(redacted_thinking_block("ZmluYWw=")),
        block_dict(text_block("Done.")),
    ]


def turn_text(k: int) -> str:
    return f"note {k} 登入" if k % 2 == 0 else ""


def planned_contents(task_id: str, mode: str) -> list[list[dict[str, Any]] | None]:
    """build_trajectory 每個 turn 原始輸入的 provider_content（未正規化；最後一個是 final turn）。

    mode："all" 每個 turn 都有；"mixed" 只有奇數 tool turn 與 final turn 有；"none" 全部為 None（如 ScriptedPolicy）。
    """
    contents: list[list[dict[str, Any]] | None] = []
    for k, (name, args) in enumerate(REFERENCE_SOLUTIONS[task_id]):
        has_pc = mode == "all" or (mode == "mixed" and k % 2 == 1)
        contents.append(tool_turn_content(k, name, args, f"toolu_{k:02d}", turn_text(k)) if has_pc else None)
    contents.append(final_turn_content() if mode in ("all", "mixed") else None)
    return contents


def zero_usage() -> Usage:
    return Usage(0, 0, 0.0, 0.0, 1)


def make_agent() -> AgentConfig:
    return AgentConfig(
        policy="hand-built-provider",
        prompt_version="test/v2",
        system_prompt="You are a triage bot. 你好",
        developer_prompt="Use the tools.",
        model_config={"model": "fake-model", "max_tokens": 256},
    )


def build_trajectory(task_id: str = ASSIGN, mode: str = "all") -> Trajectory:
    """以真實 env records 手工組裝 final_answer trajectory（每個 oracle call 一個 turn）。"""
    task = get_smoke_task(task_id)
    contents = planned_contents(task_id, mode)
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    turns: list[Turn] = []
    for k, (name, args) in enumerate(REFERENCE_SOLUTIONS[task_id]):
        n = len(env.records)
        env.call_tool(name, copy.deepcopy(args))
        steps = env.records[n:]
        message = ModelMessage(
            text=turn_text(k),
            tool_calls=(ToolCall(f"toolu_{k:02d}", name, copy.deepcopy(args)),),
            stop_reason="tool_use",
            provider_content=copy.deepcopy(contents[k]),
        )
        turns.append(Turn(index=k, message=message, tool_steps=tuple(steps), usage=zero_usage()))
    final = ModelMessage(
        text="Done.", tool_calls=(), stop_reason="end_turn", provider_content=copy.deepcopy(contents[-1])
    )
    turns.append(Turn(index=len(turns), message=final, tool_steps=(), usage=zero_usage()))
    return Trajectory(
        task=task.to_dict(),
        agent=make_agent(),
        max_turns=len(turns) + 2,
        turns=tuple(turns),
        termination="final_answer",
        final_snapshot=env.snapshot(),
        parent=None,
    )


def expected_messages(instruction: str, agent: AgentConfig, turns: Any, contents: list[Any]) -> list[dict[str, Any]]:
    """依 render_messages docstring 逐字組出的期望 conversation；provider_content 由原始輸入正規化而來。"""
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "developer", "content": agent.developer_prompt},
        {"role": "user", "content": instruction},
    ]
    for turn, pc in zip(turns, contents):
        calls = turn.message.tool_calls
        msgs.append(
            {
                "role": "assistant",
                "content": turn.message.text,
                "tool_calls": [{"id": c.id, "name": c.name, "args": json.loads(canonical_json(c.args))} for c in calls],
                "provider_content": None if pc is None else json.loads(canonical_json(pc)),
            }
        )
        for call, step in zip(calls, turn.tool_steps, strict=True):
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": canonical_json(step.observation())})
    return msgs


def set_turn_pc(turns: tuple[Turn, ...], i: int, provider_content: Any) -> tuple[Turn, ...]:
    """回傳把 turns[i] 的 provider_content 換掉的新 tuple（經 ModelMessage 建構驗證）。"""
    out = list(turns)
    out[i] = R(out[i], message=R(out[i].message, provider_content=provider_content))
    return tuple(out)


def _zero_clock() -> float:
    return 0.0


# ---------------------------------------------------------------------------
# ModelMessage.provider_content：建構
# ---------------------------------------------------------------------------


def test_provider_content_defaults_to_none():
    implicit = ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn")
    explicit = ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn", provider_content=None)
    assert implicit.provider_content is None
    assert explicit.provider_content is None
    assert implicit == explicit
    assert canonical_json(implicit.to_dict()) == canonical_json(explicit.to_dict())


ACCEPTED_CONTENTS = [
    pytest.param(lambda: [], id="empty_list"),
    pytest.param(lambda: [block_dict(text_block("hi"))], id="single_text_block"),
    pytest.param(sample_blocks, id="all_whitelisted_kinds"),
    pytest.param(lambda: [rich_block()], id="all_json_leaf_types"),
    pytest.param(lambda: [{}], id="empty_dict_block"),
    pytest.param(lambda: [{"type": "unknown_kind", "payload": [1, 2]}], id="non_whitelisted_type_not_interpreted"),
    pytest.param(lambda: [block_dict(text_block("a")), block_dict(text_block("a"))], id="duplicate_blocks"),
    pytest.param(lambda: _deep_pc(DEPTH_AT_LIMIT), id="depth_at_limit"),
]


@pytest.mark.parametrize("make_pc", ACCEPTED_CONTENTS)
def test_provider_content_accepts_lists_of_json_dicts(make_pc):
    pc = make_pc()
    expected = canonical_json(pc)
    msg = make_msg(pc)
    assert type(msg.provider_content) is list
    assert len(msg.provider_content) == len(pc)
    assert all(type(block) is dict for block in msg.provider_content)
    assert canonical_json(msg.provider_content) == expected


def test_model_message_does_not_interpret_provider_content():
    """ModelMessage 不解讀 provider_content：與 text / tool_calls 不一致也接受。一致性由產生它的 adapter 負責，
    歷史訊息的一致性檢查在 AnthropicPolicy 呼叫 client 之前（decision 009 第 7 點），不在 ModelMessage。
    """
    tool_use_only_in_pc = make_msg(
        [block_dict(tool_use_block("toolu_x", "get_issue", {"issue_id": 1}))], text="", stop_reason="end_turn"
    )
    assert tool_use_only_in_pc.tool_calls == ()
    call_without_block = make_msg(
        [block_dict(text_block("something else"))],
        text="different text",
        calls=(ToolCall("toolu_y", "get_issue", {"issue_id": 2}),),
        stop_reason="tool_use",
    )
    assert len(call_without_block.tool_calls) == 1
    assert call_without_block.text == "different text"


TYPE_ERROR_CONTENTS = [
    pytest.param(lambda: (block_dict(text_block("x")),), id="tuple_of_blocks"),
    pytest.param(lambda: (), id="empty_tuple"),
    pytest.param(lambda: block_dict(text_block("x")), id="single_dict"),
    pytest.param(lambda: "thinking", id="str"),
    pytest.param(lambda: "", id="empty_str"),
    pytest.param(lambda: b"[]", id="bytes"),
    pytest.param(lambda: 0, id="int"),
    pytest.param(lambda: False, id="bool"),
    pytest.param(lambda: 1.5, id="float"),
    pytest.param(lambda: {block_dict(text_block("x"))["text"]}, id="set"),
    pytest.param(lambda: ["text"], id="element_str"),
    pytest.param(lambda: [None], id="element_none"),
    pytest.param(lambda: [[block_dict(text_block("x"))]], id="element_list"),
    pytest.param(lambda: [("type", "text")], id="element_tuple"),
    pytest.param(lambda: [1], id="element_int"),
    pytest.param(lambda: [True], id="element_bool"),
    pytest.param(lambda: [block_dict(text_block("x")), "x"], id="valid_then_str"),
    pytest.param(lambda: [{"type": "tool_use", "input": (1, 2)}], id="nested_tuple_value"),
    pytest.param(lambda: [{"type": "x", "a": [{"b": (1,)}]}], id="deeply_nested_tuple"),
    pytest.param(lambda: [{"type": "x", "a": {1, 2}}], id="nested_set"),
    pytest.param(lambda: [{"type": "x", "a": b"raw"}], id="nested_bytes"),
    pytest.param(lambda: [{"type": "x", "a": object()}], id="nested_object"),
    pytest.param(lambda: [{"type": "x", "a": 1j}], id="nested_complex"),
    pytest.param(lambda: [{1: "a"}], id="int_key"),
    pytest.param(lambda: [{None: "a"}], id="none_key"),
    pytest.param(lambda: [{"type": "x", "a": {2: "b"}}], id="nested_int_key"),
    pytest.param(lambda: [{("k",): 1}], id="tuple_key"),
]


@pytest.mark.parametrize("make_pc", TYPE_ERROR_CONTENTS)
def test_provider_content_type_errors(make_pc):
    with pytest.raises(TypeError):
        make_msg(make_pc())


VALUE_ERROR_CONTENTS = [
    pytest.param(lambda: [{"type": "x", "v": NAN}], id="nan"),
    pytest.param(lambda: [{"type": "x", "v": INF}], id="inf"),
    pytest.param(lambda: [{"type": "x", "v": -INF}], id="neg_inf"),
    pytest.param(lambda: [{"type": "x", "v": [1, {"w": NAN}]}], id="nested_nan"),
    pytest.param(lambda: [{"type": "text", "text": "bad \ud800"}], id="surrogate_value"),
    pytest.param(lambda: [{"type": "x", "\udc00": 1}], id="surrogate_key"),
    pytest.param(lambda: [{"type": "x", "v": ["ok", ["\udfff"]]}], id="nested_surrogate"),
    pytest.param(lambda: [{"type": "x", "v": {"\ud83d": "half emoji"}}], id="nested_surrogate_key"),
    pytest.param(lambda: _deep_pc(DEPTH_PAST_LIMIT), id="depth_past_limit"),
    pytest.param(lambda: _deep_pc(DEPTH_PAST_LIMIT + 5), id="depth_well_past_limit"),
    pytest.param(lambda: _deep_pc(5000), id="depth_5000"),
    pytest.param(lambda: [{"type": "x", "v": {"a": _nested(DEPTH_PAST_LIMIT)}}], id="depth_past_limit_via_dict"),
]


@pytest.mark.parametrize("make_pc", VALUE_ERROR_CONTENTS)
def test_provider_content_value_errors(make_pc):
    with pytest.raises(ValueError):
        make_msg(make_pc())


def test_provider_content_is_stored_with_recursively_sorted_keys():
    pc = [
        {"type": "tool_use", "name": "search_issues", "input": {"z": 1, "a": {"y": 2, "b": [{"q": 1, "c": 2}]}}, "id": "t"},
        rich_block(),
        block_dict(thinking_block("sig", thinking="t")),
    ]
    msg = make_msg(pc)
    _assert_keys_sorted_recursively(msg.provider_content)
    assert _ordered_json(msg.provider_content) == canonical_json(pc)


def test_provider_content_preserves_list_order_and_number_types():
    pc = [
        block_dict(text_block("second-looking")),
        {"type": "x", "i": 1, "f": 1.0, "b": True, "zero": 0, "fzero": 0.0, "neg": -0.5, "seq": [3, 1, 2]},
        block_dict(text_block("a-first-looking")),
    ]
    msg = make_msg(pc)
    stored = msg.provider_content
    assert [block.get("text") for block in stored] == ["second-looking", None, "a-first-looking"]
    numbers = stored[1]
    assert type(numbers["i"]) is int and numbers["i"] == 1
    assert type(numbers["f"]) is float and numbers["f"] == 1.0
    assert numbers["b"] is True
    assert type(numbers["zero"]) is int
    assert type(numbers["fzero"]) is float
    assert numbers["seq"] == [3, 1, 2]
    assert canonical_json(stored) == canonical_json(pc)


def test_provider_content_does_not_alias_constructor_input():
    pc = sample_blocks()
    before = canonical_json(pc)
    msg = make_msg(pc)
    assert msg.provider_content is not pc
    assert all(stored is not original for stored, original in zip(msg.provider_content, pc))
    # 就地破壞輸入（巢狀 dict、list 本身）
    pc[0]["signature"] = "forged"
    pc[3]["input"]["issue_id"] = 999
    pc[3]["input"]["extra"] = [1]
    pc.append(block_dict(text_block("late")))
    del pc[1]
    assert canonical_json(msg.provider_content) == before


def test_model_messages_differing_only_in_provider_content_are_distinct():
    base = make_msg(sample_blocks())
    other_signature = sample_blocks()
    other_signature[0]["signature"] = "sig-β"
    variants = [
        make_msg(None),
        make_msg([]),
        make_msg(other_signature),
        make_msg(list(reversed(sample_blocks()))),
        make_msg(sample_blocks() + [block_dict(text_block("more"))]),
    ]
    rendered = {canonical_json(base.to_dict())}
    for variant in variants:
        assert variant != base
        rendered.add(canonical_json(variant.to_dict()))
    assert len(rendered) == len(variants) + 1
    assert make_msg(None) != make_msg([])


def test_provider_content_canonical_distinguishes_what_eq_confuses():
    floaty = make_msg([{"type": "x", "n": 1.0}])
    inty = make_msg([{"type": "x", "n": 1}])
    assert floaty == inty  # Python == 把 1 與 1.0 視為相同
    assert canonical_json(floaty.to_dict()) != canonical_json(inty.to_dict())
    back = ModelMessage.from_dict(json.loads(canonical_json(floaty.to_dict())))
    assert canonical_json(back.to_dict()) == canonical_json(floaty.to_dict())
    assert type(back.provider_content[0]["n"]) is float


# ---------------------------------------------------------------------------
# ModelMessage.to_dict / from_dict
# ---------------------------------------------------------------------------


def test_model_message_to_dict_has_exactly_four_keys():
    none_dict = make_msg(None, stop_reason=None).to_dict()
    assert set(none_dict) == MESSAGE_KEYS
    assert none_dict["provider_content"] is None
    assert canonical_json(none_dict) == '{"provider_content":null,"stop_reason":null,"text":"","tool_calls":[]}'

    calls = (ToolCall("toolu_01", "assign_issue", {"issue_id": 8, "assignee_id": "u4"}),)
    pc = sample_blocks()
    d = make_msg(pc, text="I'll assign it. 好", calls=calls, stop_reason="tool_use").to_dict()
    assert set(d) == MESSAGE_KEYS
    assert type(d["provider_content"]) is list
    _assert_json_native(d)
    assert canonical_json(d) == canonical_json(
        {
            "text": "I'll assign it. 好",
            "tool_calls": [c.to_dict() for c in calls],
            "stop_reason": "tool_use",
            "provider_content": pc,
        }
    )
    _assert_keys_sorted_recursively(d["provider_content"])


def test_model_message_to_dict_returns_deep_copy_of_provider_content():
    msg = make_msg(sample_blocks())
    before_msg = canonical_json(msg.to_dict())
    before_pc = canonical_json(msg.provider_content)
    d = msg.to_dict()
    assert d["provider_content"] is not msg.provider_content
    d["provider_content"][0]["signature"] = "forged"
    d["provider_content"][3]["input"]["issue_id"] = 999
    d["provider_content"].append({"type": "text", "text": "injected"})
    assert canonical_json(msg.provider_content) == before_pc
    assert canonical_json(msg.to_dict()) == before_msg
    again = msg.to_dict()
    assert again["provider_content"] is not d["provider_content"]


ROUNDTRIP_MESSAGES = [
    pytest.param(lambda: make_msg(None), id="none"),
    pytest.param(lambda: make_msg([]), id="empty_list"),
    pytest.param(lambda: make_msg(final_turn_content(), text="Done."), id="final_with_thinking"),
    pytest.param(
        lambda: make_msg(
            tool_turn_content(0, "assign_issue", {"issue_id": 8, "assignee_id": "u4"}, "toolu_00", "note 0 登入"),
            text="note 0 登入",
            calls=(ToolCall("toolu_00", "assign_issue", {"issue_id": 8, "assignee_id": "u4"}),),
            stop_reason="tool_use",
        ),
        id="tool_turn",
    ),
    pytest.param(lambda: make_msg([rich_block()], stop_reason=None), id="rich_json"),
    pytest.param(lambda: make_msg([{"type": "x", "n": 1.0, "m": 1, "t": True}]), id="number_types"),
    pytest.param(lambda: make_msg(_deep_pc(DEPTH_AT_LIMIT)), id="depth_at_limit"),
    pytest.param(lambda: make_msg(sample_blocks(), stop_reason="refusal"), id="refusal"),
]


@pytest.mark.parametrize("make_message", ROUNDTRIP_MESSAGES)
def test_model_message_roundtrip_with_provider_content(make_message):
    msg = make_message()
    direct = ModelMessage.from_dict(msg.to_dict())
    via_json = ModelMessage.from_dict(json.loads(canonical_json(msg.to_dict())))
    via_plain_dumps = ModelMessage.from_dict(json.loads(json.dumps(msg.to_dict(), ensure_ascii=False)))
    for back in (direct, via_json, via_plain_dumps):
        assert back == msg
        assert canonical_json(back.to_dict()) == canonical_json(msg.to_dict())
        assert type(back.provider_content) is type(msg.provider_content)
        assert type(back.tool_calls) is tuple


def test_model_message_from_dict_normalizes_and_does_not_alias_input():
    data = {
        "text": "",
        "tool_calls": [],
        "stop_reason": "end_turn",
        "provider_content": [{"type": "text", "text": "hi", "citations": None}, {"z": {"y": 1, "b": 2}, "a": [1]}],
    }
    expected = canonical_json(data["provider_content"])
    back = ModelMessage.from_dict(data)
    _assert_keys_sorted_recursively(back.provider_content)
    assert _ordered_json(back.provider_content) == expected
    assert back.provider_content is not data["provider_content"]
    data["provider_content"][0]["text"] = "forged"
    data["provider_content"][1]["z"]["y"] = 99
    data["provider_content"].append({})
    assert canonical_json(back.provider_content) == expected


def _v2_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "text": "",
        "tool_calls": [],
        "stop_reason": "end_turn",
        "provider_content": [{"type": "text", "text": "hi"}],
    }
    data.update(overrides)
    return data


def _v2_without(key: str) -> dict[str, Any]:
    data = _v2_dict()
    del data[key]
    return data


FROM_DICT_BAD = [
    pytest.param(lambda: _v2_without("provider_content"), id="missing_provider_content_v1_shape"),
    pytest.param(lambda: _v2_without("text"), id="missing_text"),
    pytest.param(lambda: _v2_without("tool_calls"), id="missing_tool_calls"),
    pytest.param(lambda: _v2_without("stop_reason"), id="missing_stop_reason"),
    pytest.param(lambda: _v2_dict(role="assistant"), id="extra_key"),
    pytest.param(lambda: _v2_dict(providerContent=None), id="extra_misspelled_key"),
    pytest.param(lambda: _v2_dict(provider_content=({"type": "text", "text": "hi"},)), id="pc_tuple"),
    pytest.param(lambda: _v2_dict(provider_content=()), id="pc_empty_tuple"),
    pytest.param(lambda: _v2_dict(provider_content={"type": "text", "text": "hi"}), id="pc_dict"),
    pytest.param(lambda: _v2_dict(provider_content="hi"), id="pc_str"),
    pytest.param(lambda: _v2_dict(provider_content=0), id="pc_int"),
    pytest.param(lambda: _v2_dict(provider_content=False), id="pc_bool"),
    pytest.param(lambda: _v2_dict(provider_content=["text"]), id="pc_element_str"),
    pytest.param(lambda: _v2_dict(provider_content=[None]), id="pc_element_none"),
    pytest.param(lambda: _v2_dict(provider_content=[[{}]]), id="pc_element_list"),
    pytest.param(lambda: _v2_dict(provider_content=[{"type": "x", "v": (1,)}]), id="pc_nested_tuple"),
    pytest.param(lambda: _v2_dict(provider_content=[{"type": "x", "v": {1}}]), id="pc_nested_set"),
    pytest.param(lambda: _v2_dict(provider_content=[{1: "a"}]), id="pc_int_key"),
    pytest.param(lambda: _v2_dict(provider_content=[{"type": "x", "v": NAN}]), id="pc_nan"),
    pytest.param(lambda: _v2_dict(provider_content=[{"type": "x", "v": -INF}]), id="pc_neg_inf"),
    pytest.param(lambda: _v2_dict(provider_content=[{"type": "text", "text": "\ud800"}]), id="pc_surrogate"),
    pytest.param(lambda: _v2_dict(provider_content=[{"\udc00": 1}]), id="pc_surrogate_key"),
    pytest.param(lambda: _v2_dict(provider_content=_deep_pc(DEPTH_PAST_LIMIT)), id="pc_depth_past_limit"),
    pytest.param(lambda: _v2_dict(provider_content=_deep_pc(5000)), id="pc_depth_5000"),
]


@pytest.mark.parametrize("make_data", FROM_DICT_BAD)
def test_model_message_from_dict_rejects_invalid_v2_dicts_with_value_error(make_data):
    with pytest.raises(ValueError):
        ModelMessage.from_dict(make_data())


def test_model_message_from_dict_accepts_null_provider_content():
    back = ModelMessage.from_dict(_v2_dict(provider_content=None))
    assert back.provider_content is None
    assert back == ModelMessage(text="", tool_calls=(), stop_reason="end_turn")


# ---------------------------------------------------------------------------
# PolicyStep.meta：None 或 dict；內容規則同 model_config；正規化深拷貝（agents/policy.py PolicyStep docstring）
# ---------------------------------------------------------------------------

META_SENTINEL = "META-SENTINEL 量測不進 conversation"


def sample_meta(k: int = 0) -> dict[str, Any]:
    """provider 量測風格的 meta（項目取自 decision 009 第 4 點；確切形狀由本檔選定，PolicyStep / Turn 不解讀）。

    key 刻意以反序建立；含 None、int、float、非 ASCII 與一個不該出現在 conversation 的 sentinel 字串。
    """
    return {
        "usage": {
            "output_tokens": 20,
            "input_tokens": 100 + k,
            "cache_read_input_tokens": None,
            "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 0},
        },
        "tool_use_blocks": 1,
        "stop_reason": "tool_use",
        "stop_details": None,
        "sentinel": META_SENTINEL,
        "response_model": "claude-opus-5-5",
        "request_id": f"req_{k:02d} ✓",
        "elapsed": 1.0,
    }


def end_message() -> ModelMessage:
    return ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn")


def make_step(meta: Any) -> PolicyStep:
    return PolicyStep(message=end_message(), input_tokens=0, output_tokens=0, cost_usd=0.0, meta=meta)


def make_config(model_config: Any) -> AgentConfig:
    return AgentConfig(
        policy="p", prompt_version="v", system_prompt="", developer_prompt="", model_config=model_config
    )


def test_policy_step_meta_defaults_to_none():
    implicit = PolicyStep(message=end_message())
    explicit = PolicyStep(message=end_message(), meta=None)
    assert implicit.meta is None
    assert explicit.meta is None
    assert implicit == explicit


ACCEPTED_METAS = [
    pytest.param(lambda: {}, id="empty_dict"),
    pytest.param(sample_meta, id="provider_measurements"),
    pytest.param(rich_block, id="all_json_leaf_types"),
    pytest.param(lambda: {"v": _nested(8), "w": {"x": {"y": [{"z": None}]}}}, id="moderately_nested"),
    pytest.param(lambda: {"登入": "✓", "emoji": "😀", "": ""}, id="non_ascii_and_empty_strings"),
    pytest.param(lambda: {"big": 10**30, "neg": -1, "f": -0.25}, id="numbers"),
]


@pytest.mark.parametrize("make_meta", ACCEPTED_METAS)
def test_policy_step_meta_accepts_json_objects(make_meta):
    meta = make_meta()
    expected = canonical_json(meta)
    step = make_step(meta)
    assert type(step.meta) is dict
    _assert_json_native(step.meta)
    assert canonical_json(step.meta) == expected


META_TYPE_ERRORS = [
    pytest.param(lambda: [], id="empty_list"),
    pytest.param(lambda: [sample_meta()], id="list_of_dicts"),
    pytest.param(lambda: (("k", 1),), id="tuple"),
    pytest.param(lambda: (), id="empty_tuple"),
    pytest.param(lambda: "meta", id="str"),
    pytest.param(lambda: "", id="empty_str"),
    pytest.param(lambda: b"{}", id="bytes"),
    pytest.param(lambda: 0, id="int"),
    pytest.param(lambda: False, id="bool"),
    pytest.param(lambda: 1.5, id="float"),
    pytest.param(lambda: {"a", "b"}, id="set"),
    pytest.param(lambda: object(), id="object"),
    pytest.param(lambda: {"usage": (1, 2)}, id="nested_tuple"),
    pytest.param(lambda: {"a": [{"b": (1,)}]}, id="deeply_nested_tuple"),
    pytest.param(lambda: {"a": {1, 2}}, id="nested_set"),
    pytest.param(lambda: {"a": b"raw"}, id="nested_bytes"),
    pytest.param(lambda: {"a": object()}, id="nested_object"),
    pytest.param(lambda: {"a": 1j}, id="nested_complex"),
    pytest.param(lambda: {1: "a"}, id="int_key"),
    pytest.param(lambda: {None: "a"}, id="none_key"),
    pytest.param(lambda: {True: "a"}, id="bool_key"),
    pytest.param(lambda: {("k",): 1}, id="tuple_key"),
    pytest.param(lambda: {"a": {2: "b"}}, id="nested_int_key"),
    pytest.param(lambda: {"a": [{"ok": 1}, {3: "c"}]}, id="int_key_inside_list"),
]


@pytest.mark.parametrize("make_meta", META_TYPE_ERRORS)
def test_policy_step_meta_type_errors(make_meta):
    with pytest.raises(TypeError):
        make_step(make_meta())


META_VALUE_ERRORS = [
    pytest.param(lambda: {"v": NAN}, id="nan"),
    pytest.param(lambda: {"v": INF}, id="inf"),
    pytest.param(lambda: {"v": -INF}, id="neg_inf"),
    pytest.param(lambda: {"usage": {"input_tokens": [1, {"w": NAN}]}}, id="nested_nan"),
    pytest.param(lambda: {"request_id": "bad \ud800"}, id="surrogate_value"),
    pytest.param(lambda: {"\udc00": 1}, id="surrogate_key"),
    pytest.param(lambda: {"v": ["ok", ["\udfff"]]}, id="nested_surrogate"),
    pytest.param(lambda: {"v": {"\ud83d": "half emoji"}}, id="nested_surrogate_key"),
    pytest.param(lambda: {"v": _nested(MAX_MODEL_CONFIG_DEPTH + 5)}, id="depth_well_past_limit"),
    pytest.param(lambda: {"v": _nested(5000)}, id="depth_5000"),
]


@pytest.mark.parametrize("make_meta", META_VALUE_ERRORS)
def test_policy_step_meta_value_errors(make_meta):
    with pytest.raises(ValueError):
        make_step(make_meta())


def _model_config_accepts(value: Any) -> bool:
    try:
        make_config(copy.deepcopy(value))
    except ValueError:
        return False
    return True


META_DEPTHS = range(MAX_MODEL_CONFIG_DEPTH - 3, MAX_MODEL_CONFIG_DEPTH + 3)


@pytest.mark.parametrize("n", META_DEPTHS)
def test_policy_step_meta_depth_limit_matches_model_config(n):
    """「內容規則同 model_config」：深度上限的接受 / 拒絕與 AgentConfig.model_config 逐一相同（meta 根 = model_config 根）。"""
    value = {"v": _nested(n)}
    if _model_config_accepts(value):
        step = make_step(copy.deepcopy(value))
        assert canonical_json(step.meta) == canonical_json(value)
    else:
        with pytest.raises(ValueError):
            make_step(copy.deepcopy(value))


def test_meta_depth_range_straddles_the_model_config_limit():
    """上一個參數化測試的範圍必須同時包含 model_config 接受與拒絕的深度，否則它沒有測到邊界。"""
    outcomes = {_model_config_accepts({"v": _nested(n)}) for n in META_DEPTHS}
    assert outcomes == {True, False}


def test_policy_step_meta_is_stored_with_recursively_sorted_keys():
    meta = sample_meta()
    meta["z"] = {"y": 2, "b": [{"q": 1, "c": {"s": 0, "a": 1}}]}
    step = make_step(meta)
    _assert_keys_sorted_recursively(step.meta)
    assert _ordered_json(step.meta) == canonical_json(meta)


def test_policy_step_meta_preserves_list_order_and_number_types():
    meta = {"seq": [3, 1, 2], "i": 1, "f": 1.0, "b": True, "zero": 0, "fzero": 0.0, "none": None, "big": 10**30}
    stored = make_step(meta).meta
    assert stored["seq"] == [3, 1, 2]
    assert type(stored["i"]) is int and stored["i"] == 1
    assert type(stored["f"]) is float and stored["f"] == 1.0
    assert stored["b"] is True
    assert type(stored["zero"]) is int
    assert type(stored["fzero"]) is float
    assert stored["none"] is None
    assert type(stored["big"]) is int and stored["big"] == 10**30
    assert canonical_json(stored) == canonical_json(meta)


def test_policy_step_meta_canonical_distinguishes_what_eq_confuses():
    floaty = make_step({"n": 1.0})
    inty = make_step({"n": 1})
    assert floaty == inty  # Python == 把 1 與 1.0 視為相同
    assert canonical_json(floaty.meta) != canonical_json(inty.meta)


def test_policy_step_meta_does_not_alias_constructor_input():
    meta = sample_meta()
    before = canonical_json(meta)
    step = make_step(meta)
    assert step.meta is not meta
    assert step.meta["usage"] is not meta["usage"]
    assert step.meta["usage"]["cache_creation"] is not meta["usage"]["cache_creation"]
    meta["usage"]["input_tokens"] = 999
    meta["usage"]["cache_creation"]["ephemeral_1h_input_tokens"] = 7
    meta["extra"] = [1]
    del meta["request_id"]
    assert canonical_json(step.meta) == before

    empty: dict[str, Any] = {}
    empty_step = make_step(empty)
    assert empty_step.meta == {} and empty_step.meta is not empty
    empty["late"] = 1
    assert empty_step.meta == {}


def test_policy_step_meta_is_not_part_of_the_message():
    step = make_step(sample_meta())
    assert set(step.message.to_dict()) == MESSAGE_KEYS
    assert META_SENTINEL not in canonical_json(step.message.to_dict())


# ---------------------------------------------------------------------------
# render_messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", PROVIDER_MODES)
def test_render_messages_assistant_keys_in_order_with_provider_content(mode):
    traj = build_trajectory(REASSIGN, mode)
    messages = render_messages(traj.task["instruction"], traj.agent, traj.turns)
    assert [list(m) for m in messages] == [ROLE_KEYS[m["role"]] for m in messages]
    assert sum(1 for m in messages if m["role"] == "assistant") == len(traj.turns)


@pytest.mark.parametrize("mode", PROVIDER_MODES)
def test_render_messages_matches_expected_conversation_byte_for_byte(mode):
    traj = build_trajectory(REASSIGN, mode)
    contents = planned_contents(REASSIGN, mode)
    actual = render_messages(traj.task["instruction"], traj.agent, traj.turns)
    expected = expected_messages(traj.task["instruction"], traj.agent, traj.turns, contents)
    _assert_json_native(actual)
    # 對順序敏感：provider_content 內的 key 必須已遞迴排序、assistant 的 key 依規定順序
    assert json.dumps(actual, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
    for message, pc in zip((m for m in actual if m["role"] == "assistant"), contents, strict=True):
        if pc is None:
            assert message["provider_content"] is None
        else:
            assert type(message["provider_content"]) is list
            _assert_keys_sorted_recursively(message["provider_content"])


def test_render_messages_prefix_conversations_carry_provider_content():
    traj = build_trajectory(REASSIGN, "all")
    contents = planned_contents(REASSIGN, "all")
    for k in range(len(traj.turns) + 1):
        actual = render_messages(traj.task["instruction"], traj.agent, traj.turns[:k])
        expected = expected_messages(traj.task["instruction"], traj.agent, traj.turns[:k], contents[:k])
        assert json.dumps(actual, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)


def test_render_messages_returns_deep_copies_of_provider_content():
    traj = build_trajectory(ASSIGN, "all")
    before_turns = canonical_json([t.to_dict() for t in traj.turns])
    first = render_messages(traj.task["instruction"], traj.agent, traj.turns)
    assistants = [m for m in first if m["role"] == "assistant"]
    for message, turn in zip(assistants, traj.turns, strict=True):
        assert message["provider_content"] is not turn.message.provider_content
        message["provider_content"][0]["signature"] = "forged"
        message["provider_content"].append({"type": "text", "text": "injected"})
    assert canonical_json([t.to_dict() for t in traj.turns]) == before_turns
    second = render_messages(traj.task["instruction"], traj.agent, traj.turns)
    expected = expected_messages(traj.task["instruction"], traj.agent, traj.turns, planned_contents(ASSIGN, "all"))
    assert json.dumps(second, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)


def test_scripted_policy_turns_render_provider_content_none():
    task = get_smoke_task(ASSIGN)
    policy = oracle_policy(REFERENCE_SOLUTIONS[ASSIGN])
    step = policy.next_message(render_messages(task.instruction, policy.agent_config, ()), IssueTrackerEnv.tool_specs())
    assert step.message.provider_content is None
    traj = run_episode(task, policy, max_turns=10, clock=_zero_clock)
    assert all(turn.message.provider_content is None for turn in traj.turns)
    for turn_dict in traj.to_dict()["turns"]:
        assert set(turn_dict["message"]) == MESSAGE_KEYS
        assert turn_dict["message"]["provider_content"] is None
    assistants = [m for m in traj.messages() if m["role"] == "assistant"]
    assert len(assistants) == len(traj.turns)
    for message in assistants:
        assert list(message) == ROLE_KEYS["assistant"]
        assert message["provider_content"] is None


# ---------------------------------------------------------------------------
# 格式版本
# ---------------------------------------------------------------------------


def test_formats_are_v2():
    assert TRAJECTORY_FORMAT == "agent-gym/trajectory/v2"
    assert CHECKPOINT_FORMAT == "agent-gym/checkpoint/v2"


def test_documents_declare_v2_and_reject_v1_format():
    traj = build_trajectory(ASSIGN, "all")
    d = traj.to_dict()
    assert d["format"] == TRAJECTORY_FORMAT
    d["format"] = "agent-gym/trajectory/v1"
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(d)

    cp = make_checkpoint(traj, 1)
    c = cp.to_dict()
    assert c["format"] == CHECKPOINT_FORMAT
    c["format"] = "agent-gym/checkpoint/v1"
    c["checkpoint_id"] = sha256_of({key: value for key, value in c.items() if key != "checkpoint_id"})
    with pytest.raises(CheckpointError):
        Checkpoint.from_dict(c)


# ---------------------------------------------------------------------------
# Trajectory 與 provider_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", PROVIDER_MODES)
@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_provider_trajectory_validates_and_verifies(task_id, mode):
    traj = build_trajectory(task_id, mode)
    traj.validate()
    traj.verify()


@pytest.mark.parametrize("mode", PROVIDER_MODES)
def test_trajectory_to_dict_carries_provider_content(mode):
    traj = build_trajectory(REASSIGN, mode)
    contents = planned_contents(REASSIGN, mode)
    d = traj.to_dict()
    _assert_json_native(d)
    assert d["format"] == TRAJECTORY_FORMAT
    assert len(d["turns"]) == len(contents)
    for turn_dict, pc in zip(d["turns"], contents, strict=True):
        message = turn_dict["message"]
        assert set(message) == MESSAGE_KEYS
        if pc is None:
            assert message["provider_content"] is None
        else:
            assert canonical_json(message["provider_content"]) == canonical_json(pc)
            _assert_keys_sorted_recursively(message["provider_content"])


@pytest.mark.parametrize("mode", PROVIDER_MODES)
@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_provider_trajectory_round_trips_byte_identically(task_id, mode, tmp_path: Path):
    traj = build_trajectory(task_id, mode)
    canon = canonical_json(traj.to_dict())

    for back in (
        Trajectory.from_dict(traj.to_dict()),
        Trajectory.from_dict(json.loads(canon)),
    ):
        assert back == traj
        assert canonical_json(back.to_dict()) == canon
        assert back.trajectory_id == traj.trajectory_id

    first = tmp_path / "traj.json"
    traj.save(first)
    assert first.read_bytes() == (canon + "\n").encode("utf-8")
    loaded = Trajectory.load(first)
    assert loaded == traj
    assert canonical_json(loaded.to_dict()) == canon
    assert loaded.trajectory_id == traj.trajectory_id
    assert json.dumps(loaded.messages(), ensure_ascii=False) == json.dumps(traj.messages(), ensure_ascii=False)
    for original, back in zip(traj.turns, loaded.turns, strict=True):
        assert canonical_json(back.message.provider_content) == canonical_json(original.message.provider_content)

    second = tmp_path / "traj-again.json"
    loaded.save(second)
    assert second.read_bytes() == first.read_bytes()


def test_trajectory_messages_include_provider_content():
    traj = build_trajectory(ASSIGN, "mixed")
    contents = planned_contents(ASSIGN, "mixed")
    expected = expected_messages(traj.task["instruction"], traj.agent, traj.turns, contents)
    assert json.dumps(traj.messages(), ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)


def test_trajectory_to_dict_does_not_alias_provider_content():
    traj = build_trajectory(ASSIGN, "all")
    before = canonical_json(traj.to_dict())
    d = traj.to_dict()
    d["turns"][0]["message"]["provider_content"][0]["signature"] = "forged"
    d["turns"][-1]["message"]["provider_content"].clear()
    assert canonical_json(traj.to_dict()) == before

    data = traj.to_dict()
    back = Trajectory.from_dict(data)
    data["turns"][0]["message"]["provider_content"][0]["signature"] = "forged"
    data["turns"][0]["message"]["provider_content"].append({})
    assert canonical_json(back.to_dict()) == before


def _pc_signature_changed(pc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(pc)
    out[0]["signature"] = out[0]["signature"] + "-changed"
    return out


def _pc_float_input(pc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(pc)
    out[-1]["input"]["issue_id"] = float(out[-1]["input"]["issue_id"])  # Python == 仍相等
    return out


PC_MUTATIONS: list[Any] = [
    pytest.param(0, _pc_signature_changed, id="turn0_signature"),
    pytest.param(0, lambda pc: None, id="turn0_none"),
    pytest.param(0, lambda pc: [], id="turn0_empty"),
    pytest.param(0, lambda pc: list(reversed(pc)), id="turn0_block_order"),
    pytest.param(0, lambda pc: pc[1:], id="turn0_thinking_dropped"),
    pytest.param(0, lambda pc: pc + [block_dict(redacted_thinking_block("eHg="))], id="turn0_block_added"),
    pytest.param(0, _pc_float_input, id="turn0_int_to_float"),
    pytest.param(-1, _pc_signature_changed, id="final_signature"),
    pytest.param(-1, lambda pc: None, id="final_none"),
]


@pytest.mark.parametrize(("index", "mutate"), PC_MUTATIONS)
def test_trajectory_id_changes_when_provider_content_changes(index, mutate, tmp_path: Path):
    traj = build_trajectory(ASSIGN, "all")
    contents = planned_contents(ASSIGN, "all")
    i = index % len(traj.turns)
    changed = R(traj, turns=set_turn_pc(traj.turns, i, mutate(copy.deepcopy(contents[i]))))
    # tool steps 與 env 沒變：provider_content 不被 trajectory 解讀，變體仍是合法 trajectory
    changed.validate()
    changed.verify()
    assert changed.trajectory_id != traj.trajectory_id
    assert canonical_json(changed.to_dict()) != canonical_json(traj.to_dict())
    path = tmp_path / "changed.json"
    changed.save(path)
    assert Trajectory.load(path).trajectory_id == changed.trajectory_id


def test_trajectory_id_differs_between_provider_and_scripted_shaped_turns():
    with_pc = build_trajectory(ASSIGN, "all")
    without_pc = build_trajectory(ASSIGN, "none")
    assert with_pc.records() == without_pc.records()
    assert with_pc.trajectory_id != without_pc.trajectory_id


def test_trajectory_from_dict_requires_provider_content_key():
    traj = build_trajectory(ASSIGN, "all")
    d = traj.to_dict()
    del d["turns"][0]["message"]["provider_content"]
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(d)
    with pytest.raises(TrajectoryError):
        Turn.from_dict(d["turns"][0])

    scripted = build_trajectory(ASSIGN, "none").to_dict()
    for turn_dict in scripted["turns"]:
        del turn_dict["message"]["provider_content"]
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(scripted)


BAD_SERIALIZED_PC = [
    pytest.param("hi", id="str"),
    pytest.param({"type": "text", "text": "hi"}, id="dict"),
    pytest.param(1, id="int"),
    pytest.param(["text"], id="element_str"),
    pytest.param([None], id="element_none"),
    pytest.param([[{"type": "text", "text": "hi"}]], id="element_list"),
    pytest.param([{"type": "x", "v": NAN}], id="nan"),
    pytest.param([{"type": "x", "v": INF}], id="inf"),
    pytest.param([{"type": "text", "text": "\ud800"}], id="surrogate"),
    pytest.param([{"\udc00": 1}], id="surrogate_key"),
    pytest.param(_deep_pc(DEPTH_PAST_LIMIT), id="depth_past_model_limit"),
]


@pytest.mark.parametrize("bad", BAD_SERIALIZED_PC)
def test_trajectory_and_turn_from_dict_reject_invalid_provider_content(bad):
    traj = build_trajectory(ASSIGN, "all")
    d = traj.to_dict()
    d["turns"][0]["message"]["provider_content"] = copy.deepcopy(bad)
    with pytest.raises(TrajectoryError):
        Turn.from_dict(copy.deepcopy(d["turns"][0]))
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(d)


def test_trajectory_load_rejects_nan_literal_in_provider_content(tmp_path: Path):
    traj = build_trajectory(ASSIGN, "all")
    d = traj.to_dict()
    d["turns"][0]["message"]["provider_content"].append({"type": "x", "v": "__NAN_SENTINEL__"})
    text = canonical_json(d).replace('"__NAN_SENTINEL__"', "NaN")
    assert "NaN" in text
    path = tmp_path / "nan.json"
    path.write_bytes(text.encode("utf-8"))
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


def test_trajectory_with_provider_content_at_depth_limit_round_trips(tmp_path: Path):
    traj = build_trajectory(ASSIGN, "all")
    deep = traj.turns[0].message.provider_content + _deep_pc(DEPTH_AT_LIMIT)
    traj = R(traj, turns=set_turn_pc(traj.turns, 0, deep))
    traj.verify()
    path = tmp_path / "deep.json"
    traj.save(path)
    loaded = Trajectory.load(path)
    assert canonical_json(loaded.to_dict()) == canonical_json(traj.to_dict())
    assert canonical_json(loaded.to_dict()["turns"][0]["message"]["provider_content"]) == canonical_json(deep)
    assert loaded.trajectory_id == traj.trajectory_id
    cp = make_checkpoint(loaded, 1)
    cp_path = tmp_path / "deep-cp.json"
    cp.save(cp_path)
    loaded_cp = Checkpoint.load(cp_path)
    assert canonical_json(loaded_cp.to_dict()) == canonical_json(cp.to_dict())
    assert canonical_json(loaded_cp.to_dict()["turns"][0]["message"]["provider_content"]) == canonical_json(deep)


# ---------------------------------------------------------------------------
# Turn.meta：to_dict 5 個 key、from_dict 必須有 meta、None 或 dict、Trajectory 規則 9 / 10
# ---------------------------------------------------------------------------


def with_metas(traj: Trajectory, metas: list[Any]) -> Trajectory:
    """回傳每個 turn 的 meta 依序換成 metas 的新 trajectory。"""
    assert len(metas) == len(traj.turns)
    return R(traj, turns=tuple(R(turn, meta=meta) for turn, meta in zip(traj.turns, metas, strict=True)))


def set_turn_meta(traj: Trajectory, i: int, meta: Any) -> Trajectory:
    turns = list(traj.turns)
    turns[i] = R(turns[i], meta=meta)
    return R(traj, turns=tuple(turns))


def metas_for(traj: Trajectory) -> list[dict[str, Any]]:
    return [sample_meta(k) for k in range(len(traj.turns))]


def _without(d: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != key}


def test_turn_keys_include_meta():
    assert TURN_KEYS == ("index", "message", "tool_steps", "usage", "meta")


def test_turn_meta_defaults_to_none_and_is_serialized_as_null():
    turn = build_trajectory(ASSIGN, "none").turns[0]
    assert turn.meta is None
    d = turn.to_dict()
    assert set(d) == set(TURN_KEYS)
    assert d["meta"] is None


TURN_METAS = [
    pytest.param(lambda: None, id="none"),
    pytest.param(lambda: {}, id="empty_dict"),
    pytest.param(sample_meta, id="provider_measurements"),
    pytest.param(rich_block, id="all_json_leaf_types"),
    pytest.param(lambda: {"n": 1.0, "m": 1, "t": True, "big": 10**30}, id="number_types"),
]


@pytest.mark.parametrize("make_meta", TURN_METAS)
def test_turn_to_dict_has_exactly_five_keys_and_carries_meta(make_meta):
    base = build_trajectory(ASSIGN, "none").turns[0]
    meta = make_meta()
    turn = R(base, meta=meta)
    d = turn.to_dict()
    assert set(d) == set(TURN_KEYS)
    _assert_json_native(d)
    if meta is None:
        assert d["meta"] is None
    else:
        assert type(d["meta"]) is dict
        assert canonical_json(d["meta"]) == canonical_json(meta)
    # meta 以外的 key 不受 meta 影響
    assert canonical_json(_without(d, "meta")) == canonical_json(_without(base.to_dict(), "meta"))


def test_turn_to_dict_returns_deep_copy_of_meta():
    turn = R(build_trajectory(ASSIGN, "none").turns[0], meta=sample_meta())
    before = canonical_json(turn.to_dict())
    d = turn.to_dict()
    assert d["meta"] is not turn.meta
    d["meta"]["usage"]["input_tokens"] = 999
    d["meta"]["usage"]["cache_creation"]["ephemeral_5m_input_tokens"] = 7
    d["meta"]["extra"] = [1]
    assert canonical_json(turn.to_dict()) == before
    again = turn.to_dict()
    assert again["meta"] is not d["meta"]


@pytest.mark.parametrize("make_meta", TURN_METAS)
def test_turn_round_trips_meta(make_meta):
    turn = R(build_trajectory(ASSIGN, "none").turns[0], meta=make_meta())
    canon = canonical_json(turn.to_dict())
    for data in (turn.to_dict(), json.loads(canon), json.loads(json.dumps(turn.to_dict(), ensure_ascii=False))):
        back = Turn.from_dict(data)
        assert back == turn
        assert canonical_json(back.to_dict()) == canon
        assert (back.meta is None) == (turn.meta is None)
        if turn.meta is not None:
            assert type(back.meta) is dict


def test_turn_from_dict_preserves_meta_number_types():
    turn = R(build_trajectory(ASSIGN, "none").turns[0], meta={"f": 1.0, "i": 1})
    back = Turn.from_dict(json.loads(canonical_json(turn.to_dict())))
    assert type(back.meta["f"]) is float
    assert type(back.meta["i"]) is int


def test_turn_from_dict_requires_exactly_the_five_keys():
    turn = R(build_trajectory(ASSIGN, "none").turns[0], meta=sample_meta())
    missing = turn.to_dict()
    del missing["meta"]  # v1 形狀的 turn
    with pytest.raises(TrajectoryError):
        Turn.from_dict(missing)
    extra = turn.to_dict()
    extra["metadata"] = None
    with pytest.raises(TrajectoryError):
        Turn.from_dict(extra)


def test_trajectory_from_dict_requires_turn_meta_key():
    traj = with_metas(build_trajectory(ASSIGN, "none"), metas_for(build_trajectory(ASSIGN, "none")))
    d = traj.to_dict()
    del d["turns"][1]["meta"]
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(d)
    plain = build_trajectory(ASSIGN, "none").to_dict()
    for turn_dict in plain["turns"]:
        del turn_dict["meta"]
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(plain)


# Turn.from_dict 與 Trajectory.from_dict 都必須拒絕（"meta 為 None 或 dict" 或 規則 10 的 JSON 文件規則）。
FROM_DICT_BAD_META = [
    pytest.param(lambda: [], id="empty_list"),
    pytest.param(lambda: [{"a": 1}], id="list_of_dicts"),
    pytest.param(lambda: "meta", id="str"),
    pytest.param(lambda: 0, id="int"),
    pytest.param(lambda: False, id="bool"),
    pytest.param(lambda: 1.5, id="float"),
    pytest.param(lambda: ("a", 1), id="tuple"),
    pytest.param(lambda: {"a": (1, 2)}, id="nested_tuple"),
    pytest.param(lambda: {"a": [1, {"b": (2,)}]}, id="deeply_nested_tuple"),
    pytest.param(lambda: {"a": {1, 2}}, id="nested_set"),
    pytest.param(lambda: {"a": b"raw"}, id="nested_bytes"),
    pytest.param(lambda: {"a": object()}, id="nested_object"),
    pytest.param(lambda: {1: "a"}, id="int_key"),
    pytest.param(lambda: {"a": {None: 1}}, id="nested_none_key"),
    pytest.param(lambda: {"v": _nested(5000)}, id="depth_5000"),
]


@pytest.mark.parametrize("make_bad", FROM_DICT_BAD_META)
def test_turn_and_trajectory_from_dict_reject_invalid_meta(make_bad):
    traj = build_trajectory(ASSIGN, "none")
    turn_dict = traj.to_dict()["turns"][0]
    turn_dict["meta"] = make_bad()
    with pytest.raises(TrajectoryError):
        Turn.from_dict(turn_dict)
    d = traj.to_dict()
    d["turns"][0]["meta"] = make_bad()
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(d)


# 型別上是 dict，但不是可序列化的 UTF-8 JSON（規則 9）：至少 Trajectory.from_dict 必須拒絕。
FROM_DICT_UNSERIALIZABLE_META = [
    pytest.param(lambda: {"v": NAN}, id="nan"),
    pytest.param(lambda: {"v": [1, {"w": INF}]}, id="nested_inf"),
    pytest.param(lambda: {"request_id": "bad \ud800"}, id="surrogate_value"),
    pytest.param(lambda: {"\udc00": 1}, id="surrogate_key"),
]


@pytest.mark.parametrize("make_bad", FROM_DICT_UNSERIALIZABLE_META)
def test_trajectory_from_dict_rejects_unserializable_meta(make_bad):
    d = build_trajectory(ASSIGN, "none").to_dict()
    d["turns"][-1]["meta"] = make_bad()
    with pytest.raises(TrajectoryError):
        Trajectory.from_dict(d)


def test_trajectory_load_rejects_nan_literal_in_meta(tmp_path: Path):
    traj = with_metas(build_trajectory(ASSIGN, "none"), metas_for(build_trajectory(ASSIGN, "none")))
    d = traj.to_dict()
    d["turns"][0]["meta"]["v"] = "__NAN_SENTINEL__"
    text = canonical_json(d).replace('"__NAN_SENTINEL__"', "NaN")
    assert "NaN" in text
    path = tmp_path / "nan-meta.json"
    path.write_bytes(text.encode("utf-8"))
    with pytest.raises(TrajectoryError):
        Trajectory.load(path)


# 直接建構（不經 from_dict）的 Turn 不驗證 meta；Trajectory.validate 的規則 9 / 10 必須擋下。
IN_MEMORY_BAD_META = [
    pytest.param(lambda: ("a", 1), id="tuple"),
    pytest.param(lambda: {"a": (1, 2)}, id="nested_tuple"),
    pytest.param(lambda: {"a": [1, {"b": (2,)}]}, id="deeply_nested_tuple"),
    pytest.param(lambda: {"a": {1, 2}}, id="nested_set"),
    pytest.param(lambda: {"a": b"raw"}, id="nested_bytes"),
    pytest.param(lambda: {"a": object()}, id="nested_object"),
    pytest.param(lambda: {1: "a"}, id="int_key"),
    pytest.param(lambda: {"a": {2: "b"}}, id="nested_int_key"),
    pytest.param(lambda: {"v": _nested(5000)}, id="depth_5000"),
    pytest.param(lambda: {"v": NAN}, id="nan"),
    pytest.param(lambda: {"v": -INF}, id="neg_inf"),
    pytest.param(lambda: {"\ud800": 1}, id="surrogate_key"),
    pytest.param(lambda: {"a": ["ok", "\udfff"]}, id="nested_surrogate"),
]


@pytest.mark.parametrize("index", [0, -1], ids=["first_turn", "final_turn"])
@pytest.mark.parametrize("make_bad", IN_MEMORY_BAD_META)
def test_trajectory_validate_rejects_non_json_turn_meta(make_bad, index):
    traj = build_trajectory(ASSIGN, "none")
    bad = set_turn_meta(traj, index % len(traj.turns), make_bad())
    with pytest.raises(TrajectoryError):
        bad.validate()
    with pytest.raises(TrajectoryError):
        bad.verify()
    with pytest.raises(TrajectoryError):  # make_checkpoint 先 verify，TrajectoryError 原樣拋出
        make_checkpoint(bad, 1)


@pytest.mark.parametrize("mode", ("none", "all"))
@pytest.mark.parametrize("task_id", (ASSIGN, REASSIGN))
def test_trajectory_with_turn_meta_round_trips_byte_identically(task_id, mode, tmp_path: Path):
    plain = build_trajectory(task_id, mode)
    metas = metas_for(plain)
    traj = with_metas(plain, metas)
    traj.validate()
    traj.verify()
    d = traj.to_dict()
    _assert_json_native(d)
    for turn_dict, meta in zip(d["turns"], metas, strict=True):
        assert set(turn_dict) == set(TURN_KEYS)
        assert canonical_json(turn_dict["meta"]) == canonical_json(meta)

    canon = canonical_json(d)
    for back in (Trajectory.from_dict(traj.to_dict()), Trajectory.from_dict(json.loads(canon))):
        assert back == traj
        assert canonical_json(back.to_dict()) == canon
        assert back.trajectory_id == traj.trajectory_id

    first = tmp_path / "traj-meta.json"
    traj.save(first)
    assert first.read_bytes() == (canon + "\n").encode("utf-8")
    loaded = Trajectory.load(first)
    assert loaded == traj
    assert loaded.trajectory_id == traj.trajectory_id
    for turn, meta in zip(loaded.turns, metas, strict=True):
        assert canonical_json(turn.meta) == canonical_json(meta)
    second = tmp_path / "traj-meta-again.json"
    loaded.save(second)
    assert second.read_bytes() == first.read_bytes()


def test_trajectory_to_dict_does_not_alias_turn_meta():
    traj = with_metas(build_trajectory(ASSIGN, "none"), metas_for(build_trajectory(ASSIGN, "none")))
    before = canonical_json(traj.to_dict())
    d = traj.to_dict()
    d["turns"][0]["meta"]["usage"]["input_tokens"] = 999
    d["turns"][-1]["meta"].clear()
    assert canonical_json(traj.to_dict()) == before


def test_turn_meta_is_part_of_the_record_but_not_the_conversation():
    plain = build_trajectory(REASSIGN, "all")
    measured = with_metas(plain, metas_for(plain))
    # meta 是量測：進 trajectory 紀錄（trajectory_id 是紀錄 id），但不進 conversation
    assert measured.trajectory_id != plain.trajectory_id
    assert json.dumps(measured.messages(), ensure_ascii=False) == json.dumps(plain.messages(), ensure_ascii=False)

    tweaked_meta = sample_meta(0)
    tweaked_meta["elapsed"] = 2.0
    tweaked = set_turn_meta(measured, 0, tweaked_meta)
    assert tweaked.trajectory_id != measured.trajectory_id
    assert json.dumps(tweaked.messages(), ensure_ascii=False) == json.dumps(plain.messages(), ensure_ascii=False)

    empty = set_turn_meta(plain, 0, {})
    assert empty.trajectory_id != plain.trajectory_id  # None 與 {} 是不同紀錄

    inty = set_turn_meta(plain, 0, {"x": 1})
    floaty = set_turn_meta(plain, 0, {"x": 1.0})
    assert inty == floaty  # Python == 把 1 與 1.0 視為相同
    assert inty.trajectory_id != floaty.trajectory_id


@pytest.mark.parametrize("mode", PROVIDER_MODES)
def test_render_messages_never_includes_meta(mode):
    plain = build_trajectory(REASSIGN, mode)
    measured = with_metas(plain, metas_for(plain))
    contents = planned_contents(REASSIGN, mode)
    instruction = plain.task["instruction"]
    for k in range(len(plain.turns) + 1):
        actual = render_messages(instruction, measured.agent, measured.turns[:k])
        expected = expected_messages(instruction, plain.agent, plain.turns[:k], contents[:k])
        text = json.dumps(actual, ensure_ascii=False)
        assert text == json.dumps(expected, ensure_ascii=False)
        assert META_SENTINEL not in text
        assert all("meta" not in message for message in actual)
        assert [list(m) for m in actual] == [ROLE_KEYS[m["role"]] for m in actual]


# ---------------------------------------------------------------------------
# Checkpoint 與 provider_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("all", "mixed"))
@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_provider_checkpoints_verify_and_round_trip_byte_identically(task_id, mode, tmp_path: Path):
    traj = build_trajectory(task_id, mode)
    contents = planned_contents(task_id, mode)
    for k in range(len(traj.turns)):  # 最後一個 turn 是 final answer，之後不能建立 checkpoint
        cp = make_checkpoint(traj, k)
        cp.verify()
        d = cp.to_dict()
        _assert_json_native(d)
        assert d["format"] == CHECKPOINT_FORMAT
        assert len(d["turns"]) == k
        for turn_dict, pc in zip(d["turns"], contents[:k], strict=True):
            assert set(turn_dict["message"]) == MESSAGE_KEYS
            if pc is None:
                assert turn_dict["message"]["provider_content"] is None
            else:
                assert canonical_json(turn_dict["message"]["provider_content"]) == canonical_json(pc)

        canon = canonical_json(d)
        back = Checkpoint.from_dict(json.loads(canon))
        assert back == cp
        assert canonical_json(back.to_dict()) == canon

        path = tmp_path / f"cp-{k}.json"
        cp.save(path)
        assert path.read_bytes() == (canon + "\n").encode("utf-8")
        loaded = Checkpoint.load(path)
        assert loaded == cp
        assert loaded.checkpoint_id == cp.checkpoint_id
        again = tmp_path / f"cp-{k}-again.json"
        loaded.save(again)
        assert again.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("mode", PROVIDER_MODES)
def test_checkpoint_messages_carry_provider_content(mode, tmp_path: Path):
    traj = build_trajectory(REASSIGN, mode)
    contents = planned_contents(REASSIGN, mode)
    for k in range(len(traj.turns)):
        cp = make_checkpoint(traj, k)
        expected = expected_messages(traj.task["instruction"], traj.agent, traj.turns[:k], contents[:k])
        expected_text = json.dumps(expected, ensure_ascii=False)
        assert json.dumps(cp.messages(), ensure_ascii=False) == expected_text
        path = tmp_path / f"cp-{k}.json"
        cp.save(path)
        restored = restore(Checkpoint.load(path))
        assert json.dumps(restored.messages, ensure_ascii=False) == expected_text
        assistants = [m for m in restored.messages if m["role"] == "assistant"]
        assert len(assistants) == k
        for message, pc in zip(assistants, contents[:k], strict=True):
            assert list(message) == ROLE_KEYS["assistant"]
            if pc is None:
                assert message["provider_content"] is None
            else:
                assert canonical_json(message["provider_content"]) == canonical_json(pc)


def test_checkpoint_messages_are_fresh_copies():
    traj = build_trajectory(ASSIGN, "all")
    cp = make_checkpoint(traj, 2)
    before = canonical_json(cp.to_dict())
    messages = cp.messages()
    for message in messages:
        if message["role"] == "assistant":
            message["provider_content"][0]["signature"] = "forged"
            message["provider_content"].clear()
    assert canonical_json(cp.to_dict()) == before
    cp.verify()


def test_checkpoint_id_changes_and_lineage_breaks_when_provider_content_changes():
    traj = build_trajectory(ASSIGN, "all")
    contents = planned_contents(ASSIGN, "all")
    cp = make_checkpoint(traj, 2)
    verify_lineage(cp, traj)
    for i in range(2):
        for new_pc in (_pc_signature_changed(contents[i]), None, _pc_float_input(contents[i])):
            tampered = R(cp, turns=set_turn_pc(cp.turns, i, new_pc))
            tampered.verify()  # tool steps 未變：checkpoint 自身仍一致
            assert tampered.checkpoint_id != cp.checkpoint_id
            with pytest.raises(CheckpointError):
                verify_lineage(tampered, traj)


def test_checkpoint_from_dict_detects_provider_content_tampering():
    traj = build_trajectory(ASSIGN, "all")
    cp = make_checkpoint(traj, 2)

    stale_id = cp.to_dict()
    stale_id["turns"][0]["message"]["provider_content"][0]["signature"] = "forged"
    with pytest.raises(CheckpointError):
        Checkpoint.from_dict(stale_id)

    recomputed = cp.to_dict()
    recomputed["turns"][1]["message"]["provider_content"][0]["signature"] = "forged"
    recomputed["checkpoint_id"] = sha256_of({k: v for k, v in recomputed.items() if k != "checkpoint_id"})
    forged = Checkpoint.from_dict(recomputed)  # 內部 digest 擋不住重算；由 lineage 以來源 trajectory 為錨點偵測
    assert forged.checkpoint_id != cp.checkpoint_id
    with pytest.raises(CheckpointError):
        verify_lineage(forged, traj)


def test_checkpoint_from_dict_requires_provider_content_key():
    traj = build_trajectory(ASSIGN, "all")
    cp = make_checkpoint(traj, 2)
    d = cp.to_dict()
    del d["turns"][1]["message"]["provider_content"]
    d["checkpoint_id"] = sha256_of({k: v for k, v in d.items() if k != "checkpoint_id"})
    with pytest.raises(CheckpointError):
        Checkpoint.from_dict(d)


# ---------------------------------------------------------------------------
# Checkpoint 與 Turn.meta（turns 與 Trajectory 相同編碼 → meta 在 checkpoint 內、在 checkpoint_id 內，但不在 conversation 內）
# ---------------------------------------------------------------------------


def _rehash(d: dict[str, Any]) -> dict[str, Any]:
    d["checkpoint_id"] = sha256_of(_without(d, "checkpoint_id"))
    return d


@pytest.mark.parametrize("mode", ("none", "all"))
def test_checkpoint_carries_turn_meta_but_restored_conversation_does_not(mode, tmp_path: Path):
    plain = build_trajectory(REASSIGN, mode)
    metas = metas_for(plain)
    measured = with_metas(plain, metas)
    for k in range(len(measured.turns)):
        cp = make_checkpoint(measured, k)
        plain_cp = make_checkpoint(plain, k)
        d = cp.to_dict()
        assert len(d["turns"]) == k
        for turn_dict, meta in zip(d["turns"], metas[:k], strict=True):
            assert set(turn_dict) == set(TURN_KEYS)
            assert canonical_json(turn_dict["meta"]) == canonical_json(meta)
        assert cp.checkpoint_id != plain_cp.checkpoint_id  # source_trajectory_id 不同；k > 0 時前綴的 meta 也不同

        path = tmp_path / f"cp-meta-{k}.json"
        cp.save(path)
        assert path.read_bytes() == (canonical_json(d) + "\n").encode("utf-8")
        loaded = Checkpoint.load(path)
        assert loaded == cp
        assert loaded.checkpoint_id == cp.checkpoint_id
        for turn, meta in zip(loaded.turns, metas[:k], strict=True):
            assert canonical_json(turn.meta) == canonical_json(meta)

        restored = restore(loaded)
        text = json.dumps(restored.messages, ensure_ascii=False)
        assert text == json.dumps(restore(plain_cp).messages, ensure_ascii=False)
        assert text == json.dumps(loaded.messages(), ensure_ascii=False)
        assert META_SENTINEL not in text
        assert all("meta" not in message for message in restored.messages)

        verify_lineage(loaded, measured)
        with pytest.raises(CheckpointError):
            verify_lineage(loaded, plain)


def test_checkpoint_from_dict_requires_valid_turn_meta():
    cp = make_checkpoint(with_metas(build_trajectory(ASSIGN, "none"), metas_for(build_trajectory(ASSIGN, "none"))), 2)

    missing = cp.to_dict()
    del missing["turns"][1]["meta"]
    with pytest.raises(CheckpointError):
        Checkpoint.from_dict(_rehash(missing))

    # checkpoint_id 以竄改後的內容重算，因此被拒絕的原因只能是 meta 本身
    for bad in ([1], [], "meta", 0, False, {"a": (1,)}, ("a", 1)):
        d = cp.to_dict()
        d["turns"][0]["meta"] = bad
        with pytest.raises(CheckpointError):
            Checkpoint.from_dict(_rehash(d))

    # 無法計算 digest 的內容（set、int key、過深）：必須是 CheckpointError，不得漏出 TypeError / RecursionError
    for bad in ({"a": {1, 2}}, {1: "a"}, {"v": _nested(5000)}):
        d = cp.to_dict()
        d["turns"][0]["meta"] = bad
        with pytest.raises(CheckpointError):
            Checkpoint.from_dict(d)


def test_checkpoint_meta_tampering_changes_id_and_breaks_lineage():
    source = with_metas(build_trajectory(ASSIGN, "all"), metas_for(build_trajectory(ASSIGN, "all")))
    cp = make_checkpoint(source, 2)

    stale = cp.to_dict()
    stale["turns"][0]["meta"]["usage"]["input_tokens"] = 1
    with pytest.raises(CheckpointError):
        Checkpoint.from_dict(stale)  # checkpoint_id 不符

    for i in range(2):
        recomputed = cp.to_dict()
        recomputed["turns"][i]["meta"]["elapsed"] = 99.0
        forged = Checkpoint.from_dict(_rehash(recomputed))  # meta 是量測，不經重放驗證；由 lineage 偵測
        assert forged.checkpoint_id != cp.checkpoint_id
        assert json.dumps(restore(forged).messages, ensure_ascii=False) == json.dumps(
            restore(cp).messages, ensure_ascii=False
        )
        with pytest.raises(CheckpointError):
            verify_lineage(forged, source)

        dropped = R(cp, turns=tuple(R(t, meta=None) if j == i else t for j, t in enumerate(cp.turns)))
        dropped.verify()
        assert dropped.checkpoint_id != cp.checkpoint_id
        with pytest.raises(CheckpointError):
            verify_lineage(dropped, source)


# ---------------------------------------------------------------------------
# 端到端：provider 風格 policy 的 conversation 經 save / load / restore 後逐 byte 相同（SPEC §5、decision 009）
# ---------------------------------------------------------------------------


class ProviderEchoPolicy:
    """決定性、無狀態的 provider 風格 policy：第 k 個 turn 送出 oracle 的第 k 個 call，並附上 thinking 等 provider blocks。

    只為觀察而記錄每次收到的 messages（order-sensitive JSON），不依此做決策。
    """

    def __init__(self, calls: Any) -> None:
        self._calls = tuple((name, copy.deepcopy(args)) for name, args in calls)
        self.seen: list[str] = []
        self._config = AgentConfig(
            policy="provider-echo",
            prompt_version="provider-echo/v1",
            system_prompt="You are a triage bot. 你好",
            developer_prompt="Use the tools.",
            model_config={"calls": [{"name": name, "args": args} for name, args in self._calls]},
        )

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        self.seen.append(json.dumps(messages, ensure_ascii=False))
        k = sum(1 for message in messages if message["role"] == "assistant")
        if k < len(self._calls):
            name, args = self._calls[k]
            call_id = f"toolu_{k:02d}"
            message = ModelMessage(
                text=turn_text(k),
                tool_calls=(ToolCall(call_id, name, copy.deepcopy(args)),),
                stop_reason="tool_use",
                provider_content=tool_turn_content(k, name, args, call_id, turn_text(k)),
            )
        else:
            message = ModelMessage(
                text="Done.", tool_calls=(), stop_reason="end_turn", provider_content=final_turn_content()
            )
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_restored_conversation_equals_what_provider_policy_saw(task_id, tmp_path: Path):
    calls = REFERENCE_SOLUTIONS[task_id]
    policy = ProviderEchoPolicy(calls)
    original = run_episode(get_smoke_task(task_id), policy, max_turns=len(calls) + 3, clock=_zero_clock)
    assert original.termination == "final_answer"
    assert len(policy.seen) == len(original.turns)
    contents = planned_contents(task_id, "all")
    for turn, pc in zip(original.turns, contents, strict=True):
        assert canonical_json(turn.message.provider_content) == canonical_json(pc)

    traj_path = tmp_path / "traj.json"
    original.save(traj_path)
    source = Trajectory.load(traj_path)
    assert source.trajectory_id == original.trajectory_id

    for k in range(len(source.turns)):
        cp_path = tmp_path / f"cp-{k}.json"
        make_checkpoint(source, k).save(cp_path)
        cp = Checkpoint.load(cp_path)
        restored = restore(cp)
        # 核心 invariant：restore 後的 conversation（含先前 turns 的 provider_content）與 policy 當時收到的逐 byte 相同
        assert json.dumps(restored.messages, ensure_ascii=False) == policy.seen[k]
        assistants = [m for m in restored.messages if m["role"] == "assistant"]
        assert len(assistants) == k
        for message, pc in zip(assistants, contents[:k], strict=True):
            assert canonical_json(message["provider_content"]) == canonical_json(pc)

        rerun_policy = ProviderEchoPolicy(calls)
        child = rerun(cp, rerun_policy, clock=_zero_clock)
        assert rerun_policy.seen[0] == policy.seen[k]
        assert canonical_json([t.to_dict() for t in child.turns]) == canonical_json([t.to_dict() for t in source.turns])
        assert child.termination == source.termination
        assert canonical_json(child.final_snapshot) == canonical_json(source.final_snapshot)
        verify_lineage(child, source, checkpoint=cp)


# ---------------------------------------------------------------------------
# runner：Turn.meta = PolicyStep.meta（量測進紀錄，不進下一次呼叫的 conversation）
# ---------------------------------------------------------------------------


class MeasuredProviderPolicy(ProviderEchoPolicy):
    """與 ProviderEchoPolicy 相同的決定性行為（相同 agent_config），另回報第 k 個 turn 的 sample_meta(k)。"""

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        step = super().next_message(messages, tools)
        k = sum(1 for message in messages if message["role"] == "assistant")
        return PolicyStep(
            message=step.message,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            cost_usd=step.cost_usd,
            meta=sample_meta(k),
        )


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_runner_records_policy_meta_on_turns_outside_the_conversation(task_id, tmp_path: Path):
    calls = REFERENCE_SOLUTIONS[task_id]
    task = get_smoke_task(task_id)
    policy = MeasuredProviderPolicy(calls)
    traj = run_episode(task, policy, max_turns=len(calls) + 3, clock=_zero_clock)
    assert traj.termination == "final_answer"
    assert len(traj.turns) == len(calls) + 1
    for k, turn in enumerate(traj.turns):
        assert type(turn.meta) is dict
        assert canonical_json(turn.meta) == canonical_json(sample_meta(k))
        _assert_keys_sorted_recursively(turn.meta)  # PolicyStep 已正規化
    for turn_dict in traj.to_dict()["turns"]:
        assert set(turn_dict) == set(TURN_KEYS)

    # 後續呼叫收到的 conversation 不含 meta，且與不回報 meta 的同一 policy 收到的逐 byte 相同
    assert len(policy.seen) == len(traj.turns)
    for seen in policy.seen:
        assert META_SENTINEL not in seen
        assert all("meta" not in message for message in json.loads(seen))
    echo = ProviderEchoPolicy(calls)
    plain = run_episode(task, echo, max_turns=len(calls) + 3, clock=_zero_clock)
    assert echo.seen == policy.seen
    assert json.dumps(plain.messages(), ensure_ascii=False) == json.dumps(traj.messages(), ensure_ascii=False)
    assert all(turn.meta is None for turn in plain.turns)
    assert plain.trajectory_id != traj.trajectory_id

    # save / load / checkpoint / rerun 保留 meta；restore 的 conversation 仍等於 policy 當時收到的
    path = tmp_path / "measured.json"
    traj.save(path)
    source = Trajectory.load(path)
    assert source.trajectory_id == traj.trajectory_id
    for k, turn in enumerate(source.turns):
        assert canonical_json(turn.meta) == canonical_json(sample_meta(k))
    for k in range(len(source.turns)):
        cp = make_checkpoint(source, k)
        assert json.dumps(restore(cp).messages, ensure_ascii=False) == policy.seen[k]
        rerun_policy = MeasuredProviderPolicy(calls)
        child = rerun(cp, rerun_policy, clock=_zero_clock)
        assert rerun_policy.seen[0] == policy.seen[k]
        assert canonical_json([t.to_dict() for t in child.turns]) == canonical_json([t.to_dict() for t in source.turns])
        verify_lineage(child, source, checkpoint=cp)


def test_scripted_policy_turns_have_meta_none():
    task = get_smoke_task(ASSIGN)
    policy = oracle_policy(REFERENCE_SOLUTIONS[ASSIGN])
    step = policy.next_message(render_messages(task.instruction, policy.agent_config, ()), IssueTrackerEnv.tool_specs())
    assert step.meta is None
    traj = run_episode(task, policy, max_turns=10, clock=_zero_clock)
    assert all(turn.meta is None for turn in traj.turns)
    for turn_dict in traj.to_dict()["turns"]:
        assert set(turn_dict) == set(TURN_KEYS)
        assert turn_dict["meta"] is None
    back = Trajectory.from_dict(json.loads(canonical_json(traj.to_dict())))
    assert back == traj
    assert all(turn.meta is None for turn in back.turns)
