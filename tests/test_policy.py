"""agents.policy 的契約測試（decision log 006）：ToolCall、ModelMessage、PolicyStep、AgentConfig、
Policy protocol 形狀、ScriptedPolicy、oracle_policy，以及 env.environment.check_call_preconditions。

依據：agents/policy.py 的模組 docstring「統一規則」與各 class docstring。
所有「逐 byte 相同 / 同一個 agent」的比較一律用 canonical_json 或 config_id，不用 Python ==。
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import random
from dataclasses import fields
from typing import Any

import pytest

from agents.policy import (
    SCRIPTED_POLICY,
    SCRIPTED_PROMPT_VERSION,
    AgentConfig,
    ModelMessage,
    Policy,
    PolicyStep,
    ScriptedPolicy,
    ToolCall,
    oracle_policy,
)
from env.environment import MAX_ARGS_DEPTH, IssueTrackerEnv, check_call_preconditions
from env.hashing import HASH_PREFIX, canonical_json, sha256_of
from grader.grader import grade
from tasks.smoke import REFERENCE_SOLUTIONS, SMOKE_TASK_IDS, get_smoke_task

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------- helpers


def _nested(depth: int) -> Any:
    """depth 層 list 包住 1。"""
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


DEEP = _nested(5000)  # 遠超過任何合理上限（純資料，不呼叫 stub）

# args 深度邊界（依 check_call_preconditions：args 本身為深度 0，最內層值深度 > MAX_ARGS_DEPTH 才拒絕）
ARGS_AT_LIMIT = {"x": _nested(MAX_ARGS_DEPTH - 1)}  # 最內層值剛好在深度 MAX_ARGS_DEPTH
ARGS_PAST_LIMIT = {"x": _nested(MAX_ARGS_DEPTH)}  # 多一層
LIST_ARGS_AT_LIMIT = _nested(MAX_ARGS_DEPTH)  # 頂層就是 list（非 dict args 也合法）
LIST_ARGS_PAST_LIMIT = _nested(MAX_ARGS_DEPTH + 1)
MIXED_ARGS_AT_LIMIT = {"a": [{"b": _nested(MAX_ARGS_DEPTH - 3)}]}  # dict / list 交錯


def _ordered_json(value: Any) -> str:
    """order-sensitive 的 JSON（不排序 key），用來驗證「key 已遞迴排序」。"""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _assert_keys_sorted_recursively(value: Any) -> None:
    if isinstance(value, dict):
        assert list(value) == sorted(value)
        for item in value.values():
            _assert_keys_sorted_recursively(item)
    elif isinstance(value, list):
        for item in value:
            _assert_keys_sorted_recursively(item)


def _scramble(value: Any) -> None:
    """就地破壞所有巢狀 dict / list（驗證深拷貝用）。"""
    if isinstance(value, dict):
        for item in list(value.values()):
            _scramble(item)
        value.clear()
        value["__scrambled__"] = True
    elif isinstance(value, list):
        for item in value:
            _scramble(item)
        value.clear()
        value.append("__scrambled__")


_KEY_POOL = ("zeta", "alpha", "Beta", "b", "a", "", "10", "9", "é", "登入", "_x", "a b")


def _rand_index(rng: random.Random, n: int) -> int:
    return min(int(rng.random() * n), n - 1)


def _random_scalar(rng: random.Random) -> Any:
    kind = _rand_index(rng, 9)
    if kind == 0:
        return None
    if kind == 1:
        return True
    if kind == 2:
        return False
    if kind == 3:
        return _rand_index(rng, 2001) - 1000
    if kind == 4:
        return float(_rand_index(rng, 5))  # 整數值的 float：== 會與 int 混淆
    if kind == 5:
        return round(rng.random() * 100 - 50, 3)
    if kind == 6:
        return 10**20 + _rand_index(rng, 5)
    if kind == 7:
        return _KEY_POOL[_rand_index(rng, len(_KEY_POOL))] + "!"
    return "文字" + str(_rand_index(rng, 50))


def _random_object(rng: random.Random, depth: int = 0) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for _ in range(_rand_index(rng, 6)):
        obj[_KEY_POOL[_rand_index(rng, len(_KEY_POOL))]] = _random_json(rng, depth + 1)
    return obj


def _random_json(rng: random.Random, depth: int = 0) -> Any:
    r = rng.random()
    if depth >= 3 or r < 0.4:
        return _random_scalar(rng)
    if r < 0.65:
        return [_random_json(rng, depth + 1) for _ in range(_rand_index(rng, 4))]
    return _random_object(rng, depth)


def _tools() -> tuple[dict[str, Any], ...]:
    return IssueTrackerEnv.tool_specs()


# 一般劇本：turn 1 有 3 個 call；外層 tuple、內層混用 list；args key 刻意未排序。
SCRIPT: Any = (
    (("search_issues", {"status": "open", "label": "bug"}),),
    (
        ("get_issue", {"issue_id": 3}),
        ("update_issue", {"priority": "p0", "issue_id": 3, "labels": ["security", "bug"]}),
        ["add_comment", {"issue_id": 3, "body": "登入 token 已輪替"}],
    ),
    [["assign_issue", {"issue_id": 1, "assignee_id": "u1"}]],
)

# 型別敏感的劇本：Python == 會把它們視為相同，canonical_json 不會。
FLOAT_SCRIPT: Any = [[("get_issue", {"issue_id": 1.0, "flag": True})]]
INT_SCRIPT: Any = [[("get_issue", {"issue_id": 1, "flag": 1})]]


def _expected_model_config(script: Any, final_text: str = "Done.") -> dict[str, Any]:
    return {
        "script": [[{"name": name, "args": args} for name, args in turn] for turn in script],
        "final_text": final_text,
    }


def _message_json(msg: Any) -> str:
    """不經 to_dict，直接由欄位組出 canonical 形式（順便檢查型別）。"""
    assert isinstance(msg, ModelMessage)
    assert type(msg.tool_calls) is tuple
    assert all(isinstance(c, ToolCall) for c in msg.tool_calls)
    return canonical_json(
        {
            "text": msg.text,
            "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in msg.tool_calls],
            "stop_reason": msg.stop_reason,
        }
    )


def _expected_message_json(script: Any, k: int, final_text: str = "Done.") -> str:
    if k < len(script):
        calls = [{"id": f"call_{k}_{i}", "name": name, "args": args} for i, (name, args) in enumerate(script[k])]
        return canonical_json({"text": "", "tool_calls": calls, "stop_reason": "tool_use"})
    return canonical_json({"text": final_text, "tool_calls": [], "stop_reason": "end_turn"})


def _assert_scripted_usage(step: Any) -> None:
    assert isinstance(step, PolicyStep)
    # 0 / 0 / 0.0：用 canonical 比較才分得出 0 與 0.0
    assert canonical_json([step.input_tokens, step.output_tokens, step.cost_usd]) == "[0,0,0.0]"


def _conversation(n_assistant: int, script: Any = (), instruction: str = "Fix the login bug.") -> list[dict[str, Any]]:
    """手工組出 render_messages 形狀的 conversation，含 n_assistant 則 assistant 訊息。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ""},
        {"role": "developer", "content": ""},
        {"role": "user", "content": instruction},
    ]
    for k in range(n_assistant):
        calls = script[k] if k < len(script) else ()
        tool_calls = [
            {"id": f"call_{k}_{i}", "name": name, "args": copy.deepcopy(args)} for i, (name, args) in enumerate(calls)
        ]
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for tc in tool_calls:
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": canonical_json({"ok": True, "data": {}})}
            )
    return messages


BASE_CFG: dict[str, Any] = {
    "policy": "llm",
    "prompt_version": "react/v1",
    "system_prompt": "You are a triage bot.",
    "developer_prompt": "Use the tools.",
    "model_config": {"temperature": 0.0, "model": "m-1", "max_tokens": 1024},
}
AGENT_CONFIG_FIELDS = ("policy", "prompt_version", "system_prompt", "developer_prompt", "model_config")


def _cfg_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = copy.deepcopy(BASE_CFG)
    kwargs.update(overrides)
    return kwargs


def _cfg(**overrides: Any) -> AgentConfig:
    return AgentConfig(**_cfg_kwargs(**overrides))


VALID_TC = {"id": "call_0_0", "name": "get_issue", "args": {"issue_id": 1}}


# ---------------------------------------------------------------- 常數、欄位與介面形狀


def test_scripted_constants():
    assert SCRIPTED_POLICY == "scripted"
    assert SCRIPTED_PROMPT_VERSION == "scripted/v1"


@pytest.mark.parametrize(
    "cls, names",
    [
        (ToolCall, ["id", "name", "args"]),
        (ModelMessage, ["text", "tool_calls", "stop_reason"]),
        (PolicyStep, ["message", "input_tokens", "output_tokens", "cost_usd"]),
        (AgentConfig, list(AGENT_CONFIG_FIELDS)),
    ],
)
def test_dataclass_fields_and_frozen(cls, names):
    assert [f.name for f in fields(cls)] == names
    assert cls.__dataclass_params__.frozen


def test_policy_step_usage_defaults_are_none():
    msg = ModelMessage(text="", tool_calls=(), stop_reason=None)
    step = PolicyStep(msg)
    assert step.input_tokens is None and step.output_tokens is None and step.cost_usd is None


def test_policy_protocol_shape():
    assert isinstance(Policy.__dict__["agent_config"], property)
    assert list(inspect.signature(Policy.next_message).parameters) == ["self", "messages", "tools"]
    assert isinstance(ScriptedPolicy.__dict__["agent_config"], property)
    assert list(inspect.signature(ScriptedPolicy.next_message).parameters) == ["self", "messages", "tools"]


def test_scripted_policy_and_oracle_signatures():
    params = inspect.signature(ScriptedPolicy.__init__).parameters
    assert list(params) == ["self", "script", "system_prompt", "developer_prompt", "prompt_version", "final_text"]
    for name in ("system_prompt", "developer_prompt", "prompt_version", "final_text"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["system_prompt"].default == ""
    assert params["developer_prompt"].default == ""
    assert params["prompt_version"].default == SCRIPTED_PROMPT_VERSION
    assert params["final_text"].default == "Done."
    oracle_params = inspect.signature(oracle_policy).parameters
    assert list(oracle_params) == ["oracle_calls", "kwargs"]
    assert oracle_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


# ---------------------------------------------------------------- check_call_preconditions（已實作）

PRECONDITION_OK = [
    pytest.param("get_issue", {"issue_id": 1}, id="normal"),
    pytest.param("get_issue", {}, id="empty_dict"),
    pytest.param("close_issue", {"issue_id": 1}, id="unknown_tool"),
    pytest.param("search_issues", [1, "a", None], id="list_args"),
    pytest.param("search_issues", "open", id="str_args"),
    pytest.param("search_issues", 5, id="int_args"),
    pytest.param("search_issues", None, id="none_args"),
    pytest.param("search_issues", True, id="bool_args"),
    pytest.param("search_issues", 1.5, id="float_args"),
    pytest.param("get_issue", {"issue_id": 10**30}, id="big_int"),
    pytest.param("add_comment", {"issue_id": 1, "body": "登入頁面 ✓"}, id="non_ascii"),
    pytest.param("get_issue", {"issue_id": 1, "x": _nested(MAX_ARGS_DEPTH - 2)}, id="deep_within_limit"),
    pytest.param("get_issue", ARGS_AT_LIMIT, id="deep_at_limit"),
    pytest.param("search_issues", LIST_ARGS_AT_LIMIT, id="list_deep_at_limit"),
    pytest.param("get_issue", MIXED_ARGS_AT_LIMIT, id="mixed_deep_at_limit"),
]

PRECONDITION_TYPE_ERRORS = [
    pytest.param(5, {}, id="name_int"),
    pytest.param(None, {}, id="name_none"),
    pytest.param(True, {}, id="name_bool"),
    pytest.param(b"get_issue", {}, id="name_bytes"),
    pytest.param(("get_issue",), {}, id="name_tuple"),
    pytest.param("get_issue", (1, 2), id="args_top_level_tuple"),
    pytest.param("update_issue", {"issue_id": 1, "labels": ("bug",)}, id="args_nested_tuple"),
    pytest.param("update_issue", {"issue_id": 1, "labels": {"bug"}}, id="args_set"),
    pytest.param("get_issue", {1: "x"}, id="args_int_key"),
    pytest.param("get_issue", {"a": [{None: 1}]}, id="args_nested_none_key"),
    pytest.param("get_issue", {"issue_id": b"1"}, id="args_bytes"),
    pytest.param("get_issue", {"issue_id": object()}, id="args_object"),
]

PRECONDITION_VALUE_ERRORS = [
    pytest.param("\ud800", {}, id="name_lone_surrogate"),
    pytest.param("get_issue", {"issue_id": NAN}, id="args_nan"),
    pytest.param("get_issue", {"issue_id": INF}, id="args_inf"),
    pytest.param("get_issue", [-INF], id="args_neg_inf_in_list"),
    pytest.param("add_comment", {"issue_id": 1, "body": "bad \udc00"}, id="args_surrogate_value"),
    pytest.param("add_comment", {"issue_id": 1, "\ud83d": "x"}, id="args_surrogate_key"),
    pytest.param("add_comment", {"meta": {"inner": {"\udfff": 1}}}, id="args_surrogate_nested_key"),
    pytest.param("add_comment", "\ud800", id="args_surrogate_top_level_str"),
    pytest.param("get_issue", {"x": _nested(MAX_ARGS_DEPTH + 5)}, id="args_too_deep"),
    pytest.param("get_issue", ARGS_PAST_LIMIT, id="args_one_past_limit"),
    pytest.param("search_issues", LIST_ARGS_PAST_LIMIT, id="list_args_one_past_limit"),
    pytest.param("get_issue", DEEP, id="args_far_too_deep"),
]


@pytest.mark.parametrize("name, args", PRECONDITION_OK)
def test_check_call_preconditions_accepts_json_values(name, args):
    before = _ordered_json(args)
    assert check_call_preconditions(name, args) is None
    assert _ordered_json(args) == before


@pytest.mark.parametrize("name, args", PRECONDITION_TYPE_ERRORS)
def test_check_call_preconditions_type_errors(name, args):
    with pytest.raises(TypeError):
        check_call_preconditions(name, args)


@pytest.mark.parametrize("name, args", PRECONDITION_VALUE_ERRORS)
def test_check_call_preconditions_value_errors(name, args):
    with pytest.raises(ValueError):
        check_call_preconditions(name, args)


# ---------------------------------------------------------------- ToolCall：建構與正規化


@pytest.mark.parametrize("name, args", PRECONDITION_OK)
def test_tool_call_accepts_what_preconditions_accept(name, args):
    call = ToolCall("call_0_0", name, args)
    assert call.id == "call_0_0"
    assert call.name == name
    assert canonical_json(call.args) == canonical_json(args)
    assert _ordered_json(call.args) == canonical_json(args)


@pytest.mark.parametrize("name, args", PRECONDITION_TYPE_ERRORS + PRECONDITION_VALUE_ERRORS)
def test_tool_call_propagates_precondition_errors_as_is(name, args):
    with pytest.raises((TypeError, ValueError)) as expected:
        check_call_preconditions(name, args)
    with pytest.raises((TypeError, ValueError)) as actual:
        ToolCall("call_0_0", name, args)
    # 「其 TypeError / ValueError 原樣拋出」
    assert type(actual.value) is type(expected.value)


@pytest.mark.parametrize(
    "call_id, exc",
    [
        pytest.param(5, TypeError, id="int"),
        pytest.param(None, TypeError, id="none"),
        pytest.param(True, TypeError, id="bool"),
        pytest.param(b"call", TypeError, id="bytes"),
        pytest.param(["call"], TypeError, id="list"),
        pytest.param("", ValueError, id="empty"),
        pytest.param("\ud800", ValueError, id="lone_surrogate"),
        pytest.param("call_\udfff_0", ValueError, id="embedded_surrogate"),
    ],
)
def test_tool_call_id_validation(call_id, exc):
    with pytest.raises(exc):
        ToolCall(call_id, "get_issue", {"issue_id": 1})


def test_tool_call_args_keys_are_sorted_recursively():
    args = {"z": 1, "a": {"y": [{"d": 1, "c": 2}], "b": None}, "m": True}
    call = ToolCall("c", "update_issue", args)
    assert list(call.args) == ["a", "m", "z"]
    assert list(call.args["a"]) == ["b", "y"]
    assert list(call.args["a"]["y"][0]) == ["c", "d"]
    # 輸入本身不被重排
    assert list(args) == ["z", "a", "m"]


def test_tool_call_args_preserve_json_types():
    floaty = ToolCall("c", "get_issue", {"issue_id": 1.0, "flag": True, "n": None})
    inty = ToolCall("c", "get_issue", {"issue_id": 1, "flag": 1, "n": None})
    assert canonical_json(floaty.args) == '{"flag":true,"issue_id":1.0,"n":null}'
    assert canonical_json(inty.args) == '{"flag":1,"issue_id":1,"n":null}'
    # Python == 被騙；canonical 才分得出來
    assert floaty == inty
    assert canonical_json(floaty.args) != canonical_json(inty.args)


def test_tool_call_deep_copies_args():
    args = {"issue_id": 1, "labels": ["bug"], "meta": {"k": [1, {"x": 2}]}}
    snapshot = canonical_json(args)
    call = ToolCall("c", "update_issue", args)
    assert call.args is not args
    _scramble(args)
    assert canonical_json(call.args) == snapshot


def test_tool_call_random_args_are_normalized_deep_copies():
    rng = random.Random(20261008)
    for _ in range(60):
        args = _random_json(rng) if rng.random() < 0.3 else _random_object(rng)
        before_ordered = _ordered_json(args)
        before_canonical = canonical_json(args)
        call = ToolCall("call_0_0", "search_issues", args)
        assert _ordered_json(args) == before_ordered  # 不修改輸入
        assert canonical_json(call.args) == before_canonical
        assert _ordered_json(call.args) == before_canonical  # key 已遞迴排序
        _assert_keys_sorted_recursively(call.args)
        _scramble(args)
        assert canonical_json(call.args) == before_canonical


# ---------------------------------------------------------------- ToolCall：to_dict / from_dict


def test_tool_call_to_dict_shape():
    call = ToolCall("call_1_2", "update_issue", {"priority": "p0", "issue_id": 4})
    d = call.to_dict()
    assert set(d) == {"id", "name", "args"}
    assert canonical_json(d) == canonical_json(
        {"id": "call_1_2", "name": "update_issue", "args": {"issue_id": 4, "priority": "p0"}}
    )


def test_tool_call_to_dict_is_deep_copy():
    call = ToolCall("c", "update_issue", {"issue_id": 1, "labels": ["bug"]})
    d = call.to_dict()
    assert d["args"] is not call.args
    d["args"]["labels"].append("x")
    d["args"]["issue_id"] = 99
    assert canonical_json(call.args) == '{"issue_id":1,"labels":["bug"]}'


ROUNDTRIP_CALLS = [
    pytest.param("call_0_0", "get_issue", {"issue_id": 1}, id="plain"),
    pytest.param("c", "search_issues", {"z": [1.0, True, None], "a": {"y": 2, "b": "é"}}, id="types_and_order"),
    pytest.param("c", "search_issues", [1, 2.5], id="list_args"),
    pytest.param("c", "close_issue", None, id="none_args"),
    pytest.param("c", "x", 1.0, id="float_args"),
    pytest.param("c", "get_issue", ARGS_AT_LIMIT, id="deep_at_limit"),
]


@pytest.mark.parametrize("call_id, name, args", ROUNDTRIP_CALLS)
def test_tool_call_roundtrip(call_id, name, args):
    call = ToolCall(call_id, name, args)
    direct = ToolCall.from_dict(call.to_dict())
    via_json = ToolCall.from_dict(json.loads(canonical_json(call.to_dict())))
    for back in (direct, via_json):
        assert back == call
        assert canonical_json(back.to_dict()) == canonical_json(call.to_dict())


def test_tool_call_json_roundtrip_keeps_float_distinct_from_int():
    floaty = ToolCall("c", "get_issue", {"issue_id": 1.0})
    inty = ToolCall("c", "get_issue", {"issue_id": 1})
    back = ToolCall.from_dict(json.loads(canonical_json(floaty.to_dict())))
    assert back == inty  # == 被騙
    assert canonical_json(back.to_dict()) == canonical_json(floaty.to_dict())
    assert canonical_json(back.to_dict()) != canonical_json(inty.to_dict())


def test_tool_call_from_dict_normalizes_and_deep_copies():
    data = {"id": "c", "name": "update_issue", "args": {"z": {"b": 1, "a": [2]}, "a": 0}}
    call = ToolCall.from_dict(data)
    assert list(call.args) == ["a", "z"]
    assert list(call.args["z"]) == ["a", "b"]
    snapshot = canonical_json(call.args)
    _scramble(data["args"])
    assert canonical_json(call.args) == snapshot


def _tc_with(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID_TC)
    data.update(overrides)
    return data


def _tc_without(key: str) -> dict[str, Any]:
    data = copy.deepcopy(VALID_TC)
    del data[key]
    return data


TOOL_CALL_BAD_DICTS = [
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param("call", id="str"),
    pytest.param(5, id="int"),
    pytest.param([("id", "c"), ("name", "get_issue"), ("args", {})], id="pairs"),
    pytest.param(_tc_without("id"), id="missing_id"),
    pytest.param(_tc_without("name"), id="missing_name"),
    pytest.param(_tc_without("args"), id="missing_args"),
    pytest.param(_tc_with(extra=1), id="extra_key"),
    pytest.param(_tc_with(type="function"), id="extra_type_key"),
    pytest.param(_tc_with(id=5), id="id_int"),
    pytest.param(_tc_with(id=True), id="id_bool"),
    pytest.param(_tc_with(id=None), id="id_none"),
    pytest.param(_tc_with(id=""), id="id_empty"),
    pytest.param(_tc_with(id="\ud800"), id="id_surrogate"),
    pytest.param(_tc_with(name=5), id="name_int"),
    pytest.param(_tc_with(name=None), id="name_none"),
    pytest.param(_tc_with(name="\ud800"), id="name_surrogate"),
    pytest.param(_tc_with(args=(1,)), id="args_tuple"),
    pytest.param(_tc_with(args={"labels": ("bug",)}), id="args_nested_tuple"),
    pytest.param(_tc_with(args={"s": {1}}), id="args_set"),
    pytest.param(_tc_with(args={1: "x"}), id="args_int_key"),
    pytest.param(_tc_with(args={"x": NAN}), id="args_nan"),
    pytest.param(_tc_with(args={"x": -INF}), id="args_neg_inf"),
    pytest.param(_tc_with(args={"\udc00": 1}), id="args_surrogate_key"),
    pytest.param(_tc_with(args={"x": _nested(MAX_ARGS_DEPTH + 5)}), id="args_too_deep"),
    pytest.param(_tc_with(args=ARGS_PAST_LIMIT), id="args_one_past_limit"),
]


@pytest.mark.parametrize("data", TOOL_CALL_BAD_DICTS)
def test_tool_call_from_dict_always_raises_value_error(data):
    with pytest.raises(ValueError):
        ToolCall.from_dict(data)


# ---------------------------------------------------------------- ModelMessage


def _call(i: int = 0, k: int = 0, **args: Any) -> ToolCall:
    return ToolCall(f"call_{k}_{i}", "get_issue", args or {"issue_id": i + 1})


def test_model_message_accepts_valid_values():
    final = ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn")
    assert final.tool_calls == ()
    no_reason = ModelMessage(text="", tool_calls=(), stop_reason=None)
    assert no_reason.stop_reason is None
    calls = (_call(0), _call(1))
    msg = ModelMessage(text="checking 登入", tool_calls=calls, stop_reason="tool_use")
    assert msg.tool_calls == calls
    assert type(msg.tool_calls) is tuple


# 參數用 factory（lambda）延後建立 ToolCall，避免 collection 時就執行建構驗證。
@pytest.mark.parametrize(
    "make_kwargs",
    [
        pytest.param(lambda: {"text": 5}, id="text_int"),
        pytest.param(lambda: {"text": None}, id="text_none"),
        pytest.param(lambda: {"text": b"x"}, id="text_bytes"),
        pytest.param(lambda: {"tool_calls": [ToolCall("c", "get_issue", {})]}, id="tool_calls_list"),
        pytest.param(lambda: {"tool_calls": []}, id="tool_calls_empty_list"),
        pytest.param(lambda: {"tool_calls": None}, id="tool_calls_none"),
        pytest.param(lambda: {"tool_calls": (dict(VALID_TC),)}, id="element_dict"),
        pytest.param(lambda: {"tool_calls": (None,)}, id="element_none"),
        pytest.param(lambda: {"tool_calls": (("c", "get_issue", {}),)}, id="element_tuple"),
        pytest.param(lambda: {"tool_calls": (_call(0), dict(VALID_TC))}, id="element_dict_after_valid"),
        pytest.param(lambda: {"stop_reason": 5}, id="stop_reason_int"),
        pytest.param(lambda: {"stop_reason": True}, id="stop_reason_bool"),
        pytest.param(lambda: {"stop_reason": b"tool_use"}, id="stop_reason_bytes"),
    ],
)
def test_model_message_type_errors(make_kwargs):
    base: dict[str, Any] = {"text": "", "tool_calls": (), "stop_reason": None}
    base.update(make_kwargs())
    with pytest.raises(TypeError):
        ModelMessage(**base)


@pytest.mark.parametrize(
    "make_kwargs",
    [
        pytest.param(
            lambda: {
                "tool_calls": (ToolCall("dup", "get_issue", {"issue_id": 1}), ToolCall("dup", "add_comment", {}))
            },
            id="duplicate_ids_different_calls",
        ),
        pytest.param(lambda: {"tool_calls": (_call(0), _call(0))}, id="same_call_twice"),
        pytest.param(lambda: {"tool_calls": (_call(0), _call(1), _call(0))}, id="duplicate_not_adjacent"),
        pytest.param(lambda: {"text": "bad \ud800"}, id="text_surrogate"),
        pytest.param(lambda: {"stop_reason": "\udc00"}, id="stop_reason_surrogate"),
    ],
)
def test_model_message_value_errors(make_kwargs):
    base: dict[str, Any] = {"text": "", "tool_calls": (), "stop_reason": None}
    base.update(make_kwargs())
    with pytest.raises(ValueError):
        ModelMessage(**base)


def test_model_message_to_dict_shape():
    calls = (ToolCall("call_0_0", "get_issue", {"issue_id": 1}), ToolCall("call_0_1", "search_issues", {"z": 1, "a": 2}))
    msg = ModelMessage(text="hi", tool_calls=calls, stop_reason="tool_use")
    d = msg.to_dict()
    assert set(d) == {"text", "tool_calls", "stop_reason"}
    assert type(d["tool_calls"]) is list
    assert canonical_json(d) == canonical_json(
        {"text": "hi", "tool_calls": [c.to_dict() for c in calls], "stop_reason": "tool_use"}
    )
    empty = ModelMessage(text="", tool_calls=(), stop_reason=None).to_dict()
    assert canonical_json(empty) == '{"stop_reason":null,"text":"","tool_calls":[]}'
    assert type(empty["tool_calls"]) is list


def test_model_message_to_dict_is_deep_copy():
    msg = ModelMessage(text="", tool_calls=(ToolCall("c", "update_issue", {"labels": ["bug"]}),), stop_reason=None)
    d = msg.to_dict()
    d["tool_calls"][0]["args"]["labels"].append("x")
    d["tool_calls"].append({"id": "z"})
    assert canonical_json(msg.tool_calls[0].args) == '{"labels":["bug"]}'
    assert len(msg.tool_calls) == 1


ROUNDTRIP_MESSAGES = [
    pytest.param(lambda: ModelMessage("", (), None), id="empty"),
    pytest.param(lambda: ModelMessage("Done.", (), "end_turn"), id="final"),
    pytest.param(
        lambda: ModelMessage(
            "登入",
            (
                ToolCall("call_2_0", "get_issue", {"issue_id": 1.0}),
                ToolCall("call_2_1", "update_issue", {"z": True, "a": [1, 1.0]}),
            ),
            "tool_use",
        ),
        id="calls_with_types",
    ),
]


@pytest.mark.parametrize("make_message", ROUNDTRIP_MESSAGES)
def test_model_message_roundtrip(make_message):
    msg = make_message()
    direct = ModelMessage.from_dict(msg.to_dict())
    via_json = ModelMessage.from_dict(json.loads(canonical_json(msg.to_dict())))
    for back in (direct, via_json):
        assert back == msg
        assert type(back.tool_calls) is tuple
        assert canonical_json(back.to_dict()) == canonical_json(msg.to_dict())


def test_model_message_canonical_distinguishes_what_eq_confuses():
    floaty = ModelMessage(text="", tool_calls=(ToolCall("c", "get_issue", {"issue_id": 1.0}),), stop_reason=None)
    inty = ModelMessage(text="", tool_calls=(ToolCall("c", "get_issue", {"issue_id": 1}),), stop_reason=None)
    assert floaty == inty
    assert canonical_json(floaty.to_dict()) != canonical_json(inty.to_dict())
    back = ModelMessage.from_dict(json.loads(canonical_json(floaty.to_dict())))
    assert canonical_json(back.to_dict()) == canonical_json(floaty.to_dict())


def _msg_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"text": "", "tool_calls": [copy.deepcopy(VALID_TC)], "stop_reason": "tool_use"}
    data.update(overrides)
    return data


def _msg_without(key: str) -> dict[str, Any]:
    data = _msg_dict()
    del data[key]
    return data


MODEL_MESSAGE_BAD_DICTS = [
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param("msg", id="str"),
    pytest.param(_msg_without("text"), id="missing_text"),
    pytest.param(_msg_without("tool_calls"), id="missing_tool_calls"),
    pytest.param(_msg_without("stop_reason"), id="missing_stop_reason"),
    pytest.param(_msg_dict(role="assistant"), id="extra_key"),
    pytest.param(_msg_dict(text=5), id="text_int"),
    pytest.param(_msg_dict(text=None), id="text_none"),
    pytest.param(_msg_dict(text="\ud800"), id="text_surrogate"),
    pytest.param(_msg_dict(tool_calls=(copy.deepcopy(VALID_TC),)), id="tool_calls_tuple"),
    pytest.param(_msg_dict(tool_calls=None), id="tool_calls_none"),
    pytest.param(_msg_dict(tool_calls=copy.deepcopy(VALID_TC)), id="tool_calls_dict"),
    pytest.param(_msg_dict(tool_calls="call"), id="tool_calls_str"),
    pytest.param(_msg_dict(tool_calls=["call_0_0"]), id="element_str"),
    pytest.param(_msg_dict(tool_calls=[None]), id="element_none"),
    pytest.param(_msg_dict(tool_calls=[{"id": "c"}]), id="element_missing_keys"),
    pytest.param(_msg_dict(tool_calls=[_tc_with(id=5)]), id="element_id_int"),
    pytest.param(_msg_dict(tool_calls=[_tc_with(args=(1,))]), id="element_args_tuple"),
    pytest.param(_msg_dict(tool_calls=[_tc_with(args={"x": NAN})]), id="element_args_nan"),
    pytest.param(_msg_dict(tool_calls=[copy.deepcopy(VALID_TC), copy.deepcopy(VALID_TC)]), id="duplicate_ids"),
    pytest.param(_msg_dict(stop_reason=5), id="stop_reason_int"),
    pytest.param(_msg_dict(stop_reason="\udfff"), id="stop_reason_surrogate"),
]


@pytest.mark.parametrize("data", MODEL_MESSAGE_BAD_DICTS)
def test_model_message_from_dict_always_raises_value_error(data):
    with pytest.raises(ValueError):
        ModelMessage.from_dict(data)


# ---------------------------------------------------------------- PolicyStep


def _final_message() -> ModelMessage:
    return ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}, id="zeros"),
        pytest.param({"input_tokens": 10**12, "output_tokens": 3, "cost_usd": 0.25}, id="big"),
        pytest.param({"cost_usd": 3}, id="cost_int"),
        pytest.param({"input_tokens": 5, "output_tokens": None, "cost_usd": None}, id="partial"),
    ],
)
def test_policy_step_accepts_valid_usage(kwargs):
    msg = _final_message()
    step = PolicyStep(msg, **kwargs)
    assert step.message is msg or step.message == msg
    for name, value in kwargs.items():
        assert canonical_json(getattr(step, name)) == canonical_json(value)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        pytest.param({"message": None}, TypeError, id="message_none"),
        pytest.param({"message": {"text": "", "tool_calls": [], "stop_reason": None}}, TypeError, id="message_dict"),
        pytest.param({"message": "Done."}, TypeError, id="message_str"),
        pytest.param({"input_tokens": True}, TypeError, id="input_tokens_bool"),
        pytest.param({"input_tokens": 1.0}, TypeError, id="input_tokens_float"),
        pytest.param({"input_tokens": "1"}, TypeError, id="input_tokens_str"),
        pytest.param({"output_tokens": False}, TypeError, id="output_tokens_bool"),
        pytest.param({"output_tokens": 2.0}, TypeError, id="output_tokens_float"),
        pytest.param({"cost_usd": True}, TypeError, id="cost_bool"),
        pytest.param({"cost_usd": "0.1"}, TypeError, id="cost_str"),
        pytest.param({"cost_usd": [0.1]}, TypeError, id="cost_list"),
        pytest.param({"cost_usd": NAN}, ValueError, id="cost_nan"),
        pytest.param({"cost_usd": INF}, ValueError, id="cost_inf"),
        pytest.param({"cost_usd": -INF}, ValueError, id="cost_neg_inf"),
    ],
)
def test_policy_step_validation(kwargs, exc):
    base: dict[str, Any] = {"message": _final_message()}
    base.update(kwargs)
    with pytest.raises(exc):
        PolicyStep(**base)


def test_policy_step_rejects_tool_call_as_message():
    with pytest.raises(TypeError):
        PolicyStep(ToolCall("c", "get_issue", {}))


# ---------------------------------------------------------------- AgentConfig：建構


def test_agent_config_accepts_valid_values():
    cfg = _cfg(system_prompt="", developer_prompt="", model_config={})
    assert cfg.policy == "llm"
    assert cfg.system_prompt == "" and cfg.developer_prompt == ""
    assert cfg.model_config == {}
    unicode_cfg = _cfg(system_prompt="你是分流機器人", model_config={"語言": "中文"})
    assert unicode_cfg.model_config == {"語言": "中文"}


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("policy", 5, id="policy_int"),
        pytest.param("policy", None, id="policy_none"),
        pytest.param("policy", True, id="policy_bool"),
        pytest.param("policy", b"llm", id="policy_bytes"),
        pytest.param("prompt_version", 1, id="prompt_version_int"),
        pytest.param("prompt_version", None, id="prompt_version_none"),
        pytest.param("system_prompt", None, id="system_prompt_none"),
        pytest.param("system_prompt", 0, id="system_prompt_int"),
        pytest.param("system_prompt", ["x"], id="system_prompt_list"),
        pytest.param("developer_prompt", None, id="developer_prompt_none"),
        pytest.param("developer_prompt", b"x", id="developer_prompt_bytes"),
        pytest.param("model_config", None, id="model_config_none"),
        pytest.param("model_config", [], id="model_config_list"),
        pytest.param("model_config", [("t", 1)], id="model_config_pairs"),
        pytest.param("model_config", (), id="model_config_tuple"),
        pytest.param("model_config", "x", id="model_config_str"),
        pytest.param("model_config", {"t": (1, 2)}, id="tuple_value"),
        pytest.param("model_config", {"a": [1, (2,)]}, id="tuple_in_list"),
        pytest.param("model_config", {"a": {"b": ("c",)}}, id="tuple_nested"),
        pytest.param("model_config", {"s": {1, 2}}, id="set_value"),
        pytest.param("model_config", {1: "x"}, id="int_key"),
        pytest.param("model_config", {"a": {None: 1}}, id="nested_none_key"),
        pytest.param("model_config", {"b": b"x"}, id="bytes_value"),
        pytest.param("model_config", {"o": object()}, id="object_value"),
    ],
)
def test_agent_config_type_errors(field, value):
    with pytest.raises(TypeError):
        AgentConfig(**_cfg_kwargs(**{field: value}))


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("policy", "", id="policy_empty"),
        pytest.param("prompt_version", "", id="prompt_version_empty"),
        pytest.param("policy", "\ud800", id="policy_surrogate"),
        pytest.param("prompt_version", "v\udc00", id="prompt_version_surrogate"),
        pytest.param("system_prompt", "\ud800", id="system_prompt_surrogate"),
        pytest.param("developer_prompt", "x\udfff", id="developer_prompt_surrogate"),
        pytest.param("model_config", {"t": NAN}, id="nan"),
        pytest.param("model_config", {"t": INF}, id="inf"),
        pytest.param("model_config", {"t": [-INF]}, id="neg_inf_in_list"),
        pytest.param("model_config", {"t": "\ud800"}, id="surrogate_value"),
        pytest.param("model_config", {"\ud800": 1}, id="surrogate_key"),
        pytest.param("model_config", {"a": {"\udc00": 1}}, id="surrogate_nested_key"),
        pytest.param("model_config", {"a": [{"b": "\udfff"}]}, id="surrogate_deep_value"),
        pytest.param("model_config", {"deep": DEEP}, id="far_too_deep"),
    ],
)
def test_agent_config_value_errors(field, value):
    with pytest.raises(ValueError):
        AgentConfig(**_cfg_kwargs(**{field: value}))


def test_agent_config_model_config_keys_sorted_and_deep_copied():
    model_config = {"z": {"y": [1, {"b": 2, "a": 1}]}, "a": 1.0, "m": True}
    snapshot = canonical_json(model_config)
    cfg = _cfg(model_config=model_config)
    assert cfg.model_config is not model_config
    assert list(cfg.model_config) == ["a", "m", "z"]
    assert list(cfg.model_config["z"]["y"][1]) == ["a", "b"]
    assert _ordered_json(cfg.model_config) == snapshot
    assert list(model_config) == ["z", "a", "m"]  # 輸入不被重排
    _scramble(model_config)
    assert canonical_json(cfg.model_config) == snapshot
    assert canonical_json(cfg.model_config) == '{"a":1.0,"m":true,"z":{"y":[1,{"a":1,"b":2}]}}'


def test_agent_config_random_model_configs_are_normalized():
    rng = random.Random(6006)
    for _ in range(40):
        model_config = _random_object(rng)
        before_ordered = _ordered_json(model_config)
        before_canonical = canonical_json(model_config)
        cfg = _cfg(model_config=model_config)
        assert _ordered_json(model_config) == before_ordered
        assert _ordered_json(cfg.model_config) == before_canonical
        _assert_keys_sorted_recursively(cfg.model_config)
        _scramble(model_config)
        assert canonical_json(cfg.model_config) == before_canonical


# ---------------------------------------------------------------- AgentConfig：config_id


def test_config_id_is_sha256_of_to_dict():
    cfg = _cfg()
    expected = copy.deepcopy(BASE_CFG)
    assert cfg.config_id == sha256_of(cfg.to_dict())
    assert cfg.config_id == sha256_of(expected)
    assert cfg.config_id == HASH_PREFIX + hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param({"t": 1}, {"t": 1.0}, id="int_vs_float"),
        pytest.param({"x": True}, {"x": 1}, id="bool_vs_int"),
        pytest.param({"x": False}, {"x": 0.0}, id="bool_vs_float"),
        pytest.param({"l": [1, True]}, {"l": [1.0, 1]}, id="nested_list"),
    ],
)
def test_config_id_distinguishes_what_eq_confuses(a, b):
    cfg_a = _cfg(model_config=a)
    cfg_b = _cfg(model_config=b)
    assert cfg_a == cfg_b  # Python == 被騙
    assert cfg_a.config_id != cfg_b.config_id


def test_config_id_independent_of_input_key_order():
    a = _cfg(model_config={"b": 1, "a": {"y": 2, "x": 1}})
    b = _cfg(model_config={"a": {"x": 1, "y": 2}, "b": 1})
    assert a.config_id == b.config_id


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"policy": "llm2"}, id="policy"),
        pytest.param({"prompt_version": "react/v2"}, id="prompt_version"),
        pytest.param({"system_prompt": "You are a triage bot. "}, id="system_prompt"),
        pytest.param({"developer_prompt": ""}, id="developer_prompt"),
        pytest.param({"system_prompt": "Use the tools.", "developer_prompt": "You are a triage bot."}, id="swapped"),
        pytest.param({"model_config": {"temperature": 0.0, "model": "m-1"}}, id="model_config_key_removed"),
        pytest.param({"model_config": {"temperature": 0.0, "model": "m-1", "max_tokens": 1024, "x": None}}, id="null_key_added"),
        pytest.param({"model_config": {"temperature": 0, "model": "m-1", "max_tokens": 1024}}, id="float_to_int"),
    ],
)
def test_config_id_changes_with_any_field(overrides):
    assert _cfg(**overrides).config_id != _cfg().config_id


def test_config_id_stable_through_json_roundtrip():
    cfg = _cfg(model_config={"t": 1.0, "flag": True, "stop": ["\n"]})
    back = AgentConfig.from_dict(json.loads(canonical_json(cfg.to_dict())))
    assert back.config_id == cfg.config_id
    assert canonical_json(back.model_config) == '{"flag":true,"stop":["\\n"],"t":1.0}'


# ---------------------------------------------------------------- AgentConfig：to_dict / from_dict


def test_agent_config_to_dict_shape():
    cfg = _cfg(model_config={"z": 1, "a": [1.0]})
    d = cfg.to_dict()
    assert set(d) == set(AGENT_CONFIG_FIELDS)
    assert canonical_json(d) == canonical_json({**BASE_CFG, "model_config": {"a": [1.0], "z": 1}})
    assert json.loads(canonical_json(d)) == d


def test_agent_config_to_dict_is_deep_copy():
    cfg = _cfg(model_config={"stop": ["x"], "nested": {"k": 1}})
    d = cfg.to_dict()
    assert d["model_config"] is not cfg.model_config
    d["model_config"]["stop"].append("y")
    d["model_config"]["nested"]["k"] = 2
    d["model_config"]["new"] = 1
    assert canonical_json(cfg.model_config) == '{"nested":{"k":1},"stop":["x"]}'


@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param({}, id="empty"),
        pytest.param({"temperature": 0.0, "model": "m"}, id="simple"),
        pytest.param({"script": [[{"name": "get_issue", "args": {"issue_id": 1.0}}]], "final_text": "Done."}, id="script_like"),
        pytest.param({"b": True, "a": 1, "c": None, "d": "登入"}, id="mixed"),
        # ScriptedPolicy 對「args 剛好在 MAX_ARGS_DEPTH」的劇本必須產出這種 model_config（比 args 深 4 層），
        # 因此 AgentConfig 的深度上限不得低於它
        pytest.param(_expected_model_config([[("get_issue", ARGS_AT_LIMIT)]]), id="scripted_args_at_limit"),
    ],
)
def test_agent_config_roundtrip(model_config):
    cfg = _cfg(model_config=model_config)
    direct = AgentConfig.from_dict(cfg.to_dict())
    via_json = AgentConfig.from_dict(json.loads(canonical_json(cfg.to_dict())))
    for back in (direct, via_json):
        assert back == cfg
        assert canonical_json(back.to_dict()) == canonical_json(cfg.to_dict())
        assert back.config_id == cfg.config_id


def test_agent_config_from_dict_normalizes_and_deep_copies():
    data = _cfg_kwargs(model_config={"z": [1], "a": {"d": 1, "c": 2}})
    cfg = AgentConfig.from_dict(data)
    assert list(cfg.model_config) == ["a", "z"]
    assert list(cfg.model_config["a"]) == ["c", "d"]
    snapshot = canonical_json(cfg.model_config)
    _scramble(data["model_config"])
    assert canonical_json(cfg.model_config) == snapshot


def _cfg_without(key: str) -> dict[str, Any]:
    data = _cfg_kwargs()
    del data[key]
    return data


AGENT_CONFIG_BAD_DICTS = [
    pytest.param(None, id="none"),
    pytest.param([], id="list"),
    pytest.param("cfg", id="str"),
    *[pytest.param(_cfg_without(k), id=f"missing_{k}") for k in AGENT_CONFIG_FIELDS],
    pytest.param(_cfg_kwargs(config_id="sha256:" + "0" * 64), id="extra_config_id"),
    pytest.param(_cfg_kwargs(format="x"), id="extra_format"),
    pytest.param(_cfg_kwargs(policy=""), id="policy_empty"),
    pytest.param(_cfg_kwargs(policy=5), id="policy_int"),
    pytest.param(_cfg_kwargs(policy=True), id="policy_bool"),
    pytest.param(_cfg_kwargs(prompt_version=""), id="prompt_version_empty"),
    pytest.param(_cfg_kwargs(prompt_version=None), id="prompt_version_none"),
    pytest.param(_cfg_kwargs(system_prompt=None), id="system_prompt_none"),
    pytest.param(_cfg_kwargs(developer_prompt=1), id="developer_prompt_int"),
    pytest.param(_cfg_kwargs(system_prompt="\ud800"), id="system_prompt_surrogate"),
    pytest.param(_cfg_kwargs(model_config=None), id="model_config_none"),
    pytest.param(_cfg_kwargs(model_config=[]), id="model_config_list"),
    pytest.param(_cfg_kwargs(model_config={"t": (1,)}), id="model_config_tuple"),
    pytest.param(_cfg_kwargs(model_config={1: 2}), id="model_config_int_key"),
    pytest.param(_cfg_kwargs(model_config={"t": NAN}), id="model_config_nan"),
    pytest.param(_cfg_kwargs(model_config={"a": {"\udc00": 1}}), id="model_config_surrogate_key"),
]


@pytest.mark.parametrize("data", AGENT_CONFIG_BAD_DICTS)
def test_agent_config_from_dict_always_raises_value_error(data):
    with pytest.raises(ValueError):
        AgentConfig.from_dict(data)


# ---------------------------------------------------------------- ScriptedPolicy：劇本驗證


SCRIPT_TYPE_ERRORS = [
    pytest.param("get_issue", id="script_str"),
    pytest.param(b"get_issue", id="script_bytes"),
    pytest.param(None, id="script_none"),
    pytest.param(5, id="script_int"),
    pytest.param({"turns": [[("get_issue", {})]]}, id="script_dict"),
    pytest.param(iter([[("get_issue", {})]]), id="script_iterator"),
    pytest.param(["get_issue"], id="turn_str"),
    pytest.param([None], id="turn_none"),
    pytest.param([{"name": "get_issue", "args": {}}], id="turn_dict"),
    pytest.param([iter([("get_issue", {})])], id="turn_iterator"),
    pytest.param([[("get_issue", {})], "ab"], id="second_turn_str"),
    pytest.param([["ab"]], id="call_str_len2"),
    pytest.param([[{"name": "get_issue", "args": {}}]], id="call_dict"),
    pytest.param([[None]], id="call_none"),
    pytest.param([[5]], id="call_int"),
    pytest.param([[("get_issue", (1,))]], id="args_tuple"),
    pytest.param([[("update_issue", {"issue_id": 1, "labels": ("bug",)})]], id="args_nested_tuple"),
    pytest.param([[(5, {})]], id="name_int"),
    pytest.param([[(None, {})]], id="name_none"),
    pytest.param([[("get_issue", {"a": {1, 2}})]], id="args_set"),
    pytest.param([[("get_issue", {1: "x"})]], id="args_int_key"),
]

SCRIPT_VALUE_ERRORS = [
    pytest.param([[]], id="empty_turn_list"),
    pytest.param([()], id="empty_turn_tuple"),
    pytest.param([[("get_issue", {})], []], id="second_turn_empty"),
    pytest.param([[()]], id="call_len0"),
    pytest.param([[("get_issue",)]], id="call_len1"),
    pytest.param([[["get_issue"]]], id="call_list_len1"),
    pytest.param([[("get_issue", {}, "extra")]], id="call_len3"),
    pytest.param([[("get_issue", {"x": NAN})]], id="args_nan"),
    pytest.param([[("get_issue", {"x": INF})]], id="args_inf"),
    pytest.param([[("\ud800", {})]], id="name_surrogate"),
    pytest.param([[("get_issue", {"\udc00": 1})]], id="args_surrogate_key"),
    pytest.param([[("get_issue", {"a": {"b": "\udfff"}})]], id="args_surrogate_nested_value"),
    pytest.param([[("get_issue", {"x": _nested(MAX_ARGS_DEPTH + 5)})]], id="args_too_deep"),
    pytest.param([[("get_issue", ARGS_PAST_LIMIT)]], id="args_one_past_limit"),
    pytest.param([[("get_issue", {})], [("search_issues", LIST_ARGS_PAST_LIMIT)]], id="second_turn_one_past_limit"),
]


@pytest.mark.parametrize("script", SCRIPT_TYPE_ERRORS)
def test_scripted_policy_script_type_errors(script):
    with pytest.raises(TypeError):
        ScriptedPolicy(script)


@pytest.mark.parametrize("script", SCRIPT_VALUE_ERRORS)
def test_scripted_policy_script_value_errors(script):
    with pytest.raises(ValueError):
        ScriptedPolicy(script)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        pytest.param({"prompt_version": ""}, ValueError, id="prompt_version_empty"),
        pytest.param({"system_prompt": None}, TypeError, id="system_prompt_none"),
        pytest.param({"developer_prompt": 5}, TypeError, id="developer_prompt_int"),
        pytest.param({"final_text": "bad \ud800"}, ValueError, id="final_text_surrogate"),
    ],
)
def test_scripted_policy_invalid_prompt_settings_surface_by_agent_config(kwargs, exc):
    # 建構或第一次讀 agent_config 時拋出皆可（契約只要求 agent_config 依 AgentConfig 規則）
    with pytest.raises(exc):
        ScriptedPolicy(SCRIPT, **kwargs).agent_config


# ---------------------------------------------------------------- ScriptedPolicy：agent_config


def test_scripted_agent_config_defaults():
    cfg = ScriptedPolicy(SCRIPT).agent_config
    assert isinstance(cfg, AgentConfig)
    expected = {
        "policy": SCRIPTED_POLICY,
        "prompt_version": SCRIPTED_PROMPT_VERSION,
        "system_prompt": "",
        "developer_prompt": "",
        "model_config": _expected_model_config(SCRIPT),
    }
    assert cfg.policy == "scripted"
    assert cfg.prompt_version == "scripted/v1"
    assert set(cfg.model_config) == {"script", "final_text"}
    assert canonical_json(cfg.model_config) == canonical_json(_expected_model_config(SCRIPT))
    assert canonical_json(cfg.to_dict()) == canonical_json(expected)
    assert cfg.config_id == sha256_of(expected)


def test_scripted_agent_config_custom_settings():
    policy = ScriptedPolicy(
        SCRIPT,
        system_prompt="sys 系統",
        developer_prompt="dev",
        prompt_version="exp/v2",
        final_text="All set.",
    )
    cfg = policy.agent_config
    expected = {
        "policy": "scripted",
        "prompt_version": "exp/v2",
        "system_prompt": "sys 系統",
        "developer_prompt": "dev",
        "model_config": _expected_model_config(SCRIPT, final_text="All set."),
    }
    assert canonical_json(cfg.to_dict()) == canonical_json(expected)


def test_scripted_agent_config_model_config_is_normalized():
    cfg = ScriptedPolicy(SCRIPT).agent_config
    _assert_keys_sorted_recursively(cfg.model_config)
    assert _ordered_json(cfg.model_config) == canonical_json(_expected_model_config(SCRIPT))
    update_args = cfg.model_config["script"][1][1]["args"]
    assert list(update_args) == ["issue_id", "labels", "priority"]
    assert list(cfg.model_config["script"][1][1]) == ["args", "name"]


def test_scripted_agent_config_preserves_types_where_eq_is_fooled():
    floaty = ScriptedPolicy(FLOAT_SCRIPT).agent_config
    inty = ScriptedPolicy(INT_SCRIPT).agent_config
    assert floaty == inty
    assert floaty.config_id != inty.config_id
    assert canonical_json(floaty.model_config["script"]) == (
        '[[{"args":{"flag":true,"issue_id":1.0},"name":"get_issue"}]]'
    )


def test_scripted_policy_equivalent_inputs_give_same_config():
    as_tuples = ScriptedPolicy(
        ((("update_issue", {"priority": "p0", "issue_id": 1}),), (("get_issue", {"issue_id": 2}),))
    )
    as_lists = ScriptedPolicy([[["update_issue", {"issue_id": 1, "priority": "p0"}]], [["get_issue", {"issue_id": 2}]]])
    assert as_tuples.agent_config.config_id == as_lists.agent_config.config_id
    assert canonical_json(as_tuples.agent_config.to_dict()) == canonical_json(as_lists.agent_config.to_dict())


BASE_SCRIPT: Any = [[("get_issue", {"issue_id": 1})]]


@pytest.mark.parametrize(
    "script, kwargs",
    [
        pytest.param([[("get_issue", {"issue_id": 2})]], {}, id="script_args"),
        pytest.param([[("get_issue", {"issue_id": 1})], [("get_issue", {"issue_id": 1})]], {}, id="extra_turn"),
        pytest.param([[("get_issue", {"issue_id": 1}), ("get_issue", {"issue_id": 1})]], {}, id="regrouped"),
        pytest.param(BASE_SCRIPT, {"final_text": "Finished."}, id="final_text"),
        pytest.param(BASE_SCRIPT, {"system_prompt": "s"}, id="system_prompt"),
        pytest.param(BASE_SCRIPT, {"developer_prompt": "d"}, id="developer_prompt"),
        pytest.param(BASE_SCRIPT, {"prompt_version": "scripted/v9"}, id="prompt_version"),
    ],
)
def test_scripted_config_id_changes_with_any_setting(script, kwargs):
    base = ScriptedPolicy(BASE_SCRIPT)
    variant = ScriptedPolicy(script, **kwargs)
    assert base.agent_config.config_id != variant.agent_config.config_id


DEEP_VALID_CALLS = [
    pytest.param("get_issue", {"issue_id": 1, "x": _nested(MAX_ARGS_DEPTH - 2)}, id="within_limit"),
    pytest.param("get_issue", ARGS_AT_LIMIT, id="dict_at_limit"),
    pytest.param("search_issues", LIST_ARGS_AT_LIMIT, id="list_at_limit"),
    pytest.param("get_issue", MIXED_ARGS_AT_LIMIT, id="mixed_at_limit"),
]


@pytest.mark.parametrize("name, args", DEEP_VALID_CALLS)
def test_scripted_agent_config_supports_deep_but_valid_args(name, args):
    # 劇本中的 args 只要滿足 check_call_preconditions，agent_config 就必須可建立且可 round-trip
    # （model_config 比 args 多包了 model_config / script / turn / call 四層，邊界取 args 剛好在 MAX_ARGS_DEPTH）
    check_call_preconditions(name, args)
    script = [[(name, args)]]
    cfg = ScriptedPolicy(script).agent_config
    assert canonical_json(cfg.model_config) == canonical_json(_expected_model_config(script))
    rebuilt = [
        AgentConfig(**cfg.to_dict()),
        AgentConfig.from_dict(cfg.to_dict()),
        AgentConfig.from_dict(json.loads(canonical_json(cfg.to_dict()))),
    ]
    for back in rebuilt:
        assert canonical_json(back.to_dict()) == canonical_json(cfg.to_dict())
        assert back.config_id == cfg.config_id
    assert oracle_policy([(name, args)]).agent_config.config_id == cfg.config_id


@pytest.mark.parametrize("name, args", DEEP_VALID_CALLS)
def test_next_message_plays_deep_but_valid_args(name, args):
    script = [[(name, args)]]
    step = ScriptedPolicy(script).next_message(_conversation(0), _tools())
    assert _message_json(step.message) == _expected_message_json(script, 0)
    back = ModelMessage.from_dict(json.loads(canonical_json(step.message.to_dict())))
    assert _message_json(back) == _message_json(step.message)


# ---------------------------------------------------------------- ScriptedPolicy：next_message


@pytest.mark.parametrize("k", range(len(SCRIPT) + 3))
def test_next_message_plays_turn_k(k):
    policy = ScriptedPolicy(SCRIPT)
    step = policy.next_message(_conversation(k, SCRIPT), _tools())
    _assert_scripted_usage(step)
    assert _message_json(step.message) == _expected_message_json(SCRIPT, k)
    if k < len(SCRIPT):
        assert step.message.text == ""
        assert step.message.stop_reason == "tool_use"
        assert [c.id for c in step.message.tool_calls] == [f"call_{k}_{i}" for i in range(len(SCRIPT[k]))]
    else:
        assert step.message.text == "Done."
        assert step.message.tool_calls == ()
        assert step.message.stop_reason == "end_turn"


def test_next_message_multi_call_turn_ids():
    policy = ScriptedPolicy(SCRIPT)
    step = policy.next_message(_conversation(1, SCRIPT), _tools())
    assert [c.id for c in step.message.tool_calls] == ["call_1_0", "call_1_1", "call_1_2"]
    assert [c.name for c in step.message.tool_calls] == ["get_issue", "update_issue", "add_comment"]
    assert list(step.message.tool_calls[1].args) == ["issue_id", "labels", "priority"]


def test_next_message_final_uses_custom_final_text():
    policy = ScriptedPolicy(SCRIPT, final_text="全部完成")
    step = policy.next_message(_conversation(len(SCRIPT), SCRIPT), _tools())
    assert _message_json(step.message) == canonical_json(
        {"text": "全部完成", "tool_calls": [], "stop_reason": "end_turn"}
    )
    _assert_scripted_usage(step)


def test_next_message_preserves_arg_types_where_eq_is_fooled():
    floaty = ScriptedPolicy(FLOAT_SCRIPT).next_message(_conversation(0), _tools())
    inty = ScriptedPolicy(INT_SCRIPT).next_message(_conversation(0), _tools())
    assert floaty.message == inty.message
    assert canonical_json(floaty.message.tool_calls[0].args) == '{"flag":true,"issue_id":1.0}'
    assert _message_json(floaty.message) != _message_json(inty.message)


def test_next_message_non_dict_args_are_played():
    policy = ScriptedPolicy([[("search_issues", ["open", 1.0])], [("get_issue", None)]])
    first = policy.next_message(_conversation(0), _tools())
    assert canonical_json(first.message.tool_calls[0].args) == '["open",1.0]'
    second = policy.next_message([{"role": "assistant"}], _tools())
    assert second.message.tool_calls[0].args is None


ASSISTANT = {"role": "assistant", "content": "", "tool_calls": []}


@pytest.mark.parametrize(
    "messages, k",
    [
        pytest.param([], 0, id="empty"),
        pytest.param([{"role": "assistant"}], 1, id="minimal_assistant"),
        pytest.param([{"role": "assistant"}, {"role": "assistant"}], 2, id="two_minimal"),
        pytest.param([{"role": "user", "content": "x"}] * 5, 0, id="only_users"),
        pytest.param(
            [{"role": "tool", "tool_call_id": "t", "content": "{}"}] * 4 + [dict(ASSISTANT)], 1, id="tools_then_assistant"
        ),
        pytest.param(
            [
                {"role": "system", "content": ""},
                dict(ASSISTANT),
                {"role": "user", "content": "more"},
                {"role": "developer", "content": ""},
                dict(ASSISTANT),
            ],
            2,
            id="interleaved",
        ),
        pytest.param([{"role": "Assistant"}, {"role": "assistant_"}, {"role": ""}], 0, id="near_miss_roles"),
        pytest.param([{"role": "assistant"}] * 3, 3, id="exactly_script_length"),
        pytest.param([{"role": "assistant"}] * 7, 7, id="far_past_end"),
    ],
)
def test_next_message_position_counts_only_assistant_messages(messages, k):
    policy = ScriptedPolicy(SCRIPT)
    step = policy.next_message(messages, _tools())
    assert _message_json(step.message) == _expected_message_json(SCRIPT, k)


def test_next_message_ignores_tools_argument():
    policy = ScriptedPolicy(SCRIPT)
    conv = _conversation(1, SCRIPT)
    with_specs = policy.next_message(conv, _tools())
    without = policy.next_message(conv, ())
    assert _message_json(with_specs.message) == _message_json(without.message)


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(None, id="none"),
        pytest.param(({"role": "user", "content": "x"},), id="tuple"),
        pytest.param({"role": "user", "content": "x"}, id="dict"),
        pytest.param("assistant", id="str"),
        pytest.param([{"role": "user", "content": "x"}, "assistant"], id="element_str"),
        pytest.param([["role", "assistant"]], id="element_list"),
        pytest.param([("role", "assistant")], id="element_tuple"),
        pytest.param([None], id="element_none"),
        pytest.param([{"content": "x"}], id="missing_role"),
        pytest.param([{"role": None}], id="role_none"),
        pytest.param([{"role": 1}], id="role_int"),
        pytest.param([{"role": b"assistant"}], id="role_bytes"),
        pytest.param([{"role": "assistant"}, {"role": ["assistant"]}], id="later_role_list"),
    ],
)
def test_next_message_rejects_malformed_messages(messages):
    policy = ScriptedPolicy(SCRIPT)
    with pytest.raises(TypeError):
        policy.next_message(messages, _tools())


def test_next_message_does_not_mutate_inputs():
    policy = ScriptedPolicy(SCRIPT)
    messages = _conversation(1, SCRIPT)
    messages.append({"role": "user", "z": 1.0, "content": "unsorted keys"})
    tools = _tools()
    before = (_ordered_json(messages), canonical_json(messages), canonical_json(tools))
    policy.next_message(messages, tools)
    policy.next_message(messages, tools)
    assert (_ordered_json(messages), canonical_json(messages), canonical_json(tools)) == before


def test_next_message_is_stateless_and_repeatable():
    policy = ScriptedPolicy(SCRIPT)
    conv = _conversation(1, SCRIPT)
    first = policy.next_message(conv, _tools())
    second = policy.next_message(conv, _tools())
    assert first is not second
    assert first == second
    assert _message_json(first.message) == _message_json(second.message)


def test_next_message_rewinds_with_earlier_conversation():
    policy = ScriptedPolicy(SCRIPT)
    order = [2, 0, 5, 1, 1, 0, 3]
    for k in order:
        step = policy.next_message(_conversation(k, SCRIPT), _tools())
        assert _message_json(step.message) == _expected_message_json(SCRIPT, k)
        _assert_scripted_usage(step)


def test_next_message_returns_fresh_objects():
    policy = ScriptedPolicy(SCRIPT)
    conv = _conversation(1, SCRIPT)
    first = policy.next_message(conv, _tools())
    second = policy.next_message(conv, _tools())
    assert first.message.tool_calls[1].args is not second.message.tool_calls[1].args
    first.message.tool_calls[1].args["issue_id"] = 999
    first.message.tool_calls[1].args["labels"].append("hacked")
    third = policy.next_message(conv, _tools())
    assert _message_json(third.message) == _expected_message_json(SCRIPT, 1)
    assert canonical_json(policy.agent_config.model_config) == canonical_json(_expected_model_config(SCRIPT))


def test_agent_config_returns_fresh_objects():
    policy = ScriptedPolicy(SCRIPT)
    a = policy.agent_config
    b = policy.agent_config
    assert a is not b
    assert a.model_config is not b.model_config
    a.model_config["script"][0][0]["args"]["status"] = "closed"
    a.model_config["script"].append([{"name": "get_issue", "args": {}}])
    a.model_config["final_text"] = "hacked"
    c = policy.agent_config
    assert canonical_json(c.model_config) == canonical_json(_expected_model_config(SCRIPT))
    assert c.config_id == b.config_id
    for k in (0, len(SCRIPT)):
        step = policy.next_message(_conversation(k, SCRIPT), _tools())
        assert _message_json(step.message) == _expected_message_json(SCRIPT, k)


def test_scripted_policy_deep_copies_script():
    script = [[["update_issue", {"issue_id": 1, "labels": ["bug"]}]], [("get_issue", {"issue_id": 2})]]
    original = copy.deepcopy(script)
    policy = ScriptedPolicy(script)
    script[0][0][1]["labels"].append("hacked")
    script[0][0][0] = "add_comment"
    script[1].append(("get_issue", {"issue_id": 3}))
    script.append([("get_issue", {"issue_id": 4})])
    assert canonical_json(policy.agent_config.model_config) == canonical_json(_expected_model_config(original))
    assert policy.agent_config.config_id == ScriptedPolicy(original).agent_config.config_id
    for k in range(len(original) + 1):
        step = policy.next_message(_conversation(k, original), _tools())
        assert _message_json(step.message) == _expected_message_json(original, k)


def test_scripted_policy_does_not_mutate_script():
    script = [[("update_issue", {"priority": "p0", "issue_id": 1, "labels": ["z", "a"]})]]
    before = _ordered_json(script)
    policy = ScriptedPolicy(script)
    policy.agent_config
    policy.next_message(_conversation(0), _tools())
    assert _ordered_json(script) == before


def test_independent_scripted_policies_agree():
    a = ScriptedPolicy(SCRIPT)
    b = ScriptedPolicy(copy.deepcopy(SCRIPT))
    for k in range(len(SCRIPT) + 1):
        conv = _conversation(k, SCRIPT)
        assert _message_json(a.next_message(conv, _tools()).message) == _message_json(
            b.next_message(conv, _tools()).message
        )


# ---------------------------------------------------------------- oracle_policy


def test_oracle_policy_one_call_per_turn():
    calls = REFERENCE_SOLUTIONS["smoke-reassign-departed"]
    as_script = [[call] for call in calls]
    policy = oracle_policy(calls)
    assert isinstance(policy, ScriptedPolicy)
    assert canonical_json(policy.agent_config.model_config) == canonical_json(_expected_model_config(as_script))
    for k, (name, args) in enumerate(calls):
        step = policy.next_message(_conversation(k, as_script), _tools())
        assert len(step.message.tool_calls) == 1
        (call,) = step.message.tool_calls
        assert call.id == f"call_{k}_0"
        assert call.name == name
        assert canonical_json(call.args) == canonical_json(args)
        assert step.message.stop_reason == "tool_use"
    final = policy.next_message(_conversation(len(calls), as_script), _tools())
    assert _message_json(final.message) == _expected_message_json(as_script, len(calls))


def test_oracle_policy_matches_equivalent_scripted_policy():
    calls = list(REFERENCE_SOLUTIONS["smoke-close-duplicate"])
    kwargs = {"system_prompt": "s", "developer_prompt": "d", "prompt_version": "oracle/v1", "final_text": "ok"}
    via_oracle = oracle_policy(calls, **kwargs)
    via_tuple = oracle_policy(tuple(calls), **kwargs)
    direct = ScriptedPolicy([[call] for call in calls], **kwargs)
    assert via_oracle.agent_config.config_id == direct.agent_config.config_id
    assert via_tuple.agent_config.config_id == direct.agent_config.config_id


def test_oracle_policy_forwards_kwargs():
    policy = oracle_policy(
        [("get_issue", {"issue_id": 5})],
        system_prompt="sys",
        developer_prompt="dev",
        prompt_version="oracle/v1",
        final_text="Nothing to do.",
    )
    cfg = policy.agent_config
    assert cfg.policy == SCRIPTED_POLICY
    assert (cfg.system_prompt, cfg.developer_prompt, cfg.prompt_version) == ("sys", "dev", "oracle/v1")
    assert cfg.model_config["final_text"] == "Nothing to do."
    final = policy.next_message([{"role": "assistant"}], _tools())
    assert final.message.text == "Nothing to do."


def test_oracle_policy_defaults_match_scripted_defaults():
    cfg = oracle_policy([("get_issue", {"issue_id": 5})]).agent_config
    assert cfg.prompt_version == SCRIPTED_PROMPT_VERSION
    assert cfg.system_prompt == "" and cfg.developer_prompt == ""
    assert cfg.model_config["final_text"] == "Done."


@pytest.mark.parametrize("calls", [pytest.param([], id="list"), pytest.param((), id="tuple")])
def test_oracle_policy_rejects_empty(calls):
    with pytest.raises(ValueError):
        oracle_policy(calls)


@pytest.mark.parametrize(
    "calls, exc",
    [
        pytest.param([("get_issue", (1,))], TypeError, id="args_tuple"),
        pytest.param([("get_issue",)], ValueError, id="call_len1"),
        pytest.param([("get_issue", {"x": NAN})], ValueError, id="args_nan"),
        pytest.param([("get_issue", ARGS_PAST_LIMIT)], ValueError, id="args_one_past_limit"),
    ],
)
def test_oracle_policy_validates_calls_like_scripted_policy(calls, exc):
    with pytest.raises(exc):
        oracle_policy(calls)


# ---------------------------------------------------------------- 與 env / grader 的整合（手工迴圈，不依賴 runner）


def _drive(policy: Any, task: Any, max_turns: int = 30) -> tuple[IssueTrackerEnv, list[PolicyStep]]:
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    cfg = policy.agent_config
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "developer", "content": cfg.developer_prompt},
        {"role": "user", "content": task.instruction},
    ]
    steps: list[PolicyStep] = []
    for _ in range(max_turns):
        step = policy.next_message(copy.deepcopy(messages), _tools())
        steps.append(step)
        msg = step.message
        messages.append({"role": "assistant", "content": msg.text, "tool_calls": [c.to_dict() for c in msg.tool_calls]})
        if not msg.tool_calls:
            break
        for call in msg.tool_calls:
            observation = env.call_tool(call.name, call.args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": canonical_json(observation)})
    return env, steps


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_oracle_policy_solves_smoke_task(task_id):
    task = get_smoke_task(task_id)
    reference = REFERENCE_SOLUTIONS[task_id]
    env, steps = _drive(oracle_policy(reference), task)
    assert len(steps) == len(reference) + 1
    assert all(len(s.message.tool_calls) == 1 for s in steps[:-1])
    assert steps[-1].message.stop_reason == "end_turn"
    assert canonical_json([[r.tool, r.args] for r in env.records]) == canonical_json([list(c) for c in reference])
    report = grade(task, env.records, env.current_state())
    assert report.success and report.trajectory_safe


def test_single_turn_scripted_policy_executes_all_calls_in_order():
    task_id = "smoke-reassign-departed"
    task = get_smoke_task(task_id)
    reference = REFERENCE_SOLUTIONS[task_id]
    env, steps = _drive(ScriptedPolicy([list(reference)]), task)
    assert len(steps) == 2
    assert [c.id for c in steps[0].message.tool_calls] == [f"call_0_{i}" for i in range(len(reference))]
    assert canonical_json([[r.tool, r.args] for r in env.records]) == canonical_json([list(c) for c in reference])
    assert grade(task, env.records, env.current_state()).success


# ---------------------------------------------------------------- MAX_MODEL_CONFIG_DEPTH（主 session 補充）


def _nested_list(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


def test_model_config_depth_limit_constant_exceeds_scripted_nesting():
    from agents.policy import MAX_MODEL_CONFIG_DEPTH
    from env.environment import MAX_ARGS_DEPTH

    assert MAX_MODEL_CONFIG_DEPTH > MAX_ARGS_DEPTH + 4


def test_model_config_within_depth_limit_is_accepted():
    from agents.policy import MAX_MODEL_CONFIG_DEPTH

    config = AgentConfig("p", "v1", "", "", {"x": _nested_list(MAX_MODEL_CONFIG_DEPTH - 2)})
    assert config.config_id.startswith("sha256:")


@pytest.mark.parametrize("extra", [5, 1000])
def test_model_config_beyond_depth_limit_raises_value_error(extra):
    from agents.policy import MAX_MODEL_CONFIG_DEPTH

    with pytest.raises(ValueError):
        AgentConfig("p", "v1", "", "", {"x": _nested_list(MAX_MODEL_CONFIG_DEPTH + extra)})
