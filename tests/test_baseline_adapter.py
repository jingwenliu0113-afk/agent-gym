"""agents.baseline 的單元契約測試（decision log 009，含實作前的契約審查修訂）。

範圍：固定常數與 baseline_agent_config()、build_request（request 形狀、betas / thinking、歷史訊息的一致性檢查）、
parse_response（stop_reason 規則、block 白名單、input_transformations、PolicyStep.meta、含 1h cache 的成本公式）、
AnthropicPolicy（建構檢查、client 屬性檢查、beta namespace）、make_anthropic_client，以及（importorskip）與已安裝
anthropic SDK 的 beta 型別 / 參數相容性。

依據：docs/decisions/009-baseline-agent-config.md 與 agents/baseline.py 的模組 / 函式 docstring。
- 不呼叫真正的 API、不連網、不建構任何真正的 SDK client、不讀取或要求任何 credential。SDK 測試只檢查簽名 / 型別，
  或以 ``BetaMessage.model_validate`` 建立離線的 response 物件。
- client 一律使用 tests/fake_anthropic.py 的 FakeClient（或本檔內的 fake 物件）。
- 不斷言未規定的細節（例外訊息、例外之間的檢查順序除非契約有列出等）。
- 「逐 byte 相同」以 canonical_json 比較，並放入 Python ``==`` 會被騙過的案例（1 vs 1.0、True vs 1）。
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
import types
import typing
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from agents.baseline import (
    ADAPTER_VERSION,
    ALLOWED_BLOCK_TYPES,
    ANTHROPIC_POLICY,
    BASELINE_DEVELOPER_PROMPT,
    BASELINE_MAX_TURNS,
    BASELINE_MODEL_CONFIG,
    BASELINE_PROMPT_VERSION,
    BASELINE_SYSTEM_PROMPT,
    MAX_EPISODE_ATTEMPTS,
    PRICING,
    AdapterError,
    AnthropicPolicy,
    ReplayIntegrityError,
    baseline_agent_config,
    build_request,
    make_anthropic_client,
    parse_response,
)
from agents.policy import SCRIPTED_POLICY, AgentConfig, ModelMessage, PolicyStep, ToolCall
from apps.issue_tracker.tools import TOOL_SPECS
from env.hashing import canonical_json, sha256_of
from tests.fake_anthropic import (
    FakeBlock,
    FakeCacheCreation,
    FakeClient,
    FakeMessages,
    FakeResponse,
    FakeStopDetails,
    FakeUsage,
    final_turn,
    redacted_thinking_block,
    text_block,
    thinking_block,
    tool_turn,
    tool_use_block,
)

MODEL = "claude-opus-5-5"
BETA = "thinking-binding-controls-2026-08-01"
SYS = "SYS-PROMPT: you are a test agent."
DEV = "DEV-PROMPT: operating notes."
INSTR = "Close #1 and tell u2."

# decision 009 第 1 點的固定值（以字面值寫死，避免常數漂移時測試跟著漂移）
EXPECTED_THINKING: dict[str, Any] = {
    "type": "adaptive",
    "display": "summarized",
    "block_binding": {"prefix_mismatch_behavior": "error"},
}
EXPECTED_MODEL_CONFIG: dict[str, Any] = {
    "provider": "anthropic",
    "adapter": "anthropic-messages/v1",
    "model": "claude-opus-5-5",
    "max_tokens": 16000,
    "thinking": EXPECTED_THINKING,
    "betas": ["thinking-binding-controls-2026-08-01"],
    "effort": "medium",
    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    "prompt_caching": "auto",
    "fallbacks": None,
    "client": {"max_retries": 4, "timeout_s": 600},
}
EXPECTED_PRICING: dict[str, float] = {
    "input": 4.00,
    "output": 20.00,
    "cache_write": 5.00,
    "cache_write_1h": 8.00,
    "cache_read": 0.20,
}

REQUEST_KEYS = {
    "model",
    "max_tokens",
    "system",
    "messages",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
    "cache_control",
    "betas",
}
META_KEYS = {"response_model", "request_id", "stop_reason", "stop_details", "tool_use_blocks", "usage"}
META_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_creation_1h_input_tokens",
}

OK_OBS = canonical_json({"ok": True, "data": {"id": 1, "status": "open"}})
ERR_OBS = canonical_json({"ok": False, "error": {"code": "not_found", "message": "issue 99 not found"}})
OK_OBS_2 = canonical_json({"ok": True, "data": {"id": 7, "issue_id": 1, "author_id": "bot", "body": "hi"}})

# 固定的 golden 值（由本檔依契約手工組出的 expected 計算後寫死；任何 adapter / config / prompt 的變更都會改變它們，
# 必須同時升 ADAPTER_VERSION 或 prompt_version 並另立 decision log）。
GOLDEN_REQUEST_SHA256 = "sha256:77b2d9538b595cc0ee9dcdc063ce9f57ece6799b5cff4184a92b761ecc481286"
GOLDEN_BASELINE_CONFIG_ID = "sha256:42087c4d5908eb5465afc1bb4e75339a681046fdefd55ea856a446f53e6cd83e"


# ---------------------------------------------------------------- helpers


def _base(system: str = SYS, developer: str = DEV, instruction: str = INSTR) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system},
        {"role": "developer", "content": developer},
        {"role": "user", "content": instruction},
    ]


def _assistant(text: Any, calls: list[tuple[str, str, Any]], provider_content: Any) -> dict[str, Any]:
    """render_messages v2 形狀的 assistant 訊息（agents/policy.py 模組 docstring）；輸入一律深拷貝。"""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": cid, "name": name, "args": copy.deepcopy(args)} for cid, name, args in calls],
        "provider_content": copy.deepcopy(provider_content),
    }


def _tool(call_id: str, content: Any) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant_from_message(message: ModelMessage) -> dict[str, Any]:
    """依 render_messages v2 的規則把 ModelMessage 轉成 conversation 中的 assistant 訊息。"""
    return {
        "role": "assistant",
        "content": message.text,
        "tool_calls": [call.to_dict() for call in message.tool_calls],
        "provider_content": copy.deepcopy(message.provider_content),
    }


def _expected_tools(specs: Any = TOOL_SPECS) -> list[dict[str, Any]]:
    return [
        {"name": s["name"], "description": s["description"], "input_schema": copy.deepcopy(s["input_schema"])}
        for s in specs
    ]


def _expected_request(
    messages: list[dict[str, Any]], *, system: str = SYS, developer: str | None = DEV, tools: Any = TOOL_SPECS
) -> dict[str, Any]:
    sys_blocks = [{"type": "text", "text": system}]
    if developer is not None:
        sys_blocks.append({"type": "text", "text": developer})
    return {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": sys_blocks,
        "messages": messages,
        "tools": _expected_tools(tools),
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "thinking": copy.deepcopy(EXPECTED_THINKING),
        "output_config": {"effort": "medium"},
        "cache_control": {"type": "ephemeral"},
        "betas": ["thinking-binding-controls-2026-08-01"],
    }


def _user_instr(instruction: str = INSTR) -> dict[str, Any]:
    """修訂後的契約：第一則 user 訊息的 content 就是 instruction 字串本身。"""
    return {"role": "user", "content": instruction}


def _results(*items: tuple[str, str, bool]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": content, "is_error": is_error}
            for cid, content, is_error in items
        ],
    }


def _assert_json_native(value: Any, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _assert_json_native(item, f"{path}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str, f"non-str key at {path}"
            _assert_json_native(item, f"{path}.{key}")
        return
    raise AssertionError(f"non-JSON-native value of type {type(value).__name__} at {path}")


def _baseline_mc() -> dict[str, Any]:
    return copy.deepcopy(EXPECTED_MODEL_CONFIG)


def _cfg(
    *,
    policy: str = ANTHROPIC_POLICY,
    prompt_version: str = BASELINE_PROMPT_VERSION,
    system_prompt: str = BASELINE_SYSTEM_PROMPT,
    developer_prompt: str = BASELINE_DEVELOPER_PROMPT,
    model_config: dict[str, Any] | None = None,
) -> AgentConfig:
    return AgentConfig(
        policy=policy,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        model_config=_baseline_mc() if model_config is None else model_config,
    )


# provider 原生 blocks（白名單欄位）
def _th(signature: str, thinking: str = "") -> dict[str, Any]:
    return {"type": "thinking", "thinking": thinking, "signature": signature}


def _rd(data: str) -> dict[str, Any]:
    return {"type": "redacted_thinking", "data": data}


def _tx(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tu(tool_id: str, name: str, tool_input: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


TEXT_1 = "Checking."
CALLS_1: list[tuple[str, str, Any]] = [("toolu_01", "get_issue", {"issue_id": 1})]
PROVIDER_1 = [_th("sig-1"), _tx("Checking."), _tu("toolu_01", "get_issue", {"issue_id": 1})]

TEXT_2 = ""
CALLS_2: list[tuple[str, str, Any]] = [("toolu_02", "update_issue", {"issue_id": 1, "status": "closed"})]
PROVIDER_2 = [
    _rd("opaque-blob"),
    _th("sig-2", "need to close it"),
    _tu("toolu_02", "update_issue", {"issue_id": 1, "status": "closed"}),
]

# 一則訊息兩個 tool_use、text 分散在 tool_use 前後（多 tool turn；decision 009 第 4 點）
TEXT_3 = "Closing and commenting."
CALLS_3: list[tuple[str, str, Any]] = [
    ("toolu_03", "update_issue", {"issue_id": 1, "status": "closed"}),
    ("toolu_04", "add_comment", {"issue_id": 1, "body": "hi"}),
]
PROVIDER_3 = [
    _th("sig-3"),
    _tx("Closing "),
    _tu("toolu_03", "update_issue", {"issue_id": 1, "status": "closed"}),
    _tx("and commenting."),
    _tu("toolu_04", "add_comment", {"issue_id": 1, "body": "hi"}),
]


def _a1() -> dict[str, Any]:
    return _assistant(TEXT_1, CALLS_1, PROVIDER_1)


def _a2() -> dict[str, Any]:
    return _assistant(TEXT_2, CALLS_2, PROVIDER_2)


def _a3() -> dict[str, Any]:
    return _assistant(TEXT_3, CALLS_3, PROVIDER_3)


def _provider_conversation() -> list[dict[str, Any]]:
    return _base() + [_a1(), _tool("toolu_01", OK_OBS), _a2(), _tool("toolu_02", ERR_OBS)]


def _multi_conversation() -> list[dict[str, Any]]:
    return _base() + [
        _a1(),
        _tool("toolu_01", OK_OBS),
        _a3(),
        _tool("toolu_03", OK_OBS),
        _tool("toolu_04", OK_OBS_2),
    ]


CONVERSATIONS: list[Callable[[], list[dict[str, Any]]]] = [_base, _provider_conversation, _multi_conversation]


def _single(provider_content: Any, *, text: Any = TEXT_1, calls: list[tuple[str, str, Any]] | None = None) -> list[dict[str, Any]]:
    """base + 一則 assistant（給定 provider_content）+ 每個 call 一則 OK 的 tool 訊息。"""
    calls = CALLS_1 if calls is None else calls
    return _base() + [_assistant(text, calls, provider_content)] + [_tool(cid, OK_OBS) for cid, _, _ in calls]


def _p1_with(index: int, block: Any) -> list[Any]:
    """PROVIDER_1 的第 index 個 block（0 thinking、1 text、2 tool_use）換成 block。"""
    pc: list[Any] = copy.deepcopy(PROVIDER_1)
    pc[index] = block
    return pc


def _multi(
    provider_content: Any = None, *, text: Any = TEXT_3, calls: list[tuple[str, str, Any]] | None = None,
    results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pc = PROVIDER_3 if provider_content is None else provider_content
    calls = CALLS_3 if calls is None else calls
    if results is None:
        results = [_tool("toolu_03", OK_OBS), _tool("toolu_04", OK_OBS_2)]
    return _base() + [_a1(), _tool("toolu_01", OK_OBS), _assistant(text, calls, pc)] + results


def _resp(content: list[Any], stop_reason: Any, usage: Any = None, model: str = MODEL, **kwargs: Any) -> FakeResponse:
    return FakeResponse(content=content, stop_reason=stop_reason, usage=usage or FakeUsage(10, 5), model=model, **kwargs)


# ---------------------------------------------------------------- constants & baseline_agent_config


def test_constants_match_decision_009() -> None:
    assert ANTHROPIC_POLICY == "anthropic-messages"
    assert ADAPTER_VERSION == "anthropic-messages/v1"
    assert BASELINE_PROMPT_VERSION == "baseline/v1"
    assert type(BASELINE_MAX_TURNS) is int and BASELINE_MAX_TURNS == 20
    assert type(MAX_EPISODE_ATTEMPTS) is int and MAX_EPISODE_ATTEMPTS == 2
    assert canonical_json(BASELINE_MODEL_CONFIG) == canonical_json(EXPECTED_MODEL_CONFIG)


def test_model_config_carries_adapter_version_and_600s_timeout() -> None:
    """修訂：adapter 版本、display、block_binding、betas 都在 model_config 內（因此在 config_id 內）；timeout 600 秒。"""
    assert BASELINE_MODEL_CONFIG["adapter"] == ADAPTER_VERSION
    assert BASELINE_MODEL_CONFIG["thinking"] == EXPECTED_THINKING
    assert BASELINE_MODEL_CONFIG["betas"] == [BETA]
    assert BASELINE_MODEL_CONFIG["client"] == {"max_retries": 4, "timeout_s": 600}


def test_baseline_model_config_has_no_sampling_parameters() -> None:
    """decision 009：不送 temperature / top_p / top_k；不啟用 fallbacks。"""
    for key in ("temperature", "top_p", "top_k", "top_a", "seed"):
        assert key not in BASELINE_MODEL_CONFIG
    assert BASELINE_MODEL_CONFIG["fallbacks"] is None


def test_pricing_matches_decision_009() -> None:
    assert set(PRICING) >= {"claude-opus-5-5"}
    assert PRICING["claude-opus-5-5"] == EXPECTED_PRICING


def test_allowed_block_types_whitelist() -> None:
    assert ALLOWED_BLOCK_TYPES == {
        "text": ("type", "text"),
        "thinking": ("type", "thinking", "signature"),
        "redacted_thinking": ("type", "data"),
        "tool_use": ("type", "id", "name", "input"),
    }


def test_error_classes_are_distinct() -> None:
    """AdapterError（agent_output，不重試）與 ReplayIntegrityError（harness，實驗停止）的分類不同，彼此不可為子類。"""
    assert issubclass(AdapterError, Exception)
    assert issubclass(ReplayIntegrityError, Exception)
    assert not issubclass(AdapterError, ReplayIntegrityError)
    assert not issubclass(ReplayIntegrityError, AdapterError)


def test_prompts_are_nonempty_strings() -> None:
    assert isinstance(BASELINE_SYSTEM_PROMPT, str) and BASELINE_SYSTEM_PROMPT.strip()
    assert isinstance(BASELINE_DEVELOPER_PROMPT, str) and BASELINE_DEVELOPER_PROMPT.strip()


def test_developer_prompt_states_non_interactive_run() -> None:
    """decision 009 第 2 點：說明「非互動執行：沒有人會回答問題，不呼叫 tool 的回覆就結束任務」這個環境機制事實。"""
    assert "non-interactive" in BASELINE_DEVELOPER_PROMPT
    assert "nobody will answer questions" in BASELINE_DEVELOPER_PROMPT
    assert "a reply without a tool call ends the task" in BASELINE_DEVELOPER_PROMPT


def test_prompts_contain_no_safety_instructions() -> None:
    """decision 009 第 2 點：不加入安全或「不要亂改」之類的指示（以關鍵字粗略檢查）。"""
    for prompt in (BASELINE_SYSTEM_PROMPT, BASELINE_DEVELOPER_PROMPT):
        lowered = prompt.lower()
        for word in ("safe", "careful", "caution", "destructive", "irreversible", "confirm", "permission"):
            assert word not in lowered


def test_baseline_agent_config_exact_values() -> None:
    cfg = baseline_agent_config()
    assert isinstance(cfg, AgentConfig)
    assert cfg.policy == ANTHROPIC_POLICY == "anthropic-messages"
    assert cfg.prompt_version == BASELINE_PROMPT_VERSION == "baseline/v1"
    assert cfg.system_prompt == BASELINE_SYSTEM_PROMPT
    assert cfg.developer_prompt == BASELINE_DEVELOPER_PROMPT
    assert canonical_json(cfg.model_config) == canonical_json(BASELINE_MODEL_CONFIG)
    assert canonical_json(cfg.model_config) == canonical_json(EXPECTED_MODEL_CONFIG)


def test_baseline_agent_config_id_matches_manual_construction() -> None:
    manual = AgentConfig(
        policy="anthropic-messages",
        prompt_version="baseline/v1",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        developer_prompt=BASELINE_DEVELOPER_PROMPT,
        model_config=copy.deepcopy(EXPECTED_MODEL_CONFIG),
    )
    assert baseline_agent_config().config_id == manual.config_id


def test_baseline_config_id_is_pinned() -> None:
    """golden：prompt 文字或 model_config 的任何變更都會改變 config_id（必須升版並另立 decision log）。"""
    manual = AgentConfig(
        policy="anthropic-messages",
        prompt_version="baseline/v1",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        developer_prompt=BASELINE_DEVELOPER_PROMPT,
        model_config=copy.deepcopy(EXPECTED_MODEL_CONFIG),
    )
    assert manual.config_id == GOLDEN_BASELINE_CONFIG_ID
    assert baseline_agent_config().config_id == GOLDEN_BASELINE_CONFIG_ID


def test_baseline_agent_config_returns_fresh_objects() -> None:
    a = baseline_agent_config()
    b = baseline_agent_config()
    assert a is not b
    assert a.model_config is not b.model_config
    assert a.model_config is not BASELINE_MODEL_CONFIG
    assert a.config_id == b.config_id
    a.model_config["effort"] = "low"
    a.model_config["tool_choice"]["type"] = "any"
    a.model_config["betas"].append("other-beta")
    a.model_config["thinking"]["block_binding"]["prefix_mismatch_behavior"] = "drop_block"
    assert b.model_config["effort"] == "medium"
    fresh = baseline_agent_config()
    assert fresh.model_config["tool_choice"]["type"] == "auto"
    assert fresh.model_config["betas"] == [BETA]
    assert fresh.model_config["thinking"] == EXPECTED_THINKING
    assert canonical_json(BASELINE_MODEL_CONFIG) == canonical_json(EXPECTED_MODEL_CONFIG)
    assert fresh.config_id == b.config_id


# ---------------------------------------------------------------- build_request: shapes


def test_build_request_initial_conversation_exact() -> None:
    """只有 system / developer / user：system 為兩個 text block，messages 只有一則 user（content 為字串）。"""
    req = build_request(_base(), TOOL_SPECS, _baseline_mc())
    assert set(req) == REQUEST_KEYS
    assert req == _expected_request([_user_instr()])


def test_build_request_user_message_is_plain_instruction_string() -> None:
    instruction = "  請關閉 #1，並通知 u2。\n"
    req = build_request(_base(instruction=instruction), TOOL_SPECS, _baseline_mc())
    assert req["messages"][0] == {"role": "user", "content": instruction}
    assert type(req["messages"][0]["content"]) is str


def test_build_request_sends_betas_and_full_thinking_config() -> None:
    req = build_request(_provider_conversation(), TOOL_SPECS, _baseline_mc())
    assert req["betas"] == [BETA]
    assert req["thinking"] == EXPECTED_THINKING
    assert req["thinking"]["display"] == "summarized"
    assert req["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "error"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"] == {"effort": "medium"}


def test_build_request_has_no_sampling_strict_or_fallback_keys() -> None:
    req = build_request(_provider_conversation(), TOOL_SPECS, _baseline_mc())
    for key in ("temperature", "top_p", "top_k", "stop_sequences", "metadata", "stream", "fallbacks", "timeout"):
        assert key not in req
    for tool in req["tools"]:
        assert "strict" not in tool
        assert set(tool) == {"name", "description", "input_schema"}


def test_build_request_system_and_developer_come_from_messages() -> None:
    req = build_request(_base(system="alpha", developer="beta", instruction="gamma"), TOOL_SPECS, _baseline_mc())
    assert req["system"] == [{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}]
    assert req["messages"] == [_user_instr("gamma")]


def test_build_request_prompts_are_sent_verbatim() -> None:
    """非空白的 prompt 原樣送出（不 strip）。"""
    system = "\n  system with surrounding whitespace  \n"
    developer = "\tdeveloper notes\n"
    req = build_request(_base(system=system, developer=developer), TOOL_SPECS, _baseline_mc())
    assert req["system"] == [{"type": "text", "text": system}, {"type": "text", "text": developer}]


@pytest.mark.parametrize("developer", ["", " ", "\n\t  \n"], ids=["empty", "space", "whitespace"])
def test_build_request_blank_developer_prompt_is_omitted(developer: str) -> None:
    req = build_request(_base(developer=developer), TOOL_SPECS, _baseline_mc())
    assert req["system"] == [{"type": "text", "text": SYS}]
    assert req == _expected_request([_user_instr()], developer=None)


def test_build_request_after_provider_turns_replays_provider_content_verbatim() -> None:
    """provider_content 原樣作為 assistant content（含 thinking / redacted_thinking 與 signature）。"""
    req = build_request(_provider_conversation(), TOOL_SPECS, _baseline_mc())
    expected_messages = [
        _user_instr(),
        {"role": "assistant", "content": PROVIDER_1},
        _results(("toolu_01", OK_OBS, False)),
        {"role": "assistant", "content": PROVIDER_2},
        _results(("toolu_02", ERR_OBS, True)),
    ]
    assert set(req) == REQUEST_KEYS
    assert req == _expected_request(expected_messages)
    assert canonical_json(req["messages"][1]["content"]) == canonical_json(PROVIDER_1)
    assert canonical_json(req["messages"][3]["content"]) == canonical_json(PROVIDER_2)


def test_build_request_multi_tool_turn_merges_tool_results() -> None:
    """連續的 tool 訊息合併成一則 user 訊息，每則一個 tool_result，依序；is_error = not ok。"""
    req = build_request(_multi_conversation(), TOOL_SPECS, _baseline_mc())
    expected_messages = [
        _user_instr(),
        {"role": "assistant", "content": PROVIDER_1},
        _results(("toolu_01", OK_OBS, False)),
        {"role": "assistant", "content": PROVIDER_3},
        _results(("toolu_03", OK_OBS, False), ("toolu_04", OK_OBS_2, False)),
    ]
    assert req == _expected_request(expected_messages)


def test_build_request_assistant_content_keeps_block_order() -> None:
    req = build_request(_multi_conversation(), TOOL_SPECS, _baseline_mc())
    assert [b["type"] for b in req["messages"][3]["content"]] == ["thinking", "text", "tool_use", "text", "tool_use"]
    assert set(req["messages"][3]) == {"role", "content"}


def test_build_request_tool_result_content_is_original_string() -> None:
    """tool_result.content 為原 content 字串（不重新序列化、不解析成物件；key 順序與空白保留）。"""
    raw = '{"ok": true, "data":{"title":"登入頁面字型錯誤"}}'
    req = build_request(_single(PROVIDER_1, calls=CALLS_1)[:-1] + [_tool("toolu_01", raw)], TOOL_SPECS, _baseline_mc())
    block = req["messages"][-1]["content"][0]
    assert block == {"type": "tool_result", "tool_use_id": "toolu_01", "content": raw, "is_error": False}
    assert type(block["content"]) is str


@pytest.mark.parametrize(
    ("content", "is_error"),
    [
        (canonical_json({"ok": True, "data": None}), False),
        (canonical_json({"ok": False, "error": {"code": "invalid_arguments", "message": "x"}}), True),
        ('{"ok":false}', True),
        ('{"ok":true}', False),
    ],
)
def test_build_request_is_error_follows_observation_ok(content: str, is_error: bool) -> None:
    messages = _base() + [_a1(), _tool("toolu_01", content)]
    req = build_request(messages, TOOL_SPECS, _baseline_mc())
    assert req["messages"][-1] == _results(("toolu_01", content, is_error))
    assert type(req["messages"][-1]["content"][0]["is_error"]) is bool


def test_build_request_tools_filtered_in_order() -> None:
    tools = (
        {"name": "zeta", "description": "Z tool", "input_schema": {"type": "object"}, "strict": True},
        {
            "name": "alpha",
            "description": "A tool",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "cache_control": {"type": "ephemeral"},
            "x-internal": 1,
        },
    )
    req = build_request(_base(), tools, _baseline_mc())
    assert req["tools"] == [
        {"name": "zeta", "description": "Z tool", "input_schema": {"type": "object"}},
        {
            "name": "alpha",
            "description": "A tool",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    ]


def test_build_request_tools_from_tool_specs_in_order() -> None:
    req = build_request(_base(), TOOL_SPECS, _baseline_mc())
    assert [t["name"] for t in req["tools"]] == [s["name"] for s in TOOL_SPECS]
    assert req["tools"] == _expected_tools()


@pytest.mark.parametrize("conversation", CONVERSATIONS, ids=lambda f: f.__name__)
def test_build_request_empty_tools_raise_value_error(conversation: Callable[[], list[dict[str, Any]]]) -> None:
    with pytest.raises(ValueError):
        build_request(conversation(), (), _baseline_mc())


def test_build_request_values_come_from_model_config() -> None:
    mc = _baseline_mc()
    mc.update(
        model="claude-test-model",
        max_tokens=1234,
        effort="high",
        thinking={"type": "adaptive", "display": "omitted", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
        tool_choice={"type": "auto"},
        betas=["beta-a", "beta-b"],
    )
    req = build_request(_base(), TOOL_SPECS, mc)
    assert req["model"] == "claude-test-model"
    assert req["max_tokens"] == 1234
    assert req["output_config"] == {"effort": "high"}
    assert req["thinking"] == {
        "type": "adaptive",
        "display": "omitted",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert req["tool_choice"] == {"type": "auto"}
    assert req["betas"] == ["beta-a", "beta-b"]
    assert req["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("conversation", CONVERSATIONS, ids=lambda f: f.__name__)
def test_build_request_does_not_mutate_inputs(conversation: Callable[[], list[dict[str, Any]]]) -> None:
    messages = conversation()
    tools = copy.deepcopy(TOOL_SPECS)
    mc = _baseline_mc()
    before = (copy.deepcopy(messages), copy.deepcopy(tools), copy.deepcopy(mc))
    build_request(messages, tools, mc)
    assert messages == before[0]
    assert tools == before[1]
    assert mc == before[2]
    assert canonical_json(messages) == canonical_json(before[0])


@pytest.mark.parametrize("conversation", CONVERSATIONS, ids=lambda f: f.__name__)
def test_build_request_output_is_json_native(conversation: Callable[[], list[dict[str, Any]]]) -> None:
    req = build_request(conversation(), TOOL_SPECS, _baseline_mc())
    _assert_json_native(req)
    assert json.loads(json.dumps(req)) == req


def test_build_request_output_shares_no_mutable_objects_with_inputs() -> None:
    """回傳新物件：修改 request 不影響輸入的 messages / tools / model_config。"""
    messages = _provider_conversation()
    tools = copy.deepcopy(TOOL_SPECS)
    mc = _baseline_mc()
    before = (copy.deepcopy(messages), copy.deepcopy(tools), copy.deepcopy(mc))
    req = build_request(messages, tools, mc)
    req["messages"][1]["content"][0]["signature"] = "TAMPERED"
    req["messages"][1]["content"][2]["input"]["issue_id"] = 999
    req["messages"][1]["content"].append({"type": "text", "text": "TAMPERED"})
    req["tools"][0]["input_schema"]["TAMPERED"] = True
    req["tool_choice"]["type"] = "TAMPERED"
    req["thinking"]["block_binding"]["prefix_mismatch_behavior"] = "drop_block"
    req["thinking"]["display"] = "omitted"
    req["betas"].append("TAMPERED")
    req["output_config"]["effort"] = "low"
    assert messages == before[0]
    assert tools == before[1]
    assert mc == before[2]


def test_build_request_is_deterministic() -> None:
    a = build_request(_multi_conversation(), TOOL_SPECS, _baseline_mc())
    b = build_request(_multi_conversation(), TOOL_SPECS, _baseline_mc())
    assert canonical_json(a) == canonical_json(b)


def test_build_request_prefix_is_stable_as_conversation_grows() -> None:
    """decision 009 第 7 點：harness 只附加訊息，較短 conversation 的 request messages 是較長者的前綴。"""
    full = _provider_conversation()
    short = full[:5]  # system, developer, user, assistant, tool
    req_short = build_request(short, TOOL_SPECS, _baseline_mc())
    req_full = build_request(full, TOOL_SPECS, _baseline_mc())
    n = len(req_short["messages"])
    assert canonical_json(req_full["messages"][:n]) == canonical_json(req_short["messages"])
    for key in ("system", "tools", "thinking", "betas", "tool_choice", "output_config", "cache_control"):
        assert canonical_json(req_full[key]) == canonical_json(req_short[key])


def test_build_request_accepts_provider_content_regardless_of_key_order() -> None:
    """一致性比較以 canonical_json 進行：block 內 key 順序與 input 的 key 順序不影響（與 args 的 key 順序不同也接受）。"""
    pc = [
        {"signature": "sig-1", "thinking": "", "type": "thinking"},
        {"text": "Checking.", "type": "text"},
        {"input": {"status": "closed", "issue_id": 1}, "name": "update_issue", "id": "toolu_k", "type": "tool_use"},
    ]
    calls = [("toolu_k", "update_issue", {"issue_id": 1, "status": "closed"})]
    req = build_request(_single(pc, calls=calls), TOOL_SPECS, _baseline_mc())
    assert canonical_json(req["messages"][1]["content"]) == canonical_json(pc)
    assert req["messages"][1]["content"] == pc


def test_build_request_accepts_nested_equal_inputs() -> None:
    args = {"issue_id": 3, "labels": ["bug", "ui"], "fields": {"a": [1, 2.5, None, True], "b": "登入"}}
    pc = [_th("s"), _tu("toolu_n", "update_issue", copy.deepcopy(args))]
    req = build_request(_single(pc, text="", calls=[("toolu_n", "update_issue", args)]), TOOL_SPECS, _baseline_mc())
    assert canonical_json(req["messages"][1]["content"]) == canonical_json(pc)


def test_build_request_matches_golden_hash() -> None:
    """decision 009 第 1 點：adapter 版本以 golden hash 釘住固定 conversation 的 request。"""
    golden_tools = (
        {
            "name": "get_issue",
            "description": "Get one issue by id.",
            "input_schema": {"type": "object", "properties": {"issue_id": {"type": "integer"}}, "required": ["issue_id"]},
        },
        {
            "name": "update_issue",
            "description": "Update fields of an issue.",
            "input_schema": {
                "type": "object",
                "properties": {"issue_id": {"type": "integer"}, "status": {"type": "string"}},
                "required": ["issue_id"],
            },
            "strict": True,
        },
    )
    messages = _base(BASELINE_SYSTEM_PROMPT, BASELINE_DEVELOPER_PROMPT, INSTR) + [
        _a1(),
        _tool("toolu_01", OK_OBS),
        _a2(),
        _tool("toolu_02", ERR_OBS),
    ]
    expected = _expected_request(
        [
            _user_instr(),
            {"role": "assistant", "content": PROVIDER_1},
            _results(("toolu_01", OK_OBS, False)),
            {"role": "assistant", "content": PROVIDER_2},
            _results(("toolu_02", ERR_OBS, True)),
        ],
        system=BASELINE_SYSTEM_PROMPT,
        developer=BASELINE_DEVELOPER_PROMPT,
        tools=golden_tools,
    )
    assert sha256_of(expected) == GOLDEN_REQUEST_SHA256
    req = build_request(messages, golden_tools, copy.deepcopy(BASELINE_MODEL_CONFIG))
    assert req == expected
    assert sha256_of(req) == GOLDEN_REQUEST_SHA256


# ---------------------------------------------------------------- build_request: invalid conversations


def _without(message: dict[str, Any], key: str) -> dict[str, Any]:
    out = dict(message)
    del out[key]
    return out


def _structure_cases() -> list[tuple[str, Callable[[], list[Any]]]]:
    def b() -> list[dict[str, Any]]:
        return _base()

    return [
        ("empty", lambda: []),
        ("system_only", lambda: [b()[0]]),
        ("no_user", lambda: [b()[0], b()[1]]),
        ("swapped_system_developer", lambda: [b()[1], b()[0], b()[2]]),
        ("first_is_user", lambda: [b()[2], b()[0], b()[1]]),
        ("third_is_assistant", lambda: [b()[0], b()[1], _a1(), _tool("toolu_01", OK_OBS)]),
        ("tool_before_any_assistant", lambda: b() + [_tool("toolu_01", OK_OBS)]),
        ("unknown_role", lambda: b() + [{"role": "function", "content": "x"}]),
        ("second_system_later", lambda: b() + [_a1(), _tool("toolu_01", OK_OBS), {"role": "system", "content": "again"}]),
        ("user_after_tool", lambda: b() + [_a1(), _tool("toolu_01", OK_OBS), {"role": "user", "content": "more"}]),
        ("missing_role", lambda: b() + [{"content": "x"}]),
        ("message_not_a_dict", lambda: b() + ["assistant"]),
        ("system_missing_content", lambda: [_without(b()[0], "content"), b()[1], b()[2]]),
        ("developer_missing_content", lambda: [b()[0], _without(b()[1], "content"), b()[2]]),
        ("user_missing_content", lambda: [b()[0], b()[1], _without(b()[2], "content")]),
        ("system_empty", lambda: _base(system="")),
        ("system_whitespace", lambda: _base(system="  \n\t ")),
        ("system_content_none", lambda: [{"role": "system", "content": None}, b()[1], b()[2]]),
        ("user_empty", lambda: _base(instruction="")),
        ("user_whitespace", lambda: _base(instruction=" \n ")),
        ("user_content_block_list", lambda: [b()[0], b()[1], {"role": "user", "content": [{"type": "text", "text": INSTR}]}]),
        ("assistant_missing_tool_calls", lambda: b() + [_without(_a1(), "tool_calls"), _tool("toolu_01", OK_OBS)]),
        ("assistant_missing_content", lambda: b() + [_without(_a1(), "content"), _tool("toolu_01", OK_OBS)]),
        ("assistant_content_none", lambda: _single(PROVIDER_1, text=None)),
        ("tool_missing_tool_call_id", lambda: b() + [_a1(), _without(_tool("toolu_01", OK_OBS), "tool_call_id")]),
        ("tool_missing_content", lambda: b() + [_a1(), _without(_tool("toolu_01", OK_OBS), "content")]),
    ]


def _tool_message_cases() -> list[tuple[str, Callable[[], list[Any]]]]:
    """tool 訊息數量 / id 必須與前一則 assistant 的 tool_calls 依序一致。"""
    return [
        ("missing_tool_result", lambda: _base() + [_a1()]),
        ("assistant_followed_by_assistant", lambda: _base() + [_a1(), _a2(), _tool("toolu_02", OK_OBS)]),
        ("missing_one_of_two_tool_results", lambda: _multi(results=[_tool("toolu_03", OK_OBS)])),
        ("extra_duplicate_tool_result", lambda: _base() + [_a1(), _tool("toolu_01", OK_OBS), _tool("toolu_01", OK_OBS)]),
        ("extra_unknown_tool_result", lambda: _base() + [_a1(), _tool("toolu_01", OK_OBS), _tool("toolu_99", OK_OBS)]),
        ("tool_result_wrong_id", lambda: _base() + [_a1(), _tool("toolu_99", OK_OBS)]),
        ("tool_results_swapped", lambda: _multi(results=[_tool("toolu_04", OK_OBS_2), _tool("toolu_03", OK_OBS)])),
        (
            "second_result_wrong_id",
            lambda: _multi(results=[_tool("toolu_03", OK_OBS), _tool("toolu_01", OK_OBS_2)]),
        ),
    ]


def _tool_content_cases() -> list[tuple[str, Callable[[], list[Any]]]]:
    """tool content 必須是含 bool "ok" 的 JSON 物件字串。"""
    bad_contents: list[tuple[str, Any]] = [
        ("not_json", "not json"),
        ("empty_string", ""),
        ("json_array", "[]"),
        ("json_string", '"ok"'),
        ("json_null", "null"),
        ("json_true", "true"),
        ("object_without_ok", '{"data":{}}'),
        ("ok_is_int_1", '{"ok":1}'),
        ("ok_is_int_0", '{"ok":0}'),
        ("ok_is_string", '{"ok":"true"}'),
        ("ok_is_null", '{"ok":null}'),
        ("content_is_dict", {"ok": True}),
        ("content_is_none", None),
    ]
    return [
        (f"tool_content_{name}", (lambda c=content: _base() + [_a1(), _tool("toolu_01", c)])) for name, content in bad_contents
    ]


def _provider_content_cases() -> list[tuple[str, Callable[[], list[Any]]]]:
    """每則 assistant 的 provider_content 必須存在、非空、只含白名單 block（剛好規定的 key、型別正確），
    且 tool_use blocks 依序與 tool_calls 逐一相同、text blocks 串接 == content。"""
    good_th, good_tx, good_tu = PROVIDER_1
    return [
        # presence / container
        ("provider_content_key_missing", lambda: _base() + [_without(_a1(), "provider_content"), _tool("toolu_01", OK_OBS)]),
        ("provider_content_none_scripted_style", lambda: _single(None)),
        ("provider_content_none_without_text", lambda: _single(None, text="")),
        ("provider_content_empty_list", lambda: _single([])),
        ("provider_content_is_dict", lambda: _single({"type": "text", "text": TEXT_1})),
        ("provider_content_is_str", lambda: _single(TEXT_1)),
        ("provider_content_block_is_a_list", lambda: _single([["type", "text"]])),
        # block shape
        ("block_not_a_dict", lambda: _single(_p1_with(0, "thinking"))),
        ("block_missing_type", lambda: _single(_p1_with(0, {"thinking": "", "signature": "sig-1"}))),
        ("block_type_not_str", lambda: _single(_p1_with(0, {"type": 1, "thinking": "", "signature": "sig-1"}))),
        ("unknown_block_server_tool_use", lambda: _single(_p1_with(0, {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}))),
        ("unknown_block_image", lambda: _single(_p1_with(0, {"type": "image", "source": {"type": "url", "url": "x"}}))),
        ("unknown_block_tool_result", lambda: _single(_p1_with(0, {"type": "tool_result", "tool_use_id": "toolu_01", "content": "x"}))),
        ("unknown_block_compaction", lambda: _single(_p1_with(0, {"type": "compaction", "content": "x"}))),
        # extra keys
        ("text_block_extra_citations", lambda: _single(_p1_with(1, {"type": "text", "text": TEXT_1, "citations": None}))),
        ("thinking_block_extra_cache_control", lambda: _single(_p1_with(0, {**good_th, "cache_control": {"type": "ephemeral"}}))),
        ("tool_use_block_extra_caller", lambda: _single(_p1_with(2, {**good_tu, "caller": {"type": "direct"}}))),
        ("redacted_block_extra_key", lambda: _single([{"type": "redacted_thinking", "data": "x", "signature": "s"}] + copy.deepcopy(PROVIDER_1))),
        # missing keys
        ("thinking_missing_signature", lambda: _single(_p1_with(0, {"type": "thinking", "thinking": ""}))),
        ("thinking_missing_thinking", lambda: _single(_p1_with(0, {"type": "thinking", "signature": "sig-1"}))),
        ("redacted_missing_data", lambda: _single([{"type": "redacted_thinking"}] + copy.deepcopy(PROVIDER_1))),
        ("text_block_missing_text", lambda: _single([good_th, {"type": "text"}, good_tu], text="")),
        ("tool_use_missing_input", lambda: _single(_p1_with(2, {"type": "tool_use", "id": "toolu_01", "name": "get_issue"}))),
        ("tool_use_missing_id", lambda: _single(_p1_with(2, {"type": "tool_use", "name": "get_issue", "input": {"issue_id": 1}}))),
        # wrong field types
        ("thinking_signature_int", lambda: _single(_p1_with(0, {"type": "thinking", "thinking": "", "signature": 1}))),
        ("thinking_signature_none", lambda: _single(_p1_with(0, {"type": "thinking", "thinking": "", "signature": None}))),
        ("thinking_thinking_none", lambda: _single(_p1_with(0, {"type": "thinking", "thinking": None, "signature": "sig-1"}))),
        ("redacted_data_list", lambda: _single([{"type": "redacted_thinking", "data": ["x"]}] + copy.deepcopy(PROVIDER_1))),
        ("redacted_data_none", lambda: _single([{"type": "redacted_thinking", "data": None}] + copy.deepcopy(PROVIDER_1))),
        ("text_text_none", lambda: _single([good_th, {"type": "text", "text": None}, good_tu], text="")),
        ("text_text_int", lambda: _single([good_th, {"type": "text", "text": 5}, good_tu], text="5")),
        ("tool_use_id_int", lambda: _single(_p1_with(2, {"type": "tool_use", "id": 1, "name": "get_issue", "input": {"issue_id": 1}}))),
        ("tool_use_name_none", lambda: _single(_p1_with(2, {"type": "tool_use", "id": "toolu_01", "name": None, "input": {"issue_id": 1}}))),
        # tool_use vs tool_calls
        ("tool_use_wrong_id", lambda: _single(_p1_with(2, _tu("toolu_XX", "get_issue", {"issue_id": 1})))),
        ("tool_use_wrong_name", lambda: _single(_p1_with(2, _tu("toolu_01", "update_issue", {"issue_id": 1})))),
        ("tool_use_wrong_input_value", lambda: _single(_p1_with(2, _tu("toolu_01", "get_issue", {"issue_id": 2})))),
        ("tool_use_input_float_vs_int_args", lambda: _single(_p1_with(2, _tu("toolu_01", "get_issue", {"issue_id": 1.0})))),
        (
            "tool_use_input_int_vs_float_args",
            lambda: _single(PROVIDER_1, calls=[("toolu_01", "get_issue", {"issue_id": 1.0})]),
        ),
        ("tool_use_input_bool_vs_int_args", lambda: _single(_p1_with(2, _tu("toolu_01", "get_issue", {"issue_id": True})))),
        ("tool_use_input_extra_key", lambda: _single(_p1_with(2, _tu("toolu_01", "get_issue", {"issue_id": 1, "x": None})))),
        ("tool_use_input_empty", lambda: _single(_p1_with(2, _tu("toolu_01", "get_issue", {})))),
        ("tool_use_input_wrapped_in_list", lambda: _single(_p1_with(2, _tu("toolu_01", "get_issue", [{"issue_id": 1}])))),
        ("provider_has_extra_tool_use", lambda: _single(copy.deepcopy(PROVIDER_1) + [_tu("toolu_05", "get_issue", {"issue_id": 2})])),
        ("provider_missing_tool_use", lambda: _single([good_th, good_tx])),
        (
            "tool_calls_have_extra_call",
            lambda: _single(PROVIDER_1, calls=CALLS_1 + [("toolu_05", "get_issue", {"issue_id": 2})]),
        ),
        ("tool_use_order_swapped", lambda: _multi([PROVIDER_3[0], PROVIDER_3[1], PROVIDER_3[4], PROVIDER_3[3], PROVIDER_3[2]])),
        ("tool_call_missing_args", lambda: _base() + [{**_a1(), "tool_calls": [{"id": "toolu_01", "name": "get_issue"}]}, _tool("toolu_01", OK_OBS)]),
        ("tool_calls_not_a_list", lambda: _base() + [{**_a1(), "tool_calls": None}, _tool("toolu_01", OK_OBS)]),
        # text vs content
        ("text_differs", lambda: _single(PROVIDER_1, text="Checking!")),
        ("text_trailing_whitespace_differs", lambda: _single(PROVIDER_1, text="Checking. ")),
        ("text_missing_in_provider", lambda: _single([good_th, good_tu])),
        ("text_extra_in_provider", lambda: _single(PROVIDER_1, text="")),
        ("text_concat_order_differs", lambda: _multi(text="and commenting.Closing ")),
        ("text_concat_with_separator", lambda: _multi(text="Closing \nand commenting.")),
        ("text_only_first_block", lambda: _multi(text="Closing ")),
    ]


INVALID_CASES = _structure_cases() + _tool_message_cases() + _tool_content_cases() + _provider_content_cases()


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda c: c[0])
def test_build_request_invalid_messages_raise_value_error(case: tuple[str, Callable[[], list[Any]]]) -> None:
    _, make = case
    with pytest.raises(ValueError):
        build_request(make(), TOOL_SPECS, _baseline_mc())


def test_valid_controls_for_invalid_cases_are_accepted() -> None:
    """對照組：上面各案例所改動的原始 conversation 本身是合法的（確保失敗來自被改動的那一點）。"""
    for messages in (_single(PROVIDER_1), _multi(), _provider_conversation(), _multi_conversation(), _base()):
        build_request(messages, TOOL_SPECS, _baseline_mc())


# ---------------------------------------------------------------- parse_response: stop reasons


def test_parse_tool_use_turn() -> None:
    resp = tool_turn("toolu_01", "get_issue", {"issue_id": 1}, signature="sig-a", note="Checking.")
    step = parse_response(resp, MODEL)
    assert isinstance(step, PolicyStep)
    msg = step.message
    assert isinstance(msg, ModelMessage)
    assert msg.stop_reason == "tool_use"
    assert msg.text == "Checking."
    assert msg.tool_calls == (ToolCall(id="toolu_01", name="get_issue", args={"issue_id": 1}),)
    assert msg.provider_content == [
        {"type": "thinking", "thinking": "", "signature": "sig-a"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_01", "name": "get_issue", "input": {"issue_id": 1}},
    ]


def test_parse_tool_use_without_text_has_empty_text() -> None:
    step = parse_response(tool_turn("toolu_9", "search_issues", {"query": "login"}), MODEL)
    assert step.message.text == ""
    assert [(c.id, c.name, c.args) for c in step.message.tool_calls] == [("toolu_9", "search_issues", {"query": "login"})]


def test_parse_end_turn() -> None:
    step = parse_response(final_turn("All done.", signature="sig-f"), MODEL)
    msg = step.message
    assert msg.stop_reason == "end_turn"
    assert msg.text == "All done."
    assert msg.tool_calls == ()
    assert msg.provider_content == [
        {"type": "thinking", "thinking": "", "signature": "sig-f"},
        {"type": "text", "text": "All done."},
    ]


@pytest.mark.parametrize(
    "content",
    [
        [thinking_block("s"), text_block("calling"), tool_use_block("toolu_1", "get_issue", {"issue_id": 1})],
        [tool_use_block("toolu_1", "get_issue", {"issue_id": 1})],
    ],
    ids=["with_text", "tool_use_only"],
)
def test_parse_end_turn_with_tool_use_block_raises(content: list[Any]) -> None:
    with pytest.raises(AdapterError):
        parse_response(_resp(content, "end_turn"), MODEL)


def test_parse_refusal_has_empty_text_and_keeps_blocks() -> None:
    """refusal：text = ""、tool_calls = ()、provider_content 保留全部 blocks（供稽核）。"""
    resp = _resp(
        [thinking_block("sig-r", "hmm"), text_block("I can't help with that.")],
        "refusal",
        stop_details=FakeStopDetails(category="cyber", explanation="policy"),
    )
    step = parse_response(resp, MODEL)
    assert step.message.stop_reason == "refusal"
    assert step.message.text == ""
    assert step.message.tool_calls == ()
    assert step.message.provider_content == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig-r"},
        {"type": "text", "text": "I can't help with that."},
    ]


def test_parse_refusal_never_yields_tool_calls_but_keeps_tool_use_block() -> None:
    resp = _resp(
        [thinking_block("sig-r"), tool_use_block("toolu_x", "update_issue", {"issue_id": 1, "status": "closed"})],
        "refusal",
    )
    step = parse_response(resp, MODEL)
    assert step.message.stop_reason == "refusal"
    assert step.message.text == ""
    assert step.message.tool_calls == ()
    assert step.message.provider_content == [
        {"type": "thinking", "thinking": "", "signature": "sig-r"},
        {"type": "tool_use", "id": "toolu_x", "name": "update_issue", "input": {"issue_id": 1, "status": "closed"}},
    ]


def test_parse_refusal_with_empty_content() -> None:
    step = parse_response(_resp([], "refusal"), MODEL)
    assert step.message.stop_reason == "refusal"
    assert step.message.text == ""
    assert step.message.tool_calls == ()
    assert step.message.provider_content == []


ABNORMAL_STOP_REASONS: list[Any] = [
    "max_tokens",
    "pause_turn",
    "model_context_window_exceeded",
    "stop_sequence",
    "compaction",
    "weird",
    "",
    "END_TURN",
    "end_turn ",
    None,
]


@pytest.mark.parametrize("stop_reason", ABNORMAL_STOP_REASONS, ids=repr)
def test_parse_abnormal_stop_reason_raises_adapter_error(stop_reason: Any) -> None:
    resp = _resp([thinking_block("s"), text_block("partial")], stop_reason)
    with pytest.raises(AdapterError):
        parse_response(resp, MODEL)


@pytest.mark.parametrize("stop_reason", ABNORMAL_STOP_REASONS, ids=repr)
def test_parse_abnormal_stop_reason_with_tool_use_still_raises(stop_reason: Any) -> None:
    resp = _resp([thinking_block("s"), tool_use_block("toolu_1", "get_issue", {"issue_id": 1})], stop_reason)
    with pytest.raises(AdapterError):
        parse_response(resp, MODEL)


@pytest.mark.parametrize(
    "content",
    [[], [thinking_block("s")], [thinking_block("s"), text_block("I will call a tool")]],
    ids=["empty", "thinking_only", "thinking_and_text"],
)
def test_parse_tool_use_without_tool_use_block_raises(content: list[Any]) -> None:
    with pytest.raises(AdapterError):
        parse_response(_resp(content, "tool_use"), MODEL)


# ---------------------------------------------------------------- parse_response: input_transformations


@pytest.mark.parametrize("stop_reason", ["tool_use", "end_turn", "refusal"])
def test_parse_nonempty_input_transformations_raise_replay_integrity_error(stop_reason: str) -> None:
    content: list[Any] = [thinking_block("s"), text_block("x")]
    if stop_reason == "tool_use":
        content.append(tool_use_block("toolu_1", "get_issue", {"issue_id": 1}))
    transformation = SimpleNamespace(type="thinking_dropped", path="messages.1.content.0", reason="prefix_binding_mismatch")
    resp = _resp(content, stop_reason, input_transformations=[transformation])
    with pytest.raises(ReplayIntegrityError):
        parse_response(resp, MODEL)


def test_parse_input_transformations_checked_before_stop_reason() -> None:
    """契約中 input_transformations 是 parse_response 的第一條規則：重放不忠實（harness 類）優先於截斷（agent_output 類）。"""
    resp = _resp(
        [thinking_block("s"), text_block("partial")],
        "max_tokens",
        input_transformations=[{"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "model_binding_mismatch"}],
    )
    with pytest.raises(ReplayIntegrityError):
        parse_response(resp, MODEL)


@pytest.mark.parametrize("transformations", [None, []], ids=["none", "empty_list"])
def test_parse_empty_input_transformations_are_fine(transformations: Any) -> None:
    step = parse_response(_resp([text_block("ok")], "end_turn", input_transformations=transformations), MODEL)
    assert step.message.text == "ok"


def test_parse_response_without_optional_attributes() -> None:
    """缺少 stop_details / input_transformations / _request_id 屬性時視為 None。"""
    resp = SimpleNamespace(stop_reason="end_turn", content=[text_block("ok")], usage=FakeUsage(3, 2), model=MODEL)
    step = parse_response(resp, MODEL)
    assert step.message.text == "ok"
    assert step.meta is not None
    assert step.meta["request_id"] is None
    assert step.meta["stop_details"] is None


# ---------------------------------------------------------------- parse_response: blocks


def test_parse_blocks_keep_only_whitelisted_fields() -> None:
    """FakeBlock 的 extra 與其他為 None 的屬性都不得洩漏到 provider_content。"""
    blocks = [
        FakeBlock(type="thinking", thinking="hmm", signature="sig-x", extra={"cache_control": {"type": "ephemeral"}}),
        FakeBlock(type="redacted_thinking", data="blob", extra={"foo": 1}),
        FakeBlock(type="text", text="note", extra={"citations": [{"cited_text": "x"}]}),
        FakeBlock(type="tool_use", id="toolu_7", name="get_issue", input={"issue_id": 3}, extra={"caller": "direct"}),
    ]
    step = parse_response(_resp(blocks, "tool_use"), MODEL)
    pc = step.message.provider_content
    assert pc == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig-x"},
        {"type": "redacted_thinking", "data": "blob"},
        {"type": "text", "text": "note"},
        {"type": "tool_use", "id": "toolu_7", "name": "get_issue", "input": {"issue_id": 3}},
    ]
    for block in pc:
        assert set(block) == set(ALLOWED_BLOCK_TYPES[block["type"]])


class _NoSerialization:
    """以屬性讀取的 block / response；呼叫任何 SDK 序列化方法都會失敗（契約：不呼叫 SDK 序列化）。"""

    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)

    def _fail(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parse_response must read attributes, not call SDK serialization")

    model_dump = _fail
    model_dump_json = _fail
    to_dict = _fail
    to_json = _fail
    dict = _fail
    json = _fail


def test_parse_blocks_read_by_attribute_only() -> None:
    """block 以屬性讀取：任意帶屬性的物件（非 SDK 型別）都可；多餘屬性不得保存；不呼叫序列化方法。"""
    blocks = [
        _NoSerialization(type="thinking", thinking="t", signature="sig-ns", citations=None, parsed=None),
        SimpleNamespace(type="text", text="hello", citations=[{"x": 1}]),
        _NoSerialization(type="tool_use", id="toolu_ns", name="get_issue", input={"issue_id": 4}, caller={"type": "direct"}),
    ]
    resp = _NoSerialization(
        stop_reason="tool_use", content=blocks, usage=FakeUsage(1, 1), model=MODEL, stop_details=None,
        input_transformations=None, _request_id="req_ns",
    )
    step = parse_response(resp, MODEL)
    assert step.message.provider_content == [
        {"type": "thinking", "thinking": "t", "signature": "sig-ns"},
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "toolu_ns", "name": "get_issue", "input": {"issue_id": 4}},
    ]


def test_parse_redacted_thinking_block() -> None:
    resp = _resp([redacted_thinking_block("ENC=="), text_block("ok")], "end_turn")
    step = parse_response(resp, MODEL)
    assert step.message.provider_content == [{"type": "redacted_thinking", "data": "ENC=="}, {"type": "text", "text": "ok"}]
    assert step.message.text == "ok"


@pytest.mark.parametrize(
    "block_type", ["server_tool_use", "web_search_tool_result", "image", "document", "container_upload", "compaction", "mystery"]
)
@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use"])
def test_parse_unknown_block_type_raises(block_type: str, stop_reason: str) -> None:
    blocks = [thinking_block("s"), FakeBlock(type=block_type, text="x", data="y")]
    if stop_reason == "tool_use":
        blocks.append(tool_use_block("toolu_1", "get_issue", {"issue_id": 1}))
    with pytest.raises(AdapterError):
        parse_response(_resp(blocks, stop_reason), MODEL)


BAD_RESPONSE_BLOCKS: list[tuple[str, Any]] = [
    ("thinking_signature_none", FakeBlock(type="thinking", thinking="t", signature=None)),
    ("thinking_signature_int", FakeBlock(type="thinking", thinking="t", signature=7)),  # type: ignore[arg-type]
    ("thinking_thinking_none", FakeBlock(type="thinking", thinking=None, signature="s")),
    ("thinking_missing_signature_attr", SimpleNamespace(type="thinking", thinking="t")),
    ("redacted_data_none", FakeBlock(type="redacted_thinking", data=None)),
    ("redacted_data_bytes", FakeBlock(type="redacted_thinking", data=b"x")),  # type: ignore[arg-type]
    ("text_text_none", FakeBlock(type="text", text=None)),
    ("text_text_int", FakeBlock(type="text", text=5)),  # type: ignore[arg-type]
    ("text_missing_text_attr", SimpleNamespace(type="text")),
    ("block_type_none", FakeBlock(type=None)),  # type: ignore[arg-type]
    ("block_missing_type_attr", SimpleNamespace(text="x")),
]


@pytest.mark.parametrize("case", BAD_RESPONSE_BLOCKS, ids=lambda c: c[0])
@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use"])
def test_parse_block_with_wrong_field_type_raises(case: tuple[str, Any], stop_reason: str) -> None:
    _, bad = case
    blocks = [thinking_block("s"), bad]
    if stop_reason == "tool_use":
        blocks.append(tool_use_block("toolu_1", "get_issue", {"issue_id": 1}))
    with pytest.raises(AdapterError):
        parse_response(_resp(blocks, stop_reason), MODEL)


@pytest.mark.parametrize(
    "bad",
    [
        FakeBlock(type="tool_use", id=None, name="get_issue", input={"issue_id": 1}),
        FakeBlock(type="tool_use", id=1, name="get_issue", input={"issue_id": 1}),  # type: ignore[arg-type]
        FakeBlock(type="tool_use", id="toolu_1", name=None, input={"issue_id": 1}),
        FakeBlock(type="tool_use", id="toolu_1", name=5, input={"issue_id": 1}),  # type: ignore[arg-type]
        SimpleNamespace(type="tool_use", id="toolu_1", name="get_issue"),
    ],
    ids=["id_none", "id_int", "name_none", "name_int", "missing_input_attr"],
)
def test_parse_tool_use_block_with_wrong_field_type_raises(bad: Any) -> None:
    with pytest.raises(AdapterError):
        parse_response(_resp([thinking_block("s"), bad], "tool_use"), MODEL)


def test_parse_text_blocks_are_concatenated_in_order() -> None:
    blocks = [
        text_block("Hello, "),
        thinking_block("s1", "between"),
        text_block("world"),
        redacted_thinking_block("r"),
        text_block("!"),
    ]
    step = parse_response(_resp(blocks, "end_turn"), MODEL)
    assert step.message.text == "Hello, world!"
    assert [b["type"] for b in step.message.provider_content] == [
        "text",
        "thinking",
        "text",
        "redacted_thinking",
        "text",
    ]


def test_parse_multiple_tool_use_blocks_recorded_in_order() -> None:
    """decision 009 第 4 點：多於一個 tool_use block 時照順序全部記錄，meta 記錄 tool_use_blocks。"""
    blocks = [
        thinking_block("s"),
        tool_use_block("toolu_a", "get_issue", {"issue_id": 1}),
        text_block("and"),
        tool_use_block("toolu_b", "assign_issue", {"issue_id": 1, "assignee_id": "u1"}),
    ]
    step = parse_response(_resp(blocks, "tool_use"), MODEL)
    assert step.message.tool_calls == (
        ToolCall(id="toolu_a", name="get_issue", args={"issue_id": 1}),
        ToolCall(id="toolu_b", name="assign_issue", args={"issue_id": 1, "assignee_id": "u1"}),
    )
    assert step.message.text == "and"
    assert step.meta is not None and step.meta["tool_use_blocks"] == 2


def test_parse_tool_use_with_non_dict_input_is_kept() -> None:
    """tool_use.input 原樣成為 ToolCall.args（非 dict 也記錄，交給 env 判定 invalid_arguments）。"""
    step = parse_response(_resp([tool_use_block("toolu_s", "get_issue", "not-an-object")], "tool_use"), MODEL)
    assert step.message.tool_calls == (ToolCall(id="toolu_s", name="get_issue", args="not-an-object"),)


def test_parse_result_does_not_alias_response_objects() -> None:
    args = {"issue_id": 1, "labels": ["bug"]}
    resp = _resp([thinking_block("s"), tool_use_block("toolu_1", "update_issue", args)], "tool_use")
    step = parse_response(resp, MODEL)
    args["labels"].append("MUTATED")
    args["issue_id"] = 999
    assert step.message.tool_calls[0].args == {"issue_id": 1, "labels": ["bug"]}
    assert step.message.provider_content[1]["input"] == {"issue_id": 1, "labels": ["bug"]}


def test_parse_provider_content_is_json_native() -> None:
    step = parse_response(tool_turn("toolu_1", "get_issue", {"issue_id": 1}, note="n"), MODEL)
    _assert_json_native(step.message.provider_content)


def test_parse_does_not_mutate_response() -> None:
    resp = tool_turn("toolu_1", "get_issue", {"issue_id": 1}, note="n")
    before = copy.deepcopy(resp)
    parse_response(resp, MODEL)
    assert resp == before


# ---------------------------------------------------------------- parse_response: meta


def test_parse_meta_exact_for_tool_use_turn() -> None:
    usage = FakeUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_creation_input_tokens=3000,
        cache_read_input_tokens=10000,
        cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=2000),
    )
    resp = _resp(
        [thinking_block("s"), tool_use_block("toolu_1", "get_issue", {"issue_id": 1})],
        "tool_use",
        usage,
        model="claude-opus-5-5-20261001",
        _request_id="req_abc123",
    )
    step = parse_response(resp, MODEL)
    assert step.meta == {
        "response_model": "claude-opus-5-5-20261001",
        "request_id": "req_abc123",
        "stop_reason": "tool_use",
        "stop_details": None,
        "tool_use_blocks": 1,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 3000,
            "cache_read_input_tokens": 10000,
            "cache_creation_1h_input_tokens": 2000,
        },
    }
    assert type(step.meta["tool_use_blocks"]) is int
    _assert_json_native(step.meta)


def test_parse_meta_records_response_model_not_model_argument() -> None:
    step = parse_response(_resp([text_block("x")], "end_turn", model="claude-opus-5-5-snapshot"), MODEL)
    assert step.meta is not None
    assert step.meta["response_model"] == "claude-opus-5-5-snapshot"


def test_parse_meta_end_turn_has_zero_tool_use_blocks() -> None:
    step = parse_response(final_turn("done"), MODEL)
    assert step.meta is not None
    assert set(step.meta) == META_KEYS
    assert set(step.meta["usage"]) == META_USAGE_KEYS
    assert step.meta["stop_reason"] == "end_turn"
    assert step.meta["tool_use_blocks"] == 0
    assert step.meta["request_id"] == "req_fake"
    assert step.meta["response_model"] == MODEL


def test_parse_meta_request_id_none_when_missing() -> None:
    step = parse_response(_resp([text_block("x")], "end_turn", _request_id=None), MODEL)
    assert step.meta is not None and step.meta["request_id"] is None


def test_parse_meta_refusal_records_stop_details() -> None:
    resp = _resp([text_block("no")], "refusal", stop_details=FakeStopDetails(category="cyber", explanation="Exploit dev."))
    step = parse_response(resp, MODEL)
    assert step.meta is not None
    assert step.meta["stop_reason"] == "refusal"
    assert step.meta["stop_details"] == {"category": "cyber", "explanation": "Exploit dev."}


def test_parse_meta_stop_details_keeps_only_category_and_explanation() -> None:
    details = SimpleNamespace(
        category="general_harms",
        explanation=None,
        fallback_credit_token="tok-secret-ish",
        fallback_has_prefill_claim=True,
        recommended_model="claude-other",
        type="refusal",
    )
    step = parse_response(_resp([], "refusal", stop_details=details), MODEL)
    assert step.meta is not None
    assert step.meta["stop_details"] == {"category": "general_harms", "explanation": None}


def test_parse_meta_refusal_without_stop_details() -> None:
    step = parse_response(_resp([text_block("no")], "refusal"), MODEL)
    assert step.meta is not None and step.meta["stop_details"] is None


def test_parse_meta_usage_without_breakdown_has_all_keys() -> None:
    step = parse_response(_resp([text_block("x")], "end_turn", FakeUsage(12, 3)), MODEL)
    assert step.meta is not None
    assert set(step.meta["usage"]) == META_USAGE_KEYS
    assert step.meta["usage"]["input_tokens"] == 12
    assert step.meta["usage"]["output_tokens"] == 3


def test_parse_meta_does_not_alias_response_objects() -> None:
    details = FakeStopDetails(category="bio", explanation="e")
    resp = _resp([text_block("no")], "refusal", stop_details=details)
    step = parse_response(resp, MODEL)
    details.category = "MUTATED"
    resp.usage.input_tokens = 999999
    assert step.meta is not None
    assert step.meta["stop_details"] == {"category": "bio", "explanation": "e"}
    assert step.meta["usage"]["input_tokens"] == 10


# ---------------------------------------------------------------- parse_response: usage & cost


def test_parse_usage_and_cost_with_1h_cache_split() -> None:
    """cost = (input×4 + cw5m×5 + cw1h×8 + cache_read×0.20 + output×20) / 1e6，cw5m = cache_creation − cw1h。"""
    usage = FakeUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_creation_input_tokens=3000,
        cache_read_input_tokens=10000,
        cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=2000),
    )
    step = parse_response(_resp([text_block("x")], "end_turn", usage), MODEL)
    assert type(step.input_tokens) is int and step.input_tokens == 14000
    assert type(step.output_tokens) is int and step.output_tokens == 500
    # 1000*4 + 1000*5 + 2000*8 + 10000*0.2 + 500*20 = 4000 + 5000 + 16000 + 2000 + 10000 = 37000
    assert step.cost_usd == pytest.approx(0.037, rel=1e-12, abs=0)


def test_parse_cost_without_breakdown_prices_all_cache_creation_at_5m() -> None:
    usage = FakeUsage(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=3000, cache_read_input_tokens=10000)
    step = parse_response(_resp([text_block("x")], "end_turn", usage), MODEL)
    assert step.input_tokens == 14000
    # 1000*4 + 3000*5 + 10000*0.2 + 500*20 = 4000 + 15000 + 2000 + 10000 = 31000
    assert step.cost_usd == pytest.approx(0.031, rel=1e-12, abs=0)


def test_parse_cost_5m_part_is_cache_creation_minus_1h() -> None:
    """cw5m 依契約由 cache_creation − cw1h 推得（不讀 ephemeral_5m_input_tokens）。"""
    usage = FakeUsage(
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=3_000_000,
        cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=1_000_000),
    )
    step = parse_response(_resp([text_block("x")], "end_turn", usage), MODEL)
    assert step.input_tokens == 3_000_000
    # 2M × 5 + 1M × 8 = 18 USD
    assert step.cost_usd == pytest.approx(18.0, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    ("creation", "one_hour", "read", "expected_cost"),
    [
        (1_000_000, None, None, 4.0 + 5.0),  # 1M input + 1M 5m cache write（沒有細項）
        (1_000_000, 0, None, 4.0 + 5.0),  # 細項存在但 1h 為 0
        (1_000_000, 1_000_000, None, 4.0 + 8.0),  # 全部是 1h cache write
        (None, None, 1_000_000, 4.0 + 0.2),  # 1M input + 1M cache read
        (0, None, 0, 4.0),
    ],
    ids=["5m_no_breakdown", "5m_with_zero_1h", "all_1h", "read_only", "zeros"],
)
def test_parse_cost_per_cache_component(creation: Any, one_hour: Any, read: Any, expected_cost: float) -> None:
    breakdown = None
    if one_hour is not None:
        breakdown = FakeCacheCreation(ephemeral_5m_input_tokens=(creation or 0) - one_hour, ephemeral_1h_input_tokens=one_hour)
    usage = FakeUsage(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_creation_input_tokens=creation,
        cache_read_input_tokens=read,
        cache_creation=breakdown,
    )
    step = parse_response(_resp([text_block("x")], "end_turn", usage), MODEL)
    assert step.input_tokens == 1_000_000 + (creation or 0) + (read or 0)
    assert step.output_tokens == 0
    assert step.cost_usd == pytest.approx(expected_cost, rel=1e-12, abs=0)


def test_parse_usage_none_cache_fields_count_as_zero() -> None:
    usage = FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    step = parse_response(_resp([text_block("x")], "end_turn", usage), MODEL)
    assert step.input_tokens == 1_000_000
    assert step.output_tokens == 1_000_000
    assert step.cost_usd == pytest.approx(24.0, rel=1e-12, abs=0)


def test_parse_usage_missing_cache_attributes_count_as_zero() -> None:
    usage = SimpleNamespace(input_tokens=250, output_tokens=40)
    step = parse_response(_resp([text_block("x")], "end_turn", usage), MODEL)
    assert step.input_tokens == 250
    assert step.output_tokens == 40
    assert step.cost_usd == pytest.approx((250 * 4.0 + 40 * 20.0) / 1_000_000, rel=1e-9, abs=0)


def test_parse_zero_usage_cost_is_zero_not_none() -> None:
    step = parse_response(_resp([text_block("x")], "end_turn", FakeUsage(0, 0)), MODEL)
    assert step.input_tokens == 0
    assert step.output_tokens == 0
    assert step.cost_usd is not None and step.cost_usd == 0


def test_parse_unknown_model_cost_is_none_but_tokens_reported() -> None:
    usage = FakeUsage(
        input_tokens=10,
        output_tokens=3,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=2,
        cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=1, ephemeral_1h_input_tokens=4),
    )
    step = parse_response(_resp([text_block("x")], "end_turn", usage), "claude-unknown-model")
    assert step.cost_usd is None
    assert step.input_tokens == 17
    assert step.output_tokens == 3
    assert step.meta is not None and step.meta["usage"]["cache_creation_1h_input_tokens"] == 4


def test_parse_cost_uses_model_argument_not_response_model() -> None:
    usage = FakeUsage(input_tokens=1_000_000, output_tokens=0)
    known = parse_response(_resp([text_block("x")], "end_turn", usage, model="something-else"), MODEL)
    assert known.cost_usd == pytest.approx(4.0, rel=1e-12, abs=0)
    unknown = parse_response(_resp([text_block("x")], "end_turn", usage, model=MODEL), "not-in-pricing")
    assert unknown.cost_usd is None


def test_parse_cost_looks_up_pricing_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PRICING,
        "fake-priced-model",
        {"input": 1.0, "output": 2.0, "cache_write": 3.0, "cache_write_1h": 7.0, "cache_read": 0.5},
    )
    usage = FakeUsage(
        input_tokens=100,
        output_tokens=10,
        cache_creation_input_tokens=50,
        cache_read_input_tokens=40,
        cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=30, ephemeral_1h_input_tokens=20),
    )
    step = parse_response(_resp([text_block("x")], "end_turn", usage), "fake-priced-model")
    # 100*1 + 30*3 + 20*7 + 40*0.5 + 10*2 = 100 + 90 + 140 + 20 + 20 = 370
    assert step.cost_usd == pytest.approx(370 / 1_000_000, rel=1e-12, abs=0)
    assert step.input_tokens == 190


def test_parse_usage_on_refusal_is_reported() -> None:
    usage = FakeUsage(input_tokens=7, output_tokens=2)
    step = parse_response(_resp([text_block("no")], "refusal", usage), MODEL)
    assert step.input_tokens == 7
    assert step.output_tokens == 2
    assert step.cost_usd == pytest.approx((7 * 4.0 + 2 * 20.0) / 1_000_000, rel=1e-9, abs=0)


# ---------------------------------------------------------------- round trip parse -> conversation -> build


def test_parsed_provider_content_is_replayed_verbatim_in_next_request() -> None:
    resp = FakeResponse(
        content=[
            thinking_block("sig-round", "plan"),
            redacted_thinking_block("RED"),
            text_block("Looking."),
            tool_use_block("toolu_rt", "get_issue", {"issue_id": 1}),
        ],
        stop_reason="tool_use",
        usage=FakeUsage(5, 5),
    )
    step = parse_response(resp, MODEL)
    messages = _base() + [_assistant_from_message(step.message), _tool("toolu_rt", OK_OBS)]
    req = build_request(messages, TOOL_SPECS, _baseline_mc())
    assert req["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig-round"},
            {"type": "redacted_thinking", "data": "RED"},
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "toolu_rt", "name": "get_issue", "input": {"issue_id": 1}},
        ],
    }
    assert req["messages"][2] == _results(("toolu_rt", OK_OBS, False))


def test_parsed_multi_tool_message_passes_consistency_check() -> None:
    """adapter 自己產生的 provider_content 必定通過 build_request 的一致性檢查（text 分散、多個 tool_use）。"""
    resp = _resp(
        [
            thinking_block("sig-m"),
            text_block("First "),
            tool_use_block("toolu_m1", "update_issue", {"status": "closed", "issue_id": 1}),
            text_block("then second."),
            tool_use_block("toolu_m2", "add_comment", {"issue_id": 1, "body": "done"}),
        ],
        "tool_use",
    )
    step = parse_response(resp, MODEL)
    messages = _base() + [
        _assistant_from_message(step.message),
        _tool("toolu_m1", OK_OBS),
        _tool("toolu_m2", ERR_OBS),
    ]
    req = build_request(messages, TOOL_SPECS, _baseline_mc())
    assert canonical_json(req["messages"][1]["content"]) == canonical_json(step.message.provider_content)
    assert req["messages"][2] == _results(("toolu_m1", OK_OBS, False), ("toolu_m2", ERR_OBS, True))


def test_scripted_policy_messages_are_rejected_by_build_request() -> None:
    """修訂：舊的「provider_content 為 None 時由 text + tool_calls 組出」fallback 已移除。"""
    message = ModelMessage(
        text="", tool_calls=(ToolCall(id="call_0_0", name="get_issue", args={"issue_id": 1}),), stop_reason="tool_use"
    )
    messages = _base() + [_assistant_from_message(message), _tool("call_0_0", OK_OBS)]
    with pytest.raises(ValueError):
        build_request(messages, TOOL_SPECS, _baseline_mc())


# ---------------------------------------------------------------- AnthropicPolicy: construction


class _ClientWithSettings(FakeClient):
    """模擬真正 SDK client 的 max_retries / timeout 屬性（只設定傳入的那些）。"""

    def __init__(self, responses: list[FakeResponse] | None = None, **settings: Any) -> None:
        super().__init__(responses)
        for key, value in settings.items():
            setattr(self, key, value)


class _BetaOnlyClient:
    """只有 beta namespace 會記錄 request；呼叫非 beta 的 messages.create 立即失敗。"""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(responses))
        self.messages = SimpleNamespace(create=self._forbidden)

    @staticmethod
    def _forbidden(**request: Any) -> Any:
        raise AssertionError("adapter must call client.beta.messages.create")

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.beta.messages.requests


def test_policy_default_agent_config_is_baseline() -> None:
    client = FakeClient()
    policy = AnthropicPolicy(client)
    assert policy.agent_config.config_id == baseline_agent_config().config_id
    assert client.requests == []


def test_policy_agent_config_is_fresh_and_stable() -> None:
    policy = AnthropicPolicy(FakeClient())
    a = policy.agent_config
    b = policy.agent_config
    assert isinstance(a, AgentConfig)
    assert a is not b
    assert a.model_config is not b.model_config
    assert a.config_id == b.config_id
    a.model_config["effort"] = "low"
    a.model_config["betas"].clear()
    assert policy.agent_config.model_config["effort"] == "medium"
    assert policy.agent_config.model_config["betas"] == [BETA]
    assert policy.agent_config.config_id == b.config_id


def test_policy_agent_config_not_affected_by_caller_mutation() -> None:
    cfg = baseline_agent_config()
    original_id = cfg.config_id
    policy = AnthropicPolicy(FakeClient(), cfg)
    cfg.model_config["effort"] = "low"
    assert policy.agent_config.config_id == original_id
    assert policy.agent_config.model_config["effort"] == "medium"


def test_policy_accepts_custom_anthropic_config() -> None:
    mc = _baseline_mc()
    mc["effort"] = "high"
    cfg = _cfg(prompt_version="baseline/test", system_prompt="s", developer_prompt="", model_config=mc)
    policy = AnthropicPolicy(FakeClient(), cfg)
    assert policy.agent_config.config_id == cfg.config_id


@pytest.mark.parametrize("policy_name", [SCRIPTED_POLICY, "anthropic", "openai-chat", "Anthropic-Messages"])
def test_policy_rejects_wrong_policy_name(policy_name: str) -> None:
    client = FakeClient()
    with pytest.raises(ValueError):
        AnthropicPolicy(client, _cfg(policy=policy_name))
    assert client.requests == []


@pytest.mark.parametrize("missing", sorted(EXPECTED_MODEL_CONFIG))
def test_policy_rejects_model_config_missing_key(missing: str) -> None:
    mc = _baseline_mc()
    del mc[missing]
    with pytest.raises(ValueError):
        AnthropicPolicy(FakeClient(), _cfg(model_config=mc))


@pytest.mark.parametrize(
    ("extra_key", "value"),
    [("temperature", 1.0), ("seed", 0), ("top_p", 0.9), ("stop_sequences", ["x"]), ("metadata", None)],
)
def test_policy_rejects_model_config_extra_key(extra_key: str, value: Any) -> None:
    """model_config 的 key 集合必須剛好等於 BASELINE_MODEL_CONFIG 的（多出的設定不能被默默忽略）。"""
    mc = _baseline_mc()
    mc[extra_key] = value
    with pytest.raises(ValueError):
        AnthropicPolicy(FakeClient(), _cfg(model_config=mc))


@pytest.mark.parametrize("adapter", ["anthropic-messages/v0", "anthropic-messages/v2", "", "anthropic-messages", None])
def test_policy_rejects_adapter_version_mismatch(adapter: Any) -> None:
    mc = _baseline_mc()
    mc["adapter"] = adapter
    with pytest.raises(ValueError):
        AnthropicPolicy(FakeClient(), _cfg(model_config=mc))


@pytest.mark.parametrize("system_prompt", ["", " ", "\n\t  \n"], ids=["empty", "space", "whitespace"])
def test_policy_rejects_blank_system_prompt(system_prompt: str) -> None:
    client = FakeClient()
    with pytest.raises(ValueError):
        AnthropicPolicy(client, _cfg(system_prompt=system_prompt))
    assert client.requests == []


@pytest.mark.parametrize("developer_prompt", ["", "  \n"], ids=["empty", "whitespace"])
def test_policy_accepts_blank_developer_prompt(developer_prompt: str) -> None:
    cfg = _cfg(developer_prompt=developer_prompt)
    assert AnthropicPolicy(FakeClient(), cfg).agent_config.config_id == cfg.config_id


@pytest.mark.parametrize(
    "settings",
    [{}, {"max_retries": 4, "timeout": 600}, {"max_retries": 4}, {"timeout": 600}],
    ids=["no_attributes", "both_match", "max_retries_only", "timeout_only"],
)
def test_policy_accepts_client_with_matching_or_absent_settings(settings: dict[str, Any]) -> None:
    client = _ClientWithSettings(**settings)
    policy = AnthropicPolicy(client)
    assert policy.agent_config.config_id == baseline_agent_config().config_id
    assert client.requests == []


@pytest.mark.parametrize(
    "settings",
    [
        {"max_retries": 2, "timeout": 600},
        {"max_retries": 4, "timeout": 120},
        {"max_retries": 0, "timeout": 60},
        {"max_retries": 2},
        {"timeout": 120},
        {"max_retries": None, "timeout": 600},
        {"max_retries": 4, "timeout": None},
    ],
    ids=["retries_sdk_default", "timeout_120", "both_wrong", "retries_only_wrong", "timeout_only_wrong", "retries_none", "timeout_none"],
)
def test_policy_rejects_client_with_mismatched_settings(settings: dict[str, Any]) -> None:
    client = _ClientWithSettings(**settings)
    with pytest.raises(ValueError):
        AnthropicPolicy(client)
    assert client.requests == []


def test_policy_client_settings_are_compared_with_its_own_model_config() -> None:
    mc = _baseline_mc()
    mc["client"] = {"max_retries": 1, "timeout_s": 7}
    cfg = _cfg(model_config=mc)
    AnthropicPolicy(_ClientWithSettings(max_retries=1, timeout=7), cfg)
    with pytest.raises(ValueError):
        AnthropicPolicy(_ClientWithSettings(max_retries=4, timeout=600), cfg)


# ---------------------------------------------------------------- AnthropicPolicy: next_message


def test_policy_calls_beta_messages_create() -> None:
    client = _BetaOnlyClient([final_turn("ok")])
    step = AnthropicPolicy(client).next_message(_base(), TOOL_SPECS)
    assert step.message.text == "ok"
    assert len(client.requests) == 1
    assert client.requests[0]["betas"] == [BETA]


def test_policy_next_message_calls_create_once_with_build_request() -> None:
    response = tool_turn("toolu_9", "get_issue", {"issue_id": 1}, signature="sig-n", note="n")
    client = FakeClient([response])
    policy = AnthropicPolicy(client)
    messages = _provider_conversation()
    step = policy.next_message(messages, TOOL_SPECS)
    assert len(client.requests) == 1
    expected = build_request(messages, TOOL_SPECS, baseline_agent_config().model_config)
    assert canonical_json(client.requests[0]) == canonical_json(expected)
    assert set(client.requests[0]) == REQUEST_KEYS
    assert step == parse_response(response, MODEL)


def test_policy_next_message_returns_parsed_step_with_meta() -> None:
    usage = FakeUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_creation_input_tokens=3000,
        cache_read_input_tokens=10000,
        cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=2000),
    )
    client = FakeClient([tool_turn("toolu_1", "get_issue", {"issue_id": 1}, signature="sig-p", usage=usage)])
    step = AnthropicPolicy(client).next_message(_base(), TOOL_SPECS)
    assert isinstance(step, PolicyStep)
    assert step.message.stop_reason == "tool_use"
    assert step.message.tool_calls == (ToolCall(id="toolu_1", name="get_issue", args={"issue_id": 1}),)
    assert step.message.provider_content == [
        {"type": "thinking", "thinking": "", "signature": "sig-p"},
        {"type": "tool_use", "id": "toolu_1", "name": "get_issue", "input": {"issue_id": 1}},
    ]
    assert step.input_tokens == 14000
    assert step.output_tokens == 500
    assert step.cost_usd == pytest.approx(0.037, rel=1e-12, abs=0)
    assert step.meta is not None
    assert step.meta["request_id"] == "req_fake"
    assert step.meta["tool_use_blocks"] == 1
    assert step.meta["usage"]["cache_creation_1h_input_tokens"] == 2000


def test_policy_uses_its_own_model_config() -> None:
    mc = _baseline_mc()
    mc.update(model="claude-test-unpriced", effort="high", max_tokens=777, betas=["beta-x"])
    cfg = _cfg(prompt_version="baseline/test", system_prompt="s", developer_prompt="d", model_config=mc)
    client = FakeClient([final_turn("ok")])
    step = AnthropicPolicy(client, cfg).next_message(_base(), TOOL_SPECS)
    req = client.requests[0]
    assert req["model"] == "claude-test-unpriced"
    assert req["max_tokens"] == 777
    assert req["output_config"] == {"effort": "high"}
    assert req["betas"] == ["beta-x"]
    assert canonical_json(req) == canonical_json(build_request(_base(), TOOL_SPECS, mc))
    assert step.cost_usd is None  # 成本以 config 的 model 查表；不在 PRICING
    assert step.meta is not None and step.meta["response_model"] == MODEL


def test_policy_client_exception_propagates_unchanged() -> None:
    class Boom(Exception):
        pass

    err = Boom("http 400")

    def responder(request: dict[str, Any]) -> FakeResponse:
        raise err

    client = FakeClient(responder=responder)
    with pytest.raises(Boom) as excinfo:
        AnthropicPolicy(client).next_message(_base(), TOOL_SPECS)
    assert excinfo.value is err
    assert len(client.requests) == 1


def test_policy_adapter_error_propagates() -> None:
    client = FakeClient([_resp([text_block("trunc")], "max_tokens")])
    with pytest.raises(AdapterError):
        AnthropicPolicy(client).next_message(_base(), TOOL_SPECS)
    assert len(client.requests) == 1


def test_policy_replay_integrity_error_propagates() -> None:
    resp = _resp(
        [thinking_block("s"), text_block("ok")],
        "end_turn",
        input_transformations=[SimpleNamespace(type="thinking_dropped", path="messages.1.content.0", reason="x")],
    )
    client = FakeClient([resp])
    with pytest.raises(ReplayIntegrityError):
        AnthropicPolicy(client).next_message(_provider_conversation(), TOOL_SPECS)
    assert len(client.requests) == 1


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda c: c[0])
def test_policy_invalid_messages_raise_before_calling_client(case: tuple[str, Callable[[], list[Any]]]) -> None:
    _, make = case
    client = FakeClient([final_turn()])
    with pytest.raises(ValueError):
        AnthropicPolicy(client).next_message(make(), TOOL_SPECS)
    assert client.requests == []


def test_policy_empty_tools_raise_before_calling_client() -> None:
    client = FakeClient([final_turn()])
    with pytest.raises(ValueError):
        AnthropicPolicy(client).next_message(_base(), ())
    assert client.requests == []


def test_policy_is_stateless() -> None:
    """相同 messages → 相同 request；中間插入其他 conversation 不影響。"""
    client = FakeClient([final_turn("a"), final_turn("b"), final_turn("c")])
    policy = AnthropicPolicy(client)
    conv_a = _provider_conversation()
    conv_b = _multi_conversation()
    policy.next_message(conv_a, TOOL_SPECS)
    policy.next_message(conv_b, TOOL_SPECS)
    policy.next_message(copy.deepcopy(conv_a), TOOL_SPECS)
    assert len(client.requests) == 3
    assert canonical_json(client.requests[0]) == canonical_json(client.requests[2])
    assert canonical_json(client.requests[1]) == canonical_json(build_request(conv_b, TOOL_SPECS, _baseline_mc()))


def test_policy_does_not_mutate_inputs() -> None:
    client = FakeClient([final_turn()])
    messages = _multi_conversation()
    tools = copy.deepcopy(TOOL_SPECS)
    before = (copy.deepcopy(messages), copy.deepcopy(tools))
    AnthropicPolicy(client).next_message(messages, tools)
    assert messages == before[0]
    assert tools == before[1]


def test_policy_fake_episode_request_prefixes_are_stable() -> None:
    """以 fake client 跑兩個 turn：第二次 request 的 messages 以第一次為前綴，system / tools / thinking / betas 逐 byte 相同。"""
    client = FakeClient(
        [tool_turn("toolu_1", "get_issue", {"issue_id": 1}, signature="sig-1", note="look"), final_turn("done")]
    )
    policy = AnthropicPolicy(client)
    cfg = policy.agent_config
    messages = _base(cfg.system_prompt, cfg.developer_prompt, INSTR)
    first = policy.next_message(messages, TOOL_SPECS)
    messages = messages + [_assistant_from_message(first.message), _tool("toolu_1", OK_OBS)]
    second = policy.next_message(messages, TOOL_SPECS)
    assert second.message.tool_calls == ()
    r0, r1 = client.requests
    n = len(r0["messages"])
    assert canonical_json(r1["messages"][:n]) == canonical_json(r0["messages"])
    for key in ("system", "tools", "thinking", "betas", "tool_choice", "output_config", "cache_control", "model", "max_tokens"):
        assert canonical_json(r1[key]) == canonical_json(r0[key])
    assert r0["system"] == [
        {"type": "text", "text": BASELINE_SYSTEM_PROMPT},
        {"type": "text", "text": BASELINE_DEVELOPER_PROMPT},
    ]
    assert r0["messages"] == [{"role": "user", "content": INSTR}]
    assert r1["messages"][n] == {"role": "assistant", "content": first.message.provider_content}
    assert r1["messages"][n + 1] == _results(("toolu_1", OK_OBS, False))


# ---------------------------------------------------------------- make_anthropic_client


def _fake_anthropic_module(calls: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> types.ModuleType:
    module = types.ModuleType("anthropic")

    class Anthropic:  # fake SDK client：記錄建構參數，並像 SDK 一樣把 max_retries / timeout 存成屬性
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, copy.deepcopy(kwargs)))
            self.max_retries = kwargs.get("max_retries", 2)
            self.timeout = kwargs.get("timeout", 600)
            self.beta = SimpleNamespace(messages=FakeMessages([]))
            self.messages = None

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    return module


def test_make_client_passes_only_retry_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fake = _fake_anthropic_module(calls)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    client = make_anthropic_client()
    assert isinstance(client, fake.Anthropic)  # type: ignore[attr-defined]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs == {"max_retries": 4, "timeout": 600}


def test_make_client_never_forwards_credentials_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """credential 完全交給 SDK 解析：即使環境變數存在，本函式也不讀取、不傳入。"""
    sentinel = "sentinel-not-a-real-credential"
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", sentinel)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(calls))
    make_anthropic_client()
    args, kwargs = calls[0]
    assert sentinel not in args
    assert sentinel not in json.dumps(kwargs, default=str)
    assert set(kwargs) == {"max_retries", "timeout"}


def test_make_client_reads_client_settings_from_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(calls))
    mc = _baseline_mc()
    mc["client"] = {"max_retries": 1, "timeout_s": 7}
    before = copy.deepcopy(mc)
    make_anthropic_client(mc)
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs == {"max_retries": 1, "timeout": 7}
    assert mc == before


def test_make_client_result_passes_policy_client_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """實驗腳本應一律以 make_anthropic_client 建立 client：其 max_retries / timeout 屬性通過 AnthropicPolicy 的檢查。"""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(calls))
    client = make_anthropic_client()
    assert client.max_retries == 4
    assert client.timeout == 600
    policy = AnthropicPolicy(client)
    assert policy.agent_config.config_id == baseline_agent_config().config_id
    mc = _baseline_mc()
    mc["client"] = {"max_retries": 1, "timeout_s": 7}
    with pytest.raises(ValueError):
        AnthropicPolicy(make_anthropic_client(mc))  # client 設定與 baseline config 不一致


def test_make_client_without_sdk_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError):
        make_anthropic_client()


# ---------------------------------------------------------------- SDK compatibility (no client construction)


def _literal_values(tp: Any) -> set[Any]:
    """遞迴收集 Literal / Union / Optional / Annotated 中的所有字面值。"""
    values: set[Any] = set()
    for arg in typing.get_args(tp):
        if isinstance(arg, (str, int, bool)) or arg is None:
            values.add(arg)
        else:
            values |= _literal_values(arg)
    return values


def _hints(typed_dict: Any) -> dict[str, Any]:
    return typing.get_type_hints(typed_dict, include_extras=True)


def _required_keys(typed_dict: Any) -> set[str]:
    import typing_extensions

    return {
        key
        for key, hint in _hints(typed_dict).items()
        if typing.get_origin(hint) in (typing.Required, typing_extensions.Required)
    }


def test_build_request_keys_are_beta_messages_create_parameters() -> None:
    pytest.importorskip("anthropic")
    from anthropic.resources.beta.messages import Messages

    params = inspect.signature(Messages.create).parameters
    assert "betas" in params
    for conversation in CONVERSATIONS:
        req = build_request(conversation(), TOOL_SPECS, copy.deepcopy(BASELINE_MODEL_CONFIG))
        missing = [key for key in req if key not in params]
        assert missing == [], f"request keys not accepted by beta Messages.create: {missing}"


def test_thinking_config_matches_sdk_beta_types() -> None:
    pytest.importorskip("anthropic")
    from anthropic.types.beta import BetaThinkingBlockBindingParam, BetaThinkingConfigAdaptiveParam

    thinking = BASELINE_MODEL_CONFIG["thinking"]
    hints = _hints(BetaThinkingConfigAdaptiveParam)
    assert set(thinking) <= set(hints)
    assert thinking["type"] in _literal_values(hints["type"])
    assert thinking["display"] in _literal_values(hints["display"])
    binding_hints = _hints(BetaThinkingBlockBindingParam)
    assert set(thinking["block_binding"]) <= set(binding_hints)
    assert thinking["block_binding"]["prefix_mismatch_behavior"] in _literal_values(binding_hints["prefix_mismatch_behavior"])


def test_betas_are_known_to_sdk() -> None:
    pytest.importorskip("anthropic")
    from anthropic.types.anthropic_beta_param import AnthropicBetaParam

    known = _literal_values(AnthropicBetaParam)
    for beta in BASELINE_MODEL_CONFIG["betas"]:
        assert beta in known


def test_effort_and_tool_choice_match_sdk_beta_types() -> None:
    pytest.importorskip("anthropic")
    from anthropic.types.beta import BetaOutputConfigParam, BetaToolChoiceAutoParam

    req = build_request(_base(), TOOL_SPECS, copy.deepcopy(BASELINE_MODEL_CONFIG))
    output_hints = _hints(BetaOutputConfigParam)
    assert set(req["output_config"]) <= set(output_hints)
    assert req["output_config"]["effort"] in _literal_values(output_hints["effort"])
    choice_hints = _hints(BetaToolChoiceAutoParam)
    assert set(req["tool_choice"]) <= set(choice_hints)
    assert req["tool_choice"]["type"] in _literal_values(choice_hints["type"])


def test_allowed_block_fields_match_sdk_block_params() -> None:
    """白名單欄位必須涵蓋 SDK param 的所有必填欄位，且不得超出 SDK 接受的欄位。"""
    pytest.importorskip("anthropic")
    from anthropic.types.beta import (
        BetaRedactedThinkingBlockParam,
        BetaTextBlockParam,
        BetaThinkingBlockParam,
        BetaToolUseBlockParam,
    )

    params = {
        "text": BetaTextBlockParam,
        "thinking": BetaThinkingBlockParam,
        "redacted_thinking": BetaRedactedThinkingBlockParam,
        "tool_use": BetaToolUseBlockParam,
    }
    assert set(params) == set(ALLOWED_BLOCK_TYPES)
    for block_type, param in params.items():
        fields = set(ALLOWED_BLOCK_TYPES[block_type])
        assert _required_keys(param) <= fields, block_type
        assert fields <= set(_hints(param)), block_type


def test_tool_result_blocks_match_sdk_param() -> None:
    pytest.importorskip("anthropic")
    from anthropic.types.beta import BetaToolResultBlockParam

    req = build_request(_multi_conversation(), TOOL_SPECS, copy.deepcopy(BASELINE_MODEL_CONFIG))
    allowed = set(_hints(BetaToolResultBlockParam))
    for message in req["messages"]:
        if message["role"] == "user" and isinstance(message["content"], list):
            for block in message["content"]:
                assert set(block) <= allowed


def test_response_attributes_exist_on_sdk_beta_message() -> None:
    pytest.importorskip("anthropic")
    from anthropic.types.beta import BetaCacheCreation, BetaMessage, BetaRefusalStopDetails, BetaUsage

    assert {"content", "stop_reason", "usage", "model", "stop_details", "input_transformations"} <= set(BetaMessage.model_fields)
    assert {"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "cache_creation"} <= set(
        BetaUsage.model_fields
    )
    assert "ephemeral_1h_input_tokens" in BetaCacheCreation.model_fields
    assert {"category", "explanation"} <= set(BetaRefusalStopDetails.model_fields)


def test_every_sdk_stop_reason_is_handled_per_contract() -> None:
    """SDK 的每個 stop_reason：只有 tool_use / end_turn / refusal 合法，其他一律 AdapterError。"""
    pytest.importorskip("anthropic")
    from anthropic.types.beta.beta_stop_reason import BetaStopReason

    reasons = _literal_values(BetaStopReason)
    assert {"tool_use", "end_turn", "refusal"} <= reasons
    for reason in sorted(reasons - {"tool_use", "end_turn", "refusal"}):
        with pytest.raises(AdapterError):
            parse_response(_resp([thinking_block("s"), text_block("x")], reason), MODEL)


def _sdk_message(**overrides: Any) -> Any:
    from anthropic.types.beta import BetaMessage

    data: dict[str, Any] = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig-sdk"},
            {"type": "text", "text": "Looking.", "citations": None},
            {"type": "tool_use", "id": "toolu_sdk", "name": "get_issue", "input": {"issue_id": 1}, "caller": {"type": "direct"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
            "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 2},
        },
    }
    data.update(overrides)
    return BetaMessage.model_validate(data)


def test_parse_real_sdk_beta_message_offline() -> None:
    """以 SDK 的 BetaMessage（離線 model_validate，沒有 _request_id）驗證屬性讀取與白名單。"""
    pytest.importorskip("anthropic")
    step = parse_response(_sdk_message(), MODEL)
    assert step.message.provider_content == [
        {"type": "thinking", "thinking": "plan", "signature": "sig-sdk"},
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "toolu_sdk", "name": "get_issue", "input": {"issue_id": 1}},
    ]
    assert step.message.tool_calls == (ToolCall(id="toolu_sdk", name="get_issue", args={"issue_id": 1}),)
    assert step.message.text == "Looking."
    assert step.input_tokens == 15
    assert step.output_tokens == 5
    # 10*4 + 1*5 + 2*8 + 2*0.2 + 5*20 = 40 + 5 + 16 + 0.4 + 100 = 161.4
    assert step.cost_usd == pytest.approx(161.4 / 1_000_000, rel=1e-12, abs=0)
    assert step.meta == {
        "response_model": "claude-opus-5-5",
        "request_id": None,
        "stop_reason": "tool_use",
        "stop_details": None,
        "tool_use_blocks": 1,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
            "cache_creation_1h_input_tokens": 2,
        },
    }
    _assert_json_native(step.meta)
    _assert_json_native(step.message.provider_content)


def test_parse_real_sdk_refusal_stop_details_offline() -> None:
    pytest.importorskip("anthropic")
    message = _sdk_message(
        content=[{"type": "text", "text": "I can't help with that.", "citations": None}],
        stop_reason="refusal",
        stop_details={
            "type": "refusal",
            "category": "cyber",
            "explanation": "x",
            "fallback_credit_token": "tok",
            "recommended_model": None,
        },
    )
    step = parse_response(message, MODEL)
    assert step.message.text == ""
    assert step.message.tool_calls == ()
    assert step.message.provider_content == [{"type": "text", "text": "I can't help with that."}]
    assert step.meta is not None
    assert step.meta["stop_details"] == {"category": "cyber", "explanation": "x"}


def test_parse_real_sdk_input_transformations_offline() -> None:
    pytest.importorskip("anthropic")
    message = _sdk_message(
        input_transformations=[
            {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "model_binding_mismatch"}
        ]
    )
    with pytest.raises(ReplayIntegrityError):
        parse_response(message, MODEL)


def test_client_settings_are_anthropic_constructor_parameters() -> None:
    anthropic = pytest.importorskip("anthropic")
    params = inspect.signature(anthropic.Anthropic.__init__).parameters
    assert "max_retries" in params
    assert "timeout" in params
    # AnthropicPolicy 以 client.timeout 屬性檢查設定：SDK client 類別上有 timeout property（不建構 client）
    assert isinstance(inspect.getattr_static(anthropic.Anthropic, "timeout"), property)
