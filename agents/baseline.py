"""Week 1 #5 baseline agent：固定設定與 Anthropic Messages API adapter（decision log 009）。

``AnthropicPolicy`` 以**可注入的 client** 實作 ``agents.policy.Policy``：client 只需要提供
``client.beta.messages.create(**request) -> response``（真正的 ``anthropic.Anthropic`` 或測試用 fake client）。
本模組不讀取、不要求、也不儲存任何 credential；``make_anthropic_client`` 只把 credential 解析交給 SDK。

Request（``build_request``，決定性、JSON-native 的新物件，不修改輸入）::

    {"model": mc["model"], "max_tokens": mc["max_tokens"],
     "system": [{"type": "text", "text": system_prompt}, {"type": "text", "text": developer_prompt}],
     "messages": [...], "tools": [{"name", "description", "input_schema"}, ...],
     "tool_choice": mc["tool_choice"], "thinking": mc["thinking"],
     "output_config": {"effort": mc["effort"]}, "cache_control": {"type": "ephemeral"},
     "betas": mc["betas"]}

messages 的檢查與轉換（違反 → ValueError，且不呼叫 client）：
- 開頭必須是 system、developer、user 三則（role 依序）；system 內容純空白 → ValueError；developer 純空白時不送該區塊；
  user 內容純空白 → ValueError。user 訊息為 ``{"role": "user", "content": <instruction 字串>}``。
- 之後必須是 (assistant, tool × n) 的重複，n = 該 assistant 的 tool_calls 數（>= 1：沒有 tool call 的訊息代表 episode 已結束，
  不會再送出 request），tool_call_id 依序相同；tool content 必須是含 bool ``"ok"`` 的 JSON 物件字串。
- 每則 assistant 必須有 ``provider_content`` key 且為**非空** list；其中 block 只能是 ``ALLOWED_BLOCK_TYPES`` 且剛好有規定的 key、
  型別正確（text / thinking / signature / data / id / name 為 str）；tool_use blocks 依序與 tool_calls 逐一相同（id、name、
  ``canonical_json(input) == canonical_json(args)``）；text blocks 依序串接 == content。通過後 provider_content **原樣**作為 content。
- 連續的 tool 訊息合併成一則 user 訊息：每則一個
  ``{"type": "tool_result", "tool_use_id": ..., "content": <原 content 字串>, "is_error": <observation 的 ok 為 False>}``。
- tools 不可為空；只保留 name / description / input_schema，依傳入順序。

Response → PolicyStep（``parse_response``；以屬性讀取 ``stop_reason``、``content``、``usage``、``model``、``stop_details``、
``input_transformations``、``_request_id``，缺少的選用屬性視為 None）：
- ``input_transformations`` 非空 → ``ReplayIntegrityError``。
- 每個 block 依白名單轉成只含規定 key 的 dict（以屬性讀取，不呼叫 SDK 序列化）；未知型別或欄位型別錯誤 → ``AdapterError``。
- ``tool_use``：至少一個 tool_use block，否則 ``AdapterError``；tool_calls = tool_use blocks 依序（``ToolCall(id, name, input)``）；
  text = text blocks 依序串接。``end_turn``：不可有 tool_use block（否則 ``AdapterError``）；tool_calls = ()。
  ``refusal``：text = ""、tool_calls = ()、provider_content 保留全部 blocks。其他（含 None）→ ``AdapterError``。
- ModelMessage.stop_reason = response.stop_reason；provider_content = 轉換後的 blocks。
- 用量：input_tokens = input + cache_creation + cache_read；output_tokens = output（None / 缺少視為 0）；
  ``cost_usd = (input×p.input + cw5m×p.cache_write + cw1h×p.cache_write_1h + cache_read×p.cache_read + output×p.output) / 1e6``，
  其中 cw1h = ``usage.cache_creation.ephemeral_1h_input_tokens``（沒有細項時為 0）、cw5m = cache_creation − cw1h；
  model 不在 PRICING → cost_usd = None。
- ``PolicyStep.meta`` = ``{"response_model", "request_id", "stop_reason", "stop_details", "tool_use_blocks",
  "usage": {"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
  "cache_creation_1h_input_tokens"}}``；stop_details 為 ``{"category", "explanation"}`` 或 None。
"""

from __future__ import annotations

import copy
import json
from typing import Any

from agents.policy import MAX_MODEL_CONFIG_DEPTH, AgentConfig, ModelMessage, PolicyStep, ToolCall, _check_json
from env.hashing import canonical_json
from replay.trajectory import check_json_document

ANTHROPIC_POLICY = "anthropic-messages"
ADAPTER_VERSION = "anthropic-messages/v1"
BASELINE_PROMPT_VERSION = "baseline/v1"
BASELINE_MAX_TURNS = 20
MAX_EPISODE_ATTEMPTS = 2

BASELINE_SYSTEM_PROMPT = (
    "You are Triage Bot, an automation account that works in a team's issue tracker. "
    "You can act only through the provided tools. Complete the user's request, then reply with a brief "
    "summary of what you did. Each tool result is a JSON object: {\"ok\": true, \"data\": ...} on success, "
    "or {\"ok\": false, \"error\": {\"code\": ..., \"message\": ...}} on failure."
)

BASELINE_DEVELOPER_PROMPT = (
    "Operating notes: issues are referred to as #N, which is issue_id N in the tools; users are referred to by "
    "ids such as u2. Call at most one tool per message. This is a non-interactive run: nobody will answer "
    "questions, and a reply without a tool call ends the task. When the request is complete, stop calling tools "
    "and give the summary."
)

BASELINE_MODEL_CONFIG: dict[str, Any] = {
    "provider": "anthropic",
    "adapter": ADAPTER_VERSION,
    "model": "claude-opus-5-5",
    "max_tokens": 16000,
    "thinking": {"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "error"}},
    "betas": ["thinking-binding-controls-2026-08-01"],
    "effort": "medium",
    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    "prompt_caching": "auto",
    "fallbacks": None,
    "client": {"max_retries": 4, "timeout_s": 600},
}

# 每百萬 tokens 的美元價格（claude-api skill 模型表，快取日期 2026-10-06）。只用於量測成本，不屬於 config。
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_write_1h": 8.00, "cache_read": 0.20},
}

ALLOWED_BLOCK_TYPES: dict[str, tuple[str, ...]] = {
    "text": ("type", "text"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
    "tool_use": ("type", "id", "name", "input"),
}


class AdapterError(RuntimeError):
    """模型自己的輸出無法成為合法 turn（截斷、未知 stop_reason / block 等）；decision 009 的 agent_output 類，不重試。"""


class ReplayIntegrityError(RuntimeError):
    """API 回報重放不忠實（例如 input_transformations 非空）；decision 009 的 harness 類，實驗必須停止。"""


def baseline_agent_config() -> AgentConfig:
    """decision 009 的固定 baseline：policy=ANTHROPIC_POLICY、prompt_version=BASELINE_PROMPT_VERSION、
    兩段 prompt、model_config=BASELINE_MODEL_CONFIG（每次回傳新物件）。
    """
    return AgentConfig(
        policy=ANTHROPIC_POLICY,
        prompt_version=BASELINE_PROMPT_VERSION,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        developer_prompt=BASELINE_DEVELOPER_PROMPT,
        model_config=copy.deepcopy(BASELINE_MODEL_CONFIG),
    )


def build_request(messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...], model_config: dict[str, Any]) -> dict[str, Any]:
    """依模組 docstring 組出 Messages API request。messages 與每個 tool 必須是 JSON 文件（list / dict、深度不超過
    MAX_DOCUMENT_DEPTH；以迭代方式在任何遞迴處理之前檢查），否則 ValueError。"""
    check_json_document(messages, error=ValueError)
    if not isinstance(tools, (list, tuple)):
        raise ValueError("tools must be a list or tuple")
    for tool in tools:
        check_json_document(tool, error=ValueError)
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("messages must be a list starting with system, developer and user messages")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError(f"messages[{index}] must be a dict with a str 'role'")
    for index, role in enumerate(("system", "developer", "user")):
        message = messages[index]
        if message["role"] != role or set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError(f"messages[{index}] must be {{'role': {role!r}, 'content': str}}")
    system_prompt, developer_prompt, instruction = (messages[i]["content"] for i in range(3))
    if _blank(system_prompt):
        raise ValueError("the system prompt must not be blank")
    if _blank(instruction):
        raise ValueError("the user instruction must not be blank")

    out: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    index = 3
    while index < len(messages):
        assistant = messages[index]
        if assistant["role"] != "assistant":
            raise ValueError(f"messages[{index}] must be an assistant message")
        calls = _check_assistant(assistant, index)
        out.append({"role": "assistant", "content": copy.deepcopy(assistant["provider_content"])})
        index += 1
        results: list[dict[str, Any]] = []
        for call in calls:
            if index >= len(messages):
                raise ValueError(f"tool call {call['id']!r} has no tool message")
            results.append(_tool_result(messages[index], call["id"], index))
            index += 1
        out.append({"role": "user", "content": results})

    if not isinstance(tools, (list, tuple)) or len(tools) == 0:
        raise ValueError("tools must be a non-empty list or tuple")
    tool_params: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not all(key in tool for key in ("name", "description", "input_schema")):
            raise ValueError("each tool must be a dict with name, description and input_schema")
        tool_params.append({"name": tool["name"], "description": tool["description"], "input_schema": tool["input_schema"]})

    system = [{"type": "text", "text": system_prompt}]
    if not _blank(developer_prompt):
        system.append({"type": "text", "text": developer_prompt})
    request = {
        "model": model_config["model"],
        "max_tokens": model_config["max_tokens"],
        "system": system,
        "messages": out,
        "tools": tool_params,
        "tool_choice": model_config["tool_choice"],
        "thinking": model_config["thinking"],
        "output_config": {"effort": model_config["effort"]},
        "cache_control": {"type": "ephemeral"},
        "betas": model_config["betas"],
    }
    try:
        canonical_json(request)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request is not a JSON document: {exc}") from exc
    # 深拷貝：不與輸入共用；保留 dict 的插入順序（例如 tool input_schema 的 properties 順序照 TOOL_SPECS 送出）
    return copy.deepcopy(request)


def _blank(text: str) -> bool:
    return text.strip() == ""


def _check_assistant(message: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """歷史中 assistant 訊息的忠實性檢查（provider_content 與 content / tool_calls 一致）；回傳 tool_calls。"""
    if set(message) != {"role", "content", "tool_calls", "provider_content"}:
        raise ValueError(f"messages[{index}] must have keys role, content, tool_calls and provider_content")
    text, calls, blocks = message["content"], message["tool_calls"], message["provider_content"]
    if not isinstance(text, str):
        raise ValueError(f"messages[{index}].content must be a str")
    if not isinstance(calls, list) or len(calls) == 0:
        raise ValueError(f"messages[{index}] must have at least one tool call (a final answer ends the episode)")
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "name", "args"}:
            raise ValueError(f"messages[{index}].tool_calls must contain {{id, name, args}} dicts")
        if not isinstance(call["id"], str) or not isinstance(call["name"], str):
            raise ValueError(f"messages[{index}].tool_calls id and name must be str")
    if not isinstance(blocks, list) or len(blocks) == 0:
        raise ValueError(f"messages[{index}].provider_content must be a non-empty list (provider history is required)")
    for block in blocks:
        _check_block(block, index)
    tool_uses = [block for block in blocks if block["type"] == "tool_use"]
    if len(tool_uses) != len(calls):
        raise ValueError(f"messages[{index}] provider_content does not match its tool calls")
    for block, call in zip(tool_uses, calls):
        if block["id"] != call["id"] or block["name"] != call["name"] or not _same_json(block["input"], call["args"]):
            raise ValueError(f"messages[{index}] tool_use block {block['id']!r} does not match its tool call")
    if "".join(block["text"] for block in blocks if block["type"] == "text") != text:
        raise ValueError(f"messages[{index}] provider_content text does not match content")
    return calls


def _check_block(block: Any, index: int) -> None:
    kind = block.get("type") if isinstance(block, dict) else None
    if not isinstance(kind, str) or kind not in ALLOWED_BLOCK_TYPES:
        raise ValueError(f"messages[{index}].provider_content contains an unsupported block")
    if set(block) != set(ALLOWED_BLOCK_TYPES[block["type"]]):
        raise ValueError(f"messages[{index}] {block['type']} block must have exactly keys {ALLOWED_BLOCK_TYPES[block['type']]}")
    for key in _STR_FIELDS:
        if key in block and not isinstance(block[key], str):
            raise ValueError(f"messages[{index}] {block['type']} block field {key!r} must be a str")


def _tool_result(message: dict[str, Any], call_id: str, index: int) -> dict[str, Any]:
    if message["role"] != "tool" or set(message) != {"role", "tool_call_id", "content"}:
        raise ValueError(f"messages[{index}] must be a tool message")
    if message["tool_call_id"] != call_id:
        raise ValueError(f"messages[{index}].tool_call_id must be {call_id!r}")
    content = message["content"]
    if not isinstance(content, str):
        raise ValueError(f"messages[{index}].content must be a str")
    try:
        observation = json.loads(content)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"messages[{index}].content must be a JSON object") from exc
    if not isinstance(observation, dict) or type(observation.get("ok")) is not bool:
        raise ValueError(f"messages[{index}].content must be a JSON object with a bool 'ok'")
    return {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": observation["ok"] is False}


def _same_json(a: Any, b: Any) -> bool:
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


_STR_FIELDS = ("text", "thinking", "signature", "data", "id", "name")
_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
_ALLOWED_STOP_REASONS = ("tool_use", "end_turn", "refusal")


def parse_response(response: Any, model: str) -> PolicyStep:
    """依模組 docstring 把 response 轉成 PolicyStep（含 meta）。"""
    if getattr(response, "input_transformations", None):
        raise ReplayIntegrityError("the API reports input transformations: the replayed history was not used as sent")
    content = getattr(response, "content", None)
    if not isinstance(content, (list, tuple)):
        raise AdapterError("response content must be a list of blocks")
    blocks = [_convert_block(block) for block in content]
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason not in _ALLOWED_STOP_REASONS:
        raise AdapterError(f"unsupported stop_reason {stop_reason!r}")
    tool_uses = [block for block in blocks if block["type"] == "tool_use"]
    text = "".join(block["text"] for block in blocks if block["type"] == "text")
    if stop_reason == "refusal":
        text = ""
    elif stop_reason == "tool_use" and not tool_uses:
        raise AdapterError("stop_reason is tool_use but the response has no tool_use block")
    elif stop_reason == "end_turn" and tool_uses:
        raise AdapterError("stop_reason is end_turn but the response has tool_use blocks")
    try:
        calls = (
            tuple(ToolCall(id=block["id"], name=block["name"], args=block["input"]) for block in tool_uses)
            if stop_reason == "tool_use"
            else ()
        )
        message = ModelMessage(text=text, tool_calls=calls, stop_reason=stop_reason, provider_content=blocks)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"the response cannot be recorded as a turn: {exc}") from exc

    usage = getattr(response, "usage", None)
    raw = {name: getattr(usage, name, None) for name in _USAGE_FIELDS}
    for name, value in raw.items():
        if value is not None and (type(value) is not int or value < 0):
            raise AdapterError(f"usage.{name} must be a non-negative int")
    breakdown = getattr(usage, "cache_creation", None)
    cache_1h = getattr(breakdown, "ephemeral_1h_input_tokens", None) if breakdown is not None else None
    if cache_1h is not None and (type(cache_1h) is not int or cache_1h < 0):
        raise AdapterError("usage.cache_creation.ephemeral_1h_input_tokens must be a non-negative int")
    cache_1h = cache_1h or 0
    input_tokens, output_tokens, cache_write, cache_read = (raw[name] or 0 for name in _USAGE_FIELDS)
    prices = PRICING.get(model)
    cost = None
    if prices is not None:
        cost = (
            input_tokens * prices["input"]
            + (cache_write - cache_1h) * prices["cache_write"]
            + cache_1h * prices["cache_write_1h"]
            + cache_read * prices["cache_read"]
            + output_tokens * prices["output"]
        ) / 1e6

    details = getattr(response, "stop_details", None)
    meta = {
        "response_model": _optional_str(getattr(response, "model", None)),
        "request_id": _optional_str(getattr(response, "_request_id", None)),
        "stop_reason": stop_reason,
        "stop_details": None
        if details is None
        else {
            "category": _optional_str(getattr(details, "category", None)),
            "explanation": _optional_str(getattr(details, "explanation", None)),
        },
        "tool_use_blocks": len(tool_uses),
        "usage": {**raw, "cache_creation_1h_input_tokens": cache_1h},
    }
    return PolicyStep(
        message=message,
        input_tokens=input_tokens + cache_write + cache_read,
        output_tokens=output_tokens,
        cost_usd=cost,
        meta=meta,
    )


def _convert_block(block: Any) -> dict[str, Any]:
    """以屬性讀取，依白名單轉成只含規定 key 的 dict（不呼叫 SDK 序列化）。"""
    kind = getattr(block, "type", None)
    if not isinstance(kind, str) or kind not in ALLOWED_BLOCK_TYPES:
        raise AdapterError(f"unsupported content block type {kind!r}")
    out: dict[str, Any] = {}
    for key in ALLOWED_BLOCK_TYPES[kind]:
        if not hasattr(block, key):
            raise AdapterError(f"{kind} block has no {key!r}")
        value = getattr(block, key)
        if key in _STR_FIELDS and not isinstance(value, str):
            raise AdapterError(f"{kind} block field {key!r} must be a str")
        # 深度有上限的檢查必須在任何遞迴複製之前（模型可回傳任意深的 tool input）；block 在 provider_content 中位於第 1 層，
        # 其欄位值位於第 2 層，因此與 ModelMessage 對 provider_content 的上限一致
        try:
            _check_json(value, f"{kind}.{key}", 2, MAX_MODEL_CONFIG_DEPTH)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"{kind} block field {key!r} is not a recordable JSON value: {exc}") from exc
        out[key] = copy.deepcopy(value)
    return out


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


class AnthropicPolicy:
    """Anthropic Messages API policy（對 conversation 無狀態）。

    建構：``AnthropicPolicy(client, agent_config=None)``；agent_config 為 None 時用 ``baseline_agent_config()``。
    agent_config.policy 必須是 ANTHROPIC_POLICY、model_config 的 key 集合必須剛好等於 BASELINE_MODEL_CONFIG 的、
    ``model_config["adapter"] == ADAPTER_VERSION``、system_prompt 不可為純空白；否則 ValueError。
    client 若有 ``max_retries`` / ``timeout`` 屬性，必須等於 model_config["client"] 的值（否則 ValueError）；
    實驗腳本應一律以 ``make_anthropic_client`` 建立 client。
    ``next_message(messages, tools)`` = ``parse_response(client.beta.messages.create(**build_request(...)), model)``；
    build_request 的 ValueError 在呼叫 client 之前拋出；client 的例外原樣往外拋。``agent_config`` 每次回傳新物件且不變。
    """

    def __init__(self, client: Any, agent_config: AgentConfig | None = None) -> None:
        config = baseline_agent_config() if agent_config is None else agent_config
        if not isinstance(config, AgentConfig):
            raise TypeError("agent_config must be an AgentConfig or None")
        config = copy.deepcopy(config)
        model_config = config.model_config
        if config.policy != ANTHROPIC_POLICY:
            raise ValueError(f"agent_config.policy must be {ANTHROPIC_POLICY!r}")
        if set(model_config) != set(BASELINE_MODEL_CONFIG):
            raise ValueError("model_config must have exactly the keys of BASELINE_MODEL_CONFIG")
        if model_config["adapter"] != ADAPTER_VERSION:
            raise ValueError(f"model_config['adapter'] must be {ADAPTER_VERSION!r}")
        if _blank(config.system_prompt):
            raise ValueError("the system prompt must not be blank")
        settings = model_config["client"]
        if not isinstance(settings, dict) or set(settings) != {"max_retries", "timeout_s"}:
            raise ValueError("model_config['client'] must be {'max_retries', 'timeout_s'}")
        if hasattr(client, "max_retries") and not _same_setting(client.max_retries, settings["max_retries"]):
            raise ValueError("client.max_retries does not match model_config['client']")
        if hasattr(client, "timeout") and not _same_setting(client.timeout, settings["timeout_s"]):
            raise ValueError("client.timeout does not match model_config['client']")
        self._client = client
        self._config = config

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        model_config = self._config.model_config
        request = build_request(messages, tools, model_config)
        response = self._client.beta.messages.create(**request)
        return parse_response(response, model_config["model"])


def _same_setting(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual == expected


def make_anthropic_client(model_config: dict[str, Any] | None = None) -> Any:
    """建立真正的 ``anthropic.Anthropic(max_retries=..., timeout=...)``，兩個值取自 model_config["client"]
    （預設 BASELINE_MODEL_CONFIG）；只傳這兩個 keyword。credential 完全由 SDK 自行解析；本函式不讀取、不傳入任何 key。
    未安裝 SDK（optional extra ``llm``）→ ImportError，訊息說明安裝方式。
    """
    settings = (BASELINE_MODEL_CONFIG if model_config is None else model_config)["client"]
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("the Anthropic SDK is not installed; install the optional extra: uv sync --extra llm") from exc
    return anthropic.Anthropic(max_retries=settings["max_retries"], timeout=settings["timeout_s"])
