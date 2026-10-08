"""Agent 端的資料型別與 policy 介面（decision log 006）。

Policy 必須對 conversation **無狀態**：``next_message(messages, tools)`` 只能依輸入決定輸出
（LLM 抽樣的隨機性除外），不得保留跨呼叫的隱藏狀態。因此 restore agent = 重建 conversation。

messages 的形狀由 ``replay.trajectory.render_messages`` 決定（JSON-native list of dict）::

    {"role": "system",    "content": str}
    {"role": "developer", "content": str}
    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}, ...]}
    {"role": "tool",      "tool_call_id": str, "content": str}   # content = canonical_json(observation)

trajectory/v1 的 assistant 訊息只有 text 與 tool_calls。Week 1 #5 的 baseline 不得啟用需要原樣回傳
provider 專屬 content block 的功能（例如 extended thinking）；若需要，必須另立 decision log 並升 trajectory 格式。

統一規則
========
- **JSON 值正規化**：ToolCall.args 與 AgentConfig.model_config 在建構時以 ``json.loads(canonical_json(x))``
  正規化（key 遞迴排序，同時完成深拷貝）後保存。因此 live run 與 save / load 後重建的 conversation 逐 byte 相同。
- **例外型別**：型別錯 → TypeError（例如 bool 當 int、list 當 tuple、tool_calls 元素不是 ToolCall、出現 tuple、
  model_config 不是 dict）；型別對但值錯 → ValueError（例如必填字串為空、call id 重複、NaN / inf、巢狀過深、
  字串無法 UTF-8 編碼）。
- **所有字串**（含 model_config / args 中巢狀的值與 key）必須可 UTF-8 編碼，否則 ValueError。
- 所有 ``from_dict`` 一律 raise ValueError（內層 TypeError 以 ``raise ValueError(...) from e`` 包裝）。
- 本模組所有 ``to_dict`` 的輸出都是 JSON-native，且 ``X.from_dict(x.to_dict()) == x``。
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from env.environment import check_call_preconditions
from env.hashing import canonical_json, sha256_of
from grader.schema import Usage

SCRIPTED_POLICY = "scripted"
SCRIPTED_PROMPT_VERSION = "scripted/v1"

# AgentConfig.model_config 的巢狀深度上限（ScriptedPolicy 把 args 放在 model_config 下第 4 層，
# 因此必須大於 MAX_ARGS_DEPTH + 4）。超過 → ValueError。
MAX_MODEL_CONFIG_DEPTH = 128


@dataclass(frozen=True)
class ToolCall:
    """模型要求的一次 tool call。

    建構時驗證：id 為非空、可 UTF-8 編碼的 str；name 與 args 必須滿足 ``env.environment.check_call_preconditions``
    （其 TypeError / ValueError 原樣拋出）。args 可以是任意 JSON 值（不是 dict 時 env 會記錄 invalid_arguments），
    保存的是正規化後的深拷貝。
    """

    id: str
    name: str
    args: Any

    def __post_init__(self) -> None:
        _require_str(self.id, "ToolCall.id", non_empty=True)
        check_call_preconditions(self.name, self.args)
        object.__setattr__(self, "args", _normalized(self.args))

    def to_dict(self) -> dict[str, Any]:
        """剛好 ``{"id", "name", "args"}``；args 為深拷貝。"""
        return {"id": self.id, "name": self.name, "args": copy.deepcopy(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """data 必須是 dict 且 key 剛好為上述三個；任何錯誤 raise ValueError。"""
        _require_keys(data, ("id", "name", "args"), "tool call")
        return _construct(cls, id=data["id"], name=data["name"], args=data["args"])


@dataclass(frozen=True)
class ModelMessage:
    """模型的一則 assistant 訊息（決定性內容；token / latency 等量測不在這裡）。

    建構時驗證：text 為 str；tool_calls 必須是 tuple（list → TypeError）且元素皆為 ToolCall、id 不重複（ValueError）；
    stop_reason 為 str 或 None；字串須可 UTF-8 編碼。沒有 tool call 的訊息代表 agent 結束 episode。
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str | None

    def __post_init__(self) -> None:
        _require_str(self.text, "ModelMessage.text")
        if type(self.tool_calls) is not tuple:
            raise TypeError("ModelMessage.tool_calls must be a tuple")
        for call in self.tool_calls:
            if not isinstance(call, ToolCall):
                raise TypeError("ModelMessage.tool_calls must contain ToolCall objects")
        ids = [call.id for call in self.tool_calls]
        if len(set(ids)) != len(ids):
            raise ValueError("tool call ids must be unique within a message")
        if self.stop_reason is not None:
            _require_str(self.stop_reason, "ModelMessage.stop_reason")

    def to_dict(self) -> dict[str, Any]:
        """剛好 ``{"text", "tool_calls", "stop_reason"}``；tool_calls 為 list of ToolCall.to_dict()。"""
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMessage:
        """data 必須是 dict、key 剛好為上述三個、tool_calls 為 list；任何錯誤 raise ValueError。"""
        _require_keys(data, ("text", "tool_calls", "stop_reason"), "model message")
        if not isinstance(data["tool_calls"], list):
            raise ValueError("model message tool_calls must be a list")
        calls = tuple(ToolCall.from_dict(call) for call in data["tool_calls"])
        return _construct(cls, text=data["text"], tool_calls=calls, stop_reason=data["stop_reason"])


@dataclass(frozen=True)
class PolicyStep:
    """``Policy.next_message`` 的回傳：訊息 + policy 自報的用量（未知為 None）。latency 由 runner 量測。

    建構時驗證：message 必須是 ModelMessage（TypeError）；input_tokens / output_tokens 為 ``type(x) is int`` 或 None，
    cost_usd 為有限的 int / float 或 None（規則同 ``grader.schema.Usage``）。
    """

    message: ModelMessage
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, ModelMessage):
            raise TypeError("PolicyStep.message must be a ModelMessage")
        Usage(input_tokens=self.input_tokens, output_tokens=self.output_tokens, cost_usd=self.cost_usd)


@dataclass(frozen=True)
class AgentConfig:
    """Checkpoint 重建 agent 所需的全部設定（SPEC §5：system / developer prompt、prompt version、model config）。

    建構時驗證：policy、prompt_version 為非空 str；system_prompt、developer_prompt 為 str；model_config 為 dict
    （正規化後保存），巢狀深度不超過 MAX_MODEL_CONFIG_DEPTH；所有字串可 UTF-8 編碼。
    **是否為同一個 agent 一律以 ``config_id`` 判定**（Python ``==`` 會把 1 與 1.0、True 與 1 視為相同，不可用於此判斷）。
    """

    policy: str
    prompt_version: str
    system_prompt: str
    developer_prompt: str
    model_config: dict[str, Any]

    def __post_init__(self) -> None:
        _require_str(self.policy, "AgentConfig.policy", non_empty=True)
        _require_str(self.prompt_version, "AgentConfig.prompt_version", non_empty=True)
        _require_str(self.system_prompt, "AgentConfig.system_prompt")
        _require_str(self.developer_prompt, "AgentConfig.developer_prompt")
        if not isinstance(self.model_config, dict):
            raise TypeError("AgentConfig.model_config must be a dict")
        _check_json(self.model_config, "model_config", 0, MAX_MODEL_CONFIG_DEPTH)
        object.__setattr__(self, "model_config", _normalized(self.model_config))

    @property
    def config_id(self) -> str:
        """``sha256_of(self.to_dict())``。"""
        return sha256_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """剛好 5 個 key（欄位名）；model_config 為深拷貝。"""
        return {
            "policy": self.policy,
            "prompt_version": self.prompt_version,
            "system_prompt": self.system_prompt,
            "developer_prompt": self.developer_prompt,
            "model_config": copy.deepcopy(self.model_config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """data 必須是 dict 且 key 剛好為 5 個欄位名；任何錯誤 raise ValueError。"""
        _require_keys(data, AGENT_CONFIG_FIELDS, "agent config")
        return _construct(cls, **{name: data[name] for name in AGENT_CONFIG_FIELDS})


class Policy(Protocol):
    @property
    def agent_config(self) -> AgentConfig: ...

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        """依目前 conversation 與 tool specs 產生下一則 assistant 訊息。不得修改輸入。"""
        ...


class ScriptedPolicy:
    """依固定劇本播放 tool calls 的決定性 policy（Week 1 用於 oracle 執行與 restore / rerun 驗證）。

    - ``script``：list 或 tuple（其他型別、包括 str → TypeError）；每個元素是一個 turn 的 tool calls，
      同樣必須是 list 或 tuple（str → TypeError）且非空（ValueError）；每個 call 必須是長度剛好 2 的 list 或 tuple
      （型別錯 → TypeError、長度錯 → ValueError），``(name, args)`` 須滿足 ``check_call_preconditions``。
      劇本以正規化後的深拷貝保存。
    - 播放位置 k = messages 中 role 為 "assistant" 的訊息數（因此無狀態，restore 後自然接續）。
      messages 不是 list、或任一元素不是含 str ``"role"`` 的 dict → TypeError。
      k < len(script)：回傳 ``ModelMessage(text="", tool_calls=..., stop_reason="tool_use")``，
      第 i 個 call 的 id 為 ``f"call_{k}_{i}"``；k >= len(script)：回傳
      ``ModelMessage(text=final_text, tool_calls=(), stop_reason="end_turn")``。
    - 用量：input_tokens=0、output_tokens=0、cost_usd=0.0。
    - ``agent_config``：policy=SCRIPTED_POLICY、prompt_version（預設 SCRIPTED_PROMPT_VERSION）、
      system_prompt、developer_prompt（預設 ""）、model_config =
      ``{"script": [[{"name": ..., "args": ...}, ...], ...], "final_text": final_text}``。
    - ``next_message`` 與 ``agent_config`` 每次回傳新物件，不與內部劇本共用可變物件。
    """

    def __init__(
        self,
        script: Sequence[Sequence[tuple[str, Any]]],
        *,
        system_prompt: str = "",
        developer_prompt: str = "",
        prompt_version: str = SCRIPTED_PROMPT_VERSION,
        final_text: str = "Done.",
    ) -> None:
        if not isinstance(script, (list, tuple)):
            raise TypeError("script must be a list or tuple of turns")
        turns: list[tuple[tuple[str, Any], ...]] = []
        for turn in script:
            if not isinstance(turn, (list, tuple)):
                raise TypeError("each script turn must be a list or tuple of calls")
            if len(turn) == 0:
                raise ValueError("each script turn must contain at least one call")
            calls: list[tuple[str, Any]] = []
            for call in turn:
                if not isinstance(call, (list, tuple)):
                    raise TypeError("each scripted call must be a (name, args) list or tuple")
                if len(call) != 2:
                    raise ValueError("each scripted call must have exactly two elements: (name, args)")
                name, args = call
                check_call_preconditions(name, args)
                calls.append((name, _normalized(args)))
            turns.append(tuple(calls))
        _require_str(final_text, "final_text")
        self._script = tuple(turns)
        self._final_text = final_text
        self._config = AgentConfig(
            policy=SCRIPTED_POLICY,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            model_config={
                "script": [[{"name": name, "args": args} for name, args in turn] for turn in self._script],
                "final_text": final_text,
            },
        )

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise TypeError("each message must be a dict with a str 'role'")
        k = sum(1 for message in messages if message["role"] == "assistant")
        if k < len(self._script):
            calls = tuple(
                ToolCall(id=f"call_{k}_{i}", name=name, args=args) for i, (name, args) in enumerate(self._script[k])
            )
            message = ModelMessage(text="", tool_calls=calls, stop_reason="tool_use")
        else:
            message = ModelMessage(text=self._final_text, tool_calls=(), stop_reason="end_turn")
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


def oracle_policy(oracle_calls: Sequence[tuple[str, Any]], **kwargs: Any) -> ScriptedPolicy:
    """每個 oracle call 一個 turn 的 ScriptedPolicy；kwargs 傳給 ScriptedPolicy。oracle_calls 不可為空（ValueError）。"""
    if not isinstance(oracle_calls, (list, tuple)):
        raise TypeError("oracle_calls must be a list or tuple")
    if len(oracle_calls) == 0:
        raise ValueError("oracle_calls must not be empty")
    return ScriptedPolicy([[call] for call in oracle_calls], **kwargs)


AGENT_CONFIG_FIELDS: tuple[str, ...] = ("policy", "prompt_version", "system_prompt", "developer_prompt", "model_config")


def _require_str(value: Any, what: str, *, non_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, got {type(value).__name__}")
    if non_empty and value == "":
        raise ValueError(f"{what} must not be empty")
    _require_utf8(value, what)


def _require_utf8(value: str, what: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{what} must be encodable as UTF-8") from exc


def _check_json(value: Any, what: str, depth: int, max_depth: int) -> None:
    """JSON-native 檢查：型別錯 → TypeError；NaN / inf、過深、無法 UTF-8 編碼 → ValueError。"""
    if depth > max_depth:
        raise ValueError(f"{what} must not be nested deeper than {max_depth} levels")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{what} must not contain NaN or infinity")
        return
    if isinstance(value, str):
        _require_utf8(value, what)
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item, what, depth + 1, max_depth)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{what} object keys must be str")
            _require_utf8(key, what)
            _check_json(item, what, depth + 1, max_depth)
        return
    raise TypeError(f"{what} must contain only JSON values, got {type(value).__name__}")


def _normalized(value: Any) -> Any:
    """``json.loads(canonical_json(value))``：key 遞迴排序並深拷貝。"""
    return json.loads(canonical_json(value))


def _require_keys(data: Any, keys: tuple[str, ...], what: str) -> None:
    if not isinstance(data, dict) or set(data) != set(keys):
        raise ValueError(f"{what} must be an object with keys: " + ", ".join(keys))


def _construct(cls: Any, **kwargs: Any) -> Any:
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"invalid {cls.__name__}: {exc}") from exc
