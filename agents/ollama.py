"""Core 的本機模型 adapter：Ollama 原生 ``/api/chat``（decision 026）。只用標準函式庫。

``OllamaPolicy(client, agent_config)`` 實作 ``agents.policy.Policy``。client 只需要 ``chat(request) -> dict``（真正的
``OllamaHTTPClient`` 或測試 / dry-run 用的假 client）；``verify_server`` 另外用到 ``version()``、``tags()``、``show(model)``。

Request（``build_chat_request``，決定性、JSON-native 的新物件，不修改輸入）::

    {"model": mc["model"], "messages": [...], "tools": [{"type": "function", "function": {"name", "description", "parameters"}}, ...],
     "stream": false, "think": mc["think"], "options": {**mc["options"], "seed": <這次呼叫的 seed>},
     "truncate": false, "shift": false}

messages 的檢查與轉換（違反 → ValueError，且不呼叫 client）：
- 開頭必須是 system、developer、user 三則；system 與 user 不可為純空白。system 與 developer 合成一則 system 訊息
  （``system + "\\n\\n" + developer``；developer 純空白時只送 system）。
- 之後必須是 (assistant, tool × n) 的重複，n = 該 assistant 的 tool_calls 數（>= 1），tool_call_id 依序相同。
- 每則 assistant 必須有 ``provider_content`` = ``[raw]``：raw 是模型回傳的 assistant 訊息（key 只能是 role / content / thinking /
  tool_calls），content 與 text 相同、tool_calls 依序與 ToolCall 的 name 及解析後的參數相同。raw **原樣**重送。
- tool 訊息送成 ``{"role": "tool", "content": <原 content 字串>, "tool_name": <對應 call 的 name>}``。
- tools 不可為空；只保留 name / description / input_schema。

Response → PolicyStep（``parse_chat_response``）：
- 結構逐欄檢查（不符 → ``OllamaProtocolError``，harness）：response 是 dict、``done`` 為 true、``done_reason`` 為 stop 或 length、
  ``model`` 等於設定的 model、沒有 ``remote_model`` / ``remote_host``、message 只有允許的 key 且型別正確、計數為 int >= 0。
- ``done_reason`` = length（輸出達到 num_predict）→ ``AdapterError``（agent_output，等同 Anthropic 的 max_tokens）。
- context 防護（``OllamaContextError``，harness）：``prompt_eval_count`` 必須存在且 >= 1；``prompt_eval_count + eval_count < num_ctx``。
- tool call：``function.name`` 必須是非空字串（否則 ``AdapterError``）；參數為物件照用、JSON 字串先解析、null 視為 {}、
  其他原樣（env 記為 invalid_arguments）；id = ``call_<assistant_turn>_<序號>``。
- ModelMessage：text = content、stop_reason = "tool_calls" 或 "stop"、provider_content = [raw]（只含允許的 key）。
- 用量：input = prompt_eval_count、output = eval_count（欄位不在為 0）、cost_usd = 0.0；meta 見 ``_meta``。
"""

from __future__ import annotations

import copy
import http.client
import ipaddress
import json
import secrets
import socket
from typing import Any, Callable
from urllib.parse import urlsplit

from agents.baseline import (
    BASELINE_DEVELOPER_PROMPT,
    BASELINE_PROMPT_VERSION,
    BASELINE_SYSTEM_PROMPT,
    AdapterError,
    InfrastructureError,
)
from agents.policy import MAX_MODEL_CONFIG_DEPTH, AgentConfig, ModelMessage, PolicyStep, ToolCall, _check_json
from env.hashing import canonical_json
from replay.trajectory import check_json_document

OLLAMA_POLICY = "ollama-chat"
OLLAMA_ADAPTER_VERSION = "ollama-chat/v1"

# decision 026 §5 的草案。model_digest / ollama_version 為 None 表示尚未凍結（正式 run 會拒絕）。
LOCAL_MODEL_CONFIG: dict[str, Any] = {
    "provider": "ollama",
    "adapter": OLLAMA_ADAPTER_VERSION,
    "model": "qwen3.5:9b-q4_K_M",
    "model_digest": None,
    "ollama_version": None,
    "think": False,
    "options": {
        "num_ctx": 32768,
        "num_predict": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
    },
    "client": {"timeout_s": 600},
}
REQUIRED_OPTIONS = ("num_ctx", "num_predict", "temperature")
RAW_MESSAGE_KEYS = ("role", "content", "thinking", "tool_calls")
RAW_CALL_KEYS = ("function", "id", "type")
RAW_FUNCTION_KEYS = ("name", "arguments", "index")
DONE_REASONS = ("stop", "length")
COUNT_FIELDS = ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")


class OllamaProtocolError(RuntimeError):
    """server 的回覆不符合預期的結構、模型不符或來自遠端（cloud）模型；harness 類，實驗停止。"""


class OllamaContextError(RuntimeError):
    """無法確認 prompt 完整進入模型（可能被截斷）；harness 類，實驗停止。"""


class OllamaTransportError(InfrastructureError):
    """連線失敗、逾時或 HTTP 5xx；infrastructure 類，依 decision 009 重試。"""


def local_agent_config() -> AgentConfig:
    """decision 026 的本機 agent：policy=OLLAMA_POLICY、prompt 與 Anthropic baseline 相同（baseline/v1）、
    model_config=LOCAL_MODEL_CONFIG（每次回傳新物件）。"""
    return AgentConfig(
        policy=OLLAMA_POLICY,
        prompt_version=BASELINE_PROMPT_VERSION,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        developer_prompt=BASELINE_DEVELOPER_PROMPT,
        model_config=copy.deepcopy(LOCAL_MODEL_CONFIG),
    )


def is_frozen(config: AgentConfig) -> bool:
    """model_digest 與 ollama_version 都已填入（decision 026 §5）。"""
    model_config = config.model_config
    return isinstance(model_config.get("model_digest"), str) and isinstance(model_config.get("ollama_version"), str)


def check_local_config(config: Any) -> AgentConfig:
    """OllamaPolicy 接受的設定；違反 → ValueError（型別錯 → TypeError）。回傳深拷貝。"""
    if not isinstance(config, AgentConfig):
        raise TypeError("agent_config must be an AgentConfig")
    config = copy.deepcopy(config)
    model_config = config.model_config
    if config.policy != OLLAMA_POLICY:
        raise ValueError(f"agent_config.policy must be {OLLAMA_POLICY!r}")
    if set(model_config) != set(LOCAL_MODEL_CONFIG):
        raise ValueError("model_config must have exactly the keys of LOCAL_MODEL_CONFIG")
    if model_config["adapter"] != OLLAMA_ADAPTER_VERSION or model_config["provider"] != "ollama":
        raise ValueError(f"model_config must use the adapter {OLLAMA_ADAPTER_VERSION!r}")
    model = model_config["model"]
    if not isinstance(model, str) or not model.strip() or model.lower().endswith("cloud"):
        raise ValueError("model must be a local Ollama model tag (cloud models are refused)")
    if not isinstance(model_config["think"], bool):
        raise ValueError("model_config['think'] must be a bool")
    options = model_config["options"]
    if not isinstance(options, dict) or any(name not in options for name in REQUIRED_OPTIONS) or "seed" in options:
        raise ValueError(f"model_config['options'] must set {REQUIRED_OPTIONS} and must not fix a seed")
    for name in ("num_ctx", "num_predict"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"options[{name!r}] must be an int >= 1")
    if options["num_predict"] >= options["num_ctx"]:
        raise ValueError("num_predict must be smaller than num_ctx")
    for name in ("model_digest", "ollama_version"):
        if model_config[name] is not None and not isinstance(model_config[name], str):
            raise ValueError(f"model_config[{name!r}] must be None or a str")
    settings = model_config["client"]
    if not isinstance(settings, dict) or set(settings) != {"timeout_s"}:
        raise ValueError("model_config['client'] must be {'timeout_s'}")
    if not config.system_prompt.strip():
        raise ValueError("the system prompt must not be blank")
    return config


def _parse_arguments(arguments: Any) -> Any:
    """物件照用；JSON 字串先解析，結果是物件才用；null → {}；其他原樣（env 記為 invalid_arguments）。"""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return arguments
        return parsed if isinstance(parsed, dict) else arguments
    return arguments


def build_chat_request(
    messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...], model_config: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    """依模組 docstring 組出 ``/api/chat`` request；違反 → ValueError。"""
    check_json_document(messages, error=ValueError)
    if not isinstance(tools, (list, tuple)) or not tools:
        raise ValueError("tools must be a non-empty list or tuple")
    for tool in tools:
        check_json_document(tool, error=ValueError)
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("messages must start with system, developer and user")
    head = [message.get("role") if isinstance(message, dict) else None for message in messages[:3]]
    if head != ["system", "developer", "user"]:
        raise ValueError("messages must start with system, developer and user")
    system, developer, user = (message.get("content") for message in messages[:3])
    if not all(isinstance(text, str) for text in (system, developer, user)) or not system.strip() or not user.strip():
        raise ValueError("system and user content must be non-blank strings")
    out: list[dict[str, Any]] = [
        {"role": "system", "content": system if not developer.strip() else f"{system}\n\n{developer}"},
        {"role": "user", "content": user},
    ]
    index = 3
    while index < len(messages):
        assistant = messages[index]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ValueError("expected an assistant message")
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("an assistant message inside the conversation must have tool calls")
        out.append(_raw_for(assistant, calls))
        for call in calls:
            index += 1
            if index >= len(messages):
                raise ValueError("missing tool result")
            tool = messages[index]
            if not isinstance(tool, dict) or tool.get("role") != "tool" or tool.get("tool_call_id") != call.get("id"):
                raise ValueError("tool results must follow their calls in order")
            if not isinstance(tool.get("content"), str):
                raise ValueError("tool content must be a str")
            out.append({"role": "tool", "content": tool["content"], "tool_name": call["name"]})
        index += 1
    return {
        "model": model_config["model"],
        "messages": out,
        "tools": [
            {"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}}
            for tool in tools
        ],
        "stream": False,
        "think": model_config["think"],
        "options": {**copy.deepcopy(model_config["options"]), "seed": seed},
        "truncate": False,
        "shift": False,
    }


def _raw_for(assistant: dict[str, Any], calls: list[Any]) -> dict[str, Any]:
    content = assistant.get("provider_content")
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise ValueError("an assistant message must carry provider_content = [the raw Ollama message]")
    raw = content[0]
    if not set(raw) <= set(RAW_MESSAGE_KEYS) or raw.get("role") != "assistant":
        raise ValueError("provider_content is not a raw assistant message")
    if raw.get("content", "") != assistant.get("content"):
        raise ValueError("provider_content disagrees with the message text")
    raw_calls = raw.get("tool_calls") or []
    if not isinstance(raw_calls, list) or len(raw_calls) != len(calls):
        raise ValueError("provider_content disagrees with the tool calls")
    for raw_call, call in zip(raw_calls, calls):
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        if not isinstance(function, dict) or not isinstance(call, dict):
            raise ValueError("provider_content disagrees with the tool calls")
        if function.get("name") != call.get("name") or canonical_json(_parse_arguments(function.get("arguments"))) != canonical_json(
            call.get("args")
        ):
            raise ValueError("provider_content disagrees with the tool calls")
    return copy.deepcopy(raw)


def _count(response: dict[str, Any], name: str) -> int | None:
    value = response.get(name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise OllamaProtocolError(f"{name} must be an int >= 0")
    return value


def parse_chat_response(response: Any, model_config: dict[str, Any], *, assistant_turn: int, seed: int) -> PolicyStep:
    """依模組 docstring 把 ``/api/chat`` 的回覆轉成 PolicyStep。"""
    if not isinstance(response, dict):
        raise OllamaProtocolError("the response is not a JSON object")
    if "remote_model" in response or "remote_host" in response:
        raise OllamaProtocolError("the response came from a remote (cloud) model")
    if response.get("model") != model_config["model"]:
        raise OllamaProtocolError(f"the response came from {response.get('model')!r}, not {model_config['model']!r}")
    if response.get("done") is not True or response.get("done_reason") not in DONE_REASONS:
        raise OllamaProtocolError("the response is not a finished chat reply")
    counts = {name: _count(response, name) for name in COUNT_FIELDS}
    message = response.get("message")
    if not isinstance(message, dict) or not set(message) <= set(RAW_MESSAGE_KEYS) or message.get("role") != "assistant":
        raise OllamaProtocolError("message is not an assistant message with the expected keys")
    text = message.get("content", "")
    if not isinstance(text, str) or not isinstance(message.get("thinking", ""), str):
        raise OllamaProtocolError("message content / thinking must be strings")
    raw_calls = message.get("tool_calls", [])
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise OllamaProtocolError("message tool_calls must be a list")
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or not set(raw_call) <= set(RAW_CALL_KEYS) or not isinstance(raw_call.get("function"), dict):
            raise OllamaProtocolError("each tool call must be an object with a function object")
        if not set(raw_call["function"]) <= set(RAW_FUNCTION_KEYS):
            raise OllamaProtocolError("a tool call function has unexpected keys")
    try:
        _check_json(message, "message", 0, MAX_MODEL_CONFIG_DEPTH)
    except (TypeError, ValueError) as exc:
        raise OllamaProtocolError(f"message is not valid JSON content: {exc}") from exc
    raw = json.loads(canonical_json(message))
    usage = {"input_tokens": counts["prompt_eval_count"], "output_tokens": counts["eval_count"] or 0}
    if response["done_reason"] == "length":
        error = AdapterError("the reply reached num_predict (output truncated)")
        error.billed_usage = {"cost_usd": 0.0, **usage}  # type: ignore[attr-defined]
        raise error
    num_ctx = model_config["options"]["num_ctx"]
    if not counts["prompt_eval_count"]:
        raise OllamaContextError("prompt_eval_count is missing or 0; the prompt cannot be verified")
    if counts["prompt_eval_count"] + (counts["eval_count"] or 0) >= num_ctx:
        raise OllamaContextError("prompt_eval_count + eval_count reached num_ctx; the context may have been truncated")
    calls = []
    for index, raw_call in enumerate(raw_calls):
        name = raw_call["function"].get("name")
        if not isinstance(name, str) or not name:
            error = AdapterError("a tool call has no name")
            error.billed_usage = {"cost_usd": 0.0, **usage}  # type: ignore[attr-defined]
            raise error
        calls.append(ToolCall(id=f"call_{assistant_turn}_{index}", name=name, args=_parse_arguments(raw_call["function"].get("arguments"))))
    model_message = ModelMessage(text=text, tool_calls=tuple(calls), stop_reason="tool_calls" if calls else "stop", provider_content=[raw])
    meta = {
        "response_model": response["model"],
        "done_reason": response["done_reason"],
        "seed": seed,
        "tool_use_blocks": len(calls),
        "usage": {"prompt_eval_count": counts["prompt_eval_count"], "eval_count": counts["eval_count"] or 0},
        "durations": {
            "total": counts["total_duration"],
            "load": counts["load_duration"],
            "prompt_eval": counts["prompt_eval_duration"],
            "eval": counts["eval_duration"],
        },
    }
    return PolicyStep(message=model_message, input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"], cost_usd=0.0, meta=meta)


def _default_seed() -> int:
    return secrets.randbelow(2**31 - 1) + 1


class OllamaPolicy:
    """Ollama ``/api/chat`` policy（對 conversation 無狀態；每次呼叫的 seed 隨機並記在 meta，decision 026 §5）。

    ``next_message``：組 request（ValueError 在呼叫之前）→ 送出前的 fit check：request bytes ÷ 2 + num_predict > num_ctx →
    ``OllamaContextError``（不呼叫）→ ``client.chat(request)``（例外原樣往外拋）→ ``parse_chat_response``。
    """

    def __init__(self, client: Any, agent_config: AgentConfig, *, seed_source: Callable[[], int] | None = None) -> None:
        self._config = check_local_config(agent_config)
        self._client = client
        self._seed_source = seed_source or _default_seed

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        model_config = self._config.model_config
        seed = self._seed_source()
        if type(seed) is not int or not 1 <= seed < 2**31:
            raise ValueError("seed_source must return an int in [1, 2**31)")
        request = build_chat_request(messages, tools, model_config, seed=seed)
        options = model_config["options"]
        if len(canonical_json(request).encode("utf-8")) // 2 + options["num_predict"] > options["num_ctx"]:
            raise OllamaContextError("the request may not fit in num_ctx; it was not sent")
        turn = sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "assistant")
        response = self._client.chat(request)
        return parse_chat_response(response, model_config, assistant_turn=turn, seed=seed)


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if host != "localhost":
        return False
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        return False
    return bool(addresses) and all(ipaddress.ip_address(address.split("%")[0]).is_loopback for address in addresses)


class OllamaHTTPClient:
    """只連 loopback 的 Ollama client（``http.client``，不經過 proxy 設定）。只提供 chat / version / tags / show——沒有任何
    下載、建立、刪除或複製模型的方法（decision 026 §1、MACHINE_POLICY）。

    錯誤：連線失敗、逾時、HTTP 5xx → OllamaTransportError；HTTP 4xx 的內容提到 context → OllamaContextError；
    其他 4xx 或回覆不是 JSON → OllamaProtocolError。
    """

    def __init__(self, base_url: str, *, timeout_s: float) -> None:
        parts = urlsplit(base_url) if isinstance(base_url, str) else None
        if (
            parts is None
            or parts.scheme != "http"
            or parts.hostname is None
            or parts.username is not None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or not _loopback(parts.hostname)
        ):
            raise ValueError("the Ollama URL must be http://<loopback host>:<port> (127.0.0.1, ::1 or localhost)")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a number > 0")
        self._host = parts.hostname
        self._port = parts.port or 11434
        self._timeout = timeout_s

    def chat(self, request: dict[str, Any]) -> Any:
        return self._call("POST", "/api/chat", request)

    def version(self) -> Any:
        return self._call("GET", "/api/version", None)

    def tags(self) -> Any:
        return self._call("GET", "/api/tags", None)

    def show(self, model: str) -> Any:
        return self._call("POST", "/api/show", {"model": model})

    def _call(self, method: str, path: str, body: Any) -> Any:
        payload = None if body is None else canonical_json(body).encode("utf-8")
        connection = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            headers = {"Content-Type": "application/json"} if payload is not None else {}
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            status = response.status
            data = response.read()
        except (OSError, http.client.HTTPException) as exc:
            raise OllamaTransportError(f"{method} {path} failed: {exc}") from exc
        finally:
            connection.close()
        text = data.decode("utf-8", errors="replace")
        if status >= 500:
            raise OllamaTransportError(f"{method} {path} returned HTTP {status}: {text[:300]}")
        if status >= 400:
            if "context" in text.lower():
                raise OllamaContextError(f"{method} {path} returned HTTP {status}: {text[:300]}")
            raise OllamaProtocolError(f"{method} {path} returned HTTP {status}: {text[:300]}")
        try:
            return json.loads(data)
        except ValueError as exc:
            raise OllamaProtocolError(f"{method} {path} did not return JSON") from exc


def make_ollama_client(base_url: str, model_config: dict[str, Any] | None = None) -> OllamaHTTPClient:
    """``OllamaHTTPClient(base_url, timeout_s=model_config["client"]["timeout_s"])``（預設 LOCAL_MODEL_CONFIG）。"""
    settings = (LOCAL_MODEL_CONFIG if model_config is None else model_config)["client"]
    return OllamaHTTPClient(base_url, timeout_s=settings["timeout_s"])


def verify_server(client: Any, config: AgentConfig) -> dict[str, Any]:
    """確認 server 上的模型（decision 026 §3）；回傳 ``{"ollama_version", "model", "model_digest", "quantization", "capabilities"}``。
    - ``tags()`` 必須列出這個 model tag，且不是遠端（cloud）模型；digest 以 ``sha256:<hex>`` 表示。
    - ``show(model)`` 的 capabilities 必須包含 "tools"。
    - 設定已凍結時，Ollama 版本與 digest 必須等於凍結值。
    任何不符 → ValueError。"""
    config = check_local_config(config)
    model_config = config.model_config
    model = model_config["model"]
    version = client.version()
    tags = client.tags()
    if not isinstance(version, dict) or not isinstance(version.get("version"), str):
        raise ValueError("the server did not report its version")
    entries = [entry for entry in (tags.get("models") if isinstance(tags, dict) else None) or [] if isinstance(entry, dict)]
    matches = [entry for entry in entries if model in (entry.get("name"), entry.get("model"))]
    if len(matches) != 1:
        raise ValueError(f"the server does not have exactly one model tagged {model!r} (it must not be pulled by this project)")
    entry = matches[0]
    if "remote_host" in entry or "remote_model" in entry:
        raise ValueError(f"{model!r} is a remote (cloud) model")
    digest = entry.get("digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("the server did not report the model digest")
    digest = digest if digest.startswith("sha256:") else f"sha256:{digest}"
    shown = client.show(model)
    capabilities = shown.get("capabilities") if isinstance(shown, dict) else None
    if not isinstance(capabilities, list) or "tools" not in capabilities:
        raise ValueError(f"{model!r} does not report the tools capability")
    details = shown.get("details") if isinstance(shown.get("details"), dict) else entry.get("details") or {}
    info = {
        "ollama_version": version["version"],
        "model": model,
        "model_digest": digest,
        "quantization": details.get("quantization_level") if isinstance(details, dict) else None,
        "capabilities": list(capabilities),
    }
    if is_frozen(config):
        if info["ollama_version"] != model_config["ollama_version"]:
            raise ValueError(f"Ollama {info['ollama_version']} is not the frozen version {model_config['ollama_version']}")
        if info["model_digest"] != model_config["model_digest"]:
            raise ValueError(f"{model!r} has digest {info['model_digest']}, not the frozen {model_config['model_digest']}")
    return info
