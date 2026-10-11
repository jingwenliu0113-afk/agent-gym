"""agents/ollama.py 的契約測試（decision 026）：request 組成與重送、tool calling 與參數解析、JSON 結構驗證、截斷防護、
錯誤分類、只連 loopback 與允許的端點、server 釘選檢查。

全部只用假的 client 與 127.0.0.1 上的測試 HTTP server，不連到任何外部服務。
"""

from __future__ import annotations

import copy
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Iterator

import pytest

from agents.baseline import (
    BASELINE_DEVELOPER_PROMPT,
    BASELINE_PROMPT_VERSION,
    BASELINE_SYSTEM_PROMPT,
    AdapterError,
    InfrastructureError,
)
from agents.ollama import (
    LOCAL_MODEL_CONFIG,
    OLLAMA_ADAPTER_VERSION,
    OLLAMA_POLICY,
    OllamaContextError,
    OllamaHTTPClient,
    OllamaPolicy,
    OllamaProtocolError,
    OllamaTransportError,
    build_chat_request,
    is_frozen,
    local_agent_config,
    parse_chat_response,
    verify_server,
)
from agents.policy import AgentConfig, ModelMessage, ToolCall
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from replay.aborts import classify_exception

MODEL = "qwen3.5:9b-q4_K_M"
TOOLS = IssueTrackerEnv.tool_specs()
DIGEST = "sha256:" + "ab" * 32


def _config(**changes: Any) -> AgentConfig:
    config = local_agent_config()
    model_config = copy.deepcopy(config.model_config)
    model_config.update(changes)
    return AgentConfig(config.policy, config.prompt_version, config.system_prompt, config.developer_prompt, model_config)


def _assistant(text: str, calls: list[tuple[str, str, Any]], raw_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    raw = {
        "role": "assistant",
        "content": text,
        "tool_calls": raw_calls if raw_calls is not None else [
            {"function": {"name": name, "arguments": args}} for _, name, args in calls
        ],
    }
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": call_id, "name": name, "args": args} for call_id, name, args in calls],
        "provider_content": [raw],
    }


def _conversation() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "developer", "content": BASELINE_DEVELOPER_PROMPT},
        {"role": "user", "content": "Close #3 as a duplicate of #1."},
        _assistant("", [("call_0_0", "get_issue", {"issue_id": 3})]),
        {"role": "tool", "tool_call_id": "call_0_0", "content": canonical_json({"ok": True, "data": {"id": 3}})},
    ]


def _response(message: dict[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "model": MODEL,
        "created_at": "2026-10-11T00:00:00Z",
        "message": message if message is not None else {
            "role": "assistant", "content": "", "tool_calls": [{"function": {"name": "update_issue", "arguments": {"issue_id": 3}}}],
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 5,
        "load_duration": 1,
        "prompt_eval_count": 900,
        "prompt_eval_duration": 2,
        "eval_count": 40,
        "eval_duration": 2,
    }
    response.update(changes)
    return response


# ---------------------------------------------------------------- 設定


def test_local_agent_config_is_the_draft_of_decision_026() -> None:
    config = local_agent_config()
    assert config.policy == OLLAMA_POLICY and config.prompt_version == BASELINE_PROMPT_VERSION
    assert config.system_prompt == BASELINE_SYSTEM_PROMPT and config.developer_prompt == BASELINE_DEVELOPER_PROMPT
    model_config = config.model_config
    assert model_config["provider"] == "ollama" and model_config["adapter"] == OLLAMA_ADAPTER_VERSION
    assert model_config["model"] == MODEL and set(model_config) == set(LOCAL_MODEL_CONFIG)
    assert model_config["options"]["temperature"] > 0  # greedy 會讓 rerun 都相同（decision 026 §5）
    assert "seed" not in model_config["options"]  # seed 每次呼叫隨機、記在 meta
    assert model_config["model_digest"] is None and model_config["ollama_version"] is None and is_frozen(config) is False
    assert local_agent_config() is not config and local_agent_config().config_id == config.config_id
    assert is_frozen(_config(model_digest=DIGEST, ollama_version="0.35.0")) is True


# ---------------------------------------------------------------- request


def test_build_chat_request_shape_and_determinism() -> None:
    model_config = local_agent_config().model_config
    request = build_chat_request(_conversation(), TOOLS, model_config, seed=7)
    assert set(request) == {"model", "messages", "tools", "stream", "think", "options", "truncate", "shift"}
    assert request["model"] == MODEL and request["stream"] is False and request["truncate"] is False and request["shift"] is False
    assert request["think"] is model_config["think"]
    assert request["options"] == {**model_config["options"], "seed": 7}
    assert request["messages"][0] == {"role": "system", "content": BASELINE_SYSTEM_PROMPT + "\n\n" + BASELINE_DEVELOPER_PROMPT}
    assert request["messages"][1] == {"role": "user", "content": "Close #3 as a duplicate of #1."}
    assert request["messages"][2] == _conversation()[3]["provider_content"][0]  # 原樣重送
    assert request["messages"][3] == {"role": "tool", "content": _conversation()[4]["content"], "tool_name": "get_issue"}
    assert [tool["function"]["name"] for tool in request["tools"]] == [spec["name"] for spec in TOOLS]
    first = request["tools"][0]
    assert first == {"type": "function", "function": {"name": TOOLS[0]["name"], "description": TOOLS[0]["description"],
                                                      "parameters": TOOLS[0]["input_schema"]}}
    again = build_chat_request(copy.deepcopy(_conversation()), TOOLS, model_config, seed=7)
    assert canonical_json(again) == canonical_json(request)


def test_a_blank_developer_prompt_is_not_sent() -> None:
    messages = _conversation()
    messages[1]["content"] = "   "
    request = build_chat_request(messages, TOOLS, local_agent_config().model_config, seed=1)
    assert request["messages"][0] == {"role": "system", "content": BASELINE_SYSTEM_PROMPT}


def _without_provider_content(messages: list[dict[str, Any]]) -> None:
    del messages[3]["provider_content"]


def _tool_id_mismatch(messages: list[dict[str, Any]]) -> None:
    messages[4]["tool_call_id"] = "call_9_9"


def _inconsistent_raw_call(messages: list[dict[str, Any]]) -> None:
    messages[3]["provider_content"][0]["tool_calls"][0]["function"]["name"] = "delete_everything"


def _inconsistent_text(messages: list[dict[str, Any]]) -> None:
    messages[3]["provider_content"][0]["content"] = "something else"


def _extra_raw_key(messages: list[dict[str, Any]]) -> None:
    messages[3]["provider_content"][0]["images"] = []


BAD_CONVERSATIONS: dict[str, Callable[[list[dict[str, Any]]], Any]] = {
    "no-provider-content": _without_provider_content,
    "tool-id-mismatch": _tool_id_mismatch,
    "raw-call-mismatch": _inconsistent_raw_call,
    "raw-text-mismatch": _inconsistent_text,
    "raw-extra-key": _extra_raw_key,
    "blank-system": lambda m: m[0].update(content="  "),
    "blank-user": lambda m: m[2].update(content=""),
    "wrong-order": lambda m: m.insert(1, m.pop(2)),
    "missing-tool-result": lambda m: m.pop(),
    "assistant-without-calls": lambda m: m.__setitem__(3, _assistant("done", [])),
}


@pytest.mark.parametrize("change", sorted(BAD_CONVERSATIONS))
def test_build_chat_request_rejects_malformed_conversations(change: str) -> None:
    messages = _conversation()
    BAD_CONVERSATIONS[change](messages)
    with pytest.raises(ValueError):
        build_chat_request(messages, TOOLS, local_agent_config().model_config, seed=1)


def test_build_chat_request_rejects_empty_tools() -> None:
    with pytest.raises(ValueError):
        build_chat_request(_conversation(), (), local_agent_config().model_config, seed=1)


# ---------------------------------------------------------------- response：tool calling 與 JSON 驗證


def _parse(response: dict[str, Any], t: int = 1, seed: int = 5) -> Any:
    return parse_chat_response(response, local_agent_config().model_config, assistant_turn=t, seed=seed)


def test_a_tool_call_becomes_a_tool_call_with_a_positional_id() -> None:
    step = _parse(_response(), t=1)
    assert step.message.tool_calls == (ToolCall("call_1_0", "update_issue", {"issue_id": 3}),)
    assert step.message.text == "" and step.message.stop_reason == "tool_calls"
    assert step.message.provider_content == [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "update_issue", "arguments": {"issue_id": 3}}}]}
    ]
    assert (step.input_tokens, step.output_tokens, step.cost_usd) == (900, 40, 0.0)
    assert step.meta["response_model"] == MODEL and step.meta["seed"] == 5 and step.meta["tool_use_blocks"] == 1
    assert step.meta["done_reason"] == "stop" and step.meta["durations"]["total"] == 5


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"issue_id": 3}, {"issue_id": 3}),
        ('{"issue_id": 3}', {"issue_id": 3}),
        (None, {}),
        ("not json", "not json"),
        ("[1, 2]", "[1, 2]"),
        ([1, 2], [1, 2]),
        (7, 7),
    ],
    ids=["object", "json-string", "null", "bad-string", "string-array", "array", "number"],
)
def test_tool_call_arguments(arguments: Any, expected: Any) -> None:
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_issue", "arguments": arguments}}]}
    step = _parse(_response(message))
    assert step.message.tool_calls[0].args == expected  # 不是物件 → 原樣交給 env（invalid_arguments）
    assert step.message.provider_content[0]["tool_calls"][0]["function"]["arguments"] == arguments  # 原文保留


def test_several_calls_text_and_thinking() -> None:
    message = {
        "role": "assistant", "content": "Working on it.", "thinking": "plan",
        "tool_calls": [
            {"function": {"name": "get_issue", "arguments": {"issue_id": 1}}, "id": "ollama-own-id"},
            {"function": {"name": "get_issue", "arguments": {"issue_id": 2}}},
        ],
    }
    step = _parse(_response(message), t=4)
    assert [call.id for call in step.message.tool_calls] == ["call_4_0", "call_4_1"]
    assert step.message.text == "Working on it." and step.meta["tool_use_blocks"] == 2
    assert step.message.provider_content[0]["thinking"] == "plan"
    assert step.message.provider_content[0]["tool_calls"][0]["id"] == "ollama-own-id"


def test_a_text_only_reply_ends_the_episode() -> None:
    step = _parse(_response({"role": "assistant", "content": "All done."}))
    assert step.message.tool_calls == () and step.message.text == "All done." and step.message.stop_reason == "stop"
    assert step.message.provider_content == [{"role": "assistant", "content": "All done."}]


def test_missing_eval_count_is_zero_output_tokens() -> None:
    response = _response()
    del response["eval_count"]
    assert _parse(response).output_tokens == 0


def test_output_truncated_at_num_predict_is_an_agent_output_error() -> None:
    with pytest.raises(AdapterError) as info:  # length 且輸出達到 num_predict（decision 027 §2）
        _parse(_response(done_reason="length", eval_count=LOCAL_MODEL_CONFIG["options"]["num_predict"]))
    assert info.value.billed_usage["cost_usd"] == 0.0  # type: ignore[attr-defined]
    assert classify_exception(info.value) == "agent_output"


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(done=False),
        lambda r: r.update(done_reason="load"),
        lambda r: r.update(message="hi"),
        lambda r: r["message"].update(role="user"),
        lambda r: r["message"].update(content=3),
        lambda r: r["message"].update(thinking=3),
        lambda r: r["message"].update(tool_calls={"function": {}}),
        lambda r: r["message"].update(tool_calls=["x"]),
        lambda r: r["message"].update(tool_calls=[{"function": "x"}]),
        lambda r: r["message"].update(unexpected=1),
        lambda r: r.update(prompt_eval_count="900"),
        lambda r: r.update(eval_count=-1),
        lambda r: r.update(model="qwen3:8b"),
        lambda r: r.update(remote_model="qwen3.5:cloud"),
        lambda r: r.update(remote_host="https://ollama.com"),
    ],
    ids=["not-done", "done-reason", "message-str", "role", "content-type", "thinking-type", "tool-calls-dict", "tool-call-str",
         "function-str", "unknown-message-key", "count-str", "count-negative", "other-model", "remote-model", "remote-host"],
)
def test_structural_problems_are_protocol_errors(change: Callable[[dict[str, Any]], Any]) -> None:
    response = _response()
    change(response)
    with pytest.raises(OllamaProtocolError) as info:
        _parse(response)
    assert classify_exception(info.value) == "harness"


@pytest.mark.parametrize("name", ["", None, 3])
def test_a_tool_call_without_a_name_is_an_agent_output_error(name: Any) -> None:
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": {}}}]}
    with pytest.raises(AdapterError):
        _parse(_response(message))


def test_a_response_that_is_not_an_object_is_a_protocol_error() -> None:
    with pytest.raises(OllamaProtocolError):
        _parse([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop("prompt_eval_count"),
        lambda r: r.update(prompt_eval_count=0),
        lambda r: r.update(prompt_eval_count=LOCAL_MODEL_CONFIG["options"]["num_ctx"] - 10, eval_count=10),
    ],
    ids=["count-missing", "count-zero", "at-num-ctx"],
)
def test_context_guards(change: Callable[[dict[str, Any]], Any]) -> None:
    response = _response()
    change(response)
    with pytest.raises(OllamaContextError) as info:
        _parse(response)
    assert classify_exception(info.value) == "harness"


# ---------------------------------------------------------------- OllamaPolicy


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = responses

    def chat(self, request: dict[str, Any]) -> Any:
        self.requests.append(copy.deepcopy(request))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def version(self) -> dict[str, Any]:  # 服務仍在（decision 027 §1 的探測）
        return {"version": "0.0.0-test"}


def test_policy_sends_the_built_request_with_a_fresh_seed_per_call() -> None:
    seeds = iter([11, 22])
    client = _FakeClient([_response(), _response()])
    policy = OllamaPolicy(client, local_agent_config(), seed_source=lambda: next(seeds))
    first = policy.next_message(_conversation(), TOOLS)
    second = policy.next_message(_conversation(), TOOLS)
    model_config = local_agent_config().model_config
    assert client.requests[0] == build_chat_request(_conversation(), TOOLS, model_config, seed=11)
    assert client.requests[1]["options"]["seed"] == 22 and (first.meta["seed"], second.meta["seed"]) == (11, 22)
    assert first.message.tool_calls[0].id == "call_1_0"  # 第 2 個 assistant 訊息（前面已有 1 個）
    assert policy.agent_config.config_id == local_agent_config().config_id


def test_the_default_seed_is_a_positive_31_bit_int() -> None:
    client = _FakeClient([_response()])
    step = OllamaPolicy(client, local_agent_config()).next_message(_conversation(), TOOLS)
    assert type(step.meta["seed"]) is int and 1 <= step.meta["seed"] < 2**31


def test_an_oversized_conversation_is_refused_before_any_call() -> None:
    messages = _conversation()
    messages[2]["content"] = "x" * (2 * LOCAL_MODEL_CONFIG["options"]["num_ctx"])
    client = _FakeClient([_response()])
    with pytest.raises(OllamaContextError):
        OllamaPolicy(client, local_agent_config()).next_message(messages, TOOLS)
    assert client.requests == []


def test_transport_errors_pass_through_and_are_infrastructure() -> None:
    client = _FakeClient([OllamaTransportError("connection refused")])
    with pytest.raises(OllamaTransportError) as info:
        OllamaPolicy(client, local_agent_config()).next_message(_conversation(), TOOLS)
    assert isinstance(info.value, InfrastructureError) and classify_exception(info.value) == "infrastructure"


@pytest.mark.parametrize(
    "config",
    [
        lambda: AgentConfig("anthropic-messages", BASELINE_PROMPT_VERSION, "s", "", local_agent_config().model_config),
        lambda: _config(adapter="ollama-chat/v0"),
        lambda: _config(model="qwen3.5:cloud"),
        lambda: _config(model="gpt-oss:120b-cloud"),
        lambda: _config(extra=1),
        lambda: _config(options={**LOCAL_MODEL_CONFIG["options"], "seed": 1}),
        lambda: _config(options={k: v for k, v in LOCAL_MODEL_CONFIG["options"].items() if k != "num_ctx"}),
        lambda: _config(think="yes"),
        lambda: AgentConfig(OLLAMA_POLICY, BASELINE_PROMPT_VERSION, "  ", "", local_agent_config().model_config),
    ],
    ids=["policy", "adapter", "cloud", "cloud-suffix", "extra-key", "fixed-seed", "no-num-ctx", "think-type", "blank-system"],
)
def test_policy_rejects_invalid_configs(config: Callable[[], AgentConfig]) -> None:
    with pytest.raises(ValueError):
        OllamaPolicy(_FakeClient([]), config())


# ---------------------------------------------------------------- restore 的忠實度（009 的不變式）


def test_a_restored_conversation_produces_the_same_request() -> None:
    from replay.trajectory import render_messages  # noqa: F401  (確認 conversation 形狀來自 recorder)

    messages = _conversation()
    restored = json.loads(canonical_json(messages))  # save / load 之後
    model_config = local_agent_config().model_config
    assert canonical_json(build_chat_request(restored, TOOLS, model_config, seed=3)) == canonical_json(
        build_chat_request(messages, TOOLS, model_config, seed=3)
    )
    message = ModelMessage.from_dict(
        {"text": "", "tool_calls": [{"id": "call_0_0", "name": "get_issue", "args": {"issue_id": 3}}], "stop_reason": "tool_calls",
         "provider_content": messages[3]["provider_content"]}
    )
    assert message.provider_content == messages[3]["provider_content"]


# ---------------------------------------------------------------- HTTP client：只連 loopback、只呼叫允許的端點


@pytest.mark.parametrize(
    "url",
    ["http://gpu-server:11434", "http://10.0.0.5:11434", "https://127.0.0.1:11434", "ftp://127.0.0.1", "http://example.com",
     "http://127.0.0.1:11434/api", "http://user@127.0.0.1:11434", "127.0.0.1:11434"],
)
def test_the_http_client_only_accepts_loopback_urls(url: str) -> None:
    with pytest.raises(ValueError):
        OllamaHTTPClient(url, timeout_s=5)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://[::1]:11434", "http://localhost:11434", "http://127.0.0.1:18080"])
def test_loopback_urls_are_accepted(url: str) -> None:
    OllamaHTTPClient(url, timeout_s=5)


def test_the_http_client_has_no_model_management_methods() -> None:
    client = OllamaHTTPClient("http://127.0.0.1:11434", timeout_s=5)
    for name in ("pull", "push", "create", "delete", "copy", "generate", "embed"):
        assert not hasattr(client, name)


class _Handler(BaseHTTPRequestHandler):
    routes: dict[tuple[str, str], tuple[int, bytes]] = {}
    seen: list[tuple[str, str, bytes]] = []

    def _reply(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        type(self).seen.append((method, self.path, body))
        status, payload = type(self).routes.get((method, self.path), (404, b'{"error": "not found"}'))
        if status == 0:  # 不回應（逾時）
            threading.Event().wait(3)
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._reply("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._reply("POST")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("Handler", (_Handler,), {"routes": {}, "seen": []})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_round_trip_over_http(server: tuple[str, type[_Handler]]) -> None:
    url, handler = server
    handler.routes[("POST", "/api/chat")] = (200, json.dumps(_response()).encode())
    client = OllamaHTTPClient(url, timeout_s=5)
    request = build_chat_request(_conversation(), TOOLS, local_agent_config().model_config, seed=3)
    assert client.chat(request) == _response()
    method, path, body = handler.seen[-1]
    assert (method, path) == ("POST", "/api/chat") and json.loads(body) == request


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (500, b'{"error": "boom"}', OllamaTransportError),
        (503, b"busy", OllamaTransportError),
        (400, b'{"error": "the input length exceeds the context length"}', OllamaContextError),
        (400, b'{"error": "invalid options"}', OllamaProtocolError),
        (404, b'{"error": "model not found"}', OllamaProtocolError),
        (200, b"not json", OllamaProtocolError),
    ],
    ids=["500", "503", "context-400", "other-400", "404", "bad-json"],
)
def test_http_errors_are_classified(status: int, body: bytes, error: type[Exception], server: tuple[str, type[_Handler]]) -> None:
    url, handler = server
    handler.routes[("POST", "/api/chat")] = (status, body)
    with pytest.raises(error):
        OllamaHTTPClient(url, timeout_s=5).chat({"model": MODEL})


def test_connection_refused_and_timeouts_are_transport_errors(server: tuple[str, type[_Handler]]) -> None:
    url, handler = server
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(OllamaTransportError):
        OllamaHTTPClient(f"http://127.0.0.1:{closed_port}", timeout_s=2).version()
    handler.routes[("GET", "/api/version")] = (0, b"")
    with pytest.raises(OllamaTransportError):
        OllamaHTTPClient(url, timeout_s=1).version()


def test_version_tags_and_show_use_the_documented_endpoints(server: tuple[str, type[_Handler]]) -> None:
    url, handler = server
    handler.routes[("GET", "/api/version")] = (200, b'{"version": "0.35.0"}')
    handler.routes[("GET", "/api/tags")] = (200, b'{"models": []}')
    handler.routes[("POST", "/api/show")] = (200, b'{"capabilities": ["tools"]}')
    client = OllamaHTTPClient(url, timeout_s=5)
    assert client.version() == {"version": "0.35.0"} and client.tags() == {"models": []}
    assert client.show(MODEL) == {"capabilities": ["tools"]}
    assert [(m, p) for m, p, _ in handler.seen] == [("GET", "/api/version"), ("GET", "/api/tags"), ("POST", "/api/show")]
    assert json.loads(handler.seen[-1][2]) == {"model": MODEL}


# ---------------------------------------------------------------- verify_server：釘選與能力


class _ServerFake:
    def __init__(self, *, version: str = "0.35.0", digest: str = DIGEST, capabilities: tuple[str, ...] = ("completion", "tools"),
                 models: list[dict[str, Any]] | None = None) -> None:
        self._version = version
        self._models = models if models is not None else [{"name": MODEL, "model": MODEL, "digest": digest.removeprefix("sha256:"),
                                                           "details": {"quantization_level": "Q4_K_M"}}]
        self._capabilities = capabilities

    def version(self) -> dict[str, Any]:
        return {"version": self._version}

    def tags(self) -> dict[str, Any]:
        return {"models": self._models}

    def show(self, model: str) -> dict[str, Any]:
        return {"capabilities": list(self._capabilities), "details": {"quantization_level": "Q4_K_M"}}


def test_verify_server_reports_the_running_model() -> None:
    info = verify_server(_ServerFake(), local_agent_config())
    assert info == {"ollama_version": "0.35.0", "model": MODEL, "model_digest": DIGEST, "quantization": "Q4_K_M",
                    "capabilities": ["completion", "tools"]}


def test_verify_server_checks_frozen_values() -> None:
    frozen = _config(model_digest=DIGEST, ollama_version="0.35.0")
    verify_server(_ServerFake(), frozen)
    with pytest.raises(ValueError):
        verify_server(_ServerFake(version="0.36.0"), frozen)
    with pytest.raises(ValueError):
        verify_server(_ServerFake(digest="sha256:" + "cd" * 32), frozen)


@pytest.mark.parametrize(
    "server_fake",
    [
        lambda: _ServerFake(models=[]),
        lambda: _ServerFake(capabilities=("completion",)),
        lambda: _ServerFake(models=[{"name": MODEL, "model": MODEL, "digest": "ab" * 32, "remote_host": "https://ollama.com"}]),
    ],
    ids=["model-missing", "no-tools", "remote"],
)
def test_verify_server_refuses(server_fake: Callable[[], _ServerFake]) -> None:
    with pytest.raises(ValueError):
        verify_server(server_fake(), local_agent_config())
