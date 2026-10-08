"""測試用的 fake Anthropic client（decision 009：adapter 以可注入的 client 實作，不需要 API key）。

物件形狀模仿 SDK 的 response：以屬性讀取 ``stop_reason``、``content``（blocks）、``usage``。
``FakeClient.beta.messages.create(**request)``（adapter 使用的 beta namespace；``FakeClient.messages`` 指向同一個記錄器）
會深拷貝並記錄每次 request，回傳預先排好的 response（或呼叫 responder(request) 產生）。不連網、不需要任何 credential。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    data: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any = None
    extra: dict[str, Any] = field(default_factory=dict)  # 模擬 SDK block 上其他不該被保存的屬性


def text_block(text: str) -> FakeBlock:
    return FakeBlock(type="text", text=text, extra={"citations": None})


def thinking_block(signature: str, thinking: str = "") -> FakeBlock:
    return FakeBlock(type="thinking", thinking=thinking, signature=signature)


def redacted_thinking_block(data: str) -> FakeBlock:
    return FakeBlock(type="redacted_thinking", data=data)


def tool_use_block(tool_id: str, name: str, input: Any) -> FakeBlock:
    return FakeBlock(type="tool_use", id=tool_id, name=name, input=input)


@dataclass
class FakeCacheCreation:
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation: FakeCacheCreation | None = None


@dataclass
class FakeStopDetails:
    category: str | None = None
    explanation: str | None = None


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str | None
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "claude-opus-5-5"
    stop_details: FakeStopDetails | None = None
    input_transformations: list[Any] | None = None
    _request_id: str | None = "req_fake"


def tool_turn(tool_id: str, name: str, args: Any, *, signature: str = "sig", note: str | None = None,
              usage: FakeUsage | None = None) -> FakeResponse:
    """thinking block（+ 可選 text）+ 一個 tool_use，stop_reason="tool_use"。"""
    blocks = [thinking_block(signature)]
    if note is not None:
        blocks.append(text_block(note))
    blocks.append(tool_use_block(tool_id, name, args))
    return FakeResponse(content=blocks, stop_reason="tool_use", usage=usage or FakeUsage(100, 20))


def final_turn(text: str = "Done.", *, signature: str = "sig-final", usage: FakeUsage | None = None) -> FakeResponse:
    return FakeResponse(
        content=[thinking_block(signature), text_block(text)], stop_reason="end_turn", usage=usage or FakeUsage(100, 10)
    )


class FakeMessages:
    def __init__(self, responses: list[FakeResponse] | None = None,
                 responder: Callable[[dict[str, Any]], FakeResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> FakeResponse:
        self.requests.append(copy.deepcopy(request))
        if self._responder is not None:
            return self._responder(copy.deepcopy(request))
        if not self._responses:
            raise AssertionError("FakeMessages: no more scripted responses")
        return self._responses.pop(0)


class _FakeBeta:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


class FakeClient:
    def __init__(self, responses: list[FakeResponse] | None = None,
                 responder: Callable[[dict[str, Any]], FakeResponse] | None = None) -> None:
        self.messages = FakeMessages(responses, responder)
        self.beta = _FakeBeta(self.messages)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests
