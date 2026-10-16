"""測試用：模擬在 tools/with-gpu-lock 之內（decision 029 / 031）——GPU lock 已持有、本 Project 的 server 已啟動並確認、
通道的 URL 已設定。模擬的 server 是凍結的那一個（decision 030）：``frozen_server`` 的假 client 回報凍結的版本與 digest，
wrapper 的報告也是凍結的版本（031 §3：兩者必須相同）。"""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from agents.ollama import FROZEN_MODEL_DIGEST, FROZEN_OLLAMA_VERSION
from agents.ollama_serve import SERVER_ENV
from experiments.run_baseline import GPU_LOCK_ENV, OLLAMA_URL_ENV, SERVER_INFO_ENV, OllamaOracleFakeClient
from tasks.generator import GeneratedInstance

WRAPPER_URL = "http://127.0.0.1:11435"

SERVER_INFO: dict[str, Any] = {
    "pid": 4242,
    "port": 11436,
    "version": FROZEN_OLLAMA_VERSION,
    "gpu": {"used_mib": 675, "total_mib": 16303},
    "env": dict(SERVER_ENV),
    "models": {"source": "~/ii-ollama-models", "blobs": 4},
}


def inside_wrapper(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    monkeypatch.setenv(GPU_LOCK_ENV, "held")
    monkeypatch.setenv(OLLAMA_URL_ENV, WRAPPER_URL)
    monkeypatch.setenv(SERVER_INFO_ENV, json.dumps({**SERVER_INFO, **overrides}))


def outside_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (GPU_LOCK_ENV, OLLAMA_URL_ENV, SERVER_INFO_ENV):
        monkeypatch.delenv(name, raising=False)


def frozen_server(instances: Sequence[GeneratedInstance]) -> OllamaOracleFakeClient:
    """模擬凍結的 server（版本與 digest 等於 decision 030 的凍結值）的 oracle 假 client。"""
    return OllamaOracleFakeClient(instances, version=FROZEN_OLLAMA_VERSION, digest=FROZEN_MODEL_DIGEST)

