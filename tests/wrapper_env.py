"""測試用：模擬在 tools/with-gpu-lock 之內（decision 029 / 031）——GPU lock 已持有、本 Project 的 server 已啟動並確認、
通道的 URL 已設定。版本與 dry-run 的假 client 相同（031 §3：wrapper 回報的版本必須等於 verify_server 讀到的版本）。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents.ollama_serve import SERVER_ENV
from experiments.run_baseline import FAKE_OLLAMA_VERSION, GPU_LOCK_ENV, OLLAMA_URL_ENV, SERVER_INFO_ENV

WRAPPER_URL = "http://127.0.0.1:11435"

SERVER_INFO: dict[str, Any] = {
    "pid": 4242,
    "port": 11436,
    "version": FAKE_OLLAMA_VERSION,
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
