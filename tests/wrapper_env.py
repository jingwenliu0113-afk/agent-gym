"""測試用：模擬在 tools/with-gpu-lock 之內（decision 029）——GPU lock 已持有、本 Project 的 server 已啟動並確認。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents.ollama_serve import SERVER_ENV
from experiments.run_baseline import GPU_LOCK_ENV, SERVER_INFO_ENV

SERVER_INFO: dict[str, Any] = {
    "pid": 4242,
    "port": 11436,
    "version": "0.35.0",
    "gpu": {"used_mib": 675, "total_mib": 16303},
    "env": dict(SERVER_ENV),
    "models": {"source": "~/ii-ollama-models", "blobs": 4},
}


def inside_wrapper(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    monkeypatch.setenv(GPU_LOCK_ENV, "held")
    monkeypatch.setenv(SERVER_INFO_ENV, json.dumps({**SERVER_INFO, **overrides}))


def outside_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GPU_LOCK_ENV, raising=False)
    monkeypatch.delenv(SERVER_INFO_ENV, raising=False)
