"""decision 029：agent-gym 自己的 Ollama server——設定與 CLI 的部分（server 的生命週期見 test_ollama_serve.py、
test_with_gpu_lock.py）。

- server 的環境（SERVER_ENV）屬於模型設定：``model_config["server_env"]``，config_id 涵蓋它。
- 本機 run（run_baseline / run_sweep / local_probe）只對 wrapper 啟動並確認的 server 送請求：``AGENT_GYM_OLLAMA_SERVER`` 必須存在、
  是 JSON 物件，而且 env 等於設定的 server_env；否則在任何模型呼叫之前拒絕。
- ``local_probe --verify-server``：只檢查 server（版本、模型 digest、capabilities），不載入模型。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from agents import ollama
from agents.ollama import check_local_config, local_agent_config
from agents.ollama_serve import SERVER_ENV
from agents.policy import AgentConfig
from experiments import local_probe, run_baseline
from experiments.run_baseline import (
    FAKE_OLLAMA_DIGEST,
    SERVER_INFO_ENV,
    OllamaOracleFakeClient,
    check_wrapper_server,
)
from tasks.generator import week1_pool
from tests.wrapper_env import SERVER_INFO, inside_wrapper, outside_wrapper

# ---------------------------------------------------------------- 設定


def test_the_server_environment_is_part_of_the_model_config() -> None:
    config = local_agent_config()
    assert config.model_config["server_env"] == SERVER_ENV
    changed = copy.deepcopy(config.model_config)
    changed["server_env"] = {**SERVER_ENV, "OLLAMA_NUM_PARALLEL": "2"}
    other = AgentConfig(policy=config.policy, prompt_version=config.prompt_version, system_prompt=config.system_prompt,
                        developer_prompt=config.developer_prompt, model_config=changed)
    assert other.config_id != config.config_id


@pytest.mark.parametrize(
    "server_env",
    [None, [], {"OLLAMA_NUM_PARALLEL": 1}, {"OLLAMA_HOST": "127.0.0.1:1"}, {"OLLAMA_MODELS": "/x"}, {"HOME": "/x"},
     {"not_ollama": "1"}],
    ids=["none", "list", "int-value", "host", "models", "home", "other-name"],
)
def test_invalid_server_environments_are_rejected(server_env: Any) -> None:
    config = local_agent_config()
    model_config = copy.deepcopy(config.model_config)
    model_config["server_env"] = server_env
    with pytest.raises(ValueError):
        check_local_config(AgentConfig(policy=config.policy, prompt_version=config.prompt_version,
                                       system_prompt=config.system_prompt, developer_prompt=config.developer_prompt,
                                       model_config=model_config))


# ---------------------------------------------------------------- 只對 wrapper 啟動的 server 送請求


def test_check_wrapper_server_accepts_the_wrappers_report(monkeypatch: pytest.MonkeyPatch) -> None:
    inside_wrapper(monkeypatch)
    assert check_wrapper_server(local_agent_config()) == SERVER_INFO


@pytest.mark.parametrize(
    "raw",
    [None, "", "not json", "[]", json.dumps({"env": {**SERVER_ENV, "OLLAMA_NUM_PARALLEL": "4"}}), json.dumps({"pid": 1}),
     json.dumps({**SERVER_INFO, "env": {k: v for k, v in SERVER_ENV.items() if k != "OLLAMA_NOPRUNE"}})],
    ids=["missing", "empty", "not-json", "not-object", "other-env", "no-env", "missing-key"],
)
def test_check_wrapper_server_refuses_anything_else(raw: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    if raw is None:
        monkeypatch.delenv(SERVER_INFO_ENV, raising=False)
    else:
        monkeypatch.setenv(SERVER_INFO_ENV, raw)
    with pytest.raises(ValueError):
        check_wrapper_server(local_agent_config())


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def make(base_url: str, model_config: Any = None) -> OllamaOracleFakeClient:
        urls.append(base_url)
        return OllamaOracleFakeClient(week1_pool("dev"))

    monkeypatch.setattr(run_baseline, "make_ollama_client", make)
    monkeypatch.setattr(local_probe, "make_ollama_client", make)
    return urls


def test_a_local_run_without_the_servers_report_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    monkeypatch.delenv(SERVER_INFO_ENV)
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "dev", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "x"])
    assert made == [] and not (tmp_path / "runs" / "x").exists()


def test_a_local_run_against_a_server_with_other_settings_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch, env={**SERVER_ENV, "OLLAMA_NUM_PARALLEL": "4"})
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "dev", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "x"])
    assert made == [] and not (tmp_path / "runs" / "x").exists()


def test_a_local_run_inside_the_wrapper_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list[str],
                                             capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    assert run_baseline.main(["--split", "dev", "--confirm-local-run", "--max-model-calls", "200", "--run-name", "x"]) == 0
    err = capsys.readouterr().err
    assert "local model server" in err and '"pid":4242' in err.replace(" ", "")


def test_the_truncation_check_needs_the_servers_report(monkeypatch: pytest.MonkeyPatch, made: list[str]) -> None:
    inside_wrapper(monkeypatch)
    monkeypatch.delenv(SERVER_INFO_ENV)
    with pytest.raises(SystemExit):
        local_probe.main(["--truncation-check"])
    assert made == []


# ---------------------------------------------------------------- local_probe --verify-server


class _Recording(OllamaOracleFakeClient):
    def __init__(self) -> None:
        super().__init__(week1_pool("dev"))
        self.chats = 0

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.chats += 1
        return super().chat(request)


def test_verify_server_checks_the_server_without_loading_the_model(monkeypatch: pytest.MonkeyPatch,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    inside_wrapper(monkeypatch)
    client = _Recording()
    monkeypatch.setattr(local_probe, "make_ollama_client", lambda url, mc=None: client)
    assert local_probe.main(["--verify-server"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["server"]["model_digest"] == FAKE_OLLAMA_DIGEST and "tools" in report["server"]["capabilities"]
    assert report["wrapper"] == SERVER_INFO
    assert report["config_id"] == local_agent_config().config_id and client.chats == 0


@pytest.mark.parametrize("where", ["outside", "no-report"])
def test_verify_server_needs_the_wrapper(where: str, monkeypatch: pytest.MonkeyPatch, made: list[str]) -> None:
    if where == "outside":
        outside_wrapper(monkeypatch)
    else:
        inside_wrapper(monkeypatch)
        monkeypatch.delenv(SERVER_INFO_ENV)
    with pytest.raises(SystemExit):
        local_probe.main(["--verify-server"])
    assert made == []


def test_the_frozen_config_check_still_applies(monkeypatch: pytest.MonkeyPatch, made: list[str]) -> None:
    inside_wrapper(monkeypatch)
    monkeypatch.setitem(ollama.LOCAL_MODEL_CONFIG, "model_digest", "sha256:" + "f" * 64)
    monkeypatch.setitem(ollama.LOCAL_MODEL_CONFIG, "ollama_version", "0.0.0-dry-run")
    with pytest.raises(SystemExit):
        local_probe.main(["--verify-server"])
