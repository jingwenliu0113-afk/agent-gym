"""experiments/local_probe.py 的契約測試（decision 026 §5 的凍結依據）。

- ``summarize_smoke``：彙整開發題測試的 run 目錄（不呼叫模型）：結果、每次呼叫的耗時與 token、tool call 的有效性與錯誤種類。
- ``truncation_check``：以很小的 num_ctx 送出超長 prompt，確認 server 在 ``truncate: false`` 時拒絕（而不是靜默截斷）。

全部只用假的 client，不連網、不使用 GPU。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import OllamaContextError, OllamaProtocolError, OllamaTransportError, local_agent_config
from experiments import local_probe
from experiments.local_probe import summarize_smoke, truncation_check
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from tasks.generator import week1_pool
from tests.wrapper_env import inside_wrapper, outside_wrapper

ZERO = lambda: 0.0  # noqa: E731
DEV = week1_pool("dev")


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("smoke") / "local-dev"
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config())
    return run_dir


def test_summarize_smoke_reports_outcomes_calls_and_tool_use(smoke_run: Path) -> None:
    report = summarize_smoke(smoke_run)
    assert report["split"] == "dev" and report["episodes"] == 5
    assert report["outcomes"] == {"success": 5} and report["success_rate"] == 1.0
    calls = report["calls"]
    expected_calls = sum(len(instance.oracle_calls) + 1 for instance in DEV)
    assert calls["n"] == expected_calls
    assert calls["prompt_eval_count"]["max"] == 1000 and calls["eval_count"]["median"] == 200
    assert calls["seconds"]["max"] == pytest.approx(1e-9)  # 假 client 的 total_duration = 1 ns
    tools = report["tool_calls"]
    assert tools["n"] == sum(len(instance.oracle_calls) for instance in DEV)
    assert tools["unknown_tool"] == 0 and tools["invalid_arguments"] == 0 and tools["multi_tool_turns"] == 0
    assert tools["error_codes"] == {}
    assert report["config_id"] == local_agent_config().config_id
    json.dumps(report)  # JSON-native


def test_summarize_smoke_counts_tool_errors(tmp_path: Path) -> None:
    class Sloppy(OllamaOracleFakeClient):
        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            t = sum(1 for message in request["messages"] if message["role"] == "assistant")
            first = {**request, "messages": request["messages"][:2]}  # 只用 instruction 取得回覆的骨架
            response = super().chat(first)
            if t == 0:  # 第一步：不存在的工具、參數是解析不了的字串、另一個 call 參數是陣列
                response["message"] = {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "teleport", "arguments": "{oops"}},
                    {"function": {"name": "get_issue", "arguments": [1]}},
                ]}
            else:
                response["message"] = {"role": "assistant", "content": "Done."}
            return response

    run_dir = tmp_path / "sloppy"
    run_split("dev", Sloppy(DEV), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config())
    tools = summarize_smoke(run_dir)["tool_calls"]
    assert tools["unknown_tool"] == 5 and tools["invalid_arguments"] == 5 and tools["multi_tool_turns"] == 5
    assert tools["error_codes"] == {"invalid_arguments": 5, "unknown_tool": 5}


def test_summarize_smoke_rejects_non_local_runs(tmp_path: Path) -> None:
    from experiments.run_baseline import OracleFakeClient

    run_dir = tmp_path / "anthropic"
    run_split("dev", OracleFakeClient(DEV), run_dir=run_dir, clock=ZERO)
    with pytest.raises(ValueError):
        summarize_smoke(run_dir)


class _Client:
    def __init__(self, outcome: Any) -> None:
        self.requests: list[dict[str, Any]] = []
        self._outcome = outcome

    def chat(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def test_truncation_check_sends_an_oversized_prompt_with_truncation_disabled() -> None:
    client = _Client(OllamaContextError("the input length exceeds the context length"))
    result = truncation_check(client, local_agent_config(), num_ctx=1024)
    request = client.requests[0]
    assert request["truncate"] is False and request["shift"] is False and request["options"]["num_ctx"] == 1024
    assert len(request["messages"][-1]["content"].encode()) > 4 * 1024 and "tools" not in request
    assert request["options"]["num_predict"] == 1
    assert result["result"] == "rejected" and result["num_ctx"] == 1024 and "context" in result["detail"]


def test_truncation_check_reports_a_silent_acceptance() -> None:
    reply = {"model": "qwen3.5:9b-q4_K_M", "message": {"role": "assistant", "content": "x"}, "done": True, "done_reason": "stop",
             "prompt_eval_count": 1000, "eval_count": 1}
    result = truncation_check(_Client(reply), local_agent_config(), num_ctx=1024)
    assert result["result"] == "accepted" and result["prompt_eval_count"] == 1000


@pytest.mark.parametrize("error", [OllamaTransportError("down"), OllamaProtocolError("bad")])
def test_truncation_check_lets_other_errors_through(error: Exception) -> None:
    with pytest.raises(type(error)):
        truncation_check(_Client(error), local_agent_config(), num_ctx=1024)


def test_the_cli_needs_the_gpu_lock_for_the_truncation_check(monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    outside_wrapper(monkeypatch)
    made: list[str] = []
    monkeypatch.setattr(local_probe, "make_ollama_client", lambda url, mc=None: made.append(url))
    with pytest.raises(SystemExit):
        local_probe.main(["--truncation-check"])
    assert made == []


def test_the_cli_runs_the_truncation_check_inside_the_lock(monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    class Rejecting(OllamaOracleFakeClient):  # server 檢查（version / tags / show）照常；chat 一律拒絕超長 prompt
        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            raise OllamaContextError("the input length exceeds the context length")

    inside_wrapper(monkeypatch)
    monkeypatch.setattr(local_probe, "make_ollama_client", lambda url, mc=None: Rejecting(DEV))
    assert local_probe.main(["--truncation-check"]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "rejected"


def test_the_cli_summarizes_a_run_without_the_lock(smoke_run: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    outside_wrapper(monkeypatch)
    assert local_probe.main(["--summarize", str(smoke_run)]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 5
