"""decision 027 的回歸測試（decision 026 審查確認的缺陷）。

- 遠端斷線不得變成 rollout 的最終結果：服務不可用 → OllamaServerUnavailable（harness），run 停止、可補回。
- context 防護在 done_reason length 之前；length 只有在 eval_count >= num_predict 時才是輸出截斷。
- 無法記錄的 tool call 參數：字串交給 env（invalid_arguments），物件或其他內容 → AdapterError（agent_output）。

全部只用假的 client 與 127.0.0.1 上的測試 server，不連到任何外部服務。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from agents.baseline import AdapterError
from agents.ollama import (
    LOCAL_MODEL_CONFIG,
    OllamaContextError,
    OllamaHTTPClient,
    OllamaPolicy,
    OllamaProtocolError,
    OllamaServerUnavailable,
    OllamaTransportError,
    local_agent_config,
    parse_chat_response,
)
from env.environment import IssueTrackerEnv
from experiments import run_sweep
from experiments.analyze_baseline import load_runs
from experiments.analyze_sweep import analyze_sweep_runs
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.aborts import classify_exception, read_manifest
from replay.localization import SweepPlan
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
DEV = week1_pool("dev")
MODEL = LOCAL_MODEL_CONFIG["model"]
NUM_CTX = LOCAL_MODEL_CONFIG["options"]["num_ctx"]
NUM_PREDICT = LOCAL_MODEL_CONFIG["options"]["num_predict"]
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)


class _Outage(OllamaOracleFakeClient):
    """前 ``healthy`` 次 chat 正常，之後 chat 與 version 都連不上（通道或 Ollama 中斷）。"""

    def __init__(self, instances: Any, healthy: int) -> None:
        super().__init__(instances)
        self._healthy = healthy
        self.chats = 0

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.chats += 1
        if self.chats > self._healthy:
            raise OllamaTransportError("connection refused")
        return super().chat(request)

    def version(self) -> dict[str, Any]:
        if self.chats > self._healthy:
            raise OllamaTransportError("connection refused")
        return super().version()


class _Blip(OllamaOracleFakeClient):
    """第 ``n`` 次 chat 失敗一次（服務仍在），其餘正常。"""

    def __init__(self, instances: Any, n: int) -> None:
        super().__init__(instances)
        self._n = n
        self.chats = 0

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.chats += 1
        if self.chats == self._n:
            raise OllamaTransportError("connection reset")
        return super().chat(request)


def _conversation() -> list[dict[str, Any]]:
    from agents.baseline import BASELINE_DEVELOPER_PROMPT, BASELINE_SYSTEM_PROMPT

    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "developer", "content": BASELINE_DEVELOPER_PROMPT},
        {"role": "user", "content": POOL[0].task.instruction},
    ]


# ---------------------------------------------------------------- 027 §1：服務不可用


def test_a_failed_call_with_an_unreachable_server_is_server_unavailable() -> None:
    client = _Outage(POOL, healthy=0)
    with pytest.raises(OllamaServerUnavailable) as info:
        OllamaPolicy(client, local_agent_config()).next_message(_conversation(), IssueTrackerEnv.tool_specs())
    assert classify_exception(info.value) == "harness"


def test_a_failed_call_with_a_live_server_is_still_infrastructure() -> None:
    client = _Blip(POOL, n=1)
    with pytest.raises(OllamaTransportError) as info:
        OllamaPolicy(client, local_agent_config()).next_message(_conversation(), IssueTrackerEnv.tool_specs())
    assert not isinstance(info.value, OllamaServerUnavailable) and classify_exception(info.value) == "infrastructure"


def test_a_transient_failure_is_retried_and_the_run_completes(tmp_path: Path) -> None:
    summary = run_split("eval", _Blip(POOL, n=5), run_dir=tmp_path / "e", clock=ZERO, agent_config=local_agent_config())
    assert all(result.outcome == "success" for result in summary.results) and len(summary.results) == 15
    assert max(result.attempts for result in summary.results) == 2


def test_an_outage_stops_the_baseline_without_aborting_the_remaining_rollouts(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval-a"
    with pytest.raises(OllamaServerUnavailable):
        run_split("eval", _Outage(POOL, healthy=20), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config())
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert not any(row.outcome == "aborted" and row.error_class == "infrastructure" for row in rows)  # 沒有任何 aborted 的最終結果
    assert rows[-1].error_class == "harness" and not (run_dir / "summary.json").exists()
    records = load_runs([run_dir]).records
    assert records[-1].outcome == "interrupted"
    assert {record.outcome for record in records} <= {"success", "interrupted"}
    rerun = tmp_path / "eval-b"
    run_split("eval", OllamaOracleFakeClient(POOL), run_dir=rerun, clock=ZERO, agent_config=local_agent_config())
    sources = select_sources([run_dir, rerun])
    assert all(source is not None for source in sources.values())  # 服務恢復後補回：每個 task 都有 source


def test_an_outage_mid_sweep_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "eval"
    run_split("eval", OllamaOracleFakeClient(POOL), run_dir=baseline, clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    reference = run_sweeps(select_sources(baseline), OllamaOracleFakeClient(POOL), plan=PLAN, criterion="wilson95",
                           run_dir=tmp_path / "reference", clock=ZERO, replicates=20, dry_run=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_sweep, "OllamaOracleFakeClient", lambda pool: _Outage(pool, healthy=40))
    argv = ["--baseline-run", str(baseline), "--plan", "3,2,2,0.3", "--criterion", "wilson95", "--dry-run"]
    with pytest.raises(OllamaServerUnavailable):
        run_sweep.main([*argv, "--run-name", "s1"])
    monkeypatch.setattr(run_sweep, "OllamaOracleFakeClient", OllamaOracleFakeClient)
    rows = read_manifest(tmp_path / "runs" / "s1" / "manifest.jsonl")
    assert not any(row.outcome == "aborted" and row.error_class == "infrastructure" for row in rows)
    assert run_sweep.main([*argv, "--run-name", "s2", "--resume-from", str(tmp_path / "runs" / "s1")]) == 0
    analysis = analyze_sweep_runs([tmp_path / "runs" / "s1", tmp_path / "runs" / "s2"], baseline, replicates=20)
    assert {curve.task_id: curve.result for curve in analysis.curves} == reference.results  # 與沒有中斷的排程相同


# ---------------------------------------------------------------- 027 §2：context 防護在 length 之前


def _response(**changes: Any) -> dict[str, Any]:
    response = {
        "model": MODEL,
        "message": {"role": "assistant", "content": "partial"},
        "done": True,
        "done_reason": "length",
        "prompt_eval_count": 900,
        "eval_count": NUM_PREDICT,
    }
    response.update(changes)
    return response


def _parse(response: dict[str, Any]) -> Any:
    return parse_chat_response(response, local_agent_config().model_config, assistant_turn=0, seed=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_eval_count": NUM_CTX - 100, "eval_count": 100},
        {"prompt_eval_count": None},
        {"prompt_eval_count": 0},
        {"eval_count": NUM_PREDICT - 1},
    ],
    ids=["at-num-ctx", "count-missing", "count-zero", "length-below-num-predict"],
)
def test_length_replies_go_through_the_context_guards(changes: dict[str, Any]) -> None:
    response = _response(**changes)
    if response["prompt_eval_count"] is None:
        del response["prompt_eval_count"]
    with pytest.raises(OllamaContextError) as info:
        _parse(response)
    assert classify_exception(info.value) == "harness"


def test_length_at_num_predict_is_output_truncation() -> None:
    with pytest.raises(AdapterError) as info:
        _parse(_response())
    assert classify_exception(info.value) == "agent_output" and info.value.billed_usage["cost_usd"] == 0.0  # type: ignore[attr-defined]


# ---------------------------------------------------------------- 027 §3：無法記錄的參數


def _deep(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = {"a": value}
    return value


def _call(arguments: Any) -> dict[str, Any]:
    return _response(done_reason="stop", eval_count=10, message={
        "role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_issue", "arguments": arguments}}],
    })


@pytest.mark.parametrize("depth", [65, 70, 130, 400])
def test_unrecordable_object_arguments_are_agent_output(depth: int) -> None:
    with pytest.raises(AdapterError) as info:
        _parse(_call(_deep(depth)))
    assert classify_exception(info.value) == "agent_output" and info.value.billed_usage["cost_usd"] == 0.0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "arguments",
    ['{"limit": NaN}', '{"limit": Infinity}', '{"limit": 1e400}', '{"a": 1, "a": 2}', json.dumps(_deep(70)), "[" * 100000],
    ids=["nan", "infinity", "overflow", "duplicate-key", "deep-json-string", "very-deep-string"],
)
def test_unparseable_string_arguments_go_to_the_env_as_strings(arguments: str) -> None:
    step = _parse(_call(arguments))
    assert step.message.tool_calls[0].args == arguments


def test_unencodable_text_is_agent_output() -> None:
    with pytest.raises(AdapterError):
        _parse(_response(done_reason="stop", eval_count=10, message={"role": "assistant", "content": "bad \ud800 text"}))


def test_a_bad_call_fails_only_its_rollout(tmp_path: Path) -> None:
    class Deep(OllamaOracleFakeClient):
        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            response = super().chat({**request, "messages": request["messages"][:2]})
            if request["messages"][1]["content"] == DEV[0].task.instruction:
                response["message"] = {"role": "assistant", "content": "",
                                       "tool_calls": [{"function": {"name": "get_issue", "arguments": _deep(70)}}]}
                return response
            return super().chat(request)

    summary = run_split("dev", Deep(DEV), run_dir=tmp_path / "d", clock=ZERO, agent_config=local_agent_config())
    outcomes = [result.outcome for result in summary.results]
    assert outcomes[0] == "agent_error" and outcomes[1:] == ["success"] * (len(DEV) - 1)


class _DeepBody(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = ("[" * 20000 + "]" * 20000).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def deep_server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _DeepBody)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_deeply_nested_body_is_a_protocol_error(deep_server: str) -> None:
    with pytest.raises(OllamaProtocolError):
        OllamaHTTPClient(deep_server, timeout_s=5).chat({"model": MODEL})
