"""decision 028 的回歸測試（decision 027 審查確認的缺陷）。

- 推論路徑連續失敗（version 有回應、chat 一直失敗或逾時）→ OllamaServerUnavailable，run 停止、可補回。
- 模型自己把對話拉長而超出 fit check → agent_output（只算那個 rollout 失敗）；固定部分放不下 → harness。
- local_probe 的彙整涵蓋失敗的嘗試、標示 dry-run、能處理中斷的目錄。
- wrapper：本機通道的 port 被占用 → exit 75。

全部只用假的 client、127.0.0.1 上的測試 server 與 PATH 上的假 ssh，不連到任何外部服務。
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from agents.baseline import AdapterError, BASELINE_DEVELOPER_PROMPT, BASELINE_SYSTEM_PROMPT
from agents.ollama import (
    LOCAL_MODEL_CONFIG,
    OllamaContextError,
    OllamaHTTPClient,
    OllamaPolicy,
    OllamaServerUnavailable,
    OllamaTransportError,
    local_agent_config,
)
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from experiments.analyze_baseline import load_runs
from experiments.local_probe import summarize_smoke
from experiments.run_baseline import FAKE_OLLAMA_VERSION, OllamaOracleFakeClient, run_split
from replay.aborts import classify_exception, read_manifest
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
DEV = week1_pool("dev")
TOOLS = IssueTrackerEnv.tool_specs()
NUM_PREDICT = LOCAL_MODEL_CONFIG["options"]["num_predict"]


# ---------------------------------------------------------------- 028 §1：推論路徑連續失敗


class _Server(BaseHTTPRequestHandler):
    """/api/version 正常；/api/chat 依 mode：ok（oracle 回覆）、500、hang（不回應）、blip（第 1 次 500，之後 ok）。"""

    mode = "ok"
    oracle: OllamaOracleFakeClient
    chats = 0

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"version": FAKE_OLLAMA_VERSION})

    def do_POST(self) -> None:  # noqa: N802
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        type(self).chats += 1
        mode = type(self).mode
        if mode == "500" or (mode == "blip" and type(self).chats == 1):
            self._send(500, {"error": "llama runner process has terminated: CUDA error: out of memory"})
        elif mode == "hang":
            time.sleep(2.5)
        else:
            self._send(200, type(self).oracle.chat(request))

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server(request: pytest.FixtureRequest) -> Iterator[tuple[str, type[_Server]]]:
    handler = type("Handler", (_Server,), {"mode": "ok", "oracle": OllamaOracleFakeClient([*POOL, *DEV]), "chats": 0})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_client_counts_consecutive_chat_failures(server: tuple[str, type[_Server]]) -> None:
    url, handler = server
    client = OllamaHTTPClient(url, timeout_s=5)
    handler.mode = "500"
    for expected in (1, 2):
        with pytest.raises(OllamaTransportError):
            client.chat({"model": "m"})
        assert client.consecutive_chat_failures == expected
    client.version()
    assert client.consecutive_chat_failures == 2  # version 不影響
    handler.mode = "ok"
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": DEV[0].task.instruction}]
    client.chat({"model": LOCAL_MODEL_CONFIG["model"], "messages": messages})
    assert client.consecutive_chat_failures == 0


@pytest.mark.parametrize("mode", ["500", "hang"])
def test_a_broken_inference_path_stops_the_run_without_final_aborts(mode: str, server: tuple[str, type[_Server]],
                                                                     tmp_path: Path) -> None:
    url, handler = server
    handler.mode = mode
    run_dir = tmp_path / "eval"
    with pytest.raises(OllamaServerUnavailable) as info:
        run_split("eval", OllamaHTTPClient(url, timeout_s=1), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(),
                  rollouts=3, max_model_calls=200)
    assert classify_exception(info.value) == "harness"
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert [(row.attempt, row.error_class) for row in rows] == [(1, "infrastructure"), (2, "harness")]  # 只有第一個 rollout
    assert load_runs([run_dir]).records[0].outcome == "interrupted"  # 不是最終結果，可補回
    assert handler.chats == 2


def test_a_single_failure_is_retried_and_resets_the_count(server: tuple[str, type[_Server]], tmp_path: Path) -> None:
    url, handler = server
    handler.mode = "blip"
    summary = run_split("dev", OllamaHTTPClient(url, timeout_s=5), run_dir=tmp_path / "dev", clock=ZERO,
                        agent_config=local_agent_config())
    assert [result.outcome for result in summary.results] == ["success"] * len(DEV)
    assert summary.results[0].attempts == 2


# ---------------------------------------------------------------- 028 §2：context 預算


def _conversation(extra_turns: int, padding: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "developer", "content": BASELINE_DEVELOPER_PROMPT},
        {"role": "user", "content": POOL[0].task.instruction},
    ]
    for turn in range(extra_turns):
        text = "x" * padding
        raw = {"role": "assistant", "content": text, "tool_calls": [{"function": {"name": "get_issue", "arguments": {"issue_id": 1}}}]}
        messages.append({"role": "assistant", "content": text, "tool_calls": [{"id": f"call_{turn}_0", "name": "get_issue",
                                                                                "args": {"issue_id": 1}}], "provider_content": [raw]})
        messages.append({"role": "tool", "tool_call_id": f"call_{turn}_0", "content": canonical_json({"ok": True, "data": {}})})
    return messages


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def chat(self, request: Any) -> Any:
        self.requests.append(request)
        raise AssertionError("must not be called")

    def version(self) -> Any:
        return {"version": "x"}


def test_a_conversation_the_model_grew_past_the_budget_is_agent_output() -> None:
    client = _Recorder()
    with pytest.raises(AdapterError) as info:
        OllamaPolicy(client, local_agent_config()).next_message(_conversation(extra_turns=20, padding=4000), TOOLS)
    assert classify_exception(info.value) == "agent_output"
    assert not hasattr(info.value, "billed_usage") and client.requests == []  # 沒有送出任何 call：不計費


def test_a_fixed_part_that_does_not_fit_is_a_harness_error() -> None:
    messages = _conversation(extra_turns=1, padding=10)
    messages[2]["content"] = "y" * (2 * LOCAL_MODEL_CONFIG["options"]["num_ctx"])
    client = _Recorder()
    with pytest.raises(OllamaContextError):
        OllamaPolicy(client, local_agent_config()).next_message(messages, TOOLS)
    assert client.requests == []


def test_a_looping_rollout_fails_only_itself(tmp_path: Path) -> None:
    class Looper(OllamaOracleFakeClient):
        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["messages"][1]["content"] != DEV[0].task.instruction:
                return super().chat(request)
            reply = super().chat({**request, "messages": request["messages"][:2]})
            reply["message"] = {"role": "assistant", "content": "z" * 3000, "tool_calls": [
                {"function": {"name": "search_issues", "arguments": {}}}, {"function": {"name": "get_issue", "arguments": {"issue_id": 1}}},
            ]}
            return reply

    summary = run_split("dev", Looper(DEV), run_dir=tmp_path / "dev", clock=ZERO, agent_config=local_agent_config())
    assert summary.results[0].outcome == "agent_error"
    assert [result.outcome for result in summary.results[1:]] == ["success"] * (len(DEV) - 1)


# ---------------------------------------------------------------- 028 §3：local_probe 的完整性


class _Mixed(OllamaOracleFakeClient):
    """DEV[0]：第 2 次回覆達到 num_predict；DEV[1]：第 1 次回覆的 tool call 沒有名稱；DEV[2]：第一次嘗試的第 2 次 call 逾時（服務仍在）。"""

    def __init__(self) -> None:
        super().__init__(DEV)
        self.timed_out = False

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        instruction = request["messages"][1]["content"]
        t = sum(1 for message in request["messages"] if message["role"] == "assistant")
        if instruction == DEV[2].task.instruction and t == 1 and not self.timed_out:
            self.timed_out = True
            raise OllamaTransportError("timed out")
        reply = super().chat(request)
        reply["total_duration"] = 3_000_000_000
        if instruction == DEV[0].task.instruction and t == 1:
            reply.update(done_reason="length", eval_count=NUM_PREDICT, total_duration=95_000_000_000)
        if instruction == DEV[1].task.instruction and t == 0:
            reply["message"] = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "", "arguments": {}}}]}
        return reply


def test_the_summary_covers_failed_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "dev"
    summary = run_split("dev", _Mixed(), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config())
    report = summarize_smoke(run_dir)
    assert report["dry_run"] is False
    assert report["billed_calls"] == summary.model_calls == sum(row.billed_calls for row in read_manifest(run_dir / "manifest.jsonl"))
    assert report["calls"]["coverage"] == "completed trajectories only"
    assert report["calls"]["n"] + report["calls"]["excluded"] == report["billed_calls"]
    aborts = {(entry["error_class"], entry["message"]): entry["count"] for entry in report["aborts"]}
    assert aborts[("agent_output", "the reply reached num_predict (output truncated)")] == 1
    assert aborts[("agent_output", "a tool call has no name")] == 1
    assert aborts[("infrastructure", "timed out")] == 1
    assert report["outcomes"] == {"agent_error": 2, "success": 3}


def test_the_summary_marks_dry_runs_and_reads_interrupted_runs(tmp_path: Path) -> None:
    dry = tmp_path / "dry"
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=dry, clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    assert summarize_smoke(dry)["dry_run"] is True
    (dry / "summary.json").unlink()
    report = summarize_smoke(dry)
    assert report["interrupted"] is True and report["outcomes"] == {"success": 5} and report["episodes"] == 5


# ---------------------------------------------------------------- 028 §4：wrapper 的 port 被占用


ROOT = Path(__file__).resolve().parents[1]


def test_a_busy_local_tunnel_port_is_reported_as_busy(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/usr/bin/env bash\necho 'bind [127.0.0.1]:11435: Address already in use' >&2\n"
                   "echo 'channel_setup_fwd_listener_tcpip: cannot listen to port: 11435' >&2\n"
                   "echo 'Could not request local forwarding.' >&2\nexit 255\n", encoding="utf-8")
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k != "AI_PORTFOLIO_GPU_LOCK"}
    env.update(PATH=f"{bin_dir}{os.pathsep}{env['PATH']}", WITH_GPU_LOCK_TIMEOUT_S="5")
    marker = tmp_path / "ran"
    result = subprocess.run([str(ROOT / "tools" / "with-gpu-lock"), "--", "touch", str(marker)], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 75 and not marker.exists()
    assert "port" in result.stderr.lower()
