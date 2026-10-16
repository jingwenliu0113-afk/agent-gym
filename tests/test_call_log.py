"""decision 032：逐次模型呼叫的稽核紀錄（run 目錄的 calls.jsonl）。

每次 chat 一行：哪一個 task / rollout / attempt（sweep 另加 checkpoint k）、第幾個 assistant turn、這次呼叫的 seed、request 的
sha256、結果（成功時模型的回覆與計數；失敗時例外的種類與訊息）。中止與中斷的嘗試也在其中——trajectory 只保存完成的嘗試，
所以每個 seed 都可以稽核。全部只用假的 client。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import OllamaTransportError, local_agent_config
from env.hashing import canonical_json
from experiments import local_probe, run_baseline
from experiments.call_log import CALL_LOG, CALL_LOG_FORMAT, CallLog, read_call_log
from experiments.local_probe import summarize_smoke
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.aborts import read_manifest
from replay.localization import SweepPlan
from replay.trajectory import Trajectory
from tasks.generator import week1_pool
from tests.wrapper_env import frozen_server, inside_wrapper

ZERO = lambda: 0.0  # noqa: E731
DEV = week1_pool("dev")
EVAL = week1_pool("eval")


class _Fake:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.consecutive_chat_failures = 3

    def chat(self, request: dict[str, Any]) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def version(self) -> Any:
        return {"version": "x"}


REQUEST = {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                      {"role": "assistant", "content": "a"}, {"role": "tool", "content": "t"}],
           "options": {"seed": 1234}}
RESPONSE = {"model": "m", "message": {"role": "assistant", "content": "ok"}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 3, "total_duration": 99}


def test_a_call_is_logged_with_its_seed_context_and_result(tmp_path: Path) -> None:
    log = CallLog(_Fake(RESPONSE), tmp_path / CALL_LOG)
    client = log.bind(task_id="t1", rollout_index=2, attempt=1, k=None)
    assert client.chat(REQUEST) == RESPONSE
    [record] = read_call_log(tmp_path / CALL_LOG)
    assert record["format"] == CALL_LOG_FORMAT and record["call"] == 1 and record["ok"] is True
    assert record["task_id"] == "t1" and record["rollout_index"] == 2 and record["attempt"] == 1 and record["k"] is None
    assert record["seed"] == 1234 and record["assistant_turn"] == 1
    assert record["request_sha256"] == "sha256:" + __import__("hashlib").sha256(canonical_json(REQUEST).encode()).hexdigest()
    assert record["response"]["message"] == RESPONSE["message"] and record["response"]["done_reason"] == "stop"
    assert record["response"]["prompt_eval_count"] == 10 and isinstance(record["utc_time"], str) and record["seconds"] >= 0


def test_a_failed_call_is_logged_and_the_error_is_raised(tmp_path: Path) -> None:
    log = CallLog(_Fake(OllamaTransportError("boom")), tmp_path / CALL_LOG)
    with pytest.raises(OllamaTransportError):
        log.bind(task_id="t", rollout_index=0, attempt=1, k=3).chat(REQUEST)
    [record] = read_call_log(tmp_path / CALL_LOG)
    assert record["ok"] is False and record["error_type"] == "OllamaTransportError" and "boom" in record["message"]
    assert record["seed"] == 1234 and record["k"] == 3 and "response" not in record


def test_the_bound_client_behaves_like_the_client(tmp_path: Path) -> None:
    fake = _Fake(RESPONSE)
    client = CallLog(fake, tmp_path / CALL_LOG).bind(task_id="t", rollout_index=0, attempt=1, k=None)
    assert client.version() == {"version": "x"} and client.consecutive_chat_failures == 3  # decision 028 的探測照常
    fake.consecutive_chat_failures = 0
    assert client.consecutive_chat_failures == 0


def test_the_log_starts_empty_and_is_never_overwritten(tmp_path: Path) -> None:
    CallLog(_Fake(RESPONSE), tmp_path / CALL_LOG)
    assert (tmp_path / CALL_LOG).read_bytes() == b""
    with pytest.raises(FileExistsError):
        CallLog(_Fake(RESPONSE), tmp_path / CALL_LOG)


def _turn_seeds(run_dir: Path) -> list[tuple[str, int, int, int]]:
    seeds = []
    for path in sorted((run_dir / "trajectories").glob("*.json")):
        trajectory = Trajectory.load(path)
        rollout = int(path.stem.rsplit("-r", 1)[1])
        for index, turn in enumerate(trajectory.turns):
            seeds.append((trajectory.task["task_id"], rollout, index, turn.meta["seed"]))
    return seeds


def test_every_seed_of_a_baseline_run_is_in_the_log(tmp_path: Path) -> None:
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=tmp_path / "r", clock=ZERO, agent_config=local_agent_config(),
              rollouts=2, call_log=True)
    records = read_call_log(tmp_path / "r" / CALL_LOG)
    logged = sorted((r["task_id"], r["rollout_index"], r["assistant_turn"], r["seed"]) for r in records)
    assert logged == sorted(_turn_seeds(tmp_path / "r")) and all(r["ok"] for r in records)
    assert len(records) == sum(row.billed_calls for row in read_manifest(tmp_path / "r" / "manifest.jsonl"))
    assert [r["call"] for r in records] == list(range(1, len(records) + 1))


def test_calls_of_a_failed_attempt_are_logged_too(tmp_path: Path) -> None:
    class Flaky(OllamaOracleFakeClient):
        def __init__(self) -> None:
            super().__init__(DEV)
            self.failed = False

        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            if not self.failed and sum(1 for m in request["messages"] if m["role"] == "assistant") == 1:
                self.failed = True
                raise OllamaTransportError("connection reset")
            return super().chat(request)

    run_split("dev", Flaky(), run_dir=tmp_path / "r", clock=ZERO, agent_config=local_agent_config(), call_log=True)
    records = read_call_log(tmp_path / "r" / CALL_LOG)
    [failed] = [r for r in records if not r["ok"]]
    assert failed["error_type"] == "OllamaTransportError" and failed["attempt"] == 1 and isinstance(failed["seed"], int)
    retried = [r for r in records if r["task_id"] == failed["task_id"] and r["rollout_index"] == failed["rollout_index"]]
    assert {r["attempt"] for r in retried} == {1, 2}  # 第一次嘗試的呼叫（含成功的那一次）與重試都在
    first_attempt = [r for r in retried if r["attempt"] == 1]
    assert len(first_attempt) == 2 and first_attempt[0]["ok"] is True  # trajectory 不會保存的那一次成功呼叫


def test_a_sweep_logs_the_checkpoint_of_every_call(tmp_path: Path) -> None:
    baseline = tmp_path / "eval"
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=baseline, clock=ZERO, agent_config=local_agent_config())
    run_sweeps(select_sources(baseline), OllamaOracleFakeClient(EVAL), plan=SweepPlan(3, 2, 2, 0.3), criterion="wilson95",
               run_dir=tmp_path / "s", clock=ZERO, call_log=True)
    records = read_call_log(tmp_path / "s" / CALL_LOG)
    assert records and all(isinstance(r["k"], int) and r["k"] >= 0 for r in records)
    assert {r["task_id"] for r in records} == {instance.task.task_id for instance in EVAL}
    assert len(records) == sum(row.billed_calls for row in read_manifest(tmp_path / "s" / "manifest.jsonl"))
    for record in records:  # rerun 從 checkpoint k 開始：送出的 conversation 已有 k 個 assistant turn
        assert record["assistant_turn"] >= record["k"]


def test_local_runs_from_the_cli_keep_a_call_log_and_dry_runs_do_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_baseline, "make_ollama_client", lambda url, mc=None: frozen_server(DEV))
    inside_wrapper(monkeypatch)
    assert run_baseline.main(["--split", "dev", "--confirm-local-run", "--max-model-calls", "200", "--run-name", "local"]) == 0
    assert read_call_log(tmp_path / "runs" / "local" / CALL_LOG)
    assert run_baseline.main(["--split", "dev", "--dry-run", "--run-name", "dry"]) == 0
    assert not (tmp_path / "runs" / "dry" / CALL_LOG).exists()


def test_the_smoke_summary_reports_the_call_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=tmp_path / "r", clock=ZERO, agent_config=local_agent_config(),
              call_log=True)
    report = summarize_smoke(tmp_path / "r")
    episodes = report["episodes"]
    assert report["call_log"] == {"calls": report["billed_calls"], "failed": 0, "seeds": report["billed_calls"],
                                  "attempts": episodes, "matched_attempts": episodes, "interrupted_attempts": 0,
                                  "mismatched_attempts": 0, "complete": True}
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=tmp_path / "plain", clock=ZERO, agent_config=local_agent_config())
    assert summarize_smoke(tmp_path / "plain")["call_log"] is None
    assert local_probe.main(["--summarize", str(tmp_path / "r")]) == 0
    assert json.loads(capsys.readouterr().out)["call_log"]["complete"] is True


@pytest.mark.parametrize("line", [b"{not json}\n", b'{"format": "other"}\n', b"[]\n"], ids=["bad-json", "format", "not-object"])
def test_a_corrupt_call_log_is_an_error(line: bytes, tmp_path: Path) -> None:
    (tmp_path / CALL_LOG).write_bytes(line)
    with pytest.raises(ValueError):
        read_call_log(tmp_path / CALL_LOG)


def test_a_reply_with_an_unencodable_string_is_still_logged(tmp_path: Path) -> None:
    odd = {**RESPONSE, "message": {"role": "assistant", "content": "bad \ud800 text"}}  # 單獨的 surrogate（strict JSON 讀得出來）
    log = CallLog(_Fake(odd), tmp_path / CALL_LOG)
    assert log.bind(task_id="t", rollout_index=0, attempt=1, k=None).chat(REQUEST) == odd  # adapter 照常處理它
    [record] = read_call_log(tmp_path / CALL_LOG)
    assert record["seed"] == 1234 and record["call"] == 1 and record["ok"] is True


SLOW_CALL = """
import sys, time
sys.path.insert(0, {root!r})
from pathlib import Path
from experiments.call_log import CALL_LOG, CallLog, interrupt_on_termination

class Slow:
    def chat(self, request):
        time.sleep(60)

interrupt_on_termination()
log = CallLog(Slow(), Path({out!r}) / CALL_LOG)
print("ready", flush=True)
log.bind(task_id="t", rollout_index=0, attempt=1, k=None).chat({{"options": {{"seed": 777}}, "messages": []}})
"""


@pytest.mark.parametrize("signum", ["SIGTERM", "SIGHUP"])
def test_a_call_in_flight_when_the_wrapper_ends_the_run_is_logged(signum: str, tmp_path: Path) -> None:
    """tools/with-gpu-lock 以 SIGTERM 結束指令：正在進行的呼叫（與它的 seed）照樣記下。"""
    import signal
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen([sys.executable, "-c", SLOW_CALL.format(root=root, out=str(tmp_path))], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    import time

    time.sleep(0.5)
    proc.send_signal(getattr(signal, signum))
    proc.wait(timeout=30)
    [record] = read_call_log(tmp_path / CALL_LOG)
    assert record["ok"] is False and record["error_type"] == "KeyboardInterrupt" and record["seed"] == 777


def test_reconciliation_finds_a_missing_or_seedless_record(tmp_path: Path) -> None:
    from experiments.call_log import reconcile

    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=tmp_path / "r", clock=ZERO, agent_config=local_agent_config(),
              call_log=True)
    path = tmp_path / "r" / CALL_LOG
    assert reconcile([tmp_path / "r"])["complete"] is True
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[1:]))  # 少了一筆
    result = reconcile([tmp_path / "r"])
    assert result["complete"] is False and result["mismatched_attempts"] == 1
    record = json.loads(lines[0])
    record["seed"] = None
    path.write_bytes(canonical_json(record).encode() + b"\n" + b"".join(lines[1:]))  # 有紀錄但沒有 seed
    result = reconcile([tmp_path / "r"])
    assert result["complete"] is False and result["seeds"] == result["calls"] - 1


def test_an_interrupted_attempt_is_counted_as_interrupted(tmp_path: Path) -> None:
    from experiments.call_log import reconcile

    class Stops(OllamaOracleFakeClient):
        def __init__(self) -> None:
            super().__init__(DEV)
            self.n = 0

        def chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.n += 1
            if self.n == 5:
                raise KeyboardInterrupt
            return super().chat(request)

    with pytest.raises(KeyboardInterrupt):
        run_split("dev", Stops(), run_dir=tmp_path / "r", clock=ZERO, agent_config=local_agent_config(), call_log=True)
    result = reconcile([tmp_path / "r"])
    assert result["interrupted_attempts"] == 1 and result["mismatched_attempts"] == 0 and result["complete"] is True

