"""累計呼叫上限（decision 034 §8）：``call_log.count_calls`` 與 ``run_sweep --total-model-calls``。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import local_agent_config
from experiments import run_baseline, run_sweep
from experiments.call_log import CALL_LOG, count_calls
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from tasks.generator import week1_pool
from tests.wrapper_env import frozen_server, inside_wrapper

POOL = week1_pool("eval")
REGISTERED = ["--plan", "2,10,48,0.3", "--criterion", "bonferroni95"]


def ZERO() -> float:  # noqa: N802
    return 0.0


def test_count_calls(tmp_path: Path) -> None:
    for name, lines in (("a", 3), ("b", 0)):
        (tmp_path / name).mkdir()
        (tmp_path / name / CALL_LOG).write_text(
            "".join(json.dumps({"format": "agent-gym/call/v1", "call": i + 1, "ok": True}) + "\n" for i in range(lines)), encoding="utf-8")
    assert count_calls([tmp_path / "a", tmp_path / "b"]) == 3
    assert count_calls([]) == 0
    (tmp_path / "c").mkdir()
    with pytest.raises(ValueError, match=CALL_LOG):
        count_calls([tmp_path / "c"])


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def make(base_url: str, model_config: Any = None) -> OllamaOracleFakeClient:
        urls.append(base_url)
        return frozen_server(POOL)

    monkeypatch.setattr(run_baseline, "make_ollama_client", make)
    return urls


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("cap") / "eval"
    run_split("eval", frozen_server(POOL), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config())
    return run_dir


def _calls(run_dir: Path) -> int:
    return len((run_dir / CALL_LOG).read_text(encoding="utf-8").splitlines())


def test_the_cumulative_cap_limits_each_segment(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    assert run_sweep.main(["--baseline-run", str(baseline), "--run-name", "s1", *REGISTERED, "--confirm-local-run",
                           "--max-model-calls", "20", "--total-model-calls", "35"]) == 0
    first = _calls(tmp_path / "runs" / "s1")
    assert 20 <= first < 20 + 20  # 每次嘗試前檢查上限（一段最多多出一個 episode）
    assert run_sweep.main(["--baseline-run", str(baseline), "--run-name", "s2", *REGISTERED, "--confirm-local-run",
                           "--max-model-calls", "20", "--total-model-calls", "35", "--resume-from", str(tmp_path / "runs" / "s1")]) == 0
    second = _calls(tmp_path / "runs" / "s2")
    assert second >= 35 - first and second < 35 - first + 20
    with pytest.raises(SystemExit):  # 上限已用完：不再開始新的一段
        run_sweep.main(["--baseline-run", str(baseline), "--run-name", "s3", *REGISTERED, "--confirm-local-run",
                        "--max-model-calls", "20", "--total-model-calls", str(first + second), "--resume-from",
                        str(tmp_path / "runs" / "s1"), str(tmp_path / "runs" / "s2")])
    assert not (tmp_path / "runs" / "s3").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--dry-run", "--total-model-calls", "10"],
        ["--confirm-local-run", "--total-model-calls", "10"],  # 沒有每段的上限
    ],
    ids=["dry-run", "no-segment-cap"],
)
def test_total_cap_argument_checks(extra: list[str], baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):
        run_sweep.main(["--baseline-run", str(baseline), "--run-name", "x", *REGISTERED, *extra])
    assert made == []
