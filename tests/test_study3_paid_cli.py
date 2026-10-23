"""run_sweep 的 Study 3 付費（Haiku）入口（decision 036 §6、§7）：baseline 目錄核對之後，天花板依 036 §6 判定（沒有失敗的
source，而且 baseline 完整），不呼叫只接受 Study 2 的 ``study2_analysis.analyze_baseline_runs``。

不連網、不呼叫任何模型：baseline 是 oracle 假 client 的 dry-run 輸出（run.json 改成正式），總帳、runs/ 與兩個 RUNS_ROOT 一律在
tmp；make_anthropic_client 與 run_sweeps 換成記錄參數的替身；研究的開關只在測試中打開。eval instance 只經過程式，不輸出內容。
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import haiku_agent_config
from experiments import billing, run_sweep
from experiments import study3_analysis as s3
from experiments.run_baseline import OracleFakeClient, run_split
from tasks import suites
from tasks.generator_v3 import GENERATOR_VERSION
from tasks.suites import suite_pool

FAILING = {"gen3-verified-close-H1-s2": "V3:close-qa-broken", "gen3-epic-closeout-H1-s2": "E5:blocks-instead-of-dropped"}
AGENT_ERROR = "gen3-sync-repair-H1-s2"
SWEEP = ["--plan", "2,10,48,0.3", "--criterion", "bonferroni95", "--confirm-paid-run", "--max-cost-usd", "5", "--total-cost-usd", "40",
         "--run-name", "s1"]


def ZERO() -> float:  # noqa: N802
    return 0.0


class _BrokenClient(OracleFakeClient):
    """指定的 task：每個回覆的 stop_reason 都不是登記的值 → AdapterError → agent_error（沒有 trajectory）。"""

    def __init__(self, instances: tuple, broken: str) -> None:
        super().__init__(instances)
        self._broken = next(i.task.instruction for i in instances if i.task.task_id == broken)

    def create(self, **request: Any) -> Any:
        response = super().create(**request)
        return dataclasses.replace(response, stop_reason="max_tokens") if request["messages"][0]["content"] == self._broken else response


def _formal(run_dir: Path) -> Path:
    """dry-run 的輸出改成正式的 run.json（只為了測付費入口的檢查；不是研究結果）。"""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    (run_dir / "run.json").write_text(json.dumps({**run, "dry_run": False}), encoding="utf-8")
    return run_dir


@pytest.fixture(scope="module")
def failing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """seeds 2–3 這一段，FAILING 的題目失敗（有失敗的 source）。"""
    pool = tuple(dataclasses.replace(i, oracle_calls=dict(i.negative_calls)[FAILING[i.task.task_id]]) if i.task.task_id in FAILING else i
                 for i in suite_pool("study3", "eval"))
    run_dir = tmp_path_factory.mktemp("p3f") / "b"
    run_split("eval", OracleFakeClient(pool), run_dir=run_dir, clock=ZERO, suite="study3", agent_config=haiku_agent_config(),
              dry_run=True, rollouts=1, seeds=(2, 3))
    return _formal(run_dir)


@pytest.fixture(scope="module")
def ceiling(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """96 題全部有採計的 rollout：除了一個 agent_error 全部成功（沒有失敗的 source、baseline 完整 → 036 §6 的天花板）。"""
    pool = suite_pool("study3", "eval")
    assert [i.task.instruction for i in pool].count(next(i.task.instruction for i in pool if i.task.task_id == AGENT_ERROR)) == 1
    run_dir = tmp_path_factory.mktemp("p3c") / "b"
    run_split("eval", _BrokenClient(pool, AGENT_ERROR), run_dir=run_dir, clock=ZERO, suite="study3", agent_config=haiku_agent_config(),
              dry_run=True, rollouts=1)
    results = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["results"]
    assert {r["task_id"] for r in results if r["outcome"] != "success"} == {AGENT_ERROR}
    assert {r["outcome"] for r in results if r["task_id"] == AGENT_ERROR} == {"agent_error"}
    return _formal(run_dir)


@pytest.fixture
def paid_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """工作目錄、runs/、總帳與兩個 RUNS_ROOT 都在 tmp_path；開關打開；R 改成 1（測試的 baseline 只有一個 rollout）。"""
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(billing, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(billing, "STUDY3_JOURNAL", tmp_path / "journal" / "study3.jsonl")
    monkeypatch.setattr(s3, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(s3, "ROLLOUTS", 1)
    monkeypatch.setattr(suites, "STUDY3_APPROVED", True)
    monkeypatch.setattr(suites, "STUDY3_FROZEN", True)
    seen: dict = {"made": [], "calls": []}
    monkeypatch.setattr(run_sweep, "make_anthropic_client", lambda model_config=None: seen["made"].append(model_config["model"]) or object())

    def fake_run_sweeps(sources, client, **kwargs):
        seen["calls"].append(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(run_sweep, "run_sweeps", fake_run_sweeps)
    return seen


def _install(baseline: Path, tmp_path: Path) -> None:
    shutil.copytree(baseline, tmp_path / "runs" / "b")
    floor = billing.manifest_floor(haiku_agent_config().config_id, generator=GENERATOR_VERSION)
    for i, cost in enumerate([floor]):  # 總帳至少是 runs/ 中 Study 3 的 manifest 合計（否則以「總帳被重設」拒絕）
        billing._write(billing.STUDY3_JOURNAL, {"kind": "reserve", "id": f"r{i}", "utc_time": "t", "phase": "baseline", "run": "b",
                                                "bound_usd": cost})
        billing._write(billing.STUDY3_JOURNAL, {"kind": "settle", "id": f"r{i}", "utc_time": "t", "cost_usd": cost})


def test_a_study3_paid_sweep_with_failed_sources_reaches_the_sweep(failing: Path, tmp_path: Path, paid_cli: dict,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    _install(failing, tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_sweep.main(["--baseline-run", "runs/b", *SWEEP])
    err = capsys.readouterr().err
    assert exc.value.code == 0 and "not Study 2" not in err, err
    assert paid_cli["made"] == ["claude-haiku-5-5"] and paid_cli["calls"][0]["order"] == "frozen_permutation"


def test_the_036_ceiling_counts_sources_not_episodes(ceiling: Path, tmp_path: Path, paid_cli: dict,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    """agent_error 沒有 trajectory、不是 source：沒有失敗的 source 而且 baseline 完整 → 天花板（034 的「採計的 episode 全部成功」
    在這裡不成立，但不適用於 Study 3）。在建立 client 之前拒絕。"""
    _install(ceiling, tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_sweep.main(["--baseline-run", "runs/b", *SWEEP])
    err = capsys.readouterr().err
    assert exc.value.code != 0 and "ceiling, no sweep (decision 036 §6)" in err and "not Study 2" not in err
    assert paid_cli["made"] == [] and paid_cli["calls"] == []
