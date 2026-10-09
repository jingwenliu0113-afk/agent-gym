"""第十一輪對抗式審查（decision 020 的實作與 keyless Core 完整性）確認的缺陷之回歸測試（decision 021）。

- dry-run 與付費輸出無從區分：dry-run 預演擋下付費續跑並誘導合併假結果；baseline 的 dry-run 目錄可能成為付費 sweep 的 source。
- 沒有完成任何 sweep 的祖先目錄可以被漏掉：它已計費的成本與中止嘗試從報告中消失。

「付費」目錄一律以假的 client 模擬（monkeypatch ``make_anthropic_client``）；不連網、不讀取任何 credential。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from agents.baseline import baseline_agent_config
from env.hashing import canonical_json
from experiments import run_baseline, run_sweep
from experiments.analyze_baseline import ANALYSIS_FORMAT, analyze_runs, load_runs, render_markdown
from experiments.analyze_sweep import SWEEP_ANALYSIS_FORMAT, analyze_sweep_runs
from experiments.analyze_sweep import render_markdown as render_sweep
from experiments.run_baseline import SUMMARY_FORMAT, OracleFakeClient, run_split
from experiments.run_sweep import SWEEP_SUMMARY_FORMAT, run_sweeps, select_sources
from replay.aborts import read_manifest
from replay.localization import SweepPlan
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
EVAL_IDS = [instance.task.task_id for instance in POOL]
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)
PLAN_TEXT = "3,2,2,0.3"


class _Explode:
    def __init__(self) -> None:
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        raise KeyboardInterrupt


@pytest.fixture(scope="module")
def baselines(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("baselines")
    dry, paid = root / "dry", root / "paid"
    run_split("eval", OracleFakeClient(POOL), run_dir=dry, clock=ZERO, dry_run=True)  # 較早的 dry-run 預演
    run_split("eval", OracleFakeClient(POOL), run_dir=paid, clock=ZERO)  # 之後的「付費」run（假 client 模擬）
    return {"dry": dry, "paid": paid}


def _run_json(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------- 021 §1：baseline 的 dry-run 標記


def test_format_versions() -> None:
    assert SUMMARY_FORMAT == "agent-gym/baseline-run/v2"
    assert ANALYSIS_FORMAT == "agent-gym/baseline-analysis/v3"
    assert SWEEP_SUMMARY_FORMAT == "agent-gym/sweep-run/v7"  # v7：decision 023
    assert SWEEP_ANALYSIS_FORMAT == "agent-gym/sweep-analysis/v5"  # v5：decision 025


@pytest.mark.parametrize("dry_run", [True, False])
def test_run_split_writes_run_json_before_the_first_attempt(dry_run: bool, tmp_path: Path) -> None:
    run_dir = tmp_path / "eval"
    with pytest.raises(KeyboardInterrupt):
        run_split("eval", _Explode(), run_dir=run_dir, rollouts=3, clock=ZERO, dry_run=dry_run)
    assert not (run_dir / "manifest.jsonl").exists()
    assert _run_json(run_dir) == {
        "format": SUMMARY_FORMAT,
        "split": "eval",
        "rollouts": 3,
        "config_id": baseline_agent_config().config_id,
        "dry_run": dry_run,
    }


def test_run_baseline_cli_marks_dry_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_baseline.main(["--split", "eval", "--dry-run", "--run-name", "rehearsal"]) == 0
    assert _run_json(tmp_path / "runs" / "rehearsal")["dry_run"] is True
    monkeypatch.setattr(run_baseline, "make_anthropic_client", lambda: OracleFakeClient(POOL))
    assert run_baseline.main(["--split", "eval", "--rollouts", "3", "--confirm-paid-run", "--max-cost-usd", "5",
                              "--run-name", "real"]) == 0  # 付費 eval 的 R = 3（decision 025 §2）
    assert _run_json(tmp_path / "runs" / "real")["dry_run"] is False


def test_dry_run_and_paid_baselines_cannot_be_merged(baselines: dict[str, Path]) -> None:
    for dirs in ([baselines["dry"], baselines["paid"]], [baselines["paid"], baselines["dry"]]):
        with pytest.raises(ValueError):
            load_runs(dirs)
        with pytest.raises(ValueError):
            select_sources(dirs)  # 否則較早的 dry-run 會成為「最早結束」的 rollout 0
        with pytest.raises(ValueError):
            analyze_runs(dirs, replicates=20)


def test_baseline_analysis_reports_dry_runs(baselines: dict[str, Path]) -> None:
    dry = analyze_runs([baselines["dry"]], replicates=20)
    paid = analyze_runs([baselines["paid"]], replicates=20)
    assert dry.dry_run is True and paid.dry_run is False
    assert dry.to_dict()["dry_run"] is True
    assert "dry-run" in render_markdown(dry).split("\n## ")[0] and "dry-run" not in render_markdown(paid).split("\n## ")[0]
    assert load_runs([baselines["dry"]]).dry_run is True


def _edit_run_json(run_dir: Path, edit: Callable[[dict[str, Any]], Any]) -> None:
    data = _run_json(run_dir)
    edit(data)
    (run_dir / "run.json").write_text(canonical_json(data) + "\n", encoding="utf-8")


BASELINE_RUN_JSON_CHANGES: dict[str, Callable[[Path], Any]] = {
    "missing": lambda d: (d / "run.json").unlink(),
    "format": lambda d: _edit_run_json(d, lambda x: x.update(format="agent-gym/baseline-run/v1")),
    "extra-key": lambda d: _edit_run_json(d, lambda x: x.update(extra=1)),
    "split-dev": lambda d: _edit_run_json(d, lambda x: x.update(split="dev")),
    "rollouts-mismatch": lambda d: _edit_run_json(d, lambda x: x.update(rollouts=2)),
    "rollouts-zero": lambda d: _edit_run_json(d, lambda x: x.update(rollouts=0)),
    "config": lambda d: _edit_run_json(d, lambda x: x.update(config_id="f" * 64)),
    "dry-run-int": lambda d: _edit_run_json(d, lambda x: x.update(dry_run=0)),
    "not-an-object": lambda d: (d / "run.json").write_text("[]\n", encoding="utf-8"),
}


@pytest.mark.parametrize("change", sorted(BASELINE_RUN_JSON_CHANGES))
def test_load_runs_requires_a_matching_run_json(change: str, baselines: dict[str, Path], tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(baselines["paid"], copy)
    load_runs([copy])  # 前提：未修改時可讀
    BASELINE_RUN_JSON_CHANGES[change](copy)
    with pytest.raises(ValueError):
        load_runs([copy])


def test_an_interrupted_baseline_without_summary_still_needs_run_json(baselines: dict[str, Path], tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(baselines["paid"], copy)
    (copy / "summary.json").unlink()
    load_runs([copy])
    (copy / "run.json").unlink()
    with pytest.raises(ValueError):
        load_runs([copy])


# ---------------------------------------------------------------- 021 §1：sweep 的 dry-run 標記


def _sweep_cli(baseline: Path, *extra: str) -> int:
    return run_sweep.main(["--baseline-run", str(baseline), "--plan", PLAN_TEXT, "--criterion", "wilson95", *extra])


@pytest.fixture
def fake_paid(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """以 OracleFakeClient 模擬付費 client；回傳的 list 記錄被建立的次數。這些測試的對象是 dry-run / 付費的區分（decision 021），
    為了速度用小的 plan：把登記的付費設定（decision 025 §3）暫時換成這裡的 PLAN / wilson95。"""
    made: list[int] = []
    monkeypatch.setattr(run_sweep, "REGISTERED_PLAN", PLAN)
    monkeypatch.setattr(run_sweep, "REGISTERED_CRITERION", "wilson95")

    def make() -> OracleFakeClient:
        made.append(1)
        return OracleFakeClient(POOL)

    monkeypatch.setattr(run_sweep, "make_anthropic_client", make)
    return made


def test_run_sweeps_records_the_mode(baselines: dict[str, Path], tmp_path: Path) -> None:
    sources = select_sources(baselines["paid"])
    run_sweeps(sources, OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "dry", clock=ZERO,
               replicates=20, dry_run=True)
    run_sweeps(sources, OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "paid", clock=ZERO,
               replicates=20)
    assert _run_json(tmp_path / "dry")["dry_run"] is True and _run_json(tmp_path / "paid")["dry_run"] is False


def test_a_paid_sweep_refuses_a_dry_run_baseline(baselines: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 fake_paid: list[int]) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _sweep_cli(baselines["dry"], "--confirm-paid-run", "--max-cost-usd", "5", "--run-name", "s")
    assert fake_paid == [] and not (tmp_path / "runs" / "s").exists()
    assert _sweep_cli(baselines["dry"], "--dry-run", "--run-name", "rehearsal", "--max-cost-usd", "0.1") == 0  # dry-run 可以
    assert _run_json(tmp_path / "runs" / "rehearsal")["dry_run"] is True


def test_a_dry_run_rehearsal_neither_blocks_nor_joins_the_paid_resume(baselines: dict[str, Path], tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch, fake_paid: list[int]) -> None:
    """審查情境：付費 s1 因預算停止 → 想以 --dry-run 預演續跑 → 之後的付費續跑。"""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    paid = baselines["paid"]
    assert _sweep_cli(paid, "--confirm-paid-run", "--max-cost-usd", "0.3", "--run-name", "s1") == 0
    with pytest.raises(SystemExit):  # dry-run 不能接付費目錄（預演要用 dry-run 鏈）
        _sweep_cli(paid, "--dry-run", "--resume-from", str(runs / "s1"))
    assert not (runs / "dry-run-sweep").exists()
    # 即使以 Python 直接建出一個從 s1 續跑的 dry-run 目錄，它也不會擋下付費續跑
    s1_id = _run_json(runs / "s1")["run_id"]
    run_sweeps(select_sources(paid), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=runs / "rehearsal",
               clock=ZERO, replicates=20, dry_run=True, resumed_from=[s1_id], skip=sorted(p.stem for p in (runs / "s1" / "sweeps").glob("*.json")),
               max_cost_usd=0.3)
    assert _sweep_cli(paid, "--confirm-paid-run", "--max-cost-usd", "50", "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    analysis = analyze_sweep_runs([runs / "s1", runs / "s2"], paid, replicates=20)
    assert len(analysis.curves) == 15 and analysis.dry_run is False
    with pytest.raises(ValueError):
        analyze_sweep_runs([runs / "s1", runs / "rehearsal", runs / "s2"], paid, replicates=20)


def test_a_paid_resume_refuses_dry_run_directories(baselines: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   fake_paid: list[int]) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _sweep_cli(baselines["paid"], "--dry-run", "--max-cost-usd", "0.3", "--run-name", "d1") == 0
    with pytest.raises(SystemExit):
        _sweep_cli(baselines["paid"], "--confirm-paid-run", "--max-cost-usd", "5", "--run-name", "p2", "--resume-from", str(runs / "d1"))
    assert fake_paid == [] and not (runs / "p2").exists()


def test_sweep_analysis_never_mixes_modes(baselines: dict[str, Path], tmp_path: Path) -> None:
    sources = select_sources(baselines["paid"])
    dry_sources = select_sources(baselines["dry"])
    half = EVAL_IDS[:7]
    run_sweeps(sources, OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "paid", clock=ZERO, replicates=20,
               skip=[t for t in EVAL_IDS if t not in half])
    run_sweeps(sources, OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "dry", clock=ZERO, replicates=20,
               skip=half, dry_run=True)
    with pytest.raises(ValueError):
        analyze_sweep_runs([tmp_path / "paid", tmp_path / "dry"], baselines["paid"], replicates=20)
    # 付費 sweep 不能以 dry-run baseline 分析（source 的 trajectory 相同也一樣）
    assert {t: s.trajectory_id for t, s in dry_sources.items() if s} == {t: s.trajectory_id for t, s in sources.items() if s}
    with pytest.raises(ValueError):
        analyze_sweep_runs([tmp_path / "paid"], baselines["dry"], replicates=20)
    # dry-run sweep 可以用任一種 baseline，報告標示 dry-run
    analysis = analyze_sweep_runs([tmp_path / "dry"], baselines["paid"], replicates=20)
    assert analysis.dry_run is True and analysis.to_dict()["dry_run"] is True
    assert "dry-run" in render_sweep(analysis).split("\n## ")[0]
    assert analyze_sweep_runs([tmp_path / "paid"], baselines["paid"], replicates=20).dry_run is False


# ---------------------------------------------------------------- 021 §2：祖先目錄必須齊全


def test_an_ancestor_that_completed_nothing_cannot_be_left_out(baselines: dict[str, Path], tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    paid = baselines["paid"]
    assert _sweep_cli(paid, "--dry-run", "--max-cost-usd", "0.05", "--run-name", "s1") == 0
    assert read_manifest(runs / "s1" / "manifest.jsonl") and not list((runs / "s1" / "sweeps").glob("*.json"))
    assert _sweep_cli(paid, "--dry-run", "--max-cost-usd", "0.3", "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        _sweep_cli(paid, "--dry-run", "--run-name", "s3", "--resume-from", str(runs / "s2"))
    err = capsys.readouterr().err
    assert _run_json(runs / "s1")["run_id"] in err and "s1" in err
    assert not (runs / "s3").exists()
    assert _sweep_cli(paid, "--dry-run", "--run-name", "s3", "--resume-from", str(runs / "s1"), str(runs / "s2")) == 0
    with pytest.raises(ValueError):
        analyze_sweep_runs([runs / "s2", runs / "s3"], paid, replicates=20)
    with pytest.raises(ValueError):
        analyze_sweep_runs([runs / "s3"], paid, replicates=20)
    full = analyze_sweep_runs([runs / "s1", runs / "s2", runs / "s3"], paid, replicates=20)
    every_row = [row for name in ("s1", "s2", "s3") for row in read_manifest(runs / name / "manifest.jsonl")]
    assert len(full.curves) == 15 and full.cost_usd == pytest.approx(sum(row.cost_usd for row in every_row))
