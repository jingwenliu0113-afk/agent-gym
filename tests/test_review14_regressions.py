"""decision 023 的審查確認的缺陷之回歸測試（decision 024）。

- 完全靠接續完成的 sweep 沒有 manifest：分析與續跑都拒絕它，那條付費曲線永遠進不了報告。
- summary.json / sweeps/<task>.json 不是 atomic 寫入：磁碟已滿時留下截斷的檔案，目錄因此無法讀取。

全部只用假的 client，不連網。
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import pytest

from experiments import run_sweep
from experiments.analyze_baseline import load_runs
from experiments.analyze_sweep import analyze_sweep_runs
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import main, run_sweeps, select_sources
from replay import atomic
from replay.aborts import read_manifest
from replay.localization import SweepPlan
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
EVAL_IDS = [instance.task.task_id for instance in POOL]
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)
PLAN_TEXT = "3,2,2,0.3"
REAL_REPLACE = os.replace


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("baseline") / "eval"
    run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO, dry_run=True)
    return run_dir


@pytest.fixture(scope="module")
def reference(baseline: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    return run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95",
                      run_dir=tmp_path_factory.mktemp("reference") / "sweep", clock=ZERO, replicates=20, dry_run=True)


def _cli(baseline: Path, *extra: str) -> int:
    return main(["--baseline-run", str(baseline), "--plan", PLAN_TEXT, "--criterion", "wilson95", "--dry-run", *extra])


def _assert_matches_reference(analysis: Any, reference: Any) -> None:
    assert [curve.task_id for curve in analysis.curves] == EVAL_IDS
    for curve in analysis.curves:
        assert curve.result == reference.results[curve.task_id], curve.task_id


class _Explode:
    def __init__(self) -> None:
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        raise KeyboardInterrupt


# ---------------------------------------------------------------- 024 §1 / §2：sweep 目錄一定有 manifest


def test_run_sweeps_creates_an_empty_manifest_before_the_first_attempt(baseline: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "s"
    with pytest.raises(KeyboardInterrupt):
        run_sweeps(select_sources(baseline), _Explode(), plan=PLAN, criterion="wilson95", run_dir=run_dir, clock=ZERO, dry_run=True)
    assert (run_dir / "manifest.jsonl").read_bytes() == b"" and (run_dir / "run.json").exists()
    analysis = analyze_sweep_runs([run_dir], baseline, replicates=20)  # 沒有任何列：可以分析，什麼都沒有
    assert analysis.curves == () and analysis.cost_usd == 0 and analysis.interrupted_runs == ("s",)


def test_a_sweep_completed_only_from_carried_rollouts_is_analyzable(baseline: Path, reference: Any, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """審查情境：中斷落在最後一個 source 的最後一個 rollout 記帳之後、它的 sweeps 檔寫入之前。"""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    original = run_sweep.sweep

    def interrupted_after_the_last_source(source: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(source, *args, **kwargs)
        if result.task_id == EVAL_IDS[-1]:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(run_sweep, "sweep", interrupted_after_the_last_source)
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s1")
    monkeypatch.setattr(run_sweep, "sweep", original)
    assert not (runs / "s1" / "sweeps" / f"{EVAL_IDS[-1]}.json").exists()
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    assert read_manifest(runs / "s2" / "manifest.jsonl") == ()  # 沒有任何新的 call，但 manifest 存在
    assert (runs / "s2" / "sweeps" / f"{EVAL_IDS[-1]}.json").exists()
    _assert_matches_reference(analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20), reference)
    capsys.readouterr()
    with pytest.raises(SystemExit):  # s2 完成了一個 sweep：漏給它的續跑必須被拒絕（024 §2）
        _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1"))
    assert "s2" in capsys.readouterr().err and not (runs / "s3").exists()
    assert _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1"), str(runs / "s2")) == 0
    assert not (runs / "s3").exists()  # nothing left to run


def test_a_later_directory_with_only_an_empty_manifest_does_not_block(baseline: Path, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    monkeypatch.setattr(run_sweep, "OracleFakeClient", lambda pool: _Explode())
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1"))
    monkeypatch.setattr(run_sweep, "OracleFakeClient", OracleFakeClient)
    assert read_manifest(runs / "s2" / "manifest.jsonl") == () and not list((runs / "s2" / "sweeps").glob("*.json"))
    assert _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1")) == 0


# ---------------------------------------------------------------- 024 §3 / §4：atomic 寫入、錯誤訊息含路徑


def _disk_full_on(monkeypatch: pytest.MonkeyPatch, match: Any) -> None:
    def replace(src: Any, dst: Any) -> None:
        if match(Path(dst)):
            raise OSError(errno.ENOSPC, "No space left on device")
        REAL_REPLACE(src, dst)

    monkeypatch.setattr(atomic.os, "replace", replace)


def _leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.rglob(".*.tmp"))


def test_a_failed_baseline_summary_write_leaves_an_interrupted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "eval"
    _disk_full_on(monkeypatch, lambda path: path.name == "summary.json")
    with pytest.raises(OSError):
        run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO)
    monkeypatch.undo()
    assert not (run_dir / "summary.json").exists() and _leftovers(run_dir) == []
    loaded = load_runs([run_dir])
    assert loaded.interrupted_runs == ("eval",) and len(loaded.records) == 15


def test_a_failed_sweep_summary_write_leaves_an_interrupted_run(baseline: Path, reference: Any, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "s"
    _disk_full_on(monkeypatch, lambda path: path.name == "summary.json")
    with pytest.raises(OSError):
        run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=run_dir, clock=ZERO,
                   replicates=20, dry_run=True)
    monkeypatch.undo()
    assert not (run_dir / "summary.json").exists() and _leftovers(run_dir) == []
    _assert_matches_reference(analyze_sweep_runs([run_dir], baseline, replicates=20), reference)


def test_a_failed_sweep_file_write_is_carried_over_on_resume(baseline: Path, reference: Any, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _disk_full_on(monkeypatch, lambda path: path.parent.name == "sweeps" and path.stem == EVAL_IDS[1])
    with pytest.raises(OSError):
        _cli(baseline, "--run-name", "s1")
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    assert not (runs / "s1" / "sweeps" / f"{EVAL_IDS[1]}.json").exists() and _leftovers(runs / "s1") == []
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    _assert_matches_reference(analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20), reference)


def test_run_json_is_written_atomically(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _disk_full_on(monkeypatch, lambda path: path.name == "run.json")
    with pytest.raises(OSError):
        run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                   clock=ZERO, dry_run=True)
    with pytest.raises(OSError):
        run_split("eval", OracleFakeClient(POOL), run_dir=tmp_path / "b", clock=ZERO)
    assert not (tmp_path / "s" / "run.json").exists() and not (tmp_path / "b" / "run.json").exists()
    assert _leftovers(tmp_path) == []


def test_write_atomic_keeps_the_previous_content_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "x.json"
    atomic.write_atomic(target, b"old\n")
    assert target.read_bytes() == b"old\n"
    _disk_full_on(monkeypatch, lambda path: True)
    with pytest.raises(OSError):
        atomic.write_atomic(target, b"new\n")
    assert target.read_bytes() == b"old\n" and _leftovers(tmp_path) == []


@pytest.mark.parametrize("name", ["summary.json", "run.json"])
def test_unreadable_json_errors_name_the_file(name: str, baseline: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "s"
    run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=run_dir, clock=ZERO,
               replicates=20, dry_run=True, skip=EVAL_IDS[1:])
    (run_dir / name).write_bytes(b'{"format": "agent-gym/sw')
    with pytest.raises(ValueError, match=name):
        analyze_sweep_runs([run_dir], baseline, replicates=20)
