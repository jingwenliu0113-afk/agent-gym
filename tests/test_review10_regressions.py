"""第十輪對抗式審查（decision 019 的實作）確認的缺陷之回歸測試（decision 020）。

- ``--resume-from`` 重複給時只保留最後一個值：少了一個目錄，該目錄完成的 source 被重新 sweep、再付一次費。
- 019 §3 只能發現漏給「較早」的目錄；漏給「較晚」的目錄（例如續跑被中斷後重下同一個指令）時，已完成的 source 會再付一次費。
- 一併處理：rollout 0 的同時間檢查與傳入順序無關；目標目錄已存在時提示續跑；run.json 的 run_id / resumed_from 驗證。

全部只用假的 client 與 --dry-run，不連網。
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from env.hashing import canonical_json
from experiments import run_sweep
from experiments.analyze_sweep import analyze_sweep_runs
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import SWEEP_SUMMARY_FORMAT, main, run_sweeps, select_sources
from replay.aborts import read_manifest
from replay.localization import SweepPlan
from tasks.generator import week1_pool
from tests.test_run_sweep import _InterruptAfter

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
EVAL_IDS = [instance.task.task_id for instance in POOL]
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)
PLAN_TEXT = "3,2,2,0.3"
RUN_ID = re.compile(r"[0-9a-f]{32}")


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("baseline") / "eval"
    run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO)
    return run_dir


def _cli(baseline: Path, *extra: str) -> int:
    return main(["--baseline-run", str(baseline), "--plan", PLAN_TEXT, "--criterion", "wilson95", "--dry-run", *extra])


def _completed(run_dir: Path) -> set[str]:
    return {path.stem for path in (run_dir / "sweeps").glob("*.json")}


def _manifest_tasks(run_dir: Path) -> set[str]:
    return {row.task_id for row in read_manifest(run_dir / "manifest.jsonl")}


def _run_json(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return data


def _two_budget_stops(baseline: Path, runs: Path) -> tuple[set[str], set[str]]:
    """s1 因預算停止；s2 從 s1 續跑也因預算停止。回傳兩者完成的 source。"""
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    assert _cli(baseline, "--run-name", "s2", "--max-cost-usd", "0.3", "--resume-from", str(runs / "s1")) == 0
    first, second = _completed(runs / "s1"), _completed(runs / "s2")
    assert first and second and not first & second
    return first, second


# ---------------------------------------------------------------- 020 §1：--resume-from 可重複


@pytest.mark.parametrize("order", ["newest-first", "oldest-first"])
def test_repeated_resume_from_flags_keep_every_directory(order: str, baseline: Path, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    first, second = _two_budget_stops(baseline, runs)
    names = ["s2", "s1"] if order == "newest-first" else ["s1", "s2"]
    flags = [item for name in names for item in ("--resume-from", str(runs / name))]
    assert _cli(baseline, "--run-name", "s3", *flags) == 0
    assert not _manifest_tasks(runs / "s3") & (first | second)  # 沒有任何 source 再付一次費
    analysis = analyze_sweep_runs([runs / "s1", runs / "s2", runs / "s3"], baseline, replicates=20)
    assert len(analysis.curves) == 15 and analysis.not_completed == ()


# ---------------------------------------------------------------- 020 §3：漏給較晚的目錄


def test_resume_refuses_when_a_later_directory_is_left_out(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _two_budget_stops(baseline, runs)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1"))  # 重下 s2 的指令、只改名稱
    assert not (runs / "s3").exists()
    assert "s2" in capsys.readouterr().err


def test_resume_refuses_when_an_interrupted_later_directory_is_left_out(baseline: Path, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """審查情境：續跑被 Ctrl-C 中斷（已完成幾個 source、沒有 summary），之後重下同一個續跑指令。"""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    monkeypatch.setattr(run_sweep, "OracleFakeClient", lambda pool: _InterruptAfter(OracleFakeClient(pool), 60))
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1"))
    monkeypatch.setattr(run_sweep, "OracleFakeClient", OracleFakeClient)
    assert _completed(runs / "s2") and not (runs / "s2" / "summary.json").exists()
    with pytest.raises(SystemExit):
        _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1"))
    assert not (runs / "s3").exists()
    assert _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1"), str(runs / "s2")) == 0
    assert not _manifest_tasks(runs / "s3") & (_completed(runs / "s1") | _completed(runs / "s2"))


def test_every_descendant_in_the_chain_is_detected(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _two_budget_stops(baseline, runs)
    assert _cli(baseline, "--run-name", "s3", "--max-cost-usd", "0.3", "--resume-from", str(runs / "s1"), str(runs / "s2")) == 0
    assert _completed(runs / "s3")
    for given, missing in ((["s1"], {"s2", "s3"}), (["s1", "s2"], {"s3"}), (["s2", "s1"], {"s3"})):
        capsys.readouterr()
        with pytest.raises(SystemExit):
            _cli(baseline, "--run-name", "s4", "--resume-from", *(str(runs / name) for name in given))
        err = capsys.readouterr().err
        assert all(name in err for name in missing), (given, err)
        assert not (runs / "s4").exists()
    assert _cli(baseline, "--run-name", "s4", "--resume-from", *(str(runs / n) for n in ("s1", "s2", "s3"))) == 0


def test_a_later_directory_without_a_manifest_does_not_block(baseline: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    monkeypatch.setattr(run_sweep, "OracleFakeClient", lambda pool: _InterruptAfter(OracleFakeClient(pool), 0))
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1"))
    monkeypatch.setattr(run_sweep, "OracleFakeClient", OracleFakeClient)
    assert (runs / "s2" / "run.json").exists() and read_manifest(runs / "s2" / "manifest.jsonl") == ()  # 空的 manifest（decision 024 §1）
    assert _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s1")) == 0  # s2 沒有可分析的內容（018 §4）


def test_unrelated_sweep_directories_do_not_block(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    assert _cli(baseline, "--run-name", "other", "--max-cost-usd", "0.1") == 0  # 另一個新的 sweep（不是從 s1 續跑）
    (runs / "junk").mkdir()
    (runs / "junk" / "run.json").write_text("{not json", encoding="utf-8")
    (runs / "notes.txt").write_text("x", encoding="utf-8")
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    assert not _manifest_tasks(runs / "s2") & _completed(runs / "s1")


# ---------------------------------------------------------------- 020 §2 / §4：run_id 與 resumed_from


def test_run_json_records_run_id_and_lineage(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _two_budget_stops(baseline, runs)
    one, two = _run_json(runs / "s1"), _run_json(runs / "s2")
    assert SWEEP_SUMMARY_FORMAT == one["format"]
    assert RUN_ID.fullmatch(one["run_id"]) and RUN_ID.fullmatch(two["run_id"]) and one["run_id"] != two["run_id"]
    assert one["resumed_from"] == [] and two["resumed_from"] == [one["run_id"]]
    assert _cli(baseline, "--run-name", "s3", "--resume-from", str(runs / "s2"), str(runs / "s1")) == 0
    assert _run_json(runs / "s3")["resumed_from"] == sorted([one["run_id"], two["run_id"]])


@pytest.mark.parametrize("value", ["abc", "A" * 32, 7, None, ["x"]], ids=["short", "upper", "int", "none", "list"])
def test_run_sweeps_validates_resumed_from(value: Any, baseline: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                   clock=ZERO, resumed_from=[value])
    assert not (tmp_path / "s").exists()


@pytest.fixture(scope="module")
def stopped_sweep(baseline: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("stopped") / "sweep"
    run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=run_dir, clock=ZERO,
               replicates=20, max_cost_usd=0.3)
    return run_dir


def _edit(sweep: Path, edit: Callable[[dict[str, Any]], Any]) -> None:
    data = _run_json(sweep)
    edit(data)
    (sweep / "run.json").write_text(canonical_json(data) + "\n", encoding="utf-8")


LINEAGE_CHANGES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "run-id-missing": lambda d: d.pop("run_id"),
    "run-id-short": lambda d: d.update(run_id="abc"),
    "run-id-upper": lambda d: d.update(run_id="A" * 32),
    "run-id-int": lambda d: d.update(run_id=12),
    "resumed-from-missing": lambda d: d.pop("resumed_from"),
    "resumed-from-str": lambda d: d.update(resumed_from="0" * 32),
    "resumed-from-bad-id": lambda d: d.update(resumed_from=["xyz"]),
    "resumed-from-duplicate": lambda d: d.update(resumed_from=["0" * 32, "0" * 32]),
    "resumed-from-unsorted": lambda d: d.update(resumed_from=["1" * 32, "0" * 32]),
    "resumed-from-self": lambda d: d.update(resumed_from=[d["run_id"]]),
}


@pytest.mark.parametrize("change", sorted(LINEAGE_CHANGES))
def test_analysis_validates_run_id_and_lineage(change: str, stopped_sweep: Path, baseline: Path, tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    shutil.copytree(stopped_sweep, sweep)
    analyze_sweep_runs([sweep], baseline, replicates=20)  # 前提：未修改時可分析
    _edit(sweep, LINEAGE_CHANGES[change])
    with pytest.raises(ValueError):
        analyze_sweep_runs([sweep], baseline, replicates=20)


def test_two_directories_with_the_same_run_id_are_rejected(stopped_sweep: Path, baseline: Path, tmp_path: Path) -> None:
    """複製的 sweep 目錄（inode 不同、內容相同）：成本與中止嘗試會被算兩次。"""
    interrupted = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_sweeps(select_sources(baseline), _InterruptAfter(OracleFakeClient(POOL), 3), plan=PLAN, criterion="wilson95",
                   run_dir=interrupted, clock=ZERO)
    twin = tmp_path / "twin"
    shutil.copytree(interrupted, twin)
    with pytest.raises(ValueError):
        analyze_sweep_runs([interrupted, twin], baseline, replicates=20)


# ---------------------------------------------------------------- 020 §5：同時間檢查與傳入順序無關


@pytest.fixture(scope="module")
def timed_baselines(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("timed")
    first, later = root / "first", root / "later"
    run_split("eval", OracleFakeClient(POOL), run_dir=first, clock=ZERO)
    run_split("eval", OracleFakeClient(POOL), run_dir=later, clock=ZERO)
    shutil.copytree(first, root / "first-copy")
    shutil.copytree(later, root / "later-copy")
    return {name: root / name for name in ("first", "later", "first-copy", "later-copy")}


def _ids(sources: dict[str, Any]) -> dict[str, str | None]:
    return {task_id: None if source is None else source.trajectory_id for task_id, source in sources.items()}


def test_a_tie_after_the_earliest_rollout_zero_does_not_matter_in_any_order(timed_baselines: dict[str, Path]) -> None:
    expected = _ids(select_sources(timed_baselines["first"]))
    for order in itertools.permutations(["first", "later", "later-copy"]):
        assert _ids(select_sources([timed_baselines[name] for name in order])) == expected, order


def test_a_tie_at_the_earliest_rollout_zero_is_rejected_in_every_order(timed_baselines: dict[str, Path]) -> None:
    for order in itertools.permutations(["first", "first-copy", "later"]):
        with pytest.raises(ValueError):
            select_sources([timed_baselines[name] for name in order])


# ---------------------------------------------------------------- 020 §6：目標目錄已存在


def test_an_existing_target_directory_points_to_resume(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    before = sorted(path.relative_to(tmp_path).as_posix() for path in (tmp_path / "runs" / "s1").rglob("*"))
    capsys.readouterr()
    with pytest.raises(SystemExit):
        _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3")
    assert "--resume-from" in capsys.readouterr().err
    assert sorted(path.relative_to(tmp_path).as_posix() for path in (tmp_path / "runs" / "s1").rglob("*")) == before
