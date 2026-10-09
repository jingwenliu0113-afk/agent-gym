"""最後收斂檢查（依 NEXT B.1–B.4 模擬付費流程）確認的缺陷之回歸測試（decision 023）。

- 續跑把做到一半的 source 從頭重做：已付費的 rollout 再付一次費；上限低於單一 source 的成本時永遠無法完成。
- trajectory 在 manifest 記帳之後才寫入：中斷或存檔失敗會讓目錄無法讀取。

全部只用假的 client（OracleFakeClient、包裝它的中斷 / 錯誤 client），不連網。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from experiments import run_sweep
from experiments.analyze_baseline import analyze_runs, load_runs
from experiments.analyze_sweep import analyze_sweep_runs
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import SWEEP_SUMMARY_FORMAT, main, run_sweeps, select_sources
from replay import trajectory as trajectory_module
from replay.aborts import MAX_EPISODE_ATTEMPTS, read_manifest
from replay.localization import RolloutOutcome, SweepPlan
from replay.trajectory import Trajectory
from tasks.generator import week1_pool
from tests.test_analyze_baseline import _sdk_error
from tests.test_run_sweep import _InterruptAfter

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
EVAL_IDS = [instance.task.task_id for instance in POOL]
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)
PLAN_TEXT = "3,2,2,0.3"


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("baseline") / "eval"
    run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO, dry_run=True)
    return run_dir


@pytest.fixture(scope="module")
def reference(baseline: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """沒有中斷的 sweep：續跑後的每條曲線必須與它相同（排程不變，decision 023 §1）。"""
    return run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95",
                      run_dir=tmp_path_factory.mktemp("reference") / "sweep", clock=ZERO, replicates=20, dry_run=True)


def _cli(baseline: Path, *extra: str) -> int:
    return main(["--baseline-run", str(baseline), "--plan", PLAN_TEXT, "--criterion", "wilson95", "--dry-run", *extra])


def _final(run_dir: Path) -> set[tuple[str, int, int]]:
    """manifest 中有最終結果的 (task, k, i)（decision 023 §1 的定義）。"""
    last: dict[tuple[str, int, int], Any] = {}
    for row in read_manifest(run_dir / "manifest.jsonl"):
        key = (row.task_id, row.checkpoint_turn, row.rollout_index)
        if key not in last or row.attempt > last[key].attempt:
            last[key] = row
    return {
        key for key, row in last.items()
        if row.outcome == "completed" or row.error_class == "agent_output"
        or (row.error_class == "infrastructure" and row.attempt >= MAX_EPISODE_ATTEMPTS)
    }


def _executed(run_dir: Path) -> set[tuple[str, int, int]]:
    return {(row.task_id, row.checkpoint_turn, row.rollout_index) for row in read_manifest(run_dir / "manifest.jsonl")}


def _completed(run_dir: Path) -> set[str]:
    return {path.stem for path in (run_dir / "sweeps").glob("*.json")}


def _assert_matches_reference(analysis: Any, reference: Any) -> None:
    assert [curve.task_id for curve in analysis.curves] == EVAL_IDS
    for curve in analysis.curves:
        assert curve.result == reference.results[curve.task_id], curve.task_id


def test_format_version() -> None:
    assert SWEEP_SUMMARY_FORMAT == "agent-gym/sweep-run/v7"  # v7：續跑接續做到一半的 source（decision 023）


# ---------------------------------------------------------------- 023 §1：續跑接續做到一半的 source


def test_a_resume_after_a_budget_stop_carries_over_the_partial_source(baseline: Path, reference: Any, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    partial = {task for task, _, _ in _executed(runs / "s1")} - _completed(runs / "s1")
    assert len(partial) == 1 and any(task in partial for task, _, _ in _final(runs / "s1"))  # 停在某個 source 的中途
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    assert not _executed(runs / "s2") & _final(runs / "s1")  # 已有最終結果的 rollout 不再付費
    analysis = analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20)
    _assert_matches_reference(analysis, reference)
    assert analysis.partial_sweeps == ()  # 接續進曲線的列不再是「部分完成」
    rows = [*read_manifest(runs / "s1" / "manifest.jsonl"), *read_manifest(runs / "s2" / "manifest.jsonl")]
    assert analysis.cost_usd == pytest.approx(sum(row.cost_usd for row in rows))
    assert len(rows) == sum(len(result.rollouts) for result in reference.results.values())  # 每個 rollout 只執行一次


def test_a_cap_below_one_source_still_makes_progress(baseline: Path, reference: Any, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    names: list[str] = []
    for n in range(1, 30):
        name = f"s{n}"
        resume = [item for previous in names for item in ("--resume-from", str(runs / previous))]
        assert _cli(baseline, "--run-name", name, "--max-cost-usd", "0.15", *resume) == 0
        if not (runs / name).exists():  # nothing left to run
            break
        names.append(name)
    else:
        pytest.fail("the resumes never finished the sweep")
    finals = [_final(runs / name) for name in names]
    assert sum(len(f) for f in finals) == len(set().union(*finals))  # 沒有任何 rollout 在兩個目錄都有最終結果
    analysis = analyze_sweep_runs([runs / name for name in names], baseline, replicates=20)
    _assert_matches_reference(analysis, reference)


def test_a_resume_after_ctrl_c_carries_over_the_finished_rollouts(baseline: Path, reference: Any, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(run_sweep, "OracleFakeClient", lambda pool: _InterruptAfter(OracleFakeClient(pool), 50))
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s1")
    monkeypatch.setattr(run_sweep, "OracleFakeClient", OracleFakeClient)
    assert not (runs / "s1" / "summary.json").exists() and _final(runs / "s1")
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    assert not _executed(runs / "s2") & _final(runs / "s1")
    _assert_matches_reference(analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20), reference)


class _InfraThenStop:
    """第 n 次 call 拋出 infrastructure 錯誤（重試），第 n + 1 次 call（重試的第一個 call）KeyboardInterrupt。"""

    def __init__(self, n: int) -> None:
        self._inner = OracleFakeClient(POOL)
        self._calls = 0
        self._n = n
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        self._calls += 1
        if self._calls == self._n:
            raise _sdk_error("APIConnectionError")
        if self._calls == self._n + 1:
            raise KeyboardInterrupt
        return self._inner.create(**request)


def test_a_rollout_whose_retry_was_interrupted_is_run_again(baseline: Path, reference: Any, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(run_sweep, "OracleFakeClient", lambda pool: _InfraThenStop(30))
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s1")
    monkeypatch.setattr(run_sweep, "OracleFakeClient", OracleFakeClient)
    pending = _executed(runs / "s1") - _final(runs / "s1")
    assert len(pending) == 1  # 第 1 次嘗試 infrastructure 中止、重試被中斷：不是最終結果
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    assert pending <= _executed(runs / "s2") and not _executed(runs / "s2") & _final(runs / "s1")
    analysis = analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20)
    _assert_matches_reference(analysis, reference)
    (task, _, _), = pending
    assert analysis.partial_sweeps == ({"run": "s1", "task_id": task, "rollouts": 1},)  # 沒進曲線、成本照計


def test_carried_rollouts_must_be_a_prefix_of_the_schedule(baseline: Path, tmp_path: Path) -> None:
    sources = select_sources(baseline)
    client = OracleFakeClient(POOL)
    first = EVAL_IDS[0]
    last_k = len(sources[first].turns) - 1
    carried = {first: {(last_k, 0): RolloutOutcome(last_k, 0, "success", "final_answer", False, 1)}}
    with pytest.raises(ValueError):
        run_sweeps(sources, client, plan=PLAN, criterion="wilson95", run_dir=tmp_path / "s", clock=ZERO, carried=carried,
                   dry_run=True)
    assert client.requests == []  # 在執行任何新的 rollout 之前就拒絕


@pytest.mark.parametrize(
    "carried",
    [
        {"bogus-task": {}},
        {EVAL_IDS[0]: {(0, 0): "success"}},
        {EVAL_IDS[0]: {(0, 1): RolloutOutcome(0, 0, "success", "final_answer", False, 1)}},
        {EVAL_IDS[0]: {("0", 0): RolloutOutcome(0, 0, "success", "final_answer", False, 1)}},
        {EVAL_IDS[0]: {(0, 0): RolloutOutcome(0, 0, "banana", "final_answer", False, 1)}},
        [],
    ],
    ids=["unknown-task", "not-an-outcome", "key-mismatch", "key-type", "bad-outcome", "not-a-mapping"],
)
def test_run_sweeps_validates_carried_rollouts(carried: Any, baseline: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                   clock=ZERO, carried=carried, dry_run=True)
    assert not (tmp_path / "s").exists()


def test_carried_rollouts_cannot_be_for_a_skipped_task(baseline: Path, tmp_path: Path) -> None:
    carried = {EVAL_IDS[0]: {(0, 0): RolloutOutcome(0, 0, "success", "final_answer", False, 1)}}
    with pytest.raises(ValueError):
        run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                   clock=ZERO, carried=carried, skip=[EVAL_IDS[0]], dry_run=True)


def test_analysis_rejects_a_resume_that_re_ran_an_ancestor_s_rollouts(baseline: Path, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    assert _cli(baseline, "--run-name", "s1", "--max-cost-usd", "0.3") == 0
    s1_id = json.loads((runs / "s1" / "run.json").read_text(encoding="utf-8"))["run_id"]
    run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=runs / "s2", clock=ZERO,
               replicates=20, skip=sorted(_completed(runs / "s1")), resumed_from=[s1_id], dry_run=True)  # 沒有 carried：從頭重做
    with pytest.raises(ValueError):
        analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20)


# ---------------------------------------------------------------- 023 §2：trajectory 先寫入，再記帳


def _fail_on_save(monkeypatch: pytest.MonkeyPatch, nth: int, error: BaseException) -> None:
    original = Trajectory.save
    calls = {"n": 0}

    def save(self: Trajectory, path: str | Path) -> None:
        calls["n"] += 1
        if calls["n"] == nth:
            raise error
        original(self, path)

    monkeypatch.setattr(Trajectory, "save", save)


def test_ctrl_c_while_saving_a_baseline_trajectory_leaves_a_readable_directory(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "eval"
    _fail_on_save(monkeypatch, 5, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO)
    monkeypatch.undo()
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert [row.outcome for row in rows] == ["completed"] * 4  # 第 5 個沒有記帳（同「進行中的嘗試不進 manifest」）
    assert len(load_runs([run_dir]).records) == 4
    rerun = tmp_path / "eval-2"
    run_split("eval", OracleFakeClient(POOL), run_dir=rerun, clock=ZERO)
    assert analyze_runs([run_dir, rerun], replicates=20).runs == ("eval", "eval-2")


def test_a_failed_save_is_a_harness_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "eval"
    _fail_on_save(monkeypatch, 3, OSError(28, "No space left on device"))
    with pytest.raises(OSError):
        run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO)
    monkeypatch.undo()
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert [(row.outcome, row.error_class) for row in rows] == [("completed", None)] * 2 + [("aborted", "harness")]
    assert rows[-1].billed_calls > 0  # 已計費的 call 照常記帳
    records = load_runs([run_dir]).records
    assert [record.outcome for record in records][-1] == "interrupted"


def test_ctrl_c_while_saving_a_sweep_rollout_then_resume(baseline: Path, reference: Any, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _fail_on_save(monkeypatch, 12, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _cli(baseline, "--run-name", "s1")
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    saved = {path.relative_to(runs / "s1" / "rollouts").as_posix() for path in (runs / "s1" / "rollouts").rglob("*.json")}
    assert saved == {f"{t}/k{k}-r{i}.json" for t, k, i in _final(runs / "s1")}  # manifest 說完成的都有檔案
    assert _cli(baseline, "--run-name", "s2", "--resume-from", str(runs / "s1")) == 0
    _assert_matches_reference(analyze_sweep_runs([runs / "s1", runs / "s2"], baseline, replicates=20), reference)


def test_trajectory_save_is_atomic(baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = next(source for source in select_sources(baseline).values() if source is not None)
    target = tmp_path / "t.json"
    trajectory.save(target)
    original = target.read_bytes()
    assert Trajectory.load(target) == trajectory and sorted(p.name for p in tmp_path.iterdir()) == ["t.json"]

    def broken_replace(src: Any, dst: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(trajectory_module.os, "replace", broken_replace)
    other = next(source for source in reversed(list(select_sources(baseline).values())) if source is not None)
    with pytest.raises(OSError):
        other.save(target)
    assert target.read_bytes() == original  # 目標檔不會只寫一半
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.json"]  # 暫存檔已刪除
    with pytest.raises(OSError):
        other.save(tmp_path / "new.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.json"]
    assert os.path.exists(target)
