"""Study 3 的 sweep 執行方式（decision 036 §6、§8 第 1 步 B 部分）：凍結紀錄中的排列、只跑失敗的 source、最多 25 條（跨段累計）、
開始新的 source 之前檢查剩下的累計上限 ≥ 預留額、接續的 source 不再檢查、沒有凍結紀錄就拒絕。

eval instance 只經過程式（dry-run 的 oracle 假 client 重放 oracle 或錯誤解）；測試不輸出 eval 的內容（decision 036 §4）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments import study3_plan as plan
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.localization import SweepPlan
from tasks import generator_v3 as g3
from tasks import suites
from tasks.suites import eval_task_ids, suite_pool

SMALL = SweepPlan(coarse_stride=6, n_coarse=1, n_fine=1, drop_threshold=0.3)
N_FAILING = 8


def ZERO() -> float:  # noqa: N802
    return 0.0


def _failing_pool() -> tuple:
    """排列中前 N_FAILING 題改用一個不成功的錯誤解當 oracle（只以程式判定，不輸出內容）。"""
    pool = suite_pool("study3", "eval")
    chosen = set(plan.sweep_permutation(eval_task_ids("study3"))[:N_FAILING])
    out = []
    for instance in pool:
        if instance.task.task_id in chosen:
            calls = next(c for _, c in instance.negative_calls if not g3.run_calls(instance.task, c)[1].success)
            instance = dataclasses.replace(instance, oracle_calls=calls)
        out.append(instance)
    return tuple(out)


@pytest.fixture(scope="module")
def failing() -> tuple:
    return _failing_pool()


@pytest.fixture(scope="module")
def baseline(failing, tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("s3b") / "b"
    run_split("eval", OllamaOracleFakeClient(failing), run_dir=run_dir, clock=ZERO, suite="study3", agent_config=local_agent_config(),
              dry_run=True, rollouts=1)
    return run_dir


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "036-freeze-record.json"
    path.write_text(json.dumps({"format": plan.FREEZE_FORMAT, "sweep_permutation": plan.sweep_permutation(eval_task_ids("study3")),
                                "reserve": {"local_calls": 50, "haiku_usd": 2.0}}), encoding="utf-8")
    monkeypatch.setattr(plan, "FREEZE_RECORD", path)
    return path


def _sweep(sources, failing, run_dir: Path, **kwargs):
    return run_sweeps(sources, OllamaOracleFakeClient(failing), plan=SMALL, criterion="bonferroni95", run_dir=run_dir, clock=ZERO,
                      agent_config=local_agent_config(), dry_run=True, order="frozen_permutation", **kwargs)


def test_the_permutation_is_fixed_and_shared() -> None:
    ids = eval_task_ids("study3")
    import random

    expected = sorted(ids)
    random.Random("study3/sweep-order").shuffle(expected)
    assert plan.sweep_permutation(ids) == expected == plan.sweep_permutation(list(reversed(ids)))
    assert sorted(expected) == sorted(ids) and plan.MAX_SOURCES == 25


def test_a_frozen_sweep_needs_the_freeze_record(baseline: Path, failing, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan, "FREEZE_RECORD", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="freeze record"):
        _sweep(select_sources(baseline), failing, tmp_path / "s", cumulative_left_calls=10_000)
    assert not (tmp_path / "s").exists()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": plan.FREEZE_FORMAT, "sweep_permutation": sorted(eval_task_ids("study3")),
                               "reserve": {"local_calls": 50, "haiku_usd": 2.0}}), encoding="utf-8")
    monkeypatch.setattr(plan, "FREEZE_RECORD", bad)
    with pytest.raises(ValueError, match="permutation"):
        _sweep(select_sources(baseline), failing, tmp_path / "s", cumulative_left_calls=10_000)


def test_only_failed_sources_run_in_permutation_order_up_to_the_cap(baseline: Path, failing, record: Path, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan, "MAX_SOURCES", 3)
    sources = select_sources(baseline)
    order = plan.failed_order(sources, plan.sweep_permutation(eval_task_ids("study3")))
    assert len(order) == N_FAILING
    summary = _sweep(sources, failing, tmp_path / "s", cumulative_left_calls=10_000)
    assert list(summary.completed) == order[:3] and not summary.stopped_by_budget
    resumed = _sweep(sources, failing, tmp_path / "t", cumulative_left_calls=10_000, skip=order[:2])  # 跨段：已完成 2 條
    assert list(resumed.completed) == order[2:3]


def test_a_new_source_starts_only_if_the_reserve_is_left(baseline: Path, failing, record: Path, tmp_path: Path) -> None:
    sources = select_sources(baseline)
    order = plan.failed_order(sources, plan.sweep_permutation(eval_task_ids("study3")))
    summary = _sweep(sources, failing, tmp_path / "s", cumulative_left_calls=49)  # 預留額 50：一條也不開始
    assert summary.completed == () and summary.stopped_by_budget
    summary = _sweep(sources, failing, tmp_path / "t", cumulative_left_calls=51)  # 第一條之後剩下的少於 50
    assert list(summary.completed) == order[:1] and summary.stopped_by_budget


def test_local_study3_sweeps_must_follow_the_plan(baseline: Path, failing, record: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.run_sweep as rs

    class _RealLooking:
        pass

    monkeypatch.setattr(suites, "STUDY3_APPROVED", True)
    monkeypatch.setattr(suites, "STUDY3_FROZEN", True)
    monkeypatch.setattr(rs, "make_policy", lambda client, config: None)
    sources = select_sources(baseline)
    common = dict(plan=SMALL, criterion="bonferroni95", agent_config=local_agent_config(), call_log=True, max_model_calls=3000)
    with pytest.raises(ValueError, match="frozen_permutation"):
        run_sweeps(sources, _RealLooking(), run_dir=tmp_path / "a", order="pool", cumulative_left_calls=80_000, **common)
    with pytest.raises(ValueError, match="cumulative"):
        run_sweeps(sources, _RealLooking(), run_dir=tmp_path / "b", order="frozen_permutation", **common)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_frozen_sweeps_pass_the_strict_analysis_checks(baseline: Path, failing, record: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """完成的曲線是凍結順序的前綴：到上限而結束（沒有因預算停止）、跨段續跑、在 source 之間因預留額停止，都通過嚴格核對。"""
    from experiments.analyze_sweep import allowed_orders, analyze_sweep_runs

    monkeypatch.setattr(plan, "MAX_SOURCES", 3)
    sources = select_sources(baseline)
    assert allowed_orders(sources) == ("pool", "frozen_permutation")
    first = tmp_path / "s1"
    summary = _sweep(sources, failing, first, cumulative_left_calls=10_000)
    assert len(summary.completed) == 3 and not summary.stopped_by_budget
    assert analyze_sweep_runs([first], [baseline], replicates=10) is not None
    stopped = tmp_path / "s2"
    _sweep(sources, failing, stopped, cumulative_left_calls=51)
    assert analyze_sweep_runs([stopped], [baseline], replicates=10) is not None
