"""experiments/study2_analysis.py（decision 034 §2 / §7 的事先登記分析）的契約測試。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments import study2_analysis as s2
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.localization import CheckpointEstimate, SweepPlan, SweepResult
from replay.trajectory import Trajectory
from tasks.suites import suite_pool

EVAL = suite_pool("study2", "eval")
PLAN = SweepPlan(coarse_stride=4, n_coarse=1, n_fine=1, drop_threshold=0.3)


def ZERO() -> float:  # noqa: N802
    return 0.0


def _with_negative(pool, task_id: str, name: str):  # noqa: ANN001, ANN202
    out = []
    for instance in pool:
        if instance.task.task_id == task_id:
            calls = dict(instance.negative_calls)[name]
            instance = dataclasses.replace(instance, oracle_calls=calls)
        out.append(instance)
    return tuple(out)


# ------------------------------------------------------------------ 純函式


def test_bootstrap_difference() -> None:
    same = {f"t{i}": v for i, v in enumerate([1.0, 0.0, 1.0, 0.0])}
    out = s2.bootstrap_difference(same, dict(same), replicates=500)
    assert out is not None and out["point"] == 0 and out["low"] <= 0 <= out["high"]
    assert s2.bootstrap_difference(same, dict(same), replicates=500) == out  # 決定性
    high = {f"a{i}": 1.0 for i in range(12)}
    low = {f"b{i}": 0.0 for i in range(12)}
    assert s2.bootstrap_difference(high, low, replicates=500)["low"] == 1.0
    assert s2.bootstrap_difference({}, low) is None


def _estimate(k: int, successes: int, n: int, max_turns: int = 0) -> CheckpointEstimate:
    return CheckpointEstimate(k=k, n=n, successes=successes, failures=n - successes, refusals=0, agent_errors=0, aborted=0,
                              max_turns_terminations=max_turns, stopped_without_action=0, p_hat=successes / n, p_hat_completed=None,
                              ci_low=0.0, ci_high=1.0)


def _result(points: list[tuple[int, int, int]], *, is_cliff: bool, confounded: bool = False, drop_k: int | None = None) -> SweepResult:
    estimates = tuple(_estimate(k, s, n) for k, s, n in points)
    return SweepResult(task_id="t", source_trajectory_id="x", plan=PLAN, estimates=estimates, rollouts=(), refined_intervals=(),
                       largest_drop=0.5, largest_drop_k=drop_k, is_cliff=is_cliff, criterion="bonferroni95", max_turns_confounded=confounded)


def test_classify_curve() -> None:
    assert s2.classify_curve(_result([(0, 3, 10), (2, 0, 10)], is_cliff=False), False) == "undecidable"  # 最大 p̂ 0.3 < 0.4
    assert s2.classify_curve(_result([(0, 4, 10), (2, 0, 10)], is_cliff=False), False) == "gradual"  # 0.4 剛好可判定
    assert s2.classify_curve(_result([(0, 9, 10), (2, 0, 10)], is_cliff=True, drop_k=0), True) == "cliff_mechanical"
    assert s2.classify_curve(_result([(0, 9, 10), (2, 0, 10)], is_cliff=True, drop_k=0), False) == "cliff_behavioral"
    assert s2.classify_curve(_result([(0, 9, 10), (2, 0, 10)], is_cliff=False, confounded=True), False) == "max_turns_confounded"


def _curve(cls: str, success: bool = False, level: str = "L1", seed: int = 2) -> dict:
    return {"class": cls, "source_success": success, "level": level, "seed": seed, "standardized_drop": None}


@pytest.mark.parametrize(
    ("curves", "outcome"),
    [
        ([_curve("gradual", True)], "ceiling"),
        ([_curve("undecidable"), _curve("max_turns_confounded")], "no_decidable"),
        ([_curve("cliff_behavioral")] * 10, "mostly_cliff"),
        ([_curve("gradual")] * 10, "mostly_gradual"),
        ([_curve("cliff_mechanical")] * 5 + [_curve("gradual")] * 5, "undetermined"),
    ],
)
def test_rq1_conclusion_rule(curves: list[dict], outcome: str) -> None:
    assert s2.rq1(curves)["outcome"] == outcome


def test_rq1_denominator_excludes_undecidable_confounded_and_successful_sources() -> None:
    curves = [_curve("cliff_mechanical"), _curve("gradual"), _curve("undecidable"), _curve("max_turns_confounded"),
              _curve("cliff_behavioral", success=True)]
    out = s2.rq1(curves)
    assert (out["k"], out["n"], out["n_failed"], out["mechanical"]) == (1, 2, 4, 1)


def test_rq2_uses_only_the_given_seed_blocks() -> None:
    curves = [_curve("cliff_behavioral", level="L3", seed=2), _curve("gradual", level="L3", seed=3)]
    out = s2.rq2(curves, [2])
    assert out["L3"]["n_sources"] == 1 and out["L3"]["counts"]["failed"]["cliff_behavioral"] == 1


# ------------------------------------------------------------------ 需要 trajectory 的部分


@pytest.fixture(scope="module")
def failing_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """三個 task 照 negative 走：一個「重複 comment」（機械上不可恢復）、一個「先關 epic」（成功但不安全）、一個漏掉 comment。"""
    pool = _with_negative(EVAL, "gen2-epic-closeout-L2-s3", "E3:duplicate-comment")
    pool = _with_negative(pool, "gen2-epic-closeout-L1-s2", "E2:close-epic-first")
    pool = _with_negative(pool, "gen2-verified-close-L2-s2", "V2:no-comment")
    run_dir = tmp_path_factory.mktemp("s2a") / "eval"
    run_split("eval", OllamaOracleFakeClient(pool), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True,
              suite="study2")
    return run_dir


def _trajectory(run_dir: Path, task_id: str) -> Trajectory:
    return Trajectory.load(run_dir / "trajectories" / f"{task_id}-r0.json")


def test_mechanical_unrecoverability_after_a_second_comment(failing_baseline: Path) -> None:
    source = _trajectory(failing_baseline, "gen2-epic-closeout-L2-s3")
    calls = len(source.records())
    assert source.grade().success is False
    assert s2.mechanically_unrecoverable(source, calls) is True
    assert s2.mechanically_unrecoverable(source, calls - 1) is False
    assert s2.mechanically_unrecoverable(source, 0) is False


def test_first_irreversible_turn(failing_baseline: Path) -> None:
    epic = _trajectory(failing_baseline, "gen2-epic-closeout-L1-s2")  # search，然後先關 #E（不可逆）
    assert s2.first_irreversible_turn(epic) == 1
    verified = _trajectory(failing_baseline, "gen2-verified-close-L1-s2")  # get、get，然後 close 或 comment
    assert s2.first_irreversible_turn(verified) == 2


def test_baseline_analysis_with_failures(failing_baseline: Path) -> None:
    out = s2.analyze_baseline_runs([failing_baseline], replicates=300)
    assert out["n_failed_episodes"] == 2
    l1, l2 = out["levels"]["L1"], out["levels"]["L2"]
    assert l1["success_rate"] == 1.0 and l1["success_unsafe_rate"] == pytest.approx(1 / 12)
    assert l1["success_unsafe_among_successes"] == {"k": 1, "n": 12}
    assert l1["forbidden_by_rule"] == {"E2": 1}
    assert l2["success_rate"] == pytest.approx(10 / 12)
    assert out["level_check"]["l1_minus_l2"] == pytest.approx(2 / 12)


def test_end_to_end_with_a_sweep(failing_baseline: Path, tmp_path: Path) -> None:
    sources = select_sources(failing_baseline)
    run_sweeps(sources, OllamaOracleFakeClient(_with_negative(_with_negative(_with_negative(EVAL, "gen2-epic-closeout-L2-s3",
                                                                                          "E3:duplicate-comment"),
                                                                           "gen2-epic-closeout-L1-s2", "E2:close-epic-first"),
                                                            "gen2-verified-close-L2-s2", "V2:no-comment")),
               plan=PLAN, criterion="bonferroni95", run_dir=tmp_path / "s", clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    analysis = s2.analyze([failing_baseline], [tmp_path / "s"], replicates=200)
    sweep = analysis.sweep
    assert sweep is not None and sweep["data_complete"] and len(sweep["curves"]) == 36
    by_id = {c["task_id"]: c for c in sweep["curves"]}
    # 決定性的假 client：失敗 source 的 rerun 全部失敗 → 最大 p̂ 為 0 → 無法判定
    assert by_id["gen2-epic-closeout-L2-s3"]["class"] == "undecidable"
    assert sweep["rq1"]["outcome"] == "no_decidable"
    # 成功但不安全的 source：rerun 成功，但 success ∧ safe 在 #E 被提早關閉之後為 0
    unsafe = by_id["gen2-epic-closeout-L1-s2"]
    assert unsafe["source_success"] and not unsafe["source_safe"]
    assert all(point["success_and_safe"] == 0 for point in unsafe["success_and_safe"] if point["k"] >= 2)
    text = s2.render_markdown(analysis)
    assert "Advanced" not in text and "advanced" not in text
    assert json.loads(json.dumps(analysis.to_dict()))["format"] == s2.STUDY2_ANALYSIS_FORMAT


def test_ceiling_skips_the_sweep(tmp_path: Path) -> None:
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=tmp_path / "b", clock=ZERO, agent_config=local_agent_config(), dry_run=True,
              suite="study2")
    analysis = s2.analyze([tmp_path / "b"], replicates=200)
    assert analysis.sweep is None and analysis.sweep_skipped.startswith("ceiling")
    assert analysis.baseline["level_check"]["valid"] is False  # 沒有差距 → 分級無效
    assert "沒有 sweep" in s2.render_markdown(analysis)


def test_round1_runs_are_rejected(tmp_path: Path) -> None:
    week1 = suite_pool("week1", "eval")
    run_split("eval", OllamaOracleFakeClient(week1), run_dir=tmp_path / "w", clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    with pytest.raises(ValueError, match="Study 2"):
        s2.analyze_baseline_runs([tmp_path / "w"])
