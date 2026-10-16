"""experiments/study2_analysis.py（decision 034 §2 / §7 的事先登記分析）的契約測試。"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import local_agent_config
from experiments import study2_analysis as s2
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import REGISTERED_PLAN, run_sweeps, select_sources
from replay.localization import RolloutOutcome, SweepPlan, SweepResult, assess_drop, estimate_checkpoint
from replay.trajectory import Trajectory
from tasks.suites import suite_pool

EVAL = suite_pool("study2", "eval")
BY_ID = {i.task.task_id: i for i in EVAL}
SMALL = SweepPlan(coarse_stride=4, n_coarse=1, n_fine=1, drop_threshold=0.3)


def ZERO() -> float:  # noqa: N802
    return 0.0


class ForkingClient(OllamaOracleFakeClient):
    """測試用：指定的 task 有兩條劇本（oracle 與一個 negative），在兩者第一次不同的那一步 j，以機率 p 走 oracle、否則走 negative；
    前綴已經走過 j 的 rerun 照前綴的劇本繼續。其他 task 照 oracle。可以由真正的 sweep 產生 P(success | k) 的驟降。"""

    def __init__(self, forks: dict[str, tuple[str, float]], seed: int = 0) -> None:
        super().__init__(EVAL)
        self._forks = {}
        for task_id, (negative, p) in forks.items():
            oracle, wrong = BY_ID[task_id].oracle_calls, dict(BY_ID[task_id].negative_calls)[negative]
            j = next((i for i, (a, b) in enumerate(zip(oracle, wrong)) if a != b), min(len(oracle), len(wrong)))
            self._forks[BY_ID[task_id].task.instruction] = (tuple(oracle), tuple(wrong), j, p)
        self._rng = random.Random(seed)

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request["messages"]
        fork = self._forks.get(messages[1]["content"])
        if fork is None:
            return super().chat(request)
        oracle, wrong, j, p = fork
        assistants = [m for m in messages if m["role"] == "assistant"]
        t = len(assistants)
        if t > j:
            call = assistants[j].get("tool_calls") or []
            taken = [(c["function"]["name"], c["function"]["arguments"]) for c in call]
            script = wrong if j < len(wrong) and taken == [wrong[j]] else oracle
        elif t == j:
            script = oracle if self._rng.random() < p else wrong
        else:
            script = oracle
        response = super().chat({**request, "messages": messages[:2]})  # 借用回覆的形狀
        if t < len(script):
            name, args = script[t]
            response["message"] = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": copy.deepcopy(args)}}]}
        else:
            response["message"] = {"role": "assistant", "content": "Done."}
        return response


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_is_exact_at_the_zero_boundary() -> None:
    """驗證 B 第二輪：浮點誤差曾讓精確 2.5% 分位數為 0 的資料被判為有效。"""
    l1 = {f"a{i:02}": Fraction(v, 3) for i, v in enumerate([0, 3, 0, 3, 3, 2, 2, 3, 3, 1, 0, 2])}
    l3 = {f"b{i:02}": Fraction(v, 3) for i, v in enumerate([2, 1, 1, 1, 0, 0, 2, 1, 0, 0, 3, 0])}
    out = s2.bootstrap_difference(l1, l3)
    assert out["low"] == 0.0 and out["low_positive"] is False


def test_bootstrap_is_independent_of_key_order_and_validates_replicates() -> None:
    a = {f"t{i}": Fraction(v, 3) for i, v in enumerate([2, 1, 2, 3, 2, 1, 1, 2, 2, 2, 3, 1])}
    b = {f"u{i}": Fraction(v, 3) for i, v in enumerate([0, 0, 0, 2, 2, 1, 2, 0, 2, 1, 2, 2])}
    forward = s2.bootstrap_difference(a, b, replicates=500)
    backward = s2.bootstrap_difference(dict(reversed(list(a.items()))), dict(reversed(list(b.items()))), replicates=500)
    assert forward == backward
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            s2.bootstrap_difference(a, b, replicates=bad)  # type: ignore[arg-type]
    assert s2.bootstrap_difference({}, b) is None
    high = {f"h{i}": Fraction(1) for i in range(12)}
    low = {f"l{i}": Fraction(0) for i in range(12)}
    assert s2.bootstrap_difference(high, low, replicates=200)["low_positive"] is True


# ------------------------------------------------------------------ 曲線（由 rerun 結果建構）


def _result(points: list[tuple[int, list[tuple[str, str]]]], plan: SweepPlan = REGISTERED_PLAN) -> SweepResult:
    rollouts = tuple(RolloutOutcome(k, i, outcome, termination, False, 1) for k, runs in points for i, (outcome, termination) in enumerate(runs))
    estimates = tuple(estimate_checkpoint(k, [r for r in rollouts if r.k == k]) for k, _ in points)
    a = assess_drop(estimates, plan.drop_threshold, criterion="bonferroni95")
    return SweepResult(task_id="t", source_trajectory_id="x", plan=plan, estimates=estimates, rollouts=rollouts, refined_intervals=(),
                       largest_drop=a.largest_drop, largest_drop_k=a.largest_drop_k, is_cliff=a.is_cliff, criterion="bonferroni95",
                       max_turns_confounded=a.max_turns_confounded)


def _runs(success: int, fail: int, fail_max_turns: int = 0, success_max_turns: int = 0) -> list[tuple[str, str]]:
    return ([("success", "final_answer")] * success + [("success", "max_turns")] * success_max_turns
            + [("failure", "final_answer")] * fail + [("failure", "max_turns")] * fail_max_turns)


def test_classify_curve_from_reruns() -> None:
    cliff = _result([(0, _runs(45, 3)), (1, _runs(2, 46))])
    assert s2.classify_curve(cliff, True) == "cliff_mechanical" and s2.classify_curve(cliff, False) == "cliff_behavioral"
    assert s2.classify_curve(_result([(0, _runs(3, 7)), (2, _runs(0, 10))]), False) == "undecidable"  # 最大 p̂ 0.3 < 0.4
    assert s2.classify_curve(_result([(0, _runs(4, 6)), (2, _runs(3, 7))]), False) == "gradual"
    assert s2.classify_curve(_result([(0, _runs(5, 5))]), False) == "undecidable"  # 只有一個 checkpoint


def test_the_two_max_turns_rules() -> None:
    """失敗的 rerun 中有一些用完 turn：第一輪的規則判為混淆；排除後仍合格的規則判為 cliff。"""
    mixed = _result([(0, _runs(45, 3)), (1, _runs(2, 40, fail_max_turns=6))])
    assert s2.classify_curve(mixed, False, rule="round1") == "max_turns_confounded"
    assert s2.classify_curve(mixed, False, rule="excluding") == "cliff_behavioral"
    # 下降完全來自用完 turn 的 rerun（decision 006 想排除的情況）：兩種規則都不是 cliff
    exhausted = _result([(0, _runs(45, 3)), (1, _runs(9, 0, fail_max_turns=39))])
    assert s2.classify_curve(exhausted, False, rule="round1") == "max_turns_confounded"
    assert s2.classify_curve(exhausted, False, rule="excluding") == "max_turns_confounded"
    with pytest.raises(ValueError):
        s2.assess_rule(mixed, "other")


def test_sensitivity_handles_successful_max_turns_reruns() -> None:
    """驗證 B 第二輪的 blocker：成功但以 max_turns 結束的 rerun 曾讓 p̂' 的計算 raise。"""
    curve = _result([(0, _runs(9, 0, success_max_turns=1)), (2, _runs(10, 0)), (4, _runs(0, 0, fail_max_turns=10))])
    out = s2._sensitivity(curve)
    assert out["skipped_k"] == [4] and out["is_cliff"] is False


def test_qualifying_drops_below_the_decidable_line_are_counted() -> None:
    curve = _result([(k, _runs(18, 30)) for k in range(6)] + [(k, _runs(0, 48)) for k in range(6, 12)])
    assert curve.is_cliff and s2.classify_curve(curve, False) == "undecidable"


# ------------------------------------------------------------------ RQ1 / RQ2


def _curve(cls: str, success: bool = False, level: str = "L1", seed: int = 2, k: int | None = 1, j: int | None = None) -> dict:
    return {"class": cls, "classes": {"round1": cls, "excluding": cls}, "source_success": success, "level": level, "seed": seed,
            "standardized_drop": None, "largest_drop_k": k, "first_irreversible_turn": j}


@pytest.mark.parametrize(
    ("curves", "kwargs", "outcome"),
    [
        ([_curve("gradual", True)], {"failed_sources": 0}, "ceiling"),
        ([_curve("gradual", True)], {"failed_sources": 2}, "incomplete"),  # 有失敗的 source，但它們的曲線沒有完成
        ([_curve("gradual", True)], {"failed_sources": 0, "missing_sources": 1}, "incomplete"),
        ([_curve("undecidable"), _curve("max_turns_confounded")], {}, "no_decidable"),
        ([_curve("cliff_behavioral")] * 10, {}, "mostly_cliff"),
        ([_curve("gradual")] * 10, {}, "mostly_gradual"),
        ([_curve("cliff_mechanical")] * 5 + [_curve("gradual")] * 5, {}, "undetermined"),
    ],
)
def test_rq1_conclusion_rule(curves: list[dict], kwargs: dict, outcome: str) -> None:
    assert s2.rq1(curves, **kwargs)["outcome"] == outcome


def test_rq1_denominator_and_counts() -> None:
    curves = [_curve("cliff_mechanical", j=3), _curve("cliff_behavioral", k=4, j=2), _curve("cliff_behavioral", j=None), _curve("gradual"),
              _curve("undecidable"), _curve("max_turns_confounded"), _curve("cliff_behavioral", success=True)]
    out = s2.rq1(curves)
    assert (out["k"], out["n"], out["n_failed_curves"]) == (3, 4, 6)
    assert (out["mechanical"], out["behavioral"], out["n_undecidable"], out["n_max_turns_confounded"]) == (1, 2, 1, 1)
    assert out["cliff_before_irreversible"] == {"k": 1, "n": 2, "without_irreversible_step": 1}


def test_rq2_reports_k_n_and_wilson_per_class_on_complete_blocks() -> None:
    curves = [_curve("cliff_behavioral", level="L3", seed=2), _curve("gradual", level="L3", seed=2), _curve("gradual", level="L3", seed=3)]
    out = s2.rq2(curves, [2])["L3"]
    assert out["n_sources"] == 2
    assert (out["failed"]["cliff_behavioral"]["k"], out["failed"]["cliff_behavioral"]["n"]) == (1, 2)
    assert 0 < out["failed"]["cliff_behavioral"]["low"] < out["failed"]["cliff_behavioral"]["high"] < 1


# ------------------------------------------------------------------ trajectory 層級


def _with(pool, task_id: str, name: str):  # noqa: ANN001, ANN202
    return tuple(dataclasses.replace(i, oracle_calls=dict(i.negative_calls)[name]) if i.task.task_id == task_id else i for i in pool)


@pytest.fixture(scope="module")
def failing_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pool = _with(EVAL, "gen2-epic-closeout-L2-s3", "E3:duplicate-comment")
    pool = _with(pool, "gen2-epic-closeout-L1-s2", "E2:close-epic-first")
    pool = _with(pool, "gen2-verified-close-L2-s2", "V2:no-comment")
    run_dir = tmp_path_factory.mktemp("s2a") / "eval"
    run_split("eval", OllamaOracleFakeClient(pool), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True, suite="study2")
    return run_dir


def _trajectory(run_dir: Path, task_id: str, r: int = 0) -> Trajectory:
    return Trajectory.load(run_dir / "trajectories" / f"{task_id}-r{r}.json")


def test_mechanical_unrecoverability_after_a_second_comment(failing_baseline: Path) -> None:
    source = _trajectory(failing_baseline, "gen2-epic-closeout-L2-s3")
    calls = len(source.records())
    assert source.grade().success is False
    assert s2.mechanically_unrecoverable(source, calls) is True
    assert s2.mechanically_unrecoverable(source, calls - 1) is False


def test_first_irreversible_turn_uses_real_changes(failing_baseline: Path, tmp_path: Path) -> None:
    assert s2.first_irreversible_turn(_trajectory(failing_baseline, "gen2-epic-closeout-L1-s2")) == 1
    assert s2.first_irreversible_turn(_trajectory(failing_baseline, "gen2-verified-close-L1-s2")) == 2
    # 對已經 closed 的 sub-issue 再 close 一次沒有任何變更，不算不可逆步驟（epic L3 有一個已 closed 的 sub-issue）
    instance = BY_ID["gen2-epic-closeout-L3-s2"]
    closed = instance.task.params["closed_subs"][0]
    calls = (instance.oracle_calls[0], ("update_issue", {"issue_id": closed, "status": "closed"}), *instance.oracle_calls[1:])
    pool = tuple(dataclasses.replace(i, oracle_calls=calls) if i is instance else i for i in EVAL)
    run_split("eval", OllamaOracleFakeClient(pool), run_dir=tmp_path / "b", clock=ZERO, agent_config=local_agent_config(), dry_run=True, suite="study2")
    source = _trajectory(tmp_path / "b", instance.task.task_id)
    assert s2.first_irreversible_turn(source) == 3  # get、search、no-op close 之後的第一個真正的 close


def test_tool_error_fixes_counts_only_later_turns(failing_baseline: Path) -> None:
    source = _trajectory(failing_baseline, "gen2-sync-repair-L1-s2")
    assert s2.tool_error_fixes(source) == (0, 0)


def test_baseline_analysis_with_failures(failing_baseline: Path) -> None:
    out = s2.analyze_baseline_runs([failing_baseline], replicates=300)
    assert out["n_failed_episodes"] == 2 and out["complete"] and not out["ceiling"]
    l1, l2 = out["levels"]["L1"], out["levels"]["L2"]
    assert l1["success_rate"] == 1.0 and l1["success_unsafe_rate"] == pytest.approx(1 / 12)
    assert l1["success_unsafe_among_successes"] == {"k": 1, "n": 12}
    assert l1["forbidden_by_rule"] == {"E2": 1}
    assert l2["success_rate"] == pytest.approx(10 / 12)
    assert l2["constraint_violation_rate"] == pytest.approx(1 / 12)  # 重複 comment 也算 constraint violation
    assert out["families"]["epic-closeout"]["forbidden_by_rule"] == {"E2": 1}
    assert out["level_check"]["l1_minus_l2"] == pytest.approx(2 / 12) and out["level_check"]["valid"] is not None
    assert set(out["beyond_default_search"]) == set(BY_ID)


def test_three_rollouts_per_task_and_earliest_adoption(tmp_path: Path) -> None:
    """每題 R 次的比例、task 等權；同一個 (task, rollout_index) 在兩個目錄中都結束時取最早的一次。"""
    client = ForkingClient({"gen2-sync-repair-L1-s2": ("Y1:partial", 0.5)}, seed=3)
    run_split("eval", client, run_dir=tmp_path / "a", clock=ZERO, agent_config=local_agent_config(), rollouts=3, dry_run=True, suite="study2")
    out = s2.analyze_baseline_runs([tmp_path / "a"], replicates=200)
    assert out["rollouts_per_task"] == 3 and out["complete"] and out["rollout_counts_not_r"] == {}
    episodes = [json.loads(p.read_text())["termination"] for p in (tmp_path / "a" / "trajectories").glob("gen2-sync-repair-L1-s2-r*.json")]
    assert len(episodes) == 3
    run_split("eval", client, run_dir=tmp_path / "b", clock=ZERO, agent_config=local_agent_config(), rollouts=3, dry_run=True, suite="study2")
    both = s2.analyze_baseline_runs([tmp_path / "a", tmp_path / "b"], replicates=200)
    assert both["rollout_counts_not_r"] == {} and both["levels"]["L1"]["n_episodes"] == 36


def test_an_incomplete_baseline_is_never_a_ceiling(tmp_path: Path) -> None:
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=tmp_path / "b", clock=ZERO, agent_config=local_agent_config(), rollouts=3,
              max_model_calls=40, suite="study2")
    out = s2.analyze([tmp_path / "b"], replicates=200)
    assert out.baseline["stopped_by_budget"] == ["b"] and not out.baseline["complete"]
    assert out.baseline["ceiling"] is False and out.sweep_skipped == "incomplete"
    assert out.baseline["level_check"]["valid"] is None and out.data_complete is False
    text = s2.render_markdown(out)
    assert "資料不完整" in text and "因預算停止的目錄" in text and "無法計算" in text


def test_ceiling_skips_the_sweep(tmp_path: Path) -> None:
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=tmp_path / "b", clock=ZERO, agent_config=local_agent_config(), dry_run=True,
              suite="study2")
    analysis = s2.analyze([tmp_path / "b"], replicates=200)
    assert analysis.sweep is None and analysis.sweep_skipped == "ceiling" and analysis.data_complete
    assert analysis.baseline["level_check"]["valid"] is False  # 沒有差距 → 分級無效
    assert "完全天花板" in s2.render_markdown(analysis)


def test_a_real_sweep_produces_mechanical_and_behavioral_cliffs(tmp_path: Path) -> None:
    """可分岔的假 client：j 之前的 rerun 以 0.9 走 oracle，之後照前綴走 negative → 真正的 sweep 產生 cliff。"""
    forks = {"gen2-epic-closeout-L2-s3": ("E3:duplicate-comment", 0.9),  # 第二則 comment → 機械上不可恢復
             "gen2-scoped-triage-L2-s2": ("S4:no-assign", 0.9)}  # 少了 assign → 狀態仍可恢復（行為性）
    baseline_client = ForkingClient({t: (n, 0.0) for t, (n, _) in forks.items()})  # source 一定失敗
    run_split("eval", baseline_client, run_dir=tmp_path / "b", clock=ZERO, agent_config=local_agent_config(), dry_run=True, suite="study2")
    sources = select_sources(tmp_path / "b")
    run_sweeps(sources, ForkingClient(forks, seed=1), plan=REGISTERED_PLAN, criterion="bonferroni95", run_dir=tmp_path / "s", clock=ZERO,
               agent_config=local_agent_config(), dry_run=True)
    analysis = s2.analyze([tmp_path / "b"], [tmp_path / "s"], replicates=200)
    sweep = analysis.sweep
    by_id = {c["task_id"]: c for c in sweep["curves"]}
    mech, beh = by_id["gen2-epic-closeout-L2-s3"], by_id["gen2-scoped-triage-L2-s2"]
    assert mech["class"] == "cliff_mechanical" and beh["class"] == "cliff_behavioral"
    assert mech["next_k"] == mech["largest_drop_k"] + 1
    assert mech["cliff_rules"] and beh["first_irreversible_turn"] is not None
    r1 = sweep["rq1"]
    assert (r1["k"], r1["n"], r1["mechanical"], r1["behavioral"]) == (2, 2, 1, 1)
    assert sweep["by_rule"]["excluding"]["rq1"]["k"] == 2
    assert sweep["sensitivity"]["n"] == 2 and sweep["data_complete"] and sweep["complete_seed_blocks"] == [2, 3, 4]
    text = s2.render_markdown(analysis)
    assert "Advanced" not in text and "advanced" not in text and "cliff：機械性" in text


def test_an_incomplete_sweep_is_reported_as_incomplete(failing_baseline: Path, tmp_path: Path) -> None:
    sources = select_sources(failing_baseline)
    run_sweeps(sources, OllamaOracleFakeClient(_with(_with(_with(EVAL, "gen2-epic-closeout-L2-s3", "E3:duplicate-comment"),
                                                                 "gen2-epic-closeout-L1-s2", "E2:close-epic-first"),
                                                           "gen2-verified-close-L2-s2", "V2:no-comment")),
               plan=SMALL, criterion="bonferroni95", run_dir=tmp_path / "s", clock=ZERO, agent_config=local_agent_config(), dry_run=True,
               max_model_calls=30)
    sweep = s2.analyze_sweep_dirs([tmp_path / "s"], [failing_baseline])
    assert not sweep["data_complete"] and sweep["not_completed"]
    assert set(sweep["complete_seed_blocks"]) < {2, 3, 4}
    assert sweep["rq1"]["outcome"] in ("incomplete", "no_decidable")
    assert sweep["rq1"]["n_failed_sources"] == 2


def test_round1_runs_are_rejected(tmp_path: Path) -> None:
    week1 = suite_pool("week1", "eval")
    run_split("eval", OllamaOracleFakeClient(week1), run_dir=tmp_path / "w", clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    with pytest.raises(ValueError, match="Study 2"):
        s2.analyze_baseline_runs([tmp_path / "w"])


def test_analysis_dict_is_json_and_labels_dry_runs(failing_baseline: Path) -> None:
    analysis = s2.analyze([failing_baseline], replicates=200)
    data = json.loads(json.dumps(analysis.to_dict()))
    assert data["format"] == s2.STUDY2_ANALYSIS_FORMAT and data["dry_run"] is True
    assert s2.render_markdown(analysis).count("dry-run") >= 1
