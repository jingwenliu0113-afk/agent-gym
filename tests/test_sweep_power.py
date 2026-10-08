"""decision 011：SweepPlan 偵測力模擬（experiments/sweep_power.py）的契約測試（排程本身見 tests/test_protocol.py）。"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pytest

from experiments.sweep_power import (
    CLIFF_DETECT_MIN,
    FLAG_MAX,
    PowerRow,
    Scenario,
    SimulatedSweep,
    candidate_plans,
    cliff_curve,
    default_scenarios,
    linear_curve,
    power,
    select_plan,
    simulate_sweep,
)
from replay.localization import (
    CheckpointEstimate,
    RolloutOutcome,
    SweepPlan,
    coarse_checkpoints,
    detect_cliff,
    estimate_checkpoint,
)


def _estimate(k: int, successes: int, n: int) -> CheckpointEstimate:
    outcomes = [RolloutOutcome(k, i, "success" if i < successes else "failure", "final_answer", False, 1) for i in range(n)]
    return estimate_checkpoint(k, outcomes)


# ---------------------------------------------------------------- curves / scenarios / plans


def test_cliff_and_linear_curves() -> None:
    assert cliff_curve(4, 1, 0.9, 0.1) == (0.9, 0.9, 0.1, 0.1)
    assert cliff_curve(2, 0, 0.8, 0.2) == (0.8, 0.2)
    assert linear_curve(5, 0.9, 0.1) == pytest.approx((0.9, 0.7, 0.5, 0.3, 0.1))
    assert linear_curve(1, 0.4, 0.0) == (0.4,)


@pytest.mark.parametrize("args", [(1, 0, 0.9, 0.1), (4, 3, 0.9, 0.1), (4, -1, 0.9, 0.1), (4, 1, 1.5, 0.1), (4, 1, 0.9, -0.1)])
def test_cliff_curve_rejects_bad_arguments(args: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        cliff_curve(*args)


def test_scenario_validation() -> None:
    Scenario("ok", "cliff", (0.9, 0.1), 0)
    for kwargs in (
        {"kind": "cliff", "curve": (0.9, 0.1), "cliff_k": None},
        {"kind": "flat", "curve": (0.5, 0.5), "cliff_k": 0},
        {"kind": "cliff", "curve": (0.9, 0.1), "cliff_k": 1},
        {"kind": "other", "curve": (0.5,), "cliff_k": None},
        {"kind": "flat", "curve": (), "cliff_k": None},
        {"kind": "flat", "curve": (1.2,), "cliff_k": None},
    ):
        with pytest.raises(ValueError):
            Scenario("bad", **kwargs)


def test_default_scenarios_match_decision_011() -> None:
    scenarios = default_scenarios()
    names = [s.name for s in scenarios]
    assert len(names) == len(set(names))
    cliffs = [s for s in scenarios if s.kind == "cliff"]
    # 4 種 (p_pre, p_post) × L ∈ {4, 8, 12} × 3 個位置
    assert len(cliffs) == 4 * 3 * 3
    for s in cliffs:
        m = s.cliff_k
        assert m is not None and 0 <= m <= len(s.curve) - 2
        assert len(s.curve) in (4, 8, 12)
        assert m in (0, (len(s.curve) - 1) // 2, len(s.curve) - 2)
        assert s.curve[m] - s.curve[m + 1] >= 0.5 - 1e-12
    pairs = {(s.curve[0], s.curve[-1]) for s in cliffs}
    assert pairs == {(0.9, 0.1), (0.8, 0.2), (0.7, 0.2), (0.6, 0.1)}
    gradual = [s for s in scenarios if s.kind == "gradual"]
    assert sorted(len(s.curve) for s in gradual) == [8, 12]
    for s in gradual:
        assert s.curve[0] == pytest.approx(0.9) and s.curve[-1] == pytest.approx(0.1)
        assert max(a - b for a, b in zip(s.curve, s.curve[1:])) < 0.15
    flat = [s for s in scenarios if s.kind == "flat"]
    assert sorted((s.curve[0], len(s.curve)) for s in flat) == [(0.5, 4), (0.5, 8), (0.5, 12), (0.9, 4), (0.9, 8), (0.9, 12)]
    assert all(len(set(s.curve)) == 1 for s in flat)


def test_candidate_plans_match_decision_011() -> None:
    plans = candidate_plans()
    expected = [
        SweepPlan(stride, n_coarse, n_fine, threshold)
        for stride in (1, 2, 3)
        for n_coarse in (4, 6, 8, 10)
        for n_fine in (16, 24, 32)
        for threshold in (0.25, 0.3, 0.4)
        if n_fine >= n_coarse
    ]
    assert list(plans) == expected


# ---------------------------------------------------------------- simulate_sweep


def test_simulate_sweep_flat_one_never_refines() -> None:
    plan = SweepPlan(2, 4, 16, 0.3)
    result = simulate_sweep((1.0,) * 7, plan, random.Random(0))
    assert isinstance(result, SimulatedSweep)
    assert [e.k for e in result.estimates] == coarse_checkpoints(6, 2)
    assert all(e.n == 4 and e.successes == 4 for e in result.estimates)
    assert result.refined_intervals == ()
    assert result.rollouts == 4 * len(coarse_checkpoints(6, 2))
    assert result.is_cliff is False


def test_simulate_sweep_deterministic_step_is_found_exactly() -> None:
    plan = SweepPlan(3, 4, 16, 0.3)
    curve = cliff_curve(10, 4, 1.0, 0.0)
    result = simulate_sweep(curve, plan, random.Random(1))
    # coarse = 0, 3, 6, 9；(3, 6) 下降 1.0 → refine 3..6 補到 16
    assert result.refined_intervals == ((3, 6),)
    by_k = {e.k: e for e in result.estimates}
    assert sorted(by_k) == [0, 3, 4, 5, 6, 9]
    assert all(by_k[k].n == 16 for k in (3, 4, 5, 6)) and by_k[0].n == by_k[9].n == 4
    assert result.rollouts == 4 * 4 + 12 * 2 + 16 * 2
    assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == (1.0, 4, True)


def test_simulate_sweep_uses_localization_rules() -> None:
    plan = SweepPlan(2, 6, 24, 0.25)
    rng = random.Random(7)
    for _ in range(20):
        curve = tuple(rng.random() for _ in range(rng.randint(1, 9)))
        result = simulate_sweep(curve, plan, random.Random(rng.random()))
        assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == detect_cliff(
            result.estimates, plan.drop_threshold
        )
        coarse = coarse_checkpoints(len(curve) - 1, plan.coarse_stride)
        assert set(coarse) <= {e.k for e in result.estimates}
        assert result.rollouts == sum(e.n for e in result.estimates)


def test_simulate_sweep_draw_order_is_round_robin() -> None:
    """rng.random() 依 sweep() 的 round-robin 順序消耗：coarse 每一輪依 k 遞增。"""
    plan = SweepPlan(1, 2, 2, 1.0)
    curve = (0.5, 0.5, 0.5)
    reference = random.Random(3)
    draws = [reference.random() for _ in range(6)]
    result = simulate_sweep(curve, plan, random.Random(3))
    # round 0: k=0,1,2；round 1: k=0,1,2
    expected = {k: sum(1 for i in range(2) if draws[i * 3 + k] < 0.5) for k in range(3)}
    assert {e.k: e.successes for e in result.estimates} == expected


def test_simulate_sweep_rejects_bad_curve() -> None:
    plan = SweepPlan(2, 4, 16, 0.3)
    for curve in ((), (0.5, 1.5), (-0.1,)):
        with pytest.raises(ValueError):
            simulate_sweep(curve, plan, random.Random(0))


# ---------------------------------------------------------------- power / select_plan


def test_power_is_reproducible_and_bounded() -> None:
    scenario = Scenario("c", "cliff", cliff_curve(8, 3, 0.9, 0.1), 3)
    plan = SweepPlan(2, 6, 24, 0.3)
    a = power(scenario, plan, reps=50, seed=0)
    b = power(scenario, plan, reps=50, seed=0)
    assert a == b
    assert isinstance(a, PowerRow) and a.reps == 50 and a.scenario == "c" and a.kind == "cliff" and a.plan == plan
    assert 0.0 <= a.detect_rate <= a.flag_rate <= 1.0
    assert a.mean_rollouts >= 6 * len(coarse_checkpoints(7, 2))
    flat = power(Scenario("f", "flat", (0.5,) * 4, None), plan, reps=20, seed=0)
    assert flat.detect_rate == 0.0


def test_power_rejects_bad_reps() -> None:
    scenario = Scenario("f", "flat", (0.5,) * 4, None)
    for reps in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            power(scenario, SweepPlan(2, 4, 16, 0.3), reps=reps)


def _row(plan: SweepPlan, scenario: str, kind: str, *, detect: float, flag: float, rollouts: float) -> PowerRow:
    return PowerRow(scenario, kind, plan, 100, detect, flag, rollouts)


def test_select_plan_applies_thresholds_then_cost_then_lexicographic() -> None:
    cheap_bad = SweepPlan(1, 4, 16, 0.25)
    good_a = SweepPlan(2, 6, 24, 0.3)
    good_b = SweepPlan(2, 4, 16, 0.3)
    rows = []
    for plan, detect, flag, cost in (
        (cheap_bad, 0.79, 0.0, 10.0),  # detect 不足
        (good_a, 0.9, 0.05, 50.0),
        (good_b, CLIFF_DETECT_MIN, FLAG_MAX, 40.0),
    ):
        rows.append(_row(plan, "c", "cliff", detect=detect, flag=detect, rollouts=cost))
        rows.append(_row(plan, "f", "flat", detect=0.0, flag=flag, rollouts=cost / 2))
    assert select_plan(rows) == good_b
    # 同成本 → 平均成本 → 字典序
    tie = [dataclasses.replace(r, mean_rollouts=40.0) if r.plan == good_a else r for r in rows]
    tie = [dataclasses.replace(r, mean_rollouts=20.0) if r.plan in (good_a, good_b) and r.kind == "flat" else r for r in tie]
    assert select_plan(tie) == good_b  # (2, 4, …) < (2, 6, …)


def test_select_plan_none_when_nothing_qualifies() -> None:
    plan = SweepPlan(2, 4, 16, 0.3)
    rows = [_row(plan, "c", "cliff", detect=0.5, flag=0.5, rollouts=10), _row(plan, "f", "flat", detect=0, flag=0.2, rollouts=5)]
    assert select_plan(rows) is None


def test_select_plan_requires_full_coverage() -> None:
    a, b = SweepPlan(2, 4, 16, 0.3), SweepPlan(2, 6, 24, 0.3)
    rows = [
        _row(a, "c", "cliff", detect=1, flag=1, rollouts=10),
        _row(a, "f", "flat", detect=0, flag=0, rollouts=5),
        _row(b, "c", "cliff", detect=1, flag=1, rollouts=10),
    ]
    with pytest.raises(ValueError):
        select_plan(rows)


# ---------------------------------------------------------------- 判定選項（decision 015）


def test_simulate_sweep_uses_the_selected_criterion() -> None:
    plan = SweepPlan(1, 8, 24, 0.3)
    curve = (0.95,) + (0.5,) * 6
    for criterion in ("wilson95", "bonferroni95"):
        result = simulate_sweep(curve, plan, random.Random(5), criterion=criterion)
        assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == detect_cliff(
            result.estimates, plan.drop_threshold, criterion=criterion
        )


def test_simulate_sweep_rejects_unknown_criterion() -> None:
    with pytest.raises(ValueError):
        simulate_sweep((0.5, 0.5), SweepPlan(1, 2, 2, 0.3), random.Random(0), criterion="other")
