"""replay.localization 的 pending_intervals / run_protocol（decision 013 §1–2）與精確門檻比較的契約測試。"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Callable

import pytest

from replay.localization import (
    CheckpointEstimate,
    ProtocolRun,
    RolloutOutcome,
    SweepPlan,
    coarse_checkpoints,
    detect_cliff,
    estimate_checkpoint,
    pending_intervals,
    run_protocol,
)

PLAN = SweepPlan(coarse_stride=2, n_coarse=8, n_fine=24, drop_threshold=0.3)


def _estimate(k: int, successes: int, n: int) -> CheckpointEstimate:
    outcomes = [RolloutOutcome(k, i, "success" if i < successes else "failure", "final_answer", False, 1) for i in range(n)]
    return estimate_checkpoint(k, outcomes)


def _scripted(succeed: Callable[[int, int], bool], calls: list[tuple[int, int]] | None = None):
    def rollout(k: int, i: int) -> RolloutOutcome:
        if calls is not None:
            calls.append((k, i))
        return RolloutOutcome(k, i, "success" if succeed(k, i) else "failure", "final_answer", False, 1)

    return rollout


def _exact_at_least(a: int, n_a: int, b: int, n_b: int, threshold: float) -> bool:
    return Fraction(a, n_a) - Fraction(b, n_b) >= Fraction(repr(threshold))


# ---------------------------------------------------------------- pending_intervals


def test_pending_marks_coarse_drops_at_or_above_threshold() -> None:
    plan = SweepPlan(2, 10, 16, 0.3)
    estimates = {0: _estimate(0, 10, 10), 2: _estimate(2, 8, 10), 4: _estimate(4, 2, 10), 6: _estimate(6, 0, 10)}
    assert pending_intervals(estimates, plan) == ((2, 4),)
    assert pending_intervals(estimates, SweepPlan(2, 10, 16, 0.2)) == ((0, 2), (2, 4), (4, 6))  # 1.0 − 0.8 精確等於 0.2
    assert pending_intervals(estimates, SweepPlan(2, 10, 16, 0.7)) == ()


def test_pending_ignores_fully_refined_consecutive_pairs() -> None:
    estimates = {3: _estimate(3, 16, 16), 4: _estimate(4, 0, 16)}
    assert pending_intervals(estimates, SweepPlan(1, 4, 16, 0.3)) == ()
    # 同一對、但一端少於 n_fine → 仍待加密
    estimates = {3: _estimate(3, 16, 16), 4: _estimate(4, 0, 8)}
    assert pending_intervals(estimates, SweepPlan(1, 4, 16, 0.3)) == ((3, 4),)


def test_pending_includes_gaps_even_when_both_endpoints_are_refined() -> None:
    """端點都已有 n_fine 但中間的 k 從未估計：下降位置不明 → 待加密（decision 013 §2）。"""
    estimates = {2: _estimate(2, 18, 24), 4: _estimate(4, 0, 24)}
    assert pending_intervals(estimates, PLAN) == ((2, 4),)


def test_pending_exact_boundary() -> None:
    estimates = {0: _estimate(0, 7, 10), 1: _estimate(1, 4, 10)}
    assert estimates[0].p_hat - estimates[1].p_hat < 0.3  # 前提：浮點減法落在門檻下
    assert pending_intervals(estimates, SweepPlan(1, 10, 16, 0.3)) == ((0, 1),)


@pytest.mark.parametrize("threshold", (0.25, 0.3, 0.4))
@pytest.mark.parametrize("n", (4, 6, 8, 10, 16, 24, 32))
def test_pending_matches_exact_rationals(n: int, threshold: float) -> None:
    plan = SweepPlan(1, min(n, 4), max(n + 1, 16), threshold)  # n < n_fine：只看下降是否達門檻
    for a, b in itertools.product(range(n + 1), repeat=2):
        got = pending_intervals({0: _estimate(0, a, n), 1: _estimate(1, b, n)}, plan)
        assert got == (((0, 1),) if _exact_at_least(a, n, b, n, threshold) else ()), (n, threshold, a, b)


def test_pending_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        pending_intervals({0: _estimate(1, 1, 2)}, PLAN)  # key 與 estimate.k 不符
    with pytest.raises(ValueError):
        pending_intervals({0: _estimate(0, 1, 2)}, (2, 8, 24, 0.3))  # type: ignore[arg-type]
    assert pending_intervals({}, PLAN) == ()
    assert pending_intervals({0: _estimate(0, 1, 2)}, PLAN) == ()


# ---------------------------------------------------------------- detect_cliff（精確比較）


def test_detect_cliff_exact_drop_equal_to_threshold() -> None:
    a, b = _estimate(0, 11, 24), _estimate(1, 5, 24)
    assert a.p_hat - b.p_hat < 0.25  # 前提
    drop, k, _ = detect_cliff((a, b), 0.25)
    assert k == 0 and drop >= 0.25


def test_detect_cliff_exact_threshold_with_non_overlapping_cis_is_cliff() -> None:
    a, b = _estimate(0, 469, 1000), _estimate(1, 169, 1000)
    assert a.p_hat - b.p_hat < 0.3 and a.ci_low > b.ci_high  # 前提
    drop, k, cliff = detect_cliff((a, b), 0.3)
    assert (k, cliff) == (0, True) and drop >= 0.3


def test_detect_cliff_exact_tie_picks_smaller_k() -> None:
    """7/10 − 4/10 與 5/10 − 2/10 都恰為 3/10（浮點不相等）→ 同大小，取 k 較小者。"""
    estimates = (_estimate(0, 7, 10), _estimate(1, 4, 10), _estimate(2, 5, 10), _estimate(3, 2, 10))
    drop, k, _ = detect_cliff(estimates, 0.3)
    assert (k, drop) == (0, 0.3)


@pytest.mark.parametrize("threshold", (0.25, 0.3, 0.4))
@pytest.mark.parametrize(("n_a", "n_b"), [(4, 16), (16, 4), (6, 24), (10, 32), (24, 24), (8, 32)])
def test_detect_cliff_matches_exact_rationals(n_a: int, n_b: int, threshold: float) -> None:
    for a in range(n_a + 1):
        for b in range(n_b + 1):
            ea, eb = _estimate(0, a, n_a), _estimate(1, b, n_b)
            drop, k, cliff = detect_cliff((ea, eb), threshold)
            exact = _exact_at_least(a, n_a, b, n_b, threshold)
            assert k == 0 and (drop >= threshold) == exact
            assert cliff == (exact and ea.ci_low > eb.ci_high)


# ---------------------------------------------------------------- run_protocol


def test_flat_curve_only_runs_coarse() -> None:
    calls: list[tuple[int, int]] = []
    run = run_protocol(7, PLAN, _scripted(lambda k, i: True, calls))
    assert isinstance(run, ProtocolRun)
    coarse = coarse_checkpoints(7, 2)
    assert calls == [(k, i) for i in range(PLAN.n_coarse) for k in coarse]
    assert [(o.k, o.rollout_index) for o in run.order] == calls
    assert sorted(run.outcomes) == coarse and all(len(run.outcomes[k]) == 8 for k in coarse)
    assert run.refined_intervals == ()


def test_refine_repeats_until_no_unrefined_large_drop() -> None:
    """coarse p̂ = 1, 2/8, 0, 0, 0（k = 0, 2, 4, 6, 7）：第 1 輪 refine (0, 2) 後 p̂(2) = 18/24，與未加密的 k = 4 之間下降 0.75 且
    k = 3 從未估計 → 第 2 輪 refine (2, 4)。之後所有 >= 門檻的下降都在連續的 (2, 3) 上。"""
    calls: list[tuple[int, int]] = []

    def succeed(k: int, i: int) -> bool:
        return k < 2 or (k == 2 and (i < 2 or i >= 8))

    run = run_protocol(7, PLAN, _scripted(succeed, calls))
    assert run.refined_intervals == ((0, 2), (2, 4))
    coarse = [0, 2, 4, 6, 7]
    expected = [(k, i) for i in range(8) for k in coarse]
    expected += [(1, i) for i in range(8)] + [(k, i) for i in range(8, 24) for k in (0, 1, 2)]
    expected += [(3, i) for i in range(8)] + [(k, i) for i in range(8, 24) for k in (3, 4)]
    assert calls == expected
    counts = {k: len(v) for k, v in run.outcomes.items()}
    assert counts == {0: 24, 1: 24, 2: 24, 3: 24, 4: 24, 6: 8, 7: 8}
    estimates = tuple(estimate_checkpoint(k, list(run.outcomes[k])) for k in sorted(run.outcomes))
    assert pending_intervals({e.k: e for e in estimates}, PLAN) == ()
    drop, k, cliff = detect_cliff(estimates, PLAN.drop_threshold)
    assert (drop, k, cliff) == (0.75, 2, True)


def test_shared_endpoint_is_refined_once_in_round_robin() -> None:
    calls: list[tuple[int, int]] = []
    plan = SweepPlan(2, 2, 6, 0.3)

    def succeed(k: int, i: int) -> bool:  # coarse p̂ = 1, 0.5, 0 → (0, 2) 與 (2, 4) 同時待加密
        return k == 0 or (k in (1, 2) and i % 2 == 0)

    run = run_protocol(4, plan, _scripted(succeed, calls))
    assert len(calls) == len(set(calls))
    assert run.refined_intervals[:2] == ((0, 2), (2, 4))
    first_refine = calls[6 : 6 + 2 + 4 * 5]
    assert first_refine[:3] == [(1, 0), (3, 0), (1, 1)]  # 中間點先補到 n_coarse，依 k 遞增輪流
    assert all(len(run.outcomes[k]) == 6 for k in range(5))


def test_rollout_must_return_matching_key() -> None:
    def bad(k: int, i: int) -> RolloutOutcome:
        return RolloutOutcome(k, i + 1, "success", "final_answer", False, 1)

    with pytest.raises(ValueError):
        run_protocol(3, PLAN, bad)


def test_rollout_exceptions_propagate() -> None:
    class Stop(RuntimeError):
        pass

    def stop(k: int, i: int) -> RolloutOutcome:
        raise Stop()

    with pytest.raises(Stop):
        run_protocol(3, PLAN, stop)


@pytest.mark.parametrize("last", [-1, True, 1.0])
def test_bad_last_is_rejected(last: object) -> None:
    with pytest.raises(ValueError):
        run_protocol(last, PLAN, _scripted(lambda k, i: True))  # type: ignore[arg-type]


def test_bad_plan_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_protocol(3, (2, 8, 24, 0.3), _scripted(lambda k, i: True))  # type: ignore[arg-type]
