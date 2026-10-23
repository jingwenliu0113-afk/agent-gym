"""Study 3 的統計工具（decision 036 §2 RQ2、§6、§7）：精確置換檢定、分級檢定、sweep 成本、預留額與估算、偵測力。"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from experiments import study3_stats as st
from experiments.study2_analysis import bootstrap_difference


def _brute_force_p(a: list[Fraction], b: list[Fraction]) -> Fraction:
    pooled = a + b
    observed = sum(a) / len(a) - sum(b) / len(b)
    hits = total = 0
    for chosen in itertools.combinations(range(len(pooled)), len(a)):
        rest = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        diff = sum(pooled[i] for i in chosen) / len(a) - sum(rest) / len(rest)
        hits += diff >= observed
        total += 1
    return Fraction(hits, total)


@pytest.mark.parametrize("a, b", [
    ([1, 1, Fraction(2, 3), 1, Fraction(1, 3)], [0, Fraction(1, 3), 1, Fraction(2, 3)]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([0, 0, Fraction(1, 2), 1], [1, 1, Fraction(2, 3), 1, 1]),
])
def test_the_exact_permutation_test_matches_brute_force(a: list, b: list) -> None:
    a, b = [Fraction(x) for x in a], [Fraction(x) for x in b]
    result = st.exact_permutation({str(i): v for i, v in enumerate(a)}, {str(i): v for i, v in enumerate(b)})
    assert result["p"] == pytest.approx(float(_brute_force_p(a, b)), abs=1e-12)
    assert result["point"] == pytest.approx(float(sum(a) / len(a) - sum(b) / len(b)))


def test_the_permutation_test_is_not_fooled_by_a_ceiling_level() -> None:
    """decision 036 §2：H1 全部成功、H3 有 4 題各失敗一次（差 0.042）時，bootstrap 判為有效，但精確置換檢定 p ≈ 0.057。"""
    h1 = {f"a{i}": Fraction(1) for i in range(32)}
    h3 = {f"b{i}": (Fraction(2, 3) if i < 4 else Fraction(1)) for i in range(32)}
    assert bootstrap_difference(h1, h3, replicates=10_000, seed=0)["low_positive"]
    perm = st.exact_permutation(h1, h3)
    assert perm["p"] == pytest.approx(0.057, abs=0.002)
    check = st.level_check(h1, h3)
    assert check["bootstrap"]["low_positive"] and not check["valid"]


def test_level_check_needs_both_tests_and_full_levels() -> None:
    h1 = {f"a{i}": Fraction(1) if i % 4 else Fraction(1, 3) for i in range(32)}
    h3 = {f"b{i}": Fraction(1, 3) if i % 2 else Fraction(0) for i in range(32)}
    assert st.level_check(h1, h3)["valid"]
    assert st.level_check(h1, dict(list(h3.items())[:31])) is None  # 不足 32 題：無法計算
    with pytest.raises(ValueError):
        st.exact_permutation({"a": 0.25}, {"b": 0})  # 不是 1–3 次 rollout 的成功比例


def test_the_exact_bootstrap_agrees_with_the_monte_carlo_rule() -> None:
    a = [6] * 20 + [4] * 8 + [2] * 4
    b = [6] * 14 + [2] * 10 + [0] * 8
    mc = bootstrap_difference({str(i): Fraction(v, 6) for i, v in enumerate(a)}, {str(i): Fraction(v, 6) for i, v in enumerate(b)},
                              replicates=10_000, seed=0)
    assert st.bootstrap_low_positive_exact(a, b) == mc["low_positive"] is True
    assert st.bootstrap_low_positive_exact([6] * 32, [6] * 31 + [4]) is False


def test_source_costs_follow_the_registered_cost_model() -> None:
    assert st.cost_samples(16, reps=200) == st.cost_samples(16, reps=200)  # 決定性
    mean16 = sum(st.cost_samples(16, reps=400)) / 400
    mean20 = sum(st.cost_samples(20, reps=400)) / 400
    assert 2400 < mean16 < mean20 < 3500  # decision 036 §7：每條平均約 2,800–3,100 次
    reserve = st.reserve_calls(reps=1000)
    assert 5800 < reserve < 7200  # 約 6,500 次（L = 20 的 P99）
    with pytest.raises(ValueError):
        st.source_cost(21, __import__("random").Random(0))


def test_the_estimate_asks_the_user_only_below_15_sources() -> None:
    local = st.estimate({"H1": 14.2, "H2": 16.5, "H3": 18.3}, agent="local", reps=400)
    assert local["length"] == 19 and local["n"] == math.floor(80_000 / local["c90_calls"]) and not local["ask_user"]
    haiku = st.estimate({"H1": 12.0, "H2": 15.0, "H3": 17.1}, agent="haiku", haiku_call_usd_p90=0.0003,
                        haiku_dev_spent_usd=0.5, haiku_baseline_p90_usd=2.8, reps=400)
    assert haiku["sweep_usd_left"] == pytest.approx(40 - 1 - 0.5 - 2.8) and haiku["n"] >= 15 and not haiku["ask_user"]
    # 以 c90（單條成本的 P90）× 每次呼叫成本的 P90 計算是保守的：每次 0.0006 美元時已低於 15 條，會回頭問使用者
    pricey = st.estimate({"H1": 12.0, "H2": 15.0, "H3": 17.1}, agent="haiku", haiku_call_usd_p90=0.0006,
                         haiku_dev_spent_usd=0.5, haiku_baseline_p90_usd=2.8, reps=400)
    assert pricey["n"] < 15 and pricey["ask_user"]
    with pytest.raises(ValueError):
        st.estimate({"H1": 12.0}, agent="haiku")
    with pytest.raises(ValueError):
        st.estimate({"H1": 12.0}, agent="opus")


def test_rq2_power_rows_cover_the_registered_grid() -> None:
    rows = st.rq2_power(reps=20, seed=0)
    assert len(rows) == 14  # κ 2 個 × (p₃, Δ) 中 p₁ ≤ 1 的 7 個
    assert all(r["p1"] <= 1 and 0 <= r["power"] <= min(r["bootstrap_only"], r["permutation_only"]) for r in rows)
    strong = next(r for r in rows if r["kappa"] == 10 and r["p3"] == 0.3 and r["delta"] == 0.4)
    assert strong["power"] > 0.8


def test_rq1_power_can_run_at_study3_lengths() -> None:
    """decision 036 §8 第 3 步：RQ1 的偵測力在 L = 16–20；Study 2 的預設長度不變。"""
    from experiments.study2_dev import POWER_LENGTHS, max_turns_power, summarize_power

    assert POWER_LENGTHS == (13, 16, 20)
    rows = max_turns_power(reps=2, lengths=(16, 18, 20))
    assert {r["length"] for r in rows} == {16, 18, 20}
    assert [s["length"] for s in summarize_power(rows, (16, 18, 20))] == [16, 16, 16, 16, 18, 18, 18, 18, 20, 20, 20, 20]
