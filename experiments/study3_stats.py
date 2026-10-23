"""Study 3 的統計工具（decision 036 §2 RQ2、§6 預算、§7 估算、§8 第 3 步）。不連網、不需要 GPU。

分級是否有效（RQ2）
===================
``level_check(h1, h3)``：兩個檢定都成立才算有效（decision 036 §2）。
- 034 的 task bootstrap（``study2_analysis.bootstrap_difference``：B = 10,000、seed 0、percentile、2.5% 分位數 > 0）；
- task 層級的單側精確置換檢定（``exact_permutation``）：H1 與 H3 的 task 值（採計 rollout 的成功比例，分母 1–3）乘以 6 化為
  0–6 的整數，依值分組，列舉每組選入 H1 的個數，以組合數精確計算所有分組下「H1 平均 − H3 平均」≥ 觀測值的比例；p < 0.025。
- 任一級別不足 ``TASKS_PER_LEVEL`` 題 → 無法計算（None），不是無效。

偵測力（``rq2_power``，只報告）：每題的成功機率取自平均 p、集中度 κ 的 Beta 分布（p = 1 時每題都是 1），R = 3 次獨立
Bernoulli；偵測力 = 兩個檢定同時成立的比例。模擬中的 bootstrap 以 B → ∞ 的精確分布判定（P(差 ≤ 0) < 0.025），近似正式規則的
B = 10,000 Monte Carlo（正式分析仍用 ``bootstrap_difference``）。

sweep 的成本（decision 036 §6、§7）
==================================
一條失敗 source 的成本以登記的 plan (2, 10, 48, 0.3) 模擬（``run_protocol``）：曲線的起點固定為 1.0，cliff 與 linear 各半
（034 的形狀）；checkpoint k 的 rerun 成功時花 L − k 次呼叫、失敗時花 ``MAX_TURNS`` − k 次。不使用 dev 的成功率。
- ``reserve_calls()``：預留額 = L = 20 時單條成本的 P99（所有 source 同一個值，寫進凍結紀錄）；
- ``estimate(...)``：c90 = L =（dev 各級別平均 turn 的最大值，無條件進位，最大 20）時單條成本的 P90；
  N_local = floor(80,000 ÷ c90)；N_haiku = floor((40 − 1 − dev 實際花費 − eval baseline 預估花費的 P90) ÷ (c90 × 每次呼叫成本的 P90))。
  任一個 < 15 → 回頭問使用者（decision 036 §7）。
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Any, Mapping, Sequence

from experiments.study2_analysis import bootstrap_difference
from experiments.study2_dev import REGISTERED_PLAN, cliff_curve, linear_curve
from replay.localization import RolloutOutcome, SweepPlan, run_protocol

TASKS_PER_LEVEL = 32  # 4 題型 × 8 個 eval seed
MAX_TURNS = 20
SWEEP_CAP_CALLS = 80_000
PAID_TOTAL_USD = 40.0
SAFETY_MARGIN_USD = 1.0
ASK_BELOW = 15  # 能 sweep 的失敗 source 少於這個數 → 回頭問使用者
RESERVE_LENGTH = 20
COST_REPS = 4000
POWER_P3 = (0.3, 0.6, 0.8)
POWER_DELTA = (0.1, 0.25, 0.4)
POWER_KAPPA = (2, 10)
ALPHA = 0.025


# ---------------------------------------------------------------- 分級是否有效


def _sixths(values: Mapping[str, Fraction | float]) -> list[int]:
    out = []
    for value in values.values():
        six = Fraction(value) * 6
        if six.denominator != 1 or not 0 <= six <= 6:
            raise ValueError("task values must be success fractions with denominators 1, 2 or 3")
        out.append(int(six))
    return out


def exact_permutation(h1: Mapping[str, Fraction | float], h3: Mapping[str, Fraction | float]) -> dict[str, Any]:
    """單側精確置換檢定：H0 下 H1 / H3 的標籤可交換；p = P(H1 平均 − H3 平均 ≥ 觀測值)。回傳 {"point", "p", "n1", "n3"}。"""
    a, b = _sixths(h1), _sixths(h3)
    n1, n3 = len(a), len(b)
    if not n1 or not n3:
        raise ValueError("both levels need at least one task")
    pooled = a + b
    total = sum(pooled)
    counts = [pooled.count(v) for v in range(7)]
    observed = sum(a)  # 平均差 = sum_a / n1 − (total − sum_a) / n3，對 sum_a 單調遞增
    ways_ge = 0

    def walk(v: int, chosen: int, s: int, ways: int) -> None:
        nonlocal ways_ge
        if v == 7:
            if chosen == n1 and s >= observed:
                ways_ge += ways
            return
        for c in range(min(counts[v], n1 - chosen) + 1):
            walk(v + 1, chosen + c, s + c * v, ways * math.comb(counts[v], c))

    walk(0, 0, 0, 1)
    p = Fraction(ways_ge, math.comb(n1 + n3, n1))
    point = Fraction(sum(a), 6 * n1) - Fraction(sum(b), 6 * n3)
    return {"point": float(point), "p": float(p), "n1": n1, "n3": n3, "total_sixths": total}


def level_check(h1: Mapping[str, Fraction | float], h3: Mapping[str, Fraction | float], *, replicates: int = 10_000,
                seed: int = 0) -> dict[str, Any] | None:
    """{"bootstrap", "permutation", "valid"}；任一級別不足 TASKS_PER_LEVEL 題 → None（無法計算）。"""
    if len(h1) < TASKS_PER_LEVEL or len(h3) < TASKS_PER_LEVEL:
        return None
    boot = bootstrap_difference(dict(h1), dict(h3), replicates=replicates, seed=seed)
    perm = exact_permutation(h1, h3)
    return {"bootstrap": boot, "permutation": perm, "valid": bool(boot["low_positive"]) and perm["p"] < ALPHA}


def _sum_distribution(values: Sequence[int]) -> list[float]:
    """以 values（0–6 的整數）有放回抽 len(values) 次的總和分布（精確）。"""
    n = len(values)
    step = [values.count(v) / n for v in range(7)]
    dist = [1.0]
    for _ in range(n):
        out = [0.0] * (len(dist) + 6)
        for s, ps in enumerate(dist):
            if ps:
                for v, pv in enumerate(step):
                    if pv:
                        out[s + v] += ps * pv
        dist = out
    return dist


def bootstrap_low_positive_exact(h1: Sequence[int], h3: Sequence[int]) -> bool:
    """B → ∞ 的 bootstrap：P(H1 平均 − H3 平均 ≤ 0) < 0.025（h1、h3 為 6 倍的整數值）。"""
    da, db = _sum_distribution(h1), _sum_distribution(h3)
    n1, n3 = len(h1), len(h3)
    at_most_zero = 0.0
    for sa, pa in enumerate(da):
        if not pa:
            continue
        for sb, pb in enumerate(db):
            if pb and sa * n3 <= sb * n1:
                at_most_zero += pa * pb
    return at_most_zero < ALPHA


def _task_values(rng: random.Random, p: float, kappa: float, n_tasks: int, rollouts: int = 3) -> list[int]:
    values = []
    for _ in range(n_tasks):
        q = 1.0 if p >= 1.0 else rng.betavariate(p * kappa, (1 - p) * kappa)
        k = sum(rng.random() < q for _ in range(rollouts))
        values.append(k * 6 // rollouts)
    return values


def rq2_power(*, reps: int = 1000, seed: int = 0, n_tasks: int = TASKS_PER_LEVEL) -> list[dict[str, Any]]:
    """分級檢定的偵測力（兩個檢定同時成立）；只列 p₁ = p₃ + Δ ≤ 1 的組合。"""
    rows = []
    for kappa in POWER_KAPPA:
        for p3 in POWER_P3:
            for delta in POWER_DELTA:
                p1 = round(p3 + delta, 10)
                if p1 > 1:
                    continue
                rng = random.Random(f"study3/rq2-power|{seed}|{kappa}|{p3}|{delta}")
                both = boot = perm = 0
                for _ in range(reps):
                    a, b = _task_values(rng, p1, kappa, n_tasks), _task_values(rng, p3, kappa, n_tasks)
                    ok_boot = bootstrap_low_positive_exact(a, b)
                    ok_perm = exact_permutation({str(i): Fraction(v, 6) for i, v in enumerate(a)},
                                                {str(i): Fraction(v, 6) for i, v in enumerate(b)})["p"] < ALPHA
                    boot += ok_boot
                    perm += ok_perm
                    both += ok_boot and ok_perm
                rows.append({"kappa": kappa, "p1": p1, "p3": p3, "delta": delta, "reps": reps, "power": both / reps,
                             "bootstrap_only": boot / reps, "permutation_only": perm / reps})
    return rows


# ---------------------------------------------------------------- sweep 的成本


def source_cost(length: int, rng: random.Random, plan: SweepPlan = REGISTERED_PLAN) -> int:
    """一條長度 length 的失敗 source 的 sweep 呼叫數（起點 1.0，cliff 與 linear 各半）。"""
    if not 2 <= length <= MAX_TURNS:
        raise ValueError(f"length must be within 2–{MAX_TURNS}")
    if rng.random() < 0.5:
        curve = cliff_curve(length, (length - 1) // 2, 1.0, 0.0)
    else:
        curve = linear_curve(length, 1.0, 0.0)

    def rollout(k: int, i: int) -> RolloutOutcome:
        if rng.random() < curve[k]:
            return RolloutOutcome(k, i, "success", "final_answer", False, 1)
        return RolloutOutcome(k, i, "failure", "final_answer", False, 1)

    protocol = run_protocol(length - 1, plan, rollout)
    return sum((length - k) if o.outcome == "success" else (MAX_TURNS - k) for k, outs in protocol.outcomes.items() for o in outs)


def quantile(values: Sequence[float], q: float) -> float:
    """nearest-rank 分位數。"""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(math.ceil(q * len(ordered)) - 1, 0))]


def cost_samples(length: int, *, reps: int = COST_REPS, seed: int = 0) -> list[int]:
    rng = random.Random(f"study3/sweep-cost|{seed}|{length}")
    return [source_cost(length, rng) for _ in range(reps)]


def reserve_calls(*, reps: int = COST_REPS, seed: int = 0) -> int:
    """預留額（呼叫數）= L = 20 時單條成本的 P99。"""
    return int(quantile(cost_samples(RESERVE_LENGTH, reps=reps, seed=seed), 0.99))


def estimate(level_mean_turns: Mapping[str, float], *, agent: str, haiku_call_usd_p90: float | None = None,
             haiku_dev_spent_usd: float = 0.0, haiku_baseline_p90_usd: float = 0.0, reps: int = COST_REPS,
             seed: int = 0) -> dict[str, Any]:
    """dev 之後的估算（decision 036 §7），每個 agent 各自計算。level_mean_turns：該 agent 在 dev 各級別的平均 turn。
    agent = "local"：n = floor(80,000 ÷ c90)；agent = "haiku"：n = floor(剩下可用於 sweep 的金額 ÷ (c90 × 每次呼叫成本的 P90))。
    ask_user = n < 15。"""
    if agent not in ("local", "haiku"):
        raise ValueError("agent must be 'local' or 'haiku'")
    if not level_mean_turns:
        raise ValueError("level_mean_turns must not be empty")
    length = min(MAX_TURNS, max(2, math.ceil(max(level_mean_turns.values()))))
    samples = cost_samples(length, reps=reps, seed=seed)
    c90 = quantile(samples, 0.9)
    out: dict[str, Any] = {"agent": agent, "length": length, "mean_calls": sum(samples) / len(samples), "c90_calls": c90}
    if agent == "local":
        out["n"] = math.floor(SWEEP_CAP_CALLS / c90)
    else:
        if haiku_call_usd_p90 is None or haiku_call_usd_p90 <= 0:
            raise ValueError("the Haiku estimate needs the measured P90 cost per call")
        c90_usd = c90 * haiku_call_usd_p90
        left = PAID_TOTAL_USD - SAFETY_MARGIN_USD - haiku_dev_spent_usd - haiku_baseline_p90_usd
        out.update({"c90_usd": c90_usd, "sweep_usd_left": left, "n": max(0, math.floor(left / c90_usd))})
    out["ask_user"] = out["n"] < ASK_BELOW
    return out
