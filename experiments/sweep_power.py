"""SweepPlan 的 synthetic 偵測力模擬（decision log 011）。

以已知的 P(success | k) 曲線取代 agent：checkpoint k 的每個 rollout ~ Bernoulli(curve[k])（``rng.random() < curve[k]``）。
排程（coarse → 反覆 refine）、估計與 cliff 判定一律呼叫 ``replay.localization`` 中 sweep() 使用的同一組函式。

執行：``uv run python -m experiments.sweep_power``（只用 CPU、數分鐘內完成；不呼叫任何 LLM）。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence

from replay.localization import (
    CheckpointEstimate,
    RolloutOutcome,
    SweepPlan,
    detect_cliff,
    estimate_checkpoint,
    run_protocol,
    CLIFF_CRITERIA,
    DEFAULT_CLIFF_CRITERION,
)

REPS = 400
SEED = 0
CLIFF_DETECT_MIN = 0.80
FLAG_MAX = 0.05


@dataclass(frozen=True)
class Scenario:
    """kind ∈ {"cliff", "gradual", "flat"}；curve[k] ∈ [0, 1]、len(curve) >= 1；cliff 情境的 cliff_k = m（0 <= m < len − 1），
    其他情境 cliff_k = None。違反 → ValueError。
    """

    name: str
    kind: str
    curve: tuple[float, ...]
    cliff_k: int | None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        _check_curve(self.curve)
        if self.kind == "cliff":
            if type(self.cliff_k) is not int or not 0 <= self.cliff_k < len(self.curve) - 1:
                raise ValueError("a cliff scenario needs 0 <= cliff_k < len(curve) - 1")
        elif self.cliff_k is not None:
            raise ValueError("only cliff scenarios have cliff_k")


KINDS = ("cliff", "gradual", "flat")


def _check_curve(curve: Sequence[float]) -> None:
    if len(curve) == 0:
        raise ValueError("curve must not be empty")
    for p in curve:
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise ValueError("curve values must be probabilities in [0, 1]")


@dataclass(frozen=True)
class SimulatedSweep:
    estimates: tuple[CheckpointEstimate, ...]  # 依 k 遞增
    refined_intervals: tuple[tuple[int, int], ...]
    largest_drop: float
    largest_drop_k: int | None
    is_cliff: bool
    rollouts: int  # 總 rollout 數


@dataclass(frozen=True)
class PowerRow:
    scenario: str
    kind: str
    plan: SweepPlan
    reps: int
    detect_rate: float  # is_cliff 且 largest_drop_k == cliff_k 的比例；cliff_k 為 None → 0.0
    flag_rate: float  # is_cliff 的比例
    mean_rollouts: float


def cliff_curve(length: int, m: int, p_pre: float, p_post: float) -> tuple[float, ...]:
    """curve[k] = p_pre（k <= m）、p_post（k > m）；0 <= m < length − 1、機率在 [0, 1]，否則 ValueError。"""
    if type(length) is not int or type(m) is not int or not 0 <= m < length - 1:
        raise ValueError("require 0 <= m < length - 1")
    curve = tuple(p_pre if k <= m else p_post for k in range(length))
    _check_curve(curve)
    return curve


def linear_curve(length: int, start: float, end: float) -> tuple[float, ...]:
    """由 start 線性到 end（length == 1 時為 (start,)）。"""
    if type(length) is not int or length < 1:
        raise ValueError("length must be an int >= 1")
    if length == 1:
        curve: tuple[float, ...] = (start,)
    else:
        curve = tuple(start + (end - start) * k / (length - 1) for k in range(length))
    _check_curve(curve)
    return curve


def default_scenarios() -> tuple[Scenario, ...]:
    """decision 011 第 2 點的情境（順序固定）。"""
    scenarios: list[Scenario] = []
    for p_pre, p_post in CLIFF_LEVELS:
        for length in CLIFF_LENGTHS:
            for m in sorted({0, (length - 1) // 2, length - 2}):
                scenarios.append(
                    Scenario(f"cliff-{p_pre}-{p_post}-L{length}-m{m}", "cliff", cliff_curve(length, m, p_pre, p_post), m)
                )
    for length in GRADUAL_LENGTHS:
        scenarios.append(Scenario(f"gradual-0.9-0.1-L{length}", "gradual", linear_curve(length, 0.9, 0.1), None))
    for level in FLAT_LEVELS:
        for length in FLAT_LENGTHS:
            scenarios.append(Scenario(f"flat-{level}-L{length}", "flat", (level,) * length, None))
    return tuple(scenarios)


CLIFF_LEVELS = ((0.9, 0.1), (0.8, 0.2), (0.7, 0.2), (0.6, 0.1))
CLIFF_LENGTHS = (4, 8, 12)
GRADUAL_LENGTHS = (8, 12)
FLAT_LEVELS = (0.5, 0.9)
FLAT_LENGTHS = (4, 8, 12)


def candidate_plans() -> tuple[SweepPlan, ...]:
    """decision 011 第 3 點的候選 plan（依 (stride, n_coarse, n_fine, threshold) 字典序）。"""
    return tuple(
        SweepPlan(stride, n_coarse, n_fine, threshold)
        for stride in (1, 2, 3)
        for n_coarse in (4, 6, 8, 10)
        for n_fine in (16, 24, 32)
        for threshold in (0.25, 0.3, 0.4)
        if n_fine >= n_coarse
    )


def simulate_sweep(
    curve: Sequence[float], plan: SweepPlan, rng: random.Random, *, criterion: str = DEFAULT_CLIFF_CRITERION
) -> SimulatedSweep:
    """以 sweep() 使用的同一個 ``run_protocol``（coarse → 反覆 refine，decision 013）排程；rollout i 於 checkpoint k 的結果為
    ``rng.random() < curve[k]``，依排程順序抽取。之後以同一組 estimate_checkpoint / detect_cliff 判定。last = len(curve) − 1。
    criterion：驟降判定（replay.localization.CLIFF_CRITERIA，decision 015），只影響判定、不影響排程。
    """
    _check_curve(curve)
    if criterion not in CLIFF_CRITERIA:
        raise ValueError(f"criterion must be one of {CLIFF_CRITERIA}")

    def rollout(k: int, i: int) -> RolloutOutcome:
        label = "success" if rng.random() < curve[k] else "failure"
        return RolloutOutcome(k, i, label, "final_answer", False, 1)

    protocol = run_protocol(len(curve) - 1, plan, rollout)
    estimates = tuple(estimate_checkpoint(k, list(protocol.outcomes[k])) for k in sorted(protocol.outcomes))
    largest_drop, largest_drop_k, is_cliff = detect_cliff(estimates, plan.drop_threshold, criterion=criterion)
    return SimulatedSweep(
        estimates=estimates,
        refined_intervals=protocol.refined_intervals,
        largest_drop=largest_drop,
        largest_drop_k=largest_drop_k,
        is_cliff=is_cliff,
        rollouts=len(protocol.order),
    )


def power(
    scenario: Scenario, plan: SweepPlan, *, reps: int = REPS, seed: int = SEED, criterion: str = DEFAULT_CLIFF_CRITERION
) -> PowerRow:
    """以 ``random.Random(f"{seed}|{scenario.name}|{plan}")`` 重複 reps 次 simulate_sweep（reps 為 int >= 1，否則 ValueError）。"""
    if type(reps) is not int or reps < 1:
        raise ValueError("reps must be an int >= 1")
    rng = random.Random(f"{seed}|{scenario.name}|{plan}")
    detected = flagged = rollouts = 0
    for _ in range(reps):
        result = simulate_sweep(scenario.curve, plan, rng, criterion=criterion)
        flagged += result.is_cliff
        detected += result.is_cliff and scenario.cliff_k is not None and result.largest_drop_k == scenario.cliff_k
        rollouts += result.rollouts
    return PowerRow(scenario.name, scenario.kind, plan, reps, detected / reps, flagged / reps, rollouts / reps)


def select_plan(rows: Sequence[PowerRow]) -> SweepPlan | None:
    """decision 011 第 4 點：合格（cliff detect >= CLIFF_DETECT_MIN 且非 cliff flag <= FLAG_MAX）者中，
    最大 mean_rollouts 最小 → 平均 mean_rollouts 最小 → plan 欄位字典序最小。沒有合格者 → None。
    只考慮在 rows 中出現的 plan；每個 plan 必須涵蓋 rows 中出現的所有情境，否則 ValueError。
    """
    by_plan: dict[SweepPlan, list[PowerRow]] = {}
    for row in rows:
        by_plan.setdefault(row.plan, []).append(row)
    scenarios = {row.scenario for row in rows}
    ranked = []
    for plan, plan_rows in by_plan.items():
        if {row.scenario for row in plan_rows} != scenarios or len(plan_rows) != len(scenarios):
            raise ValueError(f"plan {plan} does not cover every scenario exactly once")
        qualified = all(
            row.detect_rate >= CLIFF_DETECT_MIN if row.kind == "cliff" else row.flag_rate <= FLAG_MAX for row in plan_rows
        )
        if qualified:
            costs = [row.mean_rollouts for row in plan_rows]
            ranked.append(
                ((max(costs), math.fsum(costs) / len(costs), (plan.coarse_stride, plan.n_coarse, plan.n_fine, plan.drop_threshold)), plan)
            )
    if not ranked:
        return None
    return min(ranked, key=lambda item: item[0])[1]


EXPLORATION_PLANS = tuple(
    SweepPlan(stride, n_coarse, n_fine, threshold)
    for stride in (2, 3)
    for n_coarse in (8, 10)
    for n_fine in (48, 64)
    for threshold in (0.25, 0.3)
)


def _summary_rows(rows: Sequence[PowerRow], plans: Sequence[SweepPlan]) -> list[str]:
    lines = ["| plan | min cliff detect | max non-cliff flag | max mean rollouts | qualified |", "|---|---|---|---|---|"]
    for plan in plans:
        plan_rows = [row for row in rows if row.plan == plan]
        detect = min(row.detect_rate for row in plan_rows if row.kind == "cliff")
        flag = max(row.flag_rate for row in plan_rows if row.kind != "cliff")
        cost = max(row.mean_rollouts for row in plan_rows)
        ok = detect >= CLIFF_DETECT_MIN and flag <= FLAG_MAX
        label = f"{plan.coarse_stride},{plan.n_coarse},{plan.n_fine},{plan.drop_threshold}"
        lines.append(f"| {label} | {detect:.3f} | {flag:.3f} | {cost:.1f} | {'yes' if ok else ''} |")
    return lines


def main() -> None:  # pragma: no cover - 由 CLI 執行
    scenarios = default_scenarios()
    rows = [power(scenario, plan) for plan in candidate_plans() for scenario in scenarios]
    chosen = select_plan(rows)
    print(json.dumps({"reps": REPS, "seed": SEED, "selected": None if chosen is None else asdict(chosen)}, indent=2))
    print("\n## Pre-registered grid, decision 010 criterion (decision 011 §3–4)\n")
    print("\n".join(_summary_rows(rows, candidate_plans())))
    exploratory = [
        power(scenario, plan, criterion="bonferroni95")
        for plan in candidate_plans() + EXPLORATION_PLANS
        for scenario in scenarios
    ]
    print("\n## Option A (decision 011 exploration, selectable since decision 015): bonferroni95, grid + n_fine 48 / 64\n")
    print("\n".join(_summary_rows(exploratory, candidate_plans() + EXPLORATION_PLANS)))
    best = select_plan(exploratory)
    print(f"\nexploration select_plan: {None if best is None else asdict(best)}")


if __name__ == "__main__":  # pragma: no cover
    main()
