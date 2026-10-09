"""SPEC §6 failure localization：checkpoint rerun sweep 與 P(success | checkpoint) 估計（decision log 010）。

流程：對 source trajectory 的合法 checkpoint 做 coarse 掃描 → 對相鄰點下降 ≥ drop_threshold 的區間加密取樣 → 估計與 cliff 判定。
每個 rollout 都經由 ``replay.aborts.run_logged``（manifest）執行；排程以 round-robin 交錯 checkpoint。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping

from agents.policy import Policy
from replay.aborts import MAX_EPISODE_ATTEMPTS, RolloutKey, run_with_retries
from replay.checkpoint import Checkpoint, make_checkpoint, rerun
from replay.runner import Clock
from replay.stats import wilson_interval
from replay.trajectory import Trajectory
from tasks.schema import Task

ROLLOUT_OUTCOMES: tuple[str, ...] = ("success", "failure", "refusal", "agent_error", "aborted")

# 驟降判定（decision 015）：wilson95 = decision 010（兩點的 95% Wilson CI 不重疊）；bonferroni95 = decision 011 的選項 A
# （Wilson CI 以 z = Φ⁻¹(1 − 0.025 / m) 計算，m = 相鄰已估計點的對數）。正式實驗使用哪一個屬於使用者決策；預設維持 decision 010。
CLIFF_CRITERIA: tuple[str, ...] = ("wilson95", "bonferroni95")
DEFAULT_CLIFF_CRITERION = "wilson95"


@dataclass(frozen=True)
class SweepPlan:
    """coarse_stride、n_coarse、n_fine 為 int >= 1（bool 不算）且 n_fine >= n_coarse；0 < drop_threshold <= 1；否則 ValueError。"""

    coarse_stride: int
    n_coarse: int
    n_fine: int
    drop_threshold: float

    def __post_init__(self) -> None:
        for name in ("coarse_stride", "n_coarse", "n_fine"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"SweepPlan.{name} must be an int >= 1")
        if self.n_fine < self.n_coarse:
            raise ValueError("SweepPlan.n_fine must be >= n_coarse")
        threshold = self.drop_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError("SweepPlan.drop_threshold must satisfy 0 < drop_threshold <= 1")


DEFAULT_SYNTHETIC_PLAN = SweepPlan(coarse_stride=2, n_coarse=8, n_fine=24, drop_threshold=0.3)


@dataclass(frozen=True)
class RolloutOutcome:
    k: int
    rollout_index: int
    outcome: str  # ROLLOUT_OUTCOMES
    termination: str | None  # 有 trajectory 時
    stopped_without_action: bool  # final_answer、grade().success 為 False、續跑（index >= k 的 turns）沒有任何 mutation
    attempts: int  # 用掉的嘗試次數


@dataclass(frozen=True)
class CheckpointEstimate:
    k: int
    n: int
    successes: int
    failures: int
    refusals: int
    agent_errors: int
    aborted: int
    max_turns_terminations: int
    stopped_without_action: int
    p_hat: float  # successes / n（n == 0 時 0.0）
    p_hat_completed: float | None  # successes / (successes + failures)；分母 0 時 None
    ci_low: float  # Wilson(successes, n)
    ci_high: float


@dataclass(frozen=True)
class SweepResult:
    task_id: str
    source_trajectory_id: str
    plan: SweepPlan
    estimates: tuple[CheckpointEstimate, ...]  # 依 k 遞增；只含有 rollout 的 checkpoint
    rollouts: tuple[RolloutOutcome, ...]  # 依執行順序
    refined_intervals: tuple[tuple[int, int], ...]  # 所有 refine 輪次標記過的區間（decision 013 §2），去重、依 a 遞增
    largest_drop: float  # 相鄰已估計 checkpoint 的最大 p̂(k) − p̂(k')；少於兩點時 0.0
    largest_drop_k: int | None  # 最大下降的前一點 k
    is_cliff: bool  # assess_drop 的 is_cliff：下降合格（門檻 + CI）且非 max_turns 混淆（decision 006 / 017 §1）
    criterion: str = DEFAULT_CLIFF_CRITERION  # CLIFF_CRITERIA 之一（decision 015）
    max_turns_confounded: bool = False  # 下降合格但 max_turns 比例上升 → 不認定為 cliff（decision 006 / 017 §1）


def valid_checkpoints(trajectory: Trajectory) -> list[int]:
    """[0, len(turns) − 1]：decision 006 允許建立 checkpoint 的所有 k（trajectory 先 verify）。"""
    trajectory.verify()
    return list(range(len(trajectory.turns)))


def coarse_checkpoints(last: int, stride: int) -> list[int]:
    """{0, stride, 2·stride, …, ≤ last} ∪ {last}，遞增、不重複；last >= 0、stride >= 1（int，bool 不算），否則 ValueError。"""
    if type(last) is not int or last < 0:
        raise ValueError("last must be an int >= 0")
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be an int >= 1")
    return sorted(set(range(0, last + 1, stride)) | {last})


def classify_rollout(trajectory: Trajectory, k: int) -> tuple[str, bool]:
    """回傳 (outcome, stopped_without_action)：最後一個 turn 的 stop_reason == "refusal" → "refusal"；
    否則 grade().success → "success"，不然 "failure"。stopped_without_action 定義見 RolloutOutcome
    （mutation = changes 非空的 tool step）。0 <= k < len(turns)（int），否則 ValueError。
    """
    if type(k) is not int or not 0 <= k < len(trajectory.turns):
        raise ValueError("k must be an int with 0 <= k < len(turns)")
    success = trajectory.grade().success
    if trajectory.turns[-1].message.stop_reason == "refusal":
        outcome = "refusal"
    else:
        outcome = "success" if success else "failure"
    mutated = any(step.changes for turn in trajectory.turns[k:] for step in turn.tool_steps)
    stopped = trajectory.termination == "final_answer" and not success and not mutated
    return outcome, stopped


def estimate_checkpoint(k: int, outcomes: list[RolloutOutcome]) -> CheckpointEstimate:
    """彙總同一個 k 的 rollouts（全部的 outcome.k 必須等於 k，否則 ValueError）。"""
    for outcome in outcomes:
        if not isinstance(outcome, RolloutOutcome) or outcome.k != k:
            raise ValueError(f"every outcome must be a RolloutOutcome for checkpoint {k}")
        if outcome.outcome not in ROLLOUT_OUTCOMES:
            raise ValueError(f"unknown rollout outcome {outcome.outcome!r}")
    counts = {name: sum(1 for o in outcomes if o.outcome == name) for name in ROLLOUT_OUTCOMES}
    n = len(outcomes)
    successes, failures = counts["success"], counts["failure"]
    low, high = wilson_interval(successes, n)
    return CheckpointEstimate(
        k=k,
        n=n,
        successes=successes,
        failures=failures,
        refusals=counts["refusal"],
        agent_errors=counts["agent_error"],
        aborted=counts["aborted"],
        max_turns_terminations=sum(1 for o in outcomes if o.termination == "max_turns"),
        stopped_without_action=sum(1 for o in outcomes if o.stopped_without_action),
        p_hat=successes / n if n else 0.0,
        p_hat_completed=successes / (successes + failures) if successes + failures else None,
        ci_low=low,
        ci_high=high,
    )


def _exact_p_hat(estimate: CheckpointEstimate) -> Fraction:
    """p̂ 的精確值 successes / n（n == 0 時 0）。"""
    return Fraction(estimate.successes, estimate.n) if estimate.n else Fraction(0)


def _meets_threshold(drop: Fraction, drop_threshold: float) -> bool:
    """``drop >= drop_threshold`` 的精確判定：有限的 drop_threshold 依其十進位表示（repr）取值，0.3 即 3/10，
    而非二進位近似；因此 7/10 − 4/10 >= 0.3、11/24 − 5/24 >= 0.25 成立（浮點減法會得到略小於門檻的值）。
    int 直接精確比較；非有限值沿用 float 的比較語意（nan、+inf 永不成立）。
    """
    if not isinstance(drop_threshold, float) or not math.isfinite(drop_threshold):
        return drop >= drop_threshold
    return drop >= Fraction(repr(float(drop_threshold)))


@dataclass(frozen=True)
class DropAssessment:
    largest_drop: float  # 相鄰已估計 checkpoint 的最大 p̂(k) − p̂(k')；少於兩點時 0.0
    largest_drop_k: int | None  # 最大下降的前一點 k
    separated: bool  # largest_drop >= drop_threshold 且兩點的 CI（依 criterion）不重疊
    max_turns_confounded: bool  # separated 且後一點的 max_turns 比例嚴格大於前一點（decision 006 / 017 §1）
    is_cliff: bool  # separated 且非 max_turns_confounded


def _max_turns_rate(estimate: CheckpointEstimate) -> Fraction:
    return Fraction(estimate.max_turns_terminations, estimate.n) if estimate.n else Fraction(0)


def assess_drop(
    estimates: tuple[CheckpointEstimate, ...], drop_threshold: float, *, criterion: str = DEFAULT_CLIFF_CRITERION
) -> DropAssessment:
    """最大下降的判定（decision 010 / 013 / 015 / 017 §1）。同大小時取 k 較小者；只看最大下降（被 max_turns 混淆時
    不改看其他下降）。estimates 必須依 k 嚴格遞增、criterion 必須在 CLIFF_CRITERIA 中，否則 ValueError。
    separated：wilson95 → 兩點的 ci_low > ci_high（95% Wilson）；bonferroni95 → 以 z = Φ⁻¹(1 − 0.025 / m)（m = len(estimates) − 1）
    重算兩點的 Wilson 區間後 low > high。兩點時兩者相同。
    下降與 max_turns 比例都以精確有理數計算、比較（同大小與 ``>= drop_threshold`` 都不受浮點捨入影響，見 _meets_threshold）；
    largest_drop 回傳該精確值最接近的 float。少於兩點 → DropAssessment(0.0, None, False, False, False)。
    """
    if criterion not in CLIFF_CRITERIA:
        raise ValueError(f"criterion must be one of {CLIFF_CRITERIA}")
    ks = [estimate.k for estimate in estimates]
    if any(b <= a for a, b in zip(ks, ks[1:])):
        raise ValueError("estimates must be sorted by strictly increasing k")
    if len(estimates) < 2:
        return DropAssessment(0.0, None, False, False, False)
    exact = [_exact_p_hat(estimate) for estimate in estimates]
    drops = [before - after for before, after in zip(exact, exact[1:])]
    best_index = 0
    for index, drop in enumerate(drops):
        if drop > drops[best_index]:
            best_index = index
    best_drop = drops[best_index]
    before, after = estimates[best_index], estimates[best_index + 1]
    if criterion == "wilson95":
        ci_separated = before.ci_low > after.ci_high
    else:
        z = NormalDist().inv_cdf(1 - 0.025 / (len(estimates) - 1))
        ci_separated = wilson_interval(before.successes, before.n, z)[0] > wilson_interval(after.successes, after.n, z)[1]
    separated = _meets_threshold(best_drop, drop_threshold) and ci_separated
    confounded = separated and _max_turns_rate(after) > _max_turns_rate(before)
    return DropAssessment(float(best_drop), before.k, separated, confounded, separated and not confounded)


def detect_cliff(
    estimates: tuple[CheckpointEstimate, ...], drop_threshold: float, *, criterion: str = DEFAULT_CLIFF_CRITERION
) -> tuple[float, int | None, bool]:
    """``assess_drop`` 的 (largest_drop, largest_drop_k, is_cliff)；is_cliff 已排除 max_turns 混淆（decision 017 §1）。"""
    assessed = assess_drop(estimates, drop_threshold, criterion=criterion)
    return assessed.largest_drop, assessed.largest_drop_k, assessed.is_cliff


def pending_intervals(estimates: Mapping[int, CheckpointEstimate], plan: SweepPlan) -> tuple[tuple[int, int], ...]:
    """decision 013 §2 的待加密區間：依 k 排序的相鄰已估計點 (a, b) 中，精確下降 p̂(a) − p̂(b) >= plan.drop_threshold
    （見 _meets_threshold），且 [a, b] 內有任何 checkpoint 的 rollout 數 < plan.n_fine（從未估計的中間點視為 0）者，依 a 遞增。
    plan 必須是 SweepPlan、estimates 的每個 key 必須等於其 estimate.k；否則 ValueError。
    """
    if not isinstance(plan, SweepPlan):
        raise ValueError("plan must be a SweepPlan")
    for k, estimate in estimates.items():
        if not isinstance(estimate, CheckpointEstimate) or estimate.k != k:
            raise ValueError("estimates must map each k to its CheckpointEstimate")
    ks = sorted(estimates)
    intervals = []
    for a, b in zip(ks, ks[1:]):
        drop = _exact_p_hat(estimates[a]) - _exact_p_hat(estimates[b])
        if not _meets_threshold(drop, plan.drop_threshold):
            continue
        refined = b == a + 1 and estimates[a].n >= plan.n_fine and estimates[b].n >= plan.n_fine
        if not refined:
            intervals.append((a, b))
    return tuple(intervals)


@dataclass(frozen=True)
class ProtocolRun:
    outcomes: dict[int, tuple[RolloutOutcome, ...]]  # k → 該點的 rollouts（依 rollout_index）
    order: tuple[RolloutOutcome, ...]  # 依執行順序
    refined_intervals: tuple[tuple[int, int], ...]  # 所有輪次標記過的區間，去重、依 a 遞增


def run_protocol(last: int, plan: SweepPlan, rollout: Callable[[int, int], RolloutOutcome]) -> ProtocolRun:
    """decision 010 / 013 的掃描排程（sweep() 與 decision 011 的偵測力模擬共用）。

    1. coarse：``coarse_checkpoints(last, plan.coarse_stride)`` 的每個點目標 n_coarse。
    2. 反覆：``pending_intervals(目前所有估計, plan)`` 非空 → 區間內（含兩端）所有 checkpoint 的目標提高到 n_fine；否則結束。
    每個階段 round-robin：第 i 輪依 k 遞增，對「目前數量 == i 且 i < 目標數」的 k 呼叫 ``rollout(k, i)``。
    rollout 回傳的 RolloutOutcome 的 (k, rollout_index) 必須等於 (k, i)，否則 ValueError；rollout 的例外原樣往外拋。
    plan 必須是 SweepPlan、last 為 int >= 0，否則 ValueError。
    """
    if not isinstance(plan, SweepPlan):
        raise ValueError("plan must be a SweepPlan")
    if type(last) is not int or last < 0:
        raise ValueError("last must be an int >= 0")
    outcomes: dict[int, list[RolloutOutcome]] = {}
    order: list[RolloutOutcome] = []

    def run_round_robin(targets: dict[int, int]) -> None:
        for i in range(max(targets.values(), default=0)):
            for k in sorted(targets):
                if len(outcomes.get(k, [])) == i and i < targets[k]:
                    result = rollout(k, i)
                    if not isinstance(result, RolloutOutcome) or (result.k, result.rollout_index) != (k, i):
                        raise ValueError(f"rollout({k}, {i}) returned an outcome for a different rollout")
                    outcomes.setdefault(k, []).append(result)
                    order.append(result)

    run_round_robin({k: plan.n_coarse for k in coarse_checkpoints(last, plan.coarse_stride)})
    flagged: set[tuple[int, int]] = set()
    while True:
        estimates = {k: estimate_checkpoint(k, rollouts) for k, rollouts in outcomes.items()}
        intervals = pending_intervals(estimates, plan)
        if not intervals:
            break
        flagged.update(intervals)
        run_round_robin({k: plan.n_fine for a, b in intervals for k in range(a, b + 1)})
    return ProtocolRun(
        outcomes={k: tuple(outcomes[k]) for k in sorted(outcomes)},
        order=tuple(order),
        refined_intervals=tuple(sorted(flagged)),
    )


def check_prior(prior: Any) -> dict[tuple[int, int], RolloutOutcome]:
    """decision 023 §1：None → {}；否則必須是 (k, rollout_index)（int >= 0）→ RolloutOutcome 的 Mapping，outcome 的 (k, rollout_index)
    與 key 相同、outcome 在 ROLLOUT_OUTCOMES 中、attempts 在 1..MAX_EPISODE_ATTEMPTS；否則 ValueError。回傳可修改的 dict 複本。"""
    if prior is None:
        return {}
    if not isinstance(prior, Mapping):
        raise ValueError("prior must map (k, rollout_index) to RolloutOutcome")
    checked: dict[tuple[int, int], RolloutOutcome] = {}
    for key, outcome in prior.items():
        if not isinstance(key, tuple) or len(key) != 2 or not all(type(x) is int and x >= 0 for x in key):
            raise ValueError("prior keys must be (k, rollout_index) pairs of ints >= 0")
        if not isinstance(outcome, RolloutOutcome) or (outcome.k, outcome.rollout_index) != key:
            raise ValueError("each prior value must be the RolloutOutcome of its (k, rollout_index)")
        if outcome.outcome not in ROLLOUT_OUTCOMES or type(outcome.attempts) is not int or not 1 <= outcome.attempts <= MAX_EPISODE_ATTEMPTS:
            raise ValueError("a prior outcome has an unknown label or an invalid attempt count")
        checked[key] = outcome
    return checked


def sweep(
    source: Trajectory,
    policy_factory: Callable[[int, int, int], Policy],
    *,
    plan: SweepPlan,
    log_path: str | Path,
    clock: Clock = time.perf_counter,
    criterion: str = DEFAULT_CLIFF_CRITERION,
    rollout_dir: str | Path | None = None,
    prior: Mapping[tuple[int, int], RolloutOutcome] | None = None,
) -> SweepResult:
    """對 source 執行 coarse → refine 掃描。

    - ``policy_factory(k, rollout_index, attempt)`` 產生該次嘗試的 policy；其 agent_config.config_id 必須等於 source 的
      （否則 ValueError，不執行）。
    - 每個 (k, rollout_index) 以 ``replay.aborts.run_with_retries(lambda p: rerun(cp, p, clock=clock), ...)`` 執行
      （decision 009 的重試規則）：完成 → 以 classify_rollout 分類；未完成 → infrastructure → "aborted"、
      其他（agent_output）→ "agent_error"。harness 例外往外拋。
    - 排程與 refine 一律由 ``run_protocol(last, plan, ...)`` 決定（coarse → 反覆 refine，decision 013 §2）。
    - run_logged 的事前檢查錯誤（例如 manifest 已有相同的嘗試）原樣往外拋。
    - 驟降判定依 criterion（CLIFF_CRITERIA，decision 015），在執行任何 rollout 之前檢查（不合法 → ValueError）。
      largest_drop / largest_drop_k / is_cliff / max_turns_confounded 取自 ``assess_drop``（decision 017 §1）。
    - rollout_dir（decision 017 §2）：給定時，每個完成的 rerun trajectory 存成 ``rollout_dir/k<k>-r<i>.json``（先存檔再記帳，
      decision 023 §2）；必須不存在或為空目錄（否則 ValueError，不執行），不存在時建立。
    - prior（decision 023 §1，續跑）：(k, rollout_index) → 先前目錄中已有最終結果的 RolloutOutcome（``check_prior``）。重放排程時，
      prior 中的 (k, i) 直接採用、不執行；prior 必須剛好是排程的前綴：排程要求 prior 以外的 (k, i) 而 prior 還有剩 → 在執行之前
      ValueError；排程結束時 prior 還有剩 → ValueError。
    """
    if not isinstance(plan, SweepPlan):
        raise ValueError("plan must be a SweepPlan")
    if criterion not in CLIFF_CRITERIA:
        raise ValueError(f"criterion must be one of {CLIFF_CRITERIA}")
    remaining = check_prior(prior)
    save_dir = None if rollout_dir is None else Path(rollout_dir)
    if save_dir is not None and save_dir.exists() and (not save_dir.is_dir() or any(save_dir.iterdir())):
        raise ValueError(f"{save_dir} already exists and is not an empty directory")
    last = valid_checkpoints(source)[-1]
    task = Task.from_dict(source.task)
    config_id = source.agent.config_id
    source_id = source.trajectory_id
    checkpoints: dict[int, Checkpoint] = {}

    def run_one(k: int, rollout_index: int) -> RolloutOutcome:
        if (k, rollout_index) in remaining:
            return remaining.pop((k, rollout_index))
        if remaining:
            raise ValueError("the carried-over rollouts are not a prefix of this sweep's schedule (decision 023 §1)")
        if k not in checkpoints:
            checkpoints[k] = make_checkpoint(source, k)
        checkpoint = checkpoints[k]

        def factory(attempt: int) -> Policy:
            policy = policy_factory(k, rollout_index, attempt)
            if policy.agent_config.config_id != config_id:
                raise ValueError("policy_factory produced a policy whose config_id differs from the source agent")
            return policy

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        retried = run_with_retries(
            lambda p: rerun(checkpoint, p, clock=clock),
            factory,
            task=task,
            key=RolloutKey(checkpoint.checkpoint_id, k, source_id, rollout_index, 1),
            max_turns=checkpoint.max_turns,
            log_path=log_path,
            save_path=None if save_dir is None else save_dir / f"k{k}-r{rollout_index}.json",  # 先存檔再記帳（decision 023 §2）
        )
        if retried.trajectory is not None:
            outcome, stopped = classify_rollout(retried.trajectory, k)
            result = RolloutOutcome(k, rollout_index, outcome, retried.trajectory.termination, stopped, retried.attempts)
        else:
            label = "aborted" if retried.error_class == "infrastructure" else "agent_error"
            result = RolloutOutcome(k, rollout_index, label, None, False, retried.attempts)
        return result

    protocol = run_protocol(last, plan, run_one)
    if remaining:
        raise ValueError("some carried-over rollouts are not part of this sweep's schedule (decision 023 §1)")
    estimates = tuple(estimate_checkpoint(k, list(protocol.outcomes[k])) for k in sorted(protocol.outcomes))
    assessed = assess_drop(estimates, plan.drop_threshold, criterion=criterion)
    return SweepResult(
        task_id=task.task_id,
        source_trajectory_id=source_id,
        plan=plan,
        estimates=estimates,
        rollouts=protocol.order,
        refined_intervals=protocol.refined_intervals,
        largest_drop=assessed.largest_drop,
        largest_drop_k=assessed.largest_drop_k,
        is_cliff=assessed.is_cliff,
        criterion=criterion,
        max_turns_confounded=assessed.max_turns_confounded,
    )

