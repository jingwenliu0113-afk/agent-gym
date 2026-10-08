"""SPEC §6：在 eval baseline 的 source 上做 failure localization sweep（decision 010 / 011 / 012）。

用法::

    uv run python -m experiments.run_sweep --baseline-run runs/<eval-run> --plan STRIDE,N_COARSE,N_FINE,THRESHOLD \
        --criterion {wilson95,bonferroni95} --dry-run
        不連網：OracleFakeClient 從 checkpoint 續跑（只驗證 harness，曲線沒有研究意義）。
    uv run --extra llm python -m experiments.run_sweep --baseline-run runs/<eval-run> --plan ... \\
        --confirm-paid-run --max-cost-usd X
        真正呼叫 Anthropic API；只能在使用者確認 SweepPlan 與預算之後執行。credential 由 SDK 自行解析，本腳本不讀取任何 key。

SweepPlan 與驟降判定（--criterion）都沒有預設值：必須明確給定（decision 011 / 015 的選擇屬於使用者決策）。
輸出（gitignored）：``runs/<name>/`` 下 ``manifest.jsonl``、``sweeps/<task_id>.json``（每個 source 的 SweepResult）、
``rollouts/<task_id>/k<k>-r<i>.json``（每個完成的 rerun trajectory，decision 017 §2）、``summary.json``。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from agents.baseline import AnthropicPolicy, baseline_agent_config, make_anthropic_client
from agents.policy import AgentConfig
from env.hashing import canonical_json
from experiments.analyze_baseline import load_runs
from experiments.run_baseline import (
    CostMeter,
    Ledger,
    OracleFakeClient,
    budget_reached,
    check_budget_args,
)
from replay.localization import CLIFF_CRITERIA, SweepPlan, SweepResult, sweep
from replay.runner import Clock
from replay.stats import BootstrapResult, stratified_task_bootstrap
from replay.trajectory import Trajectory
from tasks.generator import week1_pool

COMPLETED_OUTCOMES = ("success", "failure", "refusal")

SWEEP_SUMMARY_FORMAT = "agent-gym/sweep-run/v2"  # v2：SweepResult.max_turns_confounded、rollout 存檔（decision 017）


class BudgetExceeded(RuntimeError):
    """已花費達到上限：在下一次嘗試開始之前（policy 建立時）拋出，因此不會留下該次嘗試的 manifest 紀錄。"""


def select_sources(baseline_run_dir: str | Path) -> dict[str, Trajectory | None]:
    """decision 012 §3 / 017 §3：以 ``experiments.analyze_baseline.load_runs`` 讀一個 eval baseline 目錄（summary、manifest、
    trajectory 交叉核對；沒有 summary 的中斷目錄由 manifest 重建），回傳**全部** eval task（依 eval pool 的順序）→
    rollout_index 0 完成（success / failure / refusal）的 trajectory（已 verify）；沒有 rollout 0、或 rollout 0 未完成
    （agent_error / aborted / interrupted）→ None。任何檔案不一致 → ValueError。
    """
    loaded = load_runs([baseline_run_dir])
    label = loaded.labels[0]
    records = {(record.task_id, record.rollout_index): record for record in loaded.records}
    sources: dict[str, Trajectory | None] = {}
    for instance in week1_pool("eval"):
        task_id = instance.task.task_id
        record = records.get((task_id, 0))
        if record is None or record.outcome not in COMPLETED_OUTCOMES:
            sources[task_id] = None
        else:
            sources[task_id] = loaded.trajectories[(label, task_id, 0)]
    return sources


@dataclass(frozen=True)
class SweepRunSummary:
    plan: SweepPlan
    criterion: str  # 驟降判定（replay.localization.CLIFF_CRITERIA，decision 015）
    config_id: str
    sources: tuple[str, ...]  # 依處理順序的 task_id（含缺漏者）
    missing_sources: tuple[str, ...]  # source 為 None 的 task
    completed: tuple[str, ...]  # sweep 完成的 task
    results: Mapping[str, SweepResult]  # task_id → SweepResult（只含 completed）
    families: Mapping[str, str]  # task_id → family（全部 sources）
    cliff_rate: BootstrapResult | None  # completed 的 is_cliff（0/1）以 family 分層 bootstrap；任一 family 沒有 completed → None
    mean_largest_drop: BootstrapResult | None  # 同上，值為 largest_drop
    spent_usd: float
    model_calls: int
    unknown_cost_calls: int
    stopped_by_budget: bool

    def to_dict(self) -> dict[str, Any]:
        """``{"format": SWEEP_SUMMARY_FORMAT, ...}``，JSON-native：plan 與 BootstrapResult 為欄位 dict；results 為
        task_id → SweepResult 的欄位 dict（dataclasses.asdict）。"""
        return {
            "format": SWEEP_SUMMARY_FORMAT,
            "plan": asdict(self.plan),
            "criterion": self.criterion,
            "config_id": self.config_id,
            "sources": list(self.sources),
            "missing_sources": list(self.missing_sources),
            "completed": list(self.completed),
            "results": {task_id: _jsonable(asdict(result)) for task_id, result in self.results.items()},
            "families": dict(self.families),
            "cliff_rate": None if self.cliff_rate is None else asdict(self.cliff_rate),
            "mean_largest_drop": None if self.mean_largest_drop is None else asdict(self.mean_largest_drop),
            "spent_usd": self.spent_usd,
            "model_calls": self.model_calls,
            "unknown_cost_calls": self.unknown_cost_calls,
            "stopped_by_budget": self.stopped_by_budget,
        }


def _jsonable(value: Any) -> Any:
    """tuple → list（遞迴），讓 asdict 的結果可以 canonical_json 且與讀回的 JSON 相同。"""
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def run_sweeps(
    sources: Mapping[str, Trajectory | None],
    client: Any,
    *,
    plan: SweepPlan,
    criterion: str,
    run_dir: str | Path,
    max_cost_usd: float | None = None,
    clock: Clock = time.perf_counter,
    agent_config: AgentConfig | None = None,
    replicates: int = 10000,
) -> SweepRunSummary:
    """依 sources 的順序對每個非 None 的 source 執行 ``replay.localization.sweep``（共用一個 manifest）。

    - criterion（驟降判定，CLIFF_CRITERIA 之一）沒有預設值：與 plan 一樣必須明確給定（decision 011 / 015 的選擇屬於使用者決策）。
    - 事前檢查（ValueError，不執行）：plan 為 SweepPlan；criterion 在 CLIFF_CRITERIA 中；sources 非空；每個非 None 的 source 的 agent.config_id 必須等於
      agent_config（預設 baseline）的 config_id；``check_budget_args``；run_dir 不存在或為空。
    - policy_factory(k, i, attempt)：``budget_reached(ledger, max_cost_usd)`` → raise BudgetExceeded；否則
      ``CostMeter(AnthropicPolicy(client, agent_config), ledger)``（ledger 跨所有 source 共用）。
    - BudgetExceeded → 停止（stopped_by_budget = True）：該 source 的部分 sweep 不計入 completed（rollouts 仍在 manifest 中），
      後面的 source 不再執行。
    - 每個完成的 sweep 寫入 ``sweeps/<task_id>.json``（canonical_json）；每個完成的 rerun trajectory 存成
      ``rollouts/<task_id>/k<k>-r<i>.json``（sweep 的 rollout_dir）。
    - 彙總：completed 的 task 依 family（source.task["family"]）分層 bootstrap（seed 0、``replicates``）。
    - 正常結束或因預算停止都寫 ``summary.json`` 並回傳；harness 例外原樣往外拋。
    """
    if not isinstance(plan, SweepPlan):
        raise ValueError("plan must be a SweepPlan")
    if criterion not in CLIFF_CRITERIA:
        raise ValueError(f"criterion must be one of {CLIFF_CRITERIA}")
    if not isinstance(sources, Mapping) or len(sources) == 0:
        raise ValueError("sources must be a non-empty mapping")
    config = baseline_agent_config() if agent_config is None else copy.deepcopy(agent_config)
    check_budget_args(max_cost_usd, config)
    for task_id, source in sources.items():
        if source is None:
            continue
        if not isinstance(source, Trajectory) or source.task.get("task_id") != task_id:
            raise ValueError(f"source for {task_id} must be that task's Trajectory")
        if source.agent.config_id != config.config_id:
            raise ValueError(f"source for {task_id} was produced by a different agent config")
    run_dir = Path(run_dir)
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise ValueError(f"{run_dir} already exists and is not an empty directory")
    AnthropicPolicy(client, config)  # 設定與 client 的檢查在任何呼叫之前完成

    sweeps_dir = run_dir / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "manifest.jsonl"
    ledger = Ledger()

    def policy_factory(k: int, rollout_index: int, attempt: int) -> CostMeter:
        if budget_reached(ledger, max_cost_usd):
            raise BudgetExceeded(f"spent ${ledger.cost_usd:.4f} (cap ${max_cost_usd}, unknown-cost calls {ledger.unknown_cost_calls})")
        return CostMeter(AnthropicPolicy(client, config), ledger)

    families = {task_id: _family(task_id, source) for task_id, source in sources.items()}
    results: dict[str, SweepResult] = {}
    stopped = False
    for task_id, source in sources.items():
        if source is None:
            continue
        try:
            result = sweep(
                source,
                policy_factory,
                plan=plan,
                log_path=log_path,
                clock=clock,
                criterion=criterion,
                rollout_dir=run_dir / "rollouts" / task_id,
            )
        except BudgetExceeded:
            stopped = True
            break
        results[task_id] = result
        (sweeps_dir / f"{task_id}.json").write_bytes((canonical_json(_jsonable(asdict(result))) + "\n").encode("utf-8"))

    completed = tuple(results)
    cliff_rate = mean_drop = None
    strata_all = set(families.values())
    strata_done = {families[task_id] for task_id in completed}
    if completed and strata_done == strata_all:
        strata = {task_id: families[task_id] for task_id in completed}
        cliff_rate = stratified_task_bootstrap(
            {t: float(results[t].is_cliff) for t in completed}, strata, replicates=replicates, seed=0
        )
        mean_drop = stratified_task_bootstrap(
            {t: results[t].largest_drop for t in completed}, strata, replicates=replicates, seed=0
        )
    summary = SweepRunSummary(
        plan=plan,
        criterion=criterion,
        config_id=config.config_id,
        sources=tuple(sources),
        missing_sources=tuple(task_id for task_id, source in sources.items() if source is None),
        completed=completed,
        results=results,
        families=families,
        cliff_rate=cliff_rate,
        mean_largest_drop=mean_drop,
        spent_usd=ledger.cost_usd,
        model_calls=ledger.calls,
        unknown_cost_calls=ledger.unknown_cost_calls,
        stopped_by_budget=stopped,
    )
    (run_dir / "summary.json").write_bytes((canonical_json(summary.to_dict()) + "\n").encode("utf-8"))
    return summary


def _family(task_id: str, source: Trajectory | None) -> str:
    if source is not None:
        return source.task["family"]
    for instance in week1_pool("all"):
        if instance.task.task_id == task_id:
            return instance.task.family
    raise ValueError(f"unknown task {task_id}")


def parse_plan(text: str) -> SweepPlan:
    """``"STRIDE,N_COARSE,N_FINE,THRESHOLD"`` → SweepPlan（int、int、int、float）；格式錯誤 → ValueError。"""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("plan must be STRIDE,N_COARSE,N_FINE,THRESHOLD")
    try:
        stride, n_coarse, n_fine = (int(part, 10) for part in parts[:3])
        threshold = float(parts[3])
    except ValueError as exc:
        raise ValueError(f"invalid plan {text!r}") from exc
    return SweepPlan(stride, n_coarse, n_fine, threshold)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-run", required=True, help="eval baseline 的 runs/<name> 目錄")
    parser.add_argument("--plan", required=True, help="STRIDE,N_COARSE,N_FINE,THRESHOLD（必須明確給定）")
    parser.add_argument("--criterion", required=True, choices=CLIFF_CRITERIA, help="驟降判定（必須明確給定，decision 015）")
    parser.add_argument("--run-name", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-paid-run", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=None)
    args = parser.parse_args(argv)

    plan = parse_plan(args.plan)
    sources = select_sources(args.baseline_run)
    if args.dry_run:
        client: Any = OracleFakeClient(week1_pool("eval"))
        name = args.run_name or "dry-run-sweep"
    else:
        if args.max_cost_usd is None:
            parser.error("--confirm-paid-run requires --max-cost-usd")
        client = make_anthropic_client()
        name = args.run_name or "sweep"
    summary = run_sweeps(
        sources, client, plan=plan, criterion=args.criterion, run_dir=Path("runs") / name, max_cost_usd=args.max_cost_usd
    )
    for task_id in summary.sources:
        result = summary.results.get(task_id)
        if result is None:
            print(f"{task_id}: {'missing source' if task_id in summary.missing_sources else 'not run'}")
        else:
            print(f"{task_id}: largest drop {result.largest_drop:.3f} at k={result.largest_drop_k}, cliff {result.is_cliff}")
    print(f"spent ${summary.spent_usd:.4f}; stopped by budget: {summary.stopped_by_budget}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
