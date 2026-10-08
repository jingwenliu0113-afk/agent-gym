"""SPEC §6：在 eval baseline 的 source 上做 failure localization sweep（decision 010 / 011 / 012）。

用法::

    uv run python -m experiments.run_sweep --baseline-run runs/<eval-run> [runs/<eval-rerun> ...] \
        --plan STRIDE,N_COARSE,N_FINE,THRESHOLD --criterion {wilson95,bonferroni95} --dry-run
        不連網：OracleFakeClient 從 checkpoint 續跑（只驗證 harness，曲線沒有研究意義）。
    uv run --extra llm python -m experiments.run_sweep --baseline-run runs/<eval-run> --plan ... \\
        --confirm-paid-run --max-cost-usd X
        真正呼叫 Anthropic API；只能在使用者確認 SweepPlan 與預算之後執行。credential 由 SDK 自行解析，本腳本不讀取任何 key。
    ... --run-name <new-name> --resume-from runs/<first-sweep> [runs/<second-sweep> ...]
        中斷或因預算停止後續跑（decision 018 §1 / 019 §3 / 020）：必須列出**所有**先前的 sweep 目錄（較早與較晚的都要；
        --resume-from 可重複給）。以 analyze_sweep 的完整核對讀這些目錄，plan 與 criterion 必須相同；已完成的 source 不再執行
        （不重複付費）。之後以 analyze_sweep 同時分析所有目錄。--max-cost-usd 是這一次執行的上限（先前目錄已花的金額會印出）。

--baseline-run 要列出 eval baseline 的**所有**目錄（中斷後重跑的也算；與 analyze_baseline 相同），source 依 decision 019 §1 選取。

SweepPlan 與驟降判定（--criterion）都沒有預設值：必須明確給定（decision 011 / 015 的選擇屬於使用者決策）。
輸出（gitignored）：``runs/<name>/`` 下 ``run.json``（開始時寫入的設定與略過清單，decision 019 §2）、``manifest.jsonl``、
``sweeps/<task_id>.json``（每個 source 的 SweepResult）、
``rollouts/<task_id>/k<k>-r<i>.json``（每個完成的 rerun trajectory，decision 017 §2）、``summary.json``。
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from agents.baseline import AnthropicPolicy, baseline_agent_config, make_anthropic_client
from agents.policy import AgentConfig
from env.hashing import canonical_json
from experiments.analyze_baseline import RolloutRecord, load_runs
from replay.trajectory import load_strict_json
from replay.aborts import utc_instant
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

# v2：max_turns_confounded、rollout 存檔（017）；v3：skipped（018）；v4：run.json（019）；v5：run_id / resumed_from（020）。
# run.json 與 summary.json 共用此 format
SWEEP_SUMMARY_FORMAT = "agent-gym/sweep-run/v5"
RUN_FILE = "run.json"
_RUN_ID = re.compile(r"[0-9a-f]{32}")


def is_run_id(value: Any) -> bool:
    """run.json 的 run_id：32 位小寫 hex（decision 020 §2）。"""
    return isinstance(value, str) and _RUN_ID.fullmatch(value) is not None


class BudgetExceeded(RuntimeError):
    """已花費達到上限：在下一次嘗試開始之前（policy 建立時）拋出，因此不會留下該次嘗試的 manifest 紀錄。"""


def baseline_directories(baseline_run_dirs: str | Path | Sequence[str | Path]) -> list[Path]:
    """一個目錄（str / Path）或目錄的序列 → Path 的 list；空序列 → ValueError。"""
    if isinstance(baseline_run_dirs, (str, Path)):
        return [Path(baseline_run_dirs)]
    directories = [Path(directory) for directory in baseline_run_dirs]
    if not directories:
        raise ValueError("at least one eval baseline directory is required")
    return directories


def select_sources(baseline_run_dirs: str | Path | Sequence[str | Path]) -> dict[str, Trajectory | None]:
    """decision 012 §3 / 017 §3 / 019 §1：以 ``experiments.analyze_baseline.load_runs`` 讀 eval baseline 的**所有**目錄（summary、
    manifest、trajectory 交叉核對；沒有 summary 的中斷目錄由 manifest 重建），回傳**全部** eval task（依 eval pool 的順序）→ source。

    - 每個 task 的 rollout_index 0 = 所有目錄中最早結束者（最後一次嘗試的 manifest utc_time，換算成 UTC）；interrupted（沒有結束）
      不算；最早的時間有兩條以上 → ValueError（與傳入順序無關，decision 020 §5）。
    - 它完成（success / failure / refusal）→ 它的 trajectory（已 verify）；agent_error / aborted，或沒有任何結束的 rollout 0 → None
      （不得以其他 rollout 補上，其他目錄的 rollout 0 也一樣）。
    - 任何檔案不一致 → ValueError。
    """
    loaded = load_runs(baseline_directories(baseline_run_dirs))
    finished: dict[str, list[RolloutRecord]] = {}
    for record in loaded.records:
        if record.rollout_index == 0 and record.outcome != "interrupted":
            finished.setdefault(record.task_id, []).append(record)
    first: dict[str, RolloutRecord] = {}
    for task_id, records in finished.items():
        earliest = min(utc_instant(record.utc_time) for record in records)
        at_earliest = [record for record in records if utc_instant(record.utc_time) == earliest]
        if len(at_earliest) > 1:  # 只看最早的時間：與傳入順序無關（decision 020 §5）
            raise ValueError(f"two rollout_index 0 runs of {task_id} ended at the same earliest time; the source is ambiguous")
        first[task_id] = at_earliest[0]
    sources: dict[str, Trajectory | None] = {}
    for instance in week1_pool("eval"):
        task_id = instance.task.task_id
        record = first.get(task_id)
        if record is None or record.outcome not in COMPLETED_OUTCOMES:
            sources[task_id] = None
        else:
            sources[task_id] = loaded.trajectories[(record.run, task_id, 0)]
    return sources


def run_description(
    *,
    run_id: str,
    resumed_from: Sequence[str],
    plan: SweepPlan,
    criterion: str,
    config_id: str,
    skipped: Sequence[str],
    sources: Mapping[str, Trajectory | None],
) -> dict[str, Any]:
    """``run.json`` 的內容（decision 019 §2 / 020 §2）：format、run_id、resumed_from（list）、plan（欄位 dict）、criterion、
    config_id、skipped（list）、sources（task_id → source 的 trajectory_id，沒有 source 為 None）。"""
    return {
        "format": SWEEP_SUMMARY_FORMAT,
        "run_id": run_id,
        "resumed_from": list(resumed_from),
        "plan": asdict(plan),
        "criterion": criterion,
        "config_id": config_id,
        "skipped": list(skipped),
        "sources": {task_id: None if source is None else source.trajectory_id for task_id, source in sources.items()},
    }


def read_lineage(directory: Path) -> tuple[str, tuple[str, ...]] | None:
    """寬鬆讀取 ``directory/run.json`` 的 (run_id, resumed_from)；不是目前格式的 sweep 目錄、讀不到或欄位不合法 → None
    （decision 020 §3 的後代檢查只用它來拒絕，不據以加入任何目錄）。"""
    try:
        data = load_strict_json((directory / RUN_FILE).read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != SWEEP_SUMMARY_FORMAT:
        return None
    run_id, parents = data.get("run_id"), data.get("resumed_from")
    if not is_run_id(run_id) or not isinstance(parents, list) or not all(is_run_id(parent) for parent in parents):
        return None
    return run_id, tuple(parents)


def later_sweeps(resume_from: Sequence[Path], run_dir: Path) -> list[Path]:
    """decision 020 §3：在 resume_from 各目錄與 run_dir 的上層目錄中（不遞迴），找出 run.json 的 resumed_from 含有任一
    resume_from 目錄的 run_id、有 manifest、但不在 resume_from 中的目錄（依路徑排序）。"""

    def identity(directory: Path) -> tuple[int, int] | None:
        try:
            info = os.stat(directory)
        except (OSError, ValueError):
            return None
        return info.st_dev, info.st_ino

    given = {identity(directory) for directory in resume_from} - {None}
    ids = {lineage[0] for directory in resume_from if (lineage := read_lineage(directory)) is not None}
    parents: dict[tuple[int, int], Path] = {}
    for parent in [directory.parent for directory in resume_from] + [run_dir.parent]:
        key = identity(parent)
        if key is not None:
            parents.setdefault(key, parent)
    found: list[Path] = []
    for parent in parents.values():
        try:
            children = sorted(parent.iterdir())
        except OSError:
            continue
        for child in children:
            key = identity(child)
            if key is None or key in given or not child.is_dir():
                continue
            lineage = read_lineage(child)
            if lineage is not None and set(lineage[1]) & ids and (child / "manifest.jsonl").is_file():
                found.append(child)
    return sorted(found)


@dataclass(frozen=True)
class SweepRunSummary:
    plan: SweepPlan
    criterion: str  # 驟降判定（replay.localization.CLIFF_CRITERIA，decision 015）
    config_id: str
    sources: tuple[str, ...]  # 全部 eval task，依 eval pool 的順序（含缺漏者）
    missing_sources: tuple[str, ...]  # source 為 None 的 task
    skipped: tuple[str, ...]  # 有 source、但這個目錄不執行的 task（續跑：已在其他目錄完成；decision 018 §1），依 pool 順序
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
            "skipped": list(self.skipped),
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
    skip: Collection[str] = (),
    resumed_from: Collection[str] = (),
) -> SweepRunSummary:
    """依 eval pool 的順序對每個非 None、且不在 skip 中的 source 執行 ``replay.localization.sweep``（共用一個 manifest）。

    - criterion（驟降判定，CLIFF_CRITERIA 之一）沒有預設值：與 plan 一樣必須明確給定（decision 011 / 015 的選擇屬於使用者決策）。
    - 事前檢查（ValueError，不執行）：plan 為 SweepPlan；criterion 在 CLIFF_CRITERIA 中；sources 的 key 剛好是全部 eval task
      （``select_sources`` 的形狀，decision 018 §1）；skip 是 task_id 的集合（不是 str），每個都有 source；至少有一個要執行的
      source（decision 019 §3）；每個非 None 的 source 的 agent.config_id 必須等於 agent_config（預設 baseline）的 config_id；
      ``check_budget_args``；run_dir 不存在或為空。
    - resumed_from：續跑時各 --resume-from 目錄的 run_id（不是 str 的集合；每個都是 32 位小寫 hex、不重複；否則 ValueError）。
    - 第一次嘗試之前寫入 ``run.json``（``run_description``：新的隨機 run_id、排序後的 resumed_from；decision 019 §2 / 020 §2）。
    - policy_factory(k, i, attempt)：``budget_reached(ledger, max_cost_usd)`` → raise BudgetExceeded；否則
      ``CostMeter(AnthropicPolicy(client, agent_config), ledger)``（ledger 跨所有 source 共用）。
    - BudgetExceeded → 停止（stopped_by_budget = True）：該 source 的部分 sweep 不計入 completed（rollouts 仍在 manifest 中），
      後面的 source 不再執行。
    - 每個完成的 sweep 寫入 ``sweeps/<task_id>.json``（canonical_json）；每個完成的 rerun trajectory 存成
      ``rollouts/<task_id>/k<k>-r<i>.json``（sweep 的 rollout_dir）。
    - 彙總：completed 的 task 依 family 分層 bootstrap（seed 0、``replicates``）；eval pool 的任一 family 在這個目錄沒有完成的
      sweep → None（012 §1；跨目錄的正式彙總由 analyze_sweep 重算）。
    - 正常結束或因預算停止都寫 ``summary.json`` 並回傳；harness 例外原樣往外拋。
    """
    if not isinstance(plan, SweepPlan):
        raise ValueError("plan must be a SweepPlan")
    if criterion not in CLIFF_CRITERIA:
        raise ValueError(f"criterion must be one of {CLIFF_CRITERIA}")
    pool = [instance.task.task_id for instance in week1_pool("eval")]
    if not isinstance(sources, Mapping) or set(sources) != set(pool):
        raise ValueError("sources must map every eval task (see select_sources) to its source or None")
    if isinstance(skip, (str, bytes)) or not all(isinstance(task_id, str) for task_id in skip):
        raise ValueError("skip must be a collection of task ids")
    skipped = set(skip)
    if any(task_id not in sources or sources[task_id] is None for task_id in skipped):
        raise ValueError("skip may only name eval tasks that have a source")
    if all(source is None or task_id in skipped for task_id, source in sources.items()):
        raise ValueError("nothing to run: every eval task is skipped or has no source")
    if isinstance(resumed_from, (str, bytes)) or not all(is_run_id(run_id) for run_id in resumed_from):
        raise ValueError("resumed_from must be a collection of run ids (32 lowercase hex digits)")
    parents = sorted(resumed_from)
    if len(set(parents)) != len(parents):
        raise ValueError("resumed_from must not repeat a run id")
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
    description = run_description(
        run_id=uuid.uuid4().hex,
        resumed_from=parents,
        plan=plan,
        criterion=criterion,
        config_id=config.config_id,
        skipped=[t for t in pool if t in skipped],
        sources=sources,
    )
    (run_dir / RUN_FILE).write_bytes((canonical_json(description) + "\n").encode("utf-8"))
    log_path = run_dir / "manifest.jsonl"
    ledger = Ledger()

    def policy_factory(k: int, rollout_index: int, attempt: int) -> CostMeter:
        if budget_reached(ledger, max_cost_usd):
            raise BudgetExceeded(f"spent ${ledger.cost_usd:.4f} (cap ${max_cost_usd}, unknown-cost calls {ledger.unknown_cost_calls})")
        return CostMeter(AnthropicPolicy(client, config), ledger)

    families = {instance.task.task_id: instance.task.family for instance in week1_pool("eval")}
    results: dict[str, SweepResult] = {}
    stopped = False
    for task_id in pool:
        source = sources[task_id]
        if source is None or task_id in skipped:
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
        sources=tuple(pool),
        missing_sources=tuple(task_id for task_id in pool if sources[task_id] is None),
        skipped=tuple(task_id for task_id in pool if task_id in skipped),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baseline-run", required=True, nargs="+", action="extend",
        help="eval baseline 的所有 runs/<name> 目錄（可給多個、可重複；decision 019 §1）",
    )
    parser.add_argument("--plan", required=True, help="STRIDE,N_COARSE,N_FINE,THRESHOLD（必須明確給定）")
    parser.add_argument("--criterion", required=True, choices=CLIFF_CRITERIA, help="驟降判定（必須明確給定，decision 015）")
    parser.add_argument("--run-name", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-paid-run", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--resume-from", nargs="+", action="extend", default=None,
        help="所有先前（中斷 / 因預算停止）的 sweep 目錄（可給多個、可重複；decision 018 §1 / 019 §3 / 020）",
    )
    args = parser.parse_args(argv)

    plan = parse_plan(args.plan)
    name = args.run_name or ("dry-run-sweep" if args.dry_run else "sweep")
    run_dir = Path("runs") / name
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        parser.error(
            f"{run_dir} already exists; to continue an interrupted or budget-stopped sweep, give a new --run-name and "
            "--resume-from every earlier sweep directory (decision 020 §6)"
        )
    sources = select_sources(args.baseline_run)
    skip: tuple[str, ...] = ()
    resumed_from: list[str] = []
    if args.resume_from:
        from experiments.analyze_sweep import analyze_sweep_runs  # 延遲 import：analyze_sweep 依賴本模組

        previous = analyze_sweep_runs(args.resume_from, args.baseline_run, replicates=1, plan=plan, criterion=args.criterion)
        skip = tuple(curve.task_id for curve in previous.curves)
        uncovered = sorted({t for entry in previous.resumed for t in entry["skipped"]} - set(skip))
        if uncovered:
            parser.error(
                f"{', '.join(uncovered)} were skipped by a --resume-from directory but completed in none of them; "
                "pass every earlier sweep directory to --resume-from (decision 019 §3)"
            )
        resume_dirs = [Path(directory) for directory in args.resume_from]
        left_out = later_sweeps(resume_dirs, run_dir)
        if left_out:
            parser.error(
                f"{', '.join(str(d) for d in left_out)} continued a --resume-from directory and already billed attempts; "
                "pass them to --resume-from too (decision 020 §3)"
            )
        lineages = [read_lineage(directory) for directory in resume_dirs]
        resumed_from = sorted(lineage[0] for lineage in lineages if lineage is not None)
        print(f"resuming: {len(skip)} sources already completed in {', '.join(previous.runs)}", file=sys.stderr)
        print(
            f"previously spent ${previous.cost_usd:.4f} (unknown-cost calls {previous.unknown_cost_calls}) in "
            f"{', '.join(previous.runs)}; --max-cost-usd applies to this run only",
            file=sys.stderr,
        )
        if all(source is None or task_id in skip for task_id, source in sources.items()):
            print("nothing left to run: analyze the existing directories with experiments.analyze_sweep", file=sys.stderr)
            return 0
    if args.dry_run:
        client: Any = OracleFakeClient(week1_pool("eval"))
    else:
        if args.max_cost_usd is None:
            parser.error("--confirm-paid-run requires --max-cost-usd")
        client = make_anthropic_client()
    summary = run_sweeps(
        sources, client, plan=plan, criterion=args.criterion, run_dir=run_dir, max_cost_usd=args.max_cost_usd,
        skip=skip, resumed_from=resumed_from,
    )
    for task_id in summary.sources:
        result = summary.results.get(task_id)
        if result is None:
            reason = "missing source" if task_id in summary.missing_sources else "skipped" if task_id in summary.skipped else "not run"
            print(f"{task_id}: {reason}")
        else:
            print(f"{task_id}: largest drop {result.largest_drop:.3f} at k={result.largest_drop_k}, cliff {result.is_cliff}")
    print(f"spent ${summary.spent_usd:.4f}; stopped by budget: {summary.stopped_by_budget}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
