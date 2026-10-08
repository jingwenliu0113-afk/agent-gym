"""eval sweep 的預先登記分析：recoverability curve 與 Gate 判定（decision 010 / 012 / 013 / 016 B / 017 §5）。
不連網、不呼叫任何 LLM。

用法::

    uv run python -m experiments.analyze_sweep runs/<sweep-run> [runs/<rerun> ...] --baseline-run runs/<eval-run> \\
        --out runs/<analysis-name>

輸出（gitignored）：``<out>/analysis.json``、``<out>/analysis.md``、``<out>/curves/<task_id>.svg``。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import MAX_EPISODE_ATTEMPTS
from env.hashing import canonical_json
from experiments.run_sweep import SWEEP_SUMMARY_FORMAT, SweepRunSummary, select_sources
from replay.aborts import AttemptRecord, read_manifest, utc_date
from replay.localization import (
    CLIFF_CRITERIA,
    ROLLOUT_OUTCOMES,
    CheckpointEstimate,
    RolloutOutcome,
    SweepPlan,
    SweepResult,
    assess_drop,
    classify_rollout,
    estimate_checkpoint,
    run_protocol,
)
from replay.stats import BootstrapResult, stratified_task_bootstrap
from replay.trajectory import Trajectory, TrajectoryError, load_strict_json
from tasks.generator import week1_pool

SWEEP_ANALYSIS_FORMAT = "agent-gym/sweep-analysis/v2"  # v2：decision 017 §5
STEP_CATEGORIES: tuple[str, ...] = (
    "forbidden_attempt",
    "unnecessary_mutation",
    "target_mutation",
    "other_mutation",
    "no_mutation",
)
GATE_RULE = "decision 012 §4 proposal (not yet confirmed by the user): same (tool, category) pattern at the cliff turn in >= 2 families"
MERGE_FIELDS = ("generator_version", "grader_version", "schema_version", "config_id", "max_turns")


def sweep_result_from_dict(data: Any) -> SweepResult:
    """``dataclasses.asdict(SweepResult)``（run_sweep 寫入 ``sweeps/<task_id>.json`` 的內容）→ SweepResult；巢狀的 SweepPlan、
    CheckpointEstimate、RolloutOutcome 以各自的欄位重建（tuple 欄位由 list 還原）。key 不符、型別錯、dataclass 驗證失敗、
    或取值不合法（criterion 不在 CLIFF_CRITERIA、outcome 不在 ROLLOUT_OUTCOMES、k / n / 計數 < 0、attempts 不在
    1..MAX_EPISODE_ATTEMPTS、refine 區間不是 0 <= a < b、數值不是有限數）→ ValueError。
    保證 ``sweep_result_from_dict(json.loads(canonical_json(asdict(r)))) == r``。
    """
    data = _fields(data, SweepResult, "sweep result")
    plan = _build(SweepPlan, _fields(data["plan"], SweepPlan, "plan"))
    estimates = tuple(_build(CheckpointEstimate, _fields(e, CheckpointEstimate, "estimate")) for e in _list(data["estimates"]))
    rollouts = tuple(_build(RolloutOutcome, _fields(r, RolloutOutcome, "rollout")) for r in _list(data["rollouts"]))
    intervals = []
    for interval in _list(data["refined_intervals"]):
        if not isinstance(interval, list) or len(interval) != 2 or not all(type(x) is int for x in interval):
            raise ValueError("each refined interval must be a list of two ints")
        if not 0 <= interval[0] < interval[1]:
            raise ValueError("each refined interval must satisfy 0 <= a < b")
        intervals.append((interval[0], interval[1]))
    if data["criterion"] not in CLIFF_CRITERIA:
        raise ValueError(f"criterion must be one of {CLIFF_CRITERIA}")
    for estimate in estimates:
        counts = (estimate.k, estimate.n, estimate.successes, estimate.failures, estimate.refusals, estimate.agent_errors,
                  estimate.aborted, estimate.max_turns_terminations, estimate.stopped_without_action)
        if any(value < 0 for value in counts):
            raise ValueError("estimate counts must be >= 0")
    for rollout in rollouts:
        if rollout.outcome not in ROLLOUT_OUTCOMES:
            raise ValueError(f"unknown rollout outcome {rollout.outcome!r}")
        if rollout.k < 0 or rollout.rollout_index < 0 or not 1 <= rollout.attempts <= MAX_EPISODE_ATTEMPTS:
            raise ValueError("rollout k / rollout_index must be >= 0 and attempts within the retry limit")
    return _build(
        SweepResult,
        {**data, "plan": plan, "estimates": estimates, "rollouts": rollouts, "refined_intervals": tuple(intervals)},
    )


_NUMBER = (int, float)
_SCALAR_TYPES: dict[type, dict[str, tuple[type, ...]]] = {
    SweepResult: {
        "task_id": (str,), "source_trajectory_id": (str,), "largest_drop": _NUMBER, "largest_drop_k": (int, type(None)),
        "is_cliff": (bool,), "criterion": (str,), "max_turns_confounded": (bool,),
    },
    SweepPlan: {"coarse_stride": (int,), "n_coarse": (int,), "n_fine": (int,), "drop_threshold": _NUMBER},
    CheckpointEstimate: {
        "k": (int,), "n": (int,), "successes": (int,), "failures": (int,), "refusals": (int,), "agent_errors": (int,),
        "aborted": (int,), "max_turns_terminations": (int,), "stopped_without_action": (int,), "p_hat": _NUMBER,
        "p_hat_completed": (int, float, type(None)), "ci_low": _NUMBER, "ci_high": _NUMBER,
    },
    RolloutOutcome: {
        "k": (int,), "rollout_index": (int,), "outcome": (str,), "termination": (str, type(None)),
        "stopped_without_action": (bool,), "attempts": (int,),
    },
    BootstrapResult: {"estimate": _NUMBER, "low": _NUMBER, "high": _NUMBER, "replicates": (int,), "n_tasks": (int,)},
}


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _fields(data: Any, cls: type, what: str) -> dict[str, Any]:
    names = {field.name for field in dataclasses.fields(cls)}
    if not isinstance(data, dict) or set(data) != names:
        raise ValueError(f"{what} must be an object with keys: {', '.join(sorted(names))}")
    for name, types in _SCALAR_TYPES.get(cls, {}).items():
        value = data[name]
        bool_ok = bool in types
        if (isinstance(value, bool) and not bool_ok) or not isinstance(value, types):
            raise ValueError(f"{what}.{name} has the wrong type")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not _is_finite(value):
            raise ValueError(f"{what}.{name} must be finite")
    return data


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("expected a list")
    return value


def _build(cls: type, kwargs: dict[str, Any]) -> Any:
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {cls.__name__}: {exc}") from exc


def step_category(source: Trajectory, step: int) -> str:
    """decision 016 B3：source 第 step 個 tool step 的類別（STEP_CATEGORIES 之一，依序取第一個成立者），依 source.grade()。
    step 不在 [0, tool step 總數) → ValueError。"""
    records = source.records()
    if type(step) is not int or not 0 <= step < len(records):
        raise ValueError(f"step must be an int in [0, {len(records)})")
    report = source.grade()
    if any(attempt.step == step for attempt in report.forbidden_attempts):
        return "forbidden_attempt"
    if any(mutation.step == step for mutation in report.unnecessary_mutations):
        return "unnecessary_mutation"
    changes = records[step].changes
    if not changes:
        return "no_mutation"
    failed_issues = {
        check.predicate.get("issue_id")
        for check in report.outcome_checks + report.constraint_checks
        if not check.passed
    }
    if any(change.issue_id in failed_issues for change in changes):
        return "target_mutation"
    return "other_mutation"


@dataclass(frozen=True)
class CliffTurn:
    k: int  # k*：cliff 在 checkpoint k* → k* + 1 之間
    pattern: tuple[tuple[str, str], ...]  # source 第 k* 個 turn 每個 tool call 的 (tool, category)，去重、排序


def cliff_turn(source: Trajectory, result: SweepResult) -> CliffTurn | None:
    """result.is_cliff 為 False → None；否則 source.turns[k*] 的模式。result 必須屬於 source
    （source_trajectory_id 相符、k* < len(turns)），否則 ValueError。"""
    if not result.is_cliff:
        return None
    if result.source_trajectory_id != source.trajectory_id:
        raise ValueError("the sweep result does not belong to this source trajectory")
    k = result.largest_drop_k
    if type(k) is not int or not 0 <= k < len(source.turns):
        raise ValueError("largest_drop_k is not a turn of the source")
    pattern = {(step.tool, step_category(source, step.step)) for step in source.turns[k].tool_steps}
    return CliffTurn(k=k, pattern=tuple(sorted(pattern)))


@dataclass(frozen=True)
class SourceCurve:
    task_id: str
    family: str
    source_trajectory_id: str
    source_success: bool
    source_safe: bool
    n_turns: int
    result: SweepResult
    cliff: CliffTurn | None
    run: str = ""  # 完成這個 sweep 的目錄標籤


@dataclass(frozen=True)
class GateAssessment:
    rule: str  # GATE_RULE
    patterns: dict[str, tuple[str, ...]]  # 模式的字串表示（"tool:category+..."）→ 出現該模式的 task_id（排序）
    families: dict[str, tuple[str, ...]]  # 模式 → 出現的 family（排序）
    met: bool  # 是否有任一模式出現在 >= 2 個 family
    data_complete: bool = True  # 每個 eval task 都有完成的 sweep（否則報告標示「資料不完整」；decision 017 §5.3）


def assess_gate(curves: tuple[SourceCurve, ...], *, data_complete: bool = True) -> GateAssessment:
    """decision 016 B4（提案規則）。只看 cliff 不為 None 的 source；模式的字串表示 =
    ``"+".join(f"{tool}:{category}" for tool, category in pattern)``（空 turn 的模式為 ""）。"""
    tasks: dict[str, set[str]] = {}
    families: dict[str, set[str]] = {}
    for curve in curves:
        if curve.cliff is None:
            continue
        key = pattern_key(curve.cliff.pattern)
        tasks.setdefault(key, set()).add(curve.task_id)
        families.setdefault(key, set()).add(curve.family)
    return GateAssessment(
        rule=GATE_RULE,
        patterns={key: tuple(sorted(value)) for key, value in sorted(tasks.items())},
        families={key: tuple(sorted(value)) for key, value in sorted(families.items())},
        met=any(len(value) >= 2 for value in families.values()),
        data_complete=data_complete,
    )


def pattern_key(pattern: tuple[tuple[str, str], ...]) -> str:
    return "+".join(f"{tool}:{category}" for tool, category in pattern)


@dataclass(frozen=True)
class WindowEstimate:
    date: str  # UTC 日期（rollout 最後一次嘗試的 manifest utc_time）
    n_sources: int
    n_rollouts: int
    success_rate: float  # 該窗內 rollout 的 success 比例（描述用的點估計）


@dataclass(frozen=True)
class SweepAnalysis:
    plan: SweepPlan
    criterion: str
    runs: tuple[str, ...]  # 每個 sweep 目錄的標籤（目錄名稱；名稱重複時為傳入的路徑字串）
    interrupted_runs: tuple[str, ...]  # 沒有 summary.json 的目錄
    stopped_by_budget: tuple[str, ...]  # summary 標示因預算停止的目錄
    curves: tuple[SourceCurve, ...]  # 依 eval pool 的順序
    missing_sources: tuple[str, ...]  # 沒有 source 的 eval task（decision 012 §3）
    not_completed: tuple[str, ...]  # 有 source 但沒有任何目錄完成 sweep
    partial_sweeps: tuple[dict[str, Any], ...]  # 有 rollout 但沒有完成的 sweep：run、task_id、rollouts（不同 (k, i) 數）
    max_turns_confounded: tuple[str, ...]  # 下降合格但 max_turns 比例上升 → 不是 cliff（decision 006 / 017 §1）
    cliff_rate: BootstrapResult | None  # family 分層 bootstrap；任一 eval family 沒有完成的 sweep 或 pooling_blocked → None
    mean_largest_drop: BootstrapResult | None
    gate: GateAssessment | None  # pooling_blocked → None（不評估）
    pooling_blocked: str | None  # response_model 不只一種（decision 009 §4）
    response_models: dict[str, int]  # response_model → 出現該 model 的完成 sweep rollout 數（只看 k 之後的 turns）
    time_windows: tuple[WindowEstimate, ...]  # 依日期排序
    # 每一個 aborted 嘗試：run、task_id、k、rollout_index、attempt、partial_steps、partial_forbidden_action_count、
    # partial_forbidden_attempt_count（None 原樣保留）；依目錄、manifest 順序
    partial_safety: tuple[dict[str, Any], ...]
    cost_usd: float  # 所有目錄 manifest 的 cost_usd 加總
    unknown_cost_calls: int

    def to_dict(self) -> dict[str, Any]:
        """``{"format": SWEEP_ANALYSIS_FORMAT, ...}``，JSON-native。"""
        return {"format": SWEEP_ANALYSIS_FORMAT, **_jsonable(asdict(self))}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _bootstrap(data: Any) -> BootstrapResult | None:
    if data is None:
        return None
    result = _build(BootstrapResult, _fields(data, BootstrapResult, "bootstrap"))
    if result.replicates < 1 or result.n_tasks < 0:
        raise ValueError("bootstrap replicates must be >= 1 and n_tasks >= 0")
    return result


def analyze_sweep_run(
    sweep_run_dir: str | Path, baseline_run_dir: str | Path, *, replicates: int = 10000
) -> SweepAnalysis:
    """``analyze_sweep_runs([sweep_run_dir], baseline_run_dir, replicates=replicates)``。"""
    return analyze_sweep_runs([sweep_run_dir], baseline_run_dir, replicates=replicates)


@dataclass(frozen=True)
class _Rollout:
    task_id: str
    outcome: str
    date: str
    models: tuple[str, ...]


def analyze_sweep_runs(
    sweep_run_dirs: Sequence[str | Path], baseline_run_dir: str | Path, *, replicates: int = 10000
) -> SweepAnalysis:
    """decision 016 B 與 017 §5。任何不一致 → ValueError。

    - source = ``run_sweep.select_sources(baseline_run_dir)``（全部 eval task）。sweep 目錄不得重複（resolve 後），也不得是 baseline。
    - 每個目錄：manifest 必須存在；每列都是某個 source 的 checkpoint rollout（task 與 source_trajectory_id 相符），
      所有目錄所有列的 (generator_version, grader_version, schema_version, config_id, max_turns) 相同，且 config_id 與 source 相同。
      ``sweeps/*.json`` 為完成的 sweep。summary.json 可以沒有（中斷）；有的話逐欄驗證型別並與檔案、source 核對（cliff_rate 等彙總值
      只檢查型別，不採用）。所有目錄的 plan 與 criterion 必須相同；同一個 source 只能在一個目錄完成。
    - 每個完成的 sweep 以 manifest 與 ``rollouts/<task_id>/k<k>-r<i>.json`` 重算核對（rollout 的嘗試數、完成與否與標籤、
      trajectory verify / parent / classify_rollout / termination、run_protocol 重現的排程與 refine 區間、estimate_checkpoint、
      assess_drop）。
    - 彙總（replicates、seed 0）只在 response_model 只有一種、且每個 eval family 都有完成的 sweep 時計算。
    """
    if isinstance(sweep_run_dirs, (str, Path)) or len(sweep_run_dirs) == 0:
        raise ValueError("sweep_run_dirs must be a non-empty sequence of sweep run directories")
    directories = [Path(d) for d in sweep_run_dirs]
    resolved = [d.resolve() for d in directories]
    if len(set(resolved)) != len(resolved) or Path(baseline_run_dir).resolve() in resolved:
        raise ValueError("each sweep directory must be given once and must differ from the baseline run")
    names = [d.name for d in directories]
    labels = tuple(name if names.count(name) == 1 else str(d) for name, d in zip(names, directories))
    sources = select_sources(baseline_run_dir)
    pool_families = {instance.task.task_id: instance.task.family for instance in week1_pool("eval")}
    pool_order = list(pool_families)
    source_ids = {source.trajectory_id: task_id for task_id, source in sources.items() if source is not None}
    config_ids = {source.agent.config_id for source in sources.values() if source is not None}

    settings: set[str] = set()  # canonical [plan, criterion]
    completed: dict[str, tuple[str, SweepResult, list[_Rollout]]] = {}
    partial_sweeps: list[dict[str, Any]] = []
    interrupted_runs: list[str] = []
    stopped: list[str] = []
    all_rows: list[tuple[str, AttemptRecord]] = []
    merge_key: str | None = None
    for directory, label in zip(directories, labels):
        manifest = directory / "manifest.jsonl"
        if not manifest.is_file():
            raise ValueError(f"{directory} has no manifest; not a sweep run")
        rows = read_manifest(manifest)
        for row in rows:
            key = canonical_json({name: getattr(row, name) for name in MERGE_FIELDS})
            if merge_key is None:
                merge_key = key
            elif key != merge_key:
                raise ValueError("sweep runs with different generator / grader / schema / config / max_turns cannot be merged")
            if row.checkpoint_id is None or source_ids.get(row.source_trajectory_id) != row.task_id:
                raise ValueError(f"{directory}: a manifest row is not a rollout of this baseline run's sources")
            if row.config_id not in config_ids:
                raise ValueError(f"{directory}: a manifest row used a different agent config")
        all_rows.extend((label, row) for row in rows)
        results = _read_sweep_files(directory)
        summary = _read_sweep_summary(directory)
        if summary is None:
            interrupted_runs.append(label)
        else:
            plan, criterion, stopped_by_budget = _check_summary(directory, summary, results, sources, pool_families, config_ids)
            settings.add(canonical_json([asdict(plan), criterion]))
            if stopped_by_budget:
                stopped.append(label)
        groups: dict[tuple[str, int, int], list[AttemptRecord]] = {}
        for row in rows:
            groups.setdefault((row.task_id, row.checkpoint_turn, row.rollout_index), []).append(row)
        for task_id, result in results.items():
            settings.add(canonical_json([asdict(result.plan), result.criterion]))
            if task_id in completed:
                raise ValueError(f"the sweep of {task_id} was completed in more than one directory")
            source = sources.get(task_id)
            if source is None:
                raise ValueError(f"{directory}: {task_id} has no source in the baseline run")
            completed[task_id] = (label, result, _verify(directory, result, source, groups))
        swept = {task_id for task_id, _, _ in groups}
        for task_id in sorted(swept - set(results), key=pool_order.index):
            n = len({(k, i) for t, k, i in groups if t == task_id})
            partial_sweeps.append({"run": label, "task_id": task_id, "rollouts": n})
    if len(settings) != 1:
        raise ValueError("the sweep directories used different plans or criteria (or none could be determined)")
    plan_data, criterion = load_strict_json(next(iter(settings)).encode("utf-8"))
    plan = SweepPlan(**plan_data)

    curves: list[SourceCurve] = []
    rollouts: list[_Rollout] = []
    source_models: set[str] = set()
    for task_id in pool_order:
        if task_id not in completed:
            continue
        label, result, verified = completed[task_id]
        source = sources[task_id]
        assert source is not None
        grade = source.grade()
        curves.append(
            SourceCurve(
                task_id=task_id,
                family=pool_families[task_id],
                source_trajectory_id=source.trajectory_id,
                source_success=grade.success,
                source_safe=grade.trajectory_safe,
                n_turns=len(source.turns),
                result=result,
                cliff=cliff_turn(source, result),
                run=label,
            )
        )
        rollouts.extend(verified)
        source_models.update(_models(source, 0))
    model_counts: dict[str, int] = {}
    for rollout in rollouts:
        for model in rollout.models:
            model_counts[model] = model_counts.get(model, 0) + 1
    every_model = set(model_counts) | source_models
    blocked = None
    if len(every_model) > 1:
        blocked = (
            f"sources and sweep rollouts report more than one response_model ({', '.join(sorted(every_model))}); "
            "decision 009 §4 forbids merging them"
        )
    missing = tuple(task_id for task_id, source in sources.items() if source is None)
    not_completed = tuple(task_id for task_id, source in sources.items() if source is not None and task_id not in completed)
    cliff_rate = mean_drop = None
    if blocked is None and curves and {c.family for c in curves} == set(pool_families.values()):
        strata = {c.task_id: c.family for c in curves}
        cliff_rate = stratified_task_bootstrap(
            {c.task_id: float(c.result.is_cliff) for c in curves}, strata, replicates=replicates, seed=0
        )
        mean_drop = stratified_task_bootstrap(
            {c.task_id: c.result.largest_drop for c in curves}, strata, replicates=replicates, seed=0
        )
    gate = None if blocked else assess_gate(tuple(curves), data_complete=not missing and not not_completed)
    windows: dict[str, list[_Rollout]] = {}
    for rollout in rollouts:
        windows.setdefault(rollout.date, []).append(rollout)
    return SweepAnalysis(
        plan=plan,
        criterion=criterion,
        runs=labels,
        interrupted_runs=tuple(interrupted_runs),
        stopped_by_budget=tuple(stopped),
        curves=tuple(curves),
        missing_sources=missing,
        not_completed=not_completed,
        partial_sweeps=tuple(partial_sweeps),
        max_turns_confounded=tuple(c.task_id for c in curves if c.result.max_turns_confounded),
        cliff_rate=cliff_rate,
        mean_largest_drop=mean_drop,
        gate=gate,
        pooling_blocked=blocked,
        response_models=dict(sorted(model_counts.items())),
        time_windows=tuple(
            WindowEstimate(
                date,
                len({r.task_id for r in items}),
                len(items),
                sum(1 for r in items if r.outcome == "success") / len(items),
            )
            for date, items in sorted(windows.items())
        ),
        partial_safety=tuple(
            {
                "run": label,
                "task_id": row.task_id,
                "k": row.checkpoint_turn,
                "rollout_index": row.rollout_index,
                "attempt": row.attempt,
                "partial_steps": row.partial_steps,
                "partial_forbidden_action_count": row.partial_forbidden_action_count,
                "partial_forbidden_attempt_count": row.partial_forbidden_attempt_count,
            }
            for label, row in all_rows
            if row.outcome == "aborted"
        ),
        cost_usd=math.fsum(row.cost_usd for _, row in all_rows),
        unknown_cost_calls=sum(row.unknown_cost_calls for _, row in all_rows),
    )


def _read_json(path: Path) -> Any:
    try:
        return load_strict_json(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _read_sweep_files(directory: Path) -> dict[str, SweepResult]:
    sweeps = directory / "sweeps"
    results: dict[str, SweepResult] = {}
    if not sweeps.exists():
        return results
    for path in sorted(sweeps.glob("*.json")):
        result = sweep_result_from_dict(_read_json(path))  # 先驗證結構（深層巢狀在此被擋下，不會進到遞迴的比較）
        if result.task_id != path.stem:
            raise ValueError(f"{path} holds the sweep of {result.task_id}")
        results[result.task_id] = result
    return results


def _read_sweep_summary(directory: Path) -> dict[str, Any] | None:
    path = directory / "summary.json"
    if not path.exists():
        return None
    summary = _read_json(path)
    if not isinstance(summary, dict) or summary.get("format") != SWEEP_SUMMARY_FORMAT:
        raise ValueError(f"{directory} is not a sweep run")
    return summary


def _str_list(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value) or len(set(value)) != len(value):
        raise ValueError(f"summary.{what} must be a list of distinct strings")
    return value


def _check_summary(
    directory: Path,
    summary: dict[str, Any],
    results: dict[str, SweepResult],
    sources: dict[str, Trajectory | None],
    pool_families: dict[str, str],
    config_ids: set[str],
) -> tuple[SweepPlan, str, bool]:
    """decision 017 §5.1：summary.json 逐欄驗證並與檔案、source 核對；回傳 (plan, criterion, stopped_by_budget)。"""
    expected_keys = {"format", *(field.name for field in dataclasses.fields(SweepRunSummary))}
    if set(summary) != expected_keys:
        raise ValueError(f"{directory}: summary.json must have exactly the keys {sorted(expected_keys)}")
    plan = _build(SweepPlan, _fields(summary["plan"], SweepPlan, "plan"))
    criterion = summary["criterion"]
    if criterion not in CLIFF_CRITERIA:
        raise ValueError(f"{directory}: summary.criterion must be one of {CLIFF_CRITERIA}")
    if summary["config_id"] not in config_ids:
        raise ValueError(f"{directory}: summary.config_id differs from the sources' agent config")
    listed = _str_list(summary["sources"], "sources")
    missing = _str_list(summary["missing_sources"], "missing_sources")
    done = _str_list(summary["completed"], "completed")
    if any(task_id not in pool_families for task_id in listed):
        raise ValueError(f"{directory}: summary.sources lists a task outside the eval pool")
    if missing != [task_id for task_id in listed if sources[task_id] is None]:
        raise ValueError(f"{directory}: summary.missing_sources disagrees with the baseline run")
    if set(done) != set(results) or not set(done) <= set(listed) - set(missing):
        raise ValueError(f"{directory}: summary.completed disagrees with the sweep files")
    stored = summary["results"]
    if not isinstance(stored, dict) or set(stored) != set(done):
        raise ValueError(f"{directory}: summary.results must hold exactly the completed sweeps")
    for task_id in done:
        if sweep_result_from_dict(stored[task_id]) != results[task_id]:
            raise ValueError(f"{directory}: the sweep file of {task_id} does not match the summary")
    families = summary["families"]
    if not isinstance(families, dict) or set(families) != set(listed) or any(families[t] != pool_families[t] for t in listed):
        raise ValueError(f"{directory}: summary.families disagrees with the eval pool")
    _bootstrap(summary["cliff_rate"])
    _bootstrap(summary["mean_largest_drop"])
    spent = summary["spent_usd"]
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not _is_finite(spent) or spent < 0:
        raise ValueError(f"{directory}: summary.spent_usd must be a finite number >= 0")
    for name in ("model_calls", "unknown_cost_calls"):
        if type(summary[name]) is not int or summary[name] < 0:
            raise ValueError(f"{directory}: summary.{name} must be an int >= 0")
    if not isinstance(summary["stopped_by_budget"], bool):
        raise ValueError(f"{directory}: summary.stopped_by_budget must be a bool")
    return plan, criterion, summary["stopped_by_budget"]


def _models(trajectory: Trajectory, start: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                turn.meta["response_model"]
                for turn in trajectory.turns[start:]
                if isinstance(turn.meta, dict) and isinstance(turn.meta.get("response_model"), str)
            }
        )
    )


def _verify(
    directory: Path,
    result: SweepResult,
    source: Trajectory,
    groups: dict[tuple[str, int, int], list[AttemptRecord]],
) -> list[_Rollout]:
    """decision 017 §5.2：完成的 sweep 與 manifest、存檔的 rollout 重算核對；回傳每個 rollout 的 outcome、UTC 日期與 models。"""
    task_id = result.task_id
    where = f"{directory}: sweep of {task_id}"
    if result.source_trajectory_id != source.trajectory_id:
        raise ValueError(f"{where} does not belong to the baseline run's source")
    by_key = {(r.k, r.rollout_index): r for r in result.rollouts}
    keys = {(k, i) for t, k, i in groups if t == task_id}
    if len(by_key) != len(result.rollouts) or set(by_key) != keys:
        raise ValueError(f"{where}: its rollouts do not match the manifest")
    verified: list[_Rollout] = []
    for rollout in result.rollouts:
        k, i = rollout.k, rollout.rollout_index
        attempts = sorted(groups[(task_id, k, i)], key=lambda row: row.attempt)
        last = attempts[-1]
        if [row.attempt for row in attempts] != list(range(1, len(attempts) + 1)) or any(
            row.outcome == "completed" or row.error_class != "infrastructure" for row in attempts[:-1]
        ):
            raise ValueError(f"{where}: rollout k={k} r{i} has inconsistent attempts")
        if rollout.attempts != last.attempt or not 0 <= k < len(source.turns):
            raise ValueError(f"{where}: rollout k={k} r{i} disagrees with the manifest")
        models: tuple[str, ...] = ()
        if last.outcome == "completed":
            path = directory / "rollouts" / task_id / f"k{k}-r{i}.json"
            try:
                trajectory = Trajectory.load(path)
                trajectory.verify()
            except (OSError, TrajectoryError) as exc:
                raise ValueError(f"cannot load {path}: {exc}") from exc
            parent = {"trajectory_id": source.trajectory_id, "checkpoint_id": last.checkpoint_id, "turn": k}
            if trajectory.trajectory_id != last.trajectory_id or canonical_json(trajectory.parent) != canonical_json(parent):
                raise ValueError(f"{path} does not match the manifest")
            if trajectory.termination != last.termination or rollout.termination != trajectory.termination:
                raise ValueError(f"{path}: termination disagrees with the sweep result or the manifest")
            if classify_rollout(trajectory, k) != (rollout.outcome, rollout.stopped_without_action):
                raise ValueError(f"{path}: the recorded outcome does not match the trajectory")
            models = _models(trajectory, k)
        else:
            if last.error_class == "agent_output":
                expected = "agent_error"
            elif last.error_class == "infrastructure" and last.attempt >= MAX_EPISODE_ATTEMPTS:
                expected = "aborted"
            else:
                raise ValueError(f"{where}: rollout k={k} r{i} never finished; the sweep cannot be complete")
            if (rollout.outcome, rollout.termination, rollout.stopped_without_action) != (expected, None, False):
                raise ValueError(f"{where}: rollout k={k} r{i} is labelled differently from the manifest")
        verified.append(_Rollout(task_id, rollout.outcome, utc_date(last.utc_time), models))

    def replay(k: int, i: int) -> RolloutOutcome:
        if (k, i) not in by_key:
            raise ValueError(f"{where}: the protocol needs rollout k={k} r{i}, which is missing")
        return by_key[(k, i)]

    protocol = run_protocol(len(source.turns) - 1, result.plan, replay)
    if protocol.order != result.rollouts or protocol.refined_intervals != result.refined_intervals:
        raise ValueError(f"{where}: the rollouts or refine intervals do not follow the protocol")
    estimates = tuple(estimate_checkpoint(k, list(protocol.outcomes[k])) for k in sorted(protocol.outcomes))
    if estimates != result.estimates:
        raise ValueError(f"{where}: the estimates do not match the rollouts")
    assessed = assess_drop(estimates, result.plan.drop_threshold, criterion=result.criterion)
    recorded = (result.largest_drop, result.largest_drop_k, result.max_turns_confounded, result.is_cliff)
    if recorded != (assessed.largest_drop, assessed.largest_drop_k, assessed.max_turns_confounded, assessed.is_cliff):
        raise ValueError(f"{where}: the drop / cliff verdict does not match the estimates")
    return verified


def _verdict(result: SweepResult) -> str:
    if result.is_cliff:
        return "cliff"
    if result.max_turns_confounded:
        return "no cliff (max_turns confounded)"
    return "no cliff"


def curve_svg(curve: SourceCurve) -> str:
    """單一 source 的 recoverability curve SVG（完整的 ``<svg ...>...</svg>`` 字串，決定性輸出）：x = checkpoint k，
    y = P(success | k) ∈ [0, 1]；每個已估計的 k 畫 p̂ 點與 Wilson CI 線段；cliff 時以虛線標出 k* → k* + 1；標題含 task_id、
    判定（cliff / no cliff / max_turns 混淆）與 criterion。不依賴任何外部繪圖套件。"""
    result = curve.result
    width, height, left, right, top, bottom = 640, 320, 56, 24, 40, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    last = max([e.k for e in result.estimates] + [1])

    def x(k: float) -> float:
        return left + plot_w * k / last

    def y(p: float) -> float:
        return top + plot_h * (1 - p)

    title = f"{_escape(curve.task_id)} — P(success | checkpoint k) — {_verdict(result)} ({_escape(result.criterion)})"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="22" font-size="14">{title}</text>',
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{left + plot_w}" y2="{y(0):.1f}" stroke="black"/>',
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{left}" y2="{y(1):.1f}" stroke="black"/>',
    ]
    for tick in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
        parts.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + plot_w}" y2="{y(tick):.1f}" stroke="#ddd"/>')
    for e in result.estimates:
        parts.append(f'<text x="{x(e.k):.1f}" y="{y(0) + 16:.1f}" text-anchor="middle">{e.k}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">checkpoint k</text>')
    if result.is_cliff and result.largest_drop_k is not None:
        for k in (result.largest_drop_k, result.largest_drop_k + 1):
            parts.append(
                f'<line x1="{x(k):.1f}" y1="{y(1):.1f}" x2="{x(k):.1f}" y2="{y(0):.1f}" stroke="#c00" stroke-dasharray="4 3"/>'
            )
    points = " ".join(f"{x(e.k):.1f},{y(e.p_hat):.1f}" for e in result.estimates)
    if len(result.estimates) > 1:
        parts.append(f'<polyline points="{points}" fill="none" stroke="#36c"/>')
    for e in result.estimates:
        parts.append(
            f'<line x1="{x(e.k):.1f}" y1="{y(e.ci_low):.1f}" x2="{x(e.k):.1f}" y2="{y(e.ci_high):.1f}" stroke="#36c" stroke-width="2"/>'
        )
        parts.append(f'<circle cx="{x(e.k):.1f}" cy="{y(e.p_hat):.1f}" r="4" fill="#36c"><title>k={e.k} n={e.n} p={e.p_hat:.3f}</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _count(value: int | None, steps: int | None) -> str:
    if value is not None:
        return str(value)
    return "未知（部分評分失敗）" if steps is not None else "—（沒有部分軌跡）"


def render_markdown(analysis: SweepAnalysis) -> str:
    """人讀的報告：plan 與 criterion、目錄（中斷 / 預算停止）、每個 source 的曲線表（k、n、p̂、CI、p̂_completed、各類計數、
    max_turns 比例、stopped_without_action）、largest drop / k* / 判定（含 max_turns 混淆）與 cliff turn 模式、缺漏 / 未完成 /
    部分完成的 sweep、cliff 比例與平均最大下降（CI）、Gate 評估（標示提案規則、未經確認、資料是否完整）、response_model、時間窗、
    中止嘗試的部分安全指標、成本，以及「15 題只能當初步觀察」的註記。"""
    a = analysis
    p = a.plan
    lines = [
        "# eval sweep 分析（decision 010 / 012 / 013 / 016 / 017）",
        "",
        "> eval 只有 15 題：以下只能當 infrastructure validation / 初步觀察，不得宣稱統計上的泛化。",
        "",
        f"- plan：stride {p.coarse_stride}、n_coarse {p.n_coarse}、n_fine {p.n_fine}、drop_threshold {p.drop_threshold}；"
        f"驟降判定 criterion：{a.criterion}",
        f"- sweep 目錄：{', '.join(a.runs)}；中斷（沒有 summary）：{', '.join(a.interrupted_runs) or '—'}；"
        f"因預算停止：{', '.join(a.stopped_by_budget) or '—'}",
        f"- 缺漏的 source：{', '.join(a.missing_sources) or '—'}；未完成的 sweep：{', '.join(a.not_completed) or '—'}",
        "- 部分完成（不進曲線，成本與部分安全指標照計）："
        + (", ".join(f"{e['run']} / {e['task_id']}（{e['rollouts']} rollouts）" for e in a.partial_sweeps) or "—"),
    ]
    if a.pooling_blocked:
        lines.append(f"- **不報彙總、不評估 Gate**：{a.pooling_blocked}")
    lines.append("")
    for curve in a.curves:
        r = curve.result
        lines += [
            f"## {curve.task_id}（{curve.family}；source success {curve.source_success}、safe {curve.source_safe}、"
            f"{curve.n_turns} turns；{curve.run}）",
            "",
            "| k | n | p̂ | 95% CI | p̂_completed | failure | refusal | agent_error | aborted | max_turns | stopped_without_action |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in r.estimates:
            completed = "—" if e.p_hat_completed is None else f"{e.p_hat_completed:.3f}"
            rate = e.max_turns_terminations / e.n if e.n else 0.0
            lines.append(
                f"| {e.k} | {e.n} | {e.p_hat:.3f} | [{e.ci_low:.3f}, {e.ci_high:.3f}] | {completed} | {e.failures} | {e.refusals} | "
                f"{e.agent_errors} | {e.aborted} | {rate:.3f} | {e.stopped_without_action} |"
            )
        cliff = "—" if curve.cliff is None else f"k* = {curve.cliff.k}，模式 `{pattern_key(curve.cliff.pattern)}`"
        note = "；**下降合格但 max_turns 比例上升：max_turns 混淆，不認定為 cliff（decision 006）**" if r.max_turns_confounded else ""
        lines += ["", f"largest drop {r.largest_drop:.3f}（k = {r.largest_drop_k}），cliff：{r.is_cliff}；{cliff}{note}", ""]
    lines += ["## 彙總（family 分層 task-level bootstrap）", ""]
    for name, value in (("cliff 比例", a.cliff_rate), ("平均最大下降", a.mean_largest_drop)):
        if value is None:
            reason = "response_model 不只一種" if a.pooling_blocked else "有 family 沒有完成的 sweep"
            lines.append(f"- {name}：—（{reason}）")
        else:
            lines.append(f"- {name}：{value.estimate:.3f}（95% CI [{value.low:.3f}, {value.high:.3f}]，{value.n_tasks} tasks）")
    lines.append(f"- max_turns 混淆（不計為 cliff）：{', '.join(a.max_turns_confounded) or '—'}")
    lines += ["", "## Gate 評估（提案規則，未經使用者確認）", ""]
    if a.gate is None:
        lines.append("- 不評估：response_model 不只一種（decision 009 §4）")
    else:
        lines.append(f"- 規則：{a.gate.rule}")
        for key, tasks in a.gate.patterns.items():
            lines.append(f"- `{key}`：tasks {', '.join(tasks)}；families {', '.join(a.gate.families[key])}")
        lines.append(f"- 是否滿足（提案規則下）：{a.gate.met}")
        if not a.gate.data_complete:
            lines.append("- **資料不完整**：有缺漏或未完成的 source，判定只涵蓋已完成的曲線")
    lines += [
        "",
        "## response_model 與時間窗",
        "",
        f"- response_model（出現該 model 的完成 sweep rollout 數）：`{canonical_json(a.response_models)}`",
        "",
        "| UTC date | n sources | n rollouts | success rate |",
        "|---|---|---|---|",
    ]
    for w in a.time_windows:
        lines.append(f"| {w.date} | {w.n_sources} | {w.n_rollouts} | {w.success_rate:.3f} |")
    lines += ["", "## 中止嘗試的部分安全指標（每個中止嘗試一筆；partial steps 含 checkpoint 前綴）", ""]
    for e in a.partial_safety:
        steps = e["partial_steps"]
        lines.append(
            f"- {e['run']} / {e['task_id']} k={e['k']} r{e['rollout_index']} attempt {e['attempt']}："
            f"partial steps {'—' if steps is None else steps}、"
            f"forbidden actions {_count(e['partial_forbidden_action_count'], steps)}、"
            f"forbidden attempts {_count(e['partial_forbidden_attempt_count'], steps)}"
        )
    if not a.partial_safety:
        lines.append("- （無）")
    lines += [
        "",
        f"成本（所有目錄 manifest 加總）：${a.cost_usd:.4f}；成本無法量測的已計費 call：{a.unknown_cost_calls}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_runs", nargs="+")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    analysis = analyze_sweep_runs(args.sweep_runs, args.baseline_run)
    out = Path(args.out)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(canonical_json(analysis.to_dict()) + "\n", encoding="utf-8")
    (out / "analysis.md").write_text(render_markdown(analysis), encoding="utf-8")
    for curve in analysis.curves:
        (out / "curves" / f"{curve.task_id}.svg").write_text(curve_svg(curve), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
