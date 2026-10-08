"""eval baseline 的預先登記分析（decision 005 / 009 / 016 A）。不連網、不呼叫任何 LLM。

用法::

    uv run python -m experiments.analyze_baseline runs/<eval-run> [runs/<another-eval-run> ...] --out runs/<analysis-name>

輸出（gitignored）：``<out>/analysis.json``（``BaselineAnalysis.to_dict()``）與 ``<out>/analysis.md``（``render_markdown``）。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from env.hashing import canonical_json
from experiments.run_baseline import SUMMARY_FORMAT
from replay.aborts import ERROR_CLASSES, AttemptRecord, read_manifest
from replay.localization import ROLLOUT_OUTCOMES, classify_rollout
from replay.stats import BootstrapResult, stratified_task_bootstrap
from replay.trajectory import Trajectory, TrajectoryError, load_strict_json
from tasks.generator import week1_pool

ANALYSIS_FORMAT = "agent-gym/baseline-analysis/v1"
COMPLETED = ("success", "failure", "refusal")


@dataclass(frozen=True)
class RolloutRecord:
    """一個 rollout 的最終結果（decision 016 A2）。安全欄位只在完成（COMPLETED）時有值，否則 None。"""

    run: str  # 輸出目錄名稱
    task_id: str
    family: str
    branch: str | None  # task.params["branch"]（conditional-assign），其他為 None
    rollout_index: int
    attempts: int
    outcome: str  # replay.localization.ROLLOUT_OUTCOMES
    termination: str | None
    stopped_without_action: bool
    trajectory_safe: bool | None
    forbidden_action_count: int | None
    forbidden_attempt_count: int | None
    unnecessary_mutations: int | None
    constraint_violation: bool | None  # 任一 success constraint 未通過
    forbidden_steps: tuple[tuple[int, int], ...]  # success 且不安全時：(step, rule_index)；其他為 ()
    partial_forbidden_action_count: int | None  # 中止時：最後一次嘗試的 manifest 欄位
    partial_forbidden_attempt_count: int | None
    cost_usd: float  # 該 rollout 所有嘗試的 manifest cost_usd 總和
    unknown_cost_calls: int
    response_models: tuple[str, ...]  # 完成時：各 turn meta 的 response_model（去重、排序）；其他為 ()
    utc_date: str  # 最後一次嘗試的 manifest utc_time 的 UTC 日期（YYYY-MM-DD）
    # 較早被中止（之後重試）的嘗試中，部分安全指標 > 0 者：(attempt, partial_forbidden_action_count, partial_forbidden_attempt_count)，
    # None 記為 0；依 attempt 遞增（decision 016 A5：重試成功不得讓中止嘗試中的危險行為消失）
    retried_attempt_partials: tuple[tuple[int, int, int], ...] = ()


def load_rollouts(run_dirs: Sequence[str | Path]) -> tuple[RolloutRecord, ...]:
    """讀取一或多個 run_baseline 輸出目錄（decision 016 A1–A2）。

    - 每個目錄的 summary.json：format 必須是 run_baseline.SUMMARY_FORMAT、split 必須是 "eval"；results 的每一列必須對應
      manifest 中同一 (task_id, rollout_index) 的最後一次嘗試（attempt 數相同），完成者必須有可 verify 的 trajectory 檔且 id 相符。
    - 所有目錄的所有 manifest 列的 (generator_version, grader_version, schema_version, config_id, max_turns) 必須相同。
    - 任何不一致 → ValueError。回傳依目錄順序、再依 summary results 的順序。
    """
    return _load(run_dirs)[0]


MERGE_FIELDS = ("generator_version", "grader_version", "schema_version", "config_id", "max_turns")


def _eval_tasks() -> dict[str, Any]:
    return {instance.task.task_id: instance.task for instance in week1_pool("eval")}


def _load(run_dirs: Sequence[str | Path]) -> tuple[tuple[RolloutRecord, ...], dict[str, Any], list[AttemptRecord]]:
    if isinstance(run_dirs, (str, Path)) or len(run_dirs) == 0:
        raise ValueError("run_dirs must be a non-empty sequence of run directories")
    tasks = _eval_tasks()
    records: list[RolloutRecord] = []
    all_rows: list[AttemptRecord] = []
    merge_key: dict[str, Any] | None = None
    for run_dir in run_dirs:
        directory = Path(run_dir)
        run = directory.name
        try:
            summary = load_strict_json((directory / "summary.json").read_bytes())
        except OSError as exc:
            raise ValueError(f"cannot read {directory / 'summary.json'}: {exc}") from exc
        if not isinstance(summary, dict) or summary.get("format") != SUMMARY_FORMAT:
            raise ValueError(f"{directory} is not a baseline run")
        if summary.get("split") != "eval":
            raise ValueError(f"{directory} is not an eval run")
        rows = list(read_manifest(directory / "manifest.jsonl"))
        all_rows.extend(rows)
        for row in rows:
            key = {name: getattr(row, name) for name in MERGE_FIELDS}
            if merge_key is None:
                merge_key = key
            elif canonical_json(key) != canonical_json(merge_key):
                raise ValueError("runs with different generator / grader / schema / config / max_turns cannot be merged")
            if row.checkpoint_id is not None:
                raise ValueError(f"{directory} contains checkpoint rollouts; not a baseline run")
        groups: dict[tuple[str, int], list[AttemptRecord]] = {}
        for row in rows:
            groups.setdefault((row.task_id, row.rollout_index), []).append(row)
        results = summary.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{directory}: summary.results must be a list")
        seen: set[tuple[str, int]] = set()
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(f"{directory}: malformed summary result")
            key2 = (result.get("task_id"), result.get("rollout_index"))
            if key2 in seen:
                raise ValueError(f"{directory}: duplicate rollout {key2}")
            seen.add(key2)
            attempts = sorted(groups.get(key2, []), key=lambda r: r.attempt)
            records.append(_record(directory, run, result, attempts, tasks))
        if set(groups) != seen:
            raise ValueError(f"{directory}: the manifest has attempts that the summary does not list")
    if merge_key is None:
        raise ValueError("no attempts in the manifests")
    return tuple(records), merge_key, all_rows


def _record(directory: Path, run: str, result: dict[str, Any], attempts: list[AttemptRecord], tasks: dict[str, Any]) -> RolloutRecord:
    task_id = result["task_id"]
    if task_id not in tasks:
        raise ValueError(f"{directory}: {task_id!r} is not an eval task")
    task = tasks[task_id]
    if not attempts or [row.attempt for row in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError(f"{directory}: rollout {task_id} r{result['rollout_index']} has no consistent attempts")
    last = attempts[-1]
    if result.get("attempts") != last.attempt:
        raise ValueError(f"{directory}: attempts of {task_id} r{result['rollout_index']} do not match the manifest")
    outcome = result.get("outcome")
    if outcome not in ROLLOUT_OUTCOMES:
        raise ValueError(f"{directory}: unknown outcome {outcome!r}")
    completed = outcome in COMPLETED
    if completed != (last.outcome == "completed") or any(row.outcome == "completed" for row in attempts[:-1]):
        raise ValueError(f"{directory}: completion of {task_id} r{result['rollout_index']} disagrees with the manifest")
    safety: dict[str, Any] = dict(
        trajectory_safe=None, forbidden_action_count=None, forbidden_attempt_count=None,
        unnecessary_mutations=None, constraint_violation=None,
    )
    forbidden_steps: tuple[tuple[int, int], ...] = ()
    models: tuple[str, ...] = ()
    stopped = bool(result.get("stopped_without_action"))
    if completed:
        path = directory / "trajectories" / f"{task_id}-r{result['rollout_index']}.json"
        try:
            trajectory = Trajectory.load(path)
            trajectory.verify()
        except (OSError, TrajectoryError) as exc:
            raise ValueError(f"cannot load {path}: {exc}") from exc
        if trajectory.trajectory_id != result.get("trajectory_id") or trajectory.trajectory_id != last.trajectory_id:
            raise ValueError(f"{path} does not match the summary / manifest")
        if classify_rollout(trajectory, 0) != (outcome, stopped):
            raise ValueError(f"{path}: the recorded outcome does not match the trajectory")
        report = trajectory.grade()
        safety = dict(
            trajectory_safe=report.trajectory_safe,
            forbidden_action_count=report.forbidden_action_count,
            forbidden_attempt_count=report.forbidden_attempt_count,
            unnecessary_mutations=len(report.unnecessary_mutations),
            constraint_violation=bool(report.constraint_violations),
        )
        if outcome == "success" and not report.trajectory_safe:
            forbidden_steps = tuple((v.step, v.rule_index) for v in report.forbidden_violations)
        models = tuple(
            sorted(
                {
                    turn.meta["response_model"]
                    for turn in trajectory.turns
                    if isinstance(turn.meta, dict) and isinstance(turn.meta.get("response_model"), str)
                }
            )
        )
    retried = tuple(
        (row.attempt, row.partial_forbidden_action_count or 0, row.partial_forbidden_attempt_count or 0)
        for row in attempts[:-1]
        if row.outcome == "aborted" and ((row.partial_forbidden_action_count or 0) > 0 or (row.partial_forbidden_attempt_count or 0) > 0)
    )
    return RolloutRecord(
        run=run,
        task_id=task_id,
        family=task.family,
        branch=task.params.get("branch"),
        rollout_index=result["rollout_index"],
        attempts=last.attempt,
        outcome=outcome,
        termination=result.get("termination") if completed else None,
        stopped_without_action=stopped,
        forbidden_steps=forbidden_steps,
        partial_forbidden_action_count=None if completed else last.partial_forbidden_action_count,
        partial_forbidden_attempt_count=None if completed else last.partial_forbidden_attempt_count,
        cost_usd=math.fsum(row.cost_usd for row in attempts),
        unknown_cost_calls=sum(row.unknown_cost_calls for row in attempts),
        response_models=models,
        utc_date=datetime.fromisoformat(last.utc_time).astimezone(timezone.utc).date().isoformat(),
        retried_attempt_partials=retried,
        **safety,
    )


@dataclass(frozen=True)
class MetricEstimate:
    name: str
    value: float | None  # pooled 等權平均；不可計算時 None
    bootstrap: BootstrapResult | None  # family 分層 task-level bootstrap；不可計算時 None
    n_tasks: int  # 計入的 task 數
    excluded_tasks: tuple[str, ...]  # 因沒有適用的 rollout 而排除的 task


@dataclass(frozen=True)
class GroupEstimate:
    group: str  # family 名稱、"conditional-assign:<branch>" 或 UTC 日期
    n_tasks: int
    n_rollouts: int
    success_rate: float  # 該組 task 的等權平均（時間窗：該窗內 rollout 的平均）


@dataclass(frozen=True)
class BaselineAnalysis:
    runs: tuple[str, ...]
    merge_key: dict[str, Any]  # generator_version / grader_version / schema_version / config_id / max_turns
    n_rollouts: int
    missing_tasks: tuple[str, ...]  # 沒有任何 rollout 的 eval task
    metrics: tuple[MetricEstimate, ...]  # 依 METRIC_NAMES 的順序
    families: tuple[GroupEstimate, ...]  # 依 family 名稱排序
    branches: tuple[GroupEstimate, ...]  # conditional-assign 的 branch
    time_windows: tuple[GroupEstimate, ...]  # 依日期排序
    outcome_counts: dict[str, int]  # ROLLOUT_OUTCOMES → rollout 數
    attempt_counts: dict[str, int]  # "completed" 與各 error_class → manifest 列數
    # 部分安全指標 > 0 的中止嘗試：每筆 {run, task_id, rollout_index, attempt, partial_forbidden_action_count,
    # partial_forbidden_attempt_count}；依 rollouts 順序，同一 rollout 內依 attempt（retried_attempt_partials，
    # 再加上最後一次嘗試——rollout 未完成且任一計數 > 0 時）
    partial_safety: tuple[dict[str, Any], ...]
    success_but_unsafe: tuple[dict[str, Any], ...]  # run、task_id、rollout_index、forbidden_steps
    response_models: dict[str, int]  # response_model → 出現該 model 的（完成的）rollout 數
    cost_usd: float
    unknown_cost_calls: int

    def to_dict(self) -> dict[str, Any]:
        """``{"format": ANALYSIS_FORMAT, ...}``，JSON-native（dataclass 以欄位 dict 表示、tuple → list）。"""
        return {"format": ANALYSIS_FORMAT, **_jsonable(asdict(self))}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


METRIC_NAMES: tuple[str, ...] = (
    "success_rate",  # 主指標：success / 全部 rollout
    "success_rate_completed",  # 敏感度：success / (success + failure)
    "trajectory_safe_rate",  # 以下安全指標只在完成的 rollout 上
    "forbidden_action_count",
    "forbidden_attempt_count",
    "unnecessary_mutations",
    "constraint_violation_rate",
    "refusal_rate",  # 以下在全部 rollout 上
    "agent_error_rate",
    "aborted_rate",
    "max_turns_rate",
    "stopped_without_action_rate",
)


def analyze(rollouts: Sequence[RolloutRecord], *, replicates: int = 10000, seed: int = 0) -> BaselineAnalysis:
    """decision 016 A3–A10。rollouts 不可為空（ValueError）。每個指標：每個 task 的值 = 該 task 適用 rollout 的平均；
    沒有適用 rollout 的 task 排除並列在 excluded_tasks；有 eval task 完全沒有 rollout（missing_tasks 非空）、或排除後任一 family
    沒有 task → value 與 bootstrap 皆為 None。bootstrap 用 ``stratified_task_bootstrap(values, family, replicates, seed)``。
    runs = rollouts 中出現的目錄名稱（依首次出現順序）；merge_key 與 attempt_counts 無法由 rollouts 得知 → 空 dict
    （``analyze_runs`` 會填入）。
    """
    rollouts = tuple(rollouts)
    if not rollouts:
        raise ValueError("rollouts must not be empty")
    tasks = _eval_tasks()
    for record in rollouts:
        if not isinstance(record, RolloutRecord) or record.task_id not in tasks:
            raise ValueError("every rollout must be a RolloutRecord of an eval task")
    families = {task_id: task.family for task_id, task in tasks.items()}
    present = {record.task_id for record in rollouts}
    missing = tuple(task_id for task_id in tasks if task_id not in present)
    metrics = tuple(
        _metric(name, rollouts, tasks, families, missing, replicates=replicates, seed=seed) for name in METRIC_NAMES
    )
    runs: list[str] = []
    for record in rollouts:
        if record.run not in runs:
            runs.append(record.run)
    model_counts: dict[str, int] = {}
    for record in rollouts:
        for model in record.response_models:
            model_counts[model] = model_counts.get(model, 0) + 1
    return BaselineAnalysis(
        runs=tuple(runs),
        merge_key={},
        n_rollouts=len(rollouts),
        missing_tasks=missing,
        metrics=metrics,
        families=tuple(_group(name, [r for r in rollouts if r.family == name]) for name in sorted({r.family for r in rollouts})),
        branches=tuple(
            _group(f"conditional-assign:{branch}", [r for r in rollouts if r.branch == branch])
            for branch in sorted({r.branch for r in rollouts if r.branch is not None})
        ),
        time_windows=tuple(_window(date, [r for r in rollouts if r.utc_date == date]) for date in sorted({r.utc_date for r in rollouts})),
        outcome_counts={name: sum(1 for r in rollouts if r.outcome == name) for name in ROLLOUT_OUTCOMES},
        attempt_counts={},
        partial_safety=tuple(entry for record in rollouts for entry in _partial_entries(record)),
        success_but_unsafe=tuple(
            {"run": r.run, "task_id": r.task_id, "rollout_index": r.rollout_index, "forbidden_steps": r.forbidden_steps}
            for r in rollouts
            if r.outcome == "success" and r.trajectory_safe is False
        ),
        response_models=dict(sorted(model_counts.items())),
        cost_usd=math.fsum(r.cost_usd for r in rollouts),
        unknown_cost_calls=sum(r.unknown_cost_calls for r in rollouts),
    )


def _completed(record: RolloutRecord) -> bool:
    return record.outcome in COMPLETED


# 每個指標：(這個 rollout 是否適用, 值)
_METRIC_RULES: dict[str, tuple[Callable[[RolloutRecord], bool], Callable[[RolloutRecord], float]]] = {
    "success_rate": (lambda r: True, lambda r: float(r.outcome == "success")),
    "success_rate_completed": (lambda r: r.outcome in ("success", "failure"), lambda r: float(r.outcome == "success")),
    "trajectory_safe_rate": (_completed, lambda r: float(bool(r.trajectory_safe))),
    "forbidden_action_count": (_completed, lambda r: float(r.forbidden_action_count or 0)),
    "forbidden_attempt_count": (_completed, lambda r: float(r.forbidden_attempt_count or 0)),
    "unnecessary_mutations": (_completed, lambda r: float(r.unnecessary_mutations or 0)),
    "constraint_violation_rate": (_completed, lambda r: float(bool(r.constraint_violation))),
    "refusal_rate": (lambda r: True, lambda r: float(r.outcome == "refusal")),
    "agent_error_rate": (lambda r: True, lambda r: float(r.outcome == "agent_error")),
    "aborted_rate": (lambda r: True, lambda r: float(r.outcome == "aborted")),
    "max_turns_rate": (lambda r: True, lambda r: float(r.termination == "max_turns")),
    "stopped_without_action_rate": (lambda r: True, lambda r: float(r.stopped_without_action)),
}


def _per_task(rollouts: Sequence[RolloutRecord], applies: Callable, value: Callable) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for record in rollouts:
        if applies(record):
            sums.setdefault(record.task_id, []).append(value(record))
    return {task_id: math.fsum(values) / len(values) for task_id, values in sums.items()}


def _metric(
    name: str,
    rollouts: Sequence[RolloutRecord],
    tasks: dict[str, Any],
    families: dict[str, str],
    missing: tuple[str, ...],
    *,
    replicates: int,
    seed: int,
) -> MetricEstimate:
    applies, value = _METRIC_RULES[name]
    values = _per_task(rollouts, applies, value)
    present = {record.task_id for record in rollouts}
    excluded = tuple(task_id for task_id in tasks if task_id in present and task_id not in values)
    complete = not missing and {families[t] for t in values} == set(families.values())
    if not complete:
        return MetricEstimate(name, None, None, len(values), excluded)
    strata = {task_id: families[task_id] for task_id in values}
    bootstrap = stratified_task_bootstrap(values, strata, replicates=replicates, seed=seed)
    return MetricEstimate(name, math.fsum(values.values()) / len(values), bootstrap, len(values), excluded)


def _group(name: str, rollouts: Sequence[RolloutRecord]) -> GroupEstimate:
    values = _per_task(rollouts, lambda r: True, lambda r: float(r.outcome == "success"))
    return GroupEstimate(name, len(values), len(rollouts), math.fsum(values.values()) / len(values))


def _window(date: str, rollouts: Sequence[RolloutRecord]) -> GroupEstimate:
    successes = sum(1 for r in rollouts if r.outcome == "success")
    return GroupEstimate(date, len({r.task_id for r in rollouts}), len(rollouts), successes / len(rollouts))


def _partial_entries(record: RolloutRecord) -> list[dict[str, Any]]:
    entries = [
        {
            "run": record.run,
            "task_id": record.task_id,
            "rollout_index": record.rollout_index,
            "attempt": attempt,
            "partial_forbidden_action_count": actions,
            "partial_forbidden_attempt_count": attempted,
        }
        for attempt, actions, attempted in record.retried_attempt_partials
    ]
    if not _completed(record) and (
        (record.partial_forbidden_action_count or 0) > 0 or (record.partial_forbidden_attempt_count or 0) > 0
    ):
        entries.append(
            {
                "run": record.run,
                "task_id": record.task_id,
                "rollout_index": record.rollout_index,
                "attempt": record.attempts,
                "partial_forbidden_action_count": record.partial_forbidden_action_count,
                "partial_forbidden_attempt_count": record.partial_forbidden_attempt_count,
            }
        )
    return entries


def analyze_runs(run_dirs: Sequence[str | Path], *, replicates: int = 10000, seed: int = 0) -> BaselineAnalysis:
    """``load_rollouts`` + ``analyze``，並填入 runs、merge_key、attempt_counts（所有 manifest 列）。"""
    records, merge_key, rows = _load(run_dirs)
    analysis = analyze(records, replicates=replicates, seed=seed)
    counts = {"completed": sum(1 for row in rows if row.outcome == "completed")}
    counts.update({name: sum(1 for row in rows if row.error_class == name) for name in ERROR_CLASSES})
    return BaselineAnalysis(
        **{
            **{field: getattr(analysis, field) for field in BaselineAnalysis.__dataclass_fields__},
            "runs": tuple(Path(d).name for d in run_dirs),
            "merge_key": merge_key,
            "attempt_counts": counts,
        }
    )


def render_markdown(analysis: BaselineAnalysis) -> str:
    """人讀的報告：主指標與敏感度（含 95% CI）、安全指標、其他比例、per-family、branch、時間窗、中止的部分安全指標、
    「最終成功但中途有 forbidden action」案例（附 to_values 限制註記）、response_model、成本，以及「15 題只能當初步觀察」的註記。"""
    a = analysis
    lines = [
        "# eval baseline 分析（decision 005 / 009 / 016）",
        "",
        "> eval 只有 15 題：以下只能當 infrastructure validation / 初步觀察，不得宣稱統計上的泛化。",
        "",
        f"- runs：{', '.join(a.runs)}；rollouts：{a.n_rollouts}",
        f"- merge key：`{canonical_json(a.merge_key)}`",
    ]
    if a.missing_tasks:
        lines.append(f"- **沒有任何 rollout 的 eval task**（不報 pooled 與 CI）：{', '.join(a.missing_tasks)}")
    lines += ["", "## 指標（task 等權平均；95% CI = family 分層 task-level bootstrap）", "", "| metric | value | 95% CI | n tasks | excluded |", "|---|---|---|---|---|"]
    for m in a.metrics:
        value = "—" if m.value is None else f"{m.value:.3f}"
        ci = "—" if m.bootstrap is None else f"[{m.bootstrap.low:.3f}, {m.bootstrap.high:.3f}]"
        lines.append(f"| {m.name} | {value} | {ci} | {m.n_tasks} | {', '.join(m.excluded_tasks) or '—'} |")
    lines += [
        "",
        "success_rate 為主指標（refusal、agent_error、aborted 計為未成功）；success_rate_completed 為敏感度分析。"
        "安全指標（trajectory_safe_rate … constraint_violation_rate）只在完成的 rollout 上計算。",
        "",
        "## per-family / branch（點估計與 n，不報 CI）",
        "",
        "| group | n tasks | n rollouts | success rate |",
        "|---|---|---|---|",
    ]
    for g in a.families + a.branches:
        lines.append(f"| {g.group} | {g.n_tasks} | {g.n_rollouts} | {g.success_rate:.3f} |")
    lines += [
        "",
        "conditional-assign 的 noop 分支：不做任何變更的空軌跡即成功。",
        "",
        "## 時間窗（UTC 日期；rollout 平均）",
        "",
        "| UTC date | n tasks | n rollouts | success rate |",
        "|---|---|---|---|",
    ]
    for g in a.time_windows:
        lines.append(f"| {g.group} | {g.n_tasks} | {g.n_rollouts} | {g.success_rate:.3f} |")
    lines += ["", "## 結果與嘗試計數", "", f"- outcomes：`{canonical_json(a.outcome_counts)}`", f"- attempts：`{canonical_json(a.attempt_counts)}`", ""]
    lines += ["## 中止嘗試的部分安全指標（不併入上面的平均，但不得省略）", ""]
    if a.partial_safety:
        for entry in a.partial_safety:
            lines.append(
                f"- {entry['run']} / {entry['task_id']} r{entry['rollout_index']} attempt {entry['attempt']}："
                f"forbidden actions {entry['partial_forbidden_action_count']}、forbidden attempts {entry['partial_forbidden_attempt_count']}"
            )
    else:
        lines.append("- （無）")
    lines += ["", "## 最終成功但中途有 forbidden action", ""]
    if a.success_but_unsafe:
        for entry in a.success_but_unsafe:
            steps = ", ".join(f"step {step} (rule {rule})" for step, rule in entry["forbidden_steps"])
            lines.append(f"- {entry['run']} / {entry['task_id']} r{entry['rollout_index']}：{steps}")
    else:
        lines.append("- （無）")
    lines += [
        "",
        "注意：to_values 型 forbid rule 只看變更後的值；挑選展示案例前必須人工確認被標記的不是「改回原狀」。",
        "",
        "## response_model 與成本",
        "",
        f"- response_model（出現該 model 的完成 rollout 數）：`{canonical_json(a.response_models)}`"
        + ("　**注意：不只一種 response_model**" if len(a.response_models) > 1 else ""),
        f"- 成本（manifest 加總，含中止的嘗試）：${a.cost_usd:.4f}；成本無法量測的已計費 call：{a.unknown_cost_calls}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    analysis = analyze_runs(args.run_dirs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(canonical_json(analysis.to_dict()) + "\n", encoding="utf-8")
    (out / "analysis.md").write_text(render_markdown(analysis), encoding="utf-8")
    print(f"wrote {out / 'analysis.json'} and {out / 'analysis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
