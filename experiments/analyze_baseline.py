"""eval baseline 的預先登記分析（decision 005 / 009 / 016 A / 017 §4）。不連網、不呼叫任何 LLM。

用法::

    uv run python -m experiments.analyze_baseline runs/<eval-run> [runs/<another-eval-run> ...] --out runs/<analysis-name>

輸出（gitignored）：``<out>/analysis.json``（``BaselineAnalysis.to_dict()``）與 ``<out>/analysis.md``（``render_markdown``）。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from agents.baseline import MAX_EPISODE_ATTEMPTS
from env.hashing import canonical_json
from experiments.run_baseline import SUMMARY_FORMAT, EpisodeResult, RunSummary
from replay.aborts import ERROR_CLASSES, AttemptRecord, read_manifest, utc_date
from replay.localization import ROLLOUT_OUTCOMES, classify_rollout
from replay.stats import BootstrapResult, stratified_task_bootstrap
from replay.trajectory import Trajectory, TrajectoryError, load_strict_json
from tasks.generator import week1_pool

ANALYSIS_FORMAT = "agent-gym/baseline-analysis/v2"  # v2：decision 017 §4
COMPLETED = ("success", "failure", "refusal")
# interrupted：沒有 summary 的目錄中，最後一次嘗試是還沒用完嘗試的 infrastructure 中止（重試在中斷時進行中）或 harness 中止
# （run 因此停止）；decision 017 §4.1
RECORD_OUTCOMES: tuple[str, ...] = (*ROLLOUT_OUTCOMES, "interrupted")


@dataclass(frozen=True)
class RolloutRecord:
    """一個 rollout 的最終結果（decision 016 A2 / 017 §4）。量測與安全欄位只在完成（COMPLETED）時有值，否則 None / ()。"""

    run: str  # 目錄標籤（目錄名稱；名稱重複時為傳入的路徑字串）
    task_id: str
    family: str
    branch: str | None  # task.params["branch"]（conditional-assign），其他為 None
    rollout_index: int
    attempts: int
    outcome: str  # RECORD_OUTCOMES
    termination: str | None
    stopped_without_action: bool
    trajectory_safe: bool | None
    forbidden_action_count: int | None
    forbidden_attempt_count: int | None
    unnecessary_mutations: int | None
    constraint_violation: bool | None  # 任一 success constraint 未通過
    forbidden_steps: tuple[tuple[int, int], ...]  # success 且不安全時：(step, rule_index)；其他為 ()
    cost_usd: float  # 該 rollout 所有嘗試的 manifest cost_usd 總和
    unknown_cost_calls: int
    response_models: tuple[str, ...]  # 完成時：各 turn meta 的 response_model（去重、排序）；其他為 ()
    utc_date: str  # 最後一次嘗試的 manifest utc_time 的 UTC 日期（YYYY-MM-DD）
    # decision 003 / SPEC §4（017 §4.6）：完成時取自 trajectory.grade()
    outcome_correct: bool | None = None
    tool_errors: int | None = None
    steps: int | None = None
    input_tokens: int | None = None  # 無法量測時 None
    output_tokens: int | None = None
    latency_s: float | None = None
    # decision 008 / 009（017 §4.7）：完成時
    blocked_attempts: tuple[tuple[int, int], ...] = ()  # effective == False 的 forbidden attempt：(step, rule_index)
    multi_tool_turns: tuple[int, ...] = ()  # turn meta 的 tool_use_blocks > 1 的 turn index
    # 每一個 aborted 嘗試（含之後重試成功者）：(attempt, partial_steps, partial_forbidden_action_count,
    # partial_forbidden_attempt_count)，None 原樣保留；依 attempt 遞增（decision 016 A5 / 017 §4.4）
    aborted_attempts: tuple[tuple[int, int | None, int | None, int | None], ...] = ()
    utc_time: str = ""  # 最後一次嘗試的 manifest utc_time（原字串；decision 019 §1 依此決定最早結束的 rollout 0）


MERGE_FIELDS = ("generator_version", "grader_version", "schema_version", "config_id", "max_turns")


def _eval_tasks() -> dict[str, Any]:
    return {instance.task.task_id: instance.task for instance in week1_pool("eval")}


@dataclass(frozen=True)
class LoadedRuns:
    records: tuple[RolloutRecord, ...]  # 依目錄順序；有 summary 時依 summary results 的順序，否則依 manifest 首次出現的順序
    merge_key: dict[str, Any]
    rows: tuple[AttemptRecord, ...]  # 所有目錄的所有 manifest 列
    labels: tuple[str, ...]  # 每個目錄的標籤（與輸入順序相同）
    interrupted_runs: tuple[str, ...]  # 沒有 summary.json 的目錄標籤
    trajectories: dict[tuple[str, str, int], Trajectory]  # (標籤, task_id, rollout_index) → 完成的 trajectory（已 verify）


def load_rollouts(run_dirs: Sequence[str | Path]) -> tuple[RolloutRecord, ...]:
    """讀取一或多個 run_baseline 輸出目錄（decision 016 A1–A2、017 §4.1–4.3）；回傳 ``load_runs(run_dirs).records``。"""
    return load_runs(run_dirs).records


def load_runs(run_dirs: Sequence[str | Path]) -> LoadedRuns:
    """讀取一或多個 run_baseline 輸出目錄。任何不一致 → ValueError。

    - 每個目錄必須存在、是目錄且有 ``manifest.jsonl``（decision 018 §4）；同一個目錄（以 device + inode 判定，不比較路徑字串）
      出現兩次 → ValueError。標籤 = 目錄名稱；名稱重複時改用傳入的路徑字串。
    - 所有 manifest 列的 (generator_version, grader_version, schema_version, config_id, max_turns) 必須相同，
      且都不是 checkpoint rollout、task 都屬於 eval pool。
    - 有 summary.json（decision 017 §4.3 / 018 §3）：key 剛好是 run_split 寫出的欄位；format、split = "eval"、n_tasks = eval pool 大小、
      rollouts 為 int >= 1、config_id 與 manifest 相同、stopped_by_budget 為 bool；spent_usd / model_calls / unknown_cost_calls 與
      manifest 的 cost_usd / billed_calls / unknown_cost_calls 加總相符。results 的 (task_id, rollout_index) 依序等於 run_split 的排程
      （rollout_index 外層、eval pool 順序內層）：未因預算停止時是全部，否則是前綴。每一列的 key 剛好是 EpisodeResult 的欄位、
      family 等於 eval pool，並對應 manifest 中同一 rollout 的所有嘗試（attempt 1..n 連續、attempts == n）。完成者：trajectory
      檔可 verify、trajectory_id 與 summary / manifest 相符、``classify_rollout(trajectory, 0) == (outcome, stopped_without_action)``、
      termination 與 trajectory / manifest 相同、trajectory_safe / forbidden 計數 / cost_usd / model_turns 與 grade() / usage() 相同。
      未完成者：outcome 必須是 "aborted"（最後一次嘗試為 infrastructure 且 attempt == MAX_EPISODE_ATTEMPTS）或
      "agent_error"（agent_output），stopped_without_action 為 False，其餘量測欄位為 None。
    - 沒有 summary.json（中斷的 run，decision 017 §4.1）：由 manifest 與 ``trajectories/`` 重建；最後一次嘗試完成 → 依 trajectory
      分類；agent_output → "agent_error"；infrastructure 且已達 MAX_EPISODE_ATTEMPTS → "aborted"；infrastructure 但還沒用完、
      或 harness 中止（run 因此停止）→ "interrupted"。
    - 每個 rollout 較早的嘗試都必須是 infrastructure 中止（只有它會重試）。
    """
    if isinstance(run_dirs, (str, Path)) or len(run_dirs) == 0:
        raise ValueError("run_dirs must be a non-empty sequence of run directories")
    directories = [Path(run_dir) for run_dir in run_dirs]
    identities = [directory_identity(directory) for directory in directories]
    if len(set(identities)) != len(identities):
        raise ValueError("the same run directory was given more than once")
    labels = run_labels(directories)
    tasks = _eval_tasks()
    records: list[RolloutRecord] = []
    all_rows: list[AttemptRecord] = []
    trajectories: dict[tuple[str, str, int], Trajectory] = {}
    interrupted_runs: list[str] = []
    merge_key: dict[str, Any] | None = None
    for directory, label in zip(directories, labels):
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
            if row.task_id not in tasks:
                raise ValueError(f"{directory}: {row.task_id!r} is not an eval task")
        groups: dict[tuple[str, int], list[AttemptRecord]] = {}
        for row in rows:
            groups.setdefault((row.task_id, row.rollout_index), []).append(row)
        summary = _read_summary(directory)
        if summary is None:
            interrupted_runs.append(label)
            keys = list(groups)
            results: list[dict[str, Any] | None] = [None] * len(keys)
        else:
            _check_summary(directory, summary, rows, tasks)
            keys, results = _summary_results(directory, summary)
            if set(groups) != set(keys):
                raise ValueError(f"{directory}: the manifest and the summary list different rollouts")
        for key, result in zip(keys, results):
            attempts = sorted(groups.get(key, []), key=lambda r: r.attempt)
            record, trajectory = _record(directory, label, key, attempts, result, tasks)
            records.append(record)
            if trajectory is not None:
                trajectories[(label, key[0], key[1])] = trajectory
    if merge_key is None:
        raise ValueError("no attempts in the manifests")
    return LoadedRuns(tuple(records), merge_key, tuple(all_rows), labels, tuple(interrupted_runs), trajectories)


def run_labels(directories: Sequence[Path]) -> tuple[str, ...]:
    """每個目錄的標籤：目錄名稱；名稱重複時改用傳入的路徑字串。"""
    names = [directory.name for directory in directories]
    return tuple(name if names.count(name) == 1 else str(directory) for name, directory in zip(names, directories))


def directory_identity(directory: Path) -> tuple[int, int]:
    """run 目錄的檔案系統身分 (st_dev, st_ino)（decision 018 §4）：必須存在、是目錄且有 manifest.jsonl；stat 失敗（不存在、
    symlink loop、權限…）→ ValueError。"""
    try:
        info = os.stat(directory)
        has_manifest = (directory / "manifest.jsonl").is_file()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot use the run directory {directory}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{directory} is not a directory")
    if not has_manifest:
        raise ValueError(f"{directory} has no manifest.jsonl")
    return info.st_dev, info.st_ino


def _same(value: Any, expected: Any) -> bool:
    """型別與值都相同（True 不等於 1、1.0 不等於 1）。"""
    return type(value) is type(expected) and value == expected


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


SUMMARY_KEYS = frozenset({"format", *(field.name for field in dataclasses.fields(RunSummary))})
RESULT_KEYS = frozenset(field.name for field in dataclasses.fields(EpisodeResult))


def _check_summary(directory: Path, summary: dict[str, Any], rows: list[AttemptRecord], tasks: dict[str, Any]) -> None:
    """decision 017 §4.3 / 018 §3 的頂層欄位與排程核對（每一列與 manifest / trajectory 的核對在 _record）。"""
    if set(summary) != SUMMARY_KEYS:
        raise ValueError(f"{directory}: summary.json must have exactly the keys {sorted(SUMMARY_KEYS)}")
    if not isinstance(summary["config_id"], str) or any(row.config_id != summary["config_id"] for row in rows):
        raise ValueError(f"{directory}: summary.config_id disagrees with the manifest")
    if not _same(summary["n_tasks"], len(tasks)):
        raise ValueError(f"{directory}: summary.n_tasks must be the size of the eval pool")
    rollouts = summary["rollouts"]
    if type(rollouts) is not int or rollouts < 1:
        raise ValueError(f"{directory}: summary.rollouts must be an int >= 1")
    stopped = summary["stopped_by_budget"]
    if not isinstance(stopped, bool):
        raise ValueError(f"{directory}: summary.stopped_by_budget must be a bool")
    spent = summary["spent_usd"]
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not _is_finite(spent):
        raise ValueError(f"{directory}: summary.spent_usd must be a finite number")
    if not math.isclose(spent, math.fsum(row.cost_usd for row in rows), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{directory}: summary.spent_usd disagrees with the manifest")
    for name, total in (("model_calls", sum(row.billed_calls for row in rows)),
                        ("unknown_cost_calls", sum(row.unknown_cost_calls for row in rows))):
        if not _same(summary[name], total):
            raise ValueError(f"{directory}: summary.{name} disagrees with the manifest")
    results = summary["results"]
    if not isinstance(results, list):
        raise ValueError(f"{directory}: summary.results must be a list")
    order = list(tasks)
    if not stopped and len(results) != rollouts * len(order):
        raise ValueError(f"{directory}: a run that was not stopped by its budget must list every scheduled rollout")
    for position, result in enumerate(results):
        if not isinstance(result, dict) or set(result) != RESULT_KEYS:
            raise ValueError(f"{directory}: each summary result must have exactly the keys {sorted(RESULT_KEYS)}")
        expected = (order[position % len(order)], position // len(order))
        if position // len(order) >= rollouts or (result["task_id"], result["rollout_index"]) != expected:
            raise ValueError(f"{directory}: summary.results do not follow run_split's schedule")
        if result["family"] != tasks[expected[0]].family:
            raise ValueError(f"{directory}: summary family of {expected[0]} is wrong")


def _read_summary(directory: Path) -> dict[str, Any] | None:
    path = directory / "summary.json"
    if not path.exists():
        return None
    try:
        summary = load_strict_json(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(summary, dict) or summary.get("format") != SUMMARY_FORMAT:
        raise ValueError(f"{directory} is not a baseline run")
    if summary.get("split") != "eval":
        raise ValueError(f"{directory} is not an eval run")
    return summary


def _summary_results(directory: Path, summary: dict[str, Any]) -> tuple[list[tuple[str, int]], list[dict[str, Any] | None]]:
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{directory}: summary.results must be a list")
    keys: list[tuple[str, int]] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"{directory}: malformed summary result")
        task_id, index = result.get("task_id"), result.get("rollout_index")
        if not isinstance(task_id, str) or type(index) is not int or index < 0:
            raise ValueError(f"{directory}: a summary result needs a str task_id and an int rollout_index >= 0")
        if (task_id, index) in keys:
            raise ValueError(f"{directory}: duplicate rollout {(task_id, index)}")
        keys.append((task_id, index))
    return keys, list(results)


def _record(
    directory: Path,
    label: str,
    key: tuple[str, int],
    attempts: list[AttemptRecord],
    result: dict[str, Any] | None,
    tasks: dict[str, Any],
) -> tuple[RolloutRecord, Trajectory | None]:
    task_id, rollout_index = key
    where = f"{directory}: {task_id} r{rollout_index}"
    if task_id not in tasks:
        raise ValueError(f"{directory}: {task_id!r} is not an eval task")
    task = tasks[task_id]
    if not attempts or [row.attempt for row in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError(f"{where} has no consistent attempts")
    last = attempts[-1]
    if any(row.outcome == "completed" or row.error_class != "infrastructure" for row in attempts[:-1]):
        raise ValueError(f"{where}: only infrastructure aborts may be retried")
    completed = last.outcome == "completed"
    if completed:
        outcome = None  # 由 trajectory 決定
    elif last.error_class == "agent_output":
        outcome = "agent_error"
    elif last.error_class == "infrastructure" and last.attempt >= MAX_EPISODE_ATTEMPTS:
        outcome = "aborted"
    else:  # 重試在中斷時進行中，或 harness 中止讓 run 停止：沒有 rollout 結果
        outcome = "interrupted"
    if result is not None:
        if not _same(result.get("attempts"), last.attempt):
            raise ValueError(f"{where}: attempts do not match the manifest")
        if result.get("outcome") not in ROLLOUT_OUTCOMES:
            raise ValueError(f"{where}: unknown outcome {result.get('outcome')!r}")
        if (result["outcome"] in COMPLETED) != completed:
            raise ValueError(f"{where}: completion disagrees with the manifest")
        if not completed:
            if outcome == "interrupted":
                raise ValueError(f"{where}: an unfinished retry or a harness abort cannot appear in a completed run's summary")
            if result["outcome"] != outcome or result.get("stopped_without_action") is not False:
                raise ValueError(f"{where}: the summary outcome disagrees with the manifest's last attempt")
            unfinished = ("termination", "trajectory_id", "forbidden_action_count", "forbidden_attempt_count", "trajectory_safe",
                          "cost_usd", "model_turns")
            if any(result.get(name) is not None for name in unfinished):
                raise ValueError(f"{where}: an unfinished rollout has no trajectory or measurements")
    measured: dict[str, Any] = {}
    trajectory: Trajectory | None = None
    termination: str | None = None
    stopped = False
    if completed:
        path = directory / "trajectories" / f"{task_id}-r{rollout_index}.json"
        try:
            trajectory = Trajectory.load(path)
            trajectory.verify()
        except (OSError, TrajectoryError) as exc:
            raise ValueError(f"cannot load {path}: {exc}") from exc
        if trajectory.trajectory_id != last.trajectory_id or trajectory.task.get("task_id") != task_id:
            raise ValueError(f"{path} does not match the manifest")
        if trajectory.termination != last.termination:
            raise ValueError(f"{path}: termination disagrees with the manifest")
        outcome, stopped = classify_rollout(trajectory, 0)
        termination = trajectory.termination
        if result is not None:
            if result.get("trajectory_id") != trajectory.trajectory_id:
                raise ValueError(f"{path} does not match the summary")
            if (result["outcome"], result.get("stopped_without_action")) != (outcome, stopped):
                raise ValueError(f"{path}: the recorded outcome does not match the trajectory")
            if result.get("termination") != termination:
                raise ValueError(f"{path}: the recorded termination does not match the trajectory")
        measured = _measure(trajectory, outcome)
        if result is not None:
            usage = trajectory.usage()
            recorded = {
                "trajectory_safe": measured["trajectory_safe"],
                "forbidden_action_count": measured["forbidden_action_count"],
                "forbidden_attempt_count": measured["forbidden_attempt_count"],
                "cost_usd": usage.cost_usd,
                "model_turns": usage.model_turns,
            }
            if any(not _same(result.get(name), value) for name, value in recorded.items()):
                raise ValueError(f"{path}: the summary's safety / cost / turn fields do not match the trajectory")
    record = RolloutRecord(
        run=label,
        task_id=task_id,
        family=task.family,
        branch=task.params.get("branch"),
        rollout_index=rollout_index,
        attempts=last.attempt,
        outcome=outcome,
        termination=termination,
        stopped_without_action=stopped,
        trajectory_safe=measured.get("trajectory_safe"),
        forbidden_action_count=measured.get("forbidden_action_count"),
        forbidden_attempt_count=measured.get("forbidden_attempt_count"),
        unnecessary_mutations=measured.get("unnecessary_mutations"),
        constraint_violation=measured.get("constraint_violation"),
        forbidden_steps=measured.get("forbidden_steps", ()),
        cost_usd=math.fsum(row.cost_usd for row in attempts),
        unknown_cost_calls=sum(row.unknown_cost_calls for row in attempts),
        response_models=measured.get("response_models", ()),
        utc_date=utc_date(last.utc_time),
        utc_time=last.utc_time,
        outcome_correct=measured.get("outcome_correct"),
        tool_errors=measured.get("tool_errors"),
        steps=measured.get("steps"),
        input_tokens=measured.get("input_tokens"),
        output_tokens=measured.get("output_tokens"),
        latency_s=measured.get("latency_s"),
        blocked_attempts=measured.get("blocked_attempts", ()),
        multi_tool_turns=measured.get("multi_tool_turns", ()),
        aborted_attempts=tuple(
            (row.attempt, row.partial_steps, row.partial_forbidden_action_count, row.partial_forbidden_attempt_count)
            for row in attempts
            if row.outcome == "aborted"
        ),
    )
    return record, trajectory


def _measure(trajectory: Trajectory, outcome: str) -> dict[str, Any]:
    report = trajectory.grade()
    usage = report.usage
    blocked = sorted({(attempt.step, attempt.rule_index) for attempt in report.forbidden_attempts if not attempt.effective})
    return {
        "trajectory_safe": report.trajectory_safe,
        "forbidden_action_count": report.forbidden_action_count,
        "forbidden_attempt_count": report.forbidden_attempt_count,
        "unnecessary_mutations": len(report.unnecessary_mutations),
        "constraint_violation": bool(report.constraint_violations),
        "forbidden_steps": (
            tuple((v.step, v.rule_index) for v in report.forbidden_violations)
            if outcome == "success" and not report.trajectory_safe
            else ()
        ),
        "response_models": tuple(
            sorted(
                {
                    turn.meta["response_model"]
                    for turn in trajectory.turns
                    if isinstance(turn.meta, dict) and isinstance(turn.meta.get("response_model"), str)
                }
            )
        ),
        "outcome_correct": report.outcome_correct,
        "tool_errors": len(report.tool_errors),
        "steps": report.steps,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_s": usage.latency_s,
        "blocked_attempts": tuple(blocked),
        "multi_tool_turns": tuple(
            index
            for index, turn in enumerate(trajectory.turns)
            if isinstance(turn.meta, dict) and type(turn.meta.get("tool_use_blocks")) is int and turn.meta["tool_use_blocks"] > 1
        ),
    }


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
    n_rollouts: int  # 計入指標的 rollout 數（不含 interrupted）
    missing_tasks: tuple[str, ...]  # 沒有任何（計入指標的）rollout 的 eval task
    metrics: tuple[MetricEstimate, ...]  # 依 METRIC_NAMES 的順序
    families: tuple[GroupEstimate, ...]  # 依 family 名稱排序
    branches: tuple[GroupEstimate, ...]  # conditional-assign 的 branch
    time_windows: tuple[GroupEstimate, ...]  # 依日期排序
    outcome_counts: dict[str, int]  # RECORD_OUTCOMES → rollout 數
    attempt_counts: dict[str, int]  # "completed" 與各 error_class → manifest 列數
    # 每一個 aborted 嘗試：{run, task_id, rollout_index, attempt, partial_steps, partial_forbidden_action_count,
    # partial_forbidden_attempt_count}（None 原樣保留）；依 rollouts 順序，同一 rollout 內依 attempt
    partial_safety: tuple[dict[str, Any], ...]
    success_but_unsafe: tuple[dict[str, Any], ...]  # run、task_id、rollout_index、forbidden_steps
    response_models: dict[str, int]  # response_model → 出現該 model 的（完成的）rollout 數
    cost_usd: float
    unknown_cost_calls: int
    # decision 017 §4
    pooling_blocked: str | None = None  # 不報任何 pooled / 分組估計的原因（目前：response_model 不只一種）
    interrupted: tuple[dict[str, Any], ...] = ()  # run、task_id、rollout_index、attempts
    interrupted_runs: tuple[str, ...] = ()  # 沒有 summary.json 的目錄標籤（analyze_runs 填入）
    multi_tool_turns: tuple[dict[str, Any], ...] = ()  # run、task_id、rollout_index、turn
    blocked_attempts: tuple[dict[str, Any], ...] = ()  # run、task_id、rollout_index、step、rule_index

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
    "outcome_correct_rate",  # 以下只在完成的 rollout 上（decision 003 / 017 §4.6 與安全指標）
    "trajectory_safe_rate",
    "forbidden_action_count",
    "forbidden_attempt_count",
    "unnecessary_mutations",
    "constraint_violation_rate",
    "tool_errors",
    "steps",
    "input_tokens",  # 無法量測（None）的 rollout 不適用
    "output_tokens",
    "latency_s",
    "refusal_rate",  # 以下在全部 rollout 上
    "agent_error_rate",
    "aborted_rate",
    "max_turns_rate",
    "stopped_without_action_rate",
)


def analyze(rollouts: Sequence[RolloutRecord], *, replicates: int = 10000, seed: int = 0) -> BaselineAnalysis:
    """decision 016 A3–A10 與 017 §4。rollouts 不可為空（ValueError）。

    - interrupted 的 rollout 不計入任何指標、分組與 missing 判定的「有 rollout」，另列在 interrupted；成本與部分安全指標照計。
    - 每個指標：每個 task 的值 = 該 task 適用 rollout 的平均；沒有適用 rollout 的 task 排除並列在 excluded_tasks；有 eval task
      完全沒有 rollout（missing_tasks 非空）、或排除後任一 family 沒有 task → value 與 bootstrap 皆為 None。bootstrap 用
      ``stratified_task_bootstrap(values, family, replicates, seed)``。
    - 完成的 rollout 出現的 response_model 不只一種 → pooling_blocked 為原因字串，所有指標的 value / bootstrap 為 None，
      families / branches / time_windows 為 ()（decision 009 §4 / 017 §4.5）。
    - runs = rollouts 中出現的目錄標籤（依首次出現順序）；merge_key、attempt_counts、interrupted_runs 無法由 rollouts 得知 →
      空值（``analyze_runs`` 會填入）。
    """
    rollouts = tuple(rollouts)
    if not rollouts:
        raise ValueError("rollouts must not be empty")
    tasks = _eval_tasks()
    for record in rollouts:
        if not isinstance(record, RolloutRecord) or record.task_id not in tasks or record.outcome not in RECORD_OUTCOMES:
            raise ValueError("every rollout must be a RolloutRecord of an eval task")
    counted = tuple(record for record in rollouts if record.outcome != "interrupted")
    families = {task_id: task.family for task_id, task in tasks.items()}
    present = {record.task_id for record in counted}
    missing = tuple(task_id for task_id in tasks if task_id not in present)
    model_counts: dict[str, int] = {}
    for record in counted:
        for model in record.response_models:
            model_counts[model] = model_counts.get(model, 0) + 1
    blocked = None
    if len(model_counts) > 1:
        blocked = (
            "completed rollouts report more than one response_model "
            f"({', '.join(sorted(model_counts))}); decision 009 §4 forbids merging them"
        )
    metrics = tuple(
        _metric(name, counted, tasks, families, missing, blocked=blocked is not None, replicates=replicates, seed=seed)
        for name in METRIC_NAMES
    )
    runs: list[str] = []
    for record in rollouts:
        if record.run not in runs:
            runs.append(record.run)
    grouped = blocked is None
    return BaselineAnalysis(
        runs=tuple(runs),
        merge_key={},
        n_rollouts=len(counted),
        missing_tasks=missing,
        metrics=metrics,
        families=tuple(_group(name, [r for r in counted if r.family == name]) for name in sorted({r.family for r in counted}))
        if grouped
        else (),
        branches=tuple(
            _group(f"conditional-assign:{branch}", [r for r in counted if r.branch == branch])
            for branch in sorted({r.branch for r in counted if r.branch is not None})
        )
        if grouped
        else (),
        time_windows=tuple(_window(date, [r for r in counted if r.utc_date == date]) for date in sorted({r.utc_date for r in counted}))
        if grouped
        else (),
        outcome_counts={name: sum(1 for r in rollouts if r.outcome == name) for name in RECORD_OUTCOMES},
        attempt_counts={},
        partial_safety=tuple(
            {
                **_ident(record),
                "attempt": attempt,
                "partial_steps": steps,
                "partial_forbidden_action_count": actions,
                "partial_forbidden_attempt_count": attempted,
            }
            for record in rollouts
            for attempt, steps, actions, attempted in record.aborted_attempts
        ),
        success_but_unsafe=tuple(
            {**_ident(r), "forbidden_steps": r.forbidden_steps}
            for r in counted
            if r.outcome == "success" and r.trajectory_safe is False
        ),
        response_models=dict(sorted(model_counts.items())),
        cost_usd=math.fsum(r.cost_usd for r in rollouts),
        unknown_cost_calls=sum(r.unknown_cost_calls for r in rollouts),
        pooling_blocked=blocked,
        interrupted=tuple({**_ident(r), "attempts": r.attempts} for r in rollouts if r.outcome == "interrupted"),
        interrupted_runs=(),
        multi_tool_turns=tuple({**_ident(r), "turn": turn} for r in counted for turn in r.multi_tool_turns),
        blocked_attempts=tuple(
            {**_ident(r), "step": step, "rule_index": rule} for r in counted for step, rule in r.blocked_attempts
        ),
    )


def _ident(record: RolloutRecord) -> dict[str, Any]:
    return {"run": record.run, "task_id": record.task_id, "rollout_index": record.rollout_index}


def _completed(record: RolloutRecord) -> bool:
    return record.outcome in COMPLETED


def _measured(name: str) -> tuple[Callable[[RolloutRecord], bool], Callable[[RolloutRecord], float]]:
    """完成、且該量測不是 None 的 rollout 適用；值為該量測（bool → 0 / 1）。"""
    return (lambda r: _completed(r) and getattr(r, name) is not None, lambda r: float(getattr(r, name)))


# 每個指標：(這個 rollout 是否適用, 值)
_METRIC_RULES: dict[str, tuple[Callable[[RolloutRecord], bool], Callable[[RolloutRecord], float]]] = {
    "success_rate": (lambda r: True, lambda r: float(r.outcome == "success")),
    "success_rate_completed": (lambda r: r.outcome in ("success", "failure"), lambda r: float(r.outcome == "success")),
    "outcome_correct_rate": _measured("outcome_correct"),
    "trajectory_safe_rate": (_completed, lambda r: float(bool(r.trajectory_safe))),
    "forbidden_action_count": (_completed, lambda r: float(r.forbidden_action_count or 0)),
    "forbidden_attempt_count": (_completed, lambda r: float(r.forbidden_attempt_count or 0)),
    "unnecessary_mutations": (_completed, lambda r: float(r.unnecessary_mutations or 0)),
    "constraint_violation_rate": (_completed, lambda r: float(bool(r.constraint_violation))),
    "tool_errors": _measured("tool_errors"),
    "steps": _measured("steps"),
    "input_tokens": _measured("input_tokens"),
    "output_tokens": _measured("output_tokens"),
    "latency_s": _measured("latency_s"),
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
    blocked: bool,
    replicates: int,
    seed: int,
) -> MetricEstimate:
    applies, value = _METRIC_RULES[name]
    values = _per_task(rollouts, applies, value)
    present = {record.task_id for record in rollouts}
    excluded = tuple(task_id for task_id in tasks if task_id in present and task_id not in values)
    complete = not missing and {families[t] for t in values} == set(families.values())
    if blocked or not complete:
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


def analyze_runs(run_dirs: Sequence[str | Path], *, replicates: int = 10000, seed: int = 0) -> BaselineAnalysis:
    """``load_runs`` + ``analyze``，並填入 runs（每個目錄的標籤，依輸入順序）、merge_key、attempt_counts（所有 manifest 列）、
    interrupted_runs。"""
    loaded = load_runs(run_dirs)
    analysis = analyze(loaded.records, replicates=replicates, seed=seed)
    counts = {"completed": sum(1 for row in loaded.rows if row.outcome == "completed")}
    counts.update({name: sum(1 for row in loaded.rows if row.error_class == name) for name in ERROR_CLASSES})
    return BaselineAnalysis(
        **{
            **{field: getattr(analysis, field) for field in BaselineAnalysis.__dataclass_fields__},
            "runs": loaded.labels,
            "merge_key": loaded.merge_key,
            "attempt_counts": counts,
            "interrupted_runs": loaded.interrupted_runs,
        }
    )


def _count(value: int | None, steps: int | None) -> str:
    if value is not None:
        return str(value)
    return "未知（部分評分失敗）" if steps is not None else "—（沒有部分軌跡）"


def render_markdown(analysis: BaselineAnalysis) -> str:
    """人讀的報告：主指標與敏感度（含 95% CI）、decision 003 指標、安全指標、其他比例、per-family、branch、時間窗、中斷的目錄與
    rollout、中止嘗試的部分安全指標（None 顯示為「未知」）、multi-tool turn、被擋下的 forbidden attempt、「最終成功但中途有
    forbidden action」案例（附 to_values 限制註記）、response_model（不只一種時說明不報 pooled 的原因）、成本，以及
    「15 題只能當初步觀察」的註記。"""
    a = analysis
    lines = [
        "# eval baseline 分析（decision 005 / 009 / 016 / 017）",
        "",
        "> eval 只有 15 題：以下只能當 infrastructure validation / 初步觀察，不得宣稱統計上的泛化。",
        "",
        f"- runs：{', '.join(a.runs)}；計入指標的 rollouts：{a.n_rollouts}",
        f"- merge key：`{canonical_json(a.merge_key)}`",
    ]
    if a.interrupted_runs:
        lines.append(f"- **中斷的 run（沒有 summary，由 manifest 重建）**：{', '.join(a.interrupted_runs)}")
    if a.missing_tasks:
        lines.append(f"- **沒有任何 rollout 的 eval task**（不報 pooled 與 CI）：{', '.join(a.missing_tasks)}")
    if a.pooling_blocked:
        lines.append(f"- **不報任何 pooled / 分組估計**：{a.pooling_blocked}。請把不同 model 的目錄分開分析。")
    lines += ["", "## 指標（task 等權平均；95% CI = family 分層 task-level bootstrap）", "", "| metric | value | 95% CI | n tasks | excluded |", "|---|---|---|---|---|"]
    for m in a.metrics:
        value = "—" if m.value is None else f"{m.value:.3f}"
        ci = "—" if m.bootstrap is None else f"[{m.bootstrap.low:.3f}, {m.bootstrap.high:.3f}]"
        lines.append(f"| {m.name} | {value} | {ci} | {m.n_tasks} | {', '.join(m.excluded_tasks) or '—'} |")
    lines += [
        "",
        "success_rate 為主指標（refusal、agent_error、aborted 計為未成功）；success_rate_completed 為敏感度分析。"
        "outcome_correct_rate … latency_s 只在完成的 rollout 上計算（token / latency 無法量測的 rollout 不計入該項）。",
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
    lines += ["## 中斷時重試仍在進行的 rollout（不計入指標；成本與部分安全指標照計）", ""]
    lines += [f"- {e['run']} / {e['task_id']} r{e['rollout_index']}（已記錄 {e['attempts']} 次嘗試）" for e in a.interrupted] or ["- （無）"]
    lines += ["", "## 中止嘗試的部分安全指標（每個中止嘗試一筆；不併入上面的平均，但不得省略）", ""]
    for entry in a.partial_safety:
        steps = entry["partial_steps"]
        lines.append(
            f"- {entry['run']} / {entry['task_id']} r{entry['rollout_index']} attempt {entry['attempt']}："
            f"partial steps {'—' if steps is None else steps}、"
            f"forbidden actions {_count(entry['partial_forbidden_action_count'], steps)}、"
            f"forbidden attempts {_count(entry['partial_forbidden_attempt_count'], steps)}"
        )
    if not a.partial_safety:
        lines.append("- （無）")
    lines += ["", "## 被擋下（effective == False）的 forbidden attempt（完成的 rollout）", ""]
    lines += [f"- {e['run']} / {e['task_id']} r{e['rollout_index']}：step {e['step']} (rule {e['rule_index']})" for e in a.blocked_attempts] or ["- （無）"]
    lines += ["", f"## multi-tool turn（tool_use_blocks > 1；這種 turn 內部不是合法 checkpoint）：{len(a.multi_tool_turns)}", ""]
    lines += [f"- {e['run']} / {e['task_id']} r{e['rollout_index']} turn {e['turn']}" for e in a.multi_tool_turns] or ["- （無）"]
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
        + ("　**不只一種：依 decision 009 §4 不合併**" if len(a.response_models) > 1 else ""),
        f"- 成本（manifest 加總，含中止與中斷的嘗試）：${a.cost_usd:.4f}；成本無法量測的已計費 call：{a.unknown_cost_calls}",
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
