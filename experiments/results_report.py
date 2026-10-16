"""正式結果頁與 README 的結果段落（decision 033）。不連網、不呼叫任何 LLM；輸出完全決定性。

用法::

    uv run python -m experiments.results_report --baseline-run runs/<eval> [--baseline-run ...] \\
        --sweep-run runs/<sweep-1> [--sweep-run ...] --out docs/results --readme README.md

- 分析一律經過預先登記的分析工具：``analyze_baseline.analyze_runs`` 與 ``analyze_sweep.analyze_sweep_runs``（載入時完整核對
  manifest、summary、trajectory 與續跑的血緣；不一致 → ValueError）。兩者的完整報告（``render_markdown``）原樣寫成
  ``baseline-analysis.md`` / ``sweep-analysis.md``：預先登記要報告的每一項（安全指標、排除的題目、成功但不安全的案例、中止嘗試的
  部分安全指標、中斷與續跑、資料是否完整）都以登記的工具自己的呈現為準。
- ``summary.json``：摘要（JSON-native、只含數字與短文字，不含 trajectory）；``README.md``：結果頁；``recoverability.svg``：
  15 個 source 的 recoverability curve（small multiples）。README 的 ``<!-- results:start -->`` / ``<!-- results:end -->`` 之間換成
  摘要段落：只接受正式、資料完整、Gate 有評估的結果。
- 每個 run 目錄的 ``calls.jsonl``（decision 032）：與 manifest 逐次嘗試核對（``call_log.reconcile``）。
- 失敗案例：sweep 中每個沒有成功的 rerun（依 manifest 找到保存它的目錄、核對 trajectory_id 後重新評分），baseline 中每個沒有成功的
  rollout（含中斷的），以及 baseline 中的 tool 錯誤（例子，標明模型是否在下一步修正）。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

from env.hashing import canonical_json
from experiments import analyze_baseline, analyze_sweep
from experiments.call_log import reconcile
from replay.aborts import read_manifest
from replay.trajectory import Trajectory, load_strict_json

SUMMARY_FORMAT = "agent-gym/results-summary/v1"
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
MAX_RERUN_CASES = 30
MAX_TOOL_ERROR_EXAMPLES = 3
EXCERPT = 160
MIN_SVG_WIDTH = 720
CI_LABEL = "95% Wilson CI（逐點、未做多重比較校正；cliff 判定使用 bonferroni95 的較寬區間）"
GATE_TEXT = {
    "advanced": "同一個 (tool, 類別) 模式出現在至少 2 個 family 的 cliff 上：Gate 通過，可以進 Advanced。",
    "gradual": "過半數完成的 sweep 沒有 cliff，cliff 之間也沒有重複的模式：失敗是逐步累積的，Gate 不通過，不進 Advanced。",
    "no_repeated_pattern": "有 cliff，但沒有在 2 個以上 family 重複的模式：Gate 不通過，不進 Advanced。",
    "no_data": "沒有完成的 sweep：無法判定，不進 Advanced。",
}
NO_FAILURE_TEXT = ("這組題目上沒有任何失敗：eval baseline 的每個 rollout、sweep 中從每個 checkpoint 重新執行的每個 rerun 都成功，"
                   "所以沒有可以定位的失敗——P(success | k) 在每個 checkpoint 都是 1。這是**天花板效應**（題目對這個模型太容易），"
                   "不是 replay 方法本身的結論；依 decision 012 / 025，不因結果調整題目、參數或 Gate。")


# ---------------------------------------------------------------- 讀取與查詢


def _excerpt(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= EXCERPT else text[:EXCERPT - 1] + "…"


def _failed_checks(trajectory: Trajectory) -> list[str]:
    report = trajectory.grade()
    failed = [f"{check.kind} #{check.index}: {check.detail}" for check in (*report.outcome_checks, *report.constraint_checks)
              if not check.passed]
    if report.forbidden_action_count:
        failed.append(f"forbidden actions: {report.forbidden_action_count}")
    return failed


def _last_message(trajectory: Trajectory) -> str:
    for turn in reversed(trajectory.turns):
        if turn.message.text:
            return _excerpt(turn.message.text)
    return ""


def _trajectories(run_dirs: Sequence[Path]) -> list[Trajectory]:
    return [Trajectory.load(path) for run_dir in run_dirs for path in sorted((run_dir / "trajectories").glob("*.json"))]


def _model_identity(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    for trajectory in trajectories:
        model_config = trajectory.agent.model_config
        return {key: model_config.get(key) for key in ("model", "model_digest", "ollama_version", "options", "think")}
    return {"model": None, "model_digest": None, "ollama_version": None, "options": None, "think": None}


def _rollouts_per_task(baseline_dirs: Sequence[Path]) -> list[int]:
    values = []
    for run_dir in baseline_dirs:
        run_json = load_strict_json((run_dir / "run.json").read_bytes())
        values.append(int(run_json["rollouts"]))
    return sorted(set(values))


def _tool_error_examples(trajectories: Sequence[Trajectory]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """tool 錯誤：(tool, code) 的次數，與每一種的第一個例子（錯誤的 call、錯誤訊息、模型的下一個 call、是否在下一步修正）。"""
    counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for trajectory in trajectories:
        steps = [step for turn in trajectory.turns for step in turn.tool_steps]
        for index, step in enumerate(steps):
            if step.ok or step.error is None:
                continue
            key = f"{step.tool}/{step.error['code']}"
            counts[key] = counts.get(key, 0) + 1
            if len(examples) < MAX_TOOL_ERROR_EXAMPLES and not any(e["kind"] == key for e in examples):
                following = steps[index + 1] if index + 1 < len(steps) else None
                examples.append({
                    "kind": key,
                    "task_id": trajectory.task["task_id"],
                    "call": {"tool": step.tool, "args": step.args},
                    "error": _excerpt(step.error.get("message")),
                    "next_call": None if following is None else {"tool": following.tool, "args": following.args, "ok": following.ok},
                    "fixed_next": following is not None and following.tool == step.tool and following.ok,
                })
    return dict(sorted(counts.items())), examples


class _SweepFiles:
    """sweep 目錄中保存的 rerun：依 manifest 中 completed 的列（task_id, k, rollout_index）找到保存它的目錄，並核對 trajectory_id。"""

    def __init__(self, sweep_dirs: Sequence[Path]) -> None:
        self._rows: dict[tuple[str, int, int], list[tuple[Path, str]]] = {}
        for run_dir in sweep_dirs:
            for row in read_manifest(run_dir / "manifest.jsonl"):
                if row.outcome == "completed" and row.checkpoint_turn is not None and row.trajectory_id:
                    key = (row.task_id, row.checkpoint_turn, row.rollout_index)
                    self._rows.setdefault(key, []).append((run_dir, row.trajectory_id))

    def load(self, task_id: str, k: int, rollout_index: int) -> Trajectory | None:
        for run_dir, trajectory_id in self._rows.get((task_id, k, rollout_index), []):
            path = run_dir / "rollouts" / task_id / f"k{k}-r{rollout_index}.json"
            if path.exists():
                trajectory = Trajectory.load(path)
                if trajectory.trajectory_id == trajectory_id:
                    return trajectory
        return None


# ---------------------------------------------------------------- 摘要


def _walkthrough(trajectories: Sequence[Trajectory], sources: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """最長的 source（turn 數最多；相同時取 task_id 較大者）的完整步驟（以 source_trajectory_id 找到 sweep 實際用的那條 trajectory）。"""
    if not sources:
        return None
    longest = max(sources, key=lambda source: (source["n_turns"], source["task_id"]))
    trajectory = next((t for t in trajectories if t.trajectory_id == longest["source_trajectory_id"]), None)
    if trajectory is None:
        return None
    report = trajectory.grade()
    return {
        "task_id": longest["task_id"],
        "family": longest["family"],
        "instruction": trajectory.task.get("instruction"),
        "termination": trajectory.termination,
        "turns": [{"turn": turn.index, "text": _excerpt(turn.message.text),
                   "calls": [{"tool": step.tool, "args": step.args, "ok": step.ok,
                              "error": None if step.error is None else step.error.get("code"), "changes": len(step.changes)}
                             for step in turn.tool_steps]}
                  for turn in trajectory.turns],
        "grade": {"success": report.success, "outcome_correct": report.outcome_correct, "trajectory_safe": report.trajectory_safe,
                  "steps": report.steps, "mutating_steps": report.mutating_steps,
                  "forbidden_action_count": report.forbidden_action_count, "tool_errors": len(report.tool_errors),
                  "checks": [f"{check.kind} #{check.index}: {'通過' if check.passed else '沒有通過'}"
                             for check in (*report.outcome_checks, *report.constraint_checks)]},
    }


def _build(baseline_runs: Sequence[str | Path], sweep_runs: Sequence[str | Path], replicates: int) -> tuple[dict[str, Any], str, str]:
    baseline_dirs = [Path(path) for path in baseline_runs]
    sweep_dirs = [Path(path) for path in sweep_runs]
    if analyze_baseline.baseline_suite(baseline_dirs) != "week1":
        raise ValueError("this report is the sealed round-1 report; Study 2 has its own report (decision 034 §1)")
    base_analysis = analyze_baseline.analyze_runs(baseline_dirs, replicates=replicates)
    sweep_analysis = analyze_sweep.analyze_sweep_runs(sweep_dirs, baseline_dirs, replicates=replicates)
    base, sweep = base_analysis.to_dict(), sweep_analysis.to_dict()
    baseline_trajectories = _trajectories(baseline_dirs)
    tool_error_counts, tool_error_examples = _tool_error_examples(baseline_trajectories)
    files = _SweepFiles(sweep_dirs)

    sources = []
    rerun_outcomes: dict[str, int] = {}
    rerun_cases: list[dict[str, Any]] = []
    rerun_failures = 0
    for curve in sorted(sweep["curves"], key=lambda c: c["task_id"]):
        result = curve["result"]
        for rollout in result["rollouts"]:
            rerun_outcomes[rollout["outcome"]] = rerun_outcomes.get(rollout["outcome"], 0) + 1
            if rollout["outcome"] == "success":
                continue
            rerun_failures += 1
            if len(rerun_cases) >= MAX_RERUN_CASES:
                continue
            case: dict[str, Any] = {"task_id": curve["task_id"], "family": curve["family"], "k": rollout["k"],
                                    "rollout_index": rollout["rollout_index"], "outcome": rollout["outcome"],
                                    "termination": rollout.get("termination"), "failed_checks": [], "last_message": "",
                                    "turns": None}
            trajectory = files.load(curve["task_id"], rollout["k"], rollout["rollout_index"])
            if trajectory is None:
                case["failed_checks"] = [f"沒有保存的 trajectory（{rollout['outcome']}）"]
            else:
                case["failed_checks"] = _failed_checks(trajectory) or ["（最終狀態的檢查都通過；依結束方式判定為未成功）"]
                case["last_message"] = _last_message(trajectory)
                case["turns"] = len(trajectory.turns)
            rerun_cases.append(case)
        sources.append({
            "task_id": curve["task_id"],
            "family": curve["family"],
            "n_turns": curve["n_turns"],
            "run": curve["run"],
            "source_trajectory_id": curve["source_trajectory_id"],
            "source_success": curve["source_success"],
            "is_cliff": result["is_cliff"],
            "largest_drop": result["largest_drop"],
            "largest_drop_k": result["largest_drop_k"],
            "max_turns_confounded": result["max_turns_confounded"],
            "cliff_pattern": curve["cliff"],
            "refined_intervals": result["refined_intervals"],
            "points": [{"k": e["k"], "n": e["n"], "successes": e["successes"], "p_hat": e["p_hat"], "ci_low": e["ci_low"],
                        "ci_high": e["ci_high"]} for e in sorted(result["estimates"], key=lambda e: e["k"])],
        })

    records = analyze_baseline.load_runs(baseline_dirs).records
    baseline_failures = [{"run": r.run, "task_id": r.task_id, "rollout_index": r.rollout_index, "outcome": r.outcome,
                          "termination": r.termination, "attempts": r.attempts}
                         for r in sorted(records, key=lambda r: (r.task_id, r.rollout_index, r.run)) if r.outcome != "success"]
    gate = sweep["gate"]
    summary = {
        "format": SUMMARY_FORMAT,
        "provenance": {
            "config_id": base["merge_key"]["config_id"] if base.get("merge_key") else None,
            **_model_identity(baseline_trajectories),
            "baseline_runs": base["runs"],
            "sweep_runs": sweep["runs"],
            "rollouts_per_task": _rollouts_per_task(baseline_dirs),
            "dates": sorted({w["group"] for w in base["time_windows"]} | {w["date"] for w in sweep["time_windows"]}),
            "dry_run": bool(base["dry_run"] or sweep["dry_run"]),
        },
        "calls": {"baseline": reconcile(baseline_dirs), "sweep": reconcile(sweep_dirs)},
        "baseline": {
            "rollouts": base["n_rollouts"],
            "metrics": [{"name": m["name"], "value": m["value"], "ci_low": (m.get("bootstrap") or {}).get("low"),
                         "ci_high": (m.get("bootstrap") or {}).get("high"), "n_tasks": m["n_tasks"],
                         "excluded_tasks": len(m["excluded_tasks"])} for m in base["metrics"]],
            "families": base["families"],
            "outcome_counts": base["outcome_counts"],
            "attempt_counts": base["attempt_counts"],
            "missing_tasks": base["missing_tasks"],
            "interrupted_runs": base["interrupted_runs"],
            "success_but_unsafe": len(base["success_but_unsafe"]),
            "partial_safety": len(base["partial_safety"]),
            "pooling_blocked": base["pooling_blocked"],
            "tool_error_counts": tool_error_counts,
        },
        "sweep": {
            "plan": sweep["plan"],
            "criterion": sweep["criterion"],
            "gate": None if gate is None else {key: gate[key] for key in ("outcome", "met", "rule", "patterns", "families",
                                                                         "data_complete")},
            "pooling_blocked": sweep["pooling_blocked"],
            "cliff_rate": sweep["cliff_rate"],
            "mean_largest_drop": sweep["mean_largest_drop"],
            "reruns": sum(rerun_outcomes.values()),
            "rerun_outcomes": dict(sorted(rerun_outcomes.items())),
            "not_completed": sweep["not_completed"],
            "missing_sources": sweep["missing_sources"],
            "partial_sweeps": sweep["partial_sweeps"],
            "partial_safety": len(sweep["partial_safety"]),
            "interrupted_runs": sweep["interrupted_runs"],
            "stopped_by_budget": sweep["stopped_by_budget"],
            "resumed": sweep["resumed"],
            "sources": sources,
        },
        "walkthrough": _walkthrough(baseline_trajectories, sources),
        "failure_cases": {
            "rerun_failures": rerun_failures,
            "reruns": rerun_cases,
            "baseline": baseline_failures,
            "tool_error_examples": tool_error_examples,
        },
    }
    return summary, analyze_baseline.render_markdown(base_analysis), analyze_sweep.render_markdown(sweep_analysis)


def build_summary(baseline_runs: Sequence[str | Path], sweep_runs: Sequence[str | Path], *, replicates: int = 10000) -> dict[str, Any]:
    """正式結果的摘要（JSON-native、決定性）。"""
    return _build(baseline_runs, sweep_runs, replicates)[0]


# ---------------------------------------------------------------- 結果頁


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _short(task_id: str) -> str:
    return task_id.removeprefix("gen-")


def _gate_outcome(summary: dict[str, Any]) -> str | None:
    gate = summary["sweep"]["gate"]
    return None if gate is None else gate["outcome"]


def _data_complete(summary: dict[str, Any]) -> bool:
    gate = summary["sweep"]["gate"]
    return gate is not None and bool(gate.get("data_complete"))


def _no_failures(summary: dict[str, Any]) -> bool:
    """資料完整，而且 baseline 與 sweep 都沒有任何沒有成功的 rollout / rerun（含沒有進入曲線的部分 sweep）。"""
    sweep, base = summary["sweep"], summary["baseline"]
    return (_data_complete(summary) and sweep["reruns"] > 0 and set(sweep["rerun_outcomes"]) == {"success"}
            and not sweep["partial_sweeps"] and {k for k, v in base["outcome_counts"].items() if v} == {"success"})


def _gate_sentence(summary: dict[str, Any]) -> str:
    outcome = _gate_outcome(summary)
    if outcome is None:
        return "Gate 不評估：response_model 不只一種（decision 009 §4），不得合併，不進 Advanced。"
    if not _data_complete(summary):
        sweep = summary["sweep"]
        return (f"**資料不完整**（未完成的 source {len(sweep['not_completed'])}、沒有 source {len(sweep['missing_sources'])}）："
                f"以現有資料判定為 `{outcome}`，但不能當作結論；不進 Advanced。")
    if _no_failures(summary):
        return "沒有任何 cliff：所有 rerun 都成功，沒有失敗可以定位；依機械化規則判定為 `gradual`，Gate 不通過，不進 Advanced。"
    return GATE_TEXT.get(outcome, "")


def _calls_line(label: str, stats: dict[str, Any] | None) -> str:
    if stats is None:
        return f"- {label} 的逐次呼叫紀錄：**沒有 calls.jsonl**（decision 032 之前的 run，seed 只記在完成的 trajectory）"
    state = "完整" if stats["complete"] else "**不完整**"
    return (f"- {label} 的逐次呼叫紀錄（{state}）：{stats['calls']} 次呼叫（失敗 {stats['failed']}）、{stats['seeds']} 個 seed；"
            f"manifest 的 {stats['attempts']} 次嘗試中 {stats['matched_attempts']} 次與紀錄相符、"
            f"{stats['mismatched_attempts']} 次不符；另有 {stats['interrupted_attempts']} 次被中斷的嘗試（manifest 沒有它們的列）")


def render_page(summary: dict[str, Any]) -> str:
    """docs/results/README.md（繁體中文）。"""
    prov, calls, base, sweep, cases = (summary["provenance"], summary["calls"], summary["baseline"], summary["sweep"],
                                       summary["failure_cases"])
    rollouts = prov.get("rollouts_per_task") or []
    design = f"15 題 × R = {rollouts[0]}" if len(rollouts) == 1 else "15 題"
    lines = ["# 正式結果（decision 030 凍結設定）", ""]
    if prov["dry_run"]:
        lines += ["> **[dry-run] 這一頁由 dry-run 的輸出產生，不是研究結果。**", ""]
    lines += [
        "> 由 `experiments/results_report.py` 從正式 run 目錄產生（決定性輸出）。分析經過預先登記的 `analyze_baseline` / `analyze_sweep`"
        "（載入時完整核對 manifest、summary、trajectory 與續跑血緣），它們的完整報告原樣附在 [baseline-analysis.md](baseline-analysis.md) "
        "與 [sweep-analysis.md](sweep-analysis.md)：安全指標、排除的題目、成功但不安全的案例、中止嘗試的部分安全指標都以那兩份為準。"
        "eval 只有 15 題：以下只能當初步觀察，不得宣稱統計上的泛化。",
        "",
        "## 設定與紀錄",
        "",
        f"- 模型：`{prov['model']}`，digest `{prov['model_digest']}`，Ollama {prov['ollama_version']}，think {prov['think']}",
        f"- config_id：`{prov['config_id']}`",
        f"- 推論參數：`{canonical_json(prov['options'])}`；每次呼叫的 seed 隨機並記錄（trajectory 的 meta 與 `calls.jsonl`，decision 032）",
        f"- eval baseline：{', '.join(f'`{r}`' for r in prov['baseline_runs'])}；sweep：{', '.join(f'`{r}`' for r in prov['sweep_runs'])}"
        f"；日期（UTC）：{', '.join(prov['dates'])}",
        _calls_line("eval baseline", calls["baseline"]),
        _calls_line("sweep", calls["sweep"]),
        "",
        f"## Eval baseline（{design}，{base['rollouts']} 個計入指標的 rollout）",
        "",
        "| metric | value | 95% CI（family 分層 task-level bootstrap） | tasks | 排除的 tasks |",
        "|---|---|---|---|---|",
    ]
    for metric in base["metrics"]:
        lines.append(f"| {metric['name']} | {_fmt(metric['value'])} | [{_fmt(metric['ci_low'])}, {_fmt(metric['ci_high'])}] | "
                     f"{metric['n_tasks']} | {metric['excluded_tasks']} |")
    lines += ["", "| family | tasks | rollouts | success rate |", "|---|---|---|---|"]
    for group in base["families"]:
        lines.append(f"| {group['group']} | {group['n_tasks']} | {group['n_rollouts']} | {_fmt(group['success_rate'])} |")
    lines += [
        "",
        f"- 結果：`{canonical_json(base['outcome_counts'])}`；嘗試：`{canonical_json(base['attempt_counts'])}`",
        f"- 缺少的題目：{len(base['missing_tasks'])}；中斷的目錄：{len(base['interrupted_runs'])}；成功但不安全的 rollout："
        f"{base['success_but_unsafe']}；中止嘗試的部分安全紀錄：{base['partial_safety']}（明細見 baseline-analysis.md）",
        f"- tool 錯誤（含模型在下一步自行修正者）：`{canonical_json(base['tool_error_counts'])}`",
        "",
        "## Failure localization sweep",
        "",
        f"- plan：`{canonical_json(sweep['plan'])}`，驟降判定 `{sweep['criterion']}`（decision 025 §3）",
        f"- rerun（進入曲線者）：{sweep['reruns']} 次，結果 `{canonical_json(sweep['rerun_outcomes'])}`",
        f"- 未完成的 source：{len(sweep['not_completed'])}；沒有 source：{len(sweep['missing_sources'])}；"
        f"沒有進入曲線的部分 sweep：{len(sweep['partial_sweeps'])}"
        f"（{sum(p['rollouts'] for p in sweep['partial_sweeps'])} 個 rollout）；中止嘗試的部分安全紀錄：{sweep['partial_safety']}",
        f"- 中斷的目錄：{len(sweep['interrupted_runs'])}；因預算停止：{len(sweep['stopped_by_budget'])}；續跑：{len(sweep['resumed'])}",
    ]
    cliff_rate, drop = sweep["cliff_rate"] or {}, sweep["mean_largest_drop"] or {}
    lines += [
        f"- cliff 比例：{_fmt(cliff_rate.get('estimate'))}（95% CI [{_fmt(cliff_rate.get('low'))}, {_fmt(cliff_rate.get('high'))}]）；"
        f"平均最大下降：{_fmt(drop.get('estimate'))}（95% CI [{_fmt(drop.get('low'))}, {_fmt(drop.get('high'))}]）"
        + ("" if sweep["cliff_rate"] else "（資料不完整，沒有 bootstrap）"),
        f"- **Gate（decision 025 §4）：`{_gate_outcome(summary)}`**。{_gate_sentence(summary)}",
        "",
        *([f"> {NO_FAILURE_TEXT}", ""] if _no_failures(summary) else []),
        "![recoverability curves](recoverability.svg)",
        "",
        f"每一格是一個 source（eval baseline 的 rollout 0）。點是從 checkpoint k（source 已完成 k 個 turn）重新執行的成功比例 "
        f"P(success | k)，線段是{CI_LABEL}。",
        "",
        "| source | turns | checkpoints | min P(success \\| k) | 最大下降（k） | cliff | max_turns 混淆 |",
        "|---|---|---|---|---|---|---|",
    ]
    for source in sweep["sources"]:
        low = min((p["p_hat"] for p in source["points"]), default=None)
        lines.append(f"| {source['task_id']} | {source['n_turns']} | {len(source['points'])} | {_fmt(low)} | "
                     f"{_fmt(source['largest_drop'])}（{source['largest_drop_k']}） | {'是' if source['is_cliff'] else '否'} | "
                     f"{'是' if source['max_turns_confounded'] else '否'} |")
    lines += ["", "<details><summary>每個 checkpoint 的數值（表格版的圖）</summary>", "",
              f"| source | k | n | successes | P(success \\| k) | {CI_LABEL} |", "|---|---|---|---|---|---|"]
    for source in sweep["sources"]:
        for point in source["points"]:
            lines.append(f"| {source['task_id']} | {point['k']} | {point['n']} | {point['successes']} | {_fmt(point['p_hat'])} | "
                         f"[{_fmt(point['ci_low'])}, {_fmt(point['ci_high'])}] |")
    lines += ["", "</details>", ""]
    walk = summary.get("walkthrough")
    if walk:
        grade = walk["grade"]
        lines += [
            "## 一個正式 trajectory（Task → actions → state changes → grader）",
            "",
            f"最長的 source：`{walk['task_id']}`（{walk['family']}，{len(walk['turns'])} 個 turn，結束方式 `{walk['termination']}`）。",
            "",
            f"> {' '.join(str(walk['instruction']).split())}",
            "",
            "| turn | 模型的 tool call | 結果 | 狀態變更 |",
            "|---|---|---|---|",
        ]
        for turn in walk["turns"]:
            if not turn["calls"]:
                lines.append(f"| {turn['turn']} | （最後的回答）{turn['text']} | — | — |")
            for call in turn["calls"]:
                result = "ok" if call["ok"] else f"錯誤：`{call['error']}`"
                lines.append(f"| {turn['turn']} | `{call['tool']}({canonical_json(call['args'])})` | {result} | {call['changes']} |")
        lines += [
            "",
            f"grader：success {grade['success']}、outcome_correct {grade['outcome_correct']}、trajectory_safe {grade['trajectory_safe']}；"
            f"{grade['steps']} 個 tool call（{grade['mutating_steps']} 個改變狀態），tool 錯誤 {grade['tool_errors']}，"
            f"forbidden action {grade['forbidden_action_count']}。檢查：{'；'.join(grade['checks'])}。",
            "",
        ]
    lines += ["## 失敗與異常案例", "", f"- baseline 沒有成功的 rollout（含中斷的）：{len(cases['baseline'])}"]
    for row in cases["baseline"]:
        lines.append(f"  - {row['run']} / {row['task_id']} r{row['rollout_index']}：{row['outcome']}（{row['termination']}；"
                     f"{row['attempts']} 次嘗試）")
    lines.append(f"- sweep 沒有成功的 rerun（進入曲線者）：{cases['rerun_failures']}"
                 + (f"（以下列出前 {len(cases['reruns'])} 個）" if cases["rerun_failures"] > len(cases["reruns"]) else ""))
    for case in cases["reruns"]:
        lines.append(f"  - {case['task_id']} 從 k = {case['k']} 重新執行（r{case['rollout_index']}）：{case['outcome']}"
                     f"（{case['termination']}）；沒有通過：{'；'.join(case['failed_checks'])}"
                     + (f"；最後的訊息：「{case['last_message']}」" if case["last_message"] else ""))
    lines.append(f"- 沒有進入曲線的部分 sweep：{len(sweep['partial_sweeps'])}；中止嘗試（baseline / sweep）："
                 f"{base['partial_safety']} / {sweep['partial_safety']}（明細見兩份分析）")
    if cases["tool_error_examples"]:
        lines += ["", "tool 錯誤（每一種的第一個例子；「已在下一步修正」= 下一個 call 是同一個 tool 而且成功）：", ""]
        for example in cases["tool_error_examples"]:
            following = example["next_call"]
            lines.append(f"- {example['task_id']}：`{example['call']['tool']}({canonical_json(example['call']['args'])})` → "
                         f"{example['error']}"
                         + ("" if following is None else
                            f"；下一個 call：`{following['tool']}({canonical_json(following['args'])})`"
                            f"（{'成功' if following['ok'] else '失敗'}）")
                         + ("；已在下一步修正" if example["fixed_next"] else "；沒有在下一步修正"))
    lines += [
        "",
        "## 重現",
        "",
        "```bash",
        "tools/with-gpu-lock -- uv run python -m experiments.run_baseline --split eval --rollouts 3 --confirm-local-run "
        "--max-model-calls 1800 --run-name <eval>",
        "tools/with-gpu-lock -- uv run python -m experiments.run_sweep --baseline-run runs/<eval> --plan 2,10,48,0.3 "
        "--criterion bonferroni95 --confirm-local-run --max-model-calls 3000 --run-name <sweep-1>",
        "uv run python -m experiments.results_report --baseline-run runs/<eval> --sweep-run runs/<sweep-1> --out docs/results "
        "--readme README.md",
        "```",
        "",
        "seed 每次隨機（temperature 0.7 的取樣是研究設計的一部分），所以重跑得到的是同一個分布的新樣本；個別呼叫可以由 trajectory 的 "
        "conversation 加上記錄的 seed 重送（decision 032）。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 圖


def recoverability_svg(summary: dict[str, Any]) -> str:
    """source 的 recoverability curve（small multiples：每個 family 一列）。單一序列：點 = P(success | k)、線段 = 逐點 95% Wilson CI；
    x 軸（checkpoint k）所有格共用。顏色由 CSS 變數決定，深色模式有自己的設定。"""
    sources = summary["sweep"]["sources"]
    families = sorted({s["family"] for s in sources})
    rows = [[s for s in sources if s["family"] == family] for family in families]
    columns = max((len(row) for row in rows), default=1)
    max_k = max((p["k"] for s in sources for p in s["points"]), default=0)
    panel_w, panel_h, gap_x, gap_y = 210, 118, 18, 30
    left, top, plot_left, plot_right, plot_top, plot_bottom = 16, 80, 34, 10, 22, 26
    width = max(left * 2 + columns * panel_w + (columns - 1) * gap_x, MIN_SVG_WIDTH)
    height = top + max(len(rows), 1) * (panel_h + gap_y) + 24
    dry = summary["provenance"]["dry_run"]
    gate = _gate_outcome(summary)
    title = ("[dry-run] " if dry else "") + "Recoverability：從 checkpoint k 重新執行的成功比例 P(success | k)"
    gate_text = "不評估" if gate is None else gate + ("" if _data_complete(summary) else "（資料不完整）")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" aria-labelledby="t d" font-family="-apple-system, BlinkMacSystemFont, \'Noto Sans TC\', \'Segoe UI\', sans-serif">',
        f'<title id="t">{html.escape(title)}</title>',
        f'<desc id="d">{len(sources)} 個 source 的 small multiples；每格的點是 P(success | k)，線段是逐點 95% Wilson CI。'
        f'Gate：{html.escape(gate_text)}。數值見 docs/results/README.md 的表格。</desc>',
        "<style>",
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e4e3df;--series:#2a78d6;--band:#9ec5f4;}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e88;"
        "--grid:#383835;--series:#3987e5;--band:#1c5cab;}}",
        ".bg{fill:var(--surface)}.t1{fill:var(--ink);font-size:15px;font-weight:600}.t2{fill:var(--ink2);font-size:11px}"
        ".t3{fill:var(--muted);font-size:10px}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--muted);stroke-width:1}"
        ".line{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round}.ci{stroke:var(--band);stroke-width:4;"
        "stroke-linecap:round}.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}.cliff{stroke:var(--ink2);"
        "stroke-width:1.5;stroke-dasharray:4 3}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
        f'<text class="t1" x="{left}" y="26">{html.escape(title)}</text>',
        f'<text class="t2" x="{left}" y="44">每格一個 source（eval baseline 的 rollout 0）。x：checkpoint k（source 已完成的 turn 數）；'
        f'y：P(success | k)，0–1。</text>',
        f'<text class="t2" x="{left}" y="60">點：成功比例；線段：逐點 95% Wilson CI（cliff 判定用 '
        f'{html.escape(summary["sweep"]["criterion"])}）；虛線：cliff；Gate：{html.escape(gate_text)}。</text>',
    ]
    span = max(max_k, 1)
    for r, row in enumerate(rows):
        for c, source in enumerate(row):
            x0 = left + c * (panel_w + gap_x)
            y0 = top + r * (panel_h + gap_y)
            px0, px1 = x0 + plot_left, x0 + panel_w - plot_right
            py0, py1 = y0 + plot_top, y0 + panel_h - plot_bottom

            def sx(k: float, px0: float = px0, px1: float = px1) -> float:
                return px0 + (px1 - px0) * k / span

            def sy(p: float, py0: float = py0, py1: float = py1) -> float:
                return py1 - (py1 - py0) * p

            out.append(f'<g class="panel" aria-label="{html.escape(source["task_id"])}">')
            label = _short(source["task_id"]) + ("（cliff）" if source["is_cliff"] else "")
            out.append(f'<text class="t2" x="{x0 + plot_left}" y="{y0 + 12}">{html.escape(label)}</text>')
            for level in (0.0, 0.5, 1.0):
                y = sy(level)
                out.append(f'<line class="grid" x1="{px0:.1f}" y1="{y:.1f}" x2="{px1:.1f}" y2="{y:.1f}"/>')
                out.append(f'<text class="t3" x="{px0 - 6:.1f}" y="{y + 3:.1f}" text-anchor="end">{level:g}</text>')
            for k in range(0, max_k + 1, 2 if max_k > 6 else 1):
                out.append(f'<text class="t3" x="{sx(k):.1f}" y="{py1 + 14:.1f}" text-anchor="middle">{k}</text>')
            last_k = source["n_turns"] - 1
            if last_k >= 0:
                out.append(f'<line class="axis" x1="{sx(min(last_k, span)):.1f}" y1="{py1:.1f}" x2="{sx(min(last_k, span)):.1f}" '
                           f'y2="{py1 + 4:.1f}"/>')
            if source["is_cliff"] and source["largest_drop_k"] is not None:
                xk = sx(source["largest_drop_k"] + 0.5)
                out.append(f'<line class="cliff" x1="{xk:.1f}" y1="{py0:.1f}" x2="{xk:.1f}" y2="{py1:.1f}"/>')
            points = source["points"]
            for point in points:
                x = sx(point["k"])
                out.append(f'<line class="ci" x1="{x:.1f}" y1="{sy(point["ci_low"]):.1f}" x2="{x:.1f}" y2="{sy(point["ci_high"]):.1f}"/>')
            if len(points) > 1:
                path = " ".join(f'{"M" if i == 0 else "L"}{sx(p["k"]):.1f},{sy(p["p_hat"]):.1f}' for i, p in enumerate(points))
                out.append(f'<path class="line" d="{path}"/>')
            for point in points:
                tip = (f'k = {point["k"]}：{point["successes"]}/{point["n"]} 成功，P = {point["p_hat"]:.2f}，'
                       f'逐點 95% Wilson CI [{point["ci_low"]:.2f}, {point["ci_high"]:.2f}]')
                out.append(f'<circle class="dot" cx="{sx(point["k"]):.1f}" cy="{sy(point["p_hat"]):.1f}" r="4">'
                           f'<title>{html.escape(tip)}</title></circle>')
            out.append("</g>")
    out.append(f'<text class="t3" x="{left}" y="{height - 10}">source 的最後一個 checkpoint 以 x 軸上的短刻度標示；'
               f'資料：docs/results/summary.json</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- README


def check_publishable(summary: dict[str, Any]) -> None:
    """README 只接受正式、資料完整、Gate 有評估的結果；否則 ValueError。"""
    if summary["provenance"]["dry_run"]:
        raise ValueError("dry-run outputs are not results and never go into the README")
    if _gate_outcome(summary) is None:
        raise ValueError("the Gate was not evaluated (more than one response_model); the README is not updated")
    if not _data_complete(summary):
        raise ValueError("the sweep data are incomplete (missing or not completed sources); the README is not updated")


def render_readme_block(summary: dict[str, Any]) -> str:
    """README 的結果段落（``check_publishable`` 不通過 → ValueError）。"""
    check_publishable(summary)
    base, sweep, calls = summary["baseline"], summary["sweep"], summary["calls"]
    metrics = {m["name"]: m for m in base["metrics"]}
    success = metrics.get("success_rate", {})
    safe = metrics.get("trajectory_safe_rate", {})
    gate = _gate_outcome(summary)
    cliffs = sum(1 for s in sweep["sources"] if s["is_cliff"])
    logged = list(calls.values())
    if all(stats is not None and stats["complete"] for stats in logged):
        seeds = f"共 {sum(stats['calls'] for stats in logged)} 次模型呼叫，每次的 seed 都有記錄，並與 manifest 逐次嘗試核對相符（decision 032）。"
    else:
        seeds = "逐次呼叫紀錄不完整或缺少（見結果頁的「設定與紀錄」）。"
    rollouts = summary["provenance"].get("rollouts_per_task") or []
    design = f"15 題 × R = {rollouts[0]}" if len(rollouts) == 1 else "15 題"
    gate_line = ("所有 rerun 都成功，沒有可以定位的失敗（天花板效應）；依規則不進 Advanced。" if _no_failures(summary)
                 else GATE_TEXT.get(gate or "", ""))
    return "\n".join([
        f"**正式結果**（`{summary['provenance']['model']}`，凍結設定 decision 030；15 題，只能當初步觀察）：",
        "",
        f"- eval baseline（{design}，{base['rollouts']} 個 rollout）：success rate **{_fmt(success.get('value'))}**"
        f"（95% CI [{_fmt(success.get('ci_low'))}, {_fmt(success.get('ci_high'))}]），trajectory safe rate {_fmt(safe.get('value'))}。",
        f"- failure localization sweep：{len(sweep['sources'])} 個 source、{sweep['reruns']} 次 rerun，"
        f"{cliffs} 個 source 有 cliff；**Gate：`{gate}`**——{gate_line}",
        f"- {seeds}",
        "",
        "![recoverability curves](docs/results/recoverability.svg)",
        "",
        "完整的數字、每個 checkpoint 的表格、一個正式 trajectory 的完整步驟與失敗案例：[docs/results/README.md](docs/results/README.md)。",
    ])


def embed_results(readme: str, block: str) -> str:
    """把 README 中 README_START 與 README_END 之間換成 block（標記各自剛好一個、START 在前；否則 ValueError）。"""
    if readme.count(README_START) != 1 or readme.count(README_END) != 1:
        raise ValueError("README must contain exactly one results start marker and one end marker")
    start = readme.index(README_START) + len(README_START)
    end = readme.index(README_END)
    if end < start:
        raise ValueError("the results end marker comes before the start marker")
    return readme[:start] + "\n" + block + "\n" + readme[end:]


def write_report(baseline_runs: Sequence[str | Path], sweep_runs: Sequence[str | Path], out: str | Path, *,
                 replicates: int = 10000) -> dict[str, Any]:
    summary, baseline_md, sweep_md = _build(baseline_runs, sweep_runs, replicates)
    _write(Path(out), summary, baseline_md, sweep_md)
    return summary


def _write(out: Path, summary: dict[str, Any], baseline_md: str, sweep_md: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "README.md").write_text(render_page(summary), encoding="utf-8")
    (out / "recoverability.svg").write_text(recoverability_svg(summary), encoding="utf-8")
    (out / "baseline-analysis.md").write_text(baseline_md, encoding="utf-8")
    (out / "sweep-analysis.md").write_text(sweep_md, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--sweep-run", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--readme", default=None, help="把結果段落寫進這個 README（只接受正式、資料完整的結果）")
    parser.add_argument("--allow-dry-run", action="store_true", help="允許以 dry-run 的輸出產生報告（不會寫進 README）")
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        summary, baseline_md, sweep_md = _build(args.baseline_run, args.sweep_run, args.replicates)
        if summary["provenance"]["dry_run"] and not args.allow_dry_run:
            raise ValueError("these are dry-run outputs: pass --allow-dry-run to write a (labelled) report; never into the README")
        block = render_readme_block(summary) if args.readme else None  # 在寫任何檔案之前檢查
        _write(Path(args.out), summary, baseline_md, sweep_md)
        if args.readme and block is not None:
            readme = Path(args.readme)
            readme.write_text(embed_results(readme.read_text(encoding="utf-8"), block), encoding="utf-8")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"wrote {args.out}" + (f" and {args.readme}" if args.readme else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
