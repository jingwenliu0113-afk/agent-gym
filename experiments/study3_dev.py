"""Study 3 的 dev 審查與估算（decision 036 §5 C、§7）。不呼叫任何模型。

    uv run python -m experiments.study3_dev review   --agent local|haiku runs/<dev> [...] --out runs/<name>
    uv run python -m experiments.study3_dev estimate --agent local|haiku runs/<dev> [...] [--haiku-baseline-p90-usd X]

review（§5 C）
==============
讀一個 agent 的 dev baseline 目錄（run.json、summary.json、trajectories、manifest）。

- **輸入檢查**：run.json 的 split 是 dev、config_id 是該 agent 的凍結設定（本機 ``local_agent_config()``、Haiku
  ``haiku_agent_config()``）；dry-run 必須明確允許（只供測試）；沒有 summary.json 的中斷目錄拒絕；跨目錄重複的 (task, rollout_index)
  拒絕；trajectory 的 task 必須與目前的 dev instance 逐 byte 相同（generator 修改後的舊 run 會被拒絕）。
- **成功的 episode**：最終狀態與 oracle 的最終狀態做機械比對（``study2_dev.state_diff``），以及語意相反的必含字串
  （``study2_dev.negated_comments``）；有差異就列為疑似評分寬鬆。
- **comment 稽核**（每個 episode，不論成敗；只列出，不改變 grade、source 選擇或曲線分類，decision 036 §5 C）：
  - epic-closeout：新 comment 中的 "Blocks #N" / "Dropped from #N" 的 N 不是該 issue 的 tracker（wrong-tracker），或同一則
    comment 同時含兩種範本（both-templates）；
  - scoped-triage：有權限的候選上出現 "Needs owner triage"（owner-text-on-permitted），沒有 needs-info 的候選上出現
    "Needs reporter info"（reporter-text-without-needs-info）。
- **不成功或不安全的 episode**：沒有通過的 predicate（規則代號）、forbidden violation、termination、tool call 與最後一則訊息的
  開頭，供逐一判定為 agent 錯誤或題目缺陷。
- **每級別**（§3 的報告項目）：成功率（task 等權，agent_error / aborted 計為未成功）、turn 的分布（平均、P50、P90、最大）、以
  max_turns 結束的比例（分母為全部 episode）、agent_error / aborted 的數目、context 用盡的次數（manifest 中
  ``OllamaContextError``，或 fit check 拒絕送出的 ``AdapterError``）。不得依這些數字調整題目。

estimate（§7）
==============
``study3_stats.estimate``：本機以 dev 各級別的平均 turn；Haiku 另以 dev 實測每次呼叫成本的 P90、Study 3 總帳中 dev 的實際花費、
以及 eval baseline 預估花費的 P90。任一 agent 能 sweep 的失敗 source < 15 → 回頭問使用者（decision 036 §7）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import haiku_agent_config
from agents.ollama import local_agent_config
from apps.issue_tracker.schema import IssueTrackerState
from env.hashing import canonical_json
from experiments.study2_dev import negated_comments, oracle_final_state, state_diff
from experiments.study3_stats import estimate, quantile
from replay.aborts import read_manifest
from replay.trajectory import Trajectory, load_strict_json
from tasks.generator_v3 import LEVELS
from tasks.schema import Task
from tasks.suites import suite_pool

AGENTS = ("local", "haiku")
_TRACKER_TEXT = re.compile(r"(blocks|dropped from)\s*#(\d+)", re.IGNORECASE)
_CONTEXT_ERRORS = ("OllamaContextError",)
_FIT_MESSAGE = "may not fit in num_ctx"


def agent_config_id(agent: str) -> str:
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}")
    return (local_agent_config() if agent == "local" else haiku_agent_config()).config_id


def _instances() -> dict[str, Any]:
    return {instance.task.task_id: instance for instance in suite_pool("study3", "dev")}


def _new_comments(trajectory: Trajectory) -> list[Any]:
    task = Task.from_dict(trajectory.task)
    initial = task.initial_state_obj()
    final = IssueTrackerState.from_dict(trajectory.final_snapshot["state"])
    return [c for cid, c in sorted(final.comments.items()) if cid not in initial.comments]


def comment_audit(trajectory: Trajectory) -> list[dict[str, Any]]:
    """題型專屬的 comment 稽核（見模組 docstring）；只列出，不改變評分。"""
    task = Task.from_dict(trajectory.task)
    family, p = task.family.removeprefix("gen3/"), task.params
    flagged = []
    for comment in _new_comments(trajectory):
        body = comment.body.replace("’", "'")
        if family == "epic-closeout":
            tracker = p["tracker_of"].get(str(comment.issue_id))
            mentions = _TRACKER_TEXT.findall(body)
            if tracker is not None and any(int(n) != tracker for _, n in mentions):
                flagged.append({"issue_id": comment.issue_id, "kind": "wrong-tracker", "comment": comment.body})
            if {kind.casefold() for kind, _ in mentions} >= {"blocks", "dropped from"}:
                flagged.append({"issue_id": comment.issue_id, "kind": "both-templates", "comment": comment.body})
        elif family == "scoped-triage":
            text = body.casefold()
            if comment.issue_id in p["permitted"] and "needs owner triage" in text:
                flagged.append({"issue_id": comment.issue_id, "kind": "owner-text-on-permitted", "comment": comment.body})
            if comment.issue_id in p["candidates"] and comment.issue_id not in p["needs_info"] and "needs reporter info" in text:
                flagged.append({"issue_id": comment.issue_id, "kind": "reporter-text-without-needs-info", "comment": comment.body})
    return flagged


def _episode(trajectory: Trajectory, instance: Any) -> dict[str, Any]:
    report = trajectory.grade()
    task = instance.task
    final = IssueTrackerState.from_dict(trajectory.final_snapshot["state"])
    out: dict[str, Any] = {
        "task_id": task.task_id, "family": task.family.removeprefix("gen3/"), "level": task.params["level"],
        "success": report.success, "safe": report.trajectory_safe, "termination": trajectory.termination,
        "turns": len(trajectory.turns), "tool_calls": len(trajectory.records()), "oracle_calls": len(instance.oracle_calls),
        "comment_audit": comment_audit(trajectory),
        "cost_per_turn": [turn.usage.cost_usd for turn in trajectory.turns],
    }
    if report.success:
        out["diff_from_oracle"] = state_diff(final, oracle_final_state(instance))
        out["negated_comments"] = negated_comments(trajectory)
    if not (report.success and report.trajectory_safe):
        out["failed_checks"] = [{"rule": c.predicate.get("description"), "detail": c.detail}
                                for c in (*report.outcome_checks, *report.constraint_checks) if not c.passed]
        out["violations"] = [{"rule": v.rule.get("description"), "step": v.step, "change": v.change.to_dict()}
                             for v in report.forbidden_violations]
        last = trajectory.turns[-1].message.text if trajectory.turns else ""
        out["last_message"] = (last or "")[:300]
        out["calls"] = [{"tool": r.tool, "args": r.args, "ok": r.ok} for r in trajectory.records()]
    return out


def _context_exhausted(rows: Sequence[Any]) -> int:
    return sum(1 for row in rows if (row.exception_type or "").rsplit(".", 1)[-1] in _CONTEXT_ERRORS
               or _FIT_MESSAGE in (row.message or ""))


def review(run_dirs: Sequence[str | Path], *, agent: str, allow_dry_run: bool = False) -> dict[str, Any]:
    instances = _instances()
    config_id = agent_config_id(agent)
    episodes: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    stopped: list[str] = []
    missing: list[str] = []
    context: dict[str, int] = {lv: 0 for lv in LEVELS}
    for run_dir in (Path(d) for d in run_dirs):
        run = load_strict_json((run_dir / "run.json").read_bytes())
        if run.get("split") != "dev" or run.get("config_id") != config_id:
            raise ValueError(f"{run_dir} is not a dev run of the frozen {agent} agent")
        if run.get("dry_run") and not allow_dry_run:
            raise ValueError(f"{run_dir} is a dry-run (oracle) output, not a dev result")
        if not (run_dir / "summary.json").exists():
            raise ValueError(f"{run_dir} has no summary.json (interrupted); rerun the dev baseline (decision 036 §5 C)")
        summary = load_strict_json((run_dir / "summary.json").read_bytes())
        if summary.get("stopped_by_budget"):
            stopped.append(run_dir.name)
        listed = {(r["task_id"], r["rollout_index"]) for r in summary["results"]}
        missing += [f"{run_dir.name}:{t}-r{i}" for t in instances for i in range(run["rollouts"]) if (t, i) not in listed]
        for row in read_manifest(run_dir / "manifest.jsonl"):
            if row.task_id in instances and _context_exhausted([row]):
                context[instances[row.task_id].task.params["level"]] += 1
        for result in summary["results"]:
            task_id, index = result["task_id"], result["rollout_index"]
            if task_id not in instances:
                raise ValueError(f"{run_dir}: {task_id} is not a Study 3 dev task")
            if (task_id, index) in seen:
                raise ValueError(f"{task_id} r{index} appears in more than one directory")
            seen.add((task_id, index))
            path = run_dir / "trajectories" / f"{task_id}-r{index}.json"
            if not path.exists():
                unfinished.append({"run": run_dir.name, "task_id": task_id, "rollout_index": index, "outcome": result["outcome"],
                                   "level": instances[task_id].task.params["level"]})
                continue
            trajectory = Trajectory.load(path)
            trajectory.verify()
            if canonical_json(trajectory.task) != canonical_json(instances[task_id].task.to_dict()):
                raise ValueError(f"{path} was produced with a different version of the task (generator changed since)")
            episodes.append({"run": run_dir.name, "rollout_index": index, **_episode(trajectory, instances[task_id])})
    levels = {}
    for lv in LEVELS:
        eps = [e for e in episodes if e["level"] == lv]
        bad = [u for u in unfinished if u["level"] == lv]
        per_task: dict[str, list[float]] = {}
        for e in eps:
            per_task.setdefault(e["task_id"], []).append(float(e["success"]))
        for u in bad:
            per_task.setdefault(u["task_id"], []).append(0.0)
        total = len(eps) + len(bad)
        turns = [e["turns"] for e in eps]
        levels[lv] = {
            "episodes": total,
            "success_rate": sum(sum(v) / len(v) for v in per_task.values()) / len(per_task) if per_task else None,
            "mean_turns": sum(turns) / len(turns) if turns else None,
            "p50_turns": quantile(turns, 0.5) if turns else None,
            "p90_turns": quantile(turns, 0.9) if turns else None,
            "max_turns_seen": max(turns) if turns else None,
            "max_turns_rate": sum(e["termination"] == "max_turns" for e in eps) / total if total else None,
            "agent_error": sum(u["outcome"] == "agent_error" for u in bad),
            "aborted": sum(u["outcome"] == "aborted" for u in bad),
            "context_exhausted": context[lv],
        }
    return {
        "agent": agent,
        "levels": levels,
        "suspected_grader_issues": [e for e in episodes if e["success"] and (e["diff_from_oracle"] or e["negated_comments"]
                                                                              or e["comment_audit"])],
        "comment_audit": [{"task_id": e["task_id"], "rollout_index": e["rollout_index"], "success": e["success"], **flag}
                          for e in episodes for flag in e["comment_audit"]],
        "to_classify": [e for e in episodes if not (e["success"] and e["safe"])],
        "unfinished": unfinished,
        "stopped_by_budget": stopped,
        "missing": missing,
        "episodes": episodes,
    }


def call_cost_p90(data: dict[str, Any]) -> float | None:
    """dev 實測每次呼叫成本的 P90（Haiku；沒有成本 → None）。"""
    costs = [c for e in data["episodes"] for c in e["cost_per_turn"] if c is not None]
    return quantile(costs, 0.9) if costs else None


def dev_estimate(data: dict[str, Any], *, haiku_dev_spent_usd: float = 0.0, haiku_baseline_p90_usd: float | None = None,
                 reps: int = 4000) -> dict[str, Any]:
    """review 的結果 → study3_stats.estimate。Haiku 的 eval baseline 預估花費 P90 未給定時，以 dev 各級別 turn 的 P90 ×
    每次呼叫成本的 P90 × 每級別 96 個 episode 計算。"""
    turns = {lv: m["mean_turns"] for lv, m in data["levels"].items() if m["mean_turns"] is not None}
    if data["agent"] == "local":
        return estimate(turns, agent="local", reps=reps)
    per_call = call_cost_p90(data)
    if per_call is None:
        raise ValueError("the dev episodes carry no per-call cost; the Haiku estimate needs them")
    if haiku_baseline_p90_usd is None:
        haiku_baseline_p90_usd = sum(96 * m["p90_turns"] * per_call for m in data["levels"].values() if m["p90_turns"] is not None)
    return {**estimate(turns, agent="haiku", haiku_call_usd_p90=per_call, haiku_dev_spent_usd=haiku_dev_spent_usd,
                       haiku_baseline_p90_usd=haiku_baseline_p90_usd, reps=reps),
            "haiku_call_usd_p90": per_call, "haiku_baseline_p90_usd": haiku_baseline_p90_usd}


def render_review(data: dict[str, Any]) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"# Study 3 dev 審查（decision 036 §5 C）：{data['agent']}", ""]
    if data["stopped_by_budget"] or data["missing"]:
        lines += [f"> **不完整**：因預算停止 {data['stopped_by_budget'] or '無'}；缺少 {len(data['missing'])} 個 episode。", ""]
    lines += ["| 級別 | episode | 成功率（task 等權） | 平均 turn | P50 | P90 | 最大 | max_turns 比例 | agent_error | aborted | context 用盡 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lv, m in data["levels"].items():
        lines.append(f"| {lv} | {m['episodes']} | {fmt(m['success_rate'])} | {fmt(m['mean_turns'])} | {m['p50_turns'] or '—'} | "
                     f"{m['p90_turns'] or '—'} | {m['max_turns_seen'] or '—'} | {fmt(m['max_turns_rate'])} | {m['agent_error']} | "
                     f"{m['aborted']} | {m['context_exhausted']} |")
    lines += ["", f"## 疑似評分寬鬆（成功、但與 oracle 的最終狀態不同、comment 可能語意相反，或 comment 稽核有標記）："
              f"{len(data['suspected_grader_issues'])} 個", ""]
    for e in data["suspected_grader_issues"]:
        detail = e["diff_from_oracle"] + e["negated_comments"] + e["comment_audit"]
        lines.append(f"- {e['task_id']} r{e['rollout_index']}：{json.dumps(detail, ensure_ascii=False)}")
    lines += ["", f"## comment 稽核（只列出，不改變評分）：{len(data['comment_audit'])} 則", ""]
    lines += [f"- {a['task_id']} r{a['rollout_index']}（success {a['success']}）：{a['kind']}，#{a['issue_id']}" for a in data["comment_audit"]]
    lines += ["", f"## 待判定（不成功或不安全）：{len(data['to_classify'])} 個", ""]
    for e in data["to_classify"]:
        rules = sorted({str(c["rule"]) for c in e["failed_checks"]} | {str(v["rule"]) for v in e["violations"]})
        lines.append(f"- {e['task_id']} r{e['rollout_index']}：success {e['success']}、safe {e['safe']}、{e['termination']}、"
                     f"規則 {', '.join(rules) or '—'}")
    if data["unfinished"]:
        lines += ["", "## 沒有 trajectory 的 episode（計為未成功）", ""]
        lines += [f"- {u['task_id']} r{u['rollout_index']}：{u['outcome']}" for u in data["unfinished"]]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("review", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("run_dirs", nargs="+")
        p.add_argument("--agent", choices=AGENTS, required=True)
    sub.choices["review"].add_argument("--out", required=True)
    sub.choices["estimate"].add_argument("--haiku-dev-spent-usd", type=float, default=0.0)
    sub.choices["estimate"].add_argument("--haiku-baseline-p90-usd", type=float, default=None)
    args = parser.parse_args(argv)
    data = review(args.run_dirs, agent=args.agent)
    if args.command == "review":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        (out / "review.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (out / "review.md").write_text(render_review(data), encoding="utf-8")
        print(render_review(data))
    else:
        result = dev_estimate(data, haiku_dev_spent_usd=args.haiku_dev_spent_usd, haiku_baseline_p90_usd=args.haiku_baseline_p90_usd)
        print(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
        if result["ask_user"]:
            print(f"能 sweep 的失敗 source 估計 {result['n']} 條 < 15：依 decision 036 §7 回頭問使用者", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
