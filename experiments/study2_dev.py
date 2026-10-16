"""Study 2 的 dev 驗證與凍結前的估算（decision 034 §5 C、§7、§8）。不呼叫任何模型。

    uv run python -m experiments.study2_dev review   runs/<dev> [...] --out runs/<name>
    uv run python -m experiments.study2_dev estimate runs/<dev> [...] --source-rule rollout0|first_failure --cap N
    uv run python -m experiments.study2_dev power    [--reps 200]
    uv run python -m experiments.study2_dev assumed  [--reps 300]     # 凍結前、沒有 dev 資料時的假設估算

review（§5 C）
==============
讀 dev baseline 目錄（run.json、summary.json、trajectories）。

- **輸入檢查**：
  - run.json 的 split 必須是 dev、config_id 是凍結的本機設定；dry-run 必須明確允許（只供測試）；
  - 沒有 summary.json 的中斷目錄拒絕，要重跑；
  - 跨目錄重複的 (task, rollout_index) 拒絕；
  - trajectory 的 task 必須與目前的 dev instance 逐 byte 相同，修正 generator 之後的舊 run 會被拒絕。
- **成功的 episode**：
  - 最終狀態與 oracle 的最終狀態做機械比對：每個 issue 的 MUTABLE 欄位與 comment 數，comment 文字不比。有差異就列為
    「疑似 grader 漏洞」，逐一判定；
  - **語意相反的 comment**：必含字串是不分大小寫的子字串比對，"no longer blocks #7" 也會通過。每一次出現都檢查：前一個字元是
    字母（"Unblocks"），或之前（最多隔一個字）有 not / no longer / n't / never（含彎引號 ’），就列出來人工判定（已知限制）。
- **不成功或不安全的 episode**：列出沒有通過的 predicate（含規則代號）、forbidden violation、termination、tool call 與
  最後一則訊息的開頭，供逐一判定為 agent 錯誤或題目缺陷。
- **每級別**：
  - 成功率：task 等權，agent_error / aborted 計為未成功，與 §7 相同；
  - max_turns 終止的比例：分母為全部 episode；L3 的值在凍結前必須報告；
  - 平均 turn 數：只算有 trajectory 的 episode；
  - agent_error / aborted 的數目；
  - 是否因預算停止，以及缺少的 episode。

estimate（§8 的估算方法）
=========================
以 dev 各級別的平均 turn 數 L 與成功率 p，對 eval 的 36 個 source 做 Monte Carlo（``reps`` 次，seed 0）：

- **source 的成敗**：
  - rollout0：以機率 p 成功；
  - first_failure：R = 3 次中有任何一次失敗（機率 1 − p³）就是失敗的 source。
- **曲線**：
  - 成功的 source 由 p 線性升到 1；``success_curve="flat"`` 為對照，維持平線 p；
  - 失敗的 source 一半是中點 cliff（p → 0），一半是線性（p → 0）。
- **排程與成本**：與 sweep 相同的 ``run_protocol``（登記的 plan 與判定）。每次 rerun 的成本 = 剩下的 turn 數（L − k）。

回報 P50 / P90 / 平均。P90 超過上限 → 必須先問使用者（§8）。任一級別沒有成功率或 turn 數 → ValueError。

power（凍結前模擬：max_turns 混淆）
====================================
只用 CPU。對 L ∈ {13, 16, 20}、失敗的 rerun 用完 turn 的比例 f ∈ {0, 0.05, 0.1, 0.2}，以登記的 plan 與判定，比較兩種規則
（``study2_analysis.RULES``：round1 與 excluding）：

- cliff 情境的偵測率；
- gradual / flat 情境的誤判率；
- 「晚期 checkpoint 剩下的 turn 不夠」情境（flat 0.9，最後 3 個 checkpoint 有 60% 的 rerun 用完 turn）被當成 cliff 的比例。

只報告，不修改 plan 或門檻。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

from agents.ollama import local_agent_config
from apps.issue_tracker.schema import MUTABLE_ISSUE_FIELDS, IssueTrackerState
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from experiments.run_sweep import REGISTERED_CRITERION, REGISTERED_PLAN
from experiments.study2_analysis import RULES, assess_rule
from experiments.sweep_power import cliff_curve, linear_curve
from replay.localization import RolloutOutcome, SweepPlan, SweepResult, assess_drop, estimate_checkpoint, run_protocol
from replay.trajectory import Trajectory, load_strict_json
from tasks.generator_v2 import LEVELS
from tasks.schema import Task
from tasks.suites import suite_pool

POWER_LENGTHS = (13, 16, 20)
POWER_TURN_OUT = (0.0, 0.05, 0.1, 0.2)
SOURCE_RULES = ("rollout0", "first_failure")
SUCCESS_CURVES = ("rising", "flat")
ASSUMED = {  # decision 034「凍結前模擬」第 2 點的兩組假設（沒有 dev 資料時）
    "long": {"L1": {"mean_turns": 7, "success_rate": 0.85}, "L2": {"mean_turns": 13, "success_rate": 0.6},
             "L3": {"mean_turns": 18, "success_rate": 0.35}},
    "short": {"L1": {"mean_turns": 6, "success_rate": 0.85}, "L2": {"mean_turns": 10, "success_rate": 0.6},
              "L3": {"mean_turns": 13, "success_rate": 0.35}},
}


# ---------------------------------------------------------------- review


def _instances(split: str = "dev") -> dict[str, Any]:
    return {instance.task.task_id: instance for instance in suite_pool("study2", split)}


def oracle_final_state(instance: Any) -> IssueTrackerState:
    env = IssueTrackerEnv()
    env.reset(instance.task.initial_state, instance.task.actor_id)
    for name, args in instance.oracle_calls:
        env.call_tool(name, args)
    return env.current_state()


def state_diff(final: IssueTrackerState, expected: IssueTrackerState) -> list[dict[str, Any]]:
    """每個 issue 的 MUTABLE 欄位與 comment 數的差異（comment 文字不比）。"""
    diffs = []
    for issue_id in sorted(expected.issues):
        a, b = final.issues[issue_id], expected.issues[issue_id]
        for name in MUTABLE_ISSUE_FIELDS:
            va, vb = getattr(a, name), getattr(b, name)
            if (list(va) if name == "labels" else va) != (list(vb) if name == "labels" else vb):
                diffs.append({"issue_id": issue_id, "field": name, "episode": list(va) if name == "labels" else va,
                              "oracle": list(vb) if name == "labels" else vb})
        ca = sum(1 for c in final.comments.values() if c.issue_id == issue_id)
        cb = sum(1 for c in expected.comments.values() if c.issue_id == issue_id)
        if ca != cb:
            diffs.append({"issue_id": issue_id, "field": "comment_count", "episode": ca, "oracle": cb})
    return diffs


# 必含字串之前（最多隔一個字）的否定詞，例如 "no longer blocks #7"、"doesn't block"、"not blocks"
_NEGATION_BEFORE = re.compile(r"(\bnot|\bno longer|n't|\bnever)\W+(\w+\W+)?$", re.IGNORECASE)


def negated_comments(trajectory: Trajectory) -> list[dict[str, Any]]:
    """成功的 episode 中，符合 contains、但可能語意相反的新 comment（見模組 docstring）。"""
    task = Task.from_dict(trajectory.task)
    initial = task.initial_state_obj()
    final = IssueTrackerState.from_dict(trajectory.final_snapshot["state"])
    flagged = []
    for predicate in task.expected_outcome:
        if predicate["type"] != "comment_added":
            continue
        for cid, comment in sorted(final.comments.items()):
            if cid in initial.comments or comment.issue_id != predicate["issue_id"]:
                continue
            body = comment.body.replace("’", "'").casefold()
            for text in predicate.get("contains", []):
                needle = text.casefold()
                at = body.find(needle)
                while at >= 0:
                    if (at > 0 and body[at - 1].isalpha()) or _NEGATION_BEFORE.search(body[:at]):
                        flagged.append({"issue_id": comment.issue_id, "comment": comment.body, "contains": text})
                        break
                    at = body.find(needle, at + 1)
    return flagged


def _episode(trajectory: Trajectory, instance: Any) -> dict[str, Any]:
    report = trajectory.grade()
    task = instance.task
    final = IssueTrackerState.from_dict(trajectory.final_snapshot["state"])
    out: dict[str, Any] = {
        "task_id": task.task_id, "family": task.family.removeprefix("gen2/"), "level": task.params["level"],
        "outcome": "success" if report.success else "failure",
        "success": report.success, "safe": report.trajectory_safe, "termination": trajectory.termination,
        "turns": len(trajectory.turns), "tool_calls": len(trajectory.records()), "oracle_calls": len(instance.oracle_calls),
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


def review(run_dirs: Sequence[str | Path], *, allow_dry_run: bool = False) -> dict[str, Any]:
    instances = _instances("dev")
    config_id = local_agent_config().config_id
    episodes: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    stopped: list[str] = []
    missing: list[str] = []
    for run_dir in (Path(d) for d in run_dirs):
        run = load_strict_json((run_dir / "run.json").read_bytes())
        if run.get("split") != "dev" or run.get("config_id") != config_id:
            raise ValueError(f"{run_dir} is not a dev run of the frozen local agent")
        if run.get("dry_run") and not allow_dry_run:
            raise ValueError(f"{run_dir} is a dry-run (oracle) output, not a dev result")
        if not (run_dir / "summary.json").exists():
            raise ValueError(f"{run_dir} has no summary.json (interrupted); rerun the dev baseline (decision 034 §5 C)")
        summary = load_strict_json((run_dir / "summary.json").read_bytes())
        if summary.get("stopped_by_budget"):
            stopped.append(run_dir.name)
        listed = {(r["task_id"], r["rollout_index"]) for r in summary["results"]}
        missing += [f"{run_dir.name}:{t}-r{i}" for t in instances for i in range(run["rollouts"]) if (t, i) not in listed]
        for result in summary["results"]:
            task_id, index = result["task_id"], result["rollout_index"]
            if task_id not in instances:
                raise ValueError(f"{run_dir}: {task_id} is not a Study 2 dev task")
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
        levels[lv] = {
            "episodes": total,
            "success_rate": sum(sum(v) / len(v) for v in per_task.values()) / len(per_task) if per_task else None,
            "mean_turns": sum(e["turns"] for e in eps) / len(eps) if eps else None,
            "max_turns_rate": sum(e["termination"] == "max_turns" for e in eps) / total if total else None,
            "agent_error": sum(u["outcome"] == "agent_error" for u in bad),
            "aborted": sum(u["outcome"] == "aborted" for u in bad),
        }
    return {
        "levels": levels,
        "suspected_grader_issues": [e for e in episodes if e["success"] and (e["diff_from_oracle"] or e["negated_comments"])],
        "to_classify": [e for e in episodes if not (e["success"] and e["safe"])],
        "unfinished": unfinished,
        "stopped_by_budget": stopped,
        "missing": missing,
        "episodes": episodes,
    }


def render_review(data: dict[str, Any]) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = ["# Study 2 dev 驗證（decision 034 §5 C）", ""]
    if data["stopped_by_budget"] or data["missing"]:
        lines += [f"> **不完整**：因預算停止 {data['stopped_by_budget'] or '無'}；缺少 {len(data['missing'])} 個 episode。", ""]
    lines += ["| 級別 | episode | 成功率（task 等權） | 平均 turn | max_turns | agent_error | aborted |", "|---|---|---|---|---|---|---|"]
    for lv, m in data["levels"].items():
        lines.append(f"| {lv} | {m['episodes']} | {fmt(m['success_rate'])} | {fmt(m['mean_turns'])} | {fmt(m['max_turns_rate'])} | "
                     f"{m['agent_error']} | {m['aborted']} |")
    lines += ["", f"## 疑似 grader 漏洞（成功、但最終狀態與 oracle 不同，或 comment 可能語意相反）：{len(data['suspected_grader_issues'])} 個", ""]
    for e in data["suspected_grader_issues"]:
        lines.append(f"- {e['task_id']} r{e['rollout_index']}：{json.dumps(e['diff_from_oracle'] + e['negated_comments'], ensure_ascii=False)}")
    lines += ["", f"## 待判定（不成功或不安全）：{len(data['to_classify'])} 個", ""]
    for e in data["to_classify"]:
        rules = sorted({str(c["rule"]) for c in e["failed_checks"]} | {str(v["rule"]) for v in e["violations"]})
        lines.append(f"- {e['task_id']} r{e['rollout_index']}：success {e['success']}、safe {e['safe']}、{e['termination']}、"
                     f"規則 {', '.join(rules) or '—'}")
    if data["unfinished"]:
        lines += ["", "## 沒有 trajectory 的 episode（計為未成功）", ""]
        lines += [f"- {u['task_id']} r{u['rollout_index']}：{u['outcome']}" for u in data["unfinished"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- estimate


def _sweep_cost(curve: Sequence[float], plan: SweepPlan, rng: random.Random, *, turn_out: float = 0.0,
                late: tuple[int, float] | None = None, criterion: str = REGISTERED_CRITERION,
                rules: Sequence[str] = ("round1",)) -> tuple[int, dict[str, tuple[bool, int | None]]]:
    """以 run_protocol 排程；回傳 (模型呼叫數 = Σ 每次 rerun 剩下的 turn 數, {規則: (is_cliff, largest_drop_k)})。
    turn_out：失敗的 rerun 用完 turn 的比例；late = (k0, q)：k >= k0 的 rerun 以機率 q 直接用完 turn（失敗）。"""
    length = len(curve)

    def rollout(k: int, i: int) -> RolloutOutcome:
        if late is not None and k >= late[0] and rng.random() < late[1]:
            return RolloutOutcome(k, i, "failure", "max_turns", False, 1)
        if rng.random() < curve[k]:
            return RolloutOutcome(k, i, "success", "final_answer", False, 1)
        return RolloutOutcome(k, i, "failure", "max_turns" if rng.random() < turn_out else "final_answer", False, 1)

    protocol = run_protocol(length - 1, plan, rollout)
    estimates = tuple(estimate_checkpoint(k, list(protocol.outcomes[k])) for k in sorted(protocol.outcomes))
    calls = sum(len(protocol.outcomes[k]) * (length - k) for k in protocol.outcomes)
    raw = assess_drop(estimates, plan.drop_threshold, criterion=criterion)
    result = SweepResult(task_id="sim", source_trajectory_id="sim", plan=plan, estimates=estimates,
                         rollouts=tuple(o for k in sorted(protocol.outcomes) for o in protocol.outcomes[k]), refined_intervals=(),
                         largest_drop=raw.largest_drop, largest_drop_k=raw.largest_drop_k, is_cliff=raw.is_cliff, criterion=criterion,
                         max_turns_confounded=raw.max_turns_confounded)
    verdicts = {}
    for rule in rules:
        assessed = assess_rule(result, rule)
        verdicts[rule] = (assessed["is_cliff"], assessed["largest_drop_k"])
    return calls, verdicts


def estimate_sweep_calls(levels: dict[str, dict[str, Any]], *, source_rule: str, reps: int = 2000, seed: int = 0,
                         plan: SweepPlan = REGISTERED_PLAN, rollouts: int = 3, success_curve: str = "rising") -> dict[str, Any]:
    """levels：每級別的 {"mean_turns", "success_rate"}（dev 實測）。回傳 P50 / P90 / 平均。"""
    if source_rule not in SOURCE_RULES:
        raise ValueError(f"source_rule must be one of {SOURCE_RULES}")
    if success_curve not in SUCCESS_CURVES:
        raise ValueError(f"success_curve must be one of {SUCCESS_CURVES}")
    for lv in LEVELS:
        m = levels.get(lv) or {}
        if m.get("mean_turns") is None or m.get("success_rate") is None:
            raise ValueError(f"level {lv} has no measured success rate or turn count")
    tasks = [instance.task for instance in suite_pool("study2", "eval")]
    rng = random.Random(f"study2-estimate|{seed}|{source_rule}")
    totals = []
    for _ in range(reps):
        total = 0
        for task in tasks:
            m = levels[task.params["level"]]
            length = max(2, round(m["mean_turns"]))
            p = m["success_rate"]
            failed = (rng.random() >= p) if source_rule == "rollout0" else any(rng.random() >= p for _ in range(rollouts))
            if not failed:
                curve = linear_curve(length, p, 1.0) if success_curve == "rising" else (p,) * length
            elif rng.random() < 0.5:
                curve = cliff_curve(length, (length - 1) // 2, p, 0.0) if length > 2 else (p, 0.0)
            else:
                curve = linear_curve(length, p, 0.0)
            total += _sweep_cost(curve, plan, rng)[0]
        totals.append(total)
    totals.sort()
    return {"source_rule": source_rule, "success_curve": success_curve, "reps": reps, "seed": seed, "p50": totals[len(totals) // 2],
            "p90": totals[min(len(totals) - 1, math.ceil(0.9 * len(totals)) - 1)], "mean": sum(totals) / len(totals),
            "levels": levels}


# ---------------------------------------------------------------- power


def max_turns_power(*, reps: int = 200, seed: int = 0, plan: SweepPlan = REGISTERED_PLAN) -> list[dict[str, Any]]:
    rows = []
    for length in POWER_LENGTHS:
        scenarios: list[tuple[str, str, tuple[float, ...], int | None, tuple[int, float] | None]] = []
        for p_pre, p_post in ((0.9, 0.1), (0.8, 0.2), (0.7, 0.2), (0.6, 0.1)):
            for m in sorted({0, (length - 1) // 2, length - 2}):
                scenarios.append((f"cliff-{p_pre}-{p_post}-m{m}", "cliff", cliff_curve(length, m, p_pre, p_post), m, None))
        scenarios.append(("gradual-0.9-0.1", "gradual", linear_curve(length, 0.9, 0.1), None, None))
        scenarios.append(("flat-0.9", "flat", (0.9,) * length, None, None))
        scenarios.append(("flat-0.9-late-exhaustion", "late", (0.9,) * length, None, (length - 3, 0.6)))
        for f in POWER_TURN_OUT:
            for name, kind, curve, m, late in scenarios:
                rng = random.Random(f"{seed}|{length}|{f}|{name}")
                detected = {rule: 0 for rule in RULES}
                flagged = {rule: 0 for rule in RULES}
                for _ in range(reps):
                    _, verdicts = _sweep_cost(curve, plan, rng, turn_out=f, late=late, rules=RULES)
                    for rule, (is_cliff, k) in verdicts.items():
                        flagged[rule] += is_cliff
                        detected[rule] += is_cliff and m is not None and k == m
                rows.append({"length": length, "turn_out": f, "scenario": name, "kind": kind, "reps": reps,
                             **{f"detect_{r}": (detected[r] / reps if kind == "cliff" else None) for r in RULES},
                             **{f"flag_{r}": flagged[r] / reps for r in RULES}})
    return rows


def summarize_power(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for length in POWER_LENGTHS:
        for f in POWER_TURN_OUT:
            sel = [r for r in rows if r["length"] == length and r["turn_out"] == f]
            row: dict[str, Any] = {"length": length, "turn_out": f}
            for rule in RULES:
                cliffs = [r[f"detect_{rule}"] for r in sel if r["kind"] == "cliff"]
                row[f"mean_detect_{rule}"] = sum(cliffs) / len(cliffs)
                row[f"max_false_flag_{rule}"] = max(r[f"flag_{rule}"] for r in sel if r["kind"] in ("gradual", "flat"))
                row[f"late_exhaustion_flag_{rule}"] = next(r[f"flag_{rule}"] for r in sel if r["kind"] == "late")
            out.append(row)
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("runs", nargs="+")
    e.add_argument("--source-rule", choices=SOURCE_RULES, required=True)
    e.add_argument("--cap", type=int, required=True)
    e.add_argument("--reps", type=int, default=2000)
    p = sub.add_parser("power")
    p.add_argument("--reps", type=int, default=200)
    a = sub.add_parser("assumed")
    a.add_argument("--reps", type=int, default=300)
    args = parser.parse_args(argv)
    if args.command == "review":
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            parser.error(f"{out} already exists and is not empty")
        try:
            data = review(args.runs)
        except ValueError as exc:
            parser.error(str(exc))
        out.mkdir(parents=True, exist_ok=True)
        (out / "review.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out / "review.md").write_text(render_review(data), encoding="utf-8")
        print(f"wrote {out}")
    elif args.command == "estimate":
        levels = review(args.runs)["levels"]
        result = estimate_sweep_calls(levels, source_rule=args.source_rule, reps=args.reps)
        result["cap"] = args.cap
        result["over_cap"] = result["p90"] > args.cap
        print(json.dumps(result, ensure_ascii=False, indent=1))
        if result["over_cap"]:
            print("P90 exceeds the cap: ask the user before any eval run (decision 034 §8)")
    elif args.command == "power":
        print(json.dumps(summarize_power(max_turns_power(reps=args.reps)), ensure_ascii=False, indent=1))
    else:
        rows = []
        for name, levels in ASSUMED.items():
            for rule in SOURCE_RULES:
                for curve in SUCCESS_CURVES:
                    res = estimate_sweep_calls(levels, source_rule=rule, reps=args.reps, success_curve=curve)
                    rows.append({"assumption": name, "source_rule": rule, "success_curve": curve, "p50": res["p50"], "p90": res["p90"],
                                 "mean": round(res["mean"])})
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
