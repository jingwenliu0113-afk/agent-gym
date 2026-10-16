"""Study 2 的 dev 驗證與凍結前的估算（decision 034 §5 C、§7、§8）。不呼叫任何模型。

    uv run python -m experiments.study2_dev review   runs/<dev> [...] --out runs/<name>
    uv run python -m experiments.study2_dev estimate runs/<dev> [...] --source-rule rollout0|first_failure --cap N
    uv run python -m experiments.study2_dev power    [--reps 200]

review（§5 C）
==============
讀 dev baseline 目錄的 summary.json 與 trajectories。

- **成功的 episode**：最終狀態與 oracle 的最終狀態做機械比對。比對每個 issue 的 MUTABLE 欄位與 comment 數，comment 文字不比。
  有差異就列為「疑似 grader 漏洞」，要逐一判定。
- **不成功或不安全的 episode**：列出沒有通過的 predicate（含規則代號）、forbidden violation、termination 與最後一則訊息的開頭，
  供逐一判定為 agent 錯誤或題目缺陷。
- 每級別：dev 的成功率、平均 turn 數，以及 max_turns 終止的比例。L3 的 max_turns 比例在凍結前必須報告。

estimate（§8 的估算方法）
=========================
以 dev 各級別的平均 turn 數 L 與成功率 p，對 eval 的 36 個 source 做 Monte Carlo（``reps`` 次，seed 0）：

- **source 的成敗**：
  - rollout0：以機率 p 成功；
  - first_failure：R = 3 次中有任何一次失敗（機率 1 − p³）就是失敗的 source。
- **曲線**：成功的 source 由 p 線性升到 1（k = 0 等於重新跑一次；成功 source 的最後一點幾乎一定成功）；
  失敗的 source 一半是中點 cliff（p → 0），一半是線性（p → 0）。
- **排程與成本**：與 sweep 相同的 ``run_protocol``（登記的 plan 與判定）。每次 rerun 的成本 = 剩下的 turn 數（L − k）。

回報 P50 / P90 / 平均。P90 超過上限 → 必須先問使用者（§8）。

power（§7 的敏感度：max_turns 混淆）
=====================================
只用 CPU。對 L ∈ {13, 16, 20}、失敗 rerun 用完 turn 的比例 f ∈ {0, 0.05, 0.1, 0.2}，以登記的 plan 與判定（含 max_turns 混淆規則），
報告 cliff 情境的偵測率與 gradual / flat 情境的誤判率。只報告，不修改 plan 或門檻。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from apps.issue_tracker.schema import MUTABLE_ISSUE_FIELDS, IssueTrackerState
from env.environment import IssueTrackerEnv
from experiments.run_sweep import REGISTERED_CRITERION, REGISTERED_PLAN
from experiments.sweep_power import cliff_curve, linear_curve
from replay.localization import RolloutOutcome, SweepPlan, assess_drop, estimate_checkpoint, run_protocol
from replay.trajectory import Trajectory
from tasks.generator_v2 import LEVELS
from tasks.suites import suite_pool

POWER_LENGTHS = (13, 16, 20)
POWER_TURN_OUT = (0.0, 0.05, 0.1, 0.2)
SOURCE_RULES = ("rollout0", "first_failure")


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


def _episode(trajectory: Trajectory, instance: Any) -> dict[str, Any]:
    report = trajectory.grade()
    task = instance.task
    final = IssueTrackerState.from_dict(trajectory.final_snapshot["state"])
    out: dict[str, Any] = {
        "task_id": task.task_id, "family": task.family.removeprefix("gen2/"), "level": task.params["level"],
        "success": report.success, "safe": report.trajectory_safe, "termination": trajectory.termination,
        "turns": len(trajectory.turns), "tool_calls": len(trajectory.records()), "oracle_calls": len(instance.oracle_calls),
    }
    if report.success:
        out["diff_from_oracle"] = state_diff(final, oracle_final_state(instance))
    if not (report.success and report.trajectory_safe):
        out["failed_checks"] = [{"rule": c.predicate.get("description"), "detail": c.detail}
                                for c in (*report.outcome_checks, *report.constraint_checks) if not c.passed]
        out["violations"] = [{"rule": v.rule.get("description"), "step": v.step, "change": v.change.to_dict()}
                             for v in report.forbidden_violations]
        last = trajectory.turns[-1].message.text if trajectory.turns else ""
        out["last_message"] = (last or "")[:300]
        out["calls"] = [{"tool": r.tool, "args": r.args, "ok": r.ok} for r in trajectory.records()]
    return out


def review(run_dirs: Sequence[str | Path]) -> dict[str, Any]:
    instances = _instances("dev")
    episodes: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    for run_dir in (Path(d) for d in run_dirs):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for result in summary["results"]:
            task_id, index = result["task_id"], result["rollout_index"]
            if task_id not in instances:
                raise ValueError(f"{run_dir}: {task_id} is not a Study 2 dev task")
            path = run_dir / "trajectories" / f"{task_id}-r{index}.json"
            if not path.exists():
                unfinished.append({"run": run_dir.name, "task_id": task_id, "rollout_index": index, "outcome": result["outcome"]})
                continue
            trajectory = Trajectory.load(path)
            trajectory.verify()
            episodes.append({"run": run_dir.name, "rollout_index": index, **_episode(trajectory, instances[task_id])})
    levels = {}
    for lv in LEVELS:
        eps = [e for e in episodes if e["level"] == lv]
        levels[lv] = {
            "episodes": len(eps),
            "success_rate": sum(e["success"] for e in eps) / len(eps) if eps else None,
            "mean_turns": sum(e["turns"] for e in eps) / len(eps) if eps else None,
            "max_turns_rate": sum(e["termination"] == "max_turns" for e in eps) / len(eps) if eps else None,
        }
    return {
        "levels": levels,
        "suspected_grader_issues": [e for e in episodes if e["success"] and e["diff_from_oracle"]],
        "to_classify": [e for e in episodes if not (e["success"] and e["safe"])],
        "unfinished": unfinished,
        "episodes": episodes,
    }


def render_review(data: dict[str, Any]) -> str:
    lines = ["# Study 2 dev 驗證（decision 034 §5 C）", "", "| 級別 | episode | 成功率 | 平均 turn | max_turns |", "|---|---|---|---|---|"]
    for lv, m in data["levels"].items():
        fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {lv} | {m['episodes']} | {fmt(m['success_rate'])} | {fmt(m['mean_turns'])} | {fmt(m['max_turns_rate'])} |")
    lines += ["", f"## 疑似 grader 漏洞（成功、但最終狀態與 oracle 不同）：{len(data['suspected_grader_issues'])} 個", ""]
    for e in data["suspected_grader_issues"]:
        lines.append(f"- {e['task_id']} r{e['rollout_index']}：{json.dumps(e['diff_from_oracle'], ensure_ascii=False)}")
    lines += ["", f"## 待判定（不成功或不安全）：{len(data['to_classify'])} 個", ""]
    for e in data["to_classify"]:
        rules = sorted({c["rule"] for c in e["failed_checks"]} | {v["rule"] for v in e["violations"]})
        lines.append(f"- {e['task_id']} r{e['rollout_index']}：success {e['success']}、safe {e['safe']}、{e['termination']}、"
                     f"規則 {', '.join(map(str, rules)) or '—'}")
    if data["unfinished"]:
        lines += ["", "## 沒有完成的 episode", ""] + [f"- {u['task_id']} r{u['rollout_index']}：{u['outcome']}" for u in data["unfinished"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- estimate


def _sweep_cost(curve: Sequence[float], plan: SweepPlan, rng: random.Random, *, turn_out: float = 0.0,
                criterion: str = REGISTERED_CRITERION) -> tuple[int, bool, int | None]:
    """以 run_protocol 排程；回傳 (模型呼叫數 = Σ 每次 rerun 剩下的 turn 數, is_cliff, largest_drop_k)。"""
    length = len(curve)

    def rollout(k: int, i: int) -> RolloutOutcome:
        if rng.random() < curve[k]:
            return RolloutOutcome(k, i, "success", "final_answer", False, 1)
        return RolloutOutcome(k, i, "failure", "max_turns" if rng.random() < turn_out else "final_answer", False, 1)

    protocol = run_protocol(length - 1, plan, rollout)
    estimates = tuple(estimate_checkpoint(k, list(protocol.outcomes[k])) for k in sorted(protocol.outcomes))
    assessed = assess_drop(estimates, plan.drop_threshold, criterion=criterion)
    calls = sum(len(protocol.outcomes[k]) * (length - k) for k in protocol.outcomes)
    return calls, assessed.is_cliff, assessed.largest_drop_k


def estimate_sweep_calls(levels: dict[str, dict[str, Any]], *, source_rule: str, reps: int = 2000, seed: int = 0,
                         plan: SweepPlan = REGISTERED_PLAN, rollouts: int = 3) -> dict[str, Any]:
    """levels：每級別的 {"mean_turns", "success_rate"}（dev 實測）。回傳 P50 / P90 / 平均。"""
    if source_rule not in SOURCE_RULES:
        raise ValueError(f"source_rule must be one of {SOURCE_RULES}")
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
                curve = linear_curve(length, p, 1.0)
            elif rng.random() < 0.5:
                curve = cliff_curve(length, (length - 1) // 2, p, 0.0) if length > 2 else (p, 0.0)
            else:
                curve = linear_curve(length, p, 0.0)
            total += _sweep_cost(curve, plan, rng)[0]
        totals.append(total)
    totals.sort()
    return {"source_rule": source_rule, "reps": reps, "seed": seed, "p50": totals[len(totals) // 2],
            "p90": totals[min(len(totals) - 1, math.ceil(0.9 * len(totals)) - 1)], "mean": sum(totals) / len(totals),
            "levels": levels}


# ---------------------------------------------------------------- power


def max_turns_power(*, reps: int = 200, seed: int = 0, plan: SweepPlan = REGISTERED_PLAN) -> list[dict[str, Any]]:
    rows = []
    for length in POWER_LENGTHS:
        scenarios: list[tuple[str, str, tuple[float, ...], int | None]] = []
        for p_pre, p_post in ((0.9, 0.1), (0.8, 0.2), (0.7, 0.2), (0.6, 0.1)):
            for m in sorted({0, (length - 1) // 2, length - 2}):
                scenarios.append((f"cliff-{p_pre}-{p_post}-m{m}", "cliff", cliff_curve(length, m, p_pre, p_post), m))
        scenarios.append(("gradual-0.9-0.1", "gradual", linear_curve(length, 0.9, 0.1), None))
        scenarios.append(("flat-0.9", "flat", (0.9,) * length, None))
        for f in POWER_TURN_OUT:
            for name, kind, curve, m in scenarios:
                rng = random.Random(f"{seed}|{length}|{f}|{name}")
                detected = flagged = 0
                for _ in range(reps):
                    _, is_cliff, k = _sweep_cost(curve, plan, rng, turn_out=f)
                    flagged += is_cliff
                    detected += is_cliff and m is not None and k == m
                rows.append({"length": length, "turn_out": f, "scenario": name, "kind": kind, "reps": reps,
                             "detect_rate": detected / reps if kind == "cliff" else None, "flag_rate": flagged / reps})
    return rows


def summarize_power(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for length in POWER_LENGTHS:
        for f in POWER_TURN_OUT:
            sel = [r for r in rows if r["length"] == length and r["turn_out"] == f]
            cliffs = [r["detect_rate"] for r in sel if r["kind"] == "cliff"]
            others = [r["flag_rate"] for r in sel if r["kind"] != "cliff"]
            out.append({"length": length, "turn_out": f, "mean_detect": sum(cliffs) / len(cliffs), "max_false_flag": max(others)})
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
    args = parser.parse_args(argv)
    if args.command == "review":
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            parser.error(f"{out} already exists and is not empty")
        data = review(args.runs)
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
    else:
        print(json.dumps(summarize_power(max_turns_power(reps=args.reps)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
