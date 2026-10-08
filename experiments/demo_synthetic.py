"""Keyless 示範（README 首屏用）：Task → Agent actions → state changes → grader → replay from checkpoint → P(success | k)。

**這不是 LLM 結果**：agent 是解析解已知的 synthetic policy（decision 010），用來展示整條流程與 replay 能回答的問題；
真正的 LLM 結果要等 paid runs（目前 deferred / blocked-by-paid-API）。

用法::

    uv run python -m experiments.demo_synthetic --out docs/demo

輸出：``<out>/demo.md``（README 引用的文字）與 ``<out>/curve.svg``（recoverability curve）。輸出完全決定性（固定 seed、常數 clock）。
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agents.baseline import BASELINE_MAX_TURNS
from agents.synthetic import SyntheticPolicy, checkpoint_truth, mistake_call
from env.hashing import canonical_json
from grader.schema import GradeReport
from replay.localization import SweepPlan, SweepResult, sweep
from replay.runner import run_episode
from replay.trajectory import Trajectory
from tasks.generator import GeneratedInstance, week1_pool

DEMO_TASK_ID = "gen-reassign-departed-s0"  # dev split（不計入任何統計）；oracle 7 個 call
DEMO_Q = 0.05
DEMO_MISTAKE_TURN = 3  # source 在第 3 個 turn（0-based）犯下不可逆錯誤
DEMO_PLAN = SweepPlan(coarse_stride=2, n_coarse=8, n_fine=24, drop_threshold=0.3)  # decision 010 的流程驗證預設
DEMO_CRITERION = "wilson95"
SOURCE_SEED_SEARCH_LIMIT = 10000


@dataclass(frozen=True)
class DemoResult:
    task_id: str
    instruction: str
    source_seed: int  # 決定性搜尋到的第一個 seed
    source: Trajectory
    source_grade: GradeReport
    mistake_turn: int
    sweep: SweepResult
    truth: tuple[tuple[int, float], ...]  # 每個已估計的 k 的解析解 P(success | k)


def find_source(q: float = DEMO_Q, mistake_turn: int = DEMO_MISTAKE_TURN) -> tuple[int, Trajectory]:
    """在 DEMO_TASK_ID 上以 SyntheticPolicy（rollout_seed = 0, 1, 2, …）找第一條「第一個錯誤恰好在第 mistake_turn 個 turn、
    之後沒有其他錯誤、以 final_answer 結束」的 source（max_turns = BASELINE_MAX_TURNS、clock 恆為 0.0）。
    搜尋 SOURCE_SEED_SEARCH_LIMIT 個 seed 仍找不到 → RuntimeError（q == 0 不可能犯錯 → 直接 RuntimeError）。"""
    if q == 0:
        raise RuntimeError("a mistake rate of 0 never produces a mistake")
    instance = _instance()
    mistake = mistake_call(instance.task)
    mistake_key = canonical_json(list(mistake))
    for seed in range(SOURCE_SEED_SEARCH_LIMIT):
        source = run_episode(instance.task, _policy(instance, q, seed), max_turns=BASELINE_MAX_TURNS, clock=_zero)
        if source.termination != "final_answer":
            continue
        positions = [
            i
            for i, turn in enumerate(source.turns)
            if [canonical_json([c.name, c.args]) for c in turn.message.tool_calls] == [mistake_key]
        ]
        if positions == [mistake_turn]:
            return seed, source
    raise RuntimeError(f"no source with a single mistake at turn {mistake_turn} in {SOURCE_SEED_SEARCH_LIMIT} seeds")


ROLLOUT_SEED_BASE = 1_000_000  # rollout 的 seed 與 source 的搜尋範圍 [0, SOURCE_SEED_SEARCH_LIMIT) 不重疊


def _zero() -> float:
    return 0.0


def _instance() -> GeneratedInstance:
    return next(inst for inst in week1_pool("dev") if inst.task.task_id == DEMO_TASK_ID)


def _policy(instance: GeneratedInstance, q: float, seed: int) -> SyntheticPolicy:
    return SyntheticPolicy(instance.oracle_calls, mistake=mistake_call(instance.task), mistake_rate=q, rollout_seed=seed)


def run_demo(out_dir: str | Path) -> DemoResult:
    """find_source → ``replay.localization.sweep``（DEMO_PLAN、DEMO_CRITERION；rollout policy 為相同 config 的 SyntheticPolicy，
    rollout_seed 由 (k, rollout_index, attempt) 決定且與 source 的搜尋範圍不重疊；manifest 寫在暫存目錄、結束後刪除——
    示範不是實驗紀錄）→ 解析解。寫出 ``demo.md``（render_demo）與 ``curve.svg``（experiments.analyze_sweep.curve_svg）；
    out_dir 不存在時建立，其中已有 demo 檔案時覆寫。"""
    from experiments.analyze_sweep import SourceCurve, cliff_turn, curve_svg

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instance = _instance()
    seed, source = find_source()

    def factory(k: int, rollout_index: int, attempt: int) -> SyntheticPolicy:
        return _policy(instance, DEMO_Q, ROLLOUT_SEED_BASE + k * 10_000 + rollout_index * 10 + attempt)

    with tempfile.TemporaryDirectory() as scratch:
        log_path = Path(scratch) / "manifest.jsonl"
        result = sweep(source, factory, plan=DEMO_PLAN, log_path=log_path, clock=_zero, criterion=DEMO_CRITERION)
    mistake = mistake_call(instance.task)
    truth = tuple((e.k, checkpoint_truth(source, e.k, mistake, DEMO_Q)) for e in result.estimates)
    grade = source.grade()
    demo = DemoResult(
        task_id=instance.task.task_id,
        instruction=instance.task.instruction,
        source_seed=seed,
        source=source,
        source_grade=grade,
        mistake_turn=DEMO_MISTAKE_TURN,
        sweep=result,
        truth=truth,
    )
    curve = SourceCurve(
        task_id=instance.task.task_id,
        family=instance.task.family,
        source_trajectory_id=source.trajectory_id,
        source_success=grade.success,
        source_safe=grade.trajectory_safe,
        n_turns=len(source.turns),
        result=result,
        cliff=cliff_turn(source, result),
    )
    (out / "demo.md").write_text(render_demo(demo), encoding="utf-8")
    (out / "curve.svg").write_text(curve_svg(curve), encoding="utf-8")
    return demo


def render_demo(result: DemoResult) -> str:
    """README 用的 Markdown：依序說明 1. Task（instruction 與成功 / 禁止條件摘要）、2. Agent actions（source 每個 turn 的 tool call）、
    3. State changes（每個 call 的 changes 與 state hash 前後）、4. Grader（success、trajectory_safe、forbidden / unnecessary 等）、
    5. Replay from checkpoint（估計了哪些 k；每個已估計 k 的 n、p̂、Wilson CI 與解析解；largest drop 與 k*），並明確標示「synthetic policy、不是 LLM 結果」。
    決定性輸出（不含時間或路徑）。"""
    task = result.source.task
    grade = result.source_grade
    truth = dict(result.truth)
    ks = [e.k for e in result.sweep.estimates]
    lines = [
        "> **synthetic policy 的示範，不是 LLM 結果。** agent 是解析解已知的 synthetic policy（decision 010）：照 oracle 行動，"
        f"每個 turn 以機率 q = {DEMO_Q} 改為發出不可逆的錯誤（在受保護的 issue 上留言）。真正的 LLM 結果要等 paid runs。",
        "",
        f"### 1. Task — `{result.task_id}`（dev split，不計入任何統計）",
        "",
        f"> {result.instruction}",
        "",
        f"成功條件 {len(task['expected_outcome'])} 項、success constraints {len(task['success_constraints'])} 項、"
        f"forbidden rules {len(task['forbidden'])} 項（完整定義見 task JSON）。",
        "",
        f"### 2. Agent actions — source trajectory（synthetic seed {result.source_seed}，{len(result.source.turns)} 個 turn）",
        "",
        "| turn | tool call | ok |",
        "|---|---|---|",
    ]
    for turn in result.source.turns:
        if not turn.message.tool_calls:
            lines.append(f"| {turn.index} | （結束訊息：{turn.message.text}） | |")
        for call, step in zip(turn.message.tool_calls, turn.tool_steps):
            marker = " ← 不可逆的錯誤" if turn.index == result.mistake_turn else ""
            lines.append(f"| {turn.index} | `{call.name}({canonical_json(call.args)})`{marker} | {step.ok} |")
    lines += ["", "### 3. State changes（每個成功的變更；state hash 取前 12 碼）", "", "| step | change | state hash |", "|---|---|---|"]
    for turn in result.source.turns:
        for step in turn.tool_steps:
            before, after = step.state_hash_before.split(":")[-1][:12], step.state_hash_after.split(":")[-1][:12]
            if not step.changes:
                lines.append(f"| {step.step} | （無變更：`{step.tool}`） | {before} |")
            for change in step.changes:
                value = canonical_json(change.after)
                lines.append(f"| {step.step} | issue #{change.issue_id} `{change.field}` → `{value}` | {before} → {after} |")
    lines += [
        "",
        "### 4. Grader",
        "",
        f"- success：**{grade.success}**（outcome_correct {grade.outcome_correct}、constraint violations "
        f"{len(grade.constraint_violations)}）",
        f"- trajectory_safe：{grade.trajectory_safe}；forbidden_action_count {grade.forbidden_action_count}、"
        f"forbidden_attempt_count {grade.forbidden_attempt_count}、unnecessary mutations {len(grade.unnecessary_mutations)}、"
        f"tool errors {len(grade.tool_errors)}、steps {grade.steps}",
        "",
        "最後的狀態判定失敗，但只看最後狀態看不出**從哪一步開始**救不回來——這是 replay 要回答的問題。",
        "",
        "### 5. Replay from checkpoint — P(success | k)",
        "",
        f"從 source 的 checkpoint 還原 env 與 conversation、重新 rollout：先每 {DEMO_PLAN.coarse_stride} 個 turn 取一點，"
        f"再在大下降處加密（plan {DEMO_PLAN.coarse_stride} / {DEMO_PLAN.n_coarse} / {DEMO_PLAN.n_fine} / "
        f"{DEMO_PLAN.drop_threshold}，判定 {result.sweep.criterion}）。共估計 {len(ks)} / {len(result.source.turns)} 個 checkpoint"
        f"（k = {', '.join(str(k) for k in ks)}），其餘的 k 沒有 rerun：",
        "",
        "| k | n | p̂ | Wilson 95% CI | 解析解 |",
        "|---|---|---|---|---|",
    ]
    for e in result.sweep.estimates:
        lines.append(f"| {e.k} | {e.n} | {e.p_hat:.3f} | [{e.ci_low:.3f}, {e.ci_high:.3f}] | {truth[e.k]:.3f} |")
    verdict = "cliff" if result.sweep.is_cliff else "gradual"
    lines += [
        "",
        f"最大下降 {result.sweep.largest_drop:.3f} 位於 k = {result.sweep.largest_drop_k} → {result.sweep.largest_drop_k + 1}"
        f"（判定：**{verdict}**）；source 在第 {result.mistake_turn} 個 turn 犯下不可逆錯誤，cliff 正好定位在這一步。",
        "",
    ]
    return "\n".join(lines)


README_START = "<!-- demo:start -->"
README_END = "<!-- demo:end -->"


def embed_in_readme(readme: str, demo_markdown: str) -> str:
    """把 README 中 README_START 與 README_END 之間（各自獨占一行、剛好各一個、START 在前）的內容換成 demo_markdown；
    標記不符 → ValueError。結果中 demo_markdown 前後各一個換行。"""
    if readme.count(README_START) != 1 or readme.count(README_END) != 1:
        raise ValueError("README must contain exactly one demo start marker and one end marker")
    start = readme.index(README_START) + len(README_START)
    end = readme.index(README_END)
    if end < start:
        raise ValueError("the demo end marker must come after the start marker")
    body = demo_markdown if demo_markdown.endswith("\n") else demo_markdown + "\n"
    return readme[:start] + "\n" + body + readme[end:]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--update-readme", default=None, help="把示範寫入這個 README 的 demo 區塊")
    args = parser.parse_args(argv)
    result = run_demo(args.out)
    if args.update_readme:
        path = Path(args.update_readme)
        path.write_text(embed_in_readme(path.read_text(encoding="utf-8"), render_demo(result)), encoding="utf-8")
    print(f"{result.task_id}: cliff {result.sweep.is_cliff} at k={result.sweep.largest_drop_k}; wrote {args.out}/demo.md, curve.svg")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
