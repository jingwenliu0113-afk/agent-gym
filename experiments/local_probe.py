"""本機模型的開發題測試彙整與截斷自我測試（decision 026 §5：凍結設定的依據）。

用法::

    uv run python -m experiments.local_probe --summarize runs/<local-dev-run>
        彙整一次開發題測試（不呼叫模型、不需要 GPU）：結果、每次呼叫的耗時與 token、tool call 的有效性與錯誤種類。
    tools/with-gpu-lock -- uv run python -m experiments.local_probe --verify-server
        在 wrapper 啟動的 server 上檢查 Ollama 版本、模型 digest、量化與 capabilities（不載入模型；decision 029）。
    tools/with-gpu-lock -- uv run python -m experiments.local_probe --truncation-check
        在 GPU lock 內送出一個超出小 num_ctx 的 prompt（``truncate: false``），確認 server 拒絕而不是靜默截斷。

輸出為 JSON（stdout）。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import MAX_EPISODE_ATTEMPTS
from agents.ollama import OLLAMA_POLICY, OllamaContextError, check_local_config, make_ollama_client
from agents.policy import AgentConfig
from experiments.call_log import reconcile
from apps.issue_tracker.tools import TOOL_NAMES
from replay.aborts import read_manifest
from replay.localization import classify_rollout
from replay.trajectory import Trajectory, load_strict_json

TRUNCATION_NUM_CTX = 1024


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"n": len(values), "median": statistics.median(ordered), "p90": p90, "max": ordered[-1]}


def _manifest_outcomes(run_dir: Path) -> list[dict[str, Any]]:
    """沒有 summary 的中斷目錄：由 manifest 每個 rollout 的最後一次嘗試重建結果（同 analyze_baseline 的規則，decision 017 §4.1）。"""
    last: dict[tuple[str, int], Any] = {}
    for row in read_manifest(run_dir / "manifest.jsonl"):
        key = (row.task_id, row.rollout_index)
        if key not in last or row.attempt > last[key].attempt:
            last[key] = row
    results = []
    for (task_id, rollout_index), row in last.items():
        if row.outcome == "completed":
            trajectory = Trajectory.load(run_dir / "trajectories" / f"{task_id}-r{rollout_index}.json")
            outcome = classify_rollout(trajectory, 0)[0]
            trajectory_id: str | None = row.trajectory_id
        else:
            trajectory_id = None
            if row.error_class == "agent_output":
                outcome = "agent_error"
            elif row.error_class == "infrastructure" and row.attempt >= MAX_EPISODE_ATTEMPTS:
                outcome = "aborted"
            else:
                outcome = "interrupted"
        results.append({"task_id": task_id, "rollout_index": rollout_index, "outcome": outcome, "trajectory_id": trajectory_id})
    return results


def summarize_smoke(run_dir: str | Path) -> dict[str, Any]:
    """彙整 run_baseline 的本機 run 目錄（不呼叫模型）。agent 不是本機模型 → ValueError。decision 028 §3：
    - 每次呼叫的耗時 / token 與 tool call 統計只涵蓋已完成的 trajectory（``calls.coverage``），``calls.excluded`` = 其餘計費的呼叫數；
    - ``billed_calls`` = manifest 所有嘗試的計費呼叫數；``aborts`` = 每一種中止的 (error_class, exception_type, message) 次數；
    - ``dry_run`` 取自 run.json；沒有 summary 的中斷目錄由 manifest 重建結果（``interrupted`` = True）。
    - ``call_log``（decision 032）：``call_log.reconcile``——呼叫數、失敗數、有 seed 的呼叫數，以及與 manifest 逐次嘗試核對的結果；
      沒有紀錄 → None。"""
    run_dir = Path(run_dir)
    rows = read_manifest(run_dir / "manifest.jsonl")
    run_json = load_strict_json((run_dir / "run.json").read_bytes()) if (run_dir / "run.json").exists() else {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_strict_json(summary_path.read_bytes())
        if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
            raise ValueError(f"{run_dir} is not a run_baseline directory")
        results = summary["results"]
        split, config_id = summary.get("split"), summary.get("config_id")
    else:
        summary = None
        results = _manifest_outcomes(run_dir)
        split = run_json.get("split") if isinstance(run_json, dict) else None
        config_id = rows[0].config_id if rows else None
    outcomes: dict[str, int] = {}
    seconds: list[float] = []
    prompt_counts: list[float] = []
    eval_counts: list[float] = []
    tool_calls = unknown = invalid = multi = 0
    error_codes: dict[str, int] = {}
    calls = 0
    for result in results:
        outcomes[result["outcome"]] = outcomes.get(result["outcome"], 0) + 1
        if result.get("trajectory_id") is None:
            continue
        trajectory = Trajectory.load(run_dir / "trajectories" / f"{result['task_id']}-r{result['rollout_index']}.json")
        if trajectory.agent.policy != OLLAMA_POLICY:
            raise ValueError(f"{run_dir} was not produced by the local model agent")
        for turn in trajectory.turns:
            calls += 1
            meta = turn.meta if isinstance(turn.meta, dict) else {}
            durations = meta.get("durations") or {}
            if isinstance(durations.get("total"), int):
                seconds.append(durations["total"] / 1e9)
            usage = meta.get("usage") or {}
            if isinstance(usage.get("prompt_eval_count"), int):
                prompt_counts.append(usage["prompt_eval_count"])
            if isinstance(usage.get("eval_count"), int):
                eval_counts.append(usage["eval_count"])
            if len(turn.message.tool_calls) > 1:
                multi += 1
            for step in turn.tool_steps:
                tool_calls += 1
                if step.tool not in TOOL_NAMES:
                    unknown += 1
                if not step.ok and step.error is not None:
                    code = step.error["code"]
                    error_codes[code] = error_codes.get(code, 0) + 1
                    if code == "invalid_arguments":
                        invalid += 1
    billed = sum(row.billed_calls for row in rows)
    aborts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        if row.outcome == "aborted":
            key = (row.error_class or "", row.exception_type or "", row.message or "")
            aborts[key] = aborts.get(key, 0) + 1
    episodes = len(results)
    call_log = reconcile([run_dir])  # decision 032：逐次呼叫紀錄與 manifest 逐次嘗試核對；沒有紀錄 → None
    return {
        "run": run_dir.name,
        "split": split,
        "config_id": config_id,
        "dry_run": bool(run_json.get("dry_run")) if isinstance(run_json, dict) else False,
        "interrupted": summary is None,
        "episodes": episodes,
        "outcomes": dict(sorted(outcomes.items())),
        "success_rate": outcomes.get("success", 0) / episodes if episodes else None,
        "billed_calls": billed,
        "calls": {
            "coverage": "completed trajectories only",
            "n": calls,
            "excluded": billed - calls,
            "seconds": _stats(seconds),
            "prompt_eval_count": _stats(prompt_counts),
            "eval_count": _stats(eval_counts),
        },
        "tool_calls": {
            "n": tool_calls,
            "unknown_tool": unknown,
            "invalid_arguments": invalid,
            "multi_tool_turns": multi,
            "error_codes": dict(sorted(error_codes.items())),
        },
        "aborts": [
            {"error_class": error_class, "exception_type": exception_type, "message": message, "count": count}
            for (error_class, exception_type, message), count in sorted(aborts.items())
        ],
        "call_log": call_log,
    }


def truncation_check(client: Any, config: AgentConfig, *, num_ctx: int = TRUNCATION_NUM_CTX) -> dict[str, Any]:
    """送出一則遠超過 num_ctx 的 user 訊息（約 num_ctx × 6 bytes），``truncate`` / ``shift`` 關閉、num_predict 1、不帶 tools。
    server 回 context 錯誤 → ``{"result": "rejected"}``；回覆成功 → ``{"result": "accepted", "prompt_eval_count": ...}``
    （代表 server 可能截斷，凍結前必須處理）。其他錯誤原樣往外拋。"""
    config = check_local_config(config)
    model_config = config.model_config
    content = " ".join(f"word{index}" for index in range(num_ctx))  # 約 num_ctx × 6 bytes，token 數遠超過 num_ctx
    options = {**copy.deepcopy(model_config["options"]), "num_ctx": num_ctx, "num_predict": 1, "seed": 1}
    request = {
        "model": model_config["model"],
        "messages": [{"role": "system", "content": config.system_prompt}, {"role": "user", "content": content}],
        "stream": False,
        "think": model_config["think"],
        "options": options,
        "truncate": False,
        "shift": False,
    }
    base = {"num_ctx": num_ctx, "sent_bytes": len(json.dumps(request).encode("utf-8"))}
    try:
        reply = client.chat(request)
    except OllamaContextError as exc:
        return {**base, "result": "rejected", "detail": str(exc)}
    count = reply.get("prompt_eval_count") if isinstance(reply, dict) else None
    return {**base, "result": "accepted", "prompt_eval_count": count, "detail": "the server answered an oversized prompt"}


def main(argv: Sequence[str] | None = None) -> int:
    from experiments.run_baseline import GPU_LOCK_ENV, SERVER_ERRORS, connect_local_server, ollama_url_default
    from agents.ollama import local_agent_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--summarize", metavar="RUN_DIR")
    task.add_argument("--verify-server", action="store_true")
    task.add_argument("--truncation-check", action="store_true")
    parser.add_argument("--ollama-url", default=ollama_url_default())
    args = parser.parse_args(argv)
    if args.summarize:
        report = summarize_smoke(args.summarize)
    else:
        if os.environ.get(GPU_LOCK_ENV) != "held":
            parser.error(f"this check talks to the model server: run it inside tools/with-gpu-lock ({GPU_LOCK_ENV}=held)")
        config = local_agent_config()
        try:  # wrapper 確認的通道與 server（decision 031 §3）；錯誤一律是 CLI 錯誤，不是 traceback
            client, server, wrapper_info = connect_local_server(config, args.ollama_url, make_client=make_ollama_client)
            if args.verify_server:
                report = {"config_id": config.config_id, "server": server, "wrapper": wrapper_info}
            else:
                report = truncation_check(client, config)
        except SERVER_ERRORS as exc:
            parser.error(f"the local model server is not usable: {exc}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("local_probe: done", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
