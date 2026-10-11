"""本機模型的開發題測試彙整與截斷自我測試（decision 026 §5：凍結設定的依據）。

用法::

    uv run python -m experiments.local_probe --summarize runs/<local-dev-run>
        彙整一次開發題測試（不呼叫模型、不需要 GPU）：結果、每次呼叫的耗時與 token、tool call 的有效性與錯誤種類。
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

from agents.ollama import OLLAMA_POLICY, OllamaContextError, check_local_config, make_ollama_client
from agents.policy import AgentConfig
from apps.issue_tracker.tools import TOOL_NAMES
from replay.trajectory import Trajectory, load_strict_json

TRUNCATION_NUM_CTX = 1024


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"n": len(values), "median": statistics.median(ordered), "p90": p90, "max": ordered[-1]}


def summarize_smoke(run_dir: str | Path) -> dict[str, Any]:
    """彙整 run_baseline 的本機 run 目錄（summary.json 與 trajectories/；不呼叫模型）。agent 不是本機模型 → ValueError。"""
    run_dir = Path(run_dir)
    summary = load_strict_json((run_dir / "summary.json").read_bytes())
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
        raise ValueError(f"{run_dir} is not a run_baseline directory")
    outcomes: dict[str, int] = {}
    seconds: list[float] = []
    prompt_counts: list[float] = []
    eval_counts: list[float] = []
    tool_calls = unknown = invalid = multi = 0
    error_codes: dict[str, int] = {}
    calls = 0
    for result in summary["results"]:
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
    episodes = len(summary["results"])
    return {
        "run": run_dir.name,
        "split": summary.get("split"),
        "config_id": summary.get("config_id"),
        "episodes": episodes,
        "outcomes": dict(sorted(outcomes.items())),
        "success_rate": outcomes.get("success", 0) / episodes if episodes else None,
        "calls": {
            "n": calls,
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
    from experiments.run_baseline import GPU_LOCK_ENV, ollama_url_default
    from agents.ollama import local_agent_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--summarize", metavar="RUN_DIR")
    task.add_argument("--truncation-check", action="store_true")
    parser.add_argument("--ollama-url", default=ollama_url_default())
    args = parser.parse_args(argv)
    if args.summarize:
        report = summarize_smoke(args.summarize)
    else:
        if os.environ.get(GPU_LOCK_ENV) != "held":
            parser.error(f"the truncation check loads the model: run it inside tools/with-gpu-lock ({GPU_LOCK_ENV}=held)")
        config = local_agent_config()
        report = truncation_check(make_ollama_client(args.ollama_url, config.model_config), config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("local_probe: done", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
