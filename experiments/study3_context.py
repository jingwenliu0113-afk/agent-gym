"""Study 3 的 context 檢查（decision 036 §3「context 與 agent_error」，A 類檢查）。不連網、不需要 GPU。

本機 agent 的 adapter 在 request bytes ÷ 2 + num_predict > num_ctx 時拒絕送出（``agents.ollama._fits``），所以 request 的上限是
``LIMIT_BYTES`` = 2 ×（num_ctx − num_predict）= 57,344 bytes。request 以凍結的本機 agent 組出（oracle 假 client 重放 tool call；
num_ctx 暫時放大，只為了量出超過上限的 request），assistant 的文字以 Study 2 dev 實測每個 turn 的 P95（``ASSISTANT_RESERVE_BYTES``）
補上：第 k 次 request 含前面 k 則 assistant 訊息，每則加上這個預留。

1. oracle 與每個 alternative 的完整對話，最大的 request ≤ ``BUDGET_BYTES``（0.8 × 上限）。
2. 壓力情境：oracle 前面加上 ``STRESS_SEARCHES`` 次 limit 50 的寬搜尋，最大的 request 加上 ``STRESS_TURNS`` 個 turn 的預留 ≤ 上限。

超過 turn 的正確解（long alternatives）本來就無法在 max_turns 內完成，不列入檢查。不通過的級別在 golden hash 之前修改，修改寫進
decision 036。

``request_bytes``：dev 與 eval 報告的「最大的 request 大小」（§3「context 與 agent_error」的報告）由 trajectory 重建每個 turn 送出的
request（本機：Ollama ``/api/chat`` 的形狀與 turn.meta 記錄的 seed；Haiku：Messages API 的形狀），以 canonical JSON 的 bytes 計。
被 fit check 拒絕送出的 request 沒有 trajectory，不在其中（記為 agent_error，計入 context 用盡）。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import ANTHROPIC_POLICY, build_request
from agents.ollama import OLLAMA_POLICY, OllamaPolicy, build_chat_request, local_agent_config
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from experiments.run_baseline import OllamaOracleFakeClient
from replay.runner import run_episode
from replay.trajectory import Trajectory, render_messages
from tasks.generator_v3 import MAX_TURNS, GeneratedInstanceV3

_OPTIONS = local_agent_config().model_config["options"]
LIMIT_BYTES = 2 * (_OPTIONS["num_ctx"] - _OPTIONS["num_predict"])  # 57,344
BUDGET_BYTES = int(0.8 * LIMIT_BYTES)  # 45,875
# runs/study2-dev-20261017/calls.jsonl 中成功回覆的 assistant content（JSON 字串跳脫後、不含引號）的 P95（nearest rank）；
# 660 則：P50 134、P90 622、P95 886、P99 1,660 bytes。測試在有原始紀錄的機器上重算。
ASSISTANT_RESERVE_BYTES = 886
STRESS_SEARCHES = 2
STRESS_TURNS = 20

Call = tuple[str, dict[str, Any]]


def assistant_reserve(calls_path: str | Path, quantile: float = 0.95) -> int:
    """calls.jsonl 中成功回覆的 assistant content 大小（bytes）的分位數（nearest rank）。"""
    sizes = []
    for line in Path(calls_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("ok"):
            content = row["response"]["message"].get("content", "")
            sizes.append(len(canonical_json(content).encode("utf-8")) - 2)
    if not sizes:
        raise ValueError(f"{calls_path} has no successful calls")
    sizes.sort()
    return sizes[min(len(sizes) - 1, math.ceil(quantile * len(sizes)) - 1)]


def _measuring_config() -> Any:
    config = copy.deepcopy(local_agent_config())
    config.model_config["options"]["num_ctx"] = 10**8  # 只為了量出超過上限的 request；不送到任何 server
    return config


def request_sizes(instance: GeneratedInstanceV3, calls: Sequence[Call]) -> list[int]:
    """以凍結的本機 agent 重放 calls（oracle 假 client），回傳每次 request 的 bytes（canonical JSON）。"""
    replayed = dataclasses.replace(instance, oracle_calls=tuple(calls))
    client = OllamaOracleFakeClient([replayed])
    policy = OllamaPolicy(client, _measuring_config(), seed_source=lambda: 1)
    run_episode(replayed.task, policy, max_turns=max(MAX_TURNS, len(calls) + 1), clock=lambda: 0.0)
    return [len(canonical_json(request).encode("utf-8")) for request in client.requests]


def request_bytes(trajectory: Trajectory) -> list[int]:
    """trajectory 每個 turn 送出的 request 的 bytes（canonical JSON），依 agent 的 adapter 重建（模組 docstring）。"""
    agent = trajectory.agent
    tools = IssueTrackerEnv.tool_specs()
    out = []
    for i, turn in enumerate(trajectory.turns):
        messages = render_messages(trajectory.task["instruction"], agent, trajectory.turns[:i])
        if agent.policy == OLLAMA_POLICY:
            request = build_chat_request(messages, tools, agent.model_config, seed=turn.meta["seed"])
        elif agent.policy == ANTHROPIC_POLICY:
            request = build_request(messages, tools, agent.model_config)
        else:
            raise ValueError(f"no request shape for the policy {agent.policy!r}")
        out.append(len(canonical_json(request).encode("utf-8")))
    return out


def _worst(sizes: Sequence[int], reserve: int) -> int:
    return max(size + k * reserve for k, size in enumerate(sizes))


def check_context(instance: GeneratedInstanceV3) -> dict[str, Any]:
    """{"solutions": {名稱: 最大的 request（含預留）}, "worst", "stress", "ok"}；ok = 兩項檢查都通過。"""
    reserve = ASSISTANT_RESERVE_BYTES
    solutions = {"oracle": _worst(request_sizes(instance, instance.oracle_calls), reserve)}
    for name, calls in instance.alternative_calls:
        solutions[name] = _worst(request_sizes(instance, calls), reserve)
    broad: list[Call] = [("search_issues", {"limit": 50})] * STRESS_SEARCHES
    stress = max(request_sizes(instance, [*broad, *instance.oracle_calls])) + STRESS_TURNS * reserve
    worst = max(solutions.values())
    return {"solutions": solutions, "worst": worst, "stress": stress, "ok": worst <= BUDGET_BYTES and stress <= LIMIT_BYTES}
