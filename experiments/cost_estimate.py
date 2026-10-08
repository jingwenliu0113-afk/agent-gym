"""付費 run 之前的成本估算（decision 012）；不連網、不呼叫任何 LLM。

方法：以 OracleFakeClient 的 dry-run 取得每個 task「照 oracle 走」時 adapter 實際送出的 request，量測每次 call 的 request 大小，
再加上明列的假設：

- input tokens ≈ ``len(canonical_json({system, tools, messages})) / chars_per_token``；
- 每次 call 的 output tokens = ``output_tokens``（adaptive thinking + tool call；真實值未知，以低 / 高兩組假設給範圍）；
- 前面每個 assistant turn 的 thinking 保留在歷史中：第 t 次 call 的 input 另加 ``t × output_tokens``；
- 實際 turn 數 = oracle turn 數 × ``turn_factor``（超出 oracle 的 turn 以最後一次 call 的 input 大小外插）；
- **不計 prompt caching 折扣**（全部以 base input 價計），因此是偏高的估計。

sweep 的上限：每個 source 最多 ``n_fine × L`` 個 rollout（所有 checkpoint 都被 refine），從 checkpoint k 出發的 rollout
成本 ≈ source 第 k..L−1 次 call 的成本總和；取所有 k 的平均。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from agents.baseline import BASELINE_MODEL_CONFIG, PRICING
from env.hashing import canonical_json

CHARS_PER_TOKEN = 3.5


@dataclass(frozen=True)
class EpisodeEstimate:
    task_id: str
    calls: int  # 估計的 model call 數
    input_tokens: int
    output_tokens: int
    cost_usd: float
    call_costs: tuple[float, ...]  # 每次 call 的成本（依序）


def request_tokens(request: dict, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """``ceil(len(canonical_json({"system", "tools", "messages"})) / chars_per_token)``。"""
    prompt = {key: request[key] for key in ("system", "tools", "messages")}
    return math.ceil(len(canonical_json(prompt)) / chars_per_token)


def estimate_episode(
    task_id: str,
    requests: Sequence[dict],
    *,
    output_tokens: int,
    turn_factor: float = 1.0,
    chars_per_token: float = CHARS_PER_TOKEN,
    model: str | None = None,
) -> EpisodeEstimate:
    """requests = 一個 episode 依序送出的 request（至少一個）。calls = ``ceil(len(requests) × turn_factor)``（turn_factor >= 1）；
    第 t 次 call 的 input = ``request_tokens(requests[min(t, len − 1)]) + t × output_tokens``；
    成本 = ``(input × p.input + output_tokens × p.output) / 1e6``（model 預設 BASELINE_MODEL_CONFIG["model"]，不在 PRICING → ValueError）。
    output_tokens 為 int >= 0（bool 不算）、turn_factor 為有限的數 >= 1；否則 ValueError。
    """
    if len(requests) == 0:
        raise ValueError("requests must not be empty")
    if type(output_tokens) is not int or output_tokens < 0:
        raise ValueError("output_tokens must be an int >= 0")
    if isinstance(turn_factor, bool) or not isinstance(turn_factor, (int, float)) or not math.isfinite(turn_factor) or turn_factor < 1:
        raise ValueError("turn_factor must be a finite number >= 1")
    prices = PRICING.get(BASELINE_MODEL_CONFIG["model"] if model is None else model)
    if prices is None:
        raise ValueError(f"no pricing for model {model!r}")
    sizes = [request_tokens(request, chars_per_token) for request in requests]
    calls = math.ceil(len(requests) * turn_factor)
    inputs = [sizes[min(t, len(sizes) - 1)] + t * output_tokens for t in range(calls)]
    costs = tuple((size * prices["input"] + output_tokens * prices["output"]) / 1e6 for size in inputs)
    return EpisodeEstimate(
        task_id=task_id,
        calls=calls,
        input_tokens=sum(inputs),
        output_tokens=calls * output_tokens,
        cost_usd=math.fsum(costs),
        call_costs=costs,
    )


def sweep_upper_bound(episode: EpisodeEstimate, n_fine: int) -> float:
    """``n_fine × L × mean_k(sum(call_costs[k:]))``，L = episode.calls；n_fine 為 int >= 1，否則 ValueError。"""
    if type(n_fine) is not int or n_fine < 1:
        raise ValueError("n_fine must be an int >= 1")
    costs = episode.call_costs
    return n_fine * math.fsum(math.fsum(costs[k:]) for k in range(len(costs)))


def main() -> None:  # pragma: no cover - CLI
    """以 dry-run 量測 request 大小，印出 dev / eval baseline 與 eval sweep 上限的估算（低 / 高兩組假設）。"""
    from agents.baseline import BASELINE_MAX_TURNS, AnthropicPolicy
    from experiments.run_baseline import OracleFakeClient
    from replay.runner import run_episode
    from tasks.generator import week1_pool

    scenarios = {"low": {"output_tokens": 500, "turn_factor": 1.0}, "high": {"output_tokens": 4000, "turn_factor": 2.0}}
    for split in ("dev", "eval"):
        by_task: dict[str, list[dict]] = {}
        for instance in week1_pool(split):
            client = OracleFakeClient([instance])  # 每個 task 各自一個 client：requests 不會混在一起
            run_episode(instance.task, AnthropicPolicy(client), max_turns=BASELINE_MAX_TURNS, clock=lambda: 0.0)
            by_task[instance.task.task_id] = client.requests
        for name, assumption in scenarios.items():
            episodes = [estimate_episode(task_id, reqs, **assumption) for task_id, reqs in by_task.items()]
            total = math.fsum(e.cost_usd for e in episodes)
            line = f"{split} {name}: {len(episodes)} episodes, ${total:.2f} per rollout of the split"
            if split == "eval":
                for n_fine in (24, 32):
                    bound = math.fsum(sweep_upper_bound(e, n_fine) for e in episodes)
                    line += f"; sweep upper bound (n_fine={n_fine}) ${bound:.0f}"
            print(line)


if __name__ == "__main__":  # pragma: no cover
    main()
