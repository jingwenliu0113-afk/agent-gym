"""Week 1 #5：baseline agent 在 week1 dev / eval split 上的 run（decision 009 / 012）。

用法::

    uv run python -m experiments.run_baseline --split dev --dry-run
        不連網、不需要任何 credential：以 oracle 劇本的 OracleFakeClient 走完整條流程
        （AnthropicPolicy → runner → manifest → grader → summary），只用來驗證 harness。
    uv run python -m experiments.cost_estimate
        印出付費 run 的成本估算（不連網；另一個入口）。
    uv run --extra llm python -m experiments.run_baseline --split dev --max-cost-usd 5 --confirm-paid-run
        真正呼叫 Anthropic API。只能在使用者確認模型與成本之後執行；credential 由 SDK 自行從環境變數解析，
        本腳本不讀取、不檢查、不記錄任何 key。

輸出目錄（gitignored）：``runs/<run_name>/`` 下 ``run.json``（開始時寫入：split、rollouts、config_id、dry_run；decision 021 §1）、
``manifest.jsonl``、``trajectories/<task_id>-r<i>.json``、``summary.json``。--dry-run 的輸出標記為 dry_run，永遠不能與付費輸出合併。
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import PRICING, BASELINE_MAX_TURNS, AnthropicPolicy, baseline_agent_config, make_anthropic_client
from agents.policy import AgentConfig, PolicyStep
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from replay.aborts import RolloutKey, billed_cost, run_with_retries
from replay.atomic import write_atomic
from replay.localization import classify_rollout
from replay.runner import Clock, run_episode
from tasks.generator import GeneratedInstance, week1_pool

SUMMARY_FORMAT = "agent-gym/baseline-run/v2"  # v2：run.json（decision 021 §1）；run.json 與 summary.json 共用
RUN_FILE = "run.json"


@dataclass
class Ledger:
    """一次 run 的成本帳：所有**成功回傳**的 model call（含之後中止的嘗試）。"""

    cost_usd: float = 0.0
    calls: int = 0
    unknown_cost_calls: int = 0  # PolicyStep.cost_usd 為 None 的 call 數


class CostMeter:
    """包住 policy：每次已計費的 call 記到 ledger（decision 014）：成功回傳的 next_message 加 cost_usd；帶 ``billed_usage`` 的
    例外（API 已回傳但被 adapter 拒絕）加其 cost_usd 後原樣往外拋；cost 為 None → unknown_cost_calls += 1。
    其他例外（沒有 response）不記帳、原樣往外拋。agent_config 原樣委派。
    """

    def __init__(self, inner: Any, ledger: Ledger) -> None:
        self._inner = inner
        self._ledger = ledger

    @property
    def agent_config(self) -> AgentConfig:
        return self._inner.agent_config

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        try:
            step = self._inner.next_message(messages, tools)
        except Exception as exc:
            billed, cost = billed_cost(exc)
            if billed:
                self._record(cost)
            raise
        self._record(step.cost_usd)
        return step

    def _record(self, cost: float | None) -> None:
        self._ledger.calls += 1
        if cost is None:
            self._ledger.unknown_cost_calls += 1
        else:
            self._ledger.cost_usd += cost


def budget_reached(ledger: Ledger, max_cost_usd: float | None) -> bool:
    """有上限時：已花費 >= 上限，或出現成本無法量測的已計費 call（保守停止，decision 014）。"""
    return max_cost_usd is not None and (ledger.cost_usd >= max_cost_usd or ledger.unknown_cost_calls > 0)


def check_budget_args(max_cost_usd: Any, config: AgentConfig) -> None:
    """max_cost_usd 為 None 或有限的數 > 0；有上限時 config 的模型必須在 PRICING（否則成本上限無法生效）。違反 → ValueError。"""
    if max_cost_usd is None:
        return
    if isinstance(max_cost_usd, bool) or not isinstance(max_cost_usd, (int, float)) or not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be None or a finite number > 0")
    if config.model_config.get("model") not in PRICING:
        raise ValueError("a cost cap requires a model listed in PRICING (otherwise its cost cannot be measured)")


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    family: str
    rollout_index: int
    attempts: int
    outcome: str  # replay.localization.ROLLOUT_OUTCOMES
    termination: str | None  # 完成時
    trajectory_id: str | None
    stopped_without_action: bool
    forbidden_action_count: int | None  # 完成時（grade()）
    forbidden_attempt_count: int | None
    trajectory_safe: bool | None
    cost_usd: float | None  # 完成時：trajectory.usage().cost_usd
    model_turns: int | None


@dataclass(frozen=True)
class RunSummary:
    split: str
    config_id: str
    n_tasks: int
    rollouts: int
    results: tuple[EpisodeResult, ...]  # 依執行順序
    spent_usd: float  # ledger.cost_usd
    model_calls: int  # ledger.calls
    unknown_cost_calls: int
    stopped_by_budget: bool

    def to_dict(self) -> dict[str, Any]:
        """``{"format": SUMMARY_FORMAT, ...欄位}``；results 為 list of dict（欄位名 → 值）。JSON-native。"""
        data = asdict(self)
        data["results"] = [asdict(result) for result in self.results]
        return {"format": SUMMARY_FORMAT, **data}


def run_split(
    split: str,
    client: Any,
    *,
    run_dir: str | Path,
    rollouts: int = 1,
    max_cost_usd: float | None = None,
    clock: Clock = time.perf_counter,
    agent_config: AgentConfig | None = None,
    dry_run: bool = False,
) -> RunSummary:
    """在 ``week1_pool(split)``（"dev" / "eval"）上執行 baseline。

    - 事前檢查（ValueError，不執行）：split ∈ {"dev", "eval"}；rollouts 為 int >= 1；``check_budget_args``（有上限時模型必須在
      PRICING）；run_dir 不存在或為空目錄（不覆寫既有 run，避免重複計費與混淆）；dry_run 為 bool。
    - 第一次嘗試之前寫入 ``run_dir/run.json``：format、split、rollouts、config_id、dry_run（decision 021 §1；CLI 的 --dry-run 為
      True，直接以 Python 呼叫時預設 False）。
    - 排程：rollout_index 外層、task 內層（同一輪所有 task 跑完才進下一輪；降低時間效應）。
    - 每個 episode 開始前：``budget_reached(ledger, max_cost_usd)`` → 停止排程，stopped_by_budget = True。
    - 每個 episode：``run_with_retries(lambda p: run_episode(task, p, max_turns=BASELINE_MAX_TURNS, clock=clock),
      lambda attempt: CostMeter(AnthropicPolicy(client, agent_config), ledger), key=RolloutKey(None, None, None, i, 1),
      log_path=run_dir / "manifest.jsonl")``。
    - 完成 → trajectory 存成 ``run_dir/trajectories/{task_id}-r{i}.json``；outcome 與 stopped_without_action 由
      ``classify_rollout(trajectory, 0)``；安全指標取 ``trajectory.grade()``。未完成 → infrastructure → "aborted"，否則 "agent_error"。
    - harness 例外原樣往外拋（不寫 summary）。正常結束寫 ``run_dir/summary.json``（canonical_json + "\\n"）並回傳。
    """
    if split not in ("dev", "eval"):
        raise ValueError("split must be 'dev' or 'eval'")
    if type(rollouts) is not int or rollouts < 1:
        raise ValueError("rollouts must be an int >= 1")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a bool")
    config = baseline_agent_config() if agent_config is None else copy.deepcopy(agent_config)
    check_budget_args(max_cost_usd, config)
    run_dir = Path(run_dir)
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise ValueError(f"{run_dir} already exists and is not an empty directory")
    AnthropicPolicy(client, config)  # 設定與 client 的檢查在任何呼叫之前完成

    instances = week1_pool(split)
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    description = {"format": SUMMARY_FORMAT, "split": split, "rollouts": rollouts, "config_id": config.config_id, "dry_run": dry_run}
    write_atomic(run_dir / RUN_FILE, (canonical_json(description) + "\n").encode("utf-8"))
    log_path = run_dir / "manifest.jsonl"
    ledger = Ledger()
    results: list[EpisodeResult] = []
    stopped = False
    for rollout_index in range(rollouts):
        for instance in instances:
            if budget_reached(ledger, max_cost_usd):
                stopped = True
                break
            results.append(_run_one(instance, rollout_index, client, config, ledger, trajectories_dir, log_path, clock))
        if stopped:
            break

    summary = RunSummary(
        split=split,
        config_id=config.config_id,
        n_tasks=len(instances),
        rollouts=rollouts,
        results=tuple(results),
        spent_usd=ledger.cost_usd,
        model_calls=ledger.calls,
        unknown_cost_calls=ledger.unknown_cost_calls,
        stopped_by_budget=stopped,
    )
    write_atomic(run_dir / "summary.json", (canonical_json(summary.to_dict()) + "\n").encode("utf-8"))  # decision 024 §3
    return summary


def _run_one(
    instance: GeneratedInstance,
    rollout_index: int,
    client: Any,
    config: AgentConfig,
    ledger: Ledger,
    trajectories_dir: Path,
    log_path: Path,
    clock: Clock,
) -> EpisodeResult:
    task = instance.task
    retried = run_with_retries(
        lambda policy: run_episode(task, policy, max_turns=BASELINE_MAX_TURNS, clock=clock),
        lambda attempt: CostMeter(AnthropicPolicy(client, config), ledger),
        task=task,
        key=RolloutKey(None, None, None, rollout_index, 1),
        max_turns=BASELINE_MAX_TURNS,
        log_path=log_path,
        save_path=trajectories_dir / f"{task.task_id}-r{rollout_index}.json",  # 先存檔再記帳（decision 023 §2）
    )
    trajectory = retried.trajectory
    if trajectory is None:
        outcome = "aborted" if retried.error_class == "infrastructure" else "agent_error"
        return EpisodeResult(
            task.task_id, task.family, rollout_index, retried.attempts, outcome, None, None, False, None, None, None, None, None
        )
    outcome, stopped_without_action = classify_rollout(trajectory, 0)
    report = trajectory.grade()
    usage = trajectory.usage()
    return EpisodeResult(
        task_id=task.task_id,
        family=task.family,
        rollout_index=rollout_index,
        attempts=retried.attempts,
        outcome=outcome,
        termination=trajectory.termination,
        trajectory_id=trajectory.trajectory_id,
        stopped_without_action=stopped_without_action,
        forbidden_action_count=report.forbidden_action_count,
        forbidden_attempt_count=report.forbidden_attempt_count,
        trajectory_safe=report.trajectory_safe,
        cost_usd=usage.cost_usd,
        model_turns=usage.model_turns,
    )


@dataclass(frozen=True)
class _Block:
    type: str
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any = None


@dataclass(frozen=True)
class _CacheCreation:
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


@dataclass(frozen=True)
class _Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation: _CacheCreation = field(default_factory=_CacheCreation)


@dataclass(frozen=True)
class _Response:
    content: list[_Block]
    stop_reason: str
    usage: _Usage
    model: str
    stop_details: Any = None
    input_transformations: Any = None
    _request_id: str | None = None


class OracleFakeClient:
    """dry-run 用、不連網的 client（``client.beta.messages.create(**request)``）。

    依 request 的第一則 user 訊息（instruction 字串）找出候選 task；instruction 相同的多個候選（例如 escalate-label 的不同 seed）
    以重放判別：只保留「照自己的 oracle 走、env 回傳的 tool results 與 request 中的逐 byte 相同」者，且它們的下一步必須一致
    （否則 ValueError）。t = request 中 assistant 訊息數：
    t < len(oracle) → thinking block（signature ``f"dry-run-{t}"``）+ tool_use（id ``f"toolu_dry_{t}"``、oracle[t]），stop "tool_use"；
    否則 thinking + text "Done."，stop "end_turn"。usage 固定 input 1000 / output 200、model = request["model"]、
    _request_id ``f"req_dry_{t}"``。instruction 找不到 → ValueError。``requests`` 保存每次 request 的深拷貝。
    """

    def __init__(self, instances: Sequence[GeneratedInstance]) -> None:
        self._instances = tuple(instances)
        self._requests: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self._requests

    def create(self, **request: Any) -> _Response:
        self._requests.append(copy.deepcopy(request))
        messages = request["messages"]
        t = sum(1 for message in messages if message["role"] == "assistant")
        oracle = self._oracle_for(messages, t)
        thinking = _Block(type="thinking", thinking="", signature=f"dry-run-{t}")
        if t < len(oracle):
            name, args = oracle[t]
            content = [thinking, _Block(type="tool_use", id=f"toolu_dry_{t}", name=name, input=copy.deepcopy(args))]
            stop_reason = "tool_use"
        else:
            content = [thinking, _Block(type="text", text="Done.")]
            stop_reason = "end_turn"
        return _Response(
            content=content,
            stop_reason=stop_reason,
            usage=_Usage(input_tokens=1000, output_tokens=200),
            model=request["model"],
            _request_id=f"req_dry_{t}",
        )


    def _oracle_for(self, messages: list[dict[str, Any]], t: int) -> tuple[tuple[str, Any], ...]:
        candidates = [inst for inst in self._instances if inst.task.instruction == messages[0]["content"]]
        if not candidates:
            raise ValueError("OracleFakeClient: unknown task instruction")
        results = [
            block["content"]
            for message in messages[1:]
            if message["role"] == "user" and isinstance(message["content"], list)
            for block in message["content"]
        ]
        consistent = []
        for instance in candidates:
            oracle = tuple(instance.oracle_calls)
            if len(results) > len(oracle):
                continue
            env = IssueTrackerEnv()
            env.reset(instance.task.initial_state, instance.task.actor_id)
            replayed = [canonical_json(env.call_tool(name, copy.deepcopy(args))) for name, args in oracle[: len(results)]]
            if replayed == results:
                consistent.append(oracle)
        upcoming = {canonical_json(list(oracle[t])) if t < len(oracle) else "end" for oracle in consistent}
        if len(upcoming) != 1:
            raise ValueError("OracleFakeClient: the conversation does not identify a unique next oracle step")
        return consistent[0]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--run-name", default=None, help="runs/<run-name>；預設依 split 與模式命名")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="不連網：OracleFakeClient")
    mode.add_argument("--confirm-paid-run", action="store_true", help="真正呼叫 Anthropic API（會產生費用）")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="付費 run 必填的成本上限")
    args = parser.parse_args(argv)

    if args.dry_run:
        client: Any = OracleFakeClient(week1_pool(args.split))
        name = args.run_name or f"dry-run-{args.split}"
        budget = args.max_cost_usd
    else:
        if args.max_cost_usd is None:
            parser.error("--confirm-paid-run requires --max-cost-usd")
        client = make_anthropic_client()
        name = args.run_name or f"baseline-{args.split}"
        budget = args.max_cost_usd
    summary = run_split(
        args.split, client, run_dir=Path("runs") / name, rollouts=args.rollouts, max_cost_usd=budget, dry_run=args.dry_run
    )
    for result in summary.results:
        print(
            f"{result.task_id} r{result.rollout_index}: {result.outcome} (attempts {result.attempts}, "
            f"safe {result.trajectory_safe}, forbidden {result.forbidden_action_count}/{result.forbidden_attempt_count}, "
            f"turns {result.model_turns}, cost {result.cost_usd})"
        )
    print(
        f"spent ${summary.spent_usd:.4f} over {summary.model_calls} model calls "
        f"(unknown cost: {summary.unknown_cost_calls}); stopped by budget: {summary.stopped_by_budget}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
