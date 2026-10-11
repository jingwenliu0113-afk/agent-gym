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
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import (
    ANTHROPIC_POLICY,
    BASELINE_MAX_TURNS,
    PRICING,
    AnthropicPolicy,
    baseline_agent_config,
    make_anthropic_client,
)
from agents.ollama import (
    OLLAMA_POLICY,
    OllamaHTTPClient,
    OllamaPolicy,
    is_frozen,
    local_agent_config,
    make_ollama_client,
    verify_server,
)
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
REGISTERED_EVAL_ROLLOUTS = 3  # decision 025 §2：付費 eval baseline 的 R；dev sanity 與 dry-run 不受限制


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


GPU_LOCK_ENV = "AI_PORTFOLIO_GPU_LOCK"  # 只由 GPU lock wrapper 設為 "held"（decision 026 §3、MACHINE_POLICY）
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_URL_ENV = "AGENT_GYM_OLLAMA_URL"  # tools/with-gpu-lock 設為 SSH 通道的 loopback URL
FAKE_OLLAMA_VERSION = "0.0.0-dry-run"


def ollama_url_default() -> str:
    """CLI 的 --ollama-url 預設：GPU lock wrapper 設定的通道 URL（OLLAMA_URL_ENV），否則 DEFAULT_OLLAMA_URL。"""
    return os.environ.get(OLLAMA_URL_ENV) or DEFAULT_OLLAMA_URL
FAKE_OLLAMA_DIGEST = "sha256:" + "0" * 64


def make_policy(client: Any, config: AgentConfig) -> Any:
    """依 ``config.policy`` 選擇 adapter（decision 026 §1）：OLLAMA_POLICY → OllamaPolicy；ANTHROPIC_POLICY → AnthropicPolicy
    （Optional）；其他 → ValueError。"""
    if config.policy == OLLAMA_POLICY:
        return OllamaPolicy(client, config)
    if config.policy == ANTHROPIC_POLICY:
        return AnthropicPolicy(client, config)
    raise ValueError(f"no adapter for the agent policy {config.policy!r}")


def budget_reached(ledger: Ledger, max_cost_usd: float | None, max_model_calls: int | None = None) -> bool:
    """成本上限：已花費 >= 上限，或出現成本無法量測的已計費 call（保守停止，decision 014）。呼叫次數上限（本機模型，
    decision 026 §2）：已記帳的 call 數 >= 上限。"""
    if max_model_calls is not None and ledger.calls >= max_model_calls:
        return True
    return max_cost_usd is not None and (ledger.cost_usd >= max_cost_usd or ledger.unknown_cost_calls > 0)


def check_budget_args(max_cost_usd: Any, config: AgentConfig, max_model_calls: Any = None) -> None:
    """max_cost_usd 為 None 或有限的數 > 0，有上限時 config 的模型必須在 PRICING（否則成本上限無法生效；本機模型不在其中）；
    max_model_calls 為 None 或 int >= 1。違反 → ValueError。"""
    if max_model_calls is not None and (type(max_model_calls) is not int or max_model_calls < 1):
        raise ValueError("max_model_calls must be None or an int >= 1")
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
    max_model_calls: int | None = None,
) -> RunSummary:
    """在 ``week1_pool(split)``（"dev" / "eval"）上執行 baseline。

    - 事前檢查（ValueError，不執行）：split ∈ {"dev", "eval"}；rollouts 為 int >= 1；``check_budget_args``（有上限時模型必須在
      PRICING）；run_dir 不存在或為空目錄（不覆寫既有 run，避免重複計費與混淆）；dry_run 為 bool。
    - 第一次嘗試之前寫入 ``run_dir/run.json``：format、split、rollouts、config_id、dry_run（decision 021 §1；CLI 的 --dry-run 為
      True，直接以 Python 呼叫時預設 False）。
    - 排程：rollout_index 外層、task 內層（同一輪所有 task 跑完才進下一輪；降低時間效應）。
    - 每個 episode 開始前：``budget_reached(ledger, max_cost_usd, max_model_calls)`` → 停止排程，stopped_by_budget = True。
    - policy 依 ``make_policy(client, agent_config)``（decision 026）；agent_config 預設為 decision 009 的 Anthropic baseline
      （CLI 預設改用本機模型，見 ``main``）。
    - 每個 episode：``run_with_retries(lambda p: run_episode(task, p, max_turns=BASELINE_MAX_TURNS, clock=clock),
      lambda attempt: CostMeter(make_policy(client, agent_config), ledger), key=RolloutKey(None, None, None, i, 1),
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
    check_budget_args(max_cost_usd, config, max_model_calls)
    run_dir = Path(run_dir)
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise ValueError(f"{run_dir} already exists and is not an empty directory")
    make_policy(client, config)  # 設定與 client 的檢查在任何呼叫之前完成

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
            if budget_reached(ledger, max_cost_usd, max_model_calls):
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
        lambda attempt: CostMeter(make_policy(client, config), ledger),
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


class OllamaOracleFakeClient:
    """dry-run 與測試用、不連網的 Ollama client（``chat(request)``、``version()``、``tags()``、``show(model)``；decision 026）。

    與 OracleFakeClient 相同的 oracle 判別（以 instruction 找候選 task、以重放 tool results 排除不一致者），但用 Ollama 的形狀：
    request 的第 2 則訊息（第 1 則是 system）是 instruction；t = assistant 訊息數；tool results = tool 訊息的 content。
    t < len(oracle) → ``{"role": "assistant", "content": "", "tool_calls": [{"function": {"name", "arguments"}}]}``；
    否則 content "Done."。回覆的計數固定 prompt_eval_count 1000、eval_count 200；model = request["model"]。
    version() 回 FAKE_OLLAMA_VERSION；tags() 列出一個 model（預設 local 設定的 tag、digest FAKE_OLLAMA_DIGEST）；
    show() 回 capabilities ["completion", "tools"]、quantization Q4_K_M。``requests`` 保存每次 chat request 的深拷貝。
    """

    def __init__(self, instances: Sequence[GeneratedInstance], *, model: str | None = None) -> None:
        self._oracle = OracleFakeClient(instances)
        self._model = local_agent_config().model_config["model"] if model is None else model
        self._requests: list[dict[str, Any]] = []

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self._requests

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self._requests.append(copy.deepcopy(request))
        messages = request["messages"]
        t = sum(1 for message in messages if message["role"] == "assistant")
        anthropic_shaped = [{"role": "user", "content": messages[1]["content"]}]
        anthropic_shaped += [
            {"role": "user", "content": [{"content": message["content"]}]} for message in messages[2:] if message["role"] == "tool"
        ]
        oracle = self._oracle._oracle_for(anthropic_shaped, t)
        if t < len(oracle):
            name, args = oracle[t]
            message: dict[str, Any] = {
                "role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": copy.deepcopy(args)}}],
            }
        else:
            message = {"role": "assistant", "content": "Done."}
        return {
            "model": request["model"],
            "created_at": "1970-01-01T00:00:00Z",
            "message": message,
            "done": True,
            "done_reason": "stop",
            "total_duration": 1,
            "load_duration": 0,
            "prompt_eval_count": 1000,
            "prompt_eval_duration": 1,
            "eval_count": 200,
            "eval_duration": 1,
        }

    def version(self) -> dict[str, Any]:
        return {"version": FAKE_OLLAMA_VERSION}

    def tags(self) -> dict[str, Any]:
        return {"models": [{"name": self._model, "model": self._model, "digest": FAKE_OLLAMA_DIGEST.removeprefix("sha256:"),
                            "details": {"quantization_level": "Q4_K_M"}}]}

    def show(self, model: str) -> dict[str, Any]:
        return {"capabilities": ["completion", "tools"], "details": {"quantization_level": "Q4_K_M"}}


def start_local_run(parser: argparse.ArgumentParser, config: AgentConfig, *, url: str, formal: bool,
                    max_model_calls: int | None) -> Any:
    """本機模型 run 的事前檢查（decision 026 §3），全部在任何模型呼叫之前；違反 → ``parser.error``：
    GPU lock 環境變數必須是 "held"；必須給 max_model_calls；正式 run 必須使用凍結的設定；client 必須只連 loopback；
    ``verify_server`` 通過（模型存在、不是 cloud、支援 tools；凍結時版本與 digest 相符）。回傳 client，並把 server 資訊印到 stderr。"""
    if os.environ.get(GPU_LOCK_ENV) != "held":
        parser.error(f"local model runs must run inside the GPU lock wrapper ({GPU_LOCK_ENV}=held; MACHINE_POLICY)")
    if max_model_calls is None:
        parser.error("local model runs require --max-model-calls (decision 026 §2)")
    if formal and not is_frozen(config):
        parser.error("formal local runs need the frozen model config (model_digest and ollama_version; decision 026 §5)")
    try:
        OllamaHTTPClient(url, timeout_s=1)  # 先驗證 URL（只接受 loopback），不連線
        client = make_ollama_client(url, config.model_config)
        info = verify_server(client, config)
    except ValueError as exc:
        parser.error(f"the local model server is not usable: {exc}")
    print(f"local model server: {canonical_json(info)}", file=sys.stderr)
    return client


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--run-name", default=None, help="runs/<run-name>；預設依 split 與模式命名")
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama",
                        help="ollama（Core，decision 026）或 anthropic（Optional）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="不連網：該 provider 形狀的 oracle 假 client")
    mode.add_argument("--confirm-local-run", action="store_true", help="本機模型（只限 ollama；需要 GPU lock）")
    mode.add_argument("--confirm-paid-run", action="store_true", help="真正呼叫 Anthropic API（Optional，會產生費用）")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="付費 run 必填的成本上限（只限 anthropic）")
    parser.add_argument("--max-model-calls", type=int, default=None, help="本機 run 必填的模型呼叫次數上限")
    parser.add_argument("--ollama-url", default=ollama_url_default(),
                        help="只接受 loopback；預設取 tools/with-gpu-lock 設定的通道 URL")
    args = parser.parse_args(argv)

    config = local_agent_config() if args.provider == "ollama" else baseline_agent_config()
    budget = args.max_cost_usd
    if args.dry_run:
        client: Any = (OllamaOracleFakeClient if args.provider == "ollama" else OracleFakeClient)(week1_pool(args.split))
        name = args.run_name or f"dry-run-{args.split}"
    elif args.confirm_local_run:
        if args.provider != "ollama":
            parser.error("--confirm-local-run is only for --provider ollama")
        if args.max_cost_usd is not None:
            parser.error("cost caps do not apply to local models; use --max-model-calls (decision 026 §2)")
        if args.split == "eval" and args.rollouts != REGISTERED_EVAL_ROLLOUTS:
            parser.error(f"a formal eval baseline must use --rollouts {REGISTERED_EVAL_ROLLOUTS} (decision 025 §2)")
        client = start_local_run(parser, config, url=args.ollama_url, formal=args.split == "eval",
                                 max_model_calls=args.max_model_calls)
        name = args.run_name or f"local-{args.split}"
    else:
        if args.provider != "anthropic":
            parser.error("--confirm-paid-run is only for --provider anthropic (Optional, decision 026 §4)")
        if args.max_cost_usd is None:
            parser.error("--confirm-paid-run requires --max-cost-usd")
        if args.split == "eval" and args.rollouts != REGISTERED_EVAL_ROLLOUTS:
            parser.error(f"a paid eval baseline must use --rollouts {REGISTERED_EVAL_ROLLOUTS} (decision 025 §2)")
        client = make_anthropic_client()
        name = args.run_name or f"baseline-{args.split}"
    summary = run_split(
        args.split, client, run_dir=Path("runs") / name, rollouts=args.rollouts, max_cost_usd=budget, dry_run=args.dry_run,
        agent_config=config, max_model_calls=args.max_model_calls,
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
