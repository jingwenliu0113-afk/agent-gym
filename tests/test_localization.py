"""replay/localization.py 的契約測試（SPEC §6；decision log 010，並沿用 decision 006 的 checkpoint 規則與 decision 009 的
manifest / 例外分類）。

涵蓋：SweepPlan 驗證與 DEFAULT_SYNTHETIC_PLAN；valid_checkpoints / coarse_checkpoints；classify_rollout（success / failure /
refusal、stopped_without_action 只看續跑）；estimate_checkpoint（計數、p̂、p̂_completed、Wilson CI）；detect_cliff（同大小取 k
較小者、CI 不重疊規則、少於兩點）；以及 sweep 的端到端驗證：用 ``agents.synthetic.SyntheticPolicy``（解析解已知）在 smoke tasks 與
``week1_pool("dev")`` 上確認 cliff 定位、無錯誤 source 不被標為 cliff、Wilson CI 對解析解的涵蓋率、manifest 完整性、round-robin
排程、config 不符的事前拒絕，以及 infrastructure / agent_output / harness 三種中止路徑。

另外（critic 補強）：
- 高檢定力的解析解驗證：dev 5 題、q = 0.2、stride 1、每點 16 個 rollout，檢查 Wilson 涵蓋率與合併偏差（<= 3 SE）。
- 混合重試序列（infrastructure → agent_output / harness）、部分續跑後的重試（partial_steps 含前綴、attempt 2 從乾淨的
  checkpoint 重新開始）、同一 source 第二次寫入同一 manifest 的事前拒絕、兩個 source 共用 manifest。
- 以決定性的 play policy（與 source 同 config_id、依 (k, rollout_index) 決定成敗）驗證 refine 排程：共用端點的兩個區間、
  不相交的兩個區間，以及「未 refine 的 coarse 間隔緊鄰 refine 區間」的情境（判定規則待 decision，見該測試）。
- 噪音觸發 refine 但 refine 後拒絕 cliff 的路徑；真實但小於門檻的下降不被標為 cliff；mistake 在 m = n 的 cliff。
- manifest 與獨立 rerun 的交叉核對（trajectory_id、termination、model_calls_completed、分類）與 utc_time 格式。

設計說明：
- 所有 stub 都只在測試內呼叫（模組層級只建立 dataclass 常數）。所有 episode 一律用常數 clock（``lambda: 0.0``）。
- SyntheticPolicy 的隨機數取決於 ``sha256(rollout_seed, canonical_json(messages))``，因此 source 的 seed 一律在測試執行時
  **決定性地搜尋**（從 0 遞增，取第一個符合條件者），不寫死任何 seed；統計判定都選在遠離門檻的設定上。
- rollout 的 seed 由 (seed_offset, k, rollout_index, attempt) 決定，且與 source 搜尋範圍不重疊（否則 rollout 會逐 byte 重現
  source）。噪音觸發 refine 的測試以同樣的決定性搜尋找 seed_offset，但從已知的第一個符合者（OFFSET_SEARCH_START）開始以省時間。
- 「沒有錯誤的 source 不被標為 cliff」只在固定 seed 下成立：refine 沿用觸發它的 coarse rollout（winner's curse），偽陽性機率
  不可忽略（decision 011 的 gradual 情境約 0.09–0.15）。
- 端到端 sweep 以 module 層級的 lazy cache 共用（每個 source 只跑一次）；manifest 寫在 ``tmp_path`` / ``tmp_path_factory`` 下。

輸入檢查（coarse_checkpoints 的 last / stride、classify_rollout 的 k 範圍、detect_cliff 的 k 嚴格遞增）依各函式 docstring
中的 ValueError 條款撰寫；docstring 沒有寫到的錯誤處理（例如未知的 outcome 字串、plan 型別）不測。
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import math
from collections.abc import Callable
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import MAX_EPISODE_ATTEMPTS, AdapterError
from agents.policy import AgentConfig, ModelMessage, PolicyStep, ScriptedPolicy, ToolCall, oracle_policy
from agents.synthetic import SyntheticPolicy, checkpoint_truth, mistake_call
from env.hashing import canonical_json
from replay.aborts import AttemptRecord, RolloutKey, read_manifest
from replay.checkpoint import CheckpointError, make_checkpoint, rerun
from replay.localization import (
    DEFAULT_SYNTHETIC_PLAN,
    ROLLOUT_OUTCOMES,
    CheckpointEstimate,
    RolloutOutcome,
    SweepPlan,
    SweepResult,
    classify_rollout,
    coarse_checkpoints,
    detect_cliff,
    estimate_checkpoint,
    sweep,
    valid_checkpoints,
)
from replay.runner import run_episode
from replay.stats import Z_95, wilson_interval
from replay.trajectory import Trajectory, TrajectoryError, Turn
from tasks.generator import week1_pool
from tasks.schema import Task
from tasks.smoke import REFERENCE_SOLUTIONS, SMOKE_TASK_IDS, get_smoke_task

# ---------------------------------------------------------------- 常數（純資料，不呼叫 stub）

MAX_TURNS = 20  # >= 最長 oracle（7）+ 1，因此 P(success | k) 的解析解不受 max_turns 影響（decision 010 §1）
Q = 0.05  # 端到端 sweep 用的 mistake rate：cliff 前的 P(success | m) >= 0.95^3 ≈ 0.857，遠高於偵測門檻
SEED_SEARCH_LIMIT = 3000
ROLLOUT_SEED_BASE = 10**6  # rollout seed 一律 >= 此值，與 source 搜尋範圍 [0, SEED_SEARCH_LIMIT) 不重疊
COVERAGE_EPS = 1e-12  # Wilson 下界在 0 成功時數學上為 0，容許浮點誤差（例如 1e-17）

ASSIGN = "smoke-assign-and-start"  # oracle：assign_issue(8, u4)、update_issue(8, in_progress)；受保護 issue = 1
NOOP = "smoke-conditional-noop"  # 空 trajectory 即成功

# 端到端 sweep 的 source：(task_id, 第一個錯誤所在的 turn index；None = 整條 source 沒有錯誤)。
CLIFF_CASES: tuple[tuple[str, int], ...] = (
    ("smoke-assign-and-start", 0),  # n = 2；cliff 在 k = 0 → 1
    ("smoke-assign-and-start", 2),  # m = n：所有 oracle call 之後、結束訊息之前犯錯；K = 3 為奇數 → coarse {0, 2, 3}
    ("gen-close-duplicate-s0", 2),  # n = 3；m 為 coarse 點
    ("gen-escalate-label-s0", 1),  # n = 3；m 不是 coarse 點
    ("gen-reassign-departed-s0", 5),  # n = 7；m 不是 coarse 點
    ("gen-reassign-departed-s0", 7),  # m = n = 7
)
NO_CLIFF_CASES: tuple[tuple[str, None], ...] = tuple(
    (task_id, None)
    for task_id in (
        *SMOKE_TASK_IDS,  # 含 smoke-conditional-noop（n = 1）與 smoke-reassign-departed（n = 7）
        "gen-assign-and-start-s0",
        "gen-close-duplicate-s0",
        "gen-escalate-label-s0",
        "gen-reassign-departed-s0",
        "gen-conditional-assign-s0",  # dev 的 noop 分支，n = 1
    )
)
ALL_CASES: tuple[tuple[str, int | None], ...] = CLIFF_CASES + NO_CLIFF_CASES

# 流程測試用的小計畫（q = 0 的 source 與 rollout 全部成功，不會觸發 refine）。
FLOW_PLAN = SweepPlan(coarse_stride=2, n_coarse=2, n_fine=4, drop_threshold=0.3)
# 每個合法 checkpoint 各 1 個 rollout（coarse = 全部 k）。
EVERY_K_PLAN = SweepPlan(coarse_stride=1, n_coarse=1, n_fine=1, drop_threshold=0.3)

DEV_TASK_IDS: tuple[str, ...] = (
    "gen-assign-and-start-s0",
    "gen-close-duplicate-s0",
    "gen-escalate-label-s0",
    "gen-reassign-departed-s0",
    "gen-conditional-assign-s0",
)
# 高檢定力驗證：q = 0.2 讓真值分散在約 0.17–0.8（Q = 0.05 時全部擠在 0.66–0.95，差一個指數也測不出來）。
VALIDATION_Q = 0.2
VALIDATION_PLAN = SweepPlan(coarse_stride=1, n_coarse=16, n_fine=16, drop_threshold=0.3)
# 噪音觸發 refine 的路徑：沒有錯誤的 source、真值平緩上升，coarse 的 n = 8 雜訊偶爾會讓某個區間看起來下降 >= 0.3。
FALSE_POSITIVE_TASK = "gen-reassign-departed-s0"
FALSE_POSITIVE_Q = 0.15
# 從 seed_offset = 1 開始的決定性搜尋，第一個「coarse 有標記區間」的是 65（1–64 都沒有）；每次 sweep 約 0.4 秒，因此從 65 開始搜尋
# 只為省時間，搜尋條件（前提）不變；實作或雜湊細節改變時會自動往後找。
OFFSET_SEARCH_START = 65
OFFSET_SEARCH_LIMIT = 300
# 真實但小的下降：q = 0.25、n = 7、m = 0 → cliff 前真值 0.75^8 ≈ 0.10，遠低於 drop_threshold。
SMALL_DROP_TASK = "gen-reassign-departed-s0"
SMALL_DROP_Q = 0.25
# play policy 測試的 source（q = 0，n = 7 → 8 turns，合法 k = 0..7，DEFAULT_SYNTHETIC_PLAN 的 coarse = {0, 2, 4, 6, 7}）。
SCRIPTED_TASK = "smoke-reassign-departed"
# play policy 的多區間 refine 測試用小計畫（coarse p̂ 以 2 個 rollout 即可塑形；只驗證排程，不涉及統計）。
SCRIPTED_PLAN = SweepPlan(coarse_stride=2, n_coarse=2, n_fine=6, drop_threshold=0.3)


def _clock() -> float:
    return 0.0


# ---------------------------------------------------------------- 測試替身


class _FakeAPIConnectionError(Exception):
    """模擬 anthropic SDK 重試用盡後的連線錯誤（模組名稱 anthropic、類別名稱 APIConnectionError → infrastructure）。"""


_FakeAPIConnectionError.__module__ = "anthropic"
_FakeAPIConnectionError.__name__ = "APIConnectionError"
_FakeAPIConnectionError.__qualname__ = "APIConnectionError"


class _RefusalPolicy:
    """先依序每個 turn 發一個 ``calls`` 中的 tool call，之後回傳 stop_reason == "refusal" 的結束訊息（text 空、無 tool call）。
    對 conversation 無狀態（播放位置 = assistant 訊息數）。"""

    def __init__(self, calls: tuple[tuple[str, Any], ...] = ()) -> None:
        self._calls = tuple(calls)
        self._config = AgentConfig(
            policy="test-refusal",
            prompt_version="test-refusal/v1",
            system_prompt="",
            developer_prompt="",
            model_config={"calls": [{"name": name, "args": args} for name, args in self._calls]},
        )

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        k = sum(1 for message in messages if message["role"] == "assistant")
        if k < len(self._calls):
            name, args = self._calls[k]
            message = ModelMessage(text="", tool_calls=(ToolCall(id=f"call_{k}_0", name=name, args=args),), stop_reason="tool_use")
        else:
            message = ModelMessage(text="", tool_calls=(), stop_reason="refusal")
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


class _RaisingPolicy:
    """agent_config 委派給 inner（config_id 相同）；next_message 一律拋出指定例外。"""

    def __init__(self, inner: Any, exc: BaseException) -> None:
        self._inner = inner
        self._exc = exc
        self.calls = 0

    @property
    def agent_config(self) -> AgentConfig:
        return self._inner.agent_config

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        self.calls += 1
        raise self._exc


class _SpyPolicy:
    """包住另一個 policy、記錄 next_message 次數（用於確認「不執行」）。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def agent_config(self) -> AgentConfig:
        return self._inner.agent_config

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        self.calls += 1
        return self._inner.next_message(messages, tools)


class _StepThenRaisePolicy:
    """前 ``steps`` 次 next_message 委派給 inner，之後一律拋出指定例外（計數屬於這個物件；每次嘗試一個新物件）。"""

    def __init__(self, inner: Any, steps: int, exc: BaseException) -> None:
        self._inner = inner
        self._steps = steps
        self._exc = exc
        self.calls = 0

    @property
    def agent_config(self) -> AgentConfig:
        return self._inner.agent_config

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        self.calls += 1
        if self.calls > self._steps:
            raise self._exc
        return self._inner.next_message(messages, tools)


class _PlayPolicy:
    """決定性的測試替身：``agent_config`` 為給定的 config（傳入 source.agent → 與 source 同 config_id）。
    假設 checkpoint 前綴全是 oracle call（q = 0 的 source）：位置 = assistant 訊息數。
    - succeed：播放 oracle[位置]，做完全部 oracle 後結束（"Done."）→ 成功；
    - 否則：在 checkpoint 之後的第一個 turn 發出錯誤 call（受保護 issue 留言），之後結束 → 失敗（含 k = K）。"""

    def __init__(
        self,
        config: AgentConfig,
        oracle: tuple[tuple[str, Any], ...],
        mistake: tuple[str, Any],
        *,
        start: int,
        succeed: bool,
    ) -> None:
        self._config = copy.deepcopy(config)
        self._oracle = tuple(oracle)
        self._mistake = mistake
        self._start = start
        self._succeed = succeed

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        position = sum(1 for message in messages if message["role"] == "assistant")
        call: tuple[str, Any] | None
        if self._succeed:
            call = self._oracle[position] if position < len(self._oracle) else None
        else:
            call = self._mistake if position == self._start else None
        if call is None:
            message = ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn")
        else:
            name, args = call
            tool_call = ToolCall(id=f"call_{position}_0", name=name, args=copy.deepcopy(args))
            message = ModelMessage(text="", tool_calls=(tool_call,), stop_reason="tool_use")
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


class _PlayFactory:
    """``policy_factory(k, rollout_index, attempt)`` → ``_PlayPolicy``，成敗由 ``succeed(k, rollout_index)`` 決定。"""

    def __init__(
        self, source: Trajectory, oracle: tuple[tuple[str, Any], ...], mistake: tuple[str, Any], succeed: Callable[[int, int], bool]
    ) -> None:
        self._source = source
        self._oracle = oracle
        self._mistake = mistake
        self.succeed = succeed
        self.calls: list[tuple[int, int, int]] = []

    def __call__(self, k: int, rollout_index: int, attempt: int) -> Any:
        self.calls.append((k, rollout_index, attempt))
        return _PlayPolicy(
            self._source.agent, self._oracle, self._mistake, start=k, succeed=self.succeed(k, rollout_index)
        )


class _Factory:
    """``policy_factory(k, rollout_index, attempt)``：每次嘗試一個新的 SyntheticPolicy（與 source 同 config），
    rollout_seed = ROLLOUT_SEED_BASE + seed_offset·10^7 + k·100000 + rollout_index·10 + attempt（seed_offset 預設 0）。
    ``overrides[(k, rollout_index, attempt)] = f(inner) -> policy`` 用來注入例外或不同的 config。``calls`` 記錄每次呼叫。"""

    def __init__(
        self,
        task: Task,
        oracle: tuple[tuple[str, Any], ...],
        q: float,
        overrides: dict[tuple[int, int, int], Callable[[Any], Any]] | None = None,
        *,
        seed_offset: int = 0,
    ) -> None:
        self._task = task
        self._oracle = oracle
        self._q = q
        self._overrides = dict(overrides or {})
        self._seed_offset = seed_offset
        self.calls: list[tuple[int, int, int]] = []
        self.produced: dict[tuple[int, int, int], Any] = {}

    def __call__(self, k: int, rollout_index: int, attempt: int) -> Any:
        self.calls.append((k, rollout_index, attempt))
        seed = ROLLOUT_SEED_BASE + self._seed_offset * 10**7 + k * 100_000 + rollout_index * 10 + attempt
        policy: Any = _synthetic(self._task, self._oracle, self._q, seed)
        override = self._overrides.get((k, rollout_index, attempt))
        if override is not None:
            policy = override(policy)
        self.produced[(k, rollout_index, attempt)] = policy
        return policy


# ---------------------------------------------------------------- helpers


@functools.cache
def _dev_oracles() -> dict[str, tuple[dict[str, Any], tuple[tuple[str, Any], ...]]]:
    return {inst.task.task_id: (inst.task.to_dict(), inst.oracle_calls) for inst in week1_pool("dev")}


def _task_and_oracle(task_id: str) -> tuple[Task, tuple[tuple[str, Any], ...]]:
    if task_id in SMOKE_TASK_IDS:
        return get_smoke_task(task_id), tuple(REFERENCE_SOLUTIONS[task_id])
    task_dict, oracle = _dev_oracles()[task_id]
    return Task.from_dict(copy.deepcopy(task_dict)), copy.deepcopy(oracle)


def _synthetic(task: Task, oracle: tuple[tuple[str, Any], ...], q: float, seed: int) -> SyntheticPolicy:
    return SyntheticPolicy(oracle, mistake=mistake_call(task), mistake_rate=q, rollout_seed=seed)


def _source(task: Task, oracle: tuple[tuple[str, Any], ...], q: float, seed: int) -> Trajectory:
    return run_episode(task, _synthetic(task, oracle, q, seed), max_turns=MAX_TURNS, clock=_clock)


def _is_mistake_turn(turn: Turn, mistake: tuple[str, Any]) -> bool:
    calls = turn.message.tool_calls
    return len(calls) == 1 and calls[0].name == mistake[0] and canonical_json(calls[0].args) == canonical_json(mistake[1])


def _first_mistake(trajectory: Trajectory, mistake: tuple[str, Any]) -> int | None:
    for turn in trajectory.turns:
        if _is_mistake_turn(turn, mistake):
            return turn.index
    return None


def _find_source(task: Task, oracle: tuple[tuple[str, Any], ...], q: float, first_mistake: int | None) -> Trajectory:
    """從 seed 0 遞增，回傳第一條「第一個錯誤在 turn first_mistake（None = 沒有錯誤）且以 final_answer 結束」的 source。"""
    mistake = mistake_call(task)
    for seed in range(SEED_SEARCH_LIMIT):
        source = _source(task, oracle, q, seed)
        if source.termination == "final_answer" and _first_mistake(source, mistake) == first_mistake:
            return source
    pytest.fail(f"no source seed < {SEED_SEARCH_LIMIT} with first mistake at {first_mistake} for {task.task_id}")


def _wilson_reference(successes: int, n: int) -> tuple[float, float]:
    """detect_cliff 測試自用的 Wilson score interval（與 replay.stats 無關，避免依賴另一個 stub）。"""
    if n == 0:
        return (0.0, 1.0)
    z = Z_95
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _est(k: int, successes: int, n: int) -> CheckpointEstimate:
    """欄位彼此一致的 CheckpointEstimate（只有 success / failure）。"""
    low, high = _wilson_reference(successes, n)
    return CheckpointEstimate(
        k=k,
        n=n,
        successes=successes,
        failures=n - successes,
        refusals=0,
        agent_errors=0,
        aborted=0,
        max_turns_terminations=0,
        stopped_without_action=0,
        p_hat=successes / n if n else 0.0,
        p_hat_completed=successes / n if n else None,
        ci_low=low,
        ci_high=high,
    )


def _ro(
    k: int,
    index: int,
    outcome: str,
    termination: str | None = "final_answer",
    stopped_without_action: bool = False,
    attempts: int = 1,
) -> RolloutOutcome:
    return RolloutOutcome(
        k=k,
        rollout_index=index,
        outcome=outcome,
        termination=termination,
        stopped_without_action=stopped_without_action,
        attempts=attempts,
    )


def _key_of(record: AttemptRecord) -> RolloutKey:
    return RolloutKey(**{field.name: getattr(record, field.name) for field in dataclasses.fields(RolloutKey)})


def _replace_turn(trajectory: Trajectory, index: int, **changes: Any) -> Trajectory:
    turns = list(trajectory.turns)
    turns[index] = dataclasses.replace(turns[index], **changes)
    return dataclasses.replace(trajectory, turns=tuple(turns))


def _expected_schedule(source: Trajectory, result: SweepResult) -> list[tuple[int, int]]:
    """decision 010 §2–3 / 013 §2 的排程，獨立於 run_protocol 重算（以 result 記錄的每個 (k, i) 結果重放）：
    coarse 階段每一輪 i 依 k 遞增；之後反覆找「相鄰已估計點下降 >= 門檻（精確有理數）且區間內有 k 少於 n_fine 或未估計」的區間，
    把區間內所有 k 補到 n_fine（從既有數量接續，每一輪 i 依 k 遞增），直到沒有這樣的區間。"""
    plan = result.plan
    success = {(r.k, r.rollout_index): r.outcome == "success" for r in result.rollouts}
    threshold = Fraction(repr(float(plan.drop_threshold)))
    counts: dict[int, int] = {}
    schedule: list[tuple[int, int]] = []

    def run(targets: dict[int, int]) -> None:
        for i in range(max(targets.values(), default=0)):
            for k in sorted(targets):
                if counts.get(k, 0) == i and i < targets[k]:
                    schedule.append((k, i))
                    counts[k] = i + 1

    run({k: plan.n_coarse for k in coarse_checkpoints(valid_checkpoints(source)[-1], plan.coarse_stride)})
    while True:
        ks = sorted(counts)
        p = {k: Fraction(sum(success[(k, i)] for i in range(counts[k])), counts[k]) for k in ks}
        pending = [
            (a, b)
            for a, b in zip(ks, ks[1:])
            if p[a] - p[b] >= threshold and not (b == a + 1 and counts[a] >= plan.n_fine and counts[b] >= plan.n_fine)
        ]
        if not pending:
            return schedule
        run({k: plan.n_fine for a, b in pending for k in range(a, b + 1)})


@dataclasses.dataclass
class _SweepRun:
    task: Task
    oracle: tuple[tuple[str, Any], ...]
    first_mistake: int | None
    q: float
    source: Trajectory
    result: SweepResult
    log_path: Path


class _SweepCache:
    """每個 (task_id, first_mistake, q, plan) 只跑一次 sweep（預設 q = Q、DEFAULT_SYNTHETIC_PLAN、_Factory 的 seed_offset 0）；
    失敗不快取（每次重試用新的 manifest 檔）。"""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._runs: dict[tuple[str, int | None, float, SweepPlan], _SweepRun] = {}
        self._counter = 0

    def get(
        self, task_id: str, first_mistake: int | None, *, q: float = Q, plan: SweepPlan = DEFAULT_SYNTHETIC_PLAN
    ) -> _SweepRun:
        key = (task_id, first_mistake, q, plan)
        if key not in self._runs:
            task, oracle = _task_and_oracle(task_id)
            source = _find_source(task, oracle, q, first_mistake)
            self._counter += 1
            log_path = self._root / f"{task_id}-{first_mistake}-{self._counter}.jsonl"
            result = sweep(source, _Factory(task, oracle, q), plan=plan, log_path=log_path, clock=_clock)
            self._runs[key] = _SweepRun(task, oracle, first_mistake, q, source, result, log_path)
        return self._runs[key]


@pytest.fixture(scope="module")
def sweeps(tmp_path_factory: pytest.TempPathFactory) -> _SweepCache:
    return _SweepCache(tmp_path_factory.mktemp("localization-sweeps"))


def _flow_source(q: float = 0.0) -> tuple[Task, tuple[tuple[str, Any], ...], Trajectory]:
    """q = 0 的 smoke-assign-and-start source：assign、update、結束 → 3 turns，valid k = 0..2，coarse = {0, 2}。"""
    task, oracle = _task_and_oracle(ASSIGN)
    return task, oracle, _source(task, oracle, q, 0)


# ================================================================ 常數與 SweepPlan


def test_rollout_outcomes_constant() -> None:
    assert ROLLOUT_OUTCOMES == ("success", "failure", "refusal", "agent_error", "aborted")


def test_default_synthetic_plan_values() -> None:
    plan = DEFAULT_SYNTHETIC_PLAN
    assert isinstance(plan, SweepPlan)
    assert type(plan.coarse_stride) is int and plan.coarse_stride == 2
    assert type(plan.n_coarse) is int and plan.n_coarse == 8
    assert type(plan.n_fine) is int and plan.n_fine == 24
    assert plan.drop_threshold == 0.3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coarse_stride": 1, "n_coarse": 1, "n_fine": 1, "drop_threshold": 1.0},  # 下界、n_fine == n_coarse、threshold == 1
        {"coarse_stride": 3, "n_coarse": 4, "n_fine": 40, "drop_threshold": 1e-9},
        {"coarse_stride": 2, "n_coarse": 8, "n_fine": 24, "drop_threshold": 0.3},
    ],
)
def test_sweep_plan_accepts_valid_values(kwargs: dict[str, Any]) -> None:
    plan = SweepPlan(**kwargs)
    assert dataclasses.asdict(plan) == kwargs


_VALID_PLAN = {"coarse_stride": 2, "n_coarse": 8, "n_fine": 24, "drop_threshold": 0.3}


@pytest.mark.parametrize(
    "override",
    [
        {"coarse_stride": 0},
        {"coarse_stride": -2},
        {"coarse_stride": True},
        {"coarse_stride": 2.0},
        {"n_coarse": 0},
        {"n_coarse": True},
        {"n_coarse": 8.0},
        {"n_fine": 0},
        {"n_fine": 24.0},
        {"n_coarse": 8, "n_fine": 7},  # n_fine < n_coarse
        {"n_coarse": True, "n_fine": True},  # bool 不算 int
        {"drop_threshold": 0.0},
        {"drop_threshold": -0.1},
        {"drop_threshold": 1.0000001},
        {"drop_threshold": 2.0},
        {"drop_threshold": float("nan")},
    ],
    ids=lambda o: ",".join(f"{k}={v!r}" for k, v in o.items()),
)
def test_sweep_plan_rejects_invalid_values(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SweepPlan(**{**_VALID_PLAN, **override})


# ================================================================ valid_checkpoints / coarse_checkpoints


def test_valid_checkpoints_final_answer_trajectory() -> None:
    task_id = "smoke-reassign-departed"
    task, oracle = _task_and_oracle(task_id)
    trajectory = run_episode(task, oracle_policy(oracle), max_turns=MAX_TURNS, clock=_clock)
    assert trajectory.termination == "final_answer" and len(trajectory.turns) == len(oracle) + 1
    ks = valid_checkpoints(trajectory)
    assert ks == list(range(len(trajectory.turns)))
    for k in ks:
        assert make_checkpoint(trajectory, k).turn == k
    with pytest.raises(CheckpointError):
        make_checkpoint(trajectory, len(trajectory.turns))  # final_answer 之後不是合法 checkpoint


def test_valid_checkpoints_max_turns_trajectory() -> None:
    task, oracle = _task_and_oracle("smoke-reassign-departed")
    trajectory = run_episode(task, oracle_policy(oracle), max_turns=3, clock=_clock)
    assert trajectory.termination == "max_turns" and len(trajectory.turns) == 3
    ks = valid_checkpoints(trajectory)
    assert ks == [0, 1, 2]
    for k in ks:
        assert make_checkpoint(trajectory, k).turn == k
    with pytest.raises(CheckpointError):
        make_checkpoint(trajectory, 3)  # k == max_turns


def test_valid_checkpoints_single_final_turn() -> None:
    task, _ = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, ScriptedPolicy([]), max_turns=MAX_TURNS, clock=_clock)
    assert len(trajectory.turns) == 1
    assert valid_checkpoints(trajectory) == [0]


def test_valid_checkpoints_verifies_trajectory_first() -> None:
    task, oracle = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, oracle_policy(oracle), max_turns=MAX_TURNS, clock=_clock)
    record = trajectory.turns[0].tool_steps[0]
    forged = dataclasses.replace(record, result={**copy.deepcopy(record.result), "extra": True})
    bad = _replace_turn(trajectory, 0, tool_steps=(forged,))
    bad.validate()  # 結構合法，只有重放會發現
    with pytest.raises(TrajectoryError):
        valid_checkpoints(bad)


@pytest.mark.parametrize(
    ("last", "stride", "expected"),
    [
        (0, 1, [0]),
        (0, 2, [0]),
        (1, 2, [0, 1]),
        (2, 2, [0, 2]),
        (4, 2, [0, 2, 4]),
        (5, 2, [0, 2, 4, 5]),
        (3, 1, [0, 1, 2, 3]),
        (7, 3, [0, 3, 6, 7]),
        (9, 3, [0, 3, 6, 9]),
        (2, 5, [0, 2]),
        (10, 4, [0, 4, 8, 10]),
    ],
)
def test_coarse_checkpoints_examples(last: int, stride: int, expected: list[int]) -> None:
    assert coarse_checkpoints(last, stride) == expected


@pytest.mark.parametrize(
    ("last", "stride"),
    [(-1, 1), (3, 0), (3, -2), (True, 1), (3, True), (2.0, 1), (3, 2.0)],
)
def test_coarse_checkpoints_rejects_invalid_arguments(last: Any, stride: Any) -> None:
    with pytest.raises(ValueError):
        coarse_checkpoints(last, stride)


# ================================================================ classify_rollout


def test_classify_oracle_success() -> None:
    task, oracle = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, oracle_policy(oracle), max_turns=MAX_TURNS, clock=_clock)
    assert classify_rollout(trajectory, 0) == ("success", False)
    child = rerun(make_checkpoint(trajectory, 1), oracle_policy(oracle), clock=_clock)
    assert classify_rollout(child, 1) == ("success", False)


def test_classify_success_ending_by_max_turns() -> None:
    task, oracle = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, oracle_policy(oracle), max_turns=len(oracle), clock=_clock)
    assert trajectory.termination == "max_turns" and trajectory.grade().success
    assert classify_rollout(trajectory, 0) == ("success", False)


def test_classify_immediate_stop_is_stopped_without_action() -> None:
    task, _ = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, ScriptedPolicy([]), max_turns=MAX_TURNS, clock=_clock)
    assert classify_rollout(trajectory, 0) == ("failure", True)


@pytest.mark.parametrize(
    "call",
    [
        ("get_issue", {"issue_id": 8}),  # 唯讀
        ("update_issue", {"issue_id": 999, "status": "closed"}),  # 失敗的寫入：changes 為空
    ],
    ids=["read-only", "failed-write"],
)
def test_classify_tool_steps_without_changes_are_not_mutations(call: tuple[str, Any]) -> None:
    """mutation = changes 非空的 tool step；只有唯讀或失敗的 call 就結束仍是 stopped_without_action。"""
    task, _ = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, ScriptedPolicy([[call]]), max_turns=MAX_TURNS, clock=_clock)
    assert trajectory.termination == "final_answer"
    assert not any(step.changes for turn in trajectory.turns for step in turn.tool_steps)
    assert classify_rollout(trajectory, 0) == ("failure", True)


def test_classify_failure_with_mutation_is_not_stopped_without_action() -> None:
    task, _ = _task_and_oracle(ASSIGN)
    policy = ScriptedPolicy([[("add_comment", {"issue_id": 1, "body": "hello"})]])
    trajectory = run_episode(task, policy, max_turns=MAX_TURNS, clock=_clock)
    assert trajectory.termination == "final_answer" and not trajectory.grade().success
    assert classify_rollout(trajectory, 0) == ("failure", False)


def test_classify_stopped_without_action_only_counts_the_continuation() -> None:
    """前綴（index < k）的 mutation 不算；續跑（index >= k）沒有 mutation 才是 stopped_without_action。"""
    task, oracle = _task_and_oracle(ASSIGN)
    policy = ScriptedPolicy([[oracle[0]]])  # 只做 assign，接著結束（status 沒改 → 未成功）
    source = run_episode(task, policy, max_turns=MAX_TURNS, clock=_clock)
    assert len(source.turns) == 2 and source.turns[0].tool_steps[0].changes
    from_start = rerun(make_checkpoint(source, 0), policy, clock=_clock)
    after_mutation = rerun(make_checkpoint(source, 1), policy, clock=_clock)
    assert classify_rollout(from_start, 0) == ("failure", False)
    assert classify_rollout(after_mutation, 1) == ("failure", True)


def test_classify_max_turns_failure_is_not_stopped_without_action() -> None:
    task, _ = _task_and_oracle(ASSIGN)
    read = ("get_issue", {"issue_id": 8})
    trajectory = run_episode(task, ScriptedPolicy([[read], [read]]), max_turns=2, clock=_clock)
    assert trajectory.termination == "max_turns"
    assert classify_rollout(trajectory, 0) == ("failure", False)


def test_classify_refusal_without_action() -> None:
    task, _ = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, _RefusalPolicy(), max_turns=MAX_TURNS, clock=_clock)
    assert trajectory.turns[-1].message.stop_reason == "refusal"
    assert classify_rollout(trajectory, 0) == ("refusal", True)


@pytest.mark.parametrize("task_id", [ASSIGN, NOOP])
def test_classify_refusal_takes_precedence_over_grade_success(task_id: str) -> None:
    """最後一個 turn 的 stop_reason == "refusal" → "refusal"，即使 grade().success 為 True。"""
    task, oracle = _task_and_oracle(task_id)
    calls = oracle if task_id == ASSIGN else ()  # noop：什麼都不做本身就成功
    trajectory = run_episode(task, _RefusalPolicy(calls), max_turns=MAX_TURNS, clock=_clock)
    assert trajectory.grade().success
    outcome, _ = classify_rollout(trajectory, 0)
    assert outcome == "refusal"


@pytest.mark.parametrize("k", [-1, 3, 4, 1.0])
def test_classify_rejects_k_outside_the_trajectory(k: Any) -> None:
    task, oracle = _task_and_oracle(ASSIGN)
    trajectory = run_episode(task, oracle_policy(oracle), max_turns=MAX_TURNS, clock=_clock)
    assert len(trajectory.turns) == 3
    with pytest.raises(ValueError):
        classify_rollout(trajectory, k)


# ================================================================ estimate_checkpoint


def test_estimate_checkpoint_counts_and_wilson() -> None:
    outcomes = [
        _ro(3, 0, "success"),
        _ro(3, 1, "failure", "max_turns"),
        _ro(3, 2, "success", "max_turns"),
        _ro(3, 3, "refusal", "final_answer", stopped_without_action=True),
        _ro(3, 4, "agent_error", None),
        _ro(3, 5, "failure", "final_answer", stopped_without_action=True),
        _ro(3, 6, "aborted", None, attempts=2),
        _ro(3, 7, "success"),
    ]
    est = estimate_checkpoint(3, outcomes)
    assert isinstance(est, CheckpointEstimate)
    assert (est.k, est.n, est.successes, est.failures, est.refusals, est.agent_errors, est.aborted) == (3, 8, 3, 2, 1, 1, 1)
    assert est.max_turns_terminations == 2
    assert est.stopped_without_action == 2
    assert est.p_hat == pytest.approx(3 / 8)  # refusal / agent_error / aborted 都計為非 success
    assert est.p_hat_completed == pytest.approx(3 / 5)
    low, high = wilson_interval(3, 8)
    assert est.ci_low == pytest.approx(low, abs=1e-12)
    assert est.ci_high == pytest.approx(high, abs=1e-12)
    assert est.ci_low <= est.p_hat <= est.ci_high


def test_estimate_checkpoint_order_does_not_matter() -> None:
    outcomes = [_ro(1, i, "success" if i % 3 else "failure") for i in range(9)]
    assert estimate_checkpoint(1, outcomes) == estimate_checkpoint(1, list(reversed(outcomes)))


def test_estimate_checkpoint_all_success_and_all_failure() -> None:
    full = estimate_checkpoint(0, [_ro(0, i, "success") for i in range(5)])
    assert (full.n, full.successes, full.p_hat, full.p_hat_completed) == (5, 5, 1.0, 1.0)
    assert (full.ci_low, full.ci_high) == pytest.approx(wilson_interval(5, 5), abs=1e-12)
    none = estimate_checkpoint(0, [_ro(0, i, "failure") for i in range(5)])
    assert (none.n, none.successes, none.failures, none.p_hat, none.p_hat_completed) == (5, 0, 5, 0.0, 0.0)
    assert (none.ci_low, none.ci_high) == pytest.approx(wilson_interval(0, 5), abs=1e-12)


def test_estimate_checkpoint_p_hat_completed_none_without_completed_rollouts() -> None:
    outcomes = [_ro(2, 0, "refusal", stopped_without_action=True), _ro(2, 1, "agent_error", None), _ro(2, 2, "aborted", None, attempts=2)]
    est = estimate_checkpoint(2, outcomes)
    assert (est.n, est.successes, est.failures, est.refusals, est.agent_errors, est.aborted) == (3, 0, 0, 1, 1, 1)
    assert est.p_hat == 0.0
    assert est.p_hat_completed is None
    assert (est.ci_low, est.ci_high) == pytest.approx(wilson_interval(0, 3), abs=1e-12)


def test_estimate_checkpoint_empty() -> None:
    est = estimate_checkpoint(4, [])
    assert (est.k, est.n, est.successes, est.failures, est.refusals, est.agent_errors, est.aborted) == (4, 0, 0, 0, 0, 0, 0)
    assert (est.max_turns_terminations, est.stopped_without_action) == (0, 0)
    assert est.p_hat == 0.0
    assert est.p_hat_completed is None
    assert (est.ci_low, est.ci_high) == pytest.approx(wilson_interval(0, 0))


@pytest.mark.parametrize(
    "ks",
    [[2], [3, 2], [2, 3, 3]],
    ids=["all-other-k", "one-other-k", "mixed"],
)
def test_estimate_checkpoint_rejects_k_mismatch(ks: list[int]) -> None:
    outcomes = [_ro(k, i, "success") for i, k in enumerate(ks)]
    with pytest.raises(ValueError):
        estimate_checkpoint(3, outcomes)


# ================================================================ detect_cliff


@pytest.mark.parametrize("estimates", [(), (_est(0, 8, 8),), (_est(3, 0, 24),)], ids=["empty", "one-high", "one-low"])
def test_detect_cliff_fewer_than_two_estimates(estimates: tuple[CheckpointEstimate, ...]) -> None:
    assert detect_cliff(estimates, 0.3) == (0.0, None, False)


def test_detect_cliff_clear_cliff() -> None:
    estimates = (_est(0, 20, 24), _est(1, 22, 24), _est(2, 0, 24), _est(3, 0, 24))
    drop, k, cliff = detect_cliff(estimates, 0.3)
    assert drop == pytest.approx(22 / 24)
    assert k == 1  # 最大下降的前一點
    assert cliff is True


def test_detect_cliff_uses_adjacent_estimated_checkpoints() -> None:
    """相鄰的是「已估計」的 checkpoint（k 可以不連續）；回傳前一點的 k。

    依據：SweepResult.largest_drop 的 docstring（「相鄰已估計 checkpoint 的最大 p̂(k) − p̂(k')」）。注意 decision 010 §3 寫的是
    Δ* = max_k [p̂(k) − p̂(k+1)]；兩者對「k 不連續的一對」不一致，尚無 decision log 處理（見
    test_sweep_unrefined_gap_next_to_refined_interval）。若 decision 改成只看連續的 (k, k+1)，本測試要一起改。"""
    estimates = (_est(0, 8, 8), _est(2, 8, 8), _est(5, 0, 24), _est(9, 0, 24))
    drop, k, cliff = detect_cliff(estimates, 0.3)
    assert drop == pytest.approx(1.0)
    assert k == 2
    assert cliff is True


def test_detect_cliff_requires_non_overlapping_wilson_cis() -> None:
    """下降 1.0 >= threshold，但 n = 2 的兩個 Wilson CI 重疊 → 不是 cliff。"""
    a, b = _est(0, 2, 2), _est(1, 0, 2)
    assert a.ci_low < b.ci_high  # 前提：CI 重疊
    drop, k, cliff = detect_cliff((a, b), 0.3)
    assert drop == pytest.approx(1.0)
    assert k == 0
    assert cliff is False


def test_detect_cliff_requires_drop_threshold() -> None:
    """CI 不重疊，但下降 0.2 < 0.3 → 不是 cliff。"""
    a, b = _est(0, 1000, 1000), _est(1, 800, 1000)
    assert a.ci_low > b.ci_high  # 前提：CI 不重疊
    drop, k, cliff = detect_cliff((a, b), 0.3)
    assert drop == pytest.approx(0.2)
    assert k == 0
    assert cliff is False


def test_detect_cliff_drop_equal_to_threshold_counts() -> None:
    a, b = _est(0, 1000, 1000), _est(1, 500, 1000)
    drop, k, cliff = detect_cliff((a, b), 0.5)
    assert drop == 0.5
    assert (k, cliff) == (0, True)


def test_detect_cliff_tie_picks_smaller_k() -> None:
    estimates = (_est(0, 24, 24), _est(1, 12, 24), _est(2, 24, 24), _est(3, 12, 24))
    drop, k, cliff = detect_cliff(estimates, 0.3)
    assert drop == 0.5
    assert k == 0
    assert cliff is True


def test_detect_cliff_flat_curve() -> None:
    estimates = (_est(0, 4, 8), _est(2, 4, 8), _est(4, 4, 8))
    assert detect_cliff(estimates, 0.3) == (0.0, 0, False)


@pytest.mark.parametrize(
    ("estimates", "expected"),
    [
        ((_est(0, 0, 24), _est(1, 24, 24)), (-1.0, 0, False)),  # 上升 1.0 不是下降
        ((_est(0, 0, 8), _est(2, 4, 8), _est(4, 8, 8)), (-0.5, 0, False)),  # 兩段同為 −0.5 → 取 k 較小者
    ],
    ids=["jump-up", "steady-rise"],
)
def test_detect_cliff_rise_is_not_a_cliff(
    estimates: tuple[CheckpointEstimate, ...], expected: tuple[float, int, bool]
) -> None:
    """只上升的曲線：largest_drop 是「最大的 p̂(k) − p̂(k')」本身（docstring 與 decision 010 §3 都沒有夾到 0），因此為負。
    跨 task 彙總 Δ* 時是否夾到 0 是分析層的決定（不屬於 detect_cliff 的契約）。"""
    drop, k, cliff = detect_cliff(estimates, 0.3)
    assert drop == pytest.approx(expected[0])
    assert (k, cliff) == expected[1:]


def test_detect_cliff_largest_drop_ignores_larger_rises() -> None:
    """下降是 p̂(k) − p̂(k')（前減後）：前段上升 1.0 不可蓋過後段的下降 0.5。"""
    estimates = (_est(0, 0, 24), _est(1, 24, 24), _est(2, 12, 24))
    drop, k, cliff = detect_cliff(estimates, 0.3)
    assert drop == 0.5
    assert k == 1
    assert cliff is True


@pytest.mark.parametrize("ks", [(2, 0), (1, 1), (0, 3, 2)], ids=["descending", "duplicate", "unsorted-tail"])
def test_detect_cliff_rejects_estimates_not_strictly_increasing_in_k(ks: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        detect_cliff(tuple(_est(k, 4, 8) for k in ks), 0.3)


# ================================================================ sweep：流程（q = 0，結果決定性）


def test_sweep_all_success_source_runs_coarse_only(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    assert len(source.turns) == 3
    factory = _Factory(task, oracle, 0.0)
    result = sweep(source, factory, plan=FLOW_PLAN, log_path=tmp_path / "manifest.jsonl", clock=_clock)
    assert isinstance(result, SweepResult)
    assert result.task_id == ASSIGN
    assert result.source_trajectory_id == source.trajectory_id
    assert result.plan == FLOW_PLAN
    assert [(r.k, r.rollout_index) for r in result.rollouts] == [(0, 0), (2, 0), (0, 1), (2, 1)]
    for r in result.rollouts:
        assert (r.outcome, r.termination, r.stopped_without_action, r.attempts) == ("success", "final_answer", False, 1)
    assert [e.k for e in result.estimates] == [0, 2]  # k = 0 也以 rerun 估計；只含有 rollout 的 checkpoint
    for e in result.estimates:
        assert (e.n, e.successes, e.p_hat) == (2, 2, 1.0)
    assert result.refined_intervals == ()
    assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == detect_cliff(result.estimates, 0.3)
    assert result.is_cliff is False


def test_sweep_accepts_str_log_path(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, _Factory(task, oracle, 0.0), plan=FLOW_PLAN, log_path=str(log_path), clock=_clock)
    assert len(read_manifest(log_path)) == len(result.rollouts) == 4


def test_sweep_flow_manifest_rows(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, _Factory(task, oracle, 0.0), plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    records = read_manifest(log_path)
    cp_ids = {k: make_checkpoint(source, k).checkpoint_id for k in (0, 2)}
    assert [_key_of(r) for r in records] == [
        RolloutKey(cp_ids[r.k], r.k, source.trajectory_id, r.rollout_index, 1) for r in result.rollouts
    ]
    for record in records:
        assert record.outcome == "completed"
        assert record.task_id == ASSIGN
        assert record.config_id == source.agent.config_id
        assert record.max_turns == source.max_turns
        assert record.termination == "final_answer"


def test_sweep_rejects_mismatched_config_before_any_rollout(tmp_path: Path) -> None:
    """policy_factory 的 policy 與 source 的 config_id 不同（只差 mistake_rate）→ ValueError，不執行、不寫 manifest。"""
    task, oracle, source = _flow_source()
    spies: list[_SpyPolicy] = []

    def factory(k: int, rollout_index: int, attempt: int) -> Any:
        spy = _SpyPolicy(_synthetic(task, oracle, 0.1, ROLLOUT_SEED_BASE + rollout_index))
        spies.append(spy)
        return spy

    log_path = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError):
        sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert read_manifest(log_path) == ()
    assert spies and all(spy.calls == 0 for spy in spies)


def test_sweep_rejects_different_agent_before_any_rollout(tmp_path: Path) -> None:
    _, oracle, source = _flow_source()
    spy = _SpyPolicy(oracle_policy(oracle))
    log_path = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError):
        sweep(source, lambda k, i, a: spy, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert read_manifest(log_path) == ()
    assert spy.calls == 0


def test_sweep_rejects_mismatched_config_mid_sweep_without_running_it(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    spy = _SpyPolicy(oracle_policy(oracle))
    factory = _Factory(task, oracle, 0.0, overrides={(2, 0, 1): lambda inner: spy})
    log_path = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError):
        sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert spy.calls == 0
    records = read_manifest(log_path)
    assert [(r.checkpoint_turn, r.rollout_index, r.outcome) for r in records] == [(0, 0, "completed")]


def test_sweep_retries_infrastructure_abort(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    factory = _Factory(
        task, oracle, 0.0, overrides={(0, 0, 1): lambda inner: _RaisingPolicy(inner, _FakeAPIConnectionError("connection reset"))}
    )
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert result.rollouts[0] == RolloutOutcome(
        k=0, rollout_index=0, outcome="success", termination="final_answer", stopped_without_action=False, attempts=2
    )
    assert all(r.attempts == 1 for r in result.rollouts[1:])
    assert (0, 0, 2) in factory.calls
    assert factory.produced[(0, 0, 1)].calls == 1
    records = read_manifest(log_path)
    assert [(r.checkpoint_turn, r.rollout_index, r.attempt, r.outcome) for r in records] == [
        (0, 0, 1, "aborted"),
        (0, 0, 2, "completed"),
        (2, 0, 1, "completed"),
        (0, 1, 1, "completed"),
        (2, 1, 1, "completed"),
    ]
    assert records[0].error_class == "infrastructure"
    assert _key_of(records[0]).checkpoint_id == _key_of(records[1]).checkpoint_id == make_checkpoint(source, 0).checkpoint_id
    k0 = next(e for e in result.estimates if e.k == 0)
    assert (k0.n, k0.successes, k0.aborted) == (2, 2, 0)  # 重試成功的 rollout 只算一次


def test_sweep_infrastructure_abort_on_every_attempt_is_aborted(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    overrides = {
        (0, 0, attempt): (lambda inner: _RaisingPolicy(inner, _FakeAPIConnectionError("connection reset")))
        for attempt in range(1, MAX_EPISODE_ATTEMPTS + 1)
    }
    factory = _Factory(task, oracle, 0.0, overrides=overrides)
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert result.rollouts[0] == RolloutOutcome(
        k=0, rollout_index=0, outcome="aborted", termination=None, stopped_without_action=False, attempts=MAX_EPISODE_ATTEMPTS
    )
    assert all(attempt <= MAX_EPISODE_ATTEMPTS for _, _, attempt in factory.calls)
    records = [r for r in read_manifest(log_path) if (r.checkpoint_turn, r.rollout_index) == (0, 0)]
    assert [(r.attempt, r.outcome, r.error_class) for r in records] == [
        (attempt, "aborted", "infrastructure") for attempt in range(1, MAX_EPISODE_ATTEMPTS + 1)
    ]
    k0 = next(e for e in result.estimates if e.k == 0)
    assert (k0.n, k0.successes, k0.aborted, k0.p_hat, k0.p_hat_completed) == (2, 1, 1, 0.5, 1.0)


def test_sweep_agent_output_abort_is_agent_error_without_retry(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    factory = _Factory(
        task, oracle, 0.0, overrides={(0, 0, 1): lambda inner: _RaisingPolicy(inner, AdapterError("stop_reason max_tokens"))}
    )
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert result.rollouts[0] == RolloutOutcome(
        k=0, rollout_index=0, outcome="agent_error", termination=None, stopped_without_action=False, attempts=1
    )
    assert (0, 0, 2) not in factory.calls
    records = [r for r in read_manifest(log_path) if (r.checkpoint_turn, r.rollout_index) == (0, 0)]
    assert [(r.attempt, r.outcome, r.error_class) for r in records] == [(1, "aborted", "agent_output")]
    k0 = next(e for e in result.estimates if e.k == 0)
    assert (k0.n, k0.successes, k0.agent_errors, k0.p_hat, k0.p_hat_completed) == (2, 1, 1, 0.5, 1.0)


def test_sweep_harness_exception_propagates(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    boom = RuntimeError("harness bug")
    factory = _Factory(task, oracle, 0.0, overrides={(2, 0, 1): lambda inner: _RaisingPolicy(inner, boom)})
    log_path = tmp_path / "manifest.jsonl"
    with pytest.raises(RuntimeError) as excinfo:
        sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert excinfo.value is boom
    assert (2, 0, 2) not in factory.calls
    records = read_manifest(log_path)
    assert (records[-1].checkpoint_turn, records[-1].rollout_index, records[-1].outcome, records[-1].error_class) == (
        2,
        0,
        "aborted",
        "harness",
    )


def _infra(inner: Any) -> Any:
    return _RaisingPolicy(inner, _FakeAPIConnectionError("connection reset"))


def test_sweep_infrastructure_then_agent_output_is_agent_error(tmp_path: Path) -> None:
    """結果標籤取自最後一次嘗試（RetryResult.error_class）：infrastructure → 重試 → agent_output → "agent_error"，不再重試。"""
    task, oracle, source = _flow_source()
    factory = _Factory(
        task,
        oracle,
        0.0,
        overrides={(0, 0, 1): _infra, (0, 0, 2): lambda inner: _RaisingPolicy(inner, AdapterError("stop_reason max_tokens"))},
    )
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert result.rollouts[0] == RolloutOutcome(
        k=0, rollout_index=0, outcome="agent_error", termination=None, stopped_without_action=False, attempts=2
    )
    assert (0, 0, 3) not in factory.calls
    records = [r for r in read_manifest(log_path) if (r.checkpoint_turn, r.rollout_index) == (0, 0)]
    assert [(r.attempt, r.outcome, r.error_class) for r in records] == [
        (1, "aborted", "infrastructure"),
        (2, "aborted", "agent_output"),
    ]
    k0 = next(e for e in result.estimates if e.k == 0)
    assert (k0.n, k0.successes, k0.agent_errors, k0.aborted) == (2, 1, 1, 0)


def test_sweep_infrastructure_then_harness_exception_propagates(tmp_path: Path) -> None:
    task, oracle, source = _flow_source()
    boom = RuntimeError("harness bug on retry")
    factory = _Factory(task, oracle, 0.0, overrides={(0, 0, 1): _infra, (0, 0, 2): lambda inner: _RaisingPolicy(inner, boom)})
    log_path = tmp_path / "manifest.jsonl"
    with pytest.raises(RuntimeError) as excinfo:
        sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert excinfo.value is boom
    assert (0, 0, 3) not in factory.calls
    assert [(r.checkpoint_turn, r.rollout_index, r.attempt, r.outcome, r.error_class) for r in read_manifest(log_path)] == [
        (0, 0, 1, "aborted", "infrastructure"),
        (0, 0, 2, "aborted", "harness"),
    ]


def test_sweep_retry_after_partial_continuation_restarts_from_clean_checkpoint(tmp_path: Path) -> None:
    """attempt 1 在 checkpoint k = 1 之後先執行一個 mutation（update_issue）再中止：manifest 的 partial_steps 含前綴、
    model_calls_completed 不含前綴；attempt 2 必須從乾淨的 checkpoint 重新開始（與獨立的 rerun 逐 byte 相同）。"""
    task, oracle, source = _flow_source()
    k = 1
    assert source.turns[k].tool_steps and source.turns[k].tool_steps[0].changes  # 前提：k = 1 之後的下一步是 mutation
    factory = _Factory(
        task,
        oracle,
        0.0,
        overrides={(k, 0, 1): lambda inner: _StepThenRaisePolicy(inner, 1, _FakeAPIConnectionError("connection reset"))},
    )
    log_path = tmp_path / "manifest.jsonl"
    result = sweep(source, factory, plan=EVERY_K_PLAN, log_path=log_path, clock=_clock)
    assert factory.produced[(k, 0, 1)].calls == 2  # 一次成功回傳、一次拋出
    outcome = next(r for r in result.rollouts if (r.k, r.rollout_index) == (k, 0))
    assert outcome == RolloutOutcome(
        k=k, rollout_index=0, outcome="success", termination="final_answer", stopped_without_action=False, attempts=2
    )
    rows = [r for r in read_manifest(log_path) if r.checkpoint_turn == k]
    assert [(r.attempt, r.outcome, r.error_class) for r in rows] == [(1, "aborted", "infrastructure"), (2, "completed", None)]
    checkpoint = make_checkpoint(source, k)
    assert rows[0].checkpoint_id == rows[1].checkpoint_id == checkpoint.checkpoint_id
    assert rows[0].partial_steps == sum(len(turn.tool_steps) for turn in source.turns[:k]) + 1
    assert rows[0].model_calls_completed == 1
    expected = rerun(checkpoint, _Factory(task, oracle, 0.0)(k, 0, 2), clock=_clock)
    assert rows[1].trajectory_id == expected.trajectory_id
    assert rows[1].model_calls_completed == len(expected.turns) - k


def test_sweep_same_source_twice_into_one_manifest_is_rejected_before_running(tmp_path: Path) -> None:
    """第二次對同一 source 寫入同一 manifest：第一個 rollout 的唯一鍵已存在 → run_logged 的事前檢查 ValueError 原樣往外拋
    （run_with_retries 的契約），不執行任何 rollout、不寫任何紀錄。"""
    task, oracle, source = _flow_source()
    log_path = tmp_path / "manifest.jsonl"
    sweep(source, _Factory(task, oracle, 0.0), plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    before = log_path.read_bytes()
    assert len(read_manifest(log_path)) == 4
    spies: list[_SpyPolicy] = []

    def factory(k: int, rollout_index: int, attempt: int) -> Any:
        spy = _SpyPolicy(_synthetic(task, oracle, 0.0, ROLLOUT_SEED_BASE + rollout_index))
        spies.append(spy)
        return spy

    with pytest.raises(ValueError):
        sweep(source, factory, plan=FLOW_PLAN, log_path=log_path, clock=_clock)
    assert log_path.read_bytes() == before
    assert all(spy.calls == 0 for spy in spies)


def test_sweep_two_sources_of_the_same_task_share_one_manifest(tmp_path: Path) -> None:
    """同 task、同 config 的兩條不同 source（只差 max_turns）→ checkpoint_id 不同，可寫入同一 manifest、唯一鍵不衝突。"""
    task, oracle = _task_and_oracle(ASSIGN)
    first = _source(task, oracle, 0.0, 0)
    second = run_episode(task, _synthetic(task, oracle, 0.0, 0), max_turns=10, clock=_clock)
    assert first.agent.config_id == second.agent.config_id
    assert first.trajectory_id != second.trajectory_id
    log_path = tmp_path / "manifest.jsonl"
    results = [
        sweep(src, _Factory(task, oracle, 0.0), plan=FLOW_PLAN, log_path=log_path, clock=_clock) for src in (first, second)
    ]
    records = read_manifest(log_path)
    assert len(records) == sum(len(r.rollouts) for r in results) == 8
    keys = [(r.task_id, r.config_id, r.checkpoint_id, r.rollout_index, r.attempt) for r in records]
    assert len(set(keys)) == len(keys)
    assert [r.source_trajectory_id for r in records] == [first.trajectory_id] * 4 + [second.trajectory_id] * 4
    for src, rows in ((first, records[:4]), (second, records[4:])):
        assert {r.checkpoint_id for r in rows} == {make_checkpoint(src, k).checkpoint_id for k in (0, 2)}
        assert all(r.outcome == "completed" and r.max_turns == src.max_turns for r in rows)


# ================================================================ sweep：決定性 play policy（refine 排程）


def _scripted_sweep(
    succeed: Callable[[int, int], bool], log_path: Path, plan: SweepPlan
) -> tuple[Trajectory, _PlayFactory, SweepResult]:
    task, oracle = _task_and_oracle(SCRIPTED_TASK)
    source = _source(task, oracle, 0.0, 0)
    assert len(source.turns) == len(oracle) + 1 == 8  # 前提：q = 0 → 全是 oracle call，合法 k = 0..7
    factory = _PlayFactory(source, oracle, mistake_call(task), succeed)
    result = sweep(source, factory, plan=plan, log_path=log_path, clock=_clock)
    return source, factory, result


def _check_scripted_result(source: Trajectory, factory: _PlayFactory, result: SweepResult, log_path: Path) -> None:
    """與 succeed 規則、refine 區間、排程、manifest 一致（不涉及 cliff 判定規則）。"""
    plan = result.plan
    coarse = coarse_checkpoints(valid_checkpoints(source)[-1], plan.coarse_stride)
    refined = {k for a, b in result.refined_intervals for k in range(a, b + 1)}
    assert [e.k for e in result.estimates] == sorted(set(coarse) | refined)
    for e in result.estimates:
        assert e.n == (plan.n_fine if e.k in refined else plan.n_coarse)
        indices = [r.rollout_index for r in result.rollouts if r.k == e.k]
        assert sorted(indices) == list(range(e.n))  # 共用端點不重複取樣、從既有數量接續
        assert e.successes == sum(factory.succeed(e.k, i) for i in range(e.n))
    for r in result.rollouts:
        assert r.outcome == ("success" if factory.succeed(r.k, r.rollout_index) else "failure")
        assert r.attempts == 1
    assert [(r.k, r.rollout_index) for r in result.rollouts] == _expected_schedule(source, result)
    assert len(factory.calls) == len(result.rollouts)
    records = read_manifest(log_path)
    assert len(records) == len(result.rollouts)
    assert len({_key_of(record) for record in records}) == len(records)
    assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == detect_cliff(result.estimates, plan.drop_threshold)


# coarse = {0, 2, 4, 6, 7}（SCRIPTED_PLAN：n_coarse = 2、drop_threshold = 0.3）
SCRIPTED_SHAPES: dict[str, tuple[Callable[[int, int], bool], tuple[tuple[int, int], ...]]] = {
    # coarse p̂ = 1, 0.5, 0, 0, 0 → (0, 2) 與 (2, 4) 共用端點 2
    "shared-endpoint": (lambda k, i: k < 2 or (k == 2 and i % 2 == 0), ((0, 2), (2, 4))),
    # coarse p̂ = 1, 0.5, 0.5, 0, 0 → (0, 2) 與 (4, 6) 不相交（k = 3 不被估計）
    "disjoint": (lambda k, i: k < 2 or (2 <= k <= 5 and i % 2 == 0), ((0, 2), (4, 6))),
}


@pytest.mark.parametrize("shape", sorted(SCRIPTED_SHAPES))
def test_sweep_refines_multiple_intervals(tmp_path: Path, shape: str) -> None:
    succeed, expected_intervals = SCRIPTED_SHAPES[shape]
    log_path = tmp_path / "manifest.jsonl"
    source, factory, result = _scripted_sweep(succeed, log_path, SCRIPTED_PLAN)
    assert result.refined_intervals == expected_intervals
    _check_scripted_result(source, factory, result, log_path)


def test_sweep_unrefined_gap_next_to_refined_interval(tmp_path: Path) -> None:
    """coarse p̂ = 1, 2/8, 0, 0, 0：第 1 輪只標記 (0, 2)；refine 後 p̂(2) = 18/24 = 0.75，與未加密的 p̂(4) = 0/8 下降 0.75 >= 0.3，
    且 k = 3 從未估計（下降可能在 2→3 或 3→4）。decision 013 §2：這個 gap 也必須加密 → 第 2 輪標記 (2, 4)，k = 3 被估計，
    cliff 落在連續的 (2, 3) 上。"""

    def succeed(k: int, i: int) -> bool:
        return k < 2 or (k == 2 and (i < 2 or i >= DEFAULT_SYNTHETIC_PLAN.n_coarse))

    log_path = tmp_path / "manifest.jsonl"
    source, factory, result = _scripted_sweep(succeed, log_path, DEFAULT_SYNTHETIC_PLAN)
    assert result.refined_intervals == ((0, 2), (2, 4))
    _check_scripted_result(source, factory, result, log_path)
    by_k = {e.k: e for e in result.estimates}
    assert sorted(by_k) == [0, 1, 2, 3, 4, 6, 7]
    assert (by_k[2].n, by_k[2].successes) == (24, 18)
    assert (by_k[3].n, by_k[3].successes) == (24, 0)
    assert (by_k[4].n, by_k[4].successes) == (24, 0)
    assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == (0.75, 2, True)


# ================================================================ sweep：synthetic 驗證（DEFAULT_SYNTHETIC_PLAN）


@pytest.mark.parametrize(("task_id", "m"), CLIFF_CASES)
def test_sweep_localizes_cliff_at_mistake_turn(sweeps: _SweepCache, task_id: str, m: int) -> None:
    """在第 m 個 turn 犯錯的 source：真實曲線在 k = m 與 m + 1 之間掉到 0 → refine 後 k* = m 且標為 cliff。"""
    run = sweeps.get(task_id, m)
    mistake = mistake_call(run.task)
    assert checkpoint_truth(run.source, m, mistake, Q) > 0.8  # 前提：cliff 前的真值遠高於門檻
    assert checkpoint_truth(run.source, m + 1, mistake, Q) == 0.0
    result = run.result
    assert result.is_cliff is True
    assert result.largest_drop_k == m
    assert result.largest_drop >= DEFAULT_SYNTHETIC_PLAN.drop_threshold
    assert any(a <= m and m + 1 <= b for a, b in result.refined_intervals)
    by_k = {e.k: e for e in result.estimates}
    assert by_k[m].n == by_k[m + 1].n == DEFAULT_SYNTHETIC_PLAN.n_fine
    for e in result.estimates:
        if e.k > m:
            assert e.successes == 0  # 前綴已有不可逆錯誤：任何 rollout 都不可能成功


def test_sweep_cliff_in_interval_added_for_last_checkpoint(sweeps: _SweepCache) -> None:
    """m = n 且 K 為奇數：cliff 落在 ``∪ {K}`` 規則產生的最後一個 coarse 區間 (K − 1, K)。"""
    run = sweeps.get(ASSIGN, 2)
    assert len(run.source.turns) == 4  # 前提：[o0, o1, mistake, 結束] → K = 3
    assert coarse_checkpoints(valid_checkpoints(run.source)[-1], DEFAULT_SYNTHETIC_PLAN.coarse_stride) == [0, 2, 3]
    assert (2, 3) in run.result.refined_intervals
    assert run.result.largest_drop_k == 2
    assert run.result.is_cliff is True


@pytest.mark.parametrize(("task_id", "m"), NO_CLIFF_CASES)
def test_sweep_source_without_mistake_is_not_cliff(sweeps: _SweepCache, task_id: str, m: None) -> None:
    run = sweeps.get(task_id, m)
    assert _first_mistake(run.source, mistake_call(run.task)) is None
    assert run.result.is_cliff is False


def test_sweep_estimates_cover_analytic_truth(sweeps: _SweepCache) -> None:
    """decision 010 §5：所有 source 的所有 checkpoint 估計，Wilson 95% CI 涵蓋解析解的比例 >= 0.85。"""
    covered = 0
    total = 0
    positive = 0
    for task_id, m in ALL_CASES:
        run = sweeps.get(task_id, m)
        mistake = mistake_call(run.task)
        for e in run.result.estimates:
            truth = checkpoint_truth(run.source, e.k, mistake, Q)
            total += 1
            positive += truth > 0
            covered += e.ci_low - COVERAGE_EPS <= truth <= e.ci_high + COVERAGE_EPS
    assert positive >= 30  # 驗證不是只靠真值為 0 的點（結構上至少 50 個真值 > 0 的估計）
    assert covered / total >= 0.85, f"coverage {covered}/{total}"


def test_sweep_estimates_match_analytic_truth_at_high_power(sweeps: _SweepCache) -> None:
    """decision 010 §5 的估計正確性，提高檢定力：dev 5 題、沒有錯誤的 source、q = 0.2（真值約 0.17–0.8）、stride 1、
    每點 16 個 rollout。(a) 真值 > 0 的點中 Wilson 95% CI 的涵蓋率 >= 0.85；(b) p̂ 不偏（同一 checkpoint 的 rollout 給定
    checkpoint 下 i.i.d. Bernoulli(真值)）：|Σ(successes − n·truth)| <= 3·sqrt(Σ n·truth·(1 − truth))。
    已在本機確認：解析解的指數少算 1（(1 − q)^(n − r)）時 (a)、(b) 都失敗（涵蓋 13/21、合併偏差約 −7.6 SE）；多算 1 時 (b) 失敗
    （約 3.2 SE）；正確公式為涵蓋 21/21、約 −1.1 SE。"""
    covered = total = 0
    residual = variance = 0.0
    for task_id in DEV_TASK_IDS:
        run = sweeps.get(task_id, None, q=VALIDATION_Q, plan=VALIDATION_PLAN)
        mistake = mistake_call(run.task)
        assert [e.k for e in run.result.estimates] == valid_checkpoints(run.source)
        for e in run.result.estimates:
            truth = checkpoint_truth(run.source, e.k, mistake, VALIDATION_Q)
            assert 0 < truth < 1
            assert e.n == VALIDATION_PLAN.n_coarse
            total += 1
            covered += e.ci_low <= truth <= e.ci_high
            residual += e.successes - e.n * truth
            variance += e.n * truth * (1 - truth)
    assert total >= 20
    assert covered / total >= 0.85, f"coverage {covered}/{total}"
    assert abs(residual) <= 3 * math.sqrt(variance), f"pooled residual {residual:.2f}, SE {math.sqrt(variance):.2f}"


def test_sweep_noise_triggered_refine_is_rejected(tmp_path: Path) -> None:
    """沒有錯誤的 source（真值單調上升）：coarse 的雜訊讓某個區間被標記 → refine 到 n_fine → 不被標為 cliff。
    rollout seed offset 以決定性搜尋找第一個「coarse 有標記區間」者（搜尋條件是前提，不是被斷言的結果）。"""
    task, oracle = _task_and_oracle(FALSE_POSITIVE_TASK)
    source = _find_source(task, oracle, FALSE_POSITIVE_Q, None)
    mistake = mistake_call(task)
    plan = DEFAULT_SYNTHETIC_PLAN
    for offset in range(OFFSET_SEARCH_START, OFFSET_SEARCH_START + OFFSET_SEARCH_LIMIT):
        factory = _Factory(task, oracle, FALSE_POSITIVE_Q, seed_offset=offset)
        result = sweep(source, factory, plan=plan, log_path=tmp_path / f"offset-{offset}.jsonl", clock=_clock)
        if result.refined_intervals:
            break
    else:
        pytest.fail("no rollout seed offset in the search range makes the coarse phase flag an interval")
    for a, b in result.refined_intervals:
        assert checkpoint_truth(source, a, mistake, FALSE_POSITIVE_Q) < checkpoint_truth(source, b, mistake, FALSE_POSITIVE_Q)
    refined = {k for a, b in result.refined_intervals for k in range(a, b + 1)}
    by_k = {e.k: e for e in result.estimates}
    for k in refined:
        assert by_k[k].n == plan.n_fine
    assert result.is_cliff is False


def test_sweep_real_drop_below_threshold_is_not_cliff(sweeps: _SweepCache) -> None:
    """decision 010 §5 的 cliff 定位只在 cliff 前的真值明顯高於 drop_threshold 時成立：q = 0.25、n = 7、m = 0 時真實曲線從
    0.75^8 ≈ 0.10 掉到 0，下降小於門檻 → 依 §3 的判定規則不是 cliff（gradual）。"""
    run = sweeps.get(SMALL_DROP_TASK, 0, q=SMALL_DROP_Q)
    mistake = mistake_call(run.task)
    pre = checkpoint_truth(run.source, 0, mistake, SMALL_DROP_Q)
    assert pre == pytest.approx(0.75**8)
    assert pre < DEFAULT_SYNTHETIC_PLAN.drop_threshold
    assert checkpoint_truth(run.source, 1, mistake, SMALL_DROP_Q) == 0.0
    assert run.result.is_cliff is False
    assert (run.result.largest_drop, run.result.largest_drop_k, run.result.is_cliff) == detect_cliff(
        run.result.estimates, DEFAULT_SYNTHETIC_PLAN.drop_threshold
    )


@pytest.mark.parametrize(("task_id", "m"), ALL_CASES)
def test_sweep_result_is_consistent(sweeps: _SweepCache, task_id: str, m: int | None) -> None:
    run = sweeps.get(task_id, m)
    result, plan = run.result, DEFAULT_SYNTHETIC_PLAN
    assert result.task_id == task_id
    assert result.source_trajectory_id == run.source.trajectory_id
    assert result.plan == plan
    # estimates：依 k 遞增、只含有 rollout 的 checkpoint，且每個都是該 k 的 rollouts 的彙總
    ks = [e.k for e in result.estimates]
    assert ks == sorted(set(ks)) == sorted({r.k for r in result.rollouts})
    assert ks[0] == 0
    for e in result.estimates:
        assert e == estimate_checkpoint(e.k, [r for r in result.rollouts if r.k == e.k])
    # rollout_index 從 0 連續；synthetic policy 不會中止
    for e in result.estimates:
        assert sorted(r.rollout_index for r in result.rollouts if r.k == e.k) == list(range(e.n))
    for r in result.rollouts:
        assert r.attempts == 1 and r.outcome in ("success", "failure")
        assert r.termination in ("final_answer", "max_turns")
    # refine：coarse 相鄰點 (a, b) 中 coarse 階段 p̂(a) − p̂(b) >= drop_threshold 者，依 a 遞增
    coarse = coarse_checkpoints(valid_checkpoints(run.source)[-1], plan.coarse_stride)
    coarse_p = {
        k: sum(r.outcome == "success" for r in result.rollouts if r.k == k and r.rollout_index < plan.n_coarse) / plan.n_coarse
        for k in coarse
    }
    expected_intervals = tuple((a, b) for a, b in itertools.pairwise(coarse) if coarse_p[a] - coarse_p[b] >= plan.drop_threshold)
    assert result.refined_intervals == expected_intervals
    refined = {k for a, b in result.refined_intervals for k in range(a, b + 1)}
    for e in result.estimates:
        assert e.n == (plan.n_fine if e.k in refined else plan.n_coarse)
    assert (result.largest_drop, result.largest_drop_k, result.is_cliff) == detect_cliff(result.estimates, plan.drop_threshold)


@pytest.mark.parametrize(("task_id", "m"), ALL_CASES)
def test_sweep_round_robin_schedule(sweeps: _SweepCache, task_id: str, m: int | None) -> None:
    run = sweeps.get(task_id, m)
    if m is not None:
        assert run.result.refined_intervals  # cliff source 一定有 refine 階段
    assert [(r.k, r.rollout_index) for r in run.result.rollouts] == _expected_schedule(run.source, run.result)


@pytest.mark.parametrize(("task_id", "m"), ALL_CASES)
def test_sweep_manifest_has_one_completed_row_per_rollout(sweeps: _SweepCache, task_id: str, m: int | None) -> None:
    run = sweeps.get(task_id, m)
    records = read_manifest(run.log_path)
    rollouts = run.result.rollouts
    assert len(records) == len(rollouts)  # 沒有中止：每個 rollout 剛好一列
    cp_ids = {e.k: make_checkpoint(run.source, e.k).checkpoint_id for e in run.result.estimates}
    expected_keys = [RolloutKey(cp_ids[r.k], r.k, run.source.trajectory_id, r.rollout_index, 1) for r in rollouts]
    assert [_key_of(record) for record in records] == expected_keys  # manifest 依寫入順序 = rollouts 執行順序
    assert len(set(expected_keys)) == len(expected_keys)
    for record, rollout in zip(records, rollouts):
        assert record.outcome == "completed"
        assert record.task_id == task_id
        assert record.config_id == run.source.agent.config_id
        assert record.max_turns == run.source.max_turns
        assert record.termination == rollout.termination
        assert record.trajectory_id is not None


@pytest.mark.parametrize(("task_id", "m"), ALL_CASES)
def test_sweep_manifest_matches_independent_reruns(sweeps: _SweepCache, task_id: str, m: int | None) -> None:
    """manifest 記錄的就是被分類的那條 trajectory：對每個 checkpoint 的第一個與最後一個 rollout，以同一個 policy（同 seed）
    獨立 rerun，trajectory_id / termination 相符、classify_rollout 與 RolloutOutcome 相符、model_calls_completed 等於續跑的
    turn 數（不含前綴）。utc_time 為帶 UTC 時區的 ISO 8601。"""
    run = sweeps.get(task_id, m)
    records = read_manifest(run.log_path)
    by_key = {(r.checkpoint_turn, r.rollout_index): r for r in records}
    stored = {(r.k, r.rollout_index): r for r in run.result.rollouts}
    factory = _Factory(run.task, run.oracle, run.q)
    for e in run.result.estimates:
        checkpoint = make_checkpoint(run.source, e.k)
        for i in sorted({0, e.n - 1}):
            child = rerun(checkpoint, factory(e.k, i, 1), clock=_clock)
            record = by_key[(e.k, i)]
            assert record.trajectory_id == child.trajectory_id
            assert record.termination == child.termination
            assert record.model_calls_completed == len(child.turns) - e.k
            outcome = stored[(e.k, i)]
            assert classify_rollout(child, e.k) == (outcome.outcome, outcome.stopped_without_action)
    for record in records:
        stamp = datetime.fromisoformat(record.utc_time)
        assert stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0)
