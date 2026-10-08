"""experiments/analyze_sweep.py 的契約測試（decision 010 / 012 §3–4 / 013 / 015 / 016 B）。

契約 = 模組 docstring + decision 016 B：
- ``sweep_result_from_dict``：``asdict(SweepResult)`` 的 JSON 形式 → SweepResult（巢狀 dataclass 與 tuple 還原）；key 不符、型別錯、
  dataclass 驗證失敗 → ValueError。
- ``step_category``（B3）：forbidden_attempt → unnecessary_mutation → target_mutation → other_mutation → no_mutation，取第一個成立者，
  依 source.grade()；step 超出範圍 → ValueError。
- ``cliff_turn``（B3）：cliff 在 k* → k* + 1，對應 source.turns[k*]；模式 = 該 turn 每個 tool call 的 (tool, 類別)，去重、排序。
- ``assess_gate``（B4，decision 012 §4 提案規則）：同一模式出現在 >= 2 個不同 family 的 cliff 上 → met。
- ``analyze_sweep_runs`` / ``analyze_sweep_run``（B1、B2、B5、decision 017 §5）：一或多個 sweep 目錄（manifest、可沒有的
  summary.json、sweeps/*.json、rollouts/）+ baseline 的 source；每個完成的 sweep 以 manifest 與存檔的 rollout 重算核對；
  以 eval pool 判定缺漏；不一致 → ValueError。
- ``curve_svg``、``render_markdown``、``SweepAnalysis.to_dict``：只檢查結構、關鍵字串與決定性，不檢查排版。

真實的 run 目錄（不連網、不呼叫任何 LLM）：
- baseline：``run_split("eval", _BaselineClient(...))``。client 依 week1_pool 的順序逐一播放每個 episode 的劇本（新的嘗試 = request
  中沒有 assistant 訊息）：oracle、generator 的 negative_calls、自訂的「讀取後放棄」、refusal、max_tokens 截斷（agent_error）、
  兩次 APIConnectionError（infrastructure，重試一次後 aborted）。每個 thinking block 的 signature 為 ``src|<task_id>|<t>``，
  因此 sweep 的 request 可以由前綴辨認 source 與 checkpoint k。
- sweep：``run_sweeps(..., _SweepClient(...))``。k = request 中 signature 以 ``src|`` 開頭的 assistant 訊息數。前綴與該 task 的 oracle
  前綴相同 → 交給 OracleFakeClient 續跑（成功）；前綴已偏離 oracle → 立即結束（final answer），結果完全由前綴決定。
  因此「不可逆的偏離」之後的 checkpoint 全部失敗，形成設計好的 cliff（EXPECTED_CLIFFS）；另有一個 source 在 k = 2 依序注入
  agent_error、aborted（重試一次）、refusal、success（大下降但 Wilson CI 重疊 → 不是 cliff）。
- 預算停止：同一個 baseline、較小的 plan、``criterion="bonferroni95"``、``max_cost_usd`` 只夠第一個 source 完成。

設計說明：
- 所有 stub 只在測試函式內呼叫（fixture 只用已實作的 run_split / select_sources / run_sweeps），紅燈時每個測試各自失敗。
- 分析結果以 module 層級的 cache 共用（每個 run 目錄只分析一次）。
- 期望值一律在測試中獨立計算：類別由 ``source.grade()`` 的 forbidden_attempts / unnecessary_mutations / outcome_checks /
  constraint_checks 與 ``source.records()`` 的 changes 直接推得；bootstrap 由 ``stratified_task_bootstrap`` 重算。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agents.policy import ScriptedPolicy
from env.hashing import canonical_json
from experiments.analyze_sweep import (
    GATE_RULE,
    STEP_CATEGORIES,
    SWEEP_ANALYSIS_FORMAT,
    CliffTurn,
    GateAssessment,
    SourceCurve,
    SweepAnalysis,
    WindowEstimate,
    analyze_sweep_run,
    analyze_sweep_runs,
    assess_gate,
    cliff_turn,
    curve_svg,
    render_markdown,
    step_category,
    sweep_result_from_dict,
)
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import SweepRunSummary, run_sweeps, select_sources
from replay.aborts import read_manifest, utc_date
from replay.localization import RolloutOutcome, SweepPlan, SweepResult, estimate_checkpoint
from replay.runner import run_episode
from replay.stats import stratified_task_bootstrap
from replay.trajectory import Trajectory
from tasks.generator import week1_pool
from tests.fake_anthropic import FakeResponse, FakeUsage, text_block, thinking_block, tool_use_block

ZERO = lambda: 0.0  # noqa: E731
REPLICATES = 200
CALL_COST = (1000 * 4.00 + 200 * 20.00) / 1e6  # 每次 call 固定 input 1000 / output 200 tokens（claude-opus-5-5 價格）
PLAN = SweepPlan(coarse_stride=2, n_coarse=4, n_fine=4, drop_threshold=0.5)
BUDGET_PLAN = SweepPlan(coarse_stride=3, n_coarse=1, n_fine=1, drop_threshold=0.5)

POOL = week1_pool("eval")
BY_ID = {inst.task.task_id: inst for inst in POOL}
IDS = [inst.task.task_id for inst in POOL]

T_CLOSE_THEN_START = "gen-assign-and-start-s1"  # 最終成功但中途 status → closed（不安全）
T_ORACLE_AAS = "gen-assign-and-start-s2"  # oracle
T_TOUCH_OTHER = "gen-assign-and-start-s3"  # oracle + 改動其他 issue
T_TOUCH_ORIGINAL = "gen-close-duplicate-s1"  # oracle + 改動原 issue（forbidden）
T_AGENT_ERROR = "gen-close-duplicate-s2"  # rollout 0 為 agent_error → 缺漏
T_LABEL_OVERWRITE = "gen-close-duplicate-s3"  # 覆寫 labels
T_OPEN_ONLY = "gen-escalate-label-s1"  # 只處理 open 的 target（偏離 oracle 順序）
T_READ_THEN_STOP = "gen-escalate-label-s2"  # 自訂：search + 讀取後放棄
T_REFUSAL = "gen-escalate-label-s3"  # 第一個 turn 就 refusal（完成、1 個 turn）
T_ABORTED = "gen-reassign-departed-s1"  # 兩次 APIConnectionError → aborted → 缺漏
T_NOISY = "gen-reassign-departed-s2"  # oracle；sweep 在 k = 2 注入 agent_error / aborted / refusal
T_RECOVERABLE = "gen-reassign-departed-s3"  # open-only = oracle 前綴：失敗但一直可恢復
T_EXTRA_COMMENT = "gen-conditional-assign-s1"  # oracle + 多餘留言（不可逆）
T_BLIND_ASSIGN = "gen-conditional-assign-s2"  # noop branch 卻指派（forbidden）
T_ORACLE_CA = "gen-conditional-assign-s3"  # oracle

MISSING = (T_AGENT_ERROR, T_ABORTED)
COMPLETED = tuple(t for t in IDS if t not in MISSING)

# 設計好的 cliff：task → (k*, source.turns[k*] 的模式)。每個都是「前綴偏離 oracle 後無法恢復」的位置。
EXPECTED_CLIFFS: dict[str, tuple[int, tuple[tuple[str, str], ...]]] = {
    T_CLOSE_THEN_START: (1, (("update_issue", "forbidden_attempt"),)),
    T_TOUCH_OTHER: (2, (("update_issue", "unnecessary_mutation"),)),
    T_TOUCH_ORIGINAL: (3, (("update_issue", "forbidden_attempt"),)),
    T_LABEL_OVERWRITE: (0, (("update_issue", "target_mutation"),)),
    T_OPEN_ONLY: (1, (("update_issue", "other_mutation"),)),
    T_READ_THEN_STOP: (1, (("get_issue", "no_mutation"),)),
    T_EXTRA_COMMENT: (2, (("add_comment", "unnecessary_mutation"),)),
    T_BLIND_ASSIGN: (1, (("assign_issue", "forbidden_attempt"),)),
}

MAX_TOKENS = "max_tokens"
REFUSAL = "refusal"
CONN_ERROR = "conn_error"
ORACLE = "oracle"


# ---------------------------------------------------------------- fake clients


class _APIConnectionError(Exception):
    """模擬 anthropic.APIConnectionError：replay.aborts 依類別模組與名稱分類為 infrastructure（重試一次）。"""


_APIConnectionError.__module__ = "anthropic"
_APIConnectionError.__name__ = "APIConnectionError"
_APIConnectionError.__qualname__ = "APIConnectionError"


def _usage() -> FakeUsage:
    return FakeUsage(input_tokens=1000, output_tokens=200)


def _respond(action: Any, signature: str, t: int) -> FakeResponse:
    """action：MAX_TOKENS / REFUSAL / CONN_ERROR，或依序執行的 calls（t < len → 第 t 個 call，否則 final answer）。"""
    if action == CONN_ERROR:
        raise _APIConnectionError("connection reset by peer")
    if action == MAX_TOKENS:
        return FakeResponse(content=[thinking_block(signature), text_block("Let me")], stop_reason="max_tokens", usage=_usage())
    if action == REFUSAL:
        return FakeResponse(
            content=[thinking_block(signature), text_block("I can't help with that.")], stop_reason="refusal", usage=_usage()
        )
    if t < len(action):
        name, args = action[t]
        return FakeResponse(
            content=[thinking_block(signature), tool_use_block(f"toolu_{signature}", name, copy.deepcopy(args))],
            stop_reason="tool_use",
            usage=_usage(),
        )
    return FakeResponse(content=[thinking_block(signature), text_block("Done.")], stop_reason="end_turn", usage=_usage())


class _BaselineClient:
    """依 schedule（每次嘗試一個 (instance, action)）播放 baseline episode；signature = ``src|<task_id>|<t>``。"""

    def __init__(self, schedule: list[tuple[Any, Any]]) -> None:
        self._schedule = list(schedule)
        self._current: tuple[Any, Any] | None = None
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> FakeResponse:
        messages = request["messages"]
        t = sum(1 for message in messages if message["role"] == "assistant")
        if t == 0:
            self._current = self._schedule.pop(0)
            assert messages[0]["content"] == self._current[0].task.instruction, "baseline schedule out of order"
        assert self._current is not None
        instance, action = self._current
        return _respond(action, f"src|{instance.task.task_id}|{t}", t)


def _signature(message: dict[str, Any]) -> str:
    for block in message["content"]:
        if block["type"] == "thinking":
            return block["signature"]
    return ""


class _SweepClient:
    """sweep 用：k = signature 以 ``src|`` 開頭的 assistant 訊息數。新的嘗試（沒有續跑的 assistant 訊息）時：
    overrides[(task_id, k)] 依序給出每次嘗試的 action（ORACLE = 交給 OracleFakeClient）；沒有 override 時，前綴與 oracle 前綴
    相同 → OracleFakeClient，否則立即 final answer。k = 0 與續跑中的 request 一律交給 OracleFakeClient。"""

    def __init__(self, overrides: dict[tuple[str, int], list[Any]] | None = None) -> None:
        self._inner = OracleFakeClient(POOL)
        self._overrides = {key: list(actions) for key, actions in (overrides or {}).items()}
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> FakeResponse:
        assistants = [message for message in request["messages"] if message["role"] == "assistant"]
        prefix = [message for message in assistants if _signature(message).startswith("src|")]
        k = len(prefix)
        if k and len(assistants) == k:
            task_id = _signature(prefix[0]).split("|")[1]
            signature = f"rr|{task_id}|{k}"
            actions = self._overrides.get((task_id, k))
            if actions is not None:
                action = actions.pop(0)
                if action != ORACLE:
                    return _respond(action, signature, 0)
            else:
                calls = [[b["name"], b["input"]] for m in prefix for b in m["content"] if b["type"] == "tool_use"]
                oracle = [list(call) for call in BY_ID[task_id].oracle_calls[:k]]
                if canonical_json(calls) != canonical_json(oracle):
                    return _respond((), signature, 0)
        return self._inner.create(**request)


def _negative(task_id: str, name: str) -> tuple[tuple[str, Any], ...]:
    return dict(BY_ID[task_id].negative_calls)[name]


def _baseline_schedule() -> list[tuple[Any, Any]]:
    oracle = {t: BY_ID[t].oracle_calls for t in IDS}
    read_then_stop = (oracle[T_READ_THEN_STOP][0], ("get_issue", {"issue_id": oracle[T_READ_THEN_STOP][1][1]["issue_id"]}))
    plan: dict[str, list[Any]] = {
        T_CLOSE_THEN_START: [_negative(T_CLOSE_THEN_START, "close-then-start")],
        T_ORACLE_AAS: [oracle[T_ORACLE_AAS]],
        T_TOUCH_OTHER: [_negative(T_TOUCH_OTHER, "touch-other")],
        T_TOUCH_ORIGINAL: [_negative(T_TOUCH_ORIGINAL, "touch-original")],
        T_AGENT_ERROR: [MAX_TOKENS],
        T_LABEL_OVERWRITE: [_negative(T_LABEL_OVERWRITE, "label-overwrite")],
        T_OPEN_ONLY: [_negative(T_OPEN_ONLY, "open-only")],
        T_READ_THEN_STOP: [read_then_stop],
        T_REFUSAL: [REFUSAL],
        T_ABORTED: [CONN_ERROR, CONN_ERROR],
        T_NOISY: [oracle[T_NOISY]],
        T_RECOVERABLE: [_negative(T_RECOVERABLE, "open-only")],
        T_EXTRA_COMMENT: [_negative(T_EXTRA_COMMENT, "extra-comment")],
        T_BLIND_ASSIGN: [_negative(T_BLIND_ASSIGN, "blind-assign")],
        T_ORACLE_CA: [oracle[T_ORACLE_CA]],
    }
    return [(BY_ID[t], action) for t in IDS for action in plan[t]]


SWEEP_OVERRIDES: dict[tuple[str, int], list[Any]] = {
    # rollout 0：agent_error；rollout 1：aborted（兩次嘗試）；rollout 2：refusal；rollout 3：success
    (T_NOISY, 2): [MAX_TOKENS, CONN_ERROR, CONN_ERROR, REFUSAL, ORACLE],
}


# ---------------------------------------------------------------- fixtures


@dataclasses.dataclass(frozen=True)
class _Runs:
    baseline: Path
    sweep: Path
    summary: SweepRunSummary
    sources: dict[str, Trajectory | None]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> _Runs:
    root = tmp_path_factory.mktemp("analyze-sweep")
    baseline = root / "baseline"
    run_split("eval", _BaselineClient(_baseline_schedule()), run_dir=baseline, clock=ZERO)
    sources = dict(select_sources(baseline))
    sweep_dir = root / "sweep"
    summary = run_sweeps(
        sources, _SweepClient(SWEEP_OVERRIDES), plan=PLAN, criterion="wilson95", run_dir=sweep_dir, clock=ZERO,
        replicates=REPLICATES,
    )
    return _Runs(baseline, sweep_dir, summary, sources)


@pytest.fixture(scope="module")
def budget_runs(runs: _Runs, tmp_path_factory: pytest.TempPathFactory) -> _Runs:
    sweep_dir = tmp_path_factory.mktemp("analyze-sweep-budget") / "sweep"
    summary = run_sweeps(
        runs.sources, _SweepClient(), plan=BUDGET_PLAN, criterion="bonferroni95", run_dir=sweep_dir,
        max_cost_usd=5.5 * CALL_COST, clock=ZERO, replicates=REPLICATES,
    )
    return _Runs(runs.baseline, sweep_dir, summary, runs.sources)


@pytest.fixture(scope="module")
def cache() -> dict[Path, SweepAnalysis]:
    return {}


def _analysis(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> SweepAnalysis:
    if runs.sweep not in cache:
        cache[runs.sweep] = analyze_sweep_run(runs.sweep, runs.baseline, replicates=REPLICATES)
    return cache[runs.sweep]


# ---------------------------------------------------------------- 獨立計算的期望值


def _expected_category(source: Trajectory, step: int) -> str:
    """decision 016 B3，直接由 grade report 與 records 計算。"""
    report = source.grade()
    record = source.records()[step]
    if any(attempt.step == step for attempt in report.forbidden_attempts):
        return "forbidden_attempt"
    if any(mutation.step == step for mutation in report.unnecessary_mutations):
        return "unnecessary_mutation"
    if record.changes:
        failing = {c.predicate["issue_id"] for c in (*report.outcome_checks, *report.constraint_checks) if not c.passed}
        if any(change.issue_id in failing for change in record.changes):
            return "target_mutation"
        return "other_mutation"
    return "no_mutation"


def _expected_pattern(source: Trajectory, k: int) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({(step.tool, _expected_category(source, step.step)) for step in source.turns[k].tool_steps}))


def _pattern_key(pattern: tuple[tuple[str, str], ...]) -> str:
    return "+".join(f"{tool}:{category}" for tool, category in pattern)


def _episode(task_id: str, script: list[list[tuple[str, Any]]]) -> Trajectory:
    return run_episode(BY_ID[task_id].task, ScriptedPolicy(script), max_turns=20, clock=ZERO)


def _turns(calls: Any) -> list[list[tuple[str, Any]]]:
    return [[call] for call in calls]


def _outcomes(k: int, labels: list[str]) -> list[RolloutOutcome]:
    out = []
    for i, label in enumerate(labels):
        completed = label in ("success", "failure", "refusal")
        termination = ("max_turns" if label == "failure" and i % 2 else "final_answer") if completed else None
        out.append(RolloutOutcome(k, i, label, termination, label == "failure", 2 if label == "aborted" else 1))
    return out


def _hand_result(
    task_id: str = "t-hand", source_id: str = "sha256:" + "a" * 64, *, k: int = 1, is_cliff: bool = True,
    criterion: str = "wilson95",
) -> SweepResult:
    """k → k + 1 的 4/4 → 0/4 下降。"""
    before = _outcomes(k, ["success"] * 4)
    after = _outcomes(k + 1, ["failure"] * 4)
    estimates = (estimate_checkpoint(k, before), estimate_checkpoint(k + 1, after))
    return SweepResult(
        task_id=task_id,
        source_trajectory_id=source_id,
        plan=SweepPlan(1, 4, 4, 0.5),
        estimates=estimates,
        rollouts=tuple(before + after),
        refined_intervals=(),
        largest_drop=1.0,
        largest_drop_k=k,
        is_cliff=is_cliff,
        criterion=criterion,
    )


def _varied_result() -> SweepResult:
    """各種 outcome、p̂_completed 為 None、refine 區間非空、bonferroni95。"""
    first = _outcomes(0, ["success", "success", "failure", "agent_error"])
    second = _outcomes(1, ["agent_error", "aborted", "refusal", "refusal"])
    third = _outcomes(3, ["success", "failure", "success", "failure", "success", "failure", "success", "failure"])
    estimates = (estimate_checkpoint(0, first), estimate_checkpoint(1, second), estimate_checkpoint(3, third))
    assert estimates[1].p_hat_completed is None
    return SweepResult(
        task_id="gen-escalate-label-s9",
        source_trajectory_id="sha256:" + "b" * 64,
        plan=SweepPlan(1, 4, 8, 0.25),
        estimates=estimates,
        rollouts=tuple(first + second + third),
        refined_intervals=((0, 1), (1, 3)),
        largest_drop=0.5,
        largest_drop_k=0,
        is_cliff=False,
        criterion="bonferroni95",
    )


def _empty_result() -> SweepResult:
    return SweepResult(
        task_id="gen-empty",
        source_trajectory_id="sha256:" + "c" * 64,
        plan=SweepPlan(2, 1, 1, 1.0),
        estimates=(),
        rollouts=(),
        refined_intervals=(),
        largest_drop=0.0,
        largest_drop_k=None,
        is_cliff=False,
    )


def _jsonable(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _curve(task_id: str, family: str, pattern: tuple[tuple[str, str], ...] | None) -> SourceCurve:
    result = _hand_result(task_id, k=1, is_cliff=pattern is not None)
    return SourceCurve(
        task_id=task_id,
        family=family,
        source_trajectory_id=result.source_trajectory_id,
        source_success=False,
        source_safe=True,
        n_turns=4,
        result=result,
        cliff=None if pattern is None else CliffTurn(1, pattern),
    )


# ---------------------------------------------------------------- fixture 前提（測試設計本身的檢查）


def test_fixture_premises(runs: _Runs, budget_runs: _Runs) -> None:
    summary = runs.summary
    assert summary.missing_sources == MISSING and summary.completed == COMPLETED
    assert summary.stopped_by_budget is False and summary.cliff_rate is not None
    assert {t for t in COMPLETED if summary.results[t].is_cliff} == set(EXPECTED_CLIFFS)
    for task_id, (k, _) in EXPECTED_CLIFFS.items():
        assert summary.results[task_id].largest_drop_k == k
    noisy = summary.results[T_NOISY]
    at_two = [r for r in noisy.rollouts if r.k == 2]
    assert [(r.outcome, r.attempts) for r in at_two] == [("agent_error", 1), ("aborted", 2), ("refusal", 1), ("success", 1)]
    assert noisy.largest_drop == 0.75 and noisy.is_cliff is False and noisy.refined_intervals == ((0, 2),)
    assert runs.sources[T_REFUSAL] is not None and len(runs.sources[T_REFUSAL].turns) == 1
    budget = budget_runs.summary
    assert budget.stopped_by_budget is True and budget.completed == (COMPLETED[0],) and budget.cliff_rate is None


# ---------------------------------------------------------------- sweep_result_from_dict


def test_sweep_result_from_dict_round_trips_every_sweep_file(runs: _Runs) -> None:
    written = json.loads((runs.sweep / "summary.json").read_text(encoding="utf-8"))
    for task_id in runs.summary.completed:
        data = json.loads((runs.sweep / "sweeps" / f"{task_id}.json").read_text(encoding="utf-8"))
        result = sweep_result_from_dict(data)
        assert result == runs.summary.results[task_id]
        assert isinstance(result.plan, SweepPlan)
        assert isinstance(result.estimates, tuple) and isinstance(result.rollouts, tuple)
        assert all(isinstance(r, RolloutOutcome) for r in result.rollouts)
        assert sweep_result_from_dict(written["results"][task_id]) == runs.summary.results[task_id]


@pytest.mark.parametrize("make", [_varied_result, _empty_result, lambda: _hand_result(criterion="bonferroni95")])
def test_sweep_result_from_dict_round_trip_guarantee(make: Callable[[], SweepResult]) -> None:
    result = make()
    data = json.loads(canonical_json(dataclasses.asdict(result)))
    rebuilt = sweep_result_from_dict(data)
    assert rebuilt == result
    assert type(rebuilt.estimates) is tuple and type(rebuilt.rollouts) is tuple
    assert type(rebuilt.refined_intervals) is tuple and all(type(item) is tuple for item in rebuilt.refined_intervals)
    assert data == json.loads(canonical_json(dataclasses.asdict(result)))  # 不修改輸入


def _drop(data: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k != key}


MALFORMED: dict[str, Callable[[dict[str, Any]], Any]] = {
    "not-an-object": lambda d: [d],
    "null": lambda d: None,
    "missing-key": lambda d: _drop(d, "task_id"),
    "extra-key": lambda d: {**d, "note": "x"},
    "plan-not-an-object": lambda d: {**d, "plan": [1, 4, 8, 0.25]},
    "plan-missing-key": lambda d: {**d, "plan": _drop(d["plan"], "n_fine")},
    "plan-extra-key": lambda d: {**d, "plan": {**d["plan"], "seed": 0}},
    "plan-invalid": lambda d: {**d, "plan": {**d["plan"], "n_fine": 2}},  # n_fine < n_coarse
    "estimates-not-a-list": lambda d: {**d, "estimates": {}},
    "estimate-not-an-object": lambda d: {**d, "estimates": [1, *d["estimates"][1:]]},
    "estimate-missing-key": lambda d: {**d, "estimates": [_drop(d["estimates"][0], "ci_low"), *d["estimates"][1:]]},
    "estimate-extra-key": lambda d: {**d, "estimates": [{**d["estimates"][0], "x": 1}, *d["estimates"][1:]]},
    "estimate-n-type": lambda d: {**d, "estimates": [{**d["estimates"][0], "n": "4"}, *d["estimates"][1:]]},
    "rollouts-not-a-list": lambda d: {**d, "rollouts": "none"},
    "rollout-missing-key": lambda d: {**d, "rollouts": [_drop(d["rollouts"][0], "attempts"), *d["rollouts"][1:]]},
    "rollout-extra-key": lambda d: {**d, "rollouts": [{**d["rollouts"][0], "x": 1}, *d["rollouts"][1:]]},
    "interval-wrong-length": lambda d: {**d, "refined_intervals": [[0, 1, 3]]},
    "interval-not-a-list": lambda d: {**d, "refined_intervals": [0]},
    "task-id-type": lambda d: {**d, "task_id": 5},
    "is-cliff-type": lambda d: {**d, "is_cliff": "false"},
    "largest-drop-type": lambda d: {**d, "largest_drop": "0.5"},
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_sweep_result_from_dict_rejects_malformed_data(name: str) -> None:
    data = _jsonable(dataclasses.asdict(_varied_result()))
    with pytest.raises(ValueError):
        sweep_result_from_dict(MALFORMED[name](data))


# ---------------------------------------------------------------- step_category


def test_step_category_matches_the_grade_report_on_every_source_step(runs: _Runs) -> None:
    seen: set[str] = set()
    for task_id in COMPLETED:
        source = runs.sources[task_id]
        assert source is not None
        for step in range(len(source.records())):
            category = step_category(source, step)
            assert category in STEP_CATEGORIES
            assert category == _expected_category(source, step), (task_id, step)
            seen.add(category)
    assert seen == set(STEP_CATEGORIES)  # fixture 涵蓋全部五個類別


def test_step_category_precedence_and_edge_cases() -> None:
    # forbidden attempt 即使 tool 拒絕、沒有任何 change 也算（close-duplicate：原 issue 的任何變更都 forbidden）
    o = _negative(T_TOUCH_ORIGINAL, "touch-original")[-1][1]["issue_id"]
    rejected = _episode(T_TOUCH_ORIGINAL, [[("assign_issue", {"issue_id": o, "assignee_id": "nobody"})]])
    assert rejected.records()[0].ok is False and rejected.records()[0].changes == ()
    assert step_category(rejected, 0) == "forbidden_attempt"

    # forbidden 且 unnecessary → forbidden_attempt（順序第一）
    touch_original = _episode(T_TOUCH_ORIGINAL, _turns(_negative(T_TOUCH_ORIGINAL, "touch-original")))
    report = touch_original.grade()
    assert any(m.step == 3 for m in report.unnecessary_mutations)
    assert [step_category(touch_original, s) for s in range(4)] == [
        "no_mutation", "other_mutation", "other_mutation", "forbidden_attempt"
    ]

    # unnecessary 且變更的 issue 在未通過的 constraint 中 → unnecessary_mutation；同一 trajectory 中改 target 的步驟 → other_mutation
    touch_other = _episode(T_TOUCH_OTHER, _turns(_negative(T_TOUCH_OTHER, "touch-other")))
    failing = {c.predicate["issue_id"] for c in touch_other.grade().constraint_checks if not c.passed}
    assert _negative(T_TOUCH_OTHER, "touch-other")[2][1]["issue_id"] in failing
    assert [step_category(touch_other, s) for s in range(3)] == ["other_mutation", "other_mutation", "unnecessary_mutation"]

    # target_mutation：outcome check 未通過（指派給錯的人）
    task = BY_ID[T_ORACLE_AAS].task
    target, right = BY_ID[T_ORACLE_AAS].oracle_calls[0][1]["issue_id"], BY_ID[T_ORACLE_AAS].oracle_calls[0][1]["assignee_id"]
    wrong = next(u["id"] for u in task.initial_state["users"] if u["active"] and u["id"] not in ("bot", right))
    misassigned = _episode(T_ORACLE_AAS, [[("assign_issue", {"issue_id": target, "assignee_id": wrong})]])
    assert misassigned.grade().trajectory_safe and not misassigned.grade().unnecessary_mutations
    assert step_category(misassigned, 0) == "target_mutation"

    # target_mutation：constraint check 未通過（labels 被覆寫）；之後在同一 issue 的留言也算（只看 issue，不看欄位）
    overwrite = _episode(T_LABEL_OVERWRITE, _turns(_negative(T_LABEL_OVERWRITE, "label-overwrite")))
    assert [step_category(overwrite, s) for s in range(2)] == ["target_mutation", "target_mutation"]

    # 成功的 oracle：讀取 → no_mutation、變更 → other_mutation；失敗的讀取（tool error）→ no_mutation
    oracle = _episode(T_ORACLE_CA, _turns(BY_ID[T_ORACLE_CA].oracle_calls) + [[("get_issue", {"issue_id": 9999})]])
    assert oracle.records()[2].ok is False
    assert [step_category(oracle, s) for s in range(3)] == ["no_mutation", "other_mutation", "no_mutation"]


def test_step_category_uses_global_step_numbers_across_multi_call_turns() -> None:
    calls = _negative(T_TOUCH_ORIGINAL, "touch-original")  # get, update(d), comment(d), update(o, closed)
    source = _episode(T_TOUCH_ORIGINAL, [[calls[3], calls[0]], [calls[1], calls[2]]])
    assert [step.step for step in source.turns[0].tool_steps] == [0, 1]
    assert [step_category(source, s) for s in range(4)] == [
        "forbidden_attempt", "no_mutation", "other_mutation", "other_mutation"
    ]


@pytest.mark.parametrize("where", ["negative", "total", "beyond"])
def test_step_category_rejects_steps_out_of_range(where: str) -> None:
    source = _episode(T_ORACLE_CA, _turns(BY_ID[T_ORACLE_CA].oracle_calls))
    total = len(source.records())
    assert total == 2
    step = {"negative": -1, "total": total, "beyond": total + 3}[where]
    with pytest.raises(ValueError):
        step_category(source, step)


def test_step_category_rejects_any_step_of_a_trajectory_without_tool_steps(runs: _Runs) -> None:
    refusal = runs.sources[T_REFUSAL]
    assert refusal is not None and refusal.records() == ()
    with pytest.raises(ValueError):
        step_category(refusal, 0)


# ---------------------------------------------------------------- cliff_turn


def test_cliff_turn_matches_the_independent_classification(runs: _Runs) -> None:
    for task_id in COMPLETED:
        source, result = runs.sources[task_id], runs.summary.results[task_id]
        assert source is not None
        turn = cliff_turn(source, result)
        if not result.is_cliff:
            assert turn is None, task_id
            continue
        k = result.largest_drop_k
        assert isinstance(turn, CliffTurn)
        assert turn == CliffTurn(k, _expected_pattern(source, k)), task_id


def test_cliff_turn_designed_patterns(runs: _Runs) -> None:
    observed = {}
    for task_id in COMPLETED:
        source = runs.sources[task_id]
        assert source is not None
        turn = cliff_turn(source, runs.summary.results[task_id])
        if turn is not None:
            observed[task_id] = (turn.k, turn.pattern)
    assert observed == EXPECTED_CLIFFS


def test_cliff_turn_dedupes_and_sorts_the_turn_pattern() -> None:
    calls = _negative(T_TOUCH_ORIGINAL, "touch-original")
    d = calls[0][1]["issue_id"]
    o = calls[3][1]["issue_id"]
    script = [
        [calls[0]],
        # 同一 turn、依「排序後的反序」出現：4 種 (tool, 類別)，其中兩種重複出現
        [
            ("update_issue", {"issue_id": o, "status": "closed"}),  # forbidden_attempt
            ("search_issues", {"label": "docs"}),  # no_mutation
            ("get_issue", {"issue_id": d}),  # no_mutation
            ("add_comment", {"issue_id": d, "body": calls[2][1]["body"]}),  # other_mutation（d 的檢查最終全部通過）
            ("get_issue", {"issue_id": o}),  # 重複
            ("update_issue", {"issue_id": o, "priority": "p0"}),  # 重複
        ],
        [calls[1]],
    ]
    source = _episode(T_TOUCH_ORIGINAL, script)
    result = _hand_result(T_TOUCH_ORIGINAL, source.trajectory_id, k=1)
    expected = (
        ("add_comment", "other_mutation"),
        ("get_issue", "no_mutation"),
        ("search_issues", "no_mutation"),
        ("update_issue", "forbidden_attempt"),
    )
    assert _expected_pattern(source, 1) == expected
    assert cliff_turn(source, result) == CliffTurn(1, expected)
    first = _hand_result(T_TOUCH_ORIGINAL, source.trajectory_id, k=0)
    assert cliff_turn(source, first) == CliffTurn(0, (("get_issue", "no_mutation"),))


def test_cliff_turn_is_none_without_a_cliff(runs: _Runs) -> None:
    source = runs.sources[T_EXTRA_COMMENT]
    assert source is not None
    result = dataclasses.replace(runs.summary.results[T_EXTRA_COMMENT], is_cliff=False)
    assert cliff_turn(source, result) is None
    oracle = runs.sources[T_ORACLE_CA]
    assert oracle is not None and cliff_turn(oracle, runs.summary.results[T_ORACLE_CA]) is None


@pytest.mark.parametrize("change", ["other-source", "source-id", "k-out-of-range", "k-far-out-of-range"])
def test_cliff_turn_rejects_results_of_another_source(change: str, runs: _Runs) -> None:
    source = runs.sources[T_EXTRA_COMMENT]
    assert source is not None
    result = runs.summary.results[T_EXTRA_COMMENT]
    if change == "other-source":
        result = runs.summary.results[T_BLIND_ASSIGN]
    elif change == "source-id":
        result = dataclasses.replace(result, source_trajectory_id=runs.summary.results[T_BLIND_ASSIGN].source_trajectory_id)
    elif change == "k-out-of-range":
        result = dataclasses.replace(result, largest_drop_k=len(source.turns))
    else:
        result = dataclasses.replace(result, largest_drop_k=len(source.turns) + 5)
    assert result.is_cliff is True
    with pytest.raises(ValueError):
        cliff_turn(source, result)


# ---------------------------------------------------------------- assess_gate（手造的 SourceCurve）


A = (("update_issue", "forbidden_attempt"),)
B = (("add_comment", "unnecessary_mutation"),)


def test_assess_gate_met_when_a_pattern_spans_two_families() -> None:
    curves = (
        _curve("t-c", "gen/close-duplicate", A),
        _curve("t-a", "gen/assign-and-start", A),
        _curve("t-b", "gen/assign-and-start", B),
        _curve("t-d", "gen/escalate-label", None),
    )
    gate = assess_gate(curves)
    assert isinstance(gate, GateAssessment)
    assert gate.rule == GATE_RULE
    assert gate.patterns == {"update_issue:forbidden_attempt": ("t-a", "t-c"), "add_comment:unnecessary_mutation": ("t-b",)}
    assert gate.families == {
        "update_issue:forbidden_attempt": ("gen/assign-and-start", "gen/close-duplicate"),
        "add_comment:unnecessary_mutation": ("gen/assign-and-start",),
    }
    assert gate.met is True


def test_assess_gate_not_met_when_patterns_differ_across_families() -> None:
    curves = (_curve("t-1", "gen/assign-and-start", A), _curve("t-2", "gen/close-duplicate", B))
    gate = assess_gate(curves)
    assert gate.met is False
    assert gate.patterns == {"update_issue:forbidden_attempt": ("t-1",), "add_comment:unnecessary_mutation": ("t-2",)}


def test_assess_gate_not_met_when_a_pattern_repeats_within_one_family() -> None:
    curves = (
        _curve("t-3", "gen/reassign-departed", A),
        _curve("t-1", "gen/reassign-departed", A),
        _curve("t-2", "gen/reassign-departed", A),
        _curve("t-4", "gen/escalate-label", None),  # 沒有 cliff 的 source 不計入
    )
    gate = assess_gate(curves)
    assert gate.met is False
    assert gate.patterns == {"update_issue:forbidden_attempt": ("t-1", "t-2", "t-3")}
    assert gate.families == {"update_issue:forbidden_attempt": ("gen/reassign-departed",)}


def test_assess_gate_pattern_strings_and_empty_input() -> None:
    multi = (("add_comment", "other_mutation"), ("assign_issue", "target_mutation"))
    curves = (
        _curve("t-1", "gen/reassign-departed", multi),
        _curve("t-2", "gen/assign-and-start", ()),
        _curve("t-3", "gen/close-duplicate", ()),
    )
    gate = assess_gate(curves)
    assert gate.patterns == {"add_comment:other_mutation+assign_issue:target_mutation": ("t-1",), "": ("t-2", "t-3")}
    assert gate.met is True  # 空 turn 的模式 "" 也是一種模式
    for empty in ((), (_curve("t-9", "gen/escalate-label", None),)):
        gate = assess_gate(empty)
        assert gate.patterns == {} and gate.families == {} and gate.met is False and gate.rule == GATE_RULE


# ---------------------------------------------------------------- analyze_sweep_run


def test_analysis_has_one_curve_per_completed_source(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    summary = runs.summary
    assert isinstance(analysis, SweepAnalysis)
    assert analysis.plan == PLAN and analysis.criterion == "wilson95"
    assert tuple(curve.task_id for curve in analysis.curves) == summary.completed == COMPLETED
    assert analysis.missing_sources == MISSING
    assert analysis.not_completed == () and analysis.partial_sweeps == ()
    assert analysis.runs == ("sweep",) and analysis.interrupted_runs == () and analysis.stopped_by_budget == ()
    assert analysis.resumed == () and analysis.multi_tool_turns == () and analysis.source_multi_tool_turns == ()
    rows = read_manifest(runs.sweep / "manifest.jsonl")
    assert analysis.cost_usd == pytest.approx(sum(row.cost_usd for row in rows)) == pytest.approx(summary.spent_usd)
    assert analysis.unknown_cost_calls == 0 and analysis.pooling_blocked is None and analysis.max_turns_confounded == ()
    for curve in analysis.curves:
        source = runs.sources[curve.task_id]
        assert source is not None
        report = source.grade()
        assert isinstance(curve, SourceCurve)
        assert curve.family == source.task["family"] == summary.families[curve.task_id]
        assert curve.source_trajectory_id == source.trajectory_id
        assert curve.source_success is report.success and curve.source_safe is report.trajectory_safe
        assert curve.n_turns == len(source.turns)
        assert curve.result == summary.results[curve.task_id]
        k = curve.result.largest_drop_k
        assert curve.cliff == (CliffTurn(k, _expected_pattern(source, k)) if curve.result.is_cliff else None)
    by_id = {curve.task_id: curve for curve in analysis.curves}
    assert (by_id[T_CLOSE_THEN_START].source_success, by_id[T_CLOSE_THEN_START].source_safe) == (True, False)
    assert (by_id[T_ORACLE_AAS].source_success, by_id[T_ORACLE_AAS].source_safe) == (True, True)
    assert (by_id[T_BLIND_ASSIGN].source_success, by_id[T_BLIND_ASSIGN].source_safe) == (False, False)
    assert (by_id[T_RECOVERABLE].source_success, by_id[T_RECOVERABLE].cliff) == (False, None)
    assert (by_id[T_REFUSAL].source_success, by_id[T_REFUSAL].n_turns, by_id[T_REFUSAL].cliff) == (False, 1, None)
    assert by_id[T_NOISY].cliff is None and by_id[T_NOISY].result.largest_drop == 0.75


def test_analysis_curve_points_for_a_cliff_source(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    curve = next(c for c in _analysis(runs, cache).curves if c.task_id == T_EXTRA_COMMENT)
    points = [(e.k, e.n, e.successes, e.failures, e.stopped_without_action, e.p_hat) for e in curve.result.estimates]
    # 4 個 turn（get、assign、多餘留言、final）：k = 0、2、3（stride 2 + last）；k = 3 的前綴已含留言 → 立即結束、全部失敗
    assert points == [(0, 4, 4, 0, 0, 1.0), (2, 4, 4, 0, 0, 1.0), (3, 4, 0, 4, 4, 0.0)]
    assert curve.cliff == CliffTurn(2, (("add_comment", "unnecessary_mutation"),))


def test_analysis_aggregates_reuse_the_run_sweep_bootstrap(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    summary = runs.summary
    strata = {t: summary.families[t] for t in COMPLETED}
    cliff_rate = stratified_task_bootstrap(
        {t: float(summary.results[t].is_cliff) for t in COMPLETED}, strata, replicates=REPLICATES, seed=0
    )
    mean_drop = stratified_task_bootstrap(
        {t: summary.results[t].largest_drop for t in COMPLETED}, strata, replicates=REPLICATES, seed=0
    )
    assert analysis.cliff_rate == cliff_rate == summary.cliff_rate
    assert analysis.mean_largest_drop == mean_drop == summary.mean_largest_drop
    assert analysis.cliff_rate.estimate == pytest.approx(len(EXPECTED_CLIFFS) / len(COMPLETED))
    assert analysis.cliff_rate.n_tasks == len(COMPLETED) and analysis.cliff_rate.replicates == REPLICATES


def test_analysis_gate_is_assess_gate_of_its_curves(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    assert analysis.gate == assess_gate(analysis.curves, data_complete=False)  # 有缺漏的 source（MISSING）
    assert analysis.gate.data_complete is False
    tasks: dict[str, set[str]] = {}
    families: dict[str, set[str]] = {}
    for task_id, (_, pattern) in EXPECTED_CLIFFS.items():
        tasks.setdefault(_pattern_key(pattern), set()).add(task_id)
        families.setdefault(_pattern_key(pattern), set()).add(BY_ID[task_id].task.family)
    assert analysis.gate.patterns == {key: tuple(sorted(value)) for key, value in tasks.items()}
    assert analysis.gate.families == {key: tuple(sorted(value)) for key, value in families.items()}
    assert analysis.gate.families["update_issue:forbidden_attempt"] == ("gen/assign-and-start", "gen/close-duplicate")
    assert analysis.gate.met is True and analysis.gate.rule == GATE_RULE


def test_analysis_of_a_budget_stopped_sweep(budget_runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(budget_runs, cache)
    summary = budget_runs.summary
    assert analysis.plan == BUDGET_PLAN and analysis.criterion == "bonferroni95"
    assert analysis.stopped_by_budget == ("sweep",)
    assert analysis.cost_usd == pytest.approx(summary.spent_usd)
    assert tuple(curve.task_id for curve in analysis.curves) == (COMPLETED[0],)
    assert analysis.missing_sources == MISSING
    assert analysis.not_completed == COMPLETED[1:]
    assert analysis.cliff_rate is None and analysis.mean_largest_drop is None
    assert analysis.curves[0].result == summary.results[COMPLETED[0]]
    assert analysis.gate == assess_gate(analysis.curves, data_complete=False)


def test_analysis_accepts_str_paths_and_is_deterministic(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    again = analyze_sweep_run(str(runs.sweep), str(runs.baseline), replicates=REPLICATES)
    assert again == _analysis(runs, cache)
    assert canonical_json(again.to_dict()) == canonical_json(_analysis(runs, cache).to_dict())


def _rewrite(path: Path, edit: Callable[[Any], Any]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data = edit(data) or data
    path.write_text(canonical_json(data) + "\n", encoding="utf-8")


def _set_source_id(data: dict[str, Any], value: str) -> None:
    data["source_trajectory_id"] = value


INCONSISTENT = [
    "summary-format",
    "missing-sweep-file",
    "swapped-sweep-file",
    "tampered-sweep-file",
    "malformed-sweep-file",
    "source-id",
    "baseline-as-sweep",
    "sweep-as-baseline",
]


@pytest.mark.parametrize("change", INCONSISTENT)
def test_analysis_rejects_inconsistent_inputs(change: str, runs: _Runs, tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    shutil.copytree(runs.sweep, sweep)
    baseline = runs.baseline
    victim = sweep / "sweeps" / f"{T_EXTRA_COMMENT}.json"
    if change == "summary-format":
        _rewrite(sweep / "summary.json", lambda d: d.update(format="agent-gym/sweep-run/v0"))
    elif change == "missing-sweep-file":
        victim.unlink()
    elif change == "swapped-sweep-file":
        victim.write_bytes((sweep / "sweeps" / f"{T_BLIND_ASSIGN}.json").read_bytes())
    elif change == "tampered-sweep-file":
        _rewrite(victim, lambda d: d.update(is_cliff=False))  # 與 summary 的結果不一致
    elif change == "malformed-sweep-file":
        victim.write_text(canonical_json({"task_id": T_EXTRA_COMMENT}) + "\n", encoding="utf-8")
    elif change == "source-id":  # 沒有 cliff 的 source：cliff_turn 不會先發現不一致
        fake = "sha256:" + "0" * 64
        _rewrite(sweep / "sweeps" / f"{T_ORACLE_CA}.json", lambda d: _set_source_id(d, fake))
        _rewrite(sweep / "summary.json", lambda d: _set_source_id(d["results"][T_ORACLE_CA], fake))
    elif change == "baseline-as-sweep":
        sweep = runs.baseline
    else:
        baseline = runs.sweep
    with pytest.raises(ValueError):
        analyze_sweep_run(sweep, baseline, replicates=REPLICATES)


def test_analysis_rejects_a_sweep_paired_with_another_baseline_run(runs: _Runs, tmp_path: Path) -> None:
    other = tmp_path / "other-baseline"
    run_split("eval", OracleFakeClient(POOL), run_dir=other, clock=ZERO)
    with pytest.raises(ValueError):
        analyze_sweep_run(runs.sweep, other, replicates=REPLICATES)


# ---------------------------------------------------------------- SweepAnalysis.to_dict


def test_to_dict_is_json_native_and_complete(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    data = analysis.to_dict()
    assert data["format"] == SWEEP_ANALYSIS_FORMAT
    assert json.loads(json.dumps(data, allow_nan=False)) == data  # 沒有 tuple、非 str key、NaN
    assert canonical_json(analysis.to_dict()) == canonical_json(data)
    for field in dataclasses.fields(SweepAnalysis):
        assert field.name in data
    assert data["plan"] == dataclasses.asdict(PLAN)
    assert data["criterion"] == "wilson95"
    assert data["missing_sources"] == list(MISSING) and data["not_completed"] == []
    assert data["cliff_rate"] == dataclasses.asdict(analysis.cliff_rate)
    assert data["mean_largest_drop"] == dataclasses.asdict(analysis.mean_largest_drop)
    assert data["cost_usd"] == analysis.cost_usd and data["stopped_by_budget"] == []
    gate = data["gate"]
    assert gate["rule"] == GATE_RULE and gate["met"] is True
    assert gate["patterns"] == {key: list(value) for key, value in analysis.gate.patterns.items()}
    assert gate["families"] == {key: list(value) for key, value in analysis.gate.families.items()}
    assert [curve["task_id"] for curve in data["curves"]] == list(COMPLETED)
    for item, curve in zip(data["curves"], analysis.curves):
        assert item["family"] == curve.family and item["source_trajectory_id"] == curve.source_trajectory_id
        assert item["source_success"] is curve.source_success and item["source_safe"] is curve.source_safe
        assert item["n_turns"] == curve.n_turns
        assert sweep_result_from_dict(item["result"]) == curve.result
        if curve.cliff is None:
            assert item["cliff"] is None
        else:
            assert item["cliff"] == {"k": curve.cliff.k, "pattern": [list(pair) for pair in curve.cliff.pattern]}


def test_to_dict_of_a_budget_stopped_analysis(budget_runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    data = _analysis(budget_runs, cache).to_dict()
    assert data["format"] == SWEEP_ANALYSIS_FORMAT
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["cliff_rate"] is None and data["mean_largest_drop"] is None
    assert data["not_completed"] == list(COMPLETED[1:]) and data["stopped_by_budget"] == ["sweep"]
    assert data["criterion"] == "bonferroni95" and data["gate"]["met"] is False


# ---------------------------------------------------------------- curve_svg


def _svg_root(svg: str) -> ET.Element:
    assert isinstance(svg, str)
    assert svg.lstrip().startswith("<svg") and svg.rstrip().endswith("</svg>")
    root = ET.fromstring(svg)
    assert root.tag.rsplit("}", 1)[-1] == "svg"
    return root


def _text(root: ET.Element) -> str:
    """所有文字節點，空白正規化（元素之間的換行不算內容）。"""
    return " ".join(" ".join(root.itertext()).split())


def _without_cliff(curve: SourceCurve, estimates: Any = None) -> SourceCurve:
    result = dataclasses.replace(
        curve.result, is_cliff=False, estimates=curve.result.estimates if estimates is None else estimates
    )
    return dataclasses.replace(curve, result=result, cliff=None)


def test_curve_svg_is_well_formed_and_deterministic_for_every_curve(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    for curve in _analysis(runs, cache).curves:
        svg = curve_svg(curve)
        text = _text(_svg_root(svg))
        assert curve.task_id in text and "wilson95" in text and "cliff" in text.lower()
        assert curve_svg(copy.deepcopy(curve)) == svg


def test_curve_svg_marks_the_cliff(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    cliffs = [curve for curve in _analysis(runs, cache).curves if curve.cliff is not None]
    assert {curve.task_id for curve in cliffs} == set(EXPECTED_CLIFFS)
    for curve in cliffs:
        svg = curve_svg(curve)
        plain = curve_svg(_without_cliff(curve))
        assert svg.count("dasharray") > plain.count("dasharray")  # k* → k* + 1 以虛線標出
        assert _text(_svg_root(svg)) != _text(_svg_root(plain))  # 標題標示是否 cliff


def test_curve_svg_draws_every_estimated_checkpoint(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    curve = next(c for c in _analysis(runs, cache).curves if c.task_id == T_NOISY)
    assert len(curve.result.estimates) >= 4
    full = _svg_root(curve_svg(_without_cliff(curve)))
    fewer = _svg_root(curve_svg(_without_cliff(curve, curve.result.estimates[:2])))
    # 每個已估計的 k 至少一個 p̂ 點與一條 CI 線段
    assert len(list(full.iter())) - len(list(fewer.iter())) >= 2 * (len(curve.result.estimates) - 2)


def test_curve_svg_names_the_criterion(budget_runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    curve = _analysis(budget_runs, cache).curves[0]
    text = _text(_svg_root(curve_svg(curve)))
    assert "bonferroni95" in text and curve.task_id in text


# ---------------------------------------------------------------- render_markdown


def test_render_markdown_mentions_every_required_section(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    md = render_markdown(analysis)
    assert isinstance(md, str)
    for task_id in COMPLETED + MISSING:
        assert task_id in md
    lower = md.lower()
    assert "plan" in lower and "wilson95" in md
    assert "max_turns" in md and "stopped_without_action" in md  # 曲線表的欄位
    for column in ("failure", "refusal", "agent_error", "aborted"):  # decision 017 §5.7：各類計數
        assert column in md, column
    assert "時間窗" in md and "UTC" in md and "response_model" in md
    assert "部分安全指標" in md and T_NOISY in md
    assert "ci" in lower and "cliff" in lower
    for key in analysis.gate.patterns:  # cliff turn 模式與 Gate 評估
        assert key in md
    assert "提案" in md or "proposal" in lower
    assert "未經" in md or "未確認" in md or "not yet confirmed" in lower or "unconfirmed" in lower
    assert "15" in md and ("初步" in md or "preliminary" in lower)
    assert "成本" in md or "cost" in lower
    assert render_markdown(analysis) == md


def test_render_markdown_lists_sources_not_completed(budget_runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(budget_runs, cache)
    md = render_markdown(analysis)
    for task_id in COMPLETED + MISSING:  # 完成的、未完成的、缺漏的都列出
        assert task_id in md
    assert "bonferroni95" in md
    assert "15" in md and ("初步" in md or "preliminary" in md.lower())


# ================================================================ decision 017 §5：manifest / rollout 核對、eval pool、多目錄、
# max_turns 混淆、response.model、時間窗、中止嘗試


def test_analysis_format_is_v3() -> None:
    assert SWEEP_ANALYSIS_FORMAT == "agent-gym/sweep-analysis/v3"  # v3：baseline_runs（decision 019 §4）
    assert [f.name for f in dataclasses.fields(WindowEstimate)] == ["date", "n_sources", "n_rollouts", "success_rate"]


def _only(*keep: str) -> tuple[str, ...]:
    """skip = 有 source、但不在 keep 中的 task（decision 018 §1：sources 必須涵蓋全部 eval task，其他的以 skip 略過）。"""
    return tuple(t for t in COMPLETED if t not in keep)


def _copy_sweep(runs: _Runs, tmp_path: Path, name: str = "sweep") -> Path:
    target = tmp_path / name
    shutil.copytree(runs.sweep, target)
    return target


def test_a_sweep_directory_without_summary_is_read_from_its_files(runs: _Runs, cache: dict[Path, SweepAnalysis], tmp_path: Path) -> None:
    sweep = _copy_sweep(runs, tmp_path)
    (sweep / "summary.json").unlink()
    analysis = analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)
    full = _analysis(runs, cache)
    assert analysis.interrupted_runs == ("sweep",) and analysis.stopped_by_budget == ()
    assert analysis.curves == full.curves and analysis.cliff_rate == full.cliff_rate and analysis.gate == full.gate


def test_unrun_tasks_of_a_budget_stopped_baseline_are_missing_sources(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    summary = run_split("eval", OracleFakeClient(POOL), run_dir=baseline, max_cost_usd=40 * CALL_COST, clock=ZERO)
    ran = [result.task_id for result in summary.results]
    assert summary.stopped_by_budget and 0 < len(ran) < 15
    sweep = tmp_path / "sweep"
    run_sweeps(select_sources(baseline), OracleFakeClient(POOL), plan=BUDGET_PLAN, criterion="wilson95", run_dir=sweep, clock=ZERO,
               replicates=REPLICATES)
    analysis = analyze_sweep_run(sweep, baseline, replicates=REPLICATES)
    assert analysis.missing_sources == tuple(t for t in IDS if t not in ran)
    assert tuple(curve.task_id for curve in analysis.curves) == tuple(ran)
    assert analysis.cliff_rate is None and analysis.mean_largest_drop is None  # 有 family 沒有完成的 sweep（012 §1）
    assert analysis.gate is not None and analysis.gate.data_complete is False
    assert "資料不完整" in render_markdown(analysis)


def test_sources_that_were_never_swept_are_not_completed(runs: _Runs, tmp_path: Path) -> None:
    keep = (T_ORACLE_AAS, T_EXTRA_COMMENT, T_ORACLE_CA)
    sweep = tmp_path / "subset"
    run_sweeps(runs.sources, _SweepClient(), plan=PLAN, criterion="wilson95", run_dir=sweep, clock=ZERO, replicates=REPLICATES,
               skip=_only(*keep))
    analysis = analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)
    assert tuple(curve.task_id for curve in analysis.curves) == keep
    assert analysis.missing_sources == MISSING
    assert analysis.not_completed == _only(*keep)  # skipped 但沒有在任何目錄完成 → not_completed
    assert analysis.resumed == ({"run": "subset", "skipped": _only(*keep)},)
    assert analysis.cliff_rate is None and analysis.gate is not None and analysis.gate.data_complete is False


def test_two_directories_complete_the_sweep_together(runs: _Runs, cache: dict[Path, SweepAnalysis], tmp_path: Path) -> None:
    first, second = COMPLETED[:7], COMPLETED[7:]
    for name, part in (("part-a", first), ("part-b", second)):
        run_sweeps(runs.sources, _SweepClient(SWEEP_OVERRIDES), plan=PLAN, criterion="wilson95",
                   run_dir=tmp_path / name, clock=ZERO, replicates=REPLICATES, skip=_only(*part))
    analysis = analyze_sweep_runs([tmp_path / "part-a", tmp_path / "part-b"], runs.baseline, replicates=REPLICATES)
    full = _analysis(runs, cache)
    assert analysis.runs == ("part-a", "part-b")
    assert [(c.task_id, c.result, c.cliff) for c in analysis.curves] == [(c.task_id, c.result, c.cliff) for c in full.curves]
    assert [c.run for c in analysis.curves] == ["part-a"] * len(first) + ["part-b"] * len(second)
    assert analysis.cliff_rate == full.cliff_rate and analysis.mean_largest_drop == full.mean_largest_drop
    assert analysis.gate == full.gate
    rows = [*read_manifest(tmp_path / "part-a" / "manifest.jsonl"), *read_manifest(tmp_path / "part-b" / "manifest.jsonl")]
    assert analysis.cost_usd == pytest.approx(sum(row.cost_usd for row in rows))


def test_a_source_completed_in_two_directories_is_rejected(runs: _Runs, tmp_path: Path) -> None:
    copy = _copy_sweep(runs, tmp_path, "copy")
    for dirs in ([runs.sweep, copy], [runs.sweep, runs.sweep], [runs.sweep, str(runs.sweep)]):
        with pytest.raises(ValueError):
            analyze_sweep_runs(dirs, runs.baseline, replicates=REPLICATES)
    with pytest.raises(ValueError):
        analyze_sweep_runs([], runs.baseline, replicates=REPLICATES)


def test_directories_with_different_plans_or_criteria_are_rejected(runs: _Runs, budget_runs: _Runs, tmp_path: Path) -> None:
    other = tmp_path / "other"
    run_sweeps(runs.sources, _SweepClient(), plan=PLAN, criterion="bonferroni95", run_dir=other,
               clock=ZERO, replicates=REPLICATES, skip=_only(T_ORACLE_CA))
    wilson = tmp_path / "wilson"
    run_sweeps(runs.sources, _SweepClient(), plan=PLAN, criterion="wilson95", run_dir=wilson,
               clock=ZERO, replicates=REPLICATES, skip=_only(T_ORACLE_AAS))
    analyze_sweep_runs([wilson], runs.baseline, replicates=REPLICATES)  # 單獨一個目錄是一致的
    # 各對目錄完成的 source 都不重疊，所以拒絕的原因只能是 plan（BUDGET_PLAN ≠ PLAN）或 criterion 不同
    assert T_ORACLE_CA not in budget_runs.summary.completed and T_ORACLE_AAS not in budget_runs.summary.completed
    for dirs in ([budget_runs.sweep, other], [other, wilson], [wilson, budget_runs.sweep]):
        with pytest.raises(ValueError):
            analyze_sweep_runs(dirs, runs.baseline, replicates=REPLICATES)


class _InterruptAfter:
    """前 n 次 create 照常，第 n + 1 次拋出 KeyboardInterrupt（run 中斷：不寫 summary，進行中的嘗試不進 manifest）。"""

    def __init__(self, inner: Any, n: int) -> None:
        self._inner = inner
        self._left = n
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        if self._left == 0:
            raise KeyboardInterrupt
        self._left -= 1
        return self._inner.create(**request)


def test_an_interrupted_sweep_directory_is_listed_and_billed(runs: _Runs, cache: dict[Path, SweepAnalysis], tmp_path: Path) -> None:
    interrupted = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_sweeps(runs.sources, _InterruptAfter(_SweepClient(), 6), plan=PLAN, criterion="wilson95", run_dir=interrupted,
                   clock=ZERO, replicates=REPLICATES, skip=_only(T_EXTRA_COMMENT))
    rows = read_manifest(interrupted / "manifest.jsonl")
    assert rows and not (interrupted / "summary.json").exists() and not (interrupted / "sweeps" / f"{T_EXTRA_COMMENT}.json").exists()
    analysis = analyze_sweep_runs([interrupted, runs.sweep], runs.baseline, replicates=REPLICATES)
    full = _analysis(runs, cache)
    assert analysis.interrupted_runs == ("interrupted",)
    assert analysis.partial_sweeps == (
        {"run": "interrupted", "task_id": T_EXTRA_COMMENT, "rollouts": len({(r.checkpoint_turn, r.rollout_index) for r in rows})},
    )
    assert analysis.curves == full.curves and analysis.cliff_rate == full.cliff_rate
    assert analysis.cost_usd == pytest.approx(full.cost_usd + sum(row.cost_usd for row in rows))
    assert "interrupted" in render_markdown(analysis)


def test_summary_aggregates_are_recomputed_not_copied(runs: _Runs, cache: dict[Path, SweepAnalysis], tmp_path: Path) -> None:
    sweep = _copy_sweep(runs, tmp_path)
    _rewrite(sweep / "summary.json", lambda d: d["cliff_rate"].update(estimate=0.9))
    analysis = analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)
    assert analysis.cliff_rate == _analysis(runs, cache).cliff_rate


def _edit_result(sweep: Path, task_id: str, edit: Callable[[dict[str, Any]], Any]) -> None:
    """同時修改 sweeps/<task_id>.json 與 summary.results[task_id]（保持兩者一致，只留下與 manifest / rollout 的不一致）。"""
    _rewrite(sweep / "sweeps" / f"{task_id}.json", edit)
    _rewrite(sweep / "summary.json", lambda d: edit(d["results"][task_id]))


def _add_deep_key(path: Path, depth: int) -> None:
    """在 JSON 物件最前面加入 depth 層巢狀的 list（直接改文字：canonical_json 本身無法寫出這麼深的資料）。"""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("{")
    path.write_text('{"extra":' + "[" * depth + "0" + "]" * depth + "," + text[1:], encoding="utf-8")


def _first_rollout_file(sweep: Path, task_id: str) -> Path:
    return sorted((sweep / "rollouts" / task_id).glob("*.json"))[0]


TAMPERED: dict[str, Callable[[Path], None]] = {
    "is-cliff-flipped": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d.update(is_cliff=True)),
    "confounded-flipped": lambda sw: _edit_result(sw, T_EXTRA_COMMENT, lambda d: d.update(max_turns_confounded=True, is_cliff=False)),
    "largest-drop": lambda sw: _edit_result(sw, T_EXTRA_COMMENT, lambda d: d.update(largest_drop=0.5)),
    "p-hat": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d["estimates"][0].update(p_hat=0.123)),
    "negative-n": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d["estimates"][0].update(n=-5)),
    "outcome-bogus": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d["rollouts"][0].update(outcome="banana")),
    "outcome-relabelled": lambda sw: _edit_result(sw, T_NOISY, lambda d: next(r for r in d["rollouts"] if r["outcome"] == "aborted").update(outcome="agent_error")),
    "attempts": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d["rollouts"][0].update(attempts=2)),
    "extra-rollout": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d["rollouts"].append({**d["rollouts"][0], "rollout_index": 99})),
    "refined-intervals": lambda sw: _edit_result(sw, T_ORACLE_CA, lambda d: d.update(refined_intervals=[[0, 2]])),
    "criterion-eyeball": lambda sw: (
        [_edit_result(sw, t, lambda d: d.update(criterion="eyeball")) for t in COMPLETED],
        _rewrite(sw / "summary.json", lambda d: d.update(criterion="eyeball")),
    ),
    "rollout-file-swapped": lambda sw: _first_rollout_file(sw, T_ORACLE_CA).write_bytes(_first_rollout_file(sw, T_ORACLE_AAS).read_bytes()),
    "rollout-file-missing": lambda sw: _first_rollout_file(sw, T_ORACLE_CA).unlink(),
    "rollout-file-corrupted": lambda sw: _first_rollout_file(sw, T_ORACLE_CA).write_text("{", encoding="utf-8"),
    "deep-nesting": lambda sw: _add_deep_key(sw / "sweeps" / f"{T_ORACLE_CA}.json", 5000),
    "deep-nesting-summary": lambda sw: _add_deep_key(sw / "summary.json", 5000),
    "summary-missing-criterion": lambda sw: _rewrite(sw / "summary.json", lambda d: d.pop("criterion") and None),  # pop 的回傳值不得取代整份資料
    "summary-missing-spent": lambda sw: _rewrite(sw / "summary.json", lambda d: d.pop("spent_usd") and None),  # pop 的回傳值不得取代整份資料
    "summary-spent-string": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(spent_usd="lots")),
    "summary-spent-huge": lambda sw: (sw / "summary.json").write_text(
        (sw / "summary.json").read_text(encoding="utf-8").replace('"spent_usd":', '"spent_usd":1e400,"x_":', 1), encoding="utf-8"
    ),
    "summary-stopped-string": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(stopped_by_budget="no")),
    "summary-completed-int": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(completed=3)),
    "summary-completed-nested": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(completed=[d["completed"]])),
    "summary-completed-duplicate": lambda sw: _rewrite(sw / "summary.json", lambda d: d["completed"].append(T_ORACLE_CA)),
    "summary-families-list": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(families=[])),
    "summary-family-wrong": lambda sw: _rewrite(sw / "summary.json", lambda d: d["families"].update({T_ORACLE_CA: "gen/other"})),
    "summary-sources-int": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(sources=1)),
    "summary-sources-bogus": lambda sw: _rewrite(sw / "summary.json", lambda d: d["sources"].append("bogus-task")),
    "summary-missing-tampered": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(missing_sources=[T_ORACLE_CA])),
    "summary-cliff-rate-types": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(
        cliff_rate={"estimate": "x", "low": None, "high": [], "replicates": -1, "n_tasks": True})),
    "summary-config": lambda sw: _rewrite(sw / "summary.json", lambda d: d.update(config_id="f" * 64)),
    "manifest-unknown-source": lambda sw: (sw / "manifest.jsonl").write_text(
        (sw / "manifest.jsonl").read_text(encoding="utf-8").replace(runs_source_id_placeholder(), "sha256:" + "0" * 64, 1),
        encoding="utf-8",
    ),
}


_SOURCE_ID: list[str] = []


def runs_source_id_placeholder() -> str:
    return _SOURCE_ID[0]


@pytest.mark.parametrize("change", sorted(TAMPERED))
def test_analysis_rejects_tampered_or_inconsistent_sweeps(change: str, runs: _Runs, tmp_path: Path) -> None:
    source = runs.sources[T_ORACLE_AAS]
    assert source is not None
    _SOURCE_ID[:] = [source.trajectory_id]
    sweep = _copy_sweep(runs, tmp_path)
    TAMPERED[change](sweep)
    with pytest.raises(ValueError):
        analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(criterion="eyeball"),
        lambda d: d["rollouts"][0].update(outcome="banana"),
        lambda d: d["rollouts"][0].update(attempts=0),
        lambda d: d["rollouts"][0].update(k=-1),
        lambda d: d["estimates"][0].update(k=-3),
        lambda d: d["estimates"][0].update(n=-1),
        lambda d: d.update(refined_intervals=[[5, 1]]),
        lambda d: d.update(largest_drop=10**400),
    ],
    ids=["criterion", "outcome", "attempts", "rollout-k", "estimate-k", "estimate-n", "interval", "huge-drop"],
)
def test_sweep_result_from_dict_rejects_invalid_values(edit: Callable[[dict[str, Any]], Any]) -> None:
    data = _jsonable(dataclasses.asdict(_varied_result()))
    edit(data)
    with pytest.raises(ValueError):
        sweep_result_from_dict(data)


class _LoopClient(_SweepClient):
    """T_ORACLE_AAS 從 k >= 1 續跑時一直讀同一個 issue，直到 max_turns（失敗）；其他照 _SweepClient。"""

    def create(self, **request: Any) -> FakeResponse:
        assistants = [message for message in request["messages"] if message["role"] == "assistant"]
        prefix = [message for message in assistants if _signature(message).startswith("src|")]
        if prefix and _signature(prefix[0]).split("|")[1] == T_ORACLE_AAS:
            target = BY_ID[T_ORACLE_AAS].oracle_calls[0][1]["issue_id"]
            return _respond((("get_issue", {"issue_id": target}),) * (len(assistants) + 1), f"loop|{len(assistants)}", len(assistants))
        return super().create(**request)


def test_a_max_turns_confounded_drop_is_listed_and_is_not_a_cliff(runs: _Runs, tmp_path: Path) -> None:
    sweep = tmp_path / "loop"
    plan = SweepPlan(coarse_stride=1, n_coarse=4, n_fine=4, drop_threshold=0.5)
    run_sweeps(runs.sources, _LoopClient(), plan=plan, criterion="wilson95", run_dir=sweep,
               clock=ZERO, replicates=REPLICATES, skip=_only(T_ORACLE_AAS))
    analysis = analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)
    (curve,) = analysis.curves
    by_k = {e.k: e for e in curve.result.estimates}
    assert (by_k[0].p_hat, by_k[0].max_turns_terminations, by_k[1].p_hat, by_k[1].max_turns_terminations) == (1.0, 0, 0.0, 4)
    assert curve.result.max_turns_confounded is True and curve.result.is_cliff is False and curve.cliff is None
    assert analysis.max_turns_confounded == (T_ORACLE_AAS,)
    assert analysis.gate is not None and analysis.gate.patterns == {}
    md = render_markdown(analysis)
    assert "max_turns" in md and "混淆" in md
    assert "max_turns" in _text(_svg_root(curve_svg(curve)))


class _ModelSwapClient(_SweepClient):
    """從 k >= 1 續跑的 response.model 改為 ALT_MODEL。"""

    def create(self, **request: Any) -> Any:
        response = super().create(**request)
        assistants = [message for message in request["messages"] if message["role"] == "assistant"]
        if any(_signature(message).startswith("src|") for message in assistants):
            response = dataclasses.replace(response, model=ALT_MODEL)
        return response


ALT_MODEL = "claude-opus-5-5-preview"


def test_more_than_one_response_model_blocks_aggregates_and_gate(runs: _Runs, tmp_path: Path) -> None:
    sweep = tmp_path / "alt"
    run_sweeps(runs.sources, _ModelSwapClient(), plan=PLAN, criterion="wilson95", run_dir=sweep, clock=ZERO, replicates=REPLICATES,
               skip=_only(T_EXTRA_COMMENT, T_ORACLE_CA))
    analysis = analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)
    assert analysis.pooling_blocked is not None and ALT_MODEL in analysis.pooling_blocked
    assert set(analysis.response_models) == {"claude-opus-5-5", ALT_MODEL}
    assert analysis.cliff_rate is None and analysis.mean_largest_drop is None and analysis.gate is None
    assert analysis.time_windows == ()  # decision 018 §5：時間窗也是跨 model 的合併
    assert len(analysis.curves) == 2  # 曲線照列
    md = render_markdown(analysis)
    assert "009 §4" in md and ALT_MODEL in md


def test_response_models_of_a_single_model_sweep(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    n_completed = sum(1 for c in analysis.curves for r in c.result.rollouts if r.outcome in ("success", "failure", "refusal"))
    assert analysis.response_models == {"claude-opus-5-5": n_completed}


def test_time_windows_follow_the_utc_date_of_the_last_attempt(runs: _Runs, tmp_path: Path) -> None:
    sweep = _copy_sweep(runs, tmp_path)
    path = sweep / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        late = row["task_id"] == T_NOISY and row["attempt"] == 2  # aborted rollout 的第 2 次嘗試跨到下一天
        row["utc_time"] = "2026-10-09T00:30:00+00:00" if late else "2026-10-08T23:30:00+00:00"
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    analysis = analyze_sweep_run(sweep, runs.baseline, replicates=REPLICATES)
    rollouts = [(c.task_id, r) for c in analysis.curves for r in c.result.rollouts]
    late = [(t, r) for t, r in rollouts if t == T_NOISY and r.attempts == 2]
    assert len(late) == 1
    early = [(t, r) for t, r in rollouts if (t, r) not in late]
    assert analysis.time_windows == (
        WindowEstimate("2026-10-08", len({t for t, _ in early}), len(early), sum(r.outcome == "success" for _, r in early) / len(early)),
        WindowEstimate("2026-10-09", 1, 1, 0.0),
    )


def test_default_run_has_one_time_window(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    dates = {utc_date(row.utc_time) for row in read_manifest(runs.sweep / "manifest.jsonl")}
    assert [w.date for w in analysis.time_windows] == sorted(dates)
    assert sum(w.n_rollouts for w in analysis.time_windows) == sum(len(c.result.rollouts) for c in analysis.curves)


def test_partial_safety_lists_every_aborted_sweep_attempt(runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(runs, cache)
    rows = [row for row in read_manifest(runs.sweep / "manifest.jsonl") if row.outcome == "aborted"]
    assert len(rows) == 3  # T_NOISY k = 2：agent_error 1 次、aborted 2 次
    assert analysis.partial_safety == tuple(
        {
            "run": "sweep", "task_id": row.task_id, "k": row.checkpoint_turn, "rollout_index": row.rollout_index,
            "attempt": row.attempt, "partial_steps": row.partial_steps,
            "partial_forbidden_action_count": row.partial_forbidden_action_count,
            "partial_forbidden_attempt_count": row.partial_forbidden_attempt_count,
        }
        for row in rows
    )


# ================================================================ decision 018：summary 語意、manifest 列、目錄識別、multi-tool turn


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _partial_row(sweep: Path, budget: SweepRunSummary) -> tuple[list[dict[str, Any]], int]:
    rows = _rows(sweep / "manifest.jsonl")
    index = next(i for i, row in enumerate(rows) if row["task_id"] not in budget.completed)
    return rows, index


def test_budget_fixture_has_a_partial_sweep(budget_runs: _Runs, cache: dict[Path, SweepAnalysis]) -> None:
    analysis = _analysis(budget_runs, cache)
    assert analysis.partial_sweeps and analysis.partial_sweeps[0]["task_id"] == COMPLETED[1]


def _tamper_partial_rows(change: str, sweep: Path, budget: SweepRunSummary) -> None:
    rows, i = _partial_row(sweep, budget)
    row = rows[i]
    if change == "duplicate":
        rows.insert(i + 1, dict(row))
    elif change == "attempt-gap":
        rows[i] = {**row, "attempt": 2}
    elif change == "k-out-of-range":
        rows[i] = {**row, "checkpoint_turn": 999}
    elif change == "k-none":
        rows[i] = {**row, "checkpoint_turn": None}
    elif change == "checkpoint-id":
        rows[i] = {**row, "checkpoint_id": "sha256:" + "1" * 64}
    elif change == "harness":
        rows[i] = {**row, "outcome": "aborted", "error_class": "harness", "trajectory_id": None, "termination": None,
                   "exception_type": "builtins.RuntimeError", "message": "x"}
    elif change == "completed-then-retry":
        rows.insert(i + 1, {**row, "attempt": 2})
    _write_rows(sweep / "manifest.jsonl", rows)


@pytest.mark.parametrize(
    "change", ["duplicate", "attempt-gap", "k-out-of-range", "k-none", "checkpoint-id", "harness", "completed-then-retry"]
)
def test_rows_of_unfinished_sweeps_are_validated(change: str, budget_runs: _Runs, tmp_path: Path) -> None:
    sweep = tmp_path / "budget"
    shutil.copytree(budget_runs.sweep, sweep)
    _tamper_partial_rows(change, sweep, budget_runs.summary)
    with pytest.raises(ValueError):
        analyze_sweep_run(sweep, budget_runs.baseline, replicates=REPLICATES)


def test_an_unfinished_retry_is_allowed_only_when_the_budget_stopped_it(budget_runs: _Runs, tmp_path: Path) -> None:
    sweep = tmp_path / "budget"
    shutil.copytree(budget_runs.sweep, sweep)
    rows, i = _partial_row(sweep, budget_runs.summary)
    rows[i] = {**rows[i], "outcome": "aborted", "error_class": "infrastructure", "trajectory_id": None, "termination": None,
               "exception_type": "anthropic.APIConnectionError", "message": "x"}
    _write_rows(sweep / "manifest.jsonl", rows)
    analysis = analyze_sweep_run(sweep, budget_runs.baseline, replicates=REPLICATES)  # 預算擋下了重試：可以
    assert analysis.stopped_by_budget == ("budget",)
    _rewrite(sweep / "summary.json", lambda d: d.update(stopped_by_budget=False))
    with pytest.raises(ValueError):  # 沒有因預算停止卻少跑、且重試沒做
        analyze_sweep_run(sweep, budget_runs.baseline, replicates=REPLICATES)


SUMMARY_SEMANTICS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "sources-subset": lambda d: d.update(sources=d["sources"][:5], families={t: d["families"][t] for t in d["sources"][:5]}),
    "sources-reordered": lambda d: d.update(sources=list(reversed(d["sources"]))),
    "sources-empty": lambda d: d.update(sources=[], missing_sources=[], families={}),
    "missing-reordered": lambda d: d.update(missing_sources=list(reversed(d["missing_sources"]))),
    "not-stopped-but-short": lambda d: d.update(stopped_by_budget=False),
    "skipped-with-rows": lambda d: d.update(skipped=[COMPLETED[1]]),  # 部分完成、manifest 有 rollout 的 source
    "skipped-missing-source": lambda d: d.update(skipped=[d["missing_sources"][0]]),
    "skipped-not-a-list": lambda d: d.update(skipped="x"),
    "config-unhashable": lambda d: d.update(config_id=[]),
    "config-dict": lambda d: d.update(config_id={"a": 1}),
}


@pytest.mark.parametrize("change", sorted(SUMMARY_SEMANTICS))
def test_sweep_summary_semantics_are_checked(change: str, budget_runs: _Runs, tmp_path: Path) -> None:
    sweep = tmp_path / "budget"
    shutil.copytree(budget_runs.sweep, sweep)
    _rewrite(sweep / "summary.json", SUMMARY_SEMANTICS[change])
    with pytest.raises(ValueError):
        analyze_sweep_run(sweep, budget_runs.baseline, replicates=REPLICATES)


def _case_insensitive(path: Path) -> bool:
    variant = path.parent / path.name.swapcase()
    return path.name != path.name.swapcase() and variant.exists() and variant.samefile(path)


def test_the_same_directory_through_another_spelling_is_rejected(runs: _Runs, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(runs.sweep)
    for dirs in ([runs.sweep, link], [link, runs.sweep]):
        with pytest.raises(ValueError):
            analyze_sweep_runs(dirs, runs.baseline, replicates=REPLICATES)
    baseline_link = tmp_path / "baseline-link"
    baseline_link.symlink_to(runs.baseline)
    with pytest.raises(ValueError):  # baseline 本身（換個路徑）也不能當 sweep 目錄
        analyze_sweep_runs([baseline_link], runs.baseline, replicates=REPLICATES)
    upper = runs.sweep.parent / runs.sweep.name.upper()
    if _case_insensitive(runs.sweep):  # macOS 預設不分大小寫：resolve() 不會正規化大小寫
        with pytest.raises(ValueError):
            analyze_sweep_runs([runs.sweep, upper], runs.baseline, replicates=REPLICATES)


@pytest.mark.parametrize("kind", ["missing", "loop", "file", "no-manifest"])
def test_unusable_sweep_directories_are_value_errors(kind: str, runs: _Runs, tmp_path: Path) -> None:
    target = tmp_path / "target"
    if kind == "loop":
        target.symlink_to(target)
    elif kind == "file":
        target.write_text("x", encoding="utf-8")
    elif kind == "no-manifest":
        target.mkdir()
    with pytest.raises(ValueError):
        analyze_sweep_runs([target], runs.baseline, replicates=REPLICATES)
    with pytest.raises(ValueError):
        analyze_sweep_runs([runs.sweep], target, replicates=REPLICATES)


class _DoubleReadClient:
    """T_DOUBLE：照 oracle，做完後一個 turn 同時讀兩個 issue（tool_use_blocks = 2），再結束；其他 task 交給 OracleFakeClient。
    baseline 與 sweep 共用（t = request 中的 assistant 訊息數，前綴與續跑照同一個劇本）。"""

    def __init__(self) -> None:
        self._inner = OracleFakeClient(POOL)
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        if request["messages"][0]["content"] != BY_ID[T_DOUBLE].task.instruction:
            return self._inner.create(**request)
        t = sum(1 for message in request["messages"] if message["role"] == "assistant")
        oracle = BY_ID[T_DOUBLE].oracle_calls
        signature = f"dbl|{t}"
        if t < len(oracle):
            return _respond(oracle, signature, t)
        if t == len(oracle):
            first, second = sorted(issue["id"] for issue in BY_ID[T_DOUBLE].task.initial_state["issues"])[:2]
            return FakeResponse(
                content=[thinking_block(signature), tool_use_block("toolu_a", "get_issue", {"issue_id": first}),
                         tool_use_block("toolu_b", "get_issue", {"issue_id": second})],
                stop_reason="tool_use",
                usage=_usage(),
            )
        return _respond((), signature, 0)


T_DOUBLE = "gen-escalate-label-s3"


def test_multi_tool_turns_of_sources_and_sweep_rollouts_are_listed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    run_split("eval", _DoubleReadClient(), run_dir=baseline, clock=ZERO)
    sources = select_sources(baseline)
    n = len(BY_ID[T_DOUBLE].oracle_calls)
    assert sources[T_DOUBLE] is not None and len(sources[T_DOUBLE].turns) == n + 2
    sweep = tmp_path / "sweep"
    plan = SweepPlan(coarse_stride=1, n_coarse=1, n_fine=1, drop_threshold=0.5)
    run_sweeps(sources, _DoubleReadClient(), plan=plan, criterion="wilson95", run_dir=sweep, clock=ZERO, replicates=REPLICATES,
               skip=tuple(t for t in IDS if t != T_DOUBLE))
    analysis = analyze_sweep_run(sweep, baseline, replicates=REPLICATES)
    assert analysis.source_multi_tool_turns == ({"task_id": T_DOUBLE, "turn": n},)
    (curve,) = analysis.curves
    expected = tuple(
        {"run": "sweep", "task_id": T_DOUBLE, "k": r.k, "rollout_index": r.rollout_index, "turn": n}
        for r in curve.result.rollouts
        if r.k <= n  # k 之後才有 rerun 自己的 turn；k = n + 1 的前綴已含那個 turn
    )
    assert analysis.multi_tool_turns == expected and len(expected) == n + 1
    md = render_markdown(analysis)
    assert "multi-tool" in md and f"turn {n}" in md
