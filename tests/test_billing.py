"""decision 014：已計費但被 adapter 拒絕的回應必須記帳（成本上限、manifest、runner）的回歸測試。

§6 審查確認：API 回傳（已計費）之後才由 parse_response 拋出的 AdapterError / ReplayIntegrityError（例如 max_tokens 截斷），
原本完全不記帳 → ``max_cost_usd`` 可被繞過、spent_usd 與 manifest 都低估。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import (
    PRICING,
    AdapterError,
    AnthropicPolicy,
    ReplayIntegrityError,
    baseline_agent_config,
    billed_usage,
)
from agents.policy import ModelMessage, PolicyStep, oracle_policy
from apps.issue_tracker.tools import TOOL_SPECS
from experiments.run_baseline import CostMeter, Ledger, OracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.aborts import ATTEMPT_RECORD_FORMAT, CountingPolicy, RolloutKey, read_manifest, run_logged
from replay.localization import SweepPlan
from replay.runner import run_episode
from tasks.generator import week1_pool
from tasks.smoke import REFERENCE_SOLUTIONS, get_smoke_task
from tests.fake_anthropic import (
    FakeCacheCreation,
    FakeClient,
    FakeResponse,
    FakeUsage,
    text_block,
    thinking_block,
)

P = PRICING["claude-opus-5-5"]
ZERO = lambda: 0.0  # noqa: E731
CALL_COST = (1000 * 4.00 + 200 * 20.00) / 1e6  # OracleFakeClient 的固定 usage
TRUNCATED_COST = (2000 * P["input"] + 16000 * P["output"]) / 1e6


def _truncated(usage: FakeUsage | None = None) -> FakeResponse:
    return FakeResponse(
        content=[thinking_block("sig"), text_block("cut")],
        stop_reason="max_tokens",
        usage=usage or FakeUsage(2000, 16000),
    )


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "do it"},
    ]


# ---------------------------------------------------------------- billed_usage


def test_billed_usage_matches_parse_response_cost_rules() -> None:
    usage = FakeUsage(100, 50, cache_creation_input_tokens=40, cache_read_input_tokens=10,
                      cache_creation=FakeCacheCreation(ephemeral_5m_input_tokens=30, ephemeral_1h_input_tokens=10))
    billed = billed_usage(_truncated(usage), "claude-opus-5-5")
    expected = (100 * P["input"] + 30 * P["cache_write"] + 10 * P["cache_write_1h"] + 10 * P["cache_read"] + 50 * P["output"]) / 1e6
    assert billed == {"input_tokens": 150, "output_tokens": 50, "cost_usd": pytest.approx(expected)}


def test_billed_usage_never_raises() -> None:
    assert billed_usage(_truncated(), "unpriced-model") == {"input_tokens": 2000, "output_tokens": 16000, "cost_usd": None}
    bad = FakeResponse(content=[], stop_reason="max_tokens", usage=FakeUsage(-1, 5))  # type: ignore[arg-type]
    assert billed_usage(bad, "claude-opus-5-5") == {"input_tokens": None, "output_tokens": None, "cost_usd": None}
    assert billed_usage(object(), "claude-opus-5-5") == {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


# ---------------------------------------------------------------- AnthropicPolicy attaches billed_usage


def test_rejected_response_carries_billed_usage() -> None:
    policy = AnthropicPolicy(FakeClient([_truncated()]))
    with pytest.raises(AdapterError) as info:
        policy.next_message(_messages(), TOOL_SPECS)
    assert info.value.billed_usage == {"input_tokens": 2000, "output_tokens": 16000, "cost_usd": pytest.approx(TRUNCATED_COST)}


def test_replay_integrity_error_carries_billed_usage() -> None:
    response = FakeResponse(content=[thinking_block("s"), text_block("x")], stop_reason="end_turn",
                            usage=FakeUsage(10, 5), input_transformations=[{"type": "dropped_thinking"}])
    with pytest.raises(ReplayIntegrityError) as info:
        AnthropicPolicy(FakeClient([response])).next_message(_messages(), TOOL_SPECS)
    assert info.value.billed_usage["output_tokens"] == 5


def test_client_errors_are_not_billed() -> None:
    def fail(request: dict[str, Any]) -> FakeResponse:
        raise ConnectionError("no response")

    with pytest.raises(ConnectionError) as info:
        AnthropicPolicy(FakeClient(responder=fail)).next_message(_messages(), TOOL_SPECS)
    assert not hasattr(info.value, "billed_usage")


# ---------------------------------------------------------------- CountingPolicy / CostMeter


class _Seq:
    """依序回傳 PolicyStep 或拋出例外。"""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    @property
    def agent_config(self) -> Any:
        return baseline_agent_config()

    def next_message(self, messages: Any, tools: Any) -> PolicyStep:
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return PolicyStep(ModelMessage("", (), "end_turn"), 0, 0, item)


def _billed_error(cost: float | None) -> AdapterError:
    error = AdapterError("truncated")
    error.billed_usage = {"input_tokens": 1, "output_tokens": 1, "cost_usd": cost}  # type: ignore[attr-defined]
    return error


def test_counting_policy_tracks_billed_calls_including_rejected_responses() -> None:
    counting = CountingPolicy(_Seq([0.5, None, _billed_error(0.25), ValueError("not billed"), _billed_error(None)]))
    counting.next_message([], ())
    counting.next_message([], ())
    for _ in range(3):
        with pytest.raises(Exception):
            counting.next_message([], ())
    assert counting.calls_completed == 2
    assert counting.billed_calls == 4
    assert counting.cost_usd == pytest.approx(0.75)
    assert counting.unknown_cost_calls == 2


def test_cost_meter_counts_rejected_responses() -> None:
    ledger = Ledger()
    meter = CostMeter(_Seq([0.5, _billed_error(0.25), KeyError("x"), _billed_error(None)]), ledger)
    meter.next_message([], ())
    for _ in range(3):
        with pytest.raises(Exception):
            meter.next_message([], ())
    assert (ledger.cost_usd, ledger.calls, ledger.unknown_cost_calls) == (pytest.approx(0.75), 3, 1)


# ---------------------------------------------------------------- manifest v2


def test_manifest_rows_record_attempt_billing(tmp_path: Path) -> None:
    assert ATTEMPT_RECORD_FORMAT == "agent-gym/attempt/v2"
    task = get_smoke_task("smoke-assign-and-start")
    log = tmp_path / "m.jsonl"

    def run(policy: Any) -> Any:
        return run_episode(task, policy, max_turns=20, clock=ZERO)

    client = FakeClient([_tool(0), _truncated()])
    assert run_logged(run, AnthropicPolicy(client), task=task, key=RolloutKey(None, None, None, 0, 1),
                      max_turns=20, log_path=log) is None
    (row,) = read_manifest(log)
    assert (row.error_class, row.model_calls_completed, row.billed_calls, row.unknown_cost_calls) == ("agent_output", 1, 2, 0)
    assert row.cost_usd == pytest.approx((100 * P["input"] + 20 * P["output"]) / 1e6 + TRUNCATED_COST)

    scripted = oracle_policy(REFERENCE_SOLUTIONS[task.task_id])
    assert run_logged(run, scripted, task=task, key=RolloutKey(None, None, None, 1, 1), max_turns=20, log_path=log) is not None
    completed = read_manifest(log)[1]
    assert (completed.billed_calls, completed.cost_usd, completed.unknown_cost_calls) == (3, 0.0, 0)


def _tool(t: int) -> FakeResponse:
    from tests.fake_anthropic import tool_turn

    name, args = REFERENCE_SOLUTIONS["smoke-assign-and-start"][t]
    return tool_turn(f"toolu_{t}", name, args)


# ---------------------------------------------------------------- runner 預算


class _TruncateSecond:
    """包住 OracleFakeClient：每個 episode 的第 2 次 call 改回傳已計費的 max_tokens 截斷。"""

    def __init__(self) -> None:
        self._inner = OracleFakeClient(week1_pool("dev"))
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        if sum(1 for m in request["messages"] if m["role"] == "assistant") == 1:
            return _truncated()
        return self._inner.create(**request)


def test_run_split_counts_truncated_responses_against_the_budget(tmp_path: Path) -> None:
    summary = run_split("dev", _TruncateSecond(), run_dir=tmp_path / "run", clock=ZERO)
    assert all(r.outcome == "agent_error" for r in summary.results)
    assert summary.model_calls == 2 * len(summary.results)
    assert summary.spent_usd == pytest.approx(len(summary.results) * (CALL_COST + TRUNCATED_COST))

    capped = run_split("dev", _TruncateSecond(), run_dir=tmp_path / "capped", max_cost_usd=0.5, clock=ZERO)
    assert capped.stopped_by_budget is True
    assert len(capped.results) == 2  # 第 1 個 episode 花了 ≈ $0.33，第 2 個之後超過 $0.5


def test_budget_requires_a_priced_model(tmp_path: Path) -> None:
    base = baseline_agent_config()
    unpriced = dataclasses.replace(base, model_config={**base.model_config, "model": "claude-test-unpriced"})
    client = OracleFakeClient(week1_pool("dev"))
    with pytest.raises(ValueError):
        run_split("dev", client, run_dir=tmp_path / "a", max_cost_usd=1.0, clock=ZERO, agent_config=unpriced)
    assert client.requests == []
    summary = run_split("dev", client, run_dir=tmp_path / "b", clock=ZERO, agent_config=unpriced)  # 沒有上限：允許
    assert summary.unknown_cost_calls == summary.model_calls > 0


def test_sweep_budget_requires_a_priced_model(tmp_path: Path) -> None:
    base = baseline_agent_config()
    unpriced = dataclasses.replace(base, model_config={**base.model_config, "model": "claude-test-unpriced"})
    run_dir = tmp_path / "eval"
    run_split("eval", OracleFakeClient(week1_pool("eval")), run_dir=run_dir, clock=ZERO, agent_config=unpriced)
    sources = select_sources(run_dir)
    client = OracleFakeClient(week1_pool("eval"))
    with pytest.raises(ValueError):
        run_sweeps(sources, client, plan=SweepPlan(3, 2, 2, 0.3), run_dir=tmp_path / "s", max_cost_usd=1.0,
                   clock=ZERO, agent_config=unpriced)
    assert client.requests == []
    summary = run_sweeps(sources, client, plan=SweepPlan(3, 2, 2, 0.3), run_dir=tmp_path / "t", clock=ZERO,
                         agent_config=unpriced, replicates=20)
    assert summary.unknown_cost_calls == summary.model_calls > 0


class _BadUsage:
    """第 1 次 call 回傳 usage 不合法（無法計算成本）的回應。"""

    def __init__(self) -> None:
        self._inner = OracleFakeClient(week1_pool("dev"))
        self.beta = self
        self.messages = self
        self.calls = 0

    def create(self, **request: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return FakeResponse(content=[thinking_block("s"), text_block("x")], stop_reason="end_turn",
                                usage=FakeUsage(-5, 1))  # type: ignore[arg-type]
        return self._inner.create(**request)


def test_unmeasurable_cost_stops_a_budgeted_run(tmp_path: Path) -> None:
    """有上限時，出現無法量測成本的已計費 call → 保守地停止排程（decision 014）。"""
    summary = run_split("dev", _BadUsage(), run_dir=tmp_path / "run", max_cost_usd=100.0, clock=ZERO)
    assert summary.unknown_cost_calls == 1
    assert summary.stopped_by_budget is True and len(summary.results) == 1
