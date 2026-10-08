"""experiments/run_baseline.py 的契約測試：dry-run 全流程（OracleFakeClient，不連網、不需要 credential）、預算上限、
排程、三種中止路徑、輸出檔案。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import BASELINE_MAX_TURNS, MAX_EPISODE_ATTEMPTS, AdapterError, baseline_agent_config
from agents.policy import ModelMessage, PolicyStep
from env.hashing import canonical_json
from experiments.run_baseline import (
    SUMMARY_FORMAT,
    CostMeter,
    EpisodeResult,
    Ledger,
    OracleFakeClient,
    RunSummary,
    run_split,
)
from replay.aborts import read_manifest
from replay.trajectory import Trajectory
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
CALL_COST = (1000 * 4.00 + 200 * 20.00) / 1e6  # OracleFakeClient 的固定 usage × claude-opus-5-5 價格


def _dev() -> tuple[Any, ...]:
    return week1_pool("dev")


def _client(split: str = "dev") -> OracleFakeClient:
    return OracleFakeClient(week1_pool(split))


# ---------------------------------------------------------------- OracleFakeClient


def test_oracle_client_follows_the_oracle_and_records_requests(tmp_path: Path) -> None:
    client = _client()
    summary = run_split("dev", client, run_dir=tmp_path / "run", clock=ZERO)
    n_calls = sum(len(inst.oracle_calls) + 1 for inst in _dev())
    assert len(client.requests) == n_calls == summary.model_calls
    first = client.requests[0]
    assert first["model"] == baseline_agent_config().model_config["model"]
    assert first["messages"][0] == {"role": "user", "content": _dev()[0].task.instruction}


def test_oracle_client_rejects_unknown_instruction() -> None:
    client = _client()
    request = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "not a task"}]}
    with pytest.raises(ValueError):
        client.beta.messages.create(**request)


# ---------------------------------------------------------------- dry-run 全流程


def test_dry_run_dev_succeeds_and_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    summary = run_split("dev", _client(), run_dir=run_dir, clock=ZERO)
    assert isinstance(summary, RunSummary)
    assert (summary.split, summary.n_tasks, summary.rollouts, summary.stopped_by_budget) == ("dev", 5, 1, False)
    assert summary.config_id == baseline_agent_config().config_id
    assert [r.task_id for r in summary.results] == [inst.task.task_id for inst in _dev()]
    for result, inst in zip(summary.results, _dev()):
        assert isinstance(result, EpisodeResult)
        assert result.family == inst.task.family
        assert (result.outcome, result.termination, result.attempts) == ("success", "final_answer", 1)
        assert (result.forbidden_action_count, result.forbidden_attempt_count, result.trajectory_safe) == (0, 0, True)
        assert result.model_turns == len(inst.oracle_calls) + 1
        assert result.cost_usd == pytest.approx(CALL_COST * result.model_turns)
        assert result.stopped_without_action is False
        trajectory = Trajectory.load(run_dir / "trajectories" / f"{result.task_id}-r0.json")
        assert trajectory.trajectory_id == result.trajectory_id
        trajectory.verify()
        assert trajectory.agent.config_id == summary.config_id
        assert trajectory.max_turns == BASELINE_MAX_TURNS
    assert summary.spent_usd == pytest.approx(CALL_COST * summary.model_calls)
    assert summary.unknown_cost_calls == 0

    rows = read_manifest(run_dir / "manifest.jsonl")
    assert [(r.task_id, r.rollout_index, r.attempt, r.outcome) for r in rows] == [
        (inst.task.task_id, 0, 1, "completed") for inst in _dev()
    ]
    written = (run_dir / "summary.json").read_text(encoding="utf-8")
    assert written == canonical_json(summary.to_dict()) + "\n"
    data = json.loads(written)
    assert data["format"] == SUMMARY_FORMAT
    assert len(data["results"]) == 5 and data["results"][0]["task_id"] == _dev()[0].task.task_id


def test_dry_run_is_deterministic(tmp_path: Path) -> None:
    a = run_split("dev", _client(), run_dir=tmp_path / "a", clock=ZERO)
    b = run_split("dev", _client(), run_dir=tmp_path / "b", clock=ZERO)
    assert canonical_json(a.to_dict()) == canonical_json(b.to_dict())
    for result in a.results:
        name = f"{result.task_id}-r0.json"
        assert (tmp_path / "a" / "trajectories" / name).read_bytes() == (tmp_path / "b" / "trajectories" / name).read_bytes()


def test_rollouts_are_scheduled_round_by_round(tmp_path: Path) -> None:
    summary = run_split("dev", _client(), run_dir=tmp_path / "run", rollouts=2, clock=ZERO)
    ids = [inst.task.task_id for inst in _dev()]
    assert [(r.rollout_index, r.task_id) for r in summary.results] == [(0, t) for t in ids] + [(1, t) for t in ids]
    assert summary.rollouts == 2


def test_budget_stops_scheduling(tmp_path: Path) -> None:
    summary = run_split("dev", _client(), run_dir=tmp_path / "run", max_cost_usd=CALL_COST / 2, clock=ZERO)
    assert summary.stopped_by_budget is True
    assert len(summary.results) == 1
    assert summary.spent_usd >= CALL_COST / 2
    assert len(read_manifest(tmp_path / "run" / "manifest.jsonl")) == 1


# ---------------------------------------------------------------- 事前檢查


def test_existing_non_empty_run_dir_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("x", encoding="utf-8")
    client = _client()
    with pytest.raises(ValueError):
        run_split("dev", client, run_dir=run_dir, clock=ZERO)
    assert client.requests == []
    assert sorted(p.name for p in run_dir.iterdir()) == ["keep.txt"]


def test_empty_run_dir_is_accepted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert len(run_split("dev", _client(), run_dir=run_dir, clock=ZERO).results) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "all"},
        {"split": "train"},
        {"rollouts": 0},
        {"rollouts": True},
        {"rollouts": 1.0},
        {"max_cost_usd": 0},
        {"max_cost_usd": -1.0},
        {"max_cost_usd": math.nan},
        {"max_cost_usd": math.inf},
        {"max_cost_usd": True},
    ],
    ids=repr,
)
def test_bad_arguments_are_rejected_without_running(kwargs: dict[str, Any], tmp_path: Path) -> None:
    client = _client()
    args = {"split": "dev", **kwargs}
    split = args.pop("split")
    with pytest.raises(ValueError):
        run_split(split, client, run_dir=tmp_path / "run", clock=ZERO, **args)
    assert client.requests == []
    assert not (tmp_path / "run" / "manifest.jsonl").exists()


# ---------------------------------------------------------------- 中止路徑


class _Wrap:
    """包住 OracleFakeClient：第 n 次 create 改為拋出 error 或回傳 replacement。"""

    def __init__(self, inner: OracleFakeClient, at: int, *, error: BaseException | None = None, times: int = 1) -> None:
        self._inner = inner
        self._at = at
        self._error = error
        self._times = times
        self.calls = 0
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        self.calls += 1
        if self._at <= self.calls < self._at + self._times:
            assert self._error is not None
            raise self._error
        return self._inner.beta.messages.create(**request)


def _sdk_error(name: str) -> Exception:
    return type(name, (Exception,), {"__module__": "anthropic"})(f"{name} (fake)")


def test_infrastructure_error_is_retried(tmp_path: Path) -> None:
    client = _Wrap(_client(), at=2, error=_sdk_error("APIConnectionError"))
    summary = run_split("dev", client, run_dir=tmp_path / "run", clock=ZERO)
    first = summary.results[0]
    assert (first.outcome, first.attempts) == ("success", 2)
    rows = read_manifest(tmp_path / "run" / "manifest.jsonl")
    assert [(r.task_id == first.task_id, r.attempt, r.outcome) for r in rows[:2]] == [
        (True, 1, "aborted"),
        (True, 2, "completed"),
    ]


def test_infrastructure_error_exhausts_attempts(tmp_path: Path) -> None:
    client = _Wrap(_client(), at=1, error=_sdk_error("RateLimitError"), times=MAX_EPISODE_ATTEMPTS)
    summary = run_split("dev", client, run_dir=tmp_path / "run", clock=ZERO)
    first = summary.results[0]
    assert (first.outcome, first.attempts, first.trajectory_id, first.cost_usd) == ("aborted", MAX_EPISODE_ATTEMPTS, None, None)
    assert all(r.outcome == "success" for r in summary.results[1:])


def test_agent_output_error_is_not_retried(tmp_path: Path) -> None:
    client = _Wrap(_client(), at=2, error=AdapterError("truncated"))
    summary = run_split("dev", client, run_dir=tmp_path / "run", clock=ZERO)
    first = summary.results[0]
    assert (first.outcome, first.attempts, first.termination) == ("agent_error", 1, None)
    # 第一個 call 成功回傳，已記帳；中止的嘗試也算進 spent
    assert summary.spent_usd == pytest.approx(CALL_COST * summary.model_calls)


def test_harness_error_propagates_and_writes_no_summary(tmp_path: Path) -> None:
    error = KeyError("harness bug")
    client = _Wrap(_client(), at=3, error=error)
    with pytest.raises(KeyError) as info:
        run_split("dev", client, run_dir=tmp_path / "run", clock=ZERO)
    assert info.value is error
    assert not (tmp_path / "run" / "summary.json").exists()
    rows = read_manifest(tmp_path / "run" / "manifest.jsonl")
    assert rows[-1].error_class == "harness"


# ---------------------------------------------------------------- CostMeter


class _Stub:
    def __init__(self, costs: list[Any]) -> None:
        self._costs = costs

    @property
    def agent_config(self) -> Any:
        return baseline_agent_config()

    def next_message(self, messages: Any, tools: Any) -> PolicyStep:
        cost = self._costs.pop(0)
        if isinstance(cost, BaseException):
            raise cost
        return PolicyStep(ModelMessage("", (), "end_turn"), 0, 0, cost)


def test_cost_meter_accumulates_known_and_unknown_costs() -> None:
    ledger = Ledger()
    meter = CostMeter(_Stub([0.5, None, ValueError("x"), 0.25]), ledger)
    assert meter.agent_config.config_id == baseline_agent_config().config_id
    meter.next_message([], ())
    meter.next_message([], ())
    with pytest.raises(ValueError):
        meter.next_message([], ())
    meter.next_message([], ())
    assert (ledger.cost_usd, ledger.calls, ledger.unknown_cost_calls) == (0.75, 3, 1)


def test_cost_meters_can_share_a_ledger() -> None:
    ledger = Ledger()
    CostMeter(_Stub([0.1]), ledger).next_message([], ())
    CostMeter(_Stub([0.2]), ledger).next_message([], ())
    assert ledger.cost_usd == pytest.approx(0.3) and ledger.calls == 2


def test_oracle_client_disambiguates_tasks_with_identical_instructions(tmp_path: Path) -> None:
    """eval 中 gen-escalate-label-s1 / s2 的 instruction 相同但 oracle 不同：每個 episode 都必須照自己的 oracle 走。"""
    run_dir = tmp_path / "run"
    summary = run_split("eval", _client("eval"), run_dir=run_dir, clock=ZERO)
    instructions = [inst.task.instruction for inst in week1_pool("eval")]
    assert len(set(instructions)) < len(instructions)  # 前提：確實有重複的 instruction
    for inst in week1_pool("eval"):
        trajectory = Trajectory.load(run_dir / "trajectories" / f"{inst.task.task_id}-r0.json")
        calls = [(c.name, canonical_json(c.args)) for turn in trajectory.turns for c in turn.message.tool_calls]
        assert calls == [(name, canonical_json(args)) for name, args in inst.oracle_calls]
    assert all(r.outcome == "success" and r.trajectory_safe for r in summary.results)
