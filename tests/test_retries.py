"""replay.aborts.run_with_retries 的契約測試（decision 009：只重試 infrastructure，上限 MAX_EPISODE_ATTEMPTS）。"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import MAX_EPISODE_ATTEMPTS, AdapterError
from agents.policy import PolicyStep, oracle_policy
from replay.aborts import RetryResult, RolloutKey, read_manifest, run_with_retries
from replay.runner import run_episode
from replay.trajectory import Trajectory
from tasks.smoke import REFERENCE_SOLUTIONS, get_smoke_task

TASK_ID = "smoke-assign-and-start"


def _fake_sdk_error(name: str) -> Exception:
    cls = type(name, (Exception,), {"__module__": "anthropic"})
    return cls(f"{name} (fake)")


class _Failing:
    """前 fail_at 次 next_message 照 oracle 回應，之後拋出 error。"""

    def __init__(self, error: BaseException | None, fail_at: int = 0) -> None:
        self._inner = oracle_policy(REFERENCE_SOLUTIONS[TASK_ID])
        self._error = error
        self._fail_at = fail_at
        self._calls = 0

    @property
    def agent_config(self) -> Any:
        return self._inner.agent_config

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        if self._error is not None and self._calls >= self._fail_at:
            raise self._error
        self._calls += 1
        return self._inner.next_message(messages, tools)


def _run(policy: Any) -> Trajectory:
    return run_episode(get_smoke_task(TASK_ID), policy, max_turns=10, clock=lambda: 0.0)


def _key(rollout_index: int = 0, attempt: int = 1) -> RolloutKey:
    return RolloutKey(None, None, None, rollout_index, attempt)


def _retry(factory: Any, log_path: Path, key: RolloutKey | None = None) -> RetryResult:
    return run_with_retries(
        _run, factory, task=get_smoke_task(TASK_ID), key=key or _key(), max_turns=10, log_path=log_path
    )


def test_completed_first_attempt(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    seen: list[int] = []

    def factory(attempt: int) -> Any:
        seen.append(attempt)
        return _Failing(None)

    result = _retry(factory, log)
    assert isinstance(result, RetryResult)
    assert isinstance(result.trajectory, Trajectory) and result.trajectory.grade().success
    assert (result.attempts, result.error_class) == (1, None)
    assert seen == [1]
    assert [(r.attempt, r.outcome) for r in read_manifest(log)] == [(1, "completed")]


def test_infrastructure_abort_is_retried_once_then_completes(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    policies = {1: _Failing(_fake_sdk_error("APIConnectionError"), fail_at=1), 2: _Failing(None)}
    result = _retry(lambda attempt: policies[attempt], log)
    assert result.trajectory is not None and (result.attempts, result.error_class) == (2, None)
    rows = read_manifest(log)
    assert [(r.attempt, r.outcome, r.error_class) for r in rows] == [(1, "aborted", "infrastructure"), (2, "completed", None)]


def test_infrastructure_abort_stops_at_max_attempts(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    calls: list[int] = []

    def factory(attempt: int) -> Any:
        calls.append(attempt)
        return _Failing(_fake_sdk_error("RateLimitError"))

    result = _retry(factory, log)
    assert result == RetryResult(None, MAX_EPISODE_ATTEMPTS, "infrastructure")
    assert calls == list(range(1, MAX_EPISODE_ATTEMPTS + 1))
    assert [r.attempt for r in read_manifest(log)] == list(range(1, MAX_EPISODE_ATTEMPTS + 1))


def test_agent_output_abort_is_not_retried(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    calls: list[int] = []

    def factory(attempt: int) -> Any:
        calls.append(attempt)
        return _Failing(AdapterError("truncated"), fail_at=1)

    assert _retry(factory, log) == RetryResult(None, 1, "agent_output")
    assert calls == [1]
    (row,) = read_manifest(log)
    assert (row.outcome, row.error_class, row.model_calls_completed) == ("aborted", "agent_output", 1)


def test_harness_error_propagates_without_retry(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    error = KeyError("bug")
    calls: list[int] = []

    def factory(attempt: int) -> Any:
        calls.append(attempt)
        return _Failing(error)

    with pytest.raises(KeyError) as info:
        _retry(factory, log)
    assert info.value is error
    assert calls == [1]
    assert [(r.outcome, r.error_class) for r in read_manifest(log)] == [("aborted", "harness")]


@pytest.mark.parametrize("attempt", [0, 2, 3])
def test_key_attempt_must_be_one(attempt: int, tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    with pytest.raises(ValueError):
        _retry(lambda a: _Failing(None), log, key=_key(attempt=attempt))
    assert read_manifest(log) == ()


def test_existing_key_is_rejected_before_running(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    _retry(lambda a: _Failing(None), log)
    calls: list[int] = []

    def factory(attempt: int) -> Any:
        calls.append(attempt)
        return _Failing(None)

    with pytest.raises(ValueError):
        _retry(factory, log)
    assert len(read_manifest(log)) == 1


def test_missing_manifest_row_is_a_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_logged 回傳 None 卻找不到該列 → RuntimeError（manifest 與執行不一致是 harness 問題）。"""
    import replay.aborts as aborts

    monkeypatch.setattr(aborts, "run_logged", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError):
        _retry(lambda a: _Failing(None), tmp_path / "m.jsonl")


def test_retry_key_keeps_all_other_fields(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    policies = {1: _Failing(_fake_sdk_error("InternalServerError")), 2: _Failing(None)}
    _retry(lambda attempt: policies[attempt], log, key=_key(rollout_index=3))
    assert [(r.rollout_index, r.attempt) for r in read_manifest(log)] == [(3, 1), (3, 2)]
    assert dataclasses.fields(RetryResult)
