"""decision 006 / 017 §1：伴隨 max_turns 比例上升的下降不得認定為 cliff；decision 009 / 017 §2：sweep rollout 的 trajectory 存檔。

規則（decision 017 §1）：
- separated = 最大下降 >= drop_threshold 且兩點 CI（依 criterion）不重疊（即原本的 is_cliff）；
- max_turns_confounded = separated 且後一點的 max_turns 比例（max_turns_terminations / n，精確有理數）嚴格大於前一點；
- is_cliff = separated 且非 max_turns_confounded；只看最大下降，被混淆時不改看第二大的下降。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from agents.policy import AgentConfig, ModelMessage, PolicyStep, ToolCall
from replay.aborts import read_manifest
from replay.localization import (
    CheckpointEstimate,
    DropAssessment,
    SweepPlan,
    SweepResult,
    assess_drop,
    detect_cliff,
    sweep,
)
from replay.stats import wilson_interval
from replay.trajectory import Trajectory
from tests.test_localization import SCRIPTED_TASK, _clock, _source, _task_and_oracle


def _est(k: int, successes: int, n: int, max_turns: int = 0) -> CheckpointEstimate:
    low, high = wilson_interval(successes, n)
    return CheckpointEstimate(
        k=k,
        n=n,
        successes=successes,
        failures=n - successes,
        refusals=0,
        agent_errors=0,
        aborted=0,
        max_turns_terminations=max_turns,
        stopped_without_action=0,
        p_hat=successes / n if n else 0.0,
        p_hat_completed=successes / n if n else None,
        ci_low=low,
        ci_high=high,
    )


def test_drop_assessment_fields() -> None:
    assert [f for f in DropAssessment.__dataclass_fields__] == [
        "largest_drop",
        "largest_drop_k",
        "separated",
        "max_turns_confounded",
        "is_cliff",
    ]
    assert "max_turns_confounded" in SweepResult.__dataclass_fields__


@pytest.mark.parametrize("estimates", [(), (_est(0, 8, 8),)])
def test_fewer_than_two_points(estimates: tuple[CheckpointEstimate, ...]) -> None:
    assert assess_drop(estimates, 0.3) == DropAssessment(0.0, None, False, False, False)


def test_drop_with_rising_max_turns_rate_is_not_a_cliff() -> None:
    estimates = (_est(0, 24, 24, 0), _est(1, 0, 24, 24))
    result = assess_drop(estimates, 0.3)
    assert result == DropAssessment(1.0, 0, True, True, False)
    assert detect_cliff(estimates, 0.3) == (1.0, 0, False)


def test_any_strict_increase_of_the_rate_confounds() -> None:
    estimates = (_est(0, 24, 24, 0), _est(1, 0, 24, 1))
    assert assess_drop(estimates, 0.3).max_turns_confounded is True


@pytest.mark.parametrize(
    "before_after",
    [((0, 24), (0, 24)), ((3, 24), (3, 24)), ((6, 24), (2, 24)), ((1, 8), (3, 24))],  # 相同、下降、相同比例不同 n
)
def test_equal_or_falling_max_turns_rate_keeps_the_cliff(before_after: tuple[tuple[int, int], tuple[int, int]]) -> None:
    (mt_a, n_a), (mt_b, n_b) = before_after
    estimates = (_est(0, n_a, n_a, mt_a), _est(1, 0, n_b, mt_b))
    result = assess_drop(estimates, 0.3)
    assert (result.separated, result.max_turns_confounded, result.is_cliff) == (True, False, True)


def test_rate_comparison_is_exact() -> None:
    """1/3 與 8/24 相同 → 不算上升（以精確有理數比較，不受浮點影響）。"""
    estimates = (_est(0, 24, 24, 8), _est(1, 0, 3, 1))
    result = assess_drop(estimates, 0.3)
    assert result.separated is True and result.max_turns_confounded is False


def test_confound_only_matters_when_the_drop_is_separated() -> None:
    estimates = (_est(0, 2, 2, 0), _est(1, 1, 2, 2))  # 下降 0.5 但 CI 重疊
    result = assess_drop(estimates, 0.3)
    assert result == DropAssessment(0.5, 0, False, False, False)


def test_no_fallback_to_a_smaller_clean_drop() -> None:
    """1/3 的下降（0 → 1）本身合格且未混淆，但最大下降 2/3（1 → 2）被混淆 → 不是 cliff，也不改看較小的下降。"""
    estimates = (_est(0, 24, 24), _est(1, 16, 24), _est(2, 0, 24, 24))
    assert assess_drop(estimates[:2], 0.3).is_cliff is True  # 前提：較小的下降單獨看是 cliff
    result = assess_drop(estimates, 0.3)
    assert result.largest_drop_k == 1 and result.max_turns_confounded is True and result.is_cliff is False


def test_ties_still_pick_the_smaller_k_before_the_confound_check() -> None:
    estimates = (_est(0, 24, 24), _est(1, 12, 24), _est(2, 0, 24, 24))  # 兩個 0.5 的下降；k 較小者未混淆
    result = assess_drop(estimates, 0.3)
    assert result.largest_drop_k == 0 and result.max_turns_confounded is False and result.is_cliff is True


def test_assess_drop_agrees_with_detect_cliff_and_checks_arguments() -> None:
    estimates = (_est(0, 20, 24, 0), _est(2, 22, 24, 0), _est(4, 0, 24, 2), _est(6, 0, 24, 0))
    for criterion in ("wilson95", "bonferroni95"):
        a = assess_drop(estimates, 0.3, criterion=criterion)
        assert (a.largest_drop, a.largest_drop_k, a.is_cliff) == detect_cliff(estimates, 0.3, criterion=criterion)
    with pytest.raises(ValueError):
        assess_drop(estimates, 0.3, criterion="eyeball")
    with pytest.raises(ValueError):
        assess_drop((estimates[1], estimates[0]), 0.3)


# ================================================================ end-to-end：rerun 從 k >= 1 起用完 turn 預算


class _LoopPolicy:
    """k = 0：播放 oracle 後結束（成功）。k >= 1：重複唯讀的 search_issues 直到 max_turns（失敗，termination = max_turns）。"""

    def __init__(self, config: AgentConfig, oracle: tuple[tuple[str, Any], ...], *, start: int) -> None:
        self._config = copy.deepcopy(config)
        self._oracle = oracle
        self._start = start

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        position = sum(1 for message in messages if message["role"] == "assistant")
        if self._start == 0:
            call = self._oracle[position] if position < len(self._oracle) else None
        else:
            call = ("search_issues", {"status": "open"})
        if call is None:
            message = ModelMessage(text="Done.", tool_calls=(), stop_reason="end_turn")
        else:
            name, args = call
            message = ModelMessage(
                text="", tool_calls=(ToolCall(id=f"call_{position}", name=name, args=copy.deepcopy(args)),), stop_reason="tool_use"
            )
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


PLAN = SweepPlan(coarse_stride=3, n_coarse=8, n_fine=8, drop_threshold=0.3)  # n = 8：0/8 與 8/8 的 Wilson CI 不重疊


def _loop_sweep(tmp_path: Path, **kwargs: Any) -> tuple[Trajectory, SweepResult]:
    task, oracle = _task_and_oracle(SCRIPTED_TASK)
    source = _source(task, oracle, 0.0, 0)

    def factory(k: int, rollout_index: int, attempt: int) -> _LoopPolicy:
        return _LoopPolicy(source.agent, oracle, start=k)

    return source, sweep(source, factory, plan=PLAN, log_path=tmp_path / "manifest.jsonl", clock=_clock, **kwargs)


def test_sweep_does_not_call_a_max_turns_drop_a_cliff(tmp_path: Path) -> None:
    _, result = _loop_sweep(tmp_path)
    by_k = {e.k: e for e in result.estimates}
    assert by_k[0].p_hat == 1.0 and by_k[0].max_turns_terminations == 0
    assert all(e.max_turns_terminations == e.n for k, e in by_k.items() if k >= 1)
    assert by_k[1].p_hat == 0.0  # 最後一個 checkpoint 之前（k = 7 時 oracle 已做完，續跑雖用完 turns 仍成功）
    assert result.largest_drop == 1.0 and result.largest_drop_k == 0
    assessed = assess_drop(result.estimates, PLAN.drop_threshold, criterion=result.criterion)
    assert assessed.separated is True  # 沒有 decision 006 的規則就會是 cliff
    assert result.max_turns_confounded is True
    assert result.is_cliff is False
    assessed = assess_drop(result.estimates, PLAN.drop_threshold, criterion=result.criterion)
    assert (result.largest_drop, result.largest_drop_k, result.max_turns_confounded, result.is_cliff) == (
        assessed.largest_drop,
        assessed.largest_drop_k,
        assessed.max_turns_confounded,
        assessed.is_cliff,
    )


# ================================================================ rollout_dir（decision 017 §2）


def test_sweep_saves_every_completed_rollout_trajectory(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    source, result = _loop_sweep(tmp_path, rollout_dir=rollout_dir)
    rows = {(row.checkpoint_turn, row.rollout_index): row for row in read_manifest(tmp_path / "manifest.jsonl")}
    names = sorted(path.name for path in rollout_dir.iterdir())
    assert names == sorted(f"k{r.k}-r{r.rollout_index}.json" for r in result.rollouts)
    for r in result.rollouts:
        trajectory = Trajectory.load(rollout_dir / f"k{r.k}-r{r.rollout_index}.json")
        trajectory.verify()
        assert trajectory.trajectory_id == rows[(r.k, r.rollout_index)].trajectory_id
        assert trajectory.parent is not None and trajectory.parent["turn"] == r.k
        assert trajectory.parent["trajectory_id"] == source.trajectory_id
        assert trajectory.termination == r.termination


def test_sweep_without_rollout_dir_saves_nothing(tmp_path: Path) -> None:
    _loop_sweep(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.jsonl"]


def test_sweep_rejects_a_non_empty_rollout_dir_before_running(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    (rollout_dir / "k0-r0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        _loop_sweep(tmp_path, rollout_dir=rollout_dir)
    assert not (tmp_path / "manifest.jsonl").exists()


def test_sweep_creates_a_missing_rollout_dir(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "a" / "b"
    _loop_sweep(tmp_path, rollout_dir=str(rollout_dir))
    assert any(rollout_dir.iterdir())
