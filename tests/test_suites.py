"""tasks/suites.py：第一輪（week1）與 Study 2（study2）的題目集合（decision 034）。"""

from __future__ import annotations

import pytest

from tasks import suites
from tasks.generator import week1_pool
from tasks.generator_v2 import study2_pool
from tasks.suites import (
    SUITES,
    eval_task_ids,
    suite_for_generator,
    suite_for_task_ids,
    suite_pool,
)


def test_suites_and_pools() -> None:
    assert SUITES == ("week1", "study2")
    for split in ("dev", "eval"):
        assert [i.task.task_id for i in suite_pool("week1", split)] == [i.task.task_id for i in week1_pool(split)]
        assert [i.task.task_id for i in suite_pool("study2", split)] == [i.task.task_id for i in study2_pool(split)]
    with pytest.raises(ValueError):
        suite_pool("week2", "eval")
    with pytest.raises(ValueError):
        suite_pool("study2", "all")


def test_suite_for_generator() -> None:
    assert suite_for_generator("issue-tracker-gen/v1") == "week1"
    assert suite_for_generator("issue-tracker-gen/v2") == "study2"
    with pytest.raises(ValueError):
        suite_for_generator("handwritten/v2")


def test_suite_for_task_ids_requires_exactly_an_eval_pool() -> None:
    assert suite_for_task_ids(eval_task_ids("week1")) == "week1"
    assert suite_for_task_ids(list(reversed(eval_task_ids("study2")))) == "study2"
    with pytest.raises(ValueError):
        suite_for_task_ids(eval_task_ids("study2")[:-1])
    with pytest.raises(ValueError):
        suite_for_task_ids([*eval_task_ids("week1"), *eval_task_ids("study2")])


def test_pools_are_cached_and_shared() -> None:
    assert suite_pool("study2", "eval") is suite_pool("study2", "eval")


def test_study2_model_runs_are_approved_but_not_frozen() -> None:
    """decision 034：使用者已確認（2026-10-17），dev 可以推論；凍結前不對 eval seed 做推論。"""
    assert suites.STUDY2_APPROVED is True
    assert suites.STUDY2_FROZEN is False
    assert suites.STUDY2_SWEEP_CAP == 70_000
    suites.check_model_run("study2", "dev")
    with pytest.raises(ValueError, match="frozen"):
        suites.check_model_run("study2", "eval")
    suites.check_model_run("week1", "eval")


def test_the_approval_lock_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suites, "STUDY2_APPROVED", False)
    with pytest.raises(ValueError, match="approv"):
        suites.check_model_run("study2", "dev")


def test_model_run_guard_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suites, "STUDY2_APPROVED", True)
    suites.check_model_run("study2", "dev")
    with pytest.raises(ValueError, match="frozen"):
        suites.check_model_run("study2", "eval")
    monkeypatch.setattr(suites, "STUDY2_FROZEN", True)
    suites.check_model_run("study2", "eval")
