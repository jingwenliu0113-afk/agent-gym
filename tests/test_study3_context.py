"""Study 3 的 context 檢查（decision 036 §3「context 與 agent_error」，A 類）：以本機 agent 的 fit check 計算 request 大小。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments import study3_context as ctx
from tasks import generator_v3 as g3
from tests.test_generator_v3_core import KNOBS, TOY

ROOT = Path(__file__).resolve().parent.parent
DEV_CALLS = ROOT / "runs" / "study2-dev-20261017" / "calls.jsonl"


@pytest.fixture(autouse=True)
def toy(monkeypatch: pytest.MonkeyPatch):
    KNOBS.clear()
    monkeypatch.setattr(g3, "FAMILIES", (*g3.FAMILIES, "toy"))
    monkeypatch.setitem(g3._LOADED, "toy", TOY)
    yield
    KNOBS.clear()


def test_the_limits_follow_the_frozen_local_agent() -> None:
    options = local_agent_config().model_config["options"]
    assert ctx.LIMIT_BYTES == 2 * (options["num_ctx"] - options["num_predict"]) == 57_344
    assert ctx.BUDGET_BYTES == 45_875  # 0.8 × 上限（decision 036 §3）
    assert ctx.STRESS_SEARCHES == 2 and ctx.STRESS_TURNS == 20


@pytest.mark.skipif(not DEV_CALLS.is_file(), reason="runs/ 不在 repo；只在有 Study 2 dev 原始紀錄的機器上檢查")
def test_the_assistant_reserve_is_the_study2_dev_p95() -> None:
    assert ctx.assistant_reserve(DEV_CALLS) == ctx.ASSISTANT_RESERVE_BYTES == 886


def test_request_sizes_grow_with_the_conversation() -> None:
    instance = g3.generate("toy", "H1", 1)
    sizes = ctx.request_sizes(instance, instance.oracle_calls)
    assert len(sizes) == len(instance.oracle_calls) + 1  # 每個 tool call 一次，加上最後的回覆
    assert sizes == sorted(sizes) and sizes[0] > 1000


def test_a_small_task_passes_both_checks() -> None:
    report = ctx.check_context(g3.generate("toy", "H1", 1))
    assert report["ok"] and report["worst"] <= ctx.BUDGET_BYTES and report["stress"] <= ctx.LIMIT_BYTES
    assert set(report["solutions"]) == {"oracle", "search-first", "direct"}


def test_the_checks_fail_when_the_conversation_is_too_long(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = g3.generate("toy", "H1", 1)
    monkeypatch.setattr(ctx, "BUDGET_BYTES", 2000)
    report = ctx.check_context(instance)
    assert not report["ok"] and report["worst"] > 2000
    monkeypatch.setattr(ctx, "BUDGET_BYTES", 45_875)
    monkeypatch.setattr(ctx, "ASSISTANT_RESERVE_BYTES", 3000)  # 20 個 turn × 3000 > 57,344
    report = ctx.check_context(instance)
    assert not report["ok"] and report["stress"] > ctx.LIMIT_BYTES
