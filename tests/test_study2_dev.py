"""experiments/study2_dev.py（decision 034 §5 C、§7、§8）的契約測試；不呼叫任何模型。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments import study2_dev as dev
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from tasks.suites import suite_pool

DEV = suite_pool("study2", "dev")


def ZERO() -> float:  # noqa: N802
    return 0.0


def _pool_with(changes: dict[str, tuple]) -> tuple:
    out = []
    for instance in DEV:
        if instance.task.task_id in changes:
            instance = dataclasses.replace(instance, oracle_calls=changes[instance.task.task_id])
        out.append(instance)
    return tuple(out)


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    by_id = {i.task.task_id: i for i in DEV}
    alt = dict(by_id["gen2-sync-repair-L2-s1"].alternative_calls)["comment-variant"]  # 正確、但 comment 文字不同（不算差異）
    neg = dict(by_id["gen2-scoped-triage-L1-s0"].negative_calls)["S1:edit-then-revert"]  # 成功但不安全
    fail = dict(by_id["gen2-verified-close-L2-s0"].negative_calls)["V2:no-comment"]
    pool = _pool_with({"gen2-sync-repair-L2-s1": alt, "gen2-scoped-triage-L1-s0": neg, "gen2-verified-close-L2-s0": fail})
    run_dir = tmp_path_factory.mktemp("dev") / "d"
    run_split("dev", OllamaOracleFakeClient(pool), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True,
              suite="study2")
    return run_dir


def test_review_classifies_episodes(dev_run: Path) -> None:
    data = dev.review([dev_run])
    assert len(data["episodes"]) == 24 and data["unfinished"] == []
    assert data["suspected_grader_issues"] == []  # comment 文字不同不算差異；成功的 episode 都與 oracle 相同
    ids = sorted(e["task_id"] for e in data["to_classify"])
    assert ids == ["gen2-scoped-triage-L1-s0", "gen2-verified-close-L2-s0"]
    unsafe = next(e for e in data["to_classify"] if e["task_id"] == "gen2-scoped-triage-L1-s0")
    assert unsafe["success"] and not unsafe["safe"] and {v["rule"] for v in unsafe["violations"]} == {"S1"}
    failed = next(e for e in data["to_classify"] if e["task_id"] == "gen2-verified-close-L2-s0")
    assert {c["rule"] for c in failed["failed_checks"]} == {"V2"} and failed["calls"]
    assert data["levels"]["L2"]["success_rate"] == pytest.approx(7 / 8)
    assert "疑似 grader 漏洞" in dev.render_review(data)


def test_state_diff_reports_fields_and_comment_counts() -> None:
    instance = next(i for i in DEV if i.task.task_id == "gen2-sync-repair-L2-s0")
    oracle = dev.oracle_final_state(instance)
    assert dev.state_diff(oracle, oracle) == []
    initial = instance.task.initial_state_obj()
    diffs = dev.state_diff(initial, oracle)
    assert {d["field"] for d in diffs} >= {"comment_count"}


def test_review_rejects_non_study2_runs(tmp_path: Path) -> None:
    week1 = suite_pool("week1", "dev")
    run_split("dev", OllamaOracleFakeClient(week1), run_dir=tmp_path / "w", clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    with pytest.raises(ValueError, match="Study 2"):
        dev.review([tmp_path / "w"])


def test_estimate_is_deterministic_and_orders_the_source_rules() -> None:
    levels = {"L1": {"mean_turns": 6, "success_rate": 0.85}, "L2": {"mean_turns": 10, "success_rate": 0.6},
              "L3": {"mean_turns": 13, "success_rate": 0.35}}
    a = dev.estimate_sweep_calls(levels, source_rule="rollout0", reps=30)
    assert a == dev.estimate_sweep_calls(levels, source_rule="rollout0", reps=30)
    b = dev.estimate_sweep_calls(levels, source_rule="first_failure", reps=30)
    assert a["p50"] <= a["p90"] and b["mean"] > a["mean"]  # 失敗的 source 較多 → refine 較多
    ceiling = dev.estimate_sweep_calls({lv: {"mean_turns": 6, "success_rate": 1.0} for lv in ("L1", "L2", "L3")},
                                       source_rule="first_failure", reps=5)
    assert ceiling["p90"] == ceiling["p50"]  # 全部成功：只有 coarse，沒有隨機性
    with pytest.raises(ValueError):
        dev.estimate_sweep_calls(levels, source_rule="best", reps=1)


def test_sweep_cost_counts_remaining_turns() -> None:
    import random

    calls, is_cliff, _ = dev._sweep_cost((1.0, 1.0, 1.0), dev.REGISTERED_PLAN, random.Random(0))
    # coarse：k = 0, 2（stride 2，last = 2），各 10 次；成本 3 與 1
    assert calls == 10 * 3 + 10 * 1 and is_cliff is False


def test_max_turns_power_reports_detection_and_false_flags() -> None:
    rows = dev.max_turns_power(reps=4)
    assert {r["length"] for r in rows} == set(dev.POWER_LENGTHS) and {r["turn_out"] for r in rows} == set(dev.POWER_TURN_OUT)
    summary = dev.summarize_power(rows)
    assert len(summary) == len(dev.POWER_LENGTHS) * len(dev.POWER_TURN_OUT)
    assert all(0 <= s["mean_detect"] <= 1 and 0 <= s["max_false_flag"] <= 1 for s in summary)


def test_negated_comments_are_flagged_for_review(tmp_path: Path) -> None:
    """已知限制：必含字串是子字串比對，"no longer blocks #E" 也會通過；dev 審查會把它列為疑似 grader 漏洞（decision 034 驗證 B）。"""
    by_id = {i.task.task_id: i for i in DEV}
    instance = by_id["gen2-epic-closeout-L2-s1"]  # 奇數 seed：有阻擋的分支
    epic = instance.task.params["epic"]
    calls = tuple((name, {**args, "body": f"This no longer blocks #{epic}."}) if name == "add_comment" else (name, args)
                  for name, args in instance.oracle_calls)
    run_split("dev", OllamaOracleFakeClient(_pool_with({instance.task.task_id: calls})), run_dir=tmp_path / "d", clock=ZERO,
              agent_config=local_agent_config(), dry_run=True, suite="study2")
    data = dev.review([tmp_path / "d"])
    flagged = [e for e in data["suspected_grader_issues"] if e["task_id"] == instance.task.task_id]
    assert flagged and flagged[0]["success"] and flagged[0]["negated_comments"]
    assert all(not e["negated_comments"] for e in data["episodes"] if e["task_id"] != instance.task.task_id and e["success"])
