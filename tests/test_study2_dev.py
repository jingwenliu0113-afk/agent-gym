"""experiments/study2_dev.py（decision 034 §5 C、§7、§8）的契約測試；不呼叫任何模型。"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments import study2_dev as dev
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from tasks.suites import suite_pool

DEV = suite_pool("study2", "dev")
BY_ID = {i.task.task_id: i for i in DEV}


def ZERO() -> float:  # noqa: N802
    return 0.0


def _pool_with(changes: dict[str, tuple]) -> tuple:
    return tuple(dataclasses.replace(i, oracle_calls=changes[i.task.task_id]) if i.task.task_id in changes else i for i in DEV)


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    alt = dict(BY_ID["gen2-sync-repair-L2-s1"].alternative_calls)["comment-variant"]  # 正確、但 comment 文字不同（不算差異）
    neg = dict(BY_ID["gen2-scoped-triage-L1-s0"].negative_calls)["S1:edit-then-revert"]  # 成功但不安全
    fail = dict(BY_ID["gen2-verified-close-L2-s0"].negative_calls)["V2:no-comment"]
    pool = _pool_with({"gen2-sync-repair-L2-s1": alt, "gen2-scoped-triage-L1-s0": neg, "gen2-verified-close-L2-s0": fail})
    run_dir = tmp_path_factory.mktemp("dev") / "d"
    run_split("dev", OllamaOracleFakeClient(pool), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True, suite="study2")
    return run_dir


def test_review_classifies_episodes(dev_run: Path) -> None:
    data = dev.review([dev_run], allow_dry_run=True)
    assert len(data["episodes"]) == 24 and data["unfinished"] == [] and data["missing"] == []
    assert data["suspected_grader_issues"] == []  # comment 文字不同不算差異；成功的 episode 都與 oracle 相同
    ids = sorted(e["task_id"] for e in data["to_classify"])
    assert ids == ["gen2-scoped-triage-L1-s0", "gen2-verified-close-L2-s0"]
    unsafe = next(e for e in data["to_classify"] if e["task_id"] == "gen2-scoped-triage-L1-s0")
    assert unsafe["success"] and not unsafe["safe"] and {v["rule"] for v in unsafe["violations"]} == {"S1"}
    failed = next(e for e in data["to_classify"] if e["task_id"] == "gen2-verified-close-L2-s0")
    assert {c["rule"] for c in failed["failed_checks"]} == {"V2"} and failed["calls"]
    assert data["levels"]["L2"]["success_rate"] == pytest.approx(7 / 8)
    assert "疑似 grader 漏洞" in dev.render_review(data)


def test_review_rejects_bad_inputs(dev_run: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dry-run"):
        dev.review([dev_run])  # dry-run 只供測試
    with pytest.raises(ValueError, match="more than one directory"):
        dev.review([dev_run, dev_run], allow_dry_run=True)
    copy = tmp_path / "copy"
    shutil.copytree(dev_run, copy)
    (copy / "summary.json").unlink()
    with pytest.raises(ValueError, match="interrupted"):
        dev.review([copy], allow_dry_run=True)
    week1 = suite_pool("week1", "dev")
    run_split("dev", OllamaOracleFakeClient(week1), run_dir=tmp_path / "w", clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    with pytest.raises(ValueError, match="Study 2"):
        dev.review([tmp_path / "w"], allow_dry_run=True)


def test_review_rejects_trajectories_of_another_task_version(dev_run: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(dev_run, copy)
    path = copy / "trajectories" / "gen2-epic-closeout-L1-s0-r0.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["task"]["instruction"] += " (older wording)"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        dev.review([copy], allow_dry_run=True)


def test_unfinished_episodes_count_as_failures(dev_run: Path, tmp_path: Path) -> None:
    """agent_error / aborted 沒有 trajectory，但計入成功率的分母（與 §7 相同；驗證 B 第二輪）。"""
    copy = tmp_path / "copy"
    shutil.copytree(dev_run, copy)
    summary = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    target = next(r for r in summary["results"] if r["task_id"] == "gen2-epic-closeout-L3-s0")
    target["outcome"] = "agent_error"
    (copy / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (copy / "trajectories" / "gen2-epic-closeout-L3-s0-r0.json").unlink()
    data = dev.review([copy], allow_dry_run=True)
    assert data["levels"]["L3"]["agent_error"] == 1 and data["levels"]["L3"]["episodes"] == 8
    assert data["levels"]["L3"]["success_rate"] == pytest.approx(7 / 8)
    assert "計為未成功" in dev.render_review(data)


def test_state_diff_reports_fields_and_comment_counts() -> None:
    instance = BY_ID["gen2-sync-repair-L2-s0"]
    oracle = dev.oracle_final_state(instance)
    assert dev.state_diff(oracle, oracle) == []
    diffs = dev.state_diff(instance.task.initial_state_obj(), oracle)
    assert {d["field"] for d in diffs} >= {"comment_count"}


@pytest.mark.parametrize(
    ("body", "flagged"),
    [
        ("This no longer blocks #{e}.", True),
        ("Unblocks #{e}.", True),
        ("It doesn’t blocks #{e}.", True),  # 彎引號
        ("Blocks #{e}? Update: this no longer blocks #{e}.", True),  # 第二次出現才是否定
        ("Still open. Blocks #{e}.", False),
    ],
)
def test_negated_comments_are_flagged_for_review(body: str, flagged: bool, tmp_path: Path) -> None:
    instance = BY_ID["gen2-epic-closeout-L2-s1"]  # 奇數 seed：有阻擋的分支
    epic = instance.task.params["epic"]
    calls = tuple((n, {**a, "body": body.format(e=epic)}) if n == "add_comment" else (n, a) for n, a in instance.oracle_calls)
    run_split("dev", OllamaOracleFakeClient(_pool_with({instance.task.task_id: calls})), run_dir=tmp_path / "d", clock=ZERO,
              agent_config=local_agent_config(), dry_run=True, suite="study2")
    data = dev.review([tmp_path / "d"], allow_dry_run=True)
    hit = [e for e in data["suspected_grader_issues"] if e["task_id"] == instance.task.task_id]
    assert bool(hit) is flagged


def test_estimate_is_deterministic_and_follows_the_registered_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = {"L1": {"mean_turns": 6, "success_rate": 0.85}, "L2": {"mean_turns": 10, "success_rate": 0.6},
              "L3": {"mean_turns": 13, "success_rate": 0.35}}
    a = dev.estimate_sweep_calls(levels, source_rule="rollout0", reps=20)
    assert a == dev.estimate_sweep_calls(levels, source_rule="rollout0", reps=20) and a["p50"] <= a["p90"]
    b = dev.estimate_sweep_calls(levels, source_rule="first_failure", reps=20)
    assert b["mean"] > a["mean"]  # 失敗的 source 較多 → refine 較多
    seen: list[tuple[float, ...]] = []
    monkeypatch.setattr(dev, "_sweep_cost", lambda curve, plan, rng, **kw: (seen.append(tuple(curve)) or (1, {})))
    dev.estimate_sweep_calls({lv: {"mean_turns": 5, "success_rate": 1.0} for lv in ("L1", "L2", "L3")}, source_rule="first_failure", reps=1)
    assert all(curve == (1.0,) * 5 for curve in seen) and len(seen) == 36  # 全部成功：由 p = 1 升到 1
    seen.clear()
    dev.estimate_sweep_calls({lv: {"mean_turns": 5, "success_rate": 0.0} for lv in ("L1", "L2", "L3")}, source_rule="rollout0", reps=1)
    assert all(curve[-1] == 0.0 for curve in seen)  # 失敗的 source 降到 0
    with pytest.raises(ValueError):
        dev.estimate_sweep_calls(levels, source_rule="best", reps=1)
    with pytest.raises(ValueError, match="no measured"):
        dev.estimate_sweep_calls({**levels, "L3": {"mean_turns": None, "success_rate": None}}, source_rule="rollout0", reps=1)


def test_sweep_cost_counts_remaining_turns() -> None:
    calls, verdicts = dev._sweep_cost((1.0, 1.0, 1.0), dev.REGISTERED_PLAN, random.Random(0))
    # coarse：k = 0, 2（stride 2，last = 2），各 10 次；成本 3 與 1
    assert calls == 10 * 3 + 10 * 1 and verdicts == {"round1": (False, 0)}


def test_power_compares_the_two_rules() -> None:
    rows = dev.max_turns_power(reps=6)
    assert {r["length"] for r in rows} == set(dev.POWER_LENGTHS) and {r["turn_out"] for r in rows} == set(dev.POWER_TURN_OUT)
    summary = dev.summarize_power(rows)
    assert len(summary) == len(dev.POWER_LENGTHS) * len(dev.POWER_TURN_OUT)
    for s in summary:
        for rule in ("round1", "excluding"):
            assert 0 <= s[f"mean_detect_{rule}"] <= 1 and 0 <= s[f"max_false_flag_{rule}"] <= 1
    worst = next(s for s in summary if s["length"] == 13 and s["turn_out"] == 0.2)
    assert worst["mean_detect_round1"] < worst["mean_detect_excluding"]
