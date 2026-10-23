"""Study 3 的 dev 審查與估算（decision 036 §5 C、§7）：兩個 agent、comment 稽核、各級別的 turn 與 max_turns、估算。只用 dev instance。"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from agents.baseline import haiku_agent_config
from agents.ollama import local_agent_config
from experiments import study3_dev as dev
from experiments.run_baseline import OllamaOracleFakeClient, OracleFakeClient, run_split
from experiments.study3_context import LIMIT_BYTES
from tasks.suites import suite_pool

DEV = suite_pool("study3", "dev")


def ZERO() -> float:  # noqa: N802
    return 0.0


def _replace_text(instance, old: str, new: str):
    """oracle 的 comment 文字改寫（只用 dev instance）。"""
    calls = tuple((name, {**args, "body": args["body"].replace(old, new)} if name == "add_comment" else args)
                  for name, args in instance.oracle_calls)
    return dataclasses.replace(instance, oracle_calls=calls)


def _tampered_pool() -> tuple:
    out = []
    for instance in DEV:
        task, p = instance.task, instance.task.params
        if task.task_id == "gen3-epic-closeout-H2-s0":  # #C 的 sub-issue 的 comment 寫成 #E 的 id：wrong-tracker（失敗）
            under_c = {int(i) for i, t in p["tracker_of"].items() if t == p["child"] and int(i) != p["child"]}
            sub = next(args["issue_id"] for name, args in instance.oracle_calls if name == "add_comment" and args["issue_id"] in under_c)
            calls = tuple((name, {**args, "body": re.sub(r"#\d+", f"#{p['epic']}", args["body"])}
                           if name == "add_comment" and args["issue_id"] == sub else args) for name, args in instance.oracle_calls)
            instance = dataclasses.replace(instance, oracle_calls=calls)
        elif task.task_id == "gen3-scoped-triage-H1-s0":  # 有權限的 needs-info 候選的 comment 另含 owner 文字（仍成功：只檢查「包含」）
            target = next(i for i in p["permitted"] if i in p["needs_info"])
            calls = tuple((name, {**args, "body": args["body"] + " Needs owner triage."}
                           if name == "add_comment" and args["issue_id"] == target else args) for name, args in instance.oracle_calls)
            instance = dataclasses.replace(instance, oracle_calls=calls)
        out.append(instance)
    return tuple(out)


@pytest.fixture(scope="module")
def local_dev(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("d3") / "local"
    run_split("dev", OllamaOracleFakeClient(_tampered_pool()), run_dir=run_dir, clock=ZERO, suite="study3",
              agent_config=local_agent_config(), dry_run=True, rollouts=1)
    return run_dir


@pytest.fixture(scope="module")
def haiku_dev(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("d3h") / "haiku"
    run_split("dev", OracleFakeClient(DEV), run_dir=run_dir, clock=ZERO, suite="study3", agent_config=haiku_agent_config(),
              dry_run=True, rollouts=1)
    return run_dir


def test_review_reports_every_level_and_the_audits(local_dev: Path) -> None:
    data = dev.review([local_dev], agent="local", allow_dry_run=True)
    assert set(data["levels"]) == {"H1", "H2", "H3"}
    assert all(m["episodes"] == 8 and m["max_turns_rate"] == 0 and m["context_exhausted"] == 0 for m in data["levels"].values())
    kinds = {(a["task_id"], a["kind"], a["success"]) for a in data["comment_audit"]}
    assert ("gen3-epic-closeout-H2-s0", "wrong-tracker", False) in kinds
    assert ("gen3-scoped-triage-H1-s0", "owner-text-on-permitted", True) in kinds
    assert any(e["task_id"] == "gen3-scoped-triage-H1-s0" for e in data["suspected_grader_issues"])
    assert any(e["task_id"] == "gen3-epic-closeout-H2-s0" for e in data["to_classify"])
    text = dev.render_review(data)
    assert "comment 稽核" in text and "wrong-tracker" in text
    for lv, m in data["levels"].items():  # §3：dev 也逐級別報告最大的 request 大小（由 trajectory 重建）
        sizes = [e["max_request_bytes"] for e in data["episodes"] if e["level"] == lv]
        assert m["max_request_bytes"] == max(sizes) and 0 < max(sizes) <= LIMIT_BYTES
    assert "最大 request（bytes" in text


def test_review_checks_the_agent_and_refuses_dry_runs_by_default(local_dev: Path, haiku_dev: Path) -> None:
    with pytest.raises(ValueError, match="frozen haiku agent"):
        dev.review([local_dev], agent="haiku", allow_dry_run=True)
    with pytest.raises(ValueError, match="dry-run"):
        dev.review([local_dev], agent="local")
    with pytest.raises(ValueError):
        dev.review([local_dev], agent="opus", allow_dry_run=True)
    assert dev.review([haiku_dev], agent="haiku", allow_dry_run=True)["agent"] == "haiku"


def test_review_refuses_an_old_task_version(local_dev: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "old"
    shutil.copytree(local_dev, copy)
    path = next((copy / "trajectories").glob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["task"]["instruction"] += " "
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        dev.review([copy], agent="local", allow_dry_run=True)


def test_estimates_for_both_agents(local_dev: Path, haiku_dev: Path) -> None:
    local = dev.dev_estimate(dev.review([local_dev], agent="local", allow_dry_run=True), reps=200)
    assert local["agent"] == "local" and local["n"] >= 15 and not local["ask_user"]
    haiku = dev.dev_estimate(dev.review([haiku_dev], agent="haiku", allow_dry_run=True), haiku_dev_spent_usd=0.5, reps=200)
    assert haiku["agent"] == "haiku" and haiku["haiku_call_usd_p90"] > 0 and haiku["haiku_baseline_p90_usd"] > 0
