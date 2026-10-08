"""experiments/run_sweep.py 的契約測試：source 選取（decision 012 §3）、dry-run sweep、預算停止、彙總與事前檢查。"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import AdapterError, baseline_agent_config
from agents.policy import oracle_policy
from env.hashing import canonical_json
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import (
    SWEEP_SUMMARY_FORMAT,
    BudgetExceeded,
    SweepRunSummary,
    parse_plan,
    run_sweeps,
    select_sources,
)
from replay.aborts import read_manifest
from replay.localization import SweepPlan, SweepResult
from replay.runner import run_episode
from replay.trajectory import Trajectory
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
CALL_COST = (1000 * 4.00 + 200 * 20.00) / 1e6
SMALL_PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)
EVAL_IDS = [inst.task.task_id for inst in week1_pool("eval")]


@pytest.fixture(scope="module")
def eval_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("baseline") / "eval"
    run_split("eval", OracleFakeClient(week1_pool("eval")), run_dir=run_dir, clock=ZERO)
    return run_dir


def _copy(run_dir: Path, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(run_dir, target)
    return target


# ---------------------------------------------------------------- select_sources


def test_select_sources_takes_rollout_zero_of_every_eval_task(eval_run: Path) -> None:
    sources = select_sources(eval_run)
    assert list(sources) == EVAL_IDS
    summary = json.loads((eval_run / "summary.json").read_text(encoding="utf-8"))
    ids = {r["task_id"]: r["trajectory_id"] for r in summary["results"] if r["rollout_index"] == 0}
    for task_id, trajectory in sources.items():
        assert isinstance(trajectory, Trajectory)
        assert trajectory.trajectory_id == ids[task_id]
        assert trajectory.task["task_id"] == task_id


class _FailFor:
    """OracleFakeClient，但對指定 instruction 的第一次 call 拋出 AdapterError（agent_output，不重試）。"""

    def __init__(self, instruction: str) -> None:
        self._inner = OracleFakeClient(week1_pool("eval"))
        self._instruction = instruction
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        if request["messages"][0]["content"] == self._instruction:
            raise AdapterError("cut")
        return self._inner.create(**request)


def test_select_sources_marks_incomplete_rollout_zero_as_missing(tmp_path: Path) -> None:
    target = week1_pool("eval")[0].task
    run_dir = tmp_path / "run"
    run_split("eval", _FailFor(target.instruction), run_dir=run_dir, clock=ZERO)
    sources = select_sources(run_dir)
    assert sources[target.task_id] is None
    assert all(sources[t] is not None for t in EVAL_IDS if t != target.task_id)


def test_select_sources_rejects_dev_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "dev"
    run_split("dev", OracleFakeClient(week1_pool("dev")), run_dir=run_dir, clock=ZERO)
    with pytest.raises(ValueError):
        select_sources(run_dir)


def test_select_sources_rejects_tampered_or_missing_files(eval_run: Path, tmp_path: Path) -> None:
    copy = _copy(eval_run, tmp_path)
    victim = copy / "trajectories" / f"{EVAL_IDS[1]}-r0.json"
    other = copy / "trajectories" / f"{EVAL_IDS[2]}-r0.json"
    victim.write_bytes(other.read_bytes())  # 合法的 trajectory，但 id 與 summary 不符
    with pytest.raises(ValueError):
        select_sources(copy)
    victim.unlink()
    with pytest.raises(ValueError):
        select_sources(copy)


def test_select_sources_rejects_wrong_summary_format(eval_run: Path, tmp_path: Path) -> None:
    copy = _copy(eval_run, tmp_path)
    data = json.loads((copy / "summary.json").read_text(encoding="utf-8"))
    data["format"] = "agent-gym/baseline-run/v0"
    (copy / "summary.json").write_text(canonical_json(data) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        select_sources(copy)


# ---------------------------------------------------------------- run_sweeps（dry-run）


def test_dry_run_sweeps_every_source(eval_run: Path, tmp_path: Path) -> None:
    sources = select_sources(eval_run)
    run_dir = tmp_path / "sweep"
    client = OracleFakeClient(week1_pool("eval"))
    summary = run_sweeps(sources, client, plan=SMALL_PLAN, criterion="wilson95", run_dir=run_dir, clock=ZERO, replicates=200)
    assert isinstance(summary, SweepRunSummary)
    assert summary.sources == tuple(EVAL_IDS) and summary.completed == tuple(EVAL_IDS)
    assert summary.missing_sources == () and summary.stopped_by_budget is False
    assert summary.config_id == baseline_agent_config().config_id
    assert summary.families == {inst.task.task_id: inst.task.family for inst in week1_pool("eval")}
    for task_id in EVAL_IDS:
        result = summary.results[task_id]
        assert isinstance(result, SweepResult)
        assert result.task_id == task_id and result.source_trajectory_id == sources[task_id].trajectory_id
        assert all(e.p_hat == 1.0 for e in result.estimates)  # oracle 續跑一定成功
        assert result.is_cliff is False
        written = (run_dir / "sweeps" / f"{task_id}.json").read_text(encoding="utf-8")
        assert json.loads(written)["task_id"] == task_id
    assert summary.cliff_rate is not None and summary.cliff_rate.estimate == 0.0
    assert summary.cliff_rate.n_tasks == 15 and summary.cliff_rate.replicates == 200
    assert summary.mean_largest_drop is not None and summary.mean_largest_drop.estimate == pytest.approx(0.0)
    rows = read_manifest(run_dir / "manifest.jsonl")
    assert len(rows) == sum(len(r.rollouts) for r in summary.results.values())
    assert all(row.checkpoint_id is not None and row.outcome == "completed" for row in rows)
    assert summary.model_calls == len(client.requests)
    assert summary.spent_usd == pytest.approx(CALL_COST * summary.model_calls)
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert data["format"] == SWEEP_SUMMARY_FORMAT
    assert canonical_json(data) == canonical_json(summary.to_dict())


def test_missing_sources_are_reported_and_excluded(eval_run: Path, tmp_path: Path) -> None:
    sources = dict(select_sources(eval_run))
    sources[EVAL_IDS[0]] = None
    summary = run_sweeps(sources, OracleFakeClient(week1_pool("eval")), plan=SMALL_PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                         clock=ZERO, replicates=50)
    assert summary.missing_sources == (EVAL_IDS[0],)
    assert EVAL_IDS[0] not in summary.completed and EVAL_IDS[0] not in summary.results
    assert summary.cliff_rate is not None and summary.cliff_rate.n_tasks == 14


def test_bootstrap_is_none_when_a_family_has_no_completed_sweep(eval_run: Path, tmp_path: Path) -> None:
    sources = dict(select_sources(eval_run))
    family = week1_pool("eval")[0].task.family
    for inst in week1_pool("eval"):
        if inst.task.family == family:
            sources[inst.task.task_id] = None
    summary = run_sweeps(sources, OracleFakeClient(week1_pool("eval")), plan=SMALL_PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                         clock=ZERO, replicates=50)
    assert summary.cliff_rate is None and summary.mean_largest_drop is None
    assert len(summary.completed) == 12


def test_budget_stops_before_the_next_attempt(eval_run: Path, tmp_path: Path) -> None:
    sources = select_sources(eval_run)
    run_dir = tmp_path / "s"
    summary = run_sweeps(sources, OracleFakeClient(week1_pool("eval")), plan=SMALL_PLAN, criterion="wilson95", run_dir=run_dir,
                         max_cost_usd=CALL_COST / 2, clock=ZERO, replicates=50)
    assert summary.stopped_by_budget is True
    assert summary.completed == () and summary.results == {}
    assert len(read_manifest(run_dir / "manifest.jsonl")) == 1  # 第一個 rollout 之後就停止，沒有多餘的嘗試紀錄
    assert (run_dir / "summary.json").exists()
    assert summary.cliff_rate is None


def test_budget_exceeded_is_a_runtime_error() -> None:
    assert issubclass(BudgetExceeded, RuntimeError)


# ---------------------------------------------------------------- 事前檢查


def test_source_from_another_agent_is_rejected(eval_run: Path, tmp_path: Path) -> None:
    inst = week1_pool("eval")[0]
    foreign = run_episode(inst.task, oracle_policy(inst.oracle_calls), max_turns=20, clock=ZERO)
    sources = dict(select_sources(eval_run))
    sources[inst.task.task_id] = foreign
    client = OracleFakeClient(week1_pool("eval"))
    with pytest.raises(ValueError):
        run_sweeps(sources, client, plan=SMALL_PLAN, criterion="wilson95", run_dir=tmp_path / "s", clock=ZERO)
    assert client.requests == []


@pytest.mark.parametrize(
    "change",
    ["plan", "empty", "budget-zero", "budget-nan", "budget-bool", "run-dir"],
)
def test_bad_arguments_are_rejected_without_running(change: str, eval_run: Path, tmp_path: Path) -> None:
    sources: Any = select_sources(eval_run)
    kwargs: dict[str, Any] = {"plan": SMALL_PLAN, "criterion": "wilson95", "run_dir": tmp_path / "s", "clock": ZERO}
    if change == "plan":
        kwargs["plan"] = (3, 2, 2, 0.3)
    elif change == "empty":
        sources = {}
    elif change == "budget-zero":
        kwargs["max_cost_usd"] = 0
    elif change == "budget-nan":
        kwargs["max_cost_usd"] = math.nan
    elif change == "budget-bool":
        kwargs["max_cost_usd"] = True
    else:
        (tmp_path / "s").mkdir()
        (tmp_path / "s" / "x").write_text("x", encoding="utf-8")
    client = OracleFakeClient(week1_pool("eval"))
    with pytest.raises(ValueError):
        run_sweeps(sources, client, **kwargs)
    assert client.requests == []


# ---------------------------------------------------------------- parse_plan


def test_parse_plan() -> None:
    assert parse_plan("2,10,48,0.25") == SweepPlan(2, 10, 48, 0.25)
    assert parse_plan(" 3, 8, 32, 0.3 ") == SweepPlan(3, 8, 32, 0.3)
    for bad in ("", "2,10,48", "2,10,48,0.25,1", "a,10,48,0.25", "2.0,10,48,0.25", "2,10,4,0.25", "2,10,48,1.5"):
        with pytest.raises(ValueError):
            parse_plan(bad)


def test_summary_fields_are_stable() -> None:
    assert [f.name for f in dataclasses.fields(SweepRunSummary)] == [
        "plan",
        "criterion",
        "config_id",
        "sources",
        "missing_sources",
        "completed",
        "results",
        "families",
        "cliff_rate",
        "mean_largest_drop",
        "spent_usd",
        "model_calls",
        "unknown_cost_calls",
        "stopped_by_budget",
    ]


# ---------------------------------------------------------------- 驟降判定（decision 015）


def test_criterion_is_required_and_recorded(eval_run: Path, tmp_path: Path) -> None:
    sources = select_sources(eval_run)
    with pytest.raises(TypeError):
        run_sweeps(sources, OracleFakeClient(week1_pool("eval")), plan=SMALL_PLAN, run_dir=tmp_path / "a", clock=ZERO)  # type: ignore[call-arg]
    summary = run_sweeps(sources, OracleFakeClient(week1_pool("eval")), plan=SMALL_PLAN, criterion="bonferroni95",
                         run_dir=tmp_path / "b", clock=ZERO, replicates=20)
    assert summary.criterion == "bonferroni95"
    assert all(result.criterion == "bonferroni95" for result in summary.results.values())
    assert json.loads((tmp_path / "b" / "summary.json").read_text(encoding="utf-8"))["criterion"] == "bonferroni95"


def test_unknown_criterion_is_rejected_without_running(eval_run: Path, tmp_path: Path) -> None:
    client = OracleFakeClient(week1_pool("eval"))
    with pytest.raises(ValueError):
        run_sweeps(select_sources(eval_run), client, plan=SMALL_PLAN, criterion="other", run_dir=tmp_path / "s", clock=ZERO)
    assert client.requests == []


# ---------------------------------------------------------------- rollout 存檔與格式（decision 017 §1 / §2）


def test_sweep_output_format_is_v2() -> None:
    assert SWEEP_SUMMARY_FORMAT == "agent-gym/sweep-run/v2"


def test_dry_run_saves_every_completed_rollout_trajectory(eval_run: Path, tmp_path: Path) -> None:
    sources = select_sources(eval_run)
    run_dir = tmp_path / "sweep"
    summary = run_sweeps(sources, OracleFakeClient(week1_pool("eval")), plan=SMALL_PLAN, criterion="wilson95", run_dir=run_dir,
                         clock=ZERO, replicates=50)
    rows = {
        (row.task_id, row.checkpoint_turn, row.rollout_index): row
        for row in read_manifest(run_dir / "manifest.jsonl")
        if row.outcome == "completed"
    }
    saved = sorted(path.relative_to(run_dir / "rollouts").as_posix() for path in (run_dir / "rollouts").rglob("*.json"))
    assert saved == sorted(f"{task_id}/k{k}-r{i}.json" for task_id, k, i in rows)
    for task_id in summary.completed:
        result = summary.results[task_id]
        assert result.max_turns_confounded is False
        assert json.loads((run_dir / "sweeps" / f"{task_id}.json").read_text(encoding="utf-8"))["max_turns_confounded"] is False
        for r in result.rollouts:
            trajectory = Trajectory.load(run_dir / "rollouts" / task_id / f"k{r.k}-r{r.rollout_index}.json")
            assert trajectory.trajectory_id == rows[(task_id, r.k, r.rollout_index)].trajectory_id
            assert trajectory.parent["trajectory_id"] == sources[task_id].trajectory_id
