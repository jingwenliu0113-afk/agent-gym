"""experiments/demo_synthetic.py（README 首屏的 keyless 示範）的契約測試。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from agents.synthetic import SYNTHETIC_POLICY, mistake_call
from env.hashing import canonical_json
from experiments.demo_synthetic import (
    DEMO_CRITERION,
    DEMO_MISTAKE_TURN,
    DEMO_PLAN,
    DEMO_TASK_ID,
    DemoResult,
    find_source,
    render_demo,
    run_demo,
)
from replay.trajectory import Trajectory
from tasks.generator import week1_pool


def _task():
    return next(inst.task for inst in week1_pool("dev") if inst.task.task_id == DEMO_TASK_ID)


def _call_key(call) -> str:
    return canonical_json([call.name, call.args])


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[DemoResult, Path]:
    out = tmp_path_factory.mktemp("demo") / "out"
    return run_demo(out), out


def test_demo_task_is_a_dev_task() -> None:
    assert DEMO_TASK_ID in {inst.task.task_id for inst in week1_pool("dev")}


def test_find_source_is_deterministic_and_has_one_mistake_at_the_requested_turn() -> None:
    seed, source = find_source()
    seed2, source2 = find_source()
    assert seed == seed2 and source.trajectory_id == source2.trajectory_id
    assert isinstance(source, Trajectory) and source.agent.policy == SYNTHETIC_POLICY
    assert source.termination == "final_answer"
    name, args = mistake_call(_task())
    mistake = canonical_json([name, args])
    keys = [[_call_key(c) for c in turn.message.tool_calls] for turn in source.turns]
    positions = [i for i, calls in enumerate(keys) if calls == [mistake]]
    assert positions == [DEMO_MISTAKE_TURN]
    assert source.grade().success is False


def test_find_source_rejects_impossible_requests() -> None:
    with pytest.raises(RuntimeError):
        find_source(q=0.0)  # q = 0 永遠不會犯錯


def test_demo_localizes_the_cliff_at_the_mistake_turn(demo: tuple[DemoResult, Path]) -> None:
    result, _ = demo
    assert result.task_id == DEMO_TASK_ID and result.mistake_turn == DEMO_MISTAKE_TURN
    assert result.sweep.plan == DEMO_PLAN and result.sweep.criterion == DEMO_CRITERION
    assert result.sweep.source_trajectory_id == result.source.trajectory_id
    assert result.sweep.is_cliff is True
    assert result.sweep.largest_drop_k == DEMO_MISTAKE_TURN
    truth = dict(result.truth)
    assert sorted(truth) == [e.k for e in result.sweep.estimates]
    assert all(value > 0.6 for k, value in truth.items() if k <= DEMO_MISTAKE_TURN)  # (1 − 0.05)^(n − r + 1) >= 0.95^8 ≈ 0.66
    assert all(value == 0.0 for k, value in truth.items() if k > DEMO_MISTAKE_TURN)
    covered = sum(e.ci_low <= truth[e.k] <= e.ci_high for e in result.sweep.estimates)
    assert covered / len(result.sweep.estimates) >= 0.8
    assert result.source_grade.success is False and result.instruction == _task().instruction


def test_demo_writes_markdown_and_svg(demo: tuple[DemoResult, Path]) -> None:
    result, out = demo
    text = (out / "demo.md").read_text(encoding="utf-8")
    assert text == render_demo(result)
    for needle in ("synthetic", "不是 LLM", "Task", "Agent actions", "State changes", "Grader", "Replay from checkpoint"):
        assert needle in text, needle
    assert str(out) not in text
    root = ET.fromstring((out / "curve.svg").read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")


def test_demo_output_is_byte_for_byte_reproducible(demo: tuple[DemoResult, Path], tmp_path: Path) -> None:
    _, out = demo
    second = tmp_path / "again"
    run_demo(second)
    for name in ("demo.md", "curve.svg"):
        assert (second / name).read_bytes() == (out / name).read_bytes(), name


def test_rerunning_into_the_same_directory_overwrites(tmp_path: Path) -> None:
    run_demo(tmp_path / "x")
    first = (tmp_path / "x" / "demo.md").read_bytes()
    run_demo(tmp_path / "x")
    assert (tmp_path / "x" / "demo.md").read_bytes() == first


# ---------------------------------------------------------------- README 同步


def test_embed_in_readme_replaces_the_marked_block() -> None:
    from experiments.demo_synthetic import README_END, README_START, embed_in_readme

    text = f"intro\n{README_START}\nold\n{README_END}\noutro\n"
    once = embed_in_readme(text, "NEW\n")
    assert once == f"intro\n{README_START}\nNEW\n{README_END}\noutro\n"
    assert embed_in_readme(once, "NEW\n") == once
    for bad in ("no markers", f"{README_END}\n{README_START}\n", f"{README_START}\n{README_START}\n{README_END}\n"):
        with pytest.raises(ValueError):
            embed_in_readme(bad, "x")


def test_readme_demo_block_is_up_to_date(demo: tuple[DemoResult, Path]) -> None:
    """README 首屏的示範區塊必須等於目前程式產生的 demo（避免 README 與程式不同步）；
    更新方式：``uv run python -m experiments.demo_synthetic --out docs/demo --update-readme README.zh-TW.md``。"""
    from experiments.demo_synthetic import embed_in_readme

    result, _ = demo
    readme = (Path(__file__).resolve().parents[1] / "README.zh-TW.md").read_text(encoding="utf-8")
    assert embed_in_readme(readme, render_demo(result)) == readme
    committed = Path(__file__).resolve().parents[1] / "docs" / "demo"
    assert (committed / "demo.md").read_text(encoding="utf-8") == render_demo(result)
