"""Study 3 的登記與 run 工具（decision 036 §6、§8 第 1 步 B 部分）：study3 suite、兩道開關、呼叫上限、規則 B、付費規格，
以及依 seed 區塊執行的 baseline（本機分 4 段，每段一個完整的目錄，分析時合併）。

eval instance 只經過程式（dry-run 的 oracle 假 client）；測試不輸出 eval 的內容（decision 036 §4）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments.analyze_baseline import load_runs
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from tasks import generator_v3 as g3
from tasks import suites
from tasks.suites import STUDIES, suite_for_generator, suite_pool

SEGMENTS = ((2, 3), (4, 5), (6, 7), (8, 9))


def ZERO() -> float:  # noqa: N802
    return 0.0


class _RealLooking:
    """不是 oracle 假 client 的 client（走真正推論的檢查路徑）；make_policy 在測試中換掉，所以不會被呼叫。"""


def test_study3_is_registered_with_its_switches_caps_and_rules() -> None:
    spec = STUDIES["study3"]
    assert "study3" in suites.SUITES and suite_for_generator(g3.GENERATOR_VERSION) == "study3"
    assert spec.gated and spec.approved_flag == "STUDY3_APPROVED" and spec.frozen_flag == "STUDY3_FROZEN"
    # §8 第 1–4 步完成（2026-10-23，773cd1a）之後打開；eval 等凍結紀錄（STUDY3_FROZEN 與紀錄一起，見 test_study3_freeze）
    assert suites.STUDY3_APPROVED is True
    assert dict(spec.baseline_caps) == {"dev": 2880, "eval": 2880} and spec.segment_cap == 3000
    assert suites.sweep_cap("study3") == 80_000
    assert spec.source_rule == "first_failure" and spec.paid is suites.STUDY3_PAID
    assert spec.eval_segments == SEGMENTS
    assert [i.task.task_id for i in suite_pool("study3", "eval")] == [i.task.task_id for i in g3.study3_pool("eval")]


def test_study3_model_runs_wait_for_both_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suites, "STUDY3_APPROVED", False)
    monkeypatch.setattr(suites, "STUDY3_FROZEN", False)
    for split in ("dev", "eval"):
        with pytest.raises(ValueError, match="approval of decision 036"):
            suites.check_model_run("study3", split)
    monkeypatch.setattr(suites, "STUDY3_APPROVED", True)
    suites.check_model_run("study3", "dev")
    with pytest.raises(ValueError, match="frozen record of decision 036"):
        suites.check_model_run("study3", "eval")
    monkeypatch.setattr(suites, "STUDY3_FROZEN", True)
    suites.check_model_run("study3", "eval")


def test_segments_are_registered_seed_blocks() -> None:
    assert [len(suites.segment_pool("study3", seeds)) for seeds in SEGMENTS] == [24, 24, 24, 24]
    ids = [i.task.task_id for seeds in SEGMENTS for i in suites.segment_pool("study3", seeds)]
    assert sorted(ids) == sorted(i.task.task_id for i in suite_pool("study3", "eval"))
    for bad in ((2, 4), (2,), (0, 1), (8, 9, 10)):
        with pytest.raises(ValueError, match="segment"):
            suites.segment_pool("study3", bad)
    with pytest.raises(ValueError, match="segment"):
        suites.segment_pool("study2", (2, 3))


def test_a_local_study3_eval_baseline_must_run_one_registered_segment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.run_baseline as rb

    monkeypatch.setattr(suites, "STUDY3_APPROVED", True)
    monkeypatch.setattr(suites, "STUDY3_FROZEN", True)
    monkeypatch.setattr(rb, "make_policy", lambda client, config: None)
    common = dict(agent_config=local_agent_config(), suite="study3", call_log=True, max_model_calls=2880, rollouts=3)
    with pytest.raises(ValueError, match="segment"):  # 本機的 eval baseline 必須分段（decision 036 §6）
        run_split("eval", _RealLooking(), run_dir=tmp_path / "a", **common)
    with pytest.raises(ValueError, match="segment"):
        run_split("eval", _RealLooking(), run_dir=tmp_path / "b", seeds=(2, 4), **common)
    with pytest.raises(ValueError, match="2880"):
        run_split("eval", _RealLooking(), run_dir=tmp_path / "c", seeds=(2, 3), **{**common, "max_model_calls": 4320})
    with pytest.raises(ValueError, match="segment"):  # dev 不分段
        run_split("dev", _RealLooking(), run_dir=tmp_path / "d", seeds=(0, 1), **common)
    assert not any((tmp_path / d).exists() for d in "abcd")


@pytest.fixture(scope="module")
def segment_runs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("s3")
    dirs = []
    for seeds in SEGMENTS:
        run_dir = root / f"seg-{seeds[0]}-{seeds[1]}"
        run_split("eval", OllamaOracleFakeClient(suite_pool("study3", "eval")), run_dir=run_dir, clock=ZERO, suite="study3",
                  agent_config=local_agent_config(), dry_run=True, rollouts=1, seeds=seeds)
        dirs.append(run_dir)
    return dirs


def test_a_segment_directory_records_its_seeds_and_runs_only_them(segment_runs: list[Path]) -> None:
    run = json.loads((segment_runs[0] / "run.json").read_text(encoding="utf-8"))
    assert run["seeds"] == [2, 3] and run["dry_run"] is True
    summary = json.loads((segment_runs[0] / "summary.json").read_text(encoding="utf-8"))
    expected = [i.task.task_id for i in suites.segment_pool("study3", (2, 3))]
    assert summary["n_tasks"] == 24 and [r["task_id"] for r in summary["results"]] == expected
    assert all(r["outcome"] == "success" for r in summary["results"])


def test_segment_directories_load_alone_and_together(segment_runs: list[Path]) -> None:
    one = load_runs(segment_runs[:1])
    assert one.suite == "study3" and len(one.records) == 24
    every = load_runs(segment_runs)
    assert len(every.records) == 96
    assert {r.task_id for r in every.records} == {i.task.task_id for i in suite_pool("study3", "eval")}


def test_a_tampered_segment_is_refused(segment_runs: list[Path], tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(segment_runs[0], copy)
    run = json.loads((copy / "run.json").read_text(encoding="utf-8"))
    (copy / "run.json").write_text(json.dumps({**run, "seeds": [4, 5]}), encoding="utf-8")  # 宣稱的區塊與內容不符
    with pytest.raises(ValueError):
        load_runs([copy])
    (copy / "run.json").write_text(json.dumps({**run, "seeds": [2, 4]}), encoding="utf-8")  # 不是登記的區塊
    with pytest.raises(ValueError, match="segment"):
        load_runs([copy])
