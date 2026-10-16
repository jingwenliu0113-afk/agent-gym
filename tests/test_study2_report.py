"""experiments/study2_report.py（decision 034 §1）的契約測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.ollama import local_agent_config
from experiments import study2_report as rep
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.localization import SweepPlan
from tasks.suites import suite_pool

EVAL = suite_pool("study2", "eval")
PLAN = SweepPlan(coarse_stride=4, n_coarse=1, n_fine=1, drop_threshold=0.3)


def ZERO() -> float:  # noqa: N802
    return 0.0


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("rep")
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=root / "b", clock=ZERO, agent_config=local_agent_config(), dry_run=True,
              suite="study2")
    run_sweeps(select_sources(root / "b"), OllamaOracleFakeClient(EVAL), plan=PLAN, criterion="bonferroni95", run_dir=root / "s",
               clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    return root / "b", root / "s"


def test_report_files_and_labels(runs: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, sweep = runs
    summary = rep.write_report([baseline], [sweep], tmp_path / "study2", replicates=200)
    files = sorted(p.name for p in (tmp_path / "study2").iterdir())
    assert files == ["README.md", "recoverability.svg", "summary.json"]
    page = (tmp_path / "study2" / "README.md").read_text(encoding="utf-8")
    svg = (tmp_path / "study2" / "recoverability.svg").read_text(encoding="utf-8")
    assert page.startswith("> **dry-run")
    assert "[dry-run]" in svg and svg.count('class="dot"') > 36
    for text in (page, svg, json.dumps(summary, ensure_ascii=False)):
        assert "Advanced" not in text and "advanced" not in text
    again = rep.build([baseline], [sweep], replicates=200)
    assert again == json.loads((tmp_path / "study2" / "summary.json").read_text(encoding="utf-8"))  # 決定性


def test_dry_run_never_goes_into_the_readme(runs: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, sweep = runs
    readme = tmp_path / "README.md"
    readme.write_text("# x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dry-run"):
        rep.write_report([baseline], [sweep], tmp_path / "out", readme=readme, replicates=200)
    assert readme.read_text(encoding="utf-8") == "# x\n" and not (tmp_path / "out").exists()


def test_round1_results_directory_is_refused(runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline, sweep = runs
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs" / "results").mkdir(parents=True)
    with pytest.raises(ValueError, match="round-1"):
        rep.write_report([baseline.resolve()], [sweep.resolve()], "docs/results", replicates=200)
    assert list((tmp_path / "docs" / "results").iterdir()) == []


def test_embed_replaces_only_the_study2_block() -> None:
    block = f"{rep.README_START}\nnew\n{rep.README_END}"
    readme = "a\n<!-- results:start -->\nround 1\n<!-- results:end -->\n"
    once = rep.embed(readme, block)
    assert once.startswith(readme.rstrip("\n")) and once.endswith(block + "\n")
    twice = rep.embed(once.replace("new", "old"), block)
    assert twice == once and "round 1" in twice
    with pytest.raises(ValueError):
        rep.embed(once + block, block)


def test_readme_block_for_a_formal_result(runs: tuple[Path, Path]) -> None:
    baseline, sweep = runs
    summary = rep.build([baseline], [sweep], replicates=200)
    summary["provenance"]["dry_run"] = False  # 模擬正式結果，只測段落格式
    block = rep.render_readme_block(summary)
    assert block.startswith(rep.README_START) and block.endswith(rep.README_END)
    assert "不做 Core Gate 的判定" in block and "Advanced" not in block
