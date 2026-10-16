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
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=root / "b", clock=ZERO, agent_config=local_agent_config(), dry_run=True, suite="study2")
    run_sweeps(select_sources(root / "b"), OllamaOracleFakeClient(EVAL), plan=PLAN, criterion="bonferroni95", run_dir=root / "s",
               clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    return root / "b", root / "s"


@pytest.fixture(scope="module")
def formal_baseline_dry_sweep(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("mixed")
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=root / "b", clock=ZERO, agent_config=local_agent_config(), suite="study2")
    run_sweeps(select_sources(root / "b"), OllamaOracleFakeClient(EVAL), plan=PLAN, criterion="bonferroni95", run_dir=root / "s",
               clock=ZERO, agent_config=local_agent_config(), dry_run=True, max_model_calls=40)
    return root / "b", root / "s"


def test_report_files_and_labels(runs: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, sweep = runs
    summary = rep.write_report([baseline], [sweep], tmp_path / "study2", replicates=200)
    files = sorted(p.name for p in (tmp_path / "study2").iterdir())
    assert files == ["README.md", "recoverability.svg", "summary.json"]
    page = (tmp_path / "study2" / "README.md").read_text(encoding="utf-8")
    svg = (tmp_path / "study2" / "recoverability.svg").read_text(encoding="utf-8")
    assert "> **dry-run" in page and "[dry-run]" in svg and svg.count('class="dot"') > 36
    for text in (page, svg, json.dumps(summary, ensure_ascii=False)):
        assert "Advanced" not in text and "advanced" not in text
    again = rep.build([baseline], [sweep], replicates=200)
    assert again == json.loads((tmp_path / "study2" / "summary.json").read_text(encoding="utf-8"))  # 決定性


def test_a_dry_run_sweep_marks_the_whole_report(formal_baseline_dry_sweep: tuple[Path, Path], tmp_path: Path) -> None:
    """正式 baseline + dry-run sweep：整份報告都是 dry-run，不能寫進 README（驗證 B 第二輪）。"""
    baseline, sweep = formal_baseline_dry_sweep
    readme = tmp_path / "README.md"
    readme.write_text("# x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dry-run"):
        rep.write_report([baseline], [sweep], tmp_path / "out", readme=readme, replicates=200)
    assert readme.read_text(encoding="utf-8") == "# x\n" and not (tmp_path / "out").exists()
    summary = rep.write_report([baseline], [sweep], tmp_path / "out", replicates=200)
    assert summary["provenance"]["dry_run"] is True and summary["provenance"]["data_complete"] is False
    svg = (tmp_path / "out" / "recoverability.svg").read_text(encoding="utf-8")
    assert "[dry-run]" in svg and "沒有曲線" in svg and "s4" in svg  # 未完成的 seed 區塊照實畫出


def test_round1_results_are_refused_case_insensitively_and_from_any_cwd(runs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path) -> None:
    baseline, sweep = runs
    sealed = rep.ROUND1_RESULTS
    before = {p.name: p.read_bytes() for p in sealed.iterdir() if p.is_file()}
    monkeypatch.chdir(tmp_path)
    for out in (sealed, Path(str(sealed).replace("results", "Results")), sealed / "." ):
        with pytest.raises(ValueError, match="round-1"):
            rep.write_report([baseline.resolve()], [sweep.resolve()], out, replicates=200)
    with pytest.raises(ValueError, match="never modified"):
        rep.write_report([baseline.resolve()], [sweep.resolve()], tmp_path / "o", readme=sealed / "README.md", replicates=200)
    assert {p.name: p.read_bytes() for p in sealed.iterdir() if p.is_file()} == before


def test_readme_problems_are_found_before_writing(formal_baseline_dry_sweep: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    baseline, _ = formal_baseline_dry_sweep
    readme = tmp_path / "README.md"
    block = f"{rep.README_START}\nx\n{rep.README_END}"
    readme.write_text(block + "\n" + block + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        rep.write_report([baseline], [], tmp_path / "out", readme=readme)
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="does not exist"):
        rep.write_report([baseline], [], tmp_path / "out", readme=tmp_path / "missing.md")
    assert not (tmp_path / "out").exists()


def test_formal_reports_use_the_registered_replicates(formal_baseline_dry_sweep: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, _ = formal_baseline_dry_sweep
    with pytest.raises(ValueError, match="B = 10000"):
        rep.write_report([baseline], [], tmp_path / "out", replicates=200)
    summary = rep.write_report([baseline], [], tmp_path / "out")
    assert summary["provenance"]["dry_run"] is False and summary["analysis"]["sweep_skipped"] == "ceiling"


def test_embed_replaces_only_the_study2_block() -> None:
    block = f"{rep.README_START}\nnew\n{rep.README_END}"
    readme = "a\n<!-- results:start -->\nround 1\n<!-- results:end -->\n"
    once = rep.embed(readme, block)
    assert once.startswith(readme.rstrip("\n")) and once.endswith(block + "\n")
    twice = rep.embed(once.replace("new", "old"), block)
    assert twice == once and "round 1" in twice
    with pytest.raises(ValueError):
        rep.embed(once + block, block)


def test_readme_block_for_a_formal_result(formal_baseline_dry_sweep: tuple[Path, Path]) -> None:
    baseline, _ = formal_baseline_dry_sweep
    summary = rep.build([baseline], [])
    block = rep.render_readme_block(summary)
    assert block.startswith(rep.README_START) and block.endswith(rep.README_END)
    assert "不做 Core Gate 的判定" in block and "完全天花板" in block and "Advanced" not in block
