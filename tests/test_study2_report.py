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


def test_readme_block_states_the_rq1_sample_size(formal_baseline_dry_sweep: tuple[Path, Path]) -> None:
    """RQ1 的結論必須附上分母與 CI：可判定的失敗曲線很少時，讀者要看得到（不只寫「以驟降為主」）。"""
    baseline, _ = formal_baseline_dry_sweep
    summary = json.loads(json.dumps(rep.build([baseline], [])))
    rq1 = {"outcome": "mostly_cliff", "k": 4, "n": 4, "low": 0.5101, "high": 1.0, "n_failed_sources": 5, "n_failed_curves": 5,
           "n_undecidable": 1, "n_max_turns_confounded": 0, "mechanical": 0, "behavioral": 4}
    summary["analysis"]["sweep"] = {"rq1": rq1}
    summary["analysis"]["sweep_skipped"] = None
    block = rep.render_readme_block(summary)
    assert "- RQ1：以驟降為主（cliff 比例的 95% CI 下限 > 0.5）：可判定的失敗曲線中 cliff 4 / 4，95% CI [0.510, 1.000]" in block
    assert "失敗的 source 5 條，無法判定 1 條、max_turns 混淆 0 條" in block
    summary["analysis"]["sweep"]["rq1"] = {**rq1, "outcome": "no_decidable", "k": 0, "n": 0}
    assert "可判定的失敗曲線中 cliff 0 / 0（失敗的 source 5 條" in rep.render_readme_block(summary)  # n = 0 時不附 CI


ROOT = Path(__file__).resolve().parent.parent
STUDY2_RESULTS = ROOT / "docs" / "results" / "study2"
STUDY2_EVAL = ROOT / "runs" / "study2-eval-20261017"
STUDY2_SWEEPS = [ROOT / "runs" / f"study2-sweep-20261017-{n}" for n in range(1, 7)]


def test_the_committed_study2_results_match_the_summary() -> None:
    """正式的 Study 2 報告、曲線圖與 README 段落，必須等於從已 commit 的 summary.json 重新呈現的結果（結案稽核）。Study 2 沒有 sha256 封存：這裡只檢查呈現與 summary.json 一致，
    summary.json 本身只在有原始 run 目錄時逐 byte 重新產生（下一個測試）。"""
    summary = json.loads((STUDY2_RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["dry_run"] is False and summary["provenance"]["replicates"] == 10_000
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert rep.embed(readme, rep.render_readme_block(summary)) == readme
    assert (STUDY2_RESULTS / "README.md").read_text(encoding="utf-8") == rep.render_page(summary)
    assert (STUDY2_RESULTS / "recoverability.svg").read_text(encoding="utf-8") == rep.recoverability_svg(summary)


@pytest.mark.skipif(not all(d.is_dir() for d in [STUDY2_EVAL, *STUDY2_SWEEPS]), reason="runs/ 不在 repo；只在有原始 run 目錄的機器上檢查")
def test_the_study2_summary_regenerates_byte_identical_from_the_raw_runs() -> None:
    summary = rep.build([STUDY2_EVAL], STUDY2_SWEEPS)
    text = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    assert text == (STUDY2_RESULTS / "summary.json").read_text(encoding="utf-8")
