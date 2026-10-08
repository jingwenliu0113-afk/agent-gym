"""第十二輪對抗式審查（decision 021 的實作）確認的缺陷之回歸測試（decision 022）。

- analyze_sweep 的 --out 沿用既有目錄時，留下先前 dry-run 分析的曲線 SVG；SVG 本身沒有 dry-run 標示。

全部只用假的 client，不連網。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import analyze_baseline, analyze_sweep
from experiments.analyze_sweep import analyze_sweep_runs, curve_svg
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.localization import SweepPlan
from tasks.generator import week1_pool

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("runs")
    run_split("eval", OracleFakeClient(POOL), run_dir=root / "eval", clock=ZERO, dry_run=True)
    run_sweeps(select_sources(root / "eval"), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=root / "sweep",
               clock=ZERO, replicates=20, dry_run=True)
    return {"eval": root / "eval", "sweep": root / "sweep"}


def _files(directory: Path) -> dict[str, bytes]:
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def test_analyze_sweep_refuses_a_non_empty_out_directory(runs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    out.mkdir()
    (out / "curves").mkdir()
    (out / "curves" / "old.svg").write_text("<svg/>", encoding="utf-8")
    before = _files(out)
    with pytest.raises(SystemExit):
        analyze_sweep.main([str(runs["sweep"]), "--baseline-run", str(runs["eval"]), "--out", str(out)])
    assert _files(out) == before


@pytest.mark.parametrize("existing", [False, True], ids=["absent", "empty"])
def test_analyze_sweep_writes_into_a_new_or_empty_out_directory(existing: bool, runs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    if existing:
        out.mkdir()
    assert analyze_sweep.main([str(runs["sweep"]), "--baseline-run", str(runs["eval"]), "--out", str(out)]) == 0
    assert json.loads((out / "analysis.json").read_text(encoding="utf-8"))["dry_run"] is True
    svgs = sorted((out / "curves").glob("*.svg"))
    assert len(svgs) == 15
    assert all("[dry-run]" in svg.read_text(encoding="utf-8") for svg in svgs)  # 單獨取用時也能分辨


def test_analyze_baseline_refuses_a_non_empty_out_directory(runs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "analysis"
    out.mkdir()
    (out / "analysis.md").write_text("previous", encoding="utf-8")
    with pytest.raises(SystemExit):
        analyze_baseline.main([str(runs["eval"]), "--out", str(out)])
    assert (out / "analysis.md").read_text(encoding="utf-8") == "previous"
    fresh = tmp_path / "fresh"
    assert analyze_baseline.main([str(runs["eval"]), "--out", str(fresh)]) == 0
    assert json.loads((fresh / "analysis.json").read_text(encoding="utf-8"))["dry_run"] is True


def test_curve_svg_marks_dry_runs_only_when_asked(runs: dict[str, Path]) -> None:
    curve = analyze_sweep_runs([runs["sweep"]], runs["eval"], replicates=20).curves[0]
    assert "[dry-run]" in curve_svg(curve, dry_run=True)
    assert "[dry-run]" not in curve_svg(curve) and curve_svg(curve) == curve_svg(curve, dry_run=False)
