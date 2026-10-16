"""第一輪正式結果的封存（decision 034 §1）：docs/results/round1-seal.json 釘住 repo 中的結果檔與 runs/ 中原始 run 目錄的 sha256。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import round1_seal
from experiments.round1_seal import (
    README_BLOCK,
    ROUND1_REPO_FILES,
    ROUND1_RUN_DIRS,
    SEAL_FORMAT,
    SEAL_PATH,
    build_seal,
    verify_seal,
)

ROOT = Path(__file__).resolve().parent.parent


def _seal() -> dict:
    return json.loads((ROOT / SEAL_PATH).read_text(encoding="utf-8"))


def test_seal_file_is_canonical_and_complete() -> None:
    seal = _seal()
    assert seal["format"] == SEAL_FORMAT
    assert sorted(seal["repo_files"]) == sorted([*ROUND1_REPO_FILES, README_BLOCK])
    assert sorted(seal["run_dirs"]) == sorted(ROUND1_RUN_DIRS)
    for run_dir in ROUND1_RUN_DIRS:
        files = seal["run_dirs"][run_dir]
        assert files, run_dir
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in files)
    assert (ROOT / SEAL_PATH).read_text(encoding="utf-8") == round1_seal.render(seal)


def test_repo_files_match_the_seal() -> None:
    """結果檔與 README 的結果段落永遠不得改變（第一輪封存）。"""
    report = verify_seal(ROOT, _seal(), runs=False)
    assert report["repo_mismatches"] == []


@pytest.mark.skipif(not all((ROOT / "runs" / d).is_dir() for d in ROUND1_RUN_DIRS), reason="runs/ 不在 repo；只在有原始 run 目錄的機器上檢查")
def test_run_dirs_match_the_seal() -> None:
    report = verify_seal(ROOT, _seal(), runs=True)
    assert report["run_mismatches"] == []
    assert report["run_missing"] == []
    assert report["run_extra"] == []


def test_verify_detects_changed_missing_and_extra_files(tmp_path: Path) -> None:
    (tmp_path / "docs" / "results").mkdir(parents=True)
    for name in ROUND1_REPO_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "README.md").write_text("x\n<!-- results:start -->\nblock\n<!-- results:end -->\ny\n", encoding="utf-8")
    for run_dir in ROUND1_RUN_DIRS:
        (tmp_path / "runs" / run_dir / "sub").mkdir(parents=True)
        (tmp_path / "runs" / run_dir / "a.json").write_text("a", encoding="utf-8")
        (tmp_path / "runs" / run_dir / "sub" / "b.json").write_text("b", encoding="utf-8")
    seal = build_seal(tmp_path)
    assert verify_seal(tmp_path, seal, runs=True) == {"repo_mismatches": [], "run_mismatches": [], "run_missing": [], "run_extra": []}

    # README 結果段落以外的改動不影響封存
    (tmp_path / "README.md").write_text("changed\n<!-- results:start -->\nblock\n<!-- results:end -->\n", encoding="utf-8")
    assert verify_seal(tmp_path, seal, runs=False)["repo_mismatches"] == []

    (tmp_path / "README.md").write_text("<!-- results:start -->\nother\n<!-- results:end -->\n", encoding="utf-8")
    (tmp_path / ROUND1_REPO_FILES[0]).write_text("changed", encoding="utf-8")
    first = ROUND1_RUN_DIRS[0]
    (tmp_path / "runs" / first / "a.json").write_text("changed", encoding="utf-8")
    (tmp_path / "runs" / first / "sub" / "b.json").unlink()
    (tmp_path / "runs" / first / "new.json").write_text("n", encoding="utf-8")
    report = verify_seal(tmp_path, seal, runs=True)
    assert report["repo_mismatches"] == sorted([ROUND1_REPO_FILES[0], README_BLOCK])
    assert report["run_mismatches"] == [f"{first}/a.json"]
    assert report["run_missing"] == [f"{first}/sub/b.json"]
    assert report["run_extra"] == [f"{first}/new.json"]


def test_missing_readme_block_is_a_mismatch(tmp_path: Path) -> None:
    (tmp_path / "docs" / "results").mkdir(parents=True)
    for name in ROUND1_REPO_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "README.md").write_text("<!-- results:start -->\nblock\n<!-- results:end -->\n", encoding="utf-8")
    for run_dir in ROUND1_RUN_DIRS:
        (tmp_path / "runs" / run_dir).mkdir(parents=True)
        (tmp_path / "runs" / run_dir / "a.json").write_text("a", encoding="utf-8")
    seal = build_seal(tmp_path)
    (tmp_path / "README.md").write_text("no block\n", encoding="utf-8")
    (tmp_path / ROUND1_REPO_FILES[1]).unlink()
    assert verify_seal(tmp_path, seal, runs=False)["repo_mismatches"] == sorted([ROUND1_REPO_FILES[1], README_BLOCK])


@pytest.mark.skipif(not all((ROOT / "runs" / d).is_dir() for d in ROUND1_RUN_DIRS), reason="runs/ 不在 repo；只在有原始 run 目錄的機器上檢查")
def test_round1_report_regenerates_byte_identical(tmp_path: Path) -> None:
    """目前的程式（含 task format v2 與 grader 的 unless 支援）重新產生第一輪報告，必須與封存的檔案逐 byte 相同。"""
    from experiments.results_report import render_readme_block, write_report

    summary = write_report([ROOT / "runs" / "local-eval-20261016"], [ROOT / "runs" / "local-sweep-20261016-1"], tmp_path)
    for name in ROUND1_REPO_FILES:
        assert (tmp_path / Path(name).name).read_bytes() == (ROOT / name).read_bytes(), name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert render_readme_block(summary) in readme
