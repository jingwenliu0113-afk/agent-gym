"""第一輪正式結果的封存（decision 034 §1）。

``docs/results/round1-seal.json`` 記錄：
- repo 中的結果檔（``docs/results/`` 的五個檔案）與 README 結果段落（``<!-- results:start -->`` 到 ``<!-- results:end -->``，含標記）的 sha256；
- 第一輪原始 run 目錄（``runs/``，不進 repo）中每個檔案的 sha256，路徑相對於 ``runs/``。

測試在每台機器檢查 repo 的部分；有 ``runs/`` 的機器另外檢查原始紀錄沒有被改動、刪除或增加。

    uv run python -m experiments.round1_seal --check   # 檢查（預設）
    uv run python -m experiments.round1_seal --write   # 只在建立封存時使用一次
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.results_report import README_END, README_START

SEAL_FORMAT = "agent-gym/round1-seal/v1"
SEAL_PATH = "docs/results/round1-seal.json"
ROUND1_REPO_FILES: tuple[str, ...] = (
    "docs/results/README.md",
    "docs/results/baseline-analysis.md",
    "docs/results/recoverability.svg",
    "docs/results/summary.json",
    "docs/results/sweep-analysis.md",
)
README_BLOCK = "README.md#results"
# 開發題測試（草案設定）、正式 eval baseline、正式 sweep 的原始紀錄。
ROUND1_RUN_DIRS: tuple[str, ...] = ("local-dev-smoke-20261015", "local-eval-20261016", "local-sweep-20261016-1")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _readme_block(root: Path) -> bytes | None:
    path = root / "README.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    start = text.find(README_START)
    end = text.find(README_END, start + len(README_START)) if start >= 0 else -1
    if start < 0 or end < 0:
        return None
    return text[start : end + len(README_END)].encode("utf-8")


def _repo_hashes(root: Path) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for name in ROUND1_REPO_FILES:
        path = root / name
        hashes[name] = _sha256(path.read_bytes()) if path.is_file() else None
    block = _readme_block(root)
    hashes[README_BLOCK] = None if block is None else _sha256(block)
    return hashes


def _run_hashes(root: Path, run_dir: str) -> dict[str, str]:
    base = root / "runs" / run_dir
    return {path.relative_to(base).as_posix(): _sha256(path.read_bytes()) for path in sorted(base.rglob("*")) if path.is_file()}


def build_seal(root: Path) -> dict[str, Any]:
    """目前的檔案 → 封存內容。任一檔案或 README 段落不存在 → ValueError。"""
    repo = _repo_hashes(root)
    missing = sorted(name for name, digest in repo.items() if digest is None)
    if missing:
        raise ValueError(f"cannot seal: missing {', '.join(missing)}")
    run_dirs = {}
    for run_dir in ROUND1_RUN_DIRS:
        if not (root / "runs" / run_dir).is_dir():
            raise ValueError(f"cannot seal: runs/{run_dir} not found")
        run_dirs[run_dir] = _run_hashes(root, run_dir)
    return {"format": SEAL_FORMAT, "repo_files": repo, "run_dirs": run_dirs}


def render(seal: dict[str, Any]) -> str:
    return json.dumps(seal, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def verify_seal(root: Path, seal: dict[str, Any], *, runs: bool) -> dict[str, list[str]]:
    """與封存比對。repo_mismatches：內容不同或不存在的 repo 檔案（含 README 段落）；runs=True 時另外比對 run 目錄：
    run_mismatches（內容不同）、run_missing（封存中有、目前沒有）、run_extra（目前有、封存中沒有）。路徑為 ``<run_dir>/<相對路徑>``。"""
    current = _repo_hashes(root)
    report: dict[str, list[str]] = {
        "repo_mismatches": sorted(name for name, digest in seal["repo_files"].items() if current.get(name) != digest),
        "run_mismatches": [],
        "run_missing": [],
        "run_extra": [],
    }
    if runs:
        for run_dir, expected in sorted(seal["run_dirs"].items()):
            actual = _run_hashes(root, run_dir) if (root / "runs" / run_dir).is_dir() else {}
            report["run_mismatches"] += [f"{run_dir}/{n}" for n in sorted(expected) if n in actual and actual[n] != expected[n]]
            report["run_missing"] += [f"{run_dir}/{n}" for n in sorted(expected) if n not in actual]
            report["run_extra"] += [f"{run_dir}/{n}" for n in sorted(actual) if n not in expected]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="與封存比對（預設）")
    mode.add_argument("--write", action="store_true", help="建立封存檔；已存在時拒絕")
    args = parser.parse_args(argv)
    root = Path.cwd()
    path = root / SEAL_PATH
    if args.write:
        if path.exists():
            print(f"{SEAL_PATH} already exists; the round-1 seal is never rewritten", file=sys.stderr)
            return 2
        path.write_text(render(build_seal(root)), encoding="utf-8")
        print(f"wrote {SEAL_PATH}")
        return 0
    seal = json.loads(path.read_text(encoding="utf-8"))
    # 任何一個封存的 run 目錄存在，就比對全部（缺少其中一個是錯誤）；一個都沒有時只比對 repo 的部分，並明說（結案稽核）
    runs = any((root / "runs" / d).is_dir() for d in ROUND1_RUN_DIRS)
    report = verify_seal(root, seal, runs=runs)
    problems = [f"{kind}: {name}" for kind, names in report.items() for name in names]
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(f"round-1 seal: {len(problems)} problem(s)")
    else:
        print("round-1 seal: OK" + ("" if runs else " (repo files only; the sealed runs/ directories are absent)"))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
