"""公開 repo 的檢查（CLAUDE.md「公開與授權」，2026-10-29）：授權一致、README 標示授權、不進 repo 的目錄仍被忽略、沒有寫入地點。"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_license_is_bsd_3_clause_everywhere() -> None:
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("BSD 3-Clause License\n\nCopyright (c) 2026, jingwen\n")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "BSD-3-Clause" and project["license-files"] == ["LICENSE"]
    for readme in ("README.md", "README.zh-TW.md"):
        assert "[LICENSE](LICENSE)" in (ROOT / readme).read_text(encoding="utf-8")


def test_the_two_readmes_link_to_each_other() -> None:
    """decision 037：README.md 是英文版、README.zh-TW.md 是繁體中文版（程式產生的段落都在中文版），開頭互相連結。"""
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
    assert "[繁體中文](README.zh-TW.md)" in english.split("\n## ")[0]
    assert "[English](README.md)" in chinese.split("\n## ")[0]
    assert "<!-- results:start -->" in chinese and "<!-- " not in english  # 英文版不含程式產生的段落


def test_secrets_and_run_outputs_stay_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in (".env", "*.pem", "*.key", "secrets/", "runs/"):
        assert pattern in ignored


def test_tracked_files_do_not_name_the_remote_location() -> None:
    found = subprocess.run(["git", "grep", "-l", "台中", "--", ".", ":!tests/test_publication.py"], cwd=ROOT, capture_output=True,
                           text=True)
    assert found.returncode == 1 and found.stdout == ""  # git grep：1 = 沒有符合
