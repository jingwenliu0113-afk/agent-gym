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
    assert "[LICENSE](LICENSE)" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_secrets_and_run_outputs_stay_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in (".env", "*.pem", "*.key", "secrets/", "runs/"):
        assert pattern in ignored


def test_tracked_files_do_not_name_the_remote_location() -> None:
    found = subprocess.run(["git", "grep", "-l", "台中", "--", ".", ":!tests/test_publication.py"], cwd=ROOT, capture_output=True,
                           text=True)
    assert found.returncode == 1 and found.stdout == ""  # git grep：1 = 沒有符合
