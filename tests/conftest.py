"""共用測試 fixture。

``small_world`` 是獨立於 smoke 種子世界的小型 canonical state dict，
供 tools / env / hashing 測試使用，避免這些測試依賴 smoke task 資料。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

SMALL_WORLD: dict[str, Any] = {
    "users": [
        {"id": "bot", "name": "Test Bot", "team": "automation", "active": True},
        {"id": "u1", "name": "Ann Active", "team": "backend", "active": True},
        {"id": "u2", "name": "Ian Inactive", "team": "backend", "active": False},
        {"id": "u3", "name": "Fay Frontend", "team": "frontend", "active": True},
    ],
    "labels": [
        {"name": "bug", "description": "Something is broken"},
        {"name": "security", "description": "Security relevant"},
        {"name": "ui", "description": "User interface"},
    ],
    "issues": [
        {
            "id": 1,
            "title": "Crash on login",
            "body": "App crashes when pressing the Login button.",
            "status": "open",
            "priority": "p1",
            "assignee_id": None,
            "reporter_id": "u1",
            "labels": ["bug", "ui"],
            "created_at": "2026-09-01",
        },
        {
            "id": 2,
            "title": "登入頁面字型錯誤",
            "body": "Font fallback is wrong on the LOGIN page.",
            "status": "in_progress",
            "priority": "p2",
            "assignee_id": "u1",
            "reporter_id": "u3",
            "labels": ["ui"],
            "created_at": "2026-09-02",
        },
        {
            "id": 3,
            "title": "Token leak in logs",
            "body": "Access tokens are written to debug logs.",
            "status": "open",
            "priority": "p0",
            "assignee_id": "u2",
            "reporter_id": "u1",
            "labels": ["bug", "security"],
            "created_at": "2026-09-03",
        },
        {
            "id": 4,
            "title": "Old cleanup task",
            "body": "Remove unused feature flags.",
            "status": "closed",
            "priority": "p3",
            "assignee_id": "u2",
            "reporter_id": "u2",
            "labels": [],
            "created_at": "2026-08-01",
        },
    ],
    "comments": [
        {"id": 1, "issue_id": 1, "author_id": "u1", "body": "Reproduced on build 42."},
        {"id": 2, "issue_id": 3, "author_id": "u1", "body": "Please rotate the tokens."},
    ],
}

SMALL_ACTOR = "bot"


@pytest.fixture
def small_world() -> dict[str, Any]:
    """每個測試拿到獨立的深拷貝。"""
    return copy.deepcopy(SMALL_WORLD)
