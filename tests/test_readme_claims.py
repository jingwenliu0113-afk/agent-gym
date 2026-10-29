"""英文 README 的數字必須等於已 commit 的結果（decision 037）：英文版是手寫的摘要，這裡把它引用的每個主要數字對照 summary.json。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name / "summary.json").read_text(encoding="utf-8")) if name else \
        json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def _ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def _rates(levels: dict[str, Any]) -> str:
    return " / ".join(f"{levels[lv]['success_rate']:.3f}" for lv in sorted(levels))


def test_round1_claims() -> None:
    s = _load("")
    b, sw = s["baseline"], s["sweep"]
    assert f"{b['outcome_counts']['success']}/{b['rollouts']}" in README
    assert f"{sw['reruns']} reruns" in README and f"`{sw['gate']['outcome']}`" in README
    assert f"{s['calls']['baseline']['calls'] + s['calls']['sweep']['calls']:,}" in README


def test_study2_claims() -> None:
    a = _load("study2")["analysis"]
    rq1 = a["sweep"]["rq1"]
    assert _rates(a["baseline"]["levels"]) in README
    assert f"{rq1['k']}/{rq1['n']}" in README and _ci(rq1["low"], rq1["high"]) in README
    haiku = _load("study2-haiku")
    levels = haiku["analysis"]["baseline"]["levels"]
    episodes = sum(m["n_episodes"] for m in levels.values())
    assert f"{sum(m['outcomes']['success'] for m in levels.values())}/{episodes}" in README
    assert f"${haiku['provenance']['spent_usd']['baseline']:.3f}" in README


@pytest.mark.parametrize("name", ["study3", "study3-haiku"])
def test_study3_claims(name: str) -> None:
    a = _load(name)["analysis"]
    b, rq1 = a["baseline"], a["rq1"]
    assert _rates(b["levels"]) in README
    assert f"{b['rq2']['check']['bootstrap']['point']:.3f}" in README
    assert f"{rq1['k']}/{rq1['n']}" in README and _ci(rq1["low"], rq1["high"]) in README
    if name == "study3-haiku":
        assert f"${a['sweep']['budget']['used']:.2f}" in README
