"""英文 README 的數字必須等於已 commit 的結果（decision 037）：英文版是手寫的摘要，這裡把它引用的每個主要數字對照 summary.json。
英文 README 中判定的附條件與兩個 agent 失敗 source 的重疊（decision 038）也在這裡對照 summary.json；README.zh-TW.md 不在此測試內。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
README = (ROOT / "README.md").read_text(encoding="utf-8")
PROSE = " ".join(README.split())  # 附條件是段落文字：比對時不受換行位置影響


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


UNDECIDABLE = ("undecidable", "max_turns_confounded")  # 可判定 = 不是無法判定、也不是 max_turns 混淆（README.zh-TW.md 的分母表）


def _turn_limit_episodes(groups: dict[str, Any], r: int) -> dict[str, int]:
    """以 max_turns 結束的 episode 數：max_turns_rate 是 task 等權的比例，每題都有 r 個 episode 時乘上 episode 數就是次數。"""
    assert all(m["n_episodes"] == m["n_tasks"] * r for m in groups.values())
    return {name: round(m["max_turns_rate"] * m["n_episodes"]) for name, m in groups.items()}


def test_study2_interval_caveat() -> None:
    """decision 038 附條件 4：登記的判定用 Wilson 區間；k = n 時，精確（Clopper–Pearson）95% 區間的下限是 (α / 2)^(1 / n)。"""
    rq1 = _load("study2")["analysis"]["sweep"]["rq1"]
    assert rq1["outcome"] == "mostly_cliff" and rq1["k"] == rq1["n"] and rq1["low"] > 0.5
    assert f"whose lower bound for {rq1['k']}/{rq1['n']} is {rq1['low']:.3f}" in PROSE
    exact_low = 0.025 ** (1 / rq1["n"])
    assert exact_low < 0.5 and _ci(exact_low, 1.0) in PROSE  # 包含 0.5：同一條三分規則下是「無法判斷」
    assert f"0.025^(1/{rq1['n']})" in PROSE
    assert "This is a post-hoc robustness check, not a pre-registered analysis" in PROSE


def test_study3_local_caveats() -> None:
    """decision 038 附條件 1、2：本機的 agent_error、事先登記的敏感度分析（排除 agent_error 之後的 H1 − H3 與剩下的題數），
    以及本機以 max_turns 結束的 episode。"""
    b = _load("study3")["analysis"]["baseline"]
    errors = [b["levels"][lv]["agent_error"] for lv in sorted(b["levels"])]
    episodes = sum(m["n_episodes"] for m in b["levels"].values())
    assert f"{sum(errors)} of its {episodes} evaluation episodes ended in `agent_error`" in PROSE
    assert f"({' / '.join(map(str, errors))} at H1 / H2 / H3)" in PROSE
    without, n = b["rq2_without_agent_error"], b["rq2"]["tasks_per_level"]
    assert f"H1 − H3 falls from {b['rq2']['point']:.3f} to {without['point']:.3f}" in PROSE
    assert f"only {without['n_tasks']['H2']} of the {n} H2 tasks and {without['n_tasks']['H3']} of the {n} H3 tasks" in PROSE
    capped = _turn_limit_episodes(b["levels"], b["rollouts_per_task"])
    assert capped["H1"] == 0 and capped["H2"] > 0 and capped["H3"] > 0
    assert f"The local model also ended {capped['H2']} H2 and {capped['H3']} H3 episodes at the turn limit (none at H1)" in PROSE


def test_study3_haiku_caveats() -> None:
    """decision 038 附條件 2、3：epic-closeout H3 的 turn 上限，以及可判定曲線的題型。"""
    a = _load("study3-haiku")["analysis"]
    levels, families, r = a["baseline"]["levels"], a["baseline"]["families"], a["baseline"]["rollouts_per_task"]
    capped, capped_by_family = _turn_limit_episodes(levels, r), _turn_limit_episodes(families, r)
    tasks = families["epic-closeout"]["n_tasks"] // len(levels)
    # 以 max_turns 結束的 episode 只在 H3、只在 epic-closeout，數目等於 epic-closeout H3 的全部 episode
    assert capped["H3"] == sum(capped.values()) == capped_by_family["epic-closeout"] == sum(capped_by_family.values()) == tasks * r
    assert f"All {tasks * r} Haiku episodes on epic-closeout H3 ({tasks} tasks × {r}) ended at the turn limit" in PROSE
    failed = [t for lv in sorted(levels) for t in a["sources"]["by_level"][lv]["failed"] if "-epic-closeout-" in t]
    assert failed and all("-H3-" in t for t in failed)  # epic-closeout 的失敗都在 H3
    assert len(failed) == a["rq1"]["n_undecidable"] == len(a["rq1"]["max_turns_sources"])
    fails = families["epic-closeout"]["outcomes"]["failure"]
    assert f"The {fails} of them that failed are {fails} of Haiku's {levels['H3']['outcomes']['failure']} H3 failures" in PROSE
    assert f"their {len(failed)} sources are the {a['rq1']['n_undecidable']} undecidable curves" in PROSE
    curves = Counter(c["family"] for c in a["curves"])
    decidable = Counter(c["family"] for c in a["curves"] if c["class"] not in UNDECIDABLE)
    cliffs = Counter(c["family"] for c in a["curves"] if c["class"].startswith("cliff"))
    assert len(families) == 4 and len(decidable) == 2
    assert sum(decidable.values()) == a["rq1"]["n"] and sum(cliffs.values()) == a["rq1"]["k"]
    assert (f"The {a['rq1']['n']} decidable curves come from two of the four task families (sync-repair "
            f"{decidable['sync-repair']}, scoped-triage {decidable['scoped-triage']}; cliffs {cliffs['sync-repair']} and "
            f"{cliffs['scoped-triage']})") in PROSE
    assert decidable["epic-closeout"] == 0 and a["sources"]["n_failed_by_family"]["verified-close"] == 0
    assert f"epic-closeout's {curves['epic-closeout']} curves are undecidable and verified-close had no failed source" in PROSE


def test_study3_failed_sources_overlap() -> None:
    """decision 038 的更正：兩個 agent 失敗的 source 部分重疊（並非全部來自不同的題）；兩者都有曲線的題在本機都無法判定，
    所以 cliff 比例背後的可判定曲線沒有共同的題。"""
    local, haiku = _load("study3")["analysis"], _load("study3-haiku")["analysis"]
    overlap = set(local["sources"]["failed"]) & set(haiku["sources"]["failed"])
    assert 0 < len(overlap) < min(local["sources"]["n_failed"], haiku["sources"]["n_failed"])
    assert f"({len(overlap)} of Haiku's {haiku['sources']['n_failed']} failed sources also failed for the local model;" in PROSE
    # 「本機主要在 H1 / H2」不成立：本機失敗的 source 在三個級別大致平均
    by_level = [local["sources"]["by_level"][h]["n_failed"] for h in ("H1", "H2", "H3")]
    assert sum(by_level) == local["sources"]["n_failed"]
    assert f"(its failed sources are {' / '.join(map(str, by_level))} at H1 / H2 / H3)" in PROSE
    classes = [{c["task_id"]: c["class"] for c in a["curves"]} for a in (local, haiku)]
    assert [len(c) for c in classes] == [len(local["curves"]), len(haiku["curves"])]  # 每題最多一條曲線
    both = classes[0].keys() & classes[1].keys()
    decidable = [{t for t, cls in c.items() if cls not in UNDECIDABLE} for c in classes]
    assert both and all(classes[0][t] in UNDECIDABLE for t in both) and not decidable[0] & decidable[1]
    assert [len(d) for d in decidable] == [local["rq1"]["n"], haiku["rq1"]["n"]]
    assert (f"{len(both)} tasks have curves for both, all of them undecidable for the local model; the decidable curves "
            f"behind {local['rq1']['k']}/{local['rq1']['n']} and {haiku['rq1']['k']}/{haiku['rq1']['n']} share no task)") in PROSE
