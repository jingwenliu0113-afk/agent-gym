"""generator v3 的核心（decision 036 §3 / §4 / §5 A）：決定性與確定性的重抽、turn 的檢查、超過 turn 的正確解、search 不截斷、
forbid 集合的重算、decisive_rules、級別與跨研究的檢查。用一個只在測試中登記的玩具題型（toy），不依賴四個正式題型。"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from agents.baseline import BASELINE_MAX_TURNS
from tasks import generator_v3 as g3
from tasks.generator import GeneratorError

TOY_RULES = {"H1": ("T1", "T2"), "H2": ("T1", "T2", "T3"), "H3": ("T1", "T2", "T3", "T4")}
KNOBS: dict[str, Any] = {}  # 各測試改變 toy builder 的行為


def _toy_build(ws, ps, level: str, seed: int):
    world = g3.World.generate(ws, (8, 10))
    if KNOBS.get("redraw_below") is not None and ps.random() < KNOBS["redraw_below"]:
        raise g3.Redraw()
    ids = world.ids()
    target = ids[0]
    world.issues[target]["status"] = "resolved"
    others = [i for i in ids[1:] if world.issues[i]["status"] != "closed"]
    if not others:
        raise g3.Redraw()
    other = others[0]
    instruction = f"Close issue #{target}. Do not change anything else, and do not add any comments."
    oracle = [("get_issue", {"issue_id": target}), ("update_issue", {"issue_id": target, "status": "closed"})]
    pad = KNOBS.get("oracle_pad", 0)
    oracle = [("get_issue", {"issue_id": target})] * pad + oracle
    forbidden = [g3.forbid([i for i in ids if i != target], ["status"], "scope", ["closed"])]
    spec = g3.Spec(
        instruction=instruction,
        expected=[g3.eq(target, "status", "closed", "T1")],
        constraints=[g3.unchanged(target, g3.OTHER_FIELDS), *g3.others_unchanged(world, {target})],
        forbidden=forbidden,
        scope=[{"issue_id": target, "fields": ["status"]}],
        oracle=oracle,
        negatives=[("T1:leave-open", [("get_issue", {"issue_id": target})]),
                   ("scope:close-other", [*oracle, ("update_issue", {"issue_id": other, "status": "closed"})])],
        alternatives=[("search-first", [("search_issues", {"limit": 50}), *oracle[-1:]]),
                      ("direct", oracle[-1:]),
                      *KNOBS.get("extra_alternatives", [])],
        long_alternatives=list(KNOBS.get("long_alternatives", [])),
        params={"target": target, "n_targets": {"H1": 1, "H2": 2, "H3": 3}[level], "n_decoys": 0,
                "binding_rules": ["T1"], "planted_text_issues": [],
                "decisive_rules": KNOBS.get("decisive_rules", {str(target): ["T1"]})},
    )
    return world, g3.finish(spec)


def _toy_solve(task, level, issues, complete):
    target = task.params["target"]
    return {"target": target, "status": issues[target]["status"]}


def _toy_forbid_scope(task, level):
    ids = {i["id"] for i in task.initial_state["issues"]} - {task.params["target"]}
    return KNOBS.get("forbid_scope", {("scope", ("status",)): ids})


TOY = g3.Family(
    name="toy", rules=TOY_RULES, oracle_range={"H1": (1, 13), "H2": (1, 13), "H3": (1, 13)}, build=_toy_build,
    decision=lambda task: {"target": task.params["target"], "status": "resolved"}, solve=_toy_solve,
    forbid_scope=_toy_forbid_scope, check_refs=lambda task, level: None,
    rule_sources={"T1": "new", "T2": "new", "T3": "study2-failure", "T4": "new"},
)


@pytest.fixture(autouse=True)
def toy(monkeypatch: pytest.MonkeyPatch):
    KNOBS.clear()
    monkeypatch.setattr(g3, "FAMILIES", (*g3.FAMILIES, "toy"))
    monkeypatch.setitem(g3._LOADED, "toy", TOY)
    yield
    KNOBS.clear()


def test_the_constants_match_the_frozen_agent_and_decision_036() -> None:
    assert g3.MAX_TURNS == BASELINE_MAX_TURNS == 20 and g3.TOOL_CALL_BUDGET == 19
    assert g3.MAX_ORACLE == 13 and g3.ORACLE_SLACK == 6
    assert g3.LEVELS == ("H1", "H2", "H3") and g3.DEV_SEEDS == (0, 1) and g3.EVAL_SEEDS == tuple(range(2, 10))
    assert sum(s % 2 for s in g3.EVAL_SEEDS) == len(g3.EVAL_SEEDS) // 2  # 奇偶各半
    assert g3.rng_key("toy", "H2", 3) == "study3/toy/H2/3" and g3.task_id_for("toy", "H2", 3) == "gen3-toy-H2-s3"


def test_generation_is_deterministic() -> None:
    a, b = g3.generate("toy", "H1", 5), g3.generate("toy", "H1", 5)
    assert a == b and a.task.generator_version == "issue-tracker-gen/v3" and a.task.family == "gen3/toy"
    assert g3.rng_fingerprint("toy", "H1", 5) == g3.rng_fingerprint("toy", "H1", 5)
    assert g3.generate("toy", "H1", 6).task.initial_state != a.task.initial_state


def test_redraws_are_deterministic_and_recorded() -> None:
    KNOBS["redraw_below"] = 0.6
    redrawn = [s for s in range(40) if g3.generate("toy", "H1", s).task.params["redraws"] > 0]
    assert redrawn  # 有些 seed 第一次的條件不成立
    seed = redrawn[0]
    first = g3.generate("toy", "H1", seed)
    assert g3.generate("toy", "H1", seed) == first
    fingerprint = g3.rng_fingerprint("toy", "H1", seed)
    KNOBS["redraw_below"] = None
    assert g3.generate("toy", "H1", seed).task.params["redraws"] == 0
    assert g3.rng_fingerprint("toy", "H1", seed) != fingerprint
    KNOBS["redraw_below"] = 1.1  # 永遠不成立
    with pytest.raises(GeneratorError, match="no draw"):
        g3.generate("toy", "H1", 1)


def test_the_oracle_must_leave_turn_slack() -> None:
    KNOBS["oracle_pad"] = 11  # 13 個 call：在上限
    assert len(g3.generate("toy", "H1", 1).oracle_calls) == 13
    KNOBS["oracle_pad"] = 12  # 14 個 call
    with pytest.raises(GeneratorError, match="oracle"):
        g3.generate("toy", "H1", 1)


def test_alternatives_fit_the_turn_budget_and_long_ones_are_listed_separately() -> None:
    target = g3.generate("toy", "H1", 1).task.params["target"]
    read = ("get_issue", {"issue_id": target})
    close = ("update_issue", {"issue_id": target, "status": "closed"})
    KNOBS["long_alternatives"] = [("read-each", [read] * 19 + [close])]  # 20 個 call：正確，但超過 19
    instance = g3.generate("toy", "H1", 1)
    assert [name for name, _ in instance.long_alternative_calls] == ["read-each"]
    KNOBS["long_alternatives"] = [("too-short", [read, close])]  # 沒有超過 19：應該列在 alternatives
    with pytest.raises(GeneratorError, match="long alternative"):
        g3.generate("toy", "H1", 1)
    KNOBS["long_alternatives"] = []
    KNOBS["extra_alternatives"] = [("too-long", [read] * 19 + [close])]
    with pytest.raises(GeneratorError, match="alternative 'too-long'"):
        g3.generate("toy", "H1", 1)
    KNOBS["extra_alternatives"] = [("wrong", [read])]
    with pytest.raises(GeneratorError, match="alternative 'wrong'"):
        g3.generate("toy", "H1", 1)


def test_a_truncated_search_in_a_correct_solution_is_refused() -> None:
    target = g3.generate("toy", "H1", 1).task.params["target"]
    close = ("update_issue", {"issue_id": target, "status": "closed"})
    KNOBS["extra_alternatives"] = [("first-page", [("search_issues", {"limit": 2}), close])]
    with pytest.raises(GeneratorError, match="search returned 2 of"):
        g3.generate("toy", "H1", 1)


def test_forbid_sets_are_recomputed_per_rule_and_field() -> None:
    KNOBS["forbid_scope"] = {("scope", ("status",)): {1}}
    with pytest.raises(GeneratorError, match="forbid"):
        g3.generate("toy", "H1", 1)
    KNOBS.pop("forbid_scope")
    ids = {i["id"] for i in g3.generate("toy", "H1", 1).task.initial_state["issues"]}
    KNOBS["forbid_scope"] = {("scope", "*"): ids}  # 欄位不同也算不同
    with pytest.raises(GeneratorError, match="forbid"):
        g3.generate("toy", "H1", 1)


def test_decisive_rules_must_use_the_level_s_rules() -> None:
    for bad in ({}, {"9999": ["T1"]}, {"1": ["T4"]}, {"1": []}):
        KNOBS["decisive_rules"] = bad
        with pytest.raises(GeneratorError, match="decisive_rules"):
            g3.generate("toy", "H1", 1)


def test_families_must_tag_every_rule_and_stay_within_the_oracle_cap() -> None:
    with pytest.raises(ValueError, match="source"):
        dataclasses.replace(TOY, rule_sources={"T1": "new"})
    with pytest.raises(ValueError, match="source"):
        dataclasses.replace(TOY, rule_sources={**TOY.rule_sources, "T4": "guess"})
    with pytest.raises(ValueError, match="oracle range"):
        dataclasses.replace(TOY, oracle_range={"H1": (1, 14), "H2": (1, 13), "H3": (1, 13)})
    with pytest.raises(ValueError):
        g3.family("toy2")


def test_check_levels_requires_strictly_harder_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(g3, "check_harder_than_study2", lambda family_name, h1: None)
    KNOBS["oracle_pad"] = 0
    pool = [g3.generate("toy", lv, 1) for lv in g3.LEVELS]
    with pytest.raises(GeneratorError, match="oracle length must increase"):
        g3.check_levels(pool)
    padded = [dataclasses.replace(inst, oracle_calls=inst.oracle_calls * (k + 1)) for k, inst in enumerate(pool)]
    g3.check_levels(padded)
    h2 = padded[1].task
    same_rules = dataclasses.replace(h2, params={**h2.params, "rules": padded[0].task.params["rules"]})
    with pytest.raises(GeneratorError, match="rules must increase"):
        g3.check_levels([padded[0], dataclasses.replace(padded[1], task=same_rules), padded[2]])
    fewer = dataclasses.replace(h2, params={**h2.params, "n_targets": 0})
    with pytest.raises(GeneratorError, match="n_targets"):
        g3.check_levels([padded[0], dataclasses.replace(padded[1], task=fewer), padded[2]])


def test_h1_must_be_harder_than_study2_l3() -> None:
    def h1(n_rules: int, n_calls: int) -> Any:
        inst = g3.generate("toy", "H1", 1)
        params = {**inst.task.params, "rules": [f"R{i}" for i in range(n_rules)]}
        task = dataclasses.replace(inst.task, params=params)
        return dataclasses.replace(inst, task=task, oracle_calls=inst.oracle_calls[:1] * n_calls)

    g3.check_harder_than_study2("epic-closeout", h1(5, 9))  # v2 L3：4 條規則、oracle 8
    for n_rules, n_calls in ((4, 9), (5, 8)):
        with pytest.raises(GeneratorError):
            g3.check_harder_than_study2("epic-closeout", h1(n_rules, n_calls))
    g3.check_harder_than_study2("sync-repair", h1(5, 10))  # v2 L3：4 條規則、oracle 10；sync-repair 放寬為不短於
    with pytest.raises(GeneratorError):
        g3.check_harder_than_study2("verified-close", h1(5, 9))  # v2 L3 oracle 9：其他題型必須嚴格大於
