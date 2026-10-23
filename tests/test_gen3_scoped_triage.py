"""scoped-triage（generator v3，decision 036 §3、§4、§5 A、附錄 B scoped-triage 1–15）的契約測試。

資料隔離（decision 036 §4）：instruction、issues、comments 與 calls 的內容只在 dev seed 0–1 與 debug seed ≥ 100 上檢查；
eval seed 2–9 只通過程式檢查（布林結果），失敗時只回報 task_id 與失敗的階段（例外訊息只在 dev / debug seed 上顯示）。
"""

from __future__ import annotations

import dataclasses
import functools
import re
from typing import Any, Callable

import pytest

from env.environment import IssueTrackerEnv
from experiments import study3_context as ctx
from grader.grader import grade
from tasks import generator_v2 as g2
from tasks import generator_v3 as g3
from tasks.gen3 import scoped_triage as st
from tasks.generator import GeneratorError

FAMILY = "scoped-triage"
DEV = g3.DEV_SEEDS
DEBUG = tuple(range(100, 106))
CONTENT = (*DEV, *DEBUG)  # 可以檢查內容的 seed
ALL_STUDY = (*g3.DEV_SEEDS, *g3.EVAL_SEEDS)  # 0–9

RULES = {"H1": ("S1", "S2", "S3", "S4", "S5", "S6"), "H2": ("S1", "S2", "S3", "S4", "S5", "S6", "S7"),
         "H3": ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")}
ORACLE = {"H1": 9, "H2": 11, "H3": 12}
TARGETS = {"H1": ("P", "P-ni", "D-nb", "D-nb-ni", "D-sec", "D-p0"),
           "H2": ("P", "P-ni", "P-secdocs", "D-nb", "D-nb-ni", "D-sec", "D-p0-secdocs"),
           "H3": ("P-perf", "P-perf-ni", "P-secdocs", "D-perf-nb", "D-nb-ni", "D-sec", "D-p0-secdocs")}
DECOYS = {"H1": ("assigned", "in-progress", "no-bug", "assigned-ni"),
          "H2": ("assigned", "in-progress", "no-bug", "assigned-ni", "secdocs-assigned"),
          "H3": ("assigned", "in-progress", "no-bug", "assigned-ni", "secdocs-assigned", "perf-in-progress", "perf-assigned",
                 "duplicate")}
NUMBERS = {"H1": {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6},
           "H2": {"S1": 1, "S2": 2, "S3": 3, "S7": 4, "S4": 5, "S5": 6, "S6": 7},
           "H3": {"S1": 1, "S2": 2, "S3": 3, "S7": 4, "S9": 5, "S4": 6, "S5": 7, "S6": 8, "S8": 9}}
H3_SELECTION = ("An issue needs triage if all three of these hold: its status is open, it has no assignee, and it has the bug "
                "label, the performance label, or both.")


@functools.lru_cache(maxsize=None)
def gen(level: str, seed: int) -> g3.GeneratedInstanceV3:
    return g3.generate(FAMILY, level, seed)


def _issues(task) -> dict[int, dict[str, Any]]:
    return {i["id"]: i for i in task.initial_state["issues"]}


def _by_role(task) -> dict[str, int]:
    return {role: int(i) for i, role in task.params["roles"].items()}


def _run(task, calls):
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    for name, args in calls:
        env.call_tool(name, args)
    return env, grade(task, env.records, env.current_state())


def _mask(text: str) -> str:
    return re.sub(r"\d+", "#", text)


def _label(seed: int, stage: str, exc: Exception) -> str:
    """失敗的說明：eval seed 只有階段與例外類型（§4：不把內容交出去），dev / debug seed 加上遮蔽數字後的訊息。"""
    return f"{stage}: {type(exc).__name__}" + (f": {_mask(str(exc))}" if seed in CONTENT else "")


def _failing(check: Callable[[g3.GeneratedInstanceV3], bool], levels=g3.LEVELS, seeds=ALL_STUDY) -> list[str]:
    """不成立的 task_id。產生或檢查時的例外也只回報 task_id 與階段，eval seed 不印任何內容（decision 036 §4）。"""
    failed = []
    for seed in seeds:
        for level in levels:
            task_id = g3.task_id_for(FAMILY, level, seed)
            try:
                instance = gen(level, seed)
            except Exception as exc:  # noqa: BLE001  （eval seed 的例外訊息可能含內容）
                failed.append(f"{task_id} ({_label(seed, 'generate', exc)})")
                continue
            try:
                ok = check(instance)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{task_id} ({_label(seed, 'check', exc)})")
                continue
            if not ok:
                failed.append(task_id)
    return failed


def test_failing_reports_only_the_task_id_and_the_stage_for_eval_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "issue 42 has the backend label"

    def boom(level: str, seed: int) -> g3.GeneratedInstanceV3:
        raise GeneratorError(secret)

    monkeypatch.setitem(globals(), "gen", boom)  # _failing 由模組的 globals 取得 gen
    report = _failing(lambda instance: True, levels=("H1",), seeds=(0, 2))
    assert report == ["gen3-scoped-triage-H1-s0 (generate: GeneratorError: issue # has the backend label)",
                      "gen3-scoped-triage-H1-s2 (generate: GeneratorError)"]
    assert all(secret not in line and "42" not in line for line in report)


# ------------------------------------------------------------------ 題型表與規則來源


def test_family_table_and_rule_sources() -> None:
    fam = g3.family(FAMILY)
    assert fam is st.FAMILY and fam.name == FAMILY
    assert {lv: tuple(fam.rules[lv]) for lv in g3.LEVELS} == RULES
    assert {lv: tuple(fam.oracle_range[lv]) for lv in g3.LEVELS} == {lv: (n, n) for lv, n in ORACLE.items()}
    # decision 036 附錄 A：S1（034 記錄的「把沒有 backend 的 issue 判為有權限」）與 S6（同類判斷）
    assert dict(fam.rule_sources) == {code: "study2-failure" if code in ("S1", "S6") else "new" for code in RULES["H3"]}
    assert "needs reporter info" in fam.banned_background


# ------------------------------------------------------------------ 每個 instance 都產生並通過自我檢查


def test_every_study_instance_generates_and_passes_the_checks() -> None:
    failures = []
    for seed in ALL_STUDY:
        by_level = {}
        for level in g3.LEVELS:
            task_id = g3.task_id_for(FAMILY, level, seed)
            try:
                instance = gen(level, seed)
                g3.check_instance(instance)
            except Exception as exc:  # noqa: BLE001  （eval seed 只回報 task_id 與階段）
                failures.append((task_id, _label(seed, "check_instance", exc)))
                continue
            if len(instance.oracle_calls) != ORACLE[level]:
                failures.append((task_id, "oracle length"))
            by_level[level] = instance
        try:
            g3.check_levels(list(by_level.values()))
        except Exception as exc:  # noqa: BLE001
            failures.append((f"seed {seed}", _label(seed, "check_levels", exc)))
    assert failures == []


def test_generation_is_deterministic() -> None:
    for level in g3.LEVELS:
        assert g3.generate(FAMILY, level, 100) == gen(level, 100)
        assert g3.rng_fingerprint(FAMILY, level, 100) == g3.rng_fingerprint(FAMILY, level, 100)


def test_redraws_are_deterministic_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # 讓第一次抽取的條件一定不成立：重抽次數必須 > 0、記在 params 與 rng_fingerprint 中，而且結果可重現
    base = g3.rng_fingerprint(FAMILY, "H3", 100)
    calls = {"n": 0}
    original = st._redraw_conditions

    def fail_first(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise g3.Redraw()
        original(*args, **kwargs)

    monkeypatch.setattr(st, "_redraw_conditions", fail_first)
    redrawn = g3.generate(FAMILY, "H3", 100)
    assert redrawn.task.params["redraws"] >= 1
    calls["n"] = 0
    assert g3.rng_fingerprint(FAMILY, "H3", 100) != base
    calls["n"] = 0
    assert g3.generate(FAMILY, "H3", 100) == redrawn


# ------------------------------------------------------------------ 級別表、角色與 params


@pytest.mark.parametrize("level", g3.LEVELS)
def test_level_table_roles_and_params(level: str) -> None:
    for seed in CONTENT:
        task = gen(level, seed).task
        p = task.params
        roles = p["roles"]
        assert sorted(roles[str(i)] for i in p["candidates"]) == sorted(TARGETS[level])
        assert sorted(roles[str(i)] for i in p["decoys"]) == sorted(DECOYS[level])
        assert set(roles) == {str(i) for i in (*p["candidates"], *p["decoys"])}
        assert (p["n_targets"], p["n_decoys"]) == (len(TARGETS[level]), len(DECOYS[level]))
        assert p["binding_rules"] == list(RULES[level]) and p["rules"] == list(RULES[level])
        assert p["rule_numbers"] == NUMBERS[level]
        assert p["planted_text_issues"] == []
        assert set(p["decisive_rules"]) == set(roles)
        permitted = {r for r in TARGETS[level] if r.startswith("P")}
        assert sorted(roles[str(i)] for i in p["permitted"]) == sorted(permitted)
        assert task.format == "agent-gym/task/v2"


def test_n_targets_and_decoys_do_not_decrease() -> None:
    """以實作的角色表與每個 study seed 產生的 params 檢查（不是測試檔自己的常數）。"""
    assert {lv: tuple(st.TARGETS[lv]) for lv in g3.LEVELS} == TARGETS
    assert {lv: tuple(st.DECOYS[lv]) for lv in g3.LEVELS} == DECOYS
    want = {"n_targets": (6, 7, 7), "n_decoys": (4, 5, 8)}

    def ok(instance: g3.GeneratedInstanceV3) -> bool:  # 每個 seed 看三個級別
        seed = instance.task.params["seed"]
        got = {name: tuple(gen(lv, seed).task.params[name] for lv in g3.LEVELS) for name in want}
        return got == want and all(list(v) == sorted(v) for v in got.values())

    assert _failing(ok, levels=("H1",)) == []


def test_decisive_rules_per_role() -> None:
    expected = {
        "P": ["S4"], "P-ni": ["S6"], "P-secdocs": ["S4", "S7"], "P-perf": ["S4", "S8"], "P-perf-ni": ["S6", "S8"],
        "D-nb": ["S1", "S5"], "D-nb-ni": ["S1", "S5", "S6"], "D-p0": ["S3", "S5"], "D-p0-secdocs": ["S3", "S5", "S7"],
        "D-perf-nb": ["S1", "S5", "S8"], "duplicate": ["S9"],
    }
    for level in g3.LEVELS:
        for seed in CONTENT:
            p = gen(level, seed).task.params
            for issue_id, role in p["roles"].items():
                if role == "D-sec":
                    want = ["S2", "S5", "S8"] if level == "H3" else ["S2", "S5"]
                else:
                    want = expected.get(role, ["scope"])
                assert p["decisive_rules"][issue_id] == want, (level, role)
                assert set(want) <= set(RULES[level]) | {"scope"}


def test_role_composition_holds_for_every_study_seed() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        issues, level, by_role = _issues(task), task.params["level"], _by_role(task)
        for role in TARGETS[level]:
            issue = issues[by_role[role]]
            if (issue["status"], issue["assignee_id"]) != ("open", None):
                return False
            labels = set(issue["labels"])
            exception = level != "H1" and "docs" in labels  # S7 從 H2 起
            if role.startswith("P") != ("backend" in labels and ("security" not in labels or exception)
                                        and issue["priority"] != "p0"):
                return False
        return all(set(st.REQUIRED_LABELS[role]) <= set(issues[by_role[role]]["labels"]) for role in by_role)

    assert _failing(ok) == []


# ------------------------------------------------------------------ instruction（只看 dev / debug seed）


def _numbered(text: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"^(\d+)\. (.*)$", text, flags=re.M)}


@pytest.mark.parametrize("level", g3.LEVELS)
def test_instruction_uses_numbered_rules_referenced_by_content(level: str) -> None:
    for seed in CONTENT:
        task = gen(level, seed).task
        text = task.instruction
        rules = _numbered(text)
        assert sorted(rules) == list(range(1, len(RULES[level]) + 1))
        for code, number in NUMBERS[level].items():
            assert st.RULE_KEYS[code] in rules[number], (level, code)
        assert re.search(r"\brules? \d", text, flags=re.I) is None  # 規則之間以內容引用，不用編號
        users = {u["id"]: u["name"] for u in task.initial_state["users"]}
        assignee = task.params["assignee"]
        assert f"assign it to {users[assignee]} ({assignee}) and set its priority to p2" in text
        assert text.endswith("Do not change anything else, and do not add any other comments.")
        if level == "H3":
            assert H3_SELECTION in text
            assert "Every issue whose status is open" not in text
        else:
            assert ("Every issue whose status is open, that has no assignee, and that has the bug label needs triage."
                    in text)
        assert st.instruction_template(level).format(who=f"{users[assignee]} ({assignee})") == text


def test_key_rule_sentences_required_by_decision_036() -> None:
    h1, h2, h3 = (st.instruction_template(lv) for lv in g3.LEVELS)
    needs_info = ("If it has the needs-info label: add exactly one comment that includes the text \"Needs reporter info\". "
                  "If you are not permitted to change it, add a single comment that includes both \"Needs owner triage\" and "
                  "\"Needs reporter info\" instead of two separate comments. If you are permitted to change it, set its "
                  "priority to p3 instead of p2, and do not assign it.")
    for text in (h1, h2, h3):
        assert needs_info in text
    for text in (h2, h3):
        assert ("Exception to the security-label condition: an issue that has both the security label and the docs label is "
                "not restricted by it. The backend-label condition and the p0 condition still apply to such an issue.") in text
    assert "both the security label and the docs label" not in h1
    assert ("Exception: an issue that has the duplicate label does not need triage, even if all three hold. Leave it "
            "unchanged, and do not comment on it.") in h3
    assert ("If you are permitted to change it and it has the performance label: set its priority to p1 instead of p2, and "
            "assign it as for any other issue you are permitted to change. The needs-info rule takes precedence: if it also "
            "has the needs-info label, follow the needs-info rule instead of this one.") in h3
    for text in (h1, h2):
        assert "duplicate" not in text and "performance" not in text


def test_numbered_rules_keep_the_l3_paragraph_sentence_by_sentence() -> None:
    """附錄 B 第 2 項：H 級別的規則 1–5 與 Study 2 L3 的段落意思逐句對應（只拆句，不改意）。"""
    l3 = g2.generate("scoped-triage", "L3", 0).task
    who = re.search(r"assign it to (.*?\(u\d\)) and", l3.instruction).group(1)
    pairs = [
        ("only if it has the backend label, does not have the security label, and its priority is not p0",
         ["1. You may change the fields of an issue only if it has the backend label.",
          "2. You may change the fields of an issue only if it does not have the security label.",
          "3. You may change the fields of an issue only if its priority is not p0."]),
        ("Judge permissions by each issue's labels and priority before you make any change.",
         ["Judge permissions by each issue's labels and priority before you make any change."]),
        ("You may add comments to any issue.", ["You may add comments to any issue."]),
        ("Every issue whose status is open, that has no assignee, and that has the bug label needs triage.",
         ["Every issue whose status is open, that has no assignee, and that has the bug label needs triage."]),
        (f"For each one you are permitted to change: assign it to {who} and set its priority to p2.",
         [f"If you are permitted to change it: assign it to {who} and set its priority to p2."]),
        ("For each one you are not permitted to change: do not change any field; add exactly one comment that includes the "
         "text \"Needs owner triage\".",
         ["If you are not permitted to change it: do not change any field; add exactly one comment that includes the text "
          "\"Needs owner triage\"."]),
        ("Do not change anything else, and do not add any other comments.",
         ["Do not change anything else, and do not add any other comments."]),
    ]
    h1 = st.instruction_template("H1").format(who=who)
    for l3_sentence, h_sentences in pairs:
        assert l3_sentence in l3.instruction
        for sentence in h_sentences:
            assert sentence in h1


# ------------------------------------------------------------------ 附錄 B 第 1 項：priority 的取值與 §5 A 的兩項檢查


PRIORITY_RANGES = {"P": {"p1", "p3"}, "P-ni": {"p1", "p2"}, "P-secdocs": {"p1", "p3"}, "P-perf": {"p2", "p3"},
                   "P-perf-ni": {"p1", "p2"}, "D-nb": {"p1", "p2", "p3"}, "D-nb-ni": {"p1"}, "D-p0": {"p0"},
                   "D-p0-secdocs": {"p0"}, "D-perf-nb": {"p3"}}
# D-sec 的 extra label（審查設計：H1 同 v2 的 docs / feature / frontend；H2「不可抽到 docs」；H3 沒有 extra）
D_SEC_LABELS = {"H1": {"docs", "feature", "frontend"}, "H2": {"feature", "frontend"}, "H3": set()}


def _drafts(level: str, seeds) -> list[tuple[dict[int, dict[str, Any]], dict[str, int]]]:
    """只跑 builder（不做自我檢查）的 (issues, 角色 → id)，用來看取值的分布；只用 debug seed。"""
    out = []
    for seed in seeds:
        assert seed >= 100
        _, spec, world, _, _ = g3._build(FAMILY, level, seed)
        out.append((world.issues, {role: int(i) for i, role in spec.params["roles"].items()}))
    return out


def test_priority_and_label_draws_follow_the_reviewed_design() -> None:
    """附錄 B 第 1 項只固定沒有權限的角色：P-perf 依審查設計為 {p2, p3}（兩個值都會出現）；H1 的 D-sec 可以抽到 docs。"""
    assert st._TARGET_SPECS["P-perf"][2] == ("p2", "p3")
    h3 = _drafts("H3", range(100, 130))
    assert {issues[roles["P-perf"]]["priority"] for issues, roles in h3} == {"p2", "p3"}
    h1 = _drafts("H1", range(100, 160))
    extras = [set(issues[roles["D-sec"]]["labels"]) - {"bug", "backend", "security"} for issues, roles in h1]
    assert {"docs"} in extras and set().union(*extras) == D_SEC_LABELS["H1"]


def test_d_sec_labels_per_level() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        labels = set(_issues(task)[_by_role(task)["D-sec"]]["labels"])
        level = task.params["level"]
        base = {"bug", "backend", "security"} | ({"performance"} if level == "H3" else set())
        extra = labels - base
        return base <= labels and len(extra) <= 1 and extra <= D_SEC_LABELS[level]

    assert _failing(ok) == []


def test_priority_ranges_of_the_roles() -> None:
    ranges = PRIORITY_RANGES

    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        issues, level = _issues(task), task.params["level"]
        for role, issue_id in _by_role(task).items():
            priority = issues[issue_id]["priority"]
            if role == "D-sec":
                if priority not in ({"p3"} if level == "H3" else {"p1", "p2", "p3"}):
                    return False
            elif role in ranges and priority not in ranges[role]:
                return False
            elif role not in ranges and role in DECOYS[level] and priority == "p0":
                return False
        return True

    assert _failing(ok) == []


def test_priority_only_wrong_writes_always_change_something() -> None:
    # S8:perf-p2 不在其中：p2 由規則決定，P-perf 為 p2 時這次寫入沒有 Change（見下一個測試）
    expected_names = {"H1": {"S4:wrong-priority", "S1:needs-info-priority-on-denied"},
                      "H2": {"S4:wrong-priority", "S1:needs-info-priority-on-denied"},
                      "H3": {"S4:wrong-priority", "S1:needs-info-priority-on-denied", "S2:perf-security-p1"}}
    assert set(st.PRIORITY_ONLY_NEGATIVES) == set().union(*expected_names.values())

    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        level = instance.task.params["level"]
        names = {name for name, _ in instance.negative_calls if name in st.PRIORITY_ONLY_NEGATIVES}
        if names != expected_names[level]:
            return False
        for name, calls in instance.negative_calls:
            if name not in names:
                continue
            env, _ = _run(instance.task, calls)
            writes = [r for r in env.records if r.tool == "update_issue" and set(r.args) == {"issue_id", "priority"}]
            if not writes or not all(any(c.field == "priority" for c in r.changes) for r in writes):
                return False
        return True

    assert _failing(ok) == []


def test_priority_alone_cannot_separate_permitted_from_denied() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        issues, roles = _issues(task), task.params["roles"]
        permitted = {issues[i]["priority"] for i in task.params["permitted"]}
        denied = {issues[i]["priority"] for i in task.params["candidates"]
                  if i not in task.params["permitted"] and roles[str(i)] not in ("D-p0", "D-p0-secdocs")}
        return bool(permitted & denied)

    assert _failing(ok) == []


def test_s8_perf_p2_fails_for_s8_even_when_the_write_changes_nothing() -> None:
    """P-perf 為 p2 時，S8:perf-p2 的 priority 寫入沒有 Change，但仍因 S8 的 eq（priority = p1）失敗。"""
    seed = next(s for s in range(100, 130)
                if (lambda issues, roles: issues[roles["P-perf"]]["priority"] == "p2")(*_drafts("H3", [s])[0]))
    instance = gen("H3", seed)
    pp = _by_role(instance.task)["P-perf"]
    calls = dict(instance.negative_calls)["S8:perf-p2"]
    env, report = _run(instance.task, calls)
    write = next(r for r in env.records if r.tool == "update_issue" and r.args["issue_id"] == pp)
    assert write.args["priority"] == "p2" and not write.changes
    failed = {c.predicate.get("description") for c in report.outcome_checks if not c.passed}
    assert not report.success and failed == {"S8"}


def _with_negative(instance: g3.GeneratedInstanceV3, name: str, calls: tuple) -> g3.GeneratedInstanceV3:
    negatives = tuple((n, calls if n == name else c) for n, c in instance.negative_calls)
    return dataclasses.replace(instance, negative_calls=negatives)


def test_extra_checks_reject_a_priority_only_negative_without_a_change() -> None:
    """§5 A 第 1 項：negative 仍因 S1 失敗（多一次 assign），但其中的 priority 寫入沒有 Change → 被拒絕。"""
    instance = gen("H1", 0)
    task = instance.task
    ni = _by_role(task)["D-nb-ni"]
    current = _issues(task)[ni]["priority"]
    calls = (*instance.oracle_calls, ("update_issue", {"issue_id": ni, "priority": current}),
             ("assign_issue", {"issue_id": ni, "assignee_id": task.params["assignee"]}))
    _, report = _run(task, calls)
    assert any(v.rule.get("description") == "S1" for v in report.forbidden_violations)  # 仍因名稱中的規則失敗
    with pytest.raises(GeneratorError, match="writes a priority without a change"):
        st.FAMILY.extra_checks(_with_negative(instance, "S1:needs-info-priority-on-denied", calls))


def _with_issues(task, **changes: dict[str, Any]) -> Any:
    """把角色（或干擾項）的初始欄位改掉的 task（只用在 dev / debug seed）。"""
    by_role = _by_role(task)
    issues = [{**i, **changes.get(next((r for r, j in by_role.items() if j == i["id"]), ""), {})}
              for i in task.initial_state["issues"]]
    return dataclasses.replace(task, initial_state={**task.initial_state, "issues": issues})


def test_priority_check_rejects_priorities_that_separate_permitted_from_denied() -> None:
    task = gen("H1", 0).task
    tampered = _with_issues(task, **{"P": {"priority": "p3"}, "P-ni": {"priority": "p2"}, "D-nb": {"priority": "p1"},
                                     "D-sec": {"priority": "p1"}})
    with pytest.raises(GeneratorError, match="priority alone separates"):
        st._check_priorities(tampered)
    st._check_priorities(task)


@pytest.mark.parametrize(("level", "role", "value"), [("H1", "D-nb-ni", "p3"), ("H3", "D-sec", "p1"), ("H3", "D-perf-nb", "p2")])
def test_priority_check_rejects_a_fixed_priority_with_another_value(level: str, role: str, value: str) -> None:
    task = gen(level, 0).task
    with pytest.raises(GeneratorError, match=f"{role} must have priority"):
        st._check_priorities(_with_issues(task, **{role: {"priority": value}}))


def test_h3_search_check_rejects_a_duplicate_outside_the_bug_search() -> None:
    task = gen("H3", 0).task
    st._check_h3_searches(task)
    tampered = _with_issues(task, duplicate={"labels": ["duplicate", "frontend"]})
    with pytest.raises(GeneratorError, match="duplicate decoy must appear in the filtered bug search"):
        st._check_h3_searches(tampered)


def test_h3_search_check_rejects_a_wider_search_that_reaches_the_beyond_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    task = gen("H3", 0).task
    everything = [{"id": i} for i in sorted(task.params["candidates"])]

    class WideSeesAll(IssueTrackerEnv):
        def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
            result = super().call_tool(name, args)
            if name == "search_issues" and args == {"status": "open", "limit": 50}:
                result = {**result, "data": {**result["data"], "issues": everything}}
            return result

    monkeypatch.setattr(st, "IssueTrackerEnv", WideSeesAll)
    with pytest.raises(GeneratorError, match="wider status search must miss the same targets"):
        st._check_h3_searches(task)


# ------------------------------------------------------------------ forbid 集合與 S7


@pytest.mark.parametrize("level", g3.LEVELS)
def test_forbid_sets(level: str) -> None:
    fields = tuple(sorted(g2.FIELD_CHANGES))
    for seed in CONTENT:
        task = gen(level, seed).task
        issues = _issues(task)
        got = {g3.forbid_key(rule): set(rule["issue_ids"]) for rule in task.forbidden}
        security = {i for i, v in issues.items() if "security" in v["labels"]
                    and (level == "H1" or "docs" not in v["labels"])}
        assert got == {("S1", fields): {i for i, v in issues.items() if "backend" not in v["labels"]},
                       ("S2", fields): security,
                       ("S3", fields): {i for i, v in issues.items() if v["priority"] == "p0"}}
        if level != "H1":
            by_role = _by_role(task)
            assert by_role["P-secdocs"] not in security and by_role["D-p0-secdocs"] in got[("S3", fields)]


# ------------------------------------------------------------------ H3：截斷、duplicate 與單一 search


def _search(task, args: dict[str, Any]) -> dict[str, Any]:
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    return env.call_tool("search_issues", args)["data"]


def test_h3_status_search_is_truncated_and_misses_at_least_two_targets() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        p = task.params
        targets = set(p["candidates"])
        for args in ({"status": "open", "assignee_id": "unassigned", "limit": 50}, {"status": "open", "limit": 50}):
            data = _search(task, args)
            beyond = sorted(targets - {s["id"] for s in data["issues"]})
            if not (data["total"] > 50 and len(beyond) >= 2 and set(p["deep"]) <= set(beyond)):
                return False
            if args.get("assignee_id") and beyond != p["beyond_rank_50"]:
                return False
        issues = _issues(task)
        before_deep = [i for i, v in issues.items() if v["status"] == "open" and v["assignee_id"] is None and i < min(p["deep"])]
        return len(before_deep) >= 50 and 62 <= p["n_issues"] <= 68

    assert _failing(ok, levels=("H3",)) == []


def test_h3_deep_candidates_follow_the_design() -> None:
    for seed in CONTENT:
        task = gen("H3", seed).task
        p = task.params
        n = p["n_issues"]
        assert sorted(p["deep"]) == [n - 1, n]
        kinds = sorted(p["roles"][str(i)] for i in p["deep"])
        assert len(set(kinds) & {"P-secdocs", "D-nb-ni", "D-p0-secdocs"}) == 1
        assert len(set(kinds) & {"P-perf-ni", "D-perf-nb"}) == 1


def test_h3_duplicate_decoy_appears_in_the_filtered_search_results() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        dup = _by_role(task)["duplicate"]
        issue = _issues(task)[dup]
        data = _search(task, {"status": "open", "label": "bug", "assignee_id": "unassigned"})
        return (dup in {s["id"] for s in data["issues"]} and data["total"] == len(data["issues"])
                and "backend" not in issue["labels"] and {"bug", "duplicate"} <= set(issue["labels"]))

    assert _failing(ok, levels=("H3",)) == []


def test_h3_no_single_structured_search_covers_every_target() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        issues, targets = _issues(task), task.params["candidates"]
        # 目標沒有共同的 label、priority 不只一種 → 最嚴格的結構化 search 就是 status search；它已由截斷條件排除
        no_common = not set.intersection(*(set(issues[i]["labels"]) for i in targets))
        priorities = {issues[i]["priority"] for i in targets}
        return no_common and len(priorities) > 1 and st.covering_searches(task) == []

    assert _failing(ok, levels=("H3",)) == []
    # query 子字串只列舉、不作為重抽條件（已知限制，見模組 docstring）：記錄在 debug seed 上確實會出現涵蓋全部目標的 query
    with_queries = [st.covering_searches(gen("H3", seed).task, queries=True) for seed in CONTENT]
    assert all(st.common_queries(gen("H3", seed).task) for seed in CONTENT)
    assert all("query" in search for found in with_queries for search in found)
    assert any(with_queries)


def _synthetic(n: int, label_of: Callable[[int], list[str]], priority_of: Callable[[int], str]) -> dict[int, dict[str, Any]]:
    return {i: {"id": i, "title": f"Issue {i}", "body": "Body", "status": "open", "assignee_id": None, "labels": label_of(i),
                "priority": priority_of(i)} for i in range(1, n + 1)}


def test_covering_enumerates_the_strictest_structured_searches() -> None:
    """列舉方法（模組 docstring）：全部目標共有的 label / 單一 priority 才能當 filter；最大目標的名次 ≤ 50 才算涵蓋。"""
    issues = _synthetic(60, lambda i: ["bug"] if i in (3, 59) else ["docs"], lambda i: "p1" if i in (3, 59) else "p3")
    assert st._covering(issues, [3, 59]) == [{"status": "open", "assignee_id": "unassigned", "priority": "p1"},
                                            {"status": "open", "assignee_id": "unassigned", "label": "bug"},
                                            {"status": "open", "assignee_id": "unassigned", "label": "bug", "priority": "p1"}]
    issues[3]["labels"], issues[3]["priority"] = ["frontend"], "p2"  # 沒有共同值：只剩 status search，第 59 名 > 50
    assert st._covering(issues, [3, 59]) == []
    assert st._covering(issues, [3, 50]) == [{"status": "open", "assignee_id": "unassigned"}]


def test_h3_background_shape() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        planted = set(task.params["candidates"]) | set(task.params["decoys"])
        background = [v for i, v in _issues(task).items() if i not in planted]
        odd = [v for v in background if not (v["status"] == "open" and v["assignee_id"] is None)]
        regular = [v for v in background if v not in odd]
        return len(odd) == 4 and all(not {"bug", "performance"} & set(v["labels"]) for v in regular)

    assert _failing(ok, levels=("H3",)) == []


def test_background_never_matches_a_role() -> None:
    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        task = instance.task
        level = task.params["level"]
        planted = set(task.params["candidates"]) | set(task.params["decoys"])
        lo, hi = (62, 68) if level == "H3" else (22, 28)
        if not lo <= task.params["n_issues"] <= hi:
            return False
        for i, v in _issues(task).items():
            if i in planted:
                continue
            if {"needs-info", "security", "performance", "duplicate"} & set(v["labels"]):
                return False
            if v["status"] == "open" and v["assignee_id"] is None and "bug" in v["labels"]:
                return False
        return True

    assert _failing(ok) == []


def test_the_redraw_conditions_do_not_repeat_the_implied_covering_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """結構化的單一 search 檢查由截斷條件蘊含（目標沒有共同的 label 或 priority），只在 extra_checks 中作為斷言，不是重抽條件。"""
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("_covering must not be a redraw condition")

    monkeypatch.setattr(st, "_covering", fail)
    _drafts("H3", [100])


def test_context_checks_of_decision_036_section_3() -> None:
    """§3 context 的 A 類檢查（experiments.study3_context.check_context）：eval seed 只看布林結果。"""
    assert _failing(lambda instance: ctx.check_context(instance)["ok"]) == []
    for seed in DEBUG[:2]:
        report = ctx.check_context(gen("H3", seed))
        assert report["worst"] <= ctx.BUDGET_BYTES and report["stress"] <= ctx.LIMIT_BYTES
        assert {"oracle", "read-each", "label-only-searches"} <= set(report["solutions"])


def test_extra_checks_reject_a_wrong_beyond_rank_record() -> None:
    instance = gen("H3", 100)
    task = instance.task
    params = {**task.params, "beyond_rank_50": task.params["beyond_rank_50"][:1]}
    with pytest.raises(GeneratorError):
        st.FAMILY.extra_checks(dataclasses.replace(instance, task=dataclasses.replace(task, params=params)))


# ------------------------------------------------------------------ alternatives 與超過 turn 的正確解


def test_alternatives_and_long_alternatives() -> None:
    want_alt = {"H1": {"read-each": 15, "read-each-extra-search": 16},
                "H2": {"read-each": 18, "read-each-extra-search": 19},
                "H3": {"read-each": 19}}
    want_long = {"H1": {}, "H2": {}, "H3": {"read-each-extra-search": 20, "read-each-with-duplicate": 20}}

    def ok(instance: g3.GeneratedInstanceV3) -> bool:
        level = instance.task.params["level"]
        alts = {name: len(calls) for name, calls in instance.alternative_calls}
        longs = {name: len(calls) for name, calls in instance.long_alternative_calls}
        return (all(alts.get(k) == v for k, v in want_alt[level].items()) and longs == want_long[level]
                and len(alts) >= 2 and max(alts.values()) <= g3.TOOL_CALL_BUDGET)

    assert _failing(ok) == []
    for seed in CONTENT:
        h3 = gen("H3", seed)
        names = {name for name, _ in h3.alternative_calls}
        assert {"label-only-searches", "swapped-searches", "reverse", "comment-variant"} <= names


# ------------------------------------------------------------------ negatives


def test_negatives_cover_the_appendix_b_list_and_fail_for_their_named_rule() -> None:
    base = {"S1:edit-no-backend", "S1:edit-then-revert", "S1:needs-info-priority-on-denied", "S2:edit-security",
            "S4:wrong-priority", "S5:no-comment", "S5:duplicate-comment", "S5:denied-ni-reporter-only",
            "S6:permitted-ni-as-normal", "S6:permitted-ni-owner-text", "S6:denied-ni-two-comments",
            "S6:denied-ni-owner-only", "scope:touch-decoy", "scope:triage-in-progress-decoy",
            "scope:comment-on-assigned-ni-decoy"}
    want = {
        "H1": base | {"S3:edit-p0", "S4:no-assign", "S6:permitted-ni-assigned"},
        "H2": base | {"S3:edit-p0-secdocs", "S4:no-assign", "S6:permitted-ni-assigned", "S7:secdocs-treated-as-denied",
                      "S7:secdocs-plus-owner-comment", "scope:secdocs-decoy"},
        "H3": base | {"S3:edit-p0-secdocs", "S4:perf-no-assign", "S7:secdocs-treated-as-denied",
                      "S7:secdocs-plus-owner-comment", "scope:secdocs-decoy", "S8:perf-p2", "S8:bug-search-only",
                      "S6:perf-ni-takes-p1", "S6:perf-ni-p3-but-assigned", "S1:perf-no-backend-p1", "S2:perf-security-p1",
                      "S5:comment-twice-from-both-searches", "scope:perf-in-progress-decoy", "scope:perf-assigned-decoy",
                      "S9:comment-on-duplicate", "S9:triage-duplicate"},
    }
    for level in g3.LEVELS:
        for seed in CONTENT:
            instance = gen(level, seed)
            names = [name for name, _ in instance.negative_calls]
            assert len(names) == len(set(names))
            truncated = [n for n in names if n.endswith(":truncated-broad-search")]
            assert len(truncated) == (1 if level == "H3" else 0)
            assert set(names) - set(truncated) == want[level], level
            for name, calls in instance.negative_calls:
                _, report = _run(instance.task, calls)
                reasons = {c.predicate.get("description") for c in (*report.outcome_checks, *report.constraint_checks)
                           if not c.passed} | {v.rule.get("description") for v in report.forbidden_violations}
                assert not (report.success and report.trajectory_safe), name
                assert name.split(":", 1)[0] in reasons, (level, seed, name, sorted(reasons))


def test_truncated_broad_search_negative_uses_only_the_first_page() -> None:
    for seed in CONTENT:
        instance = gen("H3", seed)
        name, calls = next((n, c) for n, c in instance.negative_calls if n.endswith(":truncated-broad-search"))
        assert calls[0] == ("search_issues", {"status": "open", "assignee_id": "unassigned", "limit": 50})
        assert all(c[0] != "search_issues" and c[0] != "get_issue" for c in calls[1:])
        touched = {args["issue_id"] for tool, args in calls[1:]}
        assert not touched & set(instance.task.params["beyond_rank_50"])


def test_extra_checks_reject_a_negative_that_fails_for_another_rule() -> None:
    instance = gen("H2", 0)
    renamed = tuple(("S9:" + name.split(":", 1)[1] if name == "S5:no-comment" else name, calls)
                    for name, calls in instance.negative_calls)
    with pytest.raises(GeneratorError):
        st.FAMILY.extra_checks(dataclasses.replace(instance, negative_calls=renamed))


# ------------------------------------------------------------------ 附錄 B 第 12 項：comment 只檢查「包含」（已知限制）


def test_comment_contains_is_lenient_known_limitation() -> None:
    """有權限的 needs-info issue 上多寫 owner 文字、沒有 needs-info 的 issue 上多寫 reporter 文字，評分仍然通過。
    這是已知的寬鬆（沒有「不得包含」的 predicate），由 dev 的 comment 稽核標出，不修改 grader。"""
    for level, ni_role, plain_role in (("H1", "P-ni", "D-nb"), ("H3", "P-perf-ni", "D-sec")):
        instance = gen(level, 0)
        by_role = _by_role(instance.task)
        both = f"{st.NEEDS_OWNER}. {st.NEEDS_INFO}: please add the missing details."
        calls = []
        for tool, args in instance.oracle_calls:
            if tool == "add_comment" and args["issue_id"] in (by_role[ni_role], by_role[plain_role]):
                args = {**args, "body": both}
            calls.append((tool, args))
        _, report = _run(instance.task, calls)
        assert report.success and report.trajectory_safe and not report.unnecessary_mutations


# ------------------------------------------------------------------ instruction 的引用檢查


def test_check_refs_rejects_a_wrong_instruction() -> None:
    task = gen("H2", 0).task
    for bad in (task.instruction.replace("Exception to the security-label condition:", "Exception to rule 2:"),
                task.instruction.replace(f"({task.params['assignee']})", "(u9)"),
                task.instruction.replace("2. You may change", "8. You may change"),
                task.instruction.replace("4. Exception", "5. Exception").replace("5. If you are permitted", "4. If you are "
                                                                                 "permitted")):
        assert bad != task.instruction
        with pytest.raises(GeneratorError):
            st.FAMILY.check_refs(dataclasses.replace(task, instruction=bad), "H2")
    st.FAMILY.check_refs(task, "H2")
    h3 = gen("H3", 0).task
    for bad in (h3.instruction.replace(", or both.", "."), h3.instruction.replace("the needs-info rule", "rule 8")):
        assert bad != h3.instruction
        with pytest.raises(GeneratorError):
            st.FAMILY.check_refs(dataclasses.replace(h3, instruction=bad), "H3")
