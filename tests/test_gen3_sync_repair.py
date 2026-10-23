"""generator v3 sync-repair（decision 036 §3 的級別表、已決定的設計選擇、附錄 B sync-repair 第 1–13 項）的契約測試。

資料隔離（decision 036 §4）：檢視內容的斷言只用 dev seed 0–1 與除錯用的 seed ≥ 100（``CONTENT_SEEDS``）。eval seed 2–9 只通過
程式檢查（generate 內的 check_instance、check_levels、結構上的數字）；失敗時只回報 task_id 與檢查名稱，不印出 instance 的內容。
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import re

import pytest

from env.environment import IssueTrackerEnv
from experiments import study3_context as ctx
from grader.grader import grade
from tasks import generator_v2 as v2
from tasks import generator_v3 as g3
from tasks.gen3 import sync_repair as sr
from tasks.generator import GeneratorError

FAMILY = "sync-repair"
ALL_SEEDS = tuple(range(10))
CONTENT_SEEDS = (0, 1, *range(100, 124))  # dev + 除錯 seed：可以檢視內容
LEVEL_RULES = {
    "H1": ["Y1", "Y2", "Y3", "Y4", "Y5"],
    "H2": ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7"],
    "H3": ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"],
}
LEVEL_ROLES = {"H1": ["A", "B", "C", "D", "E"], "H2": ["A", "B", "C", "K", "S", "D"], "H3": ["Q", "B", "S", "D", "K", "KH"]}
ORACLE_LENGTH = {"H1": 10, "H2": 11, "H3": 12}
READ_EACH_LENGTH = {"H1": 15, "H2": 17, "H3": 18}  # 附錄 B sync-repair 第 2 項
N_TARGETS = {"H1": 7, "H2": 8, "H3": 9}  # 列出的項目數（合併之前）
N_DECOYS = {"H1": 3, "H2": 4, "H3": 4}
FIELD_CHANGES = ("title", "body", "status", "priority", "assignee_id", "labels")

# ------------------------------------------------------------------ instruction 的模板（附錄 A 的逐字稿）

HEADER_ONE = ("Issue #{R} is a report from the sync job (Sync Job, sync). Every change it lists was made by mistake. "
              "Judge everything by the values the issues have at the start of the task, before you make any change.")
HEADER_TWO = ("Issues #{R1} and #{R2} are reports from the sync job (Sync Job, sync); #{R2} is from a later sync run than #{R1}. "
              "Every change they list was made by mistake. Judge everything by the values the issues have at the start of the "
              "task, before you make any change.")
RULE_1 = ("1. Undo each listed change by restoring the value from before the change. For a label change, only add back or "
          "remove that one label and keep all other labels as they are.")
RULE_2 = ("2. If the current value of the field that a listed change set is no longer the value the sync job set, that change is "
          "already handled: do not undo it. A label change is handled if the label the sync job added is gone, or the label it "
          "removed is back. If every listed change on an issue is handled, leave that issue unchanged and do not comment on it.")
RULE_3_ONE = ("3. The report can list several changes to the same issue. Judge each of them on its own under rules 1 and 2: undo "
              "every one that is not handled, even if another change on that issue is handled. You may undo changes to different "
              "fields of one issue in a single update.")
RULE_3_TWO = ("3. The reports can list several changes to the same issue, in one report or across both. Judge each of them on its "
              "own under rules 1 and 2: undo every one that is not handled, even if another change on that issue is handled. You "
              "may undo changes to different fields of one issue in a single update.")
RULE_4 = ("4. Add exactly one comment that includes the text \"Reverted sync change\" to each issue on which you undo at least one "
          "change, even if you undo several changes on it.")
RULE_5 = ("5. You must not edit issues that are closed at the start of the task. An issue whose status is closed at the start of "
          "the task is locked: do not change any of its fields, even to undo a listed change. If a locked issue has at least one "
          "listed change that is not handled, add exactly one comment to it that includes the text \"Sync revert skipped\".")
RULE_6_ONE = ("6. Exception to rule 5: if the report lists a change that set an issue's status to closed, the sync job closed that "
              "issue by mistake, so it is not locked: undo its listed changes under the other rules (this sets its status back to "
              "the value it had before the sync job closed it).")
RULE_6_TWO = ("6. Exception to rule 5: if a report lists a change that set an issue's status to closed, the sync job closed that "
              "issue by mistake, so it is not locked: undo its listed changes under the other rules (this sets its status back to "
              "the value it had before the sync job closed it).")
RULE_7 = ("7. If both reports list a change to the priority of the same issue, the sync job changed that field twice: first the "
          "change in #{R1}, then the one in #{R2}. Treat the two as one change (this replaces rules 1–3 for that field): it is "
          "handled only if the current value differs from the value set in #{R2}, and undoing it restores the value from before "
          "the change in #{R1}.")
LAST = "Do not change anything else, and do not add any other comments."
TEMPLATES = {
    "H1": "\n".join((HEADER_ONE, RULE_1, RULE_2, RULE_3_ONE, RULE_4, LAST)),
    "H2": "\n".join((HEADER_ONE, RULE_1, RULE_2, RULE_3_ONE, RULE_4, RULE_5, RULE_6_ONE, LAST)),
    "H3": "\n".join((HEADER_TWO, RULE_1, RULE_2, RULE_3_TWO, RULE_4, RULE_5, RULE_6_TWO, RULE_7, LAST)),
}


@functools.lru_cache(maxsize=None)
def _gen(level: str, seed: int) -> g3.GeneratedInstanceV3:
    return g3.generate(FAMILY, level, seed)


def _gen_checked(level: str, seed: int) -> g3.GeneratedInstanceV3:
    """eval seed 用：產生失敗時只回報 task_id 與檢查名稱。"""
    task_id = g3.task_id_for(FAMILY, level, seed)
    try:
        return _gen(level, seed)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{task_id}: {_check_name(exc, task_id)}", pytrace=False)


def _content() -> list[tuple[str, int]]:
    return [(level, seed) for level in g3.LEVELS for seed in CONTENT_SEEDS]


def _check_name(exc: BaseException, task_id: str) -> str:
    """失敗的檢查名稱：去掉 task_id 後，第一個數字、id 或括號之前的部分（不印出 eval instance 的內容）。"""
    text = str(exc).replace(f"self-check of {task_id} failed: ", "")
    return f"{type(exc).__name__}: " + re.split(r"[\d\[\]#{}('\"]", text, maxsplit=1)[0].strip()[:80]


def _run(task, calls):
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    for name, args in calls:
        env.call_tool(name, args)
    return env, grade(task, env.records, env.current_state())


def _by_role(task) -> dict[str, int]:
    return {role: int(i) for i, role in task.params["roles"].items() if role != "decoy"}


def _entries(task, role: str) -> list[dict]:
    iid = _by_role(task)[role]
    return [e for e in task.params["entries"] if e["issue_id"] == iid]


def _issue(task, iid: int) -> dict:
    return next(i for i in task.initial_state["issues"] if i["id"] == iid)


def _calls_on(calls, iid: int) -> list:
    return [c for c in calls if c[0] != "search_issues" and c[1].get("issue_id") == iid]


# ------------------------------------------------------------------ 題型的登記


def test_the_family_matches_the_decision_036_table() -> None:
    family = g3.family(FAMILY)
    assert family is sr.FAMILY and family.name == FAMILY
    assert {lv: list(family.rules[lv]) for lv in g3.LEVELS} == LEVEL_RULES
    assert dict(family.oracle_range) == {lv: (n, n) for lv, n in ORACLE_LENGTH.items()}
    assert dict(family.rule_sources) == {"Y1": "new", "Y2": "new", "Y3": "new", "Y4": "study2-failure", "Y5": "new",
                                         "Y6": "study2-failure", "Y7": "new", "Y8": "new"}
    assert {"archived", "locked", "skipped", "sync revert", "sync run"} <= set(g3.BANNED_BACKGROUND)  # 附錄 B 第 13 項


# ------------------------------------------------------------------ 全部 seed：只通過程式檢查（eval seed 不看內容）


@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_every_instance_passes_check_instance_and_check_levels(seed: int) -> None:
    pool = [_gen_checked(level, seed) for level in g3.LEVELS]
    try:
        g3.check_levels(pool)
    except GeneratorError as exc:
        pytest.fail(f"seed {seed}: {_check_name(exc, '')}", pytrace=False)


@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_structural_numbers_on_every_seed(seed: int) -> None:
    """只比對結構上的數字（由設計固定，不洩漏 eval instance 的內容）。"""
    for level in g3.LEVELS:
        instance = _gen_checked(level, seed)
        p = instance.task.params
        alternatives = dict(instance.alternative_calls)
        failed = [name for name, ok in (
            ("oracle length", len(instance.oracle_calls) == ORACLE_LENGTH[level]),
            ("rules", p["rules"] == LEVEL_RULES[level]),
            ("binding rules", p["binding_rules"] == LEVEL_RULES[level]),
            ("n_targets", p["n_targets"] == N_TARGETS[level] == len(p["entries"])),
            ("n_decoys", p["n_decoys"] == N_DECOYS[level] == len(p["decoys"])),
            ("world size", 30 <= p["n_issues"] <= 36),
            ("roles", sorted(r for r in p["roles"].values() if r in sr.ROLE_NAMES) == sorted(LEVEL_ROLES[level])),
            ("reports", len(p["reports"]) == (2 if level == "H3" else 1) and p["planted_text_issues"] == p["reports"]),
            ("read-each", len(alternatives.get("read-each", ())) == READ_EACH_LENGTH[level]),
            ("alternatives within the turn budget", sum(len(c) <= g3.TOOL_CALL_BUDGET for c in alternatives.values()) >= 2),
            ("no long alternatives", instance.long_alternative_calls == ()),
            ("decisive rules cover the roles", set(p["decisive_rules"]) == set(p["roles"])),
            ("redraws", p["redraws"] == 0),
        ) if not ok]
        assert not failed, f"{instance.task.task_id}: {failed}"


@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_the_context_fits_on_every_seed(seed: int) -> None:
    """decision 036 §3「context 與 agent_error」（§5 A）：oracle 與每個 alternative ≤ 45,875 bytes，壓力情境 ≤ 57,344 bytes。

    失敗時只回報 task_id 與檢查名稱（eval seed 不印出大小以外的任何內容）。"""
    for level in g3.LEVELS:
        instance = _gen_checked(level, seed)
        task_id = instance.task.task_id
        try:
            report = ctx.check_context(instance)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{task_id}: {_check_name(exc, task_id)}", pytrace=False)
        failed = [name for name, ok in (("conversation", report["worst"] <= ctx.BUDGET_BYTES),
                                        ("stress", report["stress"] <= ctx.LIMIT_BYTES), ("ok", report["ok"])) if not ok]
        if failed:
            pytest.fail(f"{task_id}: context {failed}", pytrace=False)


# ------------------------------------------------------------------ instruction（dev 與除錯 seed）


@pytest.mark.parametrize(("level", "seed"), _content())
def test_the_instruction_is_the_registered_template(level: str, seed: int) -> None:
    task = _gen(level, seed).task
    reports = task.params["reports"]
    ids = {"R": reports[0]} if level != "H3" else {"R1": reports[0], "R2": reports[1]}
    assert task.instruction == TEMPLATES[level].format(**ids)
    assert {task.params["roles"][str(r)] for r in reports} == set(ids)


def test_the_rule_wording_follows_appendix_b() -> None:
    h2, h3 = _gen("H2", 0).task.instruction, _gen("H3", 0).task.instruction
    for text in (_gen("H1", 0).task.instruction, h2, h3):
        assert "the field that a listed change set" in text  # 第 7 項
        assert "fewer than 50" not in text and "limit" not in text and "lists all" not in text  # 不加涵蓋說明句
    for text in (h2, h3):
        rule5 = next(line for line in text.splitlines() if line.startswith("5. "))
        assert "archived" not in text and "Reverted sync change" not in rule5  # 第 6、12 項
        assert rule5.startswith("5. You must not edit issues that are closed at the start of the task.")
        assert 'add exactly one comment to it that includes the text "Sync revert skipped"' in rule5
        assert "undo its listed changes under the other rules (this sets its status back to the value it had before the sync job closed it)" in text  # 第 8 項
    rule7 = next(line for line in h3.splitlines() if line.startswith("7. "))
    assert "(this replaces rules 1–3 for that field)" in rule7 and "status" not in rule7  # 第 8、12 項


# ------------------------------------------------------------------ 角色與種入（附錄 B 第 2、3 項）


@pytest.mark.parametrize(("level", "seed"), _content())
def test_roles_and_entries_follow_the_level_table(level: str, seed: int) -> None:
    task = _gen(level, seed).task
    p = task.params
    roles = _by_role(task)
    assert sorted(r for r in roles if r in sr.ROLE_NAMES) == sorted(LEVEL_ROLES[level])
    assert len({*p["reports"], *roles.values(), *p["decoys"]}) == len(p["reports"]) + len(LEVEL_ROLES[level]) + N_DECOYS[level]

    def fields(role: str) -> list[str]:
        return [("labels" if e["kind"].startswith("label") else e["kind"]) for e in _entries(task, role)]

    for role in LEVEL_ROLES[level]:
        entries, issue = _entries(task, role), _issue(task, roles[role])
        assert sum(e["kind"].startswith("label") for e in entries) <= 1  # 每個 issue 最多一個 label 項
        for e in entries:
            if e["handled"]:
                assert e["kind"] in ("priority", "label_added", "label_removed")  # 已處理的項只用 priority / label
            if e["kind"] == "status":
                assert (e["set"] == "closed") == (role == "S")  # 只有 S 被 sync 關閉
                assert e["was"] in ("open", "in_progress") and (role == "S" or e["set"] in ("open", "in_progress"))
        assert (issue["status"] == "closed") == (role in ("K", "KH", "S"))
        if role != "Q":
            assert len(set(fields(role))) == len(entries)  # 同一欄位不重複（rule 7 只用在 Q）
        if role == "A":
            assert sorted(fields("A")) in (["labels", "priority"], ["labels", "status"])
            assert not any(e["handled"] for e in entries) and len({e["report"] for e in entries}) == 1
        if role == "B":
            assert len(entries) == 2 and sorted(e["handled"] for e in entries) == [False, True]
        if role in ("C", "KH"):
            assert len(entries) == 1 and entries[0]["handled"]
        if role in ("D", "E", "K"):
            assert len(entries) == 1 and not entries[0]["handled"]
        if role == "K":
            assert entries[0]["kind"] != "status"
        if role == "S":
            assert sorted(fields("S")) == (["labels", "status"] if level == "H3" else ["status"])
            assert not any(e["handled"] for e in entries)
            assert len({e["report"] for e in entries}) == len(entries)  # H3：兩項分在兩份報告
        if role == "Q":
            r1, r2 = sorted(entries, key=lambda e: e["report"])
            assert [r1["kind"], r2["kind"], r1["report"], r2["report"]] == ["priority", "priority", 1, 2]
            assert r2["was"] == r1["set"] and len({r1["was"], r1["set"], r2["set"]}) == 3 and issue["priority"] == r2["set"]
    if level == "H1":
        assert any(f == "labels" for role in ("D", "E") for f in fields(role))  # D、E 至少一個 label 項


def test_h3_reports_are_ordered_runs() -> None:
    for seed in CONTENT_SEEDS:
        task = _gen("H3", seed).task
        r1, r2 = (_issue(task, i) for i in task.params["reports"])
        assert r1["body"].startswith("Sync run 1 report.\n") and r2["body"].startswith("Sync run 2 report.\n")
        assert r1["created_at"] < r2["created_at"] and r1["reporter_id"] == r2["reporter_id"] == "sync"
    for level in ("H1", "H2"):
        report = _issue(_gen(level, 0).task, _gen(level, 0).task.params["reports"][0])
        assert report["body"].startswith("Sync run report.\n") and report["reporter_id"] == "sync"


@pytest.mark.parametrize(("level", "seed"), _content())
def test_reports_are_dated_after_the_issues_they_list(level: str, seed: int) -> None:
    """報告不早於它列出的 issue：建立日期 = 列出的 issue 中最晚的建立日期加一天；H3 的 #R2 至少比 #R1 晚一天（不抽 RNG）。"""
    task = _gen(level, seed).task
    reports = task.params["reports"]
    for n, rid in enumerate(reports, start=1):
        listed = {e["issue_id"] for e in task.params["entries"] if e["report"] == n}
        latest = max(_issue(task, i)["created_at"] for i in listed)
        assert _issue(task, rid)["created_at"] > latest
    if level == "H3":
        assert _issue(task, reports[0])["created_at"] < _issue(task, reports[1])["created_at"]


@pytest.mark.parametrize(("level", "seed"), _content())
def test_the_oracle_acts_on_exactly_the_planted_roles(level: str, seed: int) -> None:
    instance = _gen(level, seed)
    task, oracle = instance.task, instance.oracle_calls
    roles = _by_role(task)
    reads = [("get_issue", {"issue_id": r}) for r in task.params["reports"]] + [("search_issues", {"limit": 50})]
    assert list(oracle[: len(reads)]) == reads
    for role in LEVEL_ROLES[level]:
        calls = _calls_on(oracle[len(reads):], roles[role])
        tools = [name for name, _ in calls]
        if role in ("C", "KH"):
            assert tools == []
        elif role == "K":
            assert tools == ["add_comment"] and "sync revert skipped" in calls[0][1]["body"].casefold()
        else:
            assert tools == ["update_issue", "add_comment"] and "reverted sync change" in calls[1][1]["body"].casefold()
    if level == "H3":
        q = roles["Q"]
        w = next(e["was"] for e in task.params["entries"] if e["issue_id"] == q and e["report"] == 1)
        assert _calls_on(oracle, q)[0][1] == {"issue_id": q, "priority": w}  # 改回 #R1 之前的值，不是中間值
    if "S" in roles:
        update = _calls_on(oracle, roles["S"])[0][1]
        was = next(e["was"] for e in _entries(task, "S") if e["kind"] == "status")
        assert update["status"] == was and ("labels" in update) == (level == "H3")


# ------------------------------------------------------------------ 規則代號（decision 036 §2）


@pytest.mark.parametrize(("level", "seed"), _content())
def test_comment_codes_follow_the_rule_7_merge_convention(level: str, seed: int) -> None:
    task = _gen(level, seed).task
    roles = _by_role(task)
    codes = {p["issue_id"]: (p["description"], p["contains"]) for p in task.expected_outcome if p["type"] == "comment_added"}
    expected = {"A": "Y5", "B": "Y5", "D": "Y2", "E": "Y2", "Q": "Y2", "S": "Y5" if level == "H3" else "Y2"}
    for role in LEVEL_ROLES[level]:
        if role in ("C", "KH"):
            assert roles[role] not in codes
        elif role == "K":
            assert codes[roles[role]] == ("Y6", ["Sync revert skipped"])
        else:
            assert codes[roles[role]] == (expected[role], ["Reverted sync change"])
    for p in task.expected_outcome:
        if p["type"] == "comment_added":
            assert p["max_count"] == 1 and p.get("min_count", 1) == 1


@pytest.mark.parametrize(("level", "seed"), _content())
def test_decisive_rules_per_role(level: str, seed: int) -> None:
    task = _gen(level, seed).task
    p = task.params
    base = {"A": {"Y1", "Y5"}, "B": {"Y4", "Y5"}, "C": {"Y4"}, "D": {"Y1", "Y2"}, "E": {"Y1", "Y2"}, "K": {"Y6"},
            "KH": {"Y4", "Y6"}, "Q": {"Y2", "Y8"}, "S": {"Y5", "Y7"} if level == "H3" else {"Y2", "Y7"}}
    for iid, role in p["roles"].items():
        if role not in sr.ROLE_NAMES:
            assert p["decisive_rules"][iid] == ["scope"]  # 報告與未列出的干擾項
            continue
        label = any(e["kind"].startswith("label") and not e["handled"] for e in _entries(task, role)) and role not in ("K", "KH")
        assert p["decisive_rules"][iid] == sorted(base[role] | ({"Y3"} if label else set())), role


@pytest.mark.parametrize(("level", "seed"), _content())
def test_forbid_sets_per_rule_issue_and_field(level: str, seed: int) -> None:
    task = _gen(level, seed).task
    roles = _by_role(task)
    rules: dict[int, set] = {}
    for rule in task.forbidden:
        for iid in rule["issue_ids"]:
            rules.setdefault(iid, set()).add(g3.forbid_key(rule))
    everything = tuple(sorted(FIELD_CHANGES))
    listed = set(roles.values()) - set(task.params["reports"])
    for i in task.initial_state["issues"]:
        if i["id"] not in listed:
            assert rules[i["id"]] == {("scope", everything)}
    for role in LEVEL_ROLES[level]:
        got = rules.get(roles[role], set())
        if role == "K":
            assert got == {("Y6", everything)}
        elif role == "KH":
            assert got == {("Y4", "*"), ("Y6", everything)}
        elif role == "C":
            assert got == {("Y4", "*")}
        else:
            touched = {"labels" if e["kind"].startswith("label") else e["kind"] for e in _entries(task, role)}
            handled = {"labels" if e["kind"].startswith("label") else e["kind"] for e in _entries(task, role) if e["handled"]}
            assert got == {("scope", tuple(sorted(set(FIELD_CHANGES) - touched))), *(("Y4", (f,)) for f in handled)}


# ------------------------------------------------------------------ negatives 與 alternatives


NEGATIVES = {
    "H1": {"Y1:partial", "Y2:no-comment", "Y2:duplicate-comment", "Y3:overwrite-labels", "Y4:revert-handled",
           "Y4:noop-revert-and-comment", "Y4:revert-then-restore", "Y4:revert-handled-field", "Y5:comment-per-change",
           "Y5:skip-mixed-issue", "scope:touch-other-then-revert", "scope:touch-decoy"},
}
NEGATIVES["H2"] = NEGATIVES["H1"] | {"Y6:undo-locked", "Y6:no-skip-comment", "Y6:both-comments",
                                     "Y7:treat-sync-closed-as-locked", "Y7:leave-closed"}
NEGATIVES["H3"] = (NEGATIVES["H2"] - {"Y4:revert-handled", "Y4:noop-revert-and-comment", "Y5:comment-per-change"}) | {
    "Y4:comment-locked-handled", "Y5:comment-per-report", "Y7:reopen-only", "Y8:restore-intermediate",
    "Y8:first-change-looks-handled"}


@pytest.mark.parametrize(("level", "seed"), _content())
def test_negatives_fail_because_of_their_rule(level: str, seed: int) -> None:
    instance = _gen(level, seed)
    task = instance.task
    negatives = dict(instance.negative_calls)
    assert NEGATIVES[level] <= set(negatives)
    for name, calls in negatives.items():
        _, report = _run(task, calls)
        reasons = {c.predicate["description"] for c in (*report.outcome_checks, *report.constraint_checks) if not c.passed}
        violated = {v.rule["description"] for v in report.forbidden_violations}
        assert name.split(":")[0] in reasons | violated, name
        if name.startswith("scope:"):
            assert violated == {"scope"} and reasons <= {"scope"}, name  # flip 只用目標與已處理欄位以外的欄位（第 10 項）
    d = _by_role(task)["D"]  # Y1:partial 固定拿掉 D（第 10 項）
    assert list(negatives["Y1:partial"]) == [c for c in instance.oracle_calls if c[1].get("issue_id") != d]


@pytest.mark.parametrize(("level", "seed"), _content())
def test_each_level_keeps_a_revertible_discriminating_rule(level: str, seed: int) -> None:
    """附錄 B 第 5 項：H1 的 Y5（B 被略過）、H2 的 Y7（S 沒有重新打開）、H3 的 Y8（Q 改成 X）都可以撤回。"""
    instance = _gen(level, seed)
    task, oracle = instance.task, instance.oracle_calls
    name, role = {"H1": ("Y5:skip-mixed-issue", "B"), "H2": ("Y7:leave-closed", "S"), "H3": ("Y8:restore-intermediate", "Q")}[level]
    calls = list(dict(instance.negative_calls)[name])
    _, wrong = _run(task, calls)
    assert not wrong.success and wrong.trajectory_safe
    failed = {c.predicate["description"] for c in (*wrong.outcome_checks, *wrong.constraint_checks) if not c.passed}
    assert name.split(":")[0] in failed
    fix = _calls_on(oracle, _by_role(task)[role])
    if level == "H3":
        fix = [c for c in fix if c[0] == "update_issue"]
    _, fixed = _run(task, [*calls, *fix])
    assert fixed.success and fixed.trajectory_safe


@pytest.mark.parametrize(("level", "seed"), _content())
def test_alternatives(level: str, seed: int) -> None:
    instance = _gen(level, seed)
    task = instance.task
    alternatives = dict(instance.alternative_calls)
    listed = sorted(int(i) for i, role in task.params["roles"].items() if role in sr.ROLE_NAMES)
    reads = len(task.params["reports"]) + 1
    read_each = alternatives["read-each"]  # 每個不同的列出 issue 各 get_issue 一次（不是每個項目一次）
    assert [c[1]["issue_id"] for c in read_each[reads: reads + len(listed)]] == listed
    assert len(read_each) == len(instance.oracle_calls) + len(listed) == READ_EACH_LENGTH[level]
    assert {"read-each", "no-search", "reverse", "comment-variant", "split-update"} <= set(alternatives)
    assert all(len(calls) <= g3.TOOL_CALL_BUDGET for calls in alternatives.values())
    assert instance.long_alternative_calls == ()


# ------------------------------------------------------------------ 判斷與自我檢查


def test_rule_7_merges_the_two_priority_changes_of_q() -> None:
    for seed in CONTENT_SEEDS:
        task = _gen("H3", seed).task
        q = _by_role(task)["Q"]
        r1 = next(e for e in task.params["entries"] if e["issue_id"] == q and e["report"] == 1)
        current = _issue(task, q)
        assert v2._handled(r1, current)  # 只看 #R1 那一行：看起來已處理
        item = next(it for it in task.params["items"] if it["issue_id"] == q)
        assert item == {"issue_id": q, "field": "priority", "target": r1["was"], "handled": False, "merged": True}
        assert task.params["actions"][str(q)] == "revert"


def test_the_decision_matches_the_planted_roles() -> None:
    for level, seed in _content():
        task = _gen(level, seed).task
        p, roles = task.params, _by_role(task)
        actions = {role: p["actions"][str(roles[role])] for role in LEVEL_ROLES[level]}
        assert actions == {role: "none" if role in ("C", "KH") else "skip" if role == "K" else "revert" for role in actions}
        assert p["locked"] == sorted(roles[r] for r in ("K", "KH") if r in roles)


def test_a_comment_code_that_breaks_the_convention_is_refused() -> None:
    instance = _gen("H1", 0)
    task = instance.task
    b = _by_role(task)["B"]  # B 有兩項：comment 必須記 Y5
    expected = tuple({**p, "description": "Y2"} if p["type"] == "comment_added" and p["issue_id"] == b else p
                     for p in task.expected_outcome)
    tampered = dataclasses.replace(instance, task=dataclasses.replace(task, expected_outcome=expected))
    with pytest.raises(GeneratorError, match="comment code"):
        g3.check_instance(tampered)


def _tamper(instance: g3.GeneratedInstanceV3, *, entries=None, issues=None, alternatives=None) -> g3.GeneratedInstanceV3:
    task = instance.task
    params = {**task.params, "entries": entries} if entries is not None else task.params
    state = {**task.initial_state, "issues": issues} if issues is not None else task.initial_state
    return dataclasses.replace(instance, task=dataclasses.replace(task, params=params, initial_state=state),
                               alternative_calls=alternatives if alternatives is not None else instance.alternative_calls)


def _with_entries(instance: g3.GeneratedInstanceV3, role: str, change) -> g3.GeneratedInstanceV3:
    """把 role 的列出項目交給 change 修改（只改 params.entries；判斷與評分不變，只有角色的檢查會發現）。"""
    iid = _by_role(instance.task)[role]
    own = [copy.deepcopy(e) for e in instance.task.params["entries"] if e["issue_id"] == iid]
    change(own)
    return _tamper(instance, entries=[e for e in instance.task.params["entries"] if e["issue_id"] != iid] + own)


def _set(i: int, **fields):
    def change(own: list[dict]) -> None:
        own[i].update(fields)
    return change


def _label_to_priority(own: list[dict]) -> None:
    for e in own:
        if e["kind"].startswith("label"):
            e.update(kind="priority", set="p0", was="p1")
            e.pop("label")


# 每個修改都避開既有的檢查（欄位重複、已處理的 status 項、status 的值），只違反角色的項目表
@pytest.mark.parametrize(("level", "role", "change"), [
    ("H1", "B", lambda own: [e.update(handled=False) for e in own]),  # B 的兩項都未處理：B 不再測逐項的 Y4
    ("H1", "C", _set(0, handled=False)),
    ("H1", "A", _set(-1, handled=True)),  # A 的 label 項（同一份報告，label 排在最後）
    ("H1", "D", lambda own: own.clear()),
    ("H2", "S", _set(0, kind="priority", set="p0", was="p1")),
    ("H2", "K", _set(0, kind="status", set="in_progress", was="open")),
    ("H3", "Q", _set(0, kind="label_added", label="docs")),
])
def test_entries_that_break_the_role_table_are_refused(level: str, role: str, change) -> None:
    with pytest.raises(GeneratorError, match="entries of a role"):
        g3.check_instance(_with_entries(_gen(level, 0), role, change))


def test_h1_without_a_label_change_on_d_or_e_is_refused() -> None:
    tampered = _with_entries(_with_entries(_gen("H1", 0), "D", _label_to_priority), "E", _label_to_priority)
    with pytest.raises(GeneratorError, match="entries of a role"):
        g3.check_instance(tampered)


def test_a_report_dated_before_a_listed_issue_is_refused() -> None:
    instance = _gen("H2", 0)
    rid = instance.task.params["reports"][0]
    issues = [{**i, "created_at": "2026-07-01"} if i["id"] == rid else i for i in instance.task.initial_state["issues"]]
    with pytest.raises(GeneratorError, match="dated"):
        g3.check_instance(_tamper(instance, issues=issues))


def test_a_read_each_that_skips_a_listed_issue_is_refused() -> None:
    instance = _gen("H3", 0)
    reads = len(instance.task.params["reports"]) + 1
    calls = list(dict(instance.alternative_calls)["read-each"])
    calls[reads] = calls[reads + 1]  # 同一個 issue 讀兩次、少讀一個：長度不變
    alternatives = tuple((name, tuple(calls) if name == "read-each" else c) for name, c in instance.alternative_calls)
    with pytest.raises(GeneratorError, match="read-each"):
        g3.check_instance(_tamper(instance, alternatives=alternatives))


def test_generation_is_deterministic() -> None:
    assert g3.generate(FAMILY, "H3", 0) == _gen("H3", 0)
    assert g3.rng_fingerprint(FAMILY, "H2", 1) == g3.rng_fingerprint(FAMILY, "H2", 1)
    assert _gen("H1", 0).task.initial_state != _gen("H1", 1).task.initial_state
