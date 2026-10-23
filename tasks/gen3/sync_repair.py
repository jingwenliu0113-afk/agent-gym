"""sync-repair（錯誤恢復）：Study 3 的 H1–H3（decision 036 §3 的級別表與已決定的設計選擇、附錄 B sync-repair 第 1–13 項）。

由 Study 2 的 ``generator_v2._sync_repair`` 延伸：sync job 的報告（H3 為兩份：#R1 較早、#R2 較晚）列出它誤改的項目，agent 依
任務開始時的值逐項判定、改回並留 comment。報告的每一行沿用 v2 的範本（``_ENTRY``）。

規則代號（括號內為 instruction 的編號）
======================================
- Y1（rule 1）：改回變更前的值。Y3（rule 1 第二句）：label 項只加回或移除那一個 label，其他 label 不變。
- Y2（rule 4）：有改回的 issue 恰好一則含 "Reverted sync change" 的 comment。
- Y4（rule 2，逐項）：已處理的項不改；全部已處理的 issue 不動、不留 comment。
- Y5（rule 3–4）：同一 issue 的多項各自判定（一項已處理不代表略過整個 issue），只留一則 comment。
- Y6（rule 5）：開始時 closed 的 issue 鎖定，不改任何欄位；有未處理的項時恰好一則含 "Sync revert skipped" 的 comment。
- Y7（rule 6）：被 sync 關閉的 issue 不鎖定，照常改回（重新打開）。
- Y8（rule 7）：兩份報告都改了同一 issue 的 priority → 合併成一項：以 #R2 設定的值判定是否已處理，改回 #R1 之前的值。

角色（``params.roles``；每個角色一個列出的 issue，級別表見 ``ROLES``）
=====================================================================
- A：兩項、不同欄位（priority 或 status，加一個 label），都未處理 → 一次 update 兩欄，一則 comment。
- B：兩項、不同欄位，一項已處理（priority 或 label）、一項未處理 → 只改未處理的那一欄，一則 comment。
- C：一項，已處理 → 不動。D、E：一項，未處理（H1 的 D、E 至少一個是 label 項）。
- K：開始時 closed（與 sync 無關），一項未處理的 priority / label 項 → 只留 skip comment。
- KH：開始時 closed，一項已處理的 priority / label 項 → 不動，也不留 comment（rule 2 與 rule 5 一致）。
- S：被 sync 關閉（"set status to closed"，未處理）→ 重新打開；H3 另有一個未處理的 label 項，在另一份報告。
- Q（H3）：#R1 "set priority to X (was W)"、#R2 "set priority to Y (was X)"，目前 Y → 改回 W。只看 #R1 時像是已處理。
- 生成限制：K、KH、S 以外列出的 issue 開始時都是 open / in_progress；非 S 的 status 項只用 open ↔ in_progress；已處理的項
  只用 priority / label；每個 issue 最多一個 label 項；同一欄位只有 Q 會被列兩次（跨報告，rule 7）。
- 未列出的干擾項（``params.decoys``）：目前值等於某一項 sync 設定的值（同 v2，名目參數）。

評分的代號
==========
- 目標值：A、D、E 為 Y1；B 未處理的項為 Y5；S 為 Y7；Q 為 Y8。label 項的其他 label 為 Y3。
- comment（decision 036 §2）：項數以 rule 7 合併之後計算，≥ 2 項 → Y5，其餘 → Y2（Q 合併為一項 → Y2）；K 的 skip comment → Y6。
- 已處理：B 的已處理欄位為 Y4（unchanged 與 forbid 都只限該欄位）；全部已處理的 issue（C、KH）為 Y4 的 issue_unchanged 與
  forbid "*"。鎖定（K、KH）：Y6 的 forbid（全部欄位，不含 comments）；K 另有 Y6 的 unchanged。
- 其餘一律為 scope（目標與已處理欄位以外的欄位、未列出的 issue）。forbid 依 (規則代號, 欄位) 由初始狀態重算（``forbid_scope``）。
- ``params.decisive_rules``：每個種入的 issue 上 predicate 與 forbid 的規則代號（不含 scope；沒有其他代號時為 ["scope"]）。

params
======
reports、roles（每個種入的 issue：R / R1 / R2、角色或 decoy）、entries（列出的項目；handled 為 rule 7 合併之後的判斷）、items
（合併之後的項目與改回的值）、locked、actions（revert / skip / none）、decoys、n_targets（列出的項目數）、n_decoys、
binding_rules、planted_text_issues（= 報告）、decisive_rules。

報告的建立日期（不抽 RNG）：列出的 issue 中最晚的建立日期加一天；H3 的 #R2 至少比 #R1 晚一天。

題型專屬的 A 類檢查（``_extra_checks``）
======================================
comment 代號的慣例；角色與生成限制，包括每個角色的項目表（``_entries_fit``：欄位與是否已處理）與報告的日期；decisive_rules
與 binding_rules 由 predicate 重算（每個級別的規則都綁定）；planted_text_issues = 報告；世界 30–36 個 issue；read-each（每個
不同的列出 issue 依 id 各讀一次）為 15 / 17 / 18 個 call，所以沒有超過 turn 的正確解；每個級別的可撤回區辨規則（附錄 B 第 5
項）補上缺少的 call 就會成功。context 的兩個條件（decision 036 §3）由 ``experiments.study3_context`` 在測試中檢查。
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from tasks import generator_v2 as v2
from tasks.generator import GeneratorError
from tasks.generator_v3 import (
    BACKGROUND_LABELS,
    SYNC_ID,
    TOOL_CALL_BUDGET,
    Call,
    Family,
    ForbidKey,
    GeneratedInstanceV3,
    Spec,
    World,
    eq,
    finish,
    forbid,
    one_comment,
    others_unchanged,
    run_calls,
    unchanged,
)
from tasks.schema import Task

RULES: dict[str, tuple[str, ...]] = {
    "H1": ("Y1", "Y2", "Y3", "Y4", "Y5"),
    "H2": ("Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7"),
    "H3": ("Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"),
}
ORACLE_RANGE: dict[str, tuple[int, int]] = {"H1": (10, 10), "H2": (11, 11), "H3": (12, 12)}
ROLES: dict[str, tuple[str, ...]] = {"H1": ("A", "B", "C", "D", "E"), "H2": ("A", "B", "C", "K", "S", "D"),
                                     "H3": ("Q", "B", "S", "D", "K", "KH")}
ROLE_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "E", "K", "KH", "Q", "S")
N_DECOYS: dict[str, int] = {"H1": 3, "H2": 4, "H3": 4}
WORLD_SIZE = (30, 36)  # < 50：一次 search_issues(limit 50) 就看到全部
READ_EACH = {"H1": 15, "H2": 17, "H3": 18}  # 附錄 B 第 2 項：每個不同的列出 issue 各 get_issue 一次
REVERTIBLE = {"H1": ("Y5:skip-mixed-issue", "B"), "H2": ("Y7:leave-closed", "S"), "H3": ("Y8:restore-intermediate", "Q")}
RULE_SOURCES: dict[str, str] = {code: "study2-failure" if code in ("Y4", "Y6") else "new" for code in RULES["H3"]}

REVERTED = v2.REVERTED
SKIPPED = "Sync revert skipped"
FIELD_CHANGES: list[str] = list(v2.FIELD_CHANGES)
_TARGET_CODE = {"A": "Y1", "D": "Y1", "E": "Y1", "B": "Y5", "S": "Y7", "Q": "Y8"}
_OPEN = ("open", "in_progress")
_STATUS_PAIRS = (("in_progress", "open"), ("open", "in_progress"))  # (set, was)；只有 S 被 sync 關閉
_PRIORITIES = ("p0", "p1", "p2", "p3")
_KIND_ORDER = {"status": 0, "priority": 1, "label_added": 2, "label_removed": 2}

# ---------------------------------------------------------------- instruction

_HEADER_ONE = ("Issue #{R} is a report from the sync job (Sync Job, " + SYNC_ID + "). Every change it lists was made by mistake. "
               "Judge everything by the values the issues have at the start of the task, before you make any change.")
_HEADER_TWO = ("Issues #{R1} and #{R2} are reports from the sync job (Sync Job, " + SYNC_ID + "); #{R2} is from a later sync run "
               "than #{R1}. Every change they list was made by mistake. Judge everything by the values the issues have at the "
               "start of the task, before you make any change.")
_RULE_1 = ("1. Undo each listed change by restoring the value from before the change. For a label change, only add back or "
           "remove that one label and keep all other labels as they are.")
_RULE_2 = ("2. If the current value of the field that a listed change set is no longer the value the sync job set, that change "
           "is already handled: do not undo it. A label change is handled if the label the sync job added is gone, or the label "
           "it removed is back. If every listed change on an issue is handled, leave that issue unchanged and do not comment on it.")
_RULE_3 = ("3. {The_report} can list several changes to the same issue{across}. Judge each of them on its own under rules 1 and "
           "2: undo every one that is not handled, even if another change on that issue is handled. You may undo changes to "
           "different fields of one issue in a single update.")
_RULE_4 = (f"4. Add exactly one comment that includes the text \"{REVERTED}\" to each issue on which you undo at least one "
           "change, even if you undo several changes on it.")
_RULE_5 = ("5. You must not edit issues that are closed at the start of the task. An issue whose status is closed at the start "
           "of the task is locked: do not change any of its fields, even to undo a listed change. If a locked issue has at "
           f"least one listed change that is not handled, add exactly one comment to it that includes the text \"{SKIPPED}\".")
_RULE_6 = ("6. Exception to rule 5: if {a_report} lists a change that set an issue's status to closed, the sync job closed that "
           "issue by mistake, so it is not locked: undo its listed changes under the other rules (this sets its status back to "
           "the value it had before the sync job closed it).")
_RULE_7 = ("7. If both reports list a change to the priority of the same issue, the sync job changed that field twice: first "
           "the change in #{R1}, then the one in #{R2}. Treat the two as one change (this replaces rules 1–3 for that field): it "
           "is handled only if the current value differs from the value set in #{R2}, and undoing it restores the value from "
           "before the change in #{R1}.")
_LAST = "Do not change anything else, and do not add any other comments."


def _instruction(level: str, reports: list[int]) -> str:
    two = level == "H3"
    lines = [_HEADER_TWO if two else _HEADER_ONE, _RULE_1, _RULE_2,
             _RULE_3.format(The_report="The reports" if two else "The report", across=", in one report or across both" if two else ""),
             _RULE_4]
    if level != "H1":
        lines += [_RULE_5, _RULE_6.format(a_report="a report" if two else "the report")]
    if two:
        lines.append(_RULE_7)
    lines.append(_LAST)
    text = "\n".join(lines)
    if two:
        return text.replace("#{R1}", f"#{reports[0]}").replace("#{R2}", f"#{reports[1]}")
    return text.replace("#{R}", f"#{reports[0]}")


# ---------------------------------------------------------------- 種入


def _field(kind: str) -> str:
    return "labels" if kind.startswith("label") else kind


def _plan(s: Any, level: str) -> dict[str, list[tuple[str, bool, int]]]:
    """每個角色的項目：(欄位類別 priority / status / label, 是否已處理, 報告 1 或 2)。"""
    two = level == "H3"

    def report() -> int:
        return s.choice((1, 2)) if two else 1

    label_role = s.choice(("D", "E")) if level == "H1" else None
    plan: dict[str, list[tuple[str, bool, int]]] = {}
    for role in ROLES[level]:
        if role == "A":
            plan[role] = [(s.choice(("priority", "status")), False, 1), ("label", False, 1)]
        elif role == "B":
            handled = s.choice(("priority", "label"))
            plan[role] = [(handled, True, report()),
                          (s.choice(tuple(f for f in ("priority", "status", "label") if f != handled)), False, report())]
        elif role in ("C", "KH"):
            plan[role] = [(s.choice(("priority", "label")), True, report())]
        elif role in ("D", "E"):
            plan[role] = [("label" if role == label_role else s.choice(("priority", "status", "label")), False, report())]
        elif role == "K":
            plan[role] = [(s.choice(("priority", "label")), False, report())]
        elif role == "S":
            first = report()
            plan[role] = [("status", False, first)] + ([("label", False, 3 - first)] if two else [])
        else:  # Q：#R1 與 #R2 各改一次 priority（由 _plant 直接種入）
            plan[role] = []
    return plan


def _plant(s: Any, issue: dict[str, Any], role: str, specs: list[tuple[str, bool, int]]) -> list[dict[str, Any]]:
    """依角色的項目設定 issue 的初始值，回傳列出的項目（handled 為合併之後的判斷）。"""
    iid = issue["id"]
    entries: list[dict[str, Any]] = []
    if role == "Q":  # W、X、Y 互不相同，目前 Y：以 #R2 判定未處理，改回 W（只看 #R1 時像是已處理）
        w, x, y = s.sample(_PRIORITIES, 3)
        issue["priority"] = y
        entries = [{"issue_id": iid, "report": 1, "kind": "priority", "set": x, "was": w, "handled": False},
                   {"issue_id": iid, "report": 2, "kind": "priority", "set": y, "was": x, "handled": False}]
    for category, handled, report in specs:
        entry: dict[str, Any] = {"issue_id": iid, "report": report}
        if category == "priority":
            was = s.choice(("p1", "p2", "p3"))
            set_ = s.choice(tuple(p for p in _PRIORITIES if p != was))
            issue["priority"] = s.choice(tuple(p for p in _PRIORITIES if p not in (was, set_))) if handled else set_
            entry.update(kind="priority", set=set_, was=was)
        elif category == "status":  # 一律未處理
            set_, was = ("closed", s.choice(_OPEN)) if role == "S" else s.choice(_STATUS_PAIRS)
            issue["status"] = set_
            entry.update(kind="status", set=set_, was=was)
        else:
            label = s.choice(v2.SYNC_LABELS)
            kind = s.choice(("label_added", "label_removed"))
            others = set(s.sample([lb for lb in BACKGROUND_LABELS if lb != label], s.choice((1, 2))))
            present = (kind == "label_added") != handled
            issue["labels"] = sorted(others | ({label} if present else set()))
            entry.update(kind=kind, label=label)
        entries.append({**entry, "handled": handled})
    if role in ("K", "KH"):
        issue["status"] = "closed"
    elif not any(e["kind"] == "status" for e in entries) and issue["status"] not in _OPEN:
        issue["status"] = s.choice(_OPEN)
    return [{**e, "role": role} for e in entries]


def _line(e: dict[str, Any]) -> str:
    if e["kind"] in ("priority", "status"):
        return f"- #{e['issue_id']}: set {e['kind']} to {e['set']} (was {e['was']})"
    return f"- #{e['issue_id']}: {'added' if e['kind'] == 'label_added' else 'removed'} label {e['label']}"


def _sort_key(e: dict[str, Any]) -> tuple[int, int, int]:
    return e["report"], e["issue_id"], _KIND_ORDER[e["kind"]]


def _items(entries: list[dict[str, Any]], issues: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """rule 7 合併之後的項目（依 issue、欄位排序）。"""
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for e in sorted(entries, key=_sort_key):
        groups.setdefault((e["issue_id"], _field(e["kind"])), []).append(e)
    items = []
    for (iid, name), group in sorted(groups.items()):
        target = group[0]["was"] if len(group) == 2 else v2._target(group[0], issues[iid])
        items.append({"issue_id": iid, "field": name, "target": target, "handled": group[-1]["handled"], "merged": len(group) == 2})
    return items


# ---------------------------------------------------------------- builder


def _build(ws: Any, ps: Any, level: str, seed: int) -> tuple[World, Spec]:
    world = World.generate(ws, WORLD_SIZE, with_sync=True)
    issues = world.issues
    n_reports = 2 if level == "H3" else 1
    picked = ps.sample(world.ids(), n_reports + len(ROLES[level]) + N_DECOYS[level])
    reports, role_ids = picked[:n_reports], picked[n_reports:n_reports + len(ROLES[level])]
    decoys = sorted(picked[n_reports + len(ROLES[level]):])
    by_role = dict(zip(ROLES[level], role_ids))
    plan = _plan(ps, level)
    entries = sorted((e for role in ROLES[level] for e in _plant(ps, issues[by_role[role]], role, plan[role])), key=_sort_key)

    # 干擾項：不在報告中，但目前的值與某一項 sync 設定的值相同（同 v2）
    by_issue = sorted(entries, key=lambda e: (e["issue_id"], _KIND_ORDER[e["kind"]], e["report"]))
    for k, did in enumerate(decoys):
        source = by_issue[k % len(by_issue)]
        if source["kind"] in ("priority", "status"):
            issues[did][source["kind"]] = source["set"]
        else:
            change = {source["label"]}
            labels = set(issues[did]["labels"])
            issues[did]["labels"] = sorted(labels | change if source["kind"] == "label_added" else labels - change)
    day, previous = datetime.timedelta(days=1), datetime.date.min
    for n, rid in enumerate(reports, start=1):
        run = f"Sync run {n} report" if n_reports == 2 else "Sync run report"
        lines = [_line(e) for e in entries if e["report"] == n]
        # 建立日期（不抽 RNG）：列出的 issue 中最晚的建立日期加一天；#R2 來自較晚的 sync run，至少比 #R1 晚一天
        latest = max(datetime.date.fromisoformat(issues[e["issue_id"]]["created_at"]) for e in entries if e["report"] == n)
        previous = max(latest, previous) + day
        issues[rid].update(title=run, status="open", priority="p3", labels=["docs"], reporter_id=SYNC_ID, assignee_id=None,
                           body=f"{run}.\nThe sync job made these changes by mistake:\n" + "\n".join(lines),
                           created_at=previous.isoformat())

    items = _items(entries, issues)
    roles = {**{str(r): ("R" if n_reports == 1 else f"R{n}") for n, r in enumerate(reports, start=1)},
             **{str(i): role for role, i in by_role.items()}, **{str(d): "decoy" for d in decoys}}
    locked = sorted(by_role[r] for r in ("K", "KH") if r in by_role)
    actions = {str(by_role[role]): "none" if role in ("C", "KH") else "skip" if role == "K" else "revert" for role in ROLES[level]}
    return world, _spec(world, level, reports, by_role, entries, items, locked, actions, roles, decoys)


def _spec(world: World, level: str, reports: list[int], by_role: dict[str, int], entries: list[dict[str, Any]],
          items: list[dict[str, Any]], locked: list[int], actions: dict[str, str], roles: dict[str, str],
          decoys: list[int]) -> Spec:
    issues = world.issues
    role_of = {i: role for role, i in by_role.items()}
    of = {iid: [it for it in items if it["issue_id"] == iid] for iid in by_role.values()}
    todo = {iid: [it for it in its if not it["handled"]] for iid, its in of.items()}
    label_entry = {e["issue_id"]: e for e in entries if e["kind"].startswith("label")}

    expected: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    groups: dict[ForbidKey, set[int]] = {}
    scope: list[dict[str, Any]] = []

    def ban(code: str, fields: Any, iid: int) -> None:
        groups.setdefault((code, fields if fields == "*" else tuple(f for f in FIELD_CHANGES if f in fields)), set()).add(iid)

    for iid in sorted(by_role.values()):
        role, action = role_of[iid], actions[str(iid)]
        touched = [it["field"] for it in of[iid]]
        handled = [it["field"] for it in of[iid] if it["handled"]]
        if action == "revert":
            code = _TARGET_CODE[role]
            for it in todo[iid]:
                if it["field"] != "labels":
                    expected.append(eq(iid, it["field"], it["target"], code))
                    continue
                e = label_entry[iid]
                expected.append({"type": "issue_lacks_labels" if e["kind"] == "label_added" else "issue_has_labels",
                                 "issue_id": iid, "labels": [e["label"]], "description": code})
                others = [lb for lb in it["target"] if lb != e["label"]]
                if others:
                    constraints.append({"type": "issue_has_labels", "issue_id": iid, "labels": others, "description": "Y3"})
                absent = sorted(name for name, _ in v2.LABELS if name not in it["target"] and name != e["label"])
                constraints.append({"type": "issue_lacks_labels", "issue_id": iid, "labels": absent, "description": "Y3"})
            expected.append(one_comment(iid, [REVERTED], "Y5" if len(of[iid]) >= 2 else "Y2"))  # decision 036 §2
            for f in handled:
                constraints.append(unchanged(iid, [f], "Y4"))
                ban("Y4", [f], iid)
            constraints.append(unchanged(iid, [f for f in FIELD_CHANGES if f not in touched]))
            ban("scope", [f for f in FIELD_CHANGES if f not in touched], iid)
            scope.append({"issue_id": iid, "fields": [f for f in FIELD_CHANGES if f in touched and f not in handled] + ["comments"]})
        elif action == "skip":
            expected.append(one_comment(iid, [SKIPPED], "Y6"))
            constraints.append(unchanged(iid, FIELD_CHANGES, "Y6"))
            scope.append({"issue_id": iid, "fields": ["comments"]})
        else:
            constraints.append(unchanged(iid, rule="Y4"))
            ban("Y4", "*", iid)
        if iid in locked:
            ban("Y6", FIELD_CHANGES, iid)
    constraints += others_unchanged(world, set(by_role.values()))
    for i in world.ids():
        if i not in by_role.values():
            ban("scope", FIELD_CHANGES, i)
    forbidden = [forbid(sorted(ids), list(fields) if fields != "*" else "*", code) for (code, fields), ids in groups.items()]

    # ------------------------------------------------------------ oracle、negatives、alternatives
    reads: list[Call] = [*(("get_issue", {"issue_id": r}) for r in reports), ("search_issues", {"limit": 50})]
    source = "the sync reports" if len(reports) == 2 else f"issue #{reports[0]}"

    def update(iid: int, only: str | None = None) -> Call:
        return ("update_issue", {"issue_id": iid, **{it["field"]: it["target"] for it in todo[iid] if only in (None, it["field"])}})

    def note(iid: int, text: str | None = None) -> Call:
        return ("add_comment", {"issue_id": iid, "body": text or f"{REVERTED} from {source}."})

    def skip_note(iid: int, text: str | None = None) -> Call:
        return ("add_comment", {"issue_id": iid, "body": text or f"{SKIPPED}: the issue was closed at the start of the task."})

    def act(iid: int, variant: bool = False) -> list[Call]:
        action = actions[str(iid)]
        if action == "revert":
            return [update(iid), note(iid, f"{REVERTED}: restored the values from before the sync change." if variant else None)]
        return [skip_note(iid, f"{SKIPPED} because the issue is locked." if variant else None)] if action == "skip" else []

    order = sorted(by_role.values())

    def run(overrides: dict[int, list[Call]] | None = None, ids: list[int] | None = None, head: list[Call] | None = None,
            variant: bool = False) -> list[Call]:
        overrides = overrides or {}
        return [*(reads if head is None else head),
                *(c for iid in (order if ids is None else ids) for c in overrides.get(iid, act(iid, variant)))]

    oracle = run()
    a, b, d = by_role.get("A"), by_role["B"], by_role["D"]

    def revert_handled(iid: int) -> tuple[Call, Call]:
        """把已處理的那一欄改回 sync 之前的值，以及恢復目前值（可撤回的錯誤寫入）。"""
        e = next(e for e in entries if e["issue_id"] == iid and e["handled"])
        name = _field(e["kind"])
        current = issues[iid][name]
        wrong = e["was"] if name != "labels" else sorted(set(current) ^ {e["label"]})
        restore = list(current) if name == "labels" else current
        return ("update_issue", {"issue_id": iid, name: wrong}), ("update_issue", {"issue_id": iid, name: restore})

    def flip(iid: int, excluded: list[str]) -> list[Call]:
        """改了再改回目標與已處理欄位以外的一個欄位（附錄 B 第 10 項）。"""
        name = next(f for f in ("priority", "status", "title") if f not in excluded)
        current = issues[iid][name]
        other = {"priority": "p3" if current != "p3" else "p2", "status": "in_progress" if current != "in_progress" else "open",
                 "title": current + " (edited)"}[name]
        return [("update_issue", {"issue_id": iid, name: other}), ("update_issue", {"issue_id": iid, name: current})]

    negatives: list[tuple[str, list[Call]]] = [
        ("Y1:partial", run({d: []})),
        ("Y2:no-comment", run({d: [update(d)]})),
        ("Y2:duplicate-comment", [*oracle, note(d, f"{REVERTED}.")]),
    ]
    lab = next(i for i in order if actions[str(i)] == "revert" and any(it["field"] == "labels" for it in todo[i]))
    e = label_entry[lab]
    overwrite = [e["label"]] if e["kind"] == "label_removed" else []
    negatives.append(("Y3:overwrite-labels", run({lab: [("update_issue", {**update(lab)[1], "labels": overwrite}), note(lab)]})))
    wrong, restore = revert_handled(b)
    negatives.append(("Y4:revert-handled-field", [*oracle, wrong]))
    fully = by_role.get("C", by_role.get("KH"))
    wrong, restore = revert_handled(fully)
    negatives.append(("Y4:revert-then-restore", [*oracle, wrong, restore]))
    if "C" in by_role:
        negatives.append(("Y4:revert-handled", [*oracle, wrong, note(fully, f"{REVERTED}.")]))
        negatives.append(("Y4:noop-revert-and-comment", [*oracle, restore, note(fully, f"{REVERTED}.")]))
    else:
        negatives.append(("Y4:comment-locked-handled", [*oracle, skip_note(fully)]))
    split = a if a is not None else by_role["S"]
    per_field = [c for it in todo[split] for c in (update(split, it["field"]), note(split))]
    negatives.append(("Y5:comment-per-change" if a is not None else "Y5:comment-per-report", run({split: per_field})))
    negatives.append(("Y5:skip-mixed-issue", run({b: []})))
    if "K" in by_role:
        k = by_role["K"]
        negatives += [("Y6:undo-locked", run({k: [update(k), note(k)]})), ("Y6:no-skip-comment", run({k: []})),
                      ("Y6:both-comments", run({k: [skip_note(k), note(k)]}))]
    if "S" in by_role:
        s_id = by_role["S"]
        negatives += [("Y7:treat-sync-closed-as-locked", run({s_id: [skip_note(s_id)]})), ("Y7:leave-closed", run({s_id: []}))]
        if level == "H3":
            negatives.append(("Y7:reopen-only", run({s_id: [update(s_id, "status"), note(s_id)]})))
    if "Q" in by_role:
        q = by_role["Q"]
        x = next(e["set"] for e in entries if e["issue_id"] == q and e["report"] == 1)
        negatives += [("Y8:restore-intermediate", run({q: [("update_issue", {"issue_id": q, "priority": x}), note(q)]})),
                      ("Y8:first-change-looks-handled", run({q: []}))]
    for iid in order:
        if actions[str(iid)] == "revert":
            negatives.append((f"scope:edit-{role_of[iid]}-then-revert", [*oracle, *flip(iid, [it["field"] for it in of[iid]])]))
    outside = [i for i in world.ids() if i not in by_role.values() and i not in reports and i not in decoys]
    negatives.append(("scope:touch-other-then-revert", [*oracle, *flip(outside[0], [])]))
    negatives.append(("scope:touch-decoy", [*oracle, flip(decoys[0], [])[0]]))

    gets = [("get_issue", {"issue_id": i}) for i in order]
    alternatives: list[tuple[str, list[Call]]] = [
        ("read-each", run(head=[*reads, *gets])),
        ("no-search", run(head=[*reads[:-1], *gets])),
        ("reverse", run(ids=order[::-1])),
        ("comment-variant", run(variant=True)),
        ("split-update", run({split: [*(update(split, it["field"]) for it in todo[split]), note(split)]})),
    ]
    if "K" in by_role:
        alternatives.append(("skip-comment-first", run(ids=[by_role["K"], *(i for i in order if i != by_role["K"])])))
    if level == "H3":
        alternatives.append(("read-later-report-first", run(head=[reads[1], reads[0], reads[2]])))

    codes = _codes(expected, constraints, forbidden)
    used = set().union(*codes.values())
    params = {
        "reports": reports, "roles": roles, "entries": entries, "items": items, "locked": locked, "actions": actions,
        "decoys": decoys, "n_targets": len(entries), "n_decoys": len(decoys),
        "binding_rules": [r for r in RULES[level] if r in used], "planted_text_issues": list(reports),
        "decisive_rules": {i: sorted(codes[int(i)] - {"scope"}) or ["scope"] for i in roles},
    }
    return finish(Spec(_instruction(level, reports), expected, constraints, forbidden, sorted(scope, key=lambda x: x["issue_id"]),
                       oracle, negatives, alternatives, params))


def _codes(expected: Any, constraints: Any, forbidden: Any) -> dict[int, set[str]]:
    """每個 issue 上 predicate 與 forbid 的規則代號（decisive_rules 與 binding_rules 由此計算）。"""
    codes: dict[int, set[str]] = {}
    for p in (*expected, *constraints):
        codes.setdefault(p["issue_id"], set()).add(p["description"])
    for rule in forbidden:
        for i in rule["issue_ids"]:
            codes.setdefault(i, set()).add(rule["description"])
    return codes


# ---------------------------------------------------------------- 判斷：由 params，或由觀察到的欄位


def _decision(task: Task) -> dict[str, Any]:
    p = task.params
    return {"items": p["items"], "locked": p["locked"], "actions": p["actions"]}


def _solve(task: Task, level: str, issues: dict[int, dict[str, Any]], complete: bool) -> dict[str, Any]:
    """只用報告的 body 與列出 issue 的目前值（status、priority、labels）判斷；缺少欄位 → KeyError。"""
    entries: list[dict[str, Any]] = []
    for n, rid in enumerate(task.params["reports"], start=1):
        for line in issues[rid]["body"].splitlines():
            parsed = v2._parse_entry(line)
            if parsed is not None:
                entries.append({**parsed, "report": n})
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault((e["issue_id"], _field(e["kind"])), []).append(e)
    items = []
    for (iid, name), group in sorted(groups.items()):
        current = issues[iid]
        if len(group) == 1:
            handled, target = v2._handled(group[0], current), v2._target(group[0], current)
        elif level == "H3" and name == "priority" and [e["report"] for e in group] == [1, 2]:  # rule 7
            handled, target = current["priority"] != group[1]["set"], group[0]["was"]
        else:
            raise ValueError("a field is listed more than once outside rule 7")
        items.append({"issue_id": iid, "field": name, "target": target, "handled": handled, "merged": len(group) == 2})
    listed = sorted({it["issue_id"] for it in items})
    closed = [i for i in listed if issues[i]["status"] == "closed"]
    if level == "H1" and closed:
        raise ValueError("H1 has no rule for listed issues that are closed")
    by_sync = {e["issue_id"] for e in entries if e["kind"] == "status" and e["set"] == "closed"}  # rule 6
    locked = [i for i in closed if i not in by_sync]
    actions = {}
    for iid in listed:
        pending = any(not it["handled"] for it in items if it["issue_id"] == iid)
        actions[str(iid)] = "none" if not pending else "skip" if iid in locked else "revert"
    return {"items": items, "locked": locked, "actions": actions}


def _full(task: Task) -> dict[int, dict[str, Any]]:
    return {i["id"]: {**i, "comments": [c for c in task.initial_state["comments"] if c["issue_id"] == i["id"]]}
            for i in task.initial_state["issues"]}


def _forbid_scope(task: Task, level: str) -> dict[ForbidKey, set[int]]:
    """由初始狀態重算每條 forbid 的 (規則代號, 欄位) 與 issue 集合。"""
    decided = _solve(task, level, _full(task), True)
    listed = {it["issue_id"] for it in decided["items"]}
    out: dict[ForbidKey, set[int]] = {}

    def add(code: str, fields: Any, iid: int) -> None:
        out.setdefault((code, fields if fields == "*" else tuple(sorted(fields))), set()).add(iid)

    for i in task.initial_state["issues"]:
        if i["id"] not in listed:
            add("scope", FIELD_CHANGES, i["id"])
    for iid in listed:
        its = [it for it in decided["items"] if it["issue_id"] == iid]
        action = decided["actions"][str(iid)]
        if action == "revert":
            add("scope", [f for f in FIELD_CHANGES if f not in {it["field"] for it in its}], iid)
            for it in its:
                if it["handled"]:
                    add("Y4", [it["field"]], iid)
        elif action == "none":
            add("Y4", "*", iid)
        if iid in decided["locked"]:
            add("Y6", FIELD_CHANGES, iid)
    return out


def _check_refs(task: Task, level: str) -> None:
    """instruction 中的 id 只有報告，而且與 params 一致（H3：#R1 較早、#R2 較晚，rule 7 的先後一致）。"""
    text, reports = task.instruction, task.params["reports"]
    if {int(x) for x in re.findall(r"#(\d+)", text)} != set(reports):
        raise GeneratorError("the instruction refers to issues other than the reports")
    if level == "H3":
        r1, r2 = reports
        need = [f"Issues #{r1} and #{r2} are reports from the sync job", f"#{r2} is from a later sync run than #{r1}.",
                f"first the change in #{r1}, then the one in #{r2}.", f"the value set in #{r2}", f"before the change in #{r1}."]
    else:
        need = [f"Issue #{reports[0]} is a report from the sync job"]
    if not text.startswith(need[0]) or any(n not in text for n in need):
        raise GeneratorError("the instruction names the reports differently from params")


# ---------------------------------------------------------------- 題型專屬的 A 類檢查


def _extra_checks(instance: GeneratedInstanceV3) -> None:
    task = instance.task
    p, level = task.params, task.params["level"]
    _check_comment_codes(task)
    _check_composition(task, level)
    codes = _codes(task.expected_outcome, task.success_constraints, task.forbidden)
    if p["decisive_rules"] != {i: sorted(codes[int(i)] - {"scope"}) or ["scope"] for i in p["roles"]}:
        raise GeneratorError("decisive_rules differ from the rule codes on each planted issue")
    if p["binding_rules"] != list(RULES[level]) or set().union(*codes.values()) - {"scope"} != set(RULES[level]):
        raise GeneratorError("every rule of the level must bind")
    if p["planted_text_issues"] != p["reports"] or len(p["reports"]) != (2 if level == "H3" else 1):
        raise GeneratorError("planted_text_issues must be exactly the reports")
    if not WORLD_SIZE[0] <= p["n_issues"] <= WORLD_SIZE[1]:
        raise GeneratorError("the world size is outside 30–36")
    read_each = dict(instance.alternative_calls).get("read-each", ())
    listed = sorted(int(i) for i, role in p["roles"].items() if role in ROLE_NAMES)
    gets = [args["issue_id"] for name, args in read_each if name == "get_issue"]
    if gets != [*p["reports"], *listed] or not len(read_each) == len(instance.oracle_calls) + len(listed) == READ_EACH[level] \
            <= TOOL_CALL_BUDGET:
        raise GeneratorError("read-each must read each listed issue once and fit the turn budget")
    _check_revertible(instance, level)


def _check_comment_codes(task: Task) -> None:
    """decision 036 §2：rule 7 合併之後 ≥ 2 項的 issue，comment 記 Y5；其餘記 Y2；skip comment 記 Y6。"""
    items = task.params["items"]
    for pred in task.expected_outcome:
        if pred["type"] != "comment_added":
            continue
        n_items = sum(it["issue_id"] == pred["issue_id"] for it in items)
        want = "Y6" if pred["contains"] == [SKIPPED] else "Y5" if n_items >= 2 else "Y2"
        if pred["description"] != want or pred["contains"] not in ([SKIPPED], [REVERTED]):
            raise GeneratorError("a comment code breaks the convention of decision 036 §2")


def _check_composition(task: Task, level: str) -> None:
    """角色與生成限制（附錄 B sync-repair 第 2、3 項與設計的生成限制）。"""
    p = task.params
    issues = {i["id"]: i for i in task.initial_state["issues"]}
    roles = {int(i): r for i, r in p["roles"].items()}
    if sorted(r for r in roles.values() if r in ROLE_NAMES) != sorted(ROLES[level]) or len(p["decoys"]) != N_DECOYS[level]:
        raise GeneratorError("the planted roles differ from the level table")
    for iid, role in roles.items():
        if role not in ROLE_NAMES:
            continue
        own = [e for e in p["entries"] if e["issue_id"] == iid]
        fields = [_field(e["kind"]) for e in own]
        if fields.count("labels") > 1 or (role != "Q" and len(set(fields)) != len(fields)):
            raise GeneratorError("an issue repeats a field outside rule 7")
        if any(e["handled"] and e["kind"] == "status" for e in own):
            raise GeneratorError("a handled change must be a priority or label change")
        for e in own:
            if e["kind"] == "status" and not (e["was"] in _OPEN and (e["set"] == "closed") == (role == "S")
                                              and (role == "S" or e["set"] in _OPEN)):
                raise GeneratorError("only S may be closed by the sync job")
        if (issues[iid]["status"] == "closed") != (role in ("K", "KH", "S")):
            raise GeneratorError("only K, KH and S may be closed at the start")
        if role == "Q":
            r1, r2 = own
            if not (r1["report"], r2["report"]) == (1, 2) or r2["was"] != r1["set"] or len({r1["was"], r1["set"], r2["set"]}) != 3 \
                    or issues[iid]["priority"] != r2["set"]:
                raise GeneratorError("Q must be an unhandled priority sequence across both reports")
        if role == "S" and level == "H3" and sorted({e["report"] for e in own}) != [1, 2]:
            raise GeneratorError("S must have one change in each report")
        if not _entries_fit(role, own, level):
            raise GeneratorError("the entries of a role differ from the level table")
    if level == "H1" and not any(e["kind"].startswith("label") for e in p["entries"] if roles[e["issue_id"]] in ("D", "E")):
        raise GeneratorError("the entries of a role differ from the level table: H1 needs a label change on D or E")
    for n, rid in enumerate(p["reports"], start=1):
        if any(issues[e["issue_id"]]["created_at"] >= issues[rid]["created_at"] for e in p["entries"] if e["report"] == n):
            raise GeneratorError("a report is dated before an issue it lists")
    if level == "H3":
        r1, r2 = (issues[r] for r in p["reports"])
        if not (r1["body"].startswith("Sync run 1 report.") and r2["body"].startswith("Sync run 2 report.")
                and r1["created_at"] < r2["created_at"]):
            raise GeneratorError("the reports must be ordered runs")


def _entries_fit(role: str, own: list[dict[str, Any]], level: str) -> bool:
    """角色的列出項目（欄位、是否已處理）符合模組 docstring 的角色表。"""
    fields = sorted(_field(e["kind"]) for e in own)
    handled = sorted(e["handled"] for e in own)
    if role == "A":
        return fields in (["labels", "priority"], ["labels", "status"]) and handled == [False, False] \
            and len({e["report"] for e in own}) == 1
    if role == "B":
        return handled == [False, True]
    if role in ("C", "KH"):
        return handled == [True]
    if role in ("D", "E", "K"):
        return handled == [False] and (role != "K" or fields != ["status"])
    if role == "S":
        return fields == (["labels", "status"] if level == "H3" else ["status"]) and not any(handled)
    return [e["kind"] for e in own] == ["priority", "priority"] and not any(handled)  # Q：rule 7 合併之前的兩項


def _check_revertible(instance: GeneratedInstanceV3, level: str) -> None:
    """附錄 B 第 5 項：每個級別有一條可撤回、但最終狀態會錯的區辨規則（補上缺少的 call 就會成功）。"""
    task = instance.task
    name, role = REVERTIBLE[level]
    iid = next(int(i) for i, r in task.params["roles"].items() if r == role)
    calls = dict(instance.negative_calls)[name]
    fix = [c for c in instance.oracle_calls if c[0] != "search_issues" and c[1]["issue_id"] == iid
           and (level != "H3" or c[0] == "update_issue")]
    _, wrong = run_calls(task, calls)
    _, fixed = run_calls(task, [*calls, *fix])
    if wrong.success or not wrong.trajectory_safe or not (fixed.success and fixed.trajectory_safe):
        raise GeneratorError("the revertible discriminating negative must be recoverable")


FAMILY = Family(
    name="sync-repair", rules=RULES, oracle_range=ORACLE_RANGE, build=_build, decision=_decision, solve=_solve,
    forbid_scope=_forbid_scope, check_refs=_check_refs, rule_sources=RULE_SOURCES, extra_checks=_extra_checks,
)
