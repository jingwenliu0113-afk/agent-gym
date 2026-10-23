"""Study 3 的 epic-closeout（decision 036 §3、附錄 B epic-closeout 1–13）：release tracker #E、巢狀的子 tracker #C 與 wontfix 的例外鏈。

由 Study 2 的 ``generator_v2._epic_closeout``（L3）延伸。instruction 編號與規則代號（T = 該 sub-issue 所屬的 tracker，也就是列出它的
"Sub-issues:" 行所在的 issue）：

- rule 1 = E1（關閉 resolved 的 sub-issue；已 closed 的不動）與 E4（resolved + needs-info 視為未完成：不改 status，留 "Blocks #T"）；
- rule 2 = E3（open / in_progress 的 sub-issue 不關閉，留恰好一則含 "Blocks #T" 的 comment）；
- rule 3 = E2（#E 只在沒有被移出的 sub-issue 都關閉之後才關閉，否則 in_progress）；rule 3 用在 #C 時記為 E6；
- rule 4 = E5（wontfix 的 sub-issue 從所屬的 tracker 移出：不改 status，留 "Dropped from #T"；不算進 rule 3）；
- rule 5 = E6（H2 起：#C 依它自己的 sub-issue 關閉，或 in_progress 並留 "Blocks #E"；#C 只有被關閉才算 closed）；
- rule 6 = E7（H3：p0 的 wontfix 不能移出，依 rule 1–3 處理）。

級別（括號內為 oracle 長度）與角色（params.roles；open 代表 open 或 in_progress）。分支由 seed 的奇偶決定（偶數 done、奇數
blocked），同一個 seed 在 H1–H3 使用同一個分支（附錄 B 第 1 項）：

| 級別 | 分支 | #E 自己的 sub-issue | #C 的 sub-issue | n_targets |
|---|---|---|---|---|
| H1（9） | done | X, W-open, W-resolved, R×4 | — | 7 |
| H1 | blocked | X, N, U, W-resolved, R×3 | — | 7 |
| H2（11） | done | C, X, W-open, R×2 | R×2, W-resolved | 8 |
| H2 | blocked | C, X, W-open, R×2（#E 只因 #C 而 in_progress） | U, N, X | 8 |
| H3（13） | done | C, X, W-open, P-resolved, R, R-p0 | R×3, W-resolved | 10 |
| H3 | blocked | C, X, N, U, W-resolved | P-open（#C 唯一的阻擋）, R-p0, R, W-open | 9 |

- X closed；R resolved → 關閉（E1）；R-p0（H3）是 p0 但沒有 wontfix 的 R，讓 "p0" 本身不是線索；U open → "Blocks #T"（E3）；
  N resolved + needs-info → "Blocks #T"（E4）；W-*：wontfix（H3 一律非 p0）→ "Dropped from #T"（E5）；P-*（H3）：wontfix + p0 →
  依 status 處理（E7）。#E、#C 開始時都是 open。
- 干擾項（resolved、不在任何 Sub-issues 行上，必須不動；n_decoys 4 / 5 / 6）：decoy-follow-up-E（body 結尾 "Follow-up to #E."）、
  decoy-plain×2、decoy-wontfix（非 p0）；H2 起加 decoy-follow-up-C；H3 再加 decoy-wontfix-p0。
- 背景 issue 的 status 只有 open / in_progress / closed，所以 resolved 的 issue 一定是種入的 sub-issue 或干擾項。

附錄 B 與 §3 的處理：
- 拿掉 body 中 "Part of #E/#C" 的線索；#C 的標題不含 "tracking"；不加 status 摘要句；rule 4 寫 "dropped from its tracker"；rule 5
  依審查改寫，不會讀成無條件關閉（第 7、11 項）。
- wontfix 的組成不變式（§5 A，第 9 項）：種入的 issue 先以確定性的方式（不抽 RNG）移除 wontfix 與 needs-info，再只加到 W、P、
  wontfix 干擾項（wontfix）與 N（needs-info）；``extra_checks`` 以硬檢查確認。背景 issue 仍可能帶 wontfix，membership 只看
  Sub-issues 行，所以不影響角色。
- forbid 集合依 (規則代號, 欄位) 由初始狀態重算：E2 = {#E}、E3 = open 的非 wontfix sub-issue、E4 = N、E5 = W、E6 = {#C}、
  E7 = open 的 P、scope = 干擾項（第 10 項）。tracker 的 forbid（rule 3 的順序）連同 unless 清單由 ``extra_checks`` 從
  Sub-issues 行重算：done 時列出沒有被移出的 sub-issue（#E 的清單含 #C 與 X），blocked 時沒有 unless。
- negatives：E5:close-dropped 關閉任一個 W（有 resolved 的 W 就用它）；E3 / E5 的 wrong-tracker-text 帶 ``WRONG_TRACKER`` 標記
  （第 5、12 項；標記只用於 negative 的自我檢查，報告的 wrong-tracker 分類依 §2 的機械定義；``extra_checks`` 確認它只新增一則
  寫另一個 tracker id 的 comment，而且只因那則 comment 而失敗）。只錯在順序、最終狀態正確的 trajectory 是 RQ3 的事件，不列為
  negative（第 4 項；``extra_checks`` 評分每個 negative，拒絕成功、而且只違反帶 unless 的 forbid 的 negative）。
- H3 的 read-each（每個不同的 sub-issue 各 get_issue 一次）為 22（done）/ 21（blocked）個 call，列為超過 turn 的正確解（§3 (c)）。

decisive_rules（decision 036 §2 次要結果 2；歸屬標籤，不是原因）：角色要求的處理所屬的規則，加上該角色刻意測試的例外鏈（條件成立
但被例外推翻，或只有相近的線索）。例如 W-resolved 記 [E1, E5, E7]（resolved 本應關閉、wontfix 移出、H3 的 p0 例外不成立），
R-p0 記 [E1, E7]，wontfix 干擾項記 [scope, E5]。#E 另記決定它最終 status 的機制：done 分支的 E5（open 的 W 不阻擋 #E），以及
#C 決定 #E 的級別（H2 兩個分支與 H3 done）的 E6；H3 blocked 的 #E 直接被 N、U 阻擋，不記 E6（§3 的限制）。#C 在 H3 blocked
另記 E7（P 是唯一的阻擋）。

已知限制（寫進 decision 036）：E3 / E4 只在奇數 seed、E5 作為 #E 的決定因素與 E7 的 resolved P 只在偶數 seed 有作用；H3 奇數 seed
的 #C → #E 傳遞不決定結果；四個 comment 字串以子字串評分（例如同時含 "Blocks #T" 與 "Dropped from #T" 的 comment 會通過）；
只錯在順序的 trajectory 是 success（RQ3）；H3 的 read-each 超過 19 個 call。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tasks.generator import GeneratorError, _Sampler
from tasks.generator_v3 import (
    LEVELS,
    OTHER_FIELDS,
    OTHER_NO_COMMENTS,
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

NAME = "epic-closeout"
RULES: dict[str, tuple[str, ...]] = {
    "H1": ("E1", "E2", "E3", "E4", "E5"),
    "H2": ("E1", "E2", "E3", "E4", "E5", "E6"),
    "H3": ("E1", "E2", "E3", "E4", "E5", "E6", "E7"),
}
ORACLE_RANGE: dict[str, tuple[int, int]] = {"H1": (9, 9), "H2": (11, 11), "H3": (13, 13)}
# 規則來源（decision 036 附錄 A）：E2 是 034 記錄「加了 comment 就當成已關閉」這類失敗的代號（dev L2-s1、eval L3-s3），
# E6 是同類判斷下移到 #C；兩者標為 Study 2 的失敗類型，其餘為 new。
RULE_SOURCES = {"E1": "new", "E2": "study2-failure", "E3": "new", "E4": "new", "E5": "new", "E6": "study2-failure", "E7": "new"}
RULE_NUMBERS = {"1": ("E1", "E4"), "2": ("E3",), "3": ("E2", "E6"), "4": ("E5",), "5": ("E6",), "6": ("E7",)}  # 附錄 A 的對照

WORLD_SIZE = (22, 28)  # 同 L3；n ≤ 28 < 50，一次 search_issues(limit=50) 就看得到全部
SUB_LINE = "Sub-issues:"
BLOCKS = "Blocks #{t}"
DROPPED = "Dropped from #{t}"
WRONG_TRACKER = "wrong-tracker"  # wrong-tracker negative 的標記（附錄 B 第 12 項）

_OPEN = ("open", "in_progress")
_NOT_P0 = ("p1", "p2", "p3")
# (級別, 分支) → (#E 自己的 sub-issue，不含 #C；#C 的 sub-issue)
_SUBS: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {
    ("H1", "done"): (("X", "W-open", "W-resolved", "R", "R", "R", "R"), ()),
    ("H1", "blocked"): (("X", "N", "U", "W-resolved", "R", "R", "R"), ()),
    ("H2", "done"): (("X", "W-open", "R", "R"), ("R", "R", "W-resolved")),
    ("H2", "blocked"): (("X", "W-open", "R", "R"), ("U", "N", "X")),
    ("H3", "done"): (("X", "W-open", "P-resolved", "R", "R-p0"), ("R", "R", "R", "W-resolved")),
    ("H3", "blocked"): (("X", "N", "U", "W-resolved"), ("P-open", "R-p0", "R", "W-open")),
}
_DECOYS: dict[str, tuple[str, ...]] = {
    "H1": ("decoy-follow-up-E", "decoy-plain", "decoy-plain", "decoy-wontfix"),
    "H2": ("decoy-follow-up-E", "decoy-plain", "decoy-plain", "decoy-wontfix", "decoy-follow-up-C"),
    "H3": ("decoy-follow-up-E", "decoy-plain", "decoy-plain", "decoy-wontfix", "decoy-follow-up-C", "decoy-wontfix-p0"),
}
_CLOSE = {"R": "E1", "R-p0": "E1", "P-resolved": "E7"}  # 要關閉的角色 → 規則代號
_BLOCK = {"U": "E3", "N": "E4", "P-open": "E7"}  # 要留 "Blocks #T" 的角色 → 規則代號
_DROP = ("W-open", "W-resolved")
_WONTFIX = ("W-open", "W-resolved", "P-open", "P-resolved", "decoy-wontfix", "decoy-wontfix-p0")
_P0 = ("P-open", "P-resolved", "R-p0", "decoy-wontfix-p0")
_STATUS = {"X": "closed", "R": "resolved", "R-p0": "resolved", "N": "resolved", "W-resolved": "resolved",
           "P-resolved": "resolved"}
_DECISIVE = {
    "X": ("E1",), "R": ("E1",), "R-p0": ("E1", "E7"), "U": ("E3",), "N": ("E1", "E4"),
    "W-open": ("E3", "E5", "E7"), "W-resolved": ("E1", "E5", "E7"), "P-open": ("E3", "E5", "E7"),
    "P-resolved": ("E1", "E5", "E7"), "decoy-follow-up-E": ("scope",), "decoy-follow-up-C": ("scope",),
    "decoy-plain": ("scope",), "decoy-wontfix": ("scope", "E5"), "decoy-wontfix-p0": ("scope", "E5", "E7"),
}


# ---------------------------------------------------------------- instruction


def instruction(level: str, epic: Any, child: Any = None) -> str:
    """級別的 instruction；epic / child 為 id（模板以 "<E>"、"<C>" 代入，見 ``TEMPLATES``）。"""
    e, c = f"#{epic}", f"#{child}"
    rule1 = ("1. Close every sub-issue whose status is resolved (set its status to closed). A resolved sub-issue that has the "
             "needs-info label is not finished: do not change its status (do not close or reopen it), and treat it like an open "
             "sub-issue under rules 2 and 3. Sub-issues that are already closed need no action.\n")
    final = "Do not change anything else, and do not add any other comments."
    if level == "H1":
        return (
            f"Issue {e} tracks a release. The \"Sub-issues:\" line in the body of {e} lists its sub-issues; only those issues "
            "are sub-issues. Wrap up the release:\n"
            f"{rule1}"
            "2. Do not close sub-issues that are open or in_progress. Instead, add exactly one comment to each of them that "
            f"includes the text \"Blocks {e}\".\n"
            f"3. Close {e} only after all of its sub-issues that are not dropped (rule 4) are closed. If any of them cannot be "
            f"closed, set {e} to in_progress instead of closing it.\n"
            f"4. A sub-issue that has the wontfix label is dropped from its tracker, {e}, whatever its status (open, "
            "in_progress or resolved). Rules 1 and 2 do not apply to it: do not change its status, and add exactly one comment "
            f"to it that includes the text \"Dropped from {e}\".\n"
            f"{final}"
        )
    rule6 = ("6. Exception to rule 4: a wontfix sub-issue whose priority is p0 cannot be dropped. Rule 4 does not apply to it; "
             "handle it with rules 1, 2 and 3 like any other sub-issue.\n") if level == "H3" else ""
    return (
        f"Issue {e} tracks a release. The \"Sub-issues:\" line in the body of {e} lists the sub-issues of {e}. One of them, "
        f"{c}, is itself a tracker: the \"Sub-issues:\" line in the body of {c} lists the sub-issues of {c}. Only the issues "
        f"on a tracker's own \"Sub-issues:\" line are its sub-issues; the sub-issues of {c} are not sub-issues of {e}. Below, "
        "\"sub-issue\" means a sub-issue of either tracker, and \"its tracker\" is the tracker whose line lists it. Wrap up "
        "the release:\n"
        f"{rule1}"
        "2. Do not close sub-issues that are open or in_progress. Instead, add exactly one comment to each of them that "
        f"includes the text \"Blocks {c}\" if its tracker is {c}, or \"Blocks {e}\" if its tracker is {e}.\n"
        "3. Close a tracker only after all of its sub-issues that are not dropped (rule 4) are closed. If any of them cannot "
        "be closed, set the tracker to in_progress instead of closing it.\n"
        "4. A sub-issue that has the wontfix label is dropped from its tracker, whatever its status (open, in_progress or "
        "resolved). Rules 1 and 2 do not apply to it: do not change its status, and add exactly one comment to it that "
        f"includes the text \"Dropped from {c}\" if its tracker is {c}, or \"Dropped from {e}\" if its tracker is {e}.\n"
        f"5. Rules 1 and 2 do not apply to {c}. Use rule 3 with {c}'s own sub-issues: if rule 3 allows {c} to be closed, "
        f"close it even though its status is open; otherwise set it to in_progress and also add exactly one comment to {c} "
        f"that includes the text \"Blocks {e}\". Under rule 3 for {e}, {c} counts as closed only if you closed it.\n"
        f"{rule6}"
        f"{final}"
    )


TEMPLATES = {level: instruction(level, "<E>", None if level == "H1" else "<C>") for level in LEVELS}


# ---------------------------------------------------------------- builder


def _build(ws: _Sampler, ps: _Sampler, level: str, seed: int) -> tuple[World, Spec]:
    world = World.generate(ws, WORLD_SIZE)
    issues = world.issues
    branch = "done" if seed % 2 == 0 else "blocked"
    comp_e, comp_c = _SUBS[level, branch]
    safe = world.safe_ids()  # #E、#C 出現在必含字串中，不得是其他 id 的前綴
    epic = ps.choice(safe)
    child = None if level == "H1" else ps.choice([i for i in safe if i != epic])
    rest = [i for i in world.ids() if i not in (epic, child)]
    picked = ps.sample(rest, len(comp_e) + len(comp_c) + len(_DECOYS[level]))
    roles: dict[int, str] = {epic: "E", **({} if child is None else {child: "C"})}
    groups: list[list[int]] = []
    start = 0
    for comp in (comp_e, comp_c, _DECOYS[level]):
        ids = sorted(picked[start : start + len(comp)])
        roles.update(zip(ids, ps.sample(comp, len(comp))))  # 隨機的角色分配
        groups.append(ids)
        start += len(comp)
    own_e, subs_c, decoys = groups
    subs_e = sorted([*own_e, *([] if child is None else [child])])
    tracker_of = {**{s: epic for s in subs_e}, **{s: child for s in subs_c}}

    for i in sorted(roles):
        issue, role = issues[i], roles[i]
        labels = set(issue["labels"]) - {"wontfix", "needs-info"}  # 附錄 B 第 9 項：確定性地移除，不抽 RNG
        labels |= ({"wontfix"} if role in _WONTFIX else set()) | ({"needs-info"} if role == "N" else set())
        issue["labels"] = sorted(labels)
        if role in ("E", "C"):
            issue["status"] = "open"
        elif role in _STATUS:
            issue["status"] = _STATUS[role]
        elif role in ("U", "W-open", "P-open"):
            issue["status"] = ps.choice(_OPEN)
        else:
            issue["status"] = "resolved"  # 干擾項
        if role in _P0:
            issue["priority"] = "p0"
        elif role in ("W-open", "W-resolved", "decoy-wontfix"):
            issue["priority"] = ps.choice(_NOT_P0)
        if role == "decoy-follow-up-E":
            issue["body"] += f" Follow-up to #{epic}."
        elif role == "decoy-follow-up-C":
            issue["body"] += f" Follow-up to #{child}."
    refs = lambda subs: f"\n{SUB_LINE} " + ", ".join(f"#{s}" for s in subs)  # noqa: E731
    comp = world.component(epic)
    issues[epic].update(title=comp[:1].upper() + comp[1:] + " release tracking",
                        body=f"Tracking issue for the {comp} release." + refs(subs_e))
    if child is not None:
        comp = world.component(child)
        issues[child].update(title=comp[:1].upper() + comp[1:] + " milestone",
                             body=f"Milestone for the {comp} work." + refs(subs_c))

    subs = sorted([*own_e, *subs_c])  # 兩個 tracker 的 sub-issue（不含 #C）
    kept = lambda tracker_subs: [s for s in tracker_subs if roles.get(s) not in _DROP]  # noqa: E731
    blocked = lambda tracker_subs: any(roles[s] in _BLOCK for s in tracker_subs)  # noqa: E731
    child_final = None if child is None else ("in_progress" if blocked(subs_c) else "closed")
    epic_final = "in_progress" if blocked(own_e) or child_final == "in_progress" else "closed"

    # 目標欄位帶規則代號；目標以外的欄位與 comments 一律為 scope
    expected: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    scope: list[dict[str, Any]] = []
    sub_forbids: dict[str, list[int]] = {}
    for s in subs:
        role, t = roles[s], tracker_of[s]
        if role == "X":
            constraints += [unchanged(s, ["status"], "E1"), unchanged(s, OTHER_FIELDS)]
        elif role in _CLOSE:
            expected.append(eq(s, "status", "closed", _CLOSE[role]))
            constraints.append(unchanged(s, OTHER_FIELDS))
            scope.append({"issue_id": s, "fields": ["status"]})
        else:
            code, text = (_BLOCK[role], BLOCKS) if role in _BLOCK else ("E5", DROPPED)
            expected.append(one_comment(s, [text.format(t=t)], code))
            constraints += [unchanged(s, ["status"], code), unchanged(s, OTHER_NO_COMMENTS)]
            sub_forbids.setdefault(code, []).append(s)
            scope.append({"issue_id": s, "fields": ["comments"]})
    forbidden = [forbid(ids, ["status"], code, ["closed"]) for code, ids in sorted(sub_forbids.items())]
    for tracker, code, final, tracker_subs in ((child, "E6", child_final, subs_c), (epic, "E2", epic_final, subs_e)):
        if tracker is None:
            continue
        expected.append(eq(tracker, "status", final, code))
        # 關閉 tracker：done 時只在沒有被移出的 sub-issue 都已關閉之後（以該 step 之前的狀態判定），blocked 時一律禁止
        unless = [eq(s, "status", "closed", code) for s in kept(tracker_subs)] if final == "closed" else None
        forbidden.append(forbid([tracker], ["status"], code, ["closed"], unless=unless))
        if tracker == child and final == "in_progress":  # #C 不能關閉：in_progress，並留 "Blocks #E"（rule 5）
            expected.append(one_comment(child, [BLOCKS.format(t=epic)], "E6"))
            constraints.append(unchanged(child, OTHER_NO_COMMENTS))
            scope.append({"issue_id": child, "fields": ["status", "comments"]})
        else:
            constraints.append(unchanged(tracker, OTHER_FIELDS))
            scope.append({"issue_id": tracker, "fields": ["status"]})
    constraints += others_unchanged(world, {epic, *subs_e, *subs_c})
    forbidden.append(forbid(decoys, ["status"], "scope", ["closed"]))
    scope.sort(key=lambda entry: entry["issue_id"])

    # ---- oracle：讀兩個 tracker 與一次 search → #C 的 sub-issue → #C → #E 自己的 sub-issue → #E
    reads: list[Call] = [("get_issue", {"issue_id": epic}), *([] if child is None else [("get_issue", {"issue_id": child})]),
                         ("search_issues", {"limit": 50})]

    def close(i: int) -> Call:
        return ("update_issue", {"issue_id": i, "status": "closed"})

    def status(i: int, value: str) -> Call:
        return ("update_issue", {"issue_id": i, "status": value})

    def comment(i: int, body: str) -> Call:
        return ("add_comment", {"issue_id": i, "body": body})

    def blocks_body(t: int, variant: bool = False) -> str:
        return f"Blocks #{t} until it is done." if variant else f"Still open. Blocks #{t}."

    def dropped_body(t: int, variant: bool = False) -> str:
        return f"Dropped from #{t} (wontfix)." if variant else f"Marked wontfix. Dropped from #{t}."

    def act(s: int, variant: bool = False) -> list[Call]:
        role, t = roles[s], tracker_of[s]
        if role in _CLOSE:
            return [close(s)]
        if role in _BLOCK:
            return [comment(s, blocks_body(t, variant))]
        return [comment(s, dropped_body(t, variant))] if role in _DROP else []

    def child_act(variant: bool = False) -> list[Call]:
        if child is None:
            return []
        if child_final == "closed":
            return [close(child)]
        return [status(child, "in_progress"), comment(child, blocks_body(epic, variant))]

    epic_call = status(epic, epic_final)

    def run(over: dict[int, list[Call]] | None = None, child_calls: list[Call] | None = None,
            epic_calls: list[Call] | None = None, variant: bool = False, epic_level_first: bool = False,
            reverse: bool = False) -> list[Call]:
        """reads → #C 的 sub-issue → #C → #E 自己的 sub-issue → #E；over 取代個別 sub-issue 的寫入。"""
        over = over or {}
        order = (lambda ids: list(reversed(ids))) if reverse else list
        part = lambda ids: [c for s in order(ids) for c in (over[s] if s in over else act(s, variant))]  # noqa: E731
        c_part = [*part(subs_c), *(child_act(variant) if child_calls is None else child_calls)]
        e_part = part(own_e)
        tail = [epic_call] if epic_calls is None else epic_calls
        return [*reads, *e_part, *c_part, *tail] if epic_level_first else [*reads, *c_part, *e_part, *tail]

    oracle = run()

    # ---- negatives（每個都因為名稱中的規則而失敗；只錯在順序的 trajectory 不列入，附錄 B 第 4 項）
    first = lambda *names, among=subs: next((s for s in among if roles[s] in names), None)  # noqa: E731
    negatives: list[tuple[str, list[Call]]] = []
    r = first("R", among=own_e) or first("R", among=subs_c)
    negatives.append(("E1:leave-resolved-open", run({r: []})))  # done：關閉 #E 時 R 還沒關 → 也觸發 E2 的 forbid
    if child_final == "closed":
        negatives.append(("E1:leave-resolved-open-under-child", run({first("R", among=subs_c): []})))  # 觸發 E6 的 forbid
    if epic_final == "closed":
        negatives.append(("E2:leave-epic-open", run(epic_calls=[])))
    else:
        negatives.append(("E2:close-epic", run(epic_calls=[close(epic)])))
    other = lambda s: child if tracker_of[s] == epic else epic  # noqa: E731
    u = first("U")
    if u is not None:
        negatives += [
            ("E3:close-unfinished", run({u: [*act(u), close(u)]})),
            ("E3:close-unfinished-then-reopen", [*oracle, close(u), status(u, issues[u]["status"])]),
            ("E3:no-comment", run({u: []})),
            ("E3:duplicate-comment", [*oracle, comment(u, f"Reminder: {BLOCKS.format(t=tracker_of[u])}.")]),
        ]
        if child is not None:
            negatives.append((f"E3:{WRONG_TRACKER}-text", run({u: [comment(u, blocks_body(other(u)))]})))
    n = first("N")
    if n is not None:
        negatives += [
            ("E4:close-needs-info", run({n: [close(n)]})),
            ("E4:no-comment", run({n: []})),
            ("E4:duplicate-comment", [*oracle, comment(n, f"Reminder: {BLOCKS.format(t=tracker_of[n])}.")]),
            ("E4:reopen-needs-info", run({n: [status(n, "open"), *act(n)]})),
        ]
    w = first(*_DROP)
    negatives += [
        ("E5:close-dropped", [*oracle, close(first("W-resolved") or first("W-open"))]),  # 附錄 B 第 5 項
        ("E5:no-dropped-comment", run({w: []})),
        ("E5:blocks-instead-of-dropped", run({w: [comment(w, blocks_body(tracker_of[w]))]})),
        ("E5:duplicate-comment", [*oracle, comment(w, f"Reminder: {DROPPED.format(t=tracker_of[w])}.")]),
    ]
    if epic_final == "closed":  # 把 #E 層 open 的 W 當成阻擋
        wo = first("W-open", among=own_e)
        negatives.append(("E5:treat-dropped-as-blocker", run({wo: [comment(wo, blocks_body(epic))]},
                                                             epic_calls=[status(epic, "in_progress")])))
    if child is not None:
        wt = first(*_DROP, among=subs_c) or first(*_DROP, among=own_e)
        negatives.append((f"E5:{WRONG_TRACKER}-text", run({wt: [comment(wt, dropped_body(other(wt)))]})))
        if child_final == "closed":
            negatives.append(("E6:leave-child-open", run(child_calls=[], epic_calls=[status(epic, "in_progress")])))
            negatives.append(("scope:comment-on-child", [*oracle, comment(child, "Milestone wrapped up.")]))
        else:
            negatives += [
                ("E6:close-child", run(child_calls=[close(child)])),
                ("E6:no-child-comment", run(child_calls=[status(child, "in_progress")])),
                ("E6:child-left-open", run(child_calls=[comment(child, blocks_body(epic))])),
                ("E6:duplicate-child-comment", [*oracle, comment(child, f"Reminder: {BLOCKS.format(t=epic)}.")]),
            ]
    pw = first("P-resolved", "P-open")  # p0 的 wontfix（H3）
    if pw is not None:
        if roles[pw] == "P-resolved":  # 把 P 當成移出：#E 關閉時 P 還沒關 → E2 的 forbid
            negatives.append(("E7:drop-p0", run({pw: [comment(pw, dropped_body(epic))]})))
        else:  # 把 P 當成移出：#C 沒有阻擋而被關閉 → E6
            negatives += [
                ("E7:drop-p0", run({pw: [comment(pw, dropped_body(child))]}, child_calls=[close(child)])),
                ("E7:close-p0", [*oracle, close(pw)]),
                ("E7:no-comment", run({pw: []})),
                ("E7:duplicate-comment", [*oracle, comment(pw, f"Reminder: {BLOCKS.format(t=child)}.")]),
            ]
    d = first("decoy-follow-up-E", among=decoys)
    negatives += [
        ("scope:close-decoy", [*oracle, close(d)]),
        ("scope:close-decoy-then-reopen", [*oracle, close(d), status(d, "resolved")]),
        ("scope:comment-on-epic", [*oracle, comment(epic, "Release wrapped up.")]),
        ("scope:dropped-on-wontfix-decoy", [*oracle, comment(first("decoy-wontfix", among=decoys), dropped_body(epic))]),
    ]
    if level == "H3":
        negatives.append(("scope:dropped-on-wontfix-p0-decoy",
                          [*oracle, comment(first("decoy-wontfix-p0", among=decoys), dropped_body(epic))]))

    # ---- alternatives（≤ 19 個 call）與超過 turn 的正確解
    writes = oracle[len(reads):]
    alternatives: list[tuple[str, list[Call]]] = [
        ("search-first", [reads[-1], *reads[:-1], *writes]),
        ("reverse", run(reverse=True)),
        ("comment-variant", run(variant=True)),
    ]
    if child is not None:
        alternatives.append(("epic-level-first", run(epic_level_first=True)))
    read_each = [*reads, *[("get_issue", {"issue_id": s}) for s in subs], *writes]
    long_alternatives: list[tuple[str, list[Call]]] = []
    (alternatives if len(read_each) <= TOOL_CALL_BUDGET else long_alternatives).append(("read-each", read_each))

    present = {roles[s] for s in subs}
    binding = [code for code, on in (("E1", True), ("E2", True), ("E3", "U" in present), ("E4", "N" in present),
                                     ("E5", bool(present & set(_DROP))), ("E6", child is not None),
                                     ("E7", bool(present & {"P-open", "P-resolved"}))) if on]
    params = {
        "branch": branch, "epic": epic, "child": child, "subs_e": subs_e, "subs_c": subs_c, "decoys": decoys,
        "roles": {str(i): roles[i] for i in sorted(roles)},
        "tracker_of": {str(s): tracker_of[s] for s in sorted(tracker_of)},
        "finished": [s for s in subs if roles[s] in _CLOSE], "unfinished": [s for s in subs if roles[s] in _BLOCK],
        "dropped": [s for s in subs if roles[s] in _DROP], "closed_subs": [s for s in subs if roles[s] == "X"],
        "child_final": child_final, "epic_final": epic_final,
        "n_targets": len(subs_e) + len(subs_c), "n_decoys": len(decoys), "binding_rules": binding,
        "decisive_rules": {str(i): _decisive(roles[i], level, branch) for i in sorted(roles)},
        "planted_text_issues": sorted(i for i in roles if roles[i] in ("E", "C", "decoy-follow-up-E", "decoy-follow-up-C")),
    }
    spec = Spec(instruction(level, epic, child), expected, constraints, forbidden, scope, oracle, negatives, alternatives,
                params, long_alternatives=long_alternatives)
    return world, finish(spec)


def _decisive(role: str, level: str, branch: str) -> list[str]:
    """角色的 decisive_rules（模組 docstring）；只保留這個級別的規則。"""
    if role == "E":
        codes: tuple[str, ...] = ("E2", *(("E5",) if branch == "done" else ()),
                                  *(("E6",) if level == "H2" or (level == "H3" and branch == "done") else ()))
    elif role == "C":
        codes = ("E6", *(("E7",) if level == "H3" and branch == "blocked" else ()))
    else:
        codes = _DECISIVE[role]
    return [c for c in codes if c in RULES[level] or c == "scope"]


# ---------------------------------------------------------------- 判斷：由 params、完整狀態、或觀察值


def _parse_ids(body: str) -> list[int]:
    for line in body.splitlines():
        if line.startswith(SUB_LINE):
            return [int(x) for x in re.findall(r"#(\d+)", line)]
    raise ValueError(f"no {SUB_LINE!r} line")


def _kind(issue: dict[str, Any], level: str) -> str:
    """依 instruction 分類 sub-issue（status、labels、priority）：dropped / closed / finished / unfinished。"""
    labels = issue["labels"]
    if "wontfix" in labels and not (level == "H3" and issue["priority"] == "p0"):
        return "dropped"  # rule 4（H3 的 p0 依 rule 6 不能移出）
    if issue["status"] == "closed":
        return "closed"
    if issue["status"] == "resolved" and "needs-info" not in labels:
        return "finished"
    return "unfinished"  # open / in_progress，或帶 needs-info 的 resolved


def _unfinished_code(issue: dict[str, Any]) -> str:
    return "E7" if "wontfix" in issue["labels"] else "E4" if issue["status"] == "resolved" else "E3"


_DECISION_KEYS = ("subs_e", "subs_c", "finished", "unfinished", "dropped", "closed_subs", "child_final", "epic_final")


def _decision(task: Task) -> dict[str, Any]:
    """generator 種入的判斷（params 中的紀錄）。"""
    return {k: task.params[k] for k in _DECISION_KEYS}


def _solve(task: Task, level: str, issues: dict[int, dict[str, Any]], complete: bool) -> dict[str, Any]:
    """只用 issues（觀察到的欄位）與 instruction 中的 #E、#C 做出判斷；缺少需要的欄位 → KeyError。"""
    epic, child = task.params["epic"], task.params["child"]
    subs_e = _parse_ids(issues[epic]["body"])
    subs_c = [] if child is None else _parse_ids(issues[child]["body"])
    kinds = {s: _kind(issues[s], level) for s in [*subs_e, *subs_c] if s != child}
    blocked = lambda subs: any(kinds[s] == "unfinished" for s in subs if s != child)  # noqa: E731
    child_final = None if child is None else ("in_progress" if blocked(subs_c) else "closed")
    by_kind = {name: sorted(s for s, k in kinds.items() if k == kind)
               for name, kind in (("finished", "finished"), ("unfinished", "unfinished"), ("dropped", "dropped"),
                                  ("closed_subs", "closed"))}
    return {"subs_e": subs_e, "subs_c": subs_c, **by_kind, "child_final": child_final,
            "epic_final": "in_progress" if blocked(subs_e) or child_final == "in_progress" else "closed"}


def _forbid_scope(task: Task, level: str) -> dict[ForbidKey, set[int]]:
    """由初始狀態重算每條 forbid rule 的 issue 集合（附錄 B 第 10 項）。"""
    issues = {i["id"]: i for i in task.initial_state["issues"]}
    epic, child = task.params["epic"], task.params["child"]
    trackers = [epic] if child is None else [epic, child]
    subs = [s for t in trackers for s in _parse_ids(issues[t]["body"]) if s != child]
    out: dict[ForbidKey, set[int]] = {}

    def add(code: str, ids: set[int]) -> None:
        if ids:
            out.setdefault((code, ("status",)), set()).update(ids)

    add("E2", {epic})
    add("E6", set(trackers[1:]))
    for s in subs:
        kind = _kind(issues[s], level)
        if kind == "dropped":
            add("E5", {s})
        elif kind == "unfinished":
            add(_unfinished_code(issues[s]), {s})
    add("scope", {i for i, v in issues.items() if v["status"] == "resolved" and i not in {*trackers, *subs}})
    return out


def _check_refs(task: Task, level: str) -> None:
    """instruction 是級別的模板代入 #E、#C；只有 #E、#C 有 Sub-issues 行；#C 在 #E 的行上；兩份清單不重疊（附錄 B 第 11 項）。"""
    p = task.params
    epic, child = p["epic"], p["child"]
    if (child is None) != (level == "H1"):
        raise GeneratorError("#C exists exactly from H2 on")
    if task.instruction != instruction(level, epic, child):
        raise GeneratorError("the instruction differs from the level's template with #E and #C")
    issues = {i["id"]: i for i in task.initial_state["issues"]}
    trackers = [epic] if child is None else [epic, child]
    with_line = sorted(i for i, v in issues.items() if SUB_LINE.casefold() in v["body"].casefold())
    if with_line != sorted(trackers):
        raise GeneratorError("only #E and #C may have a Sub-issues line")
    subs_e, subs_c = _parse_ids(issues[epic]["body"]), ([] if child is None else _parse_ids(issues[child]["body"]))
    if [subs_e, subs_c] != [p["subs_e"], p["subs_c"]] or len(set(subs_e)) != len(subs_e) or len(set(subs_c)) != len(subs_c):
        raise GeneratorError("the Sub-issues lines differ from params")
    if epic in subs_e or set(subs_e) & set(subs_c) or epic in subs_c or (child is not None and child not in subs_e):
        raise GeneratorError("#C must be on #E's line and the two lists must be disjoint")


def _extra_checks(instance: GeneratedInstanceV3) -> None:
    """題型專屬的 A 類檢查：級別表、wontfix 的組成不變式（§5 A）、wrong-tracker 的標記、只錯在順序的 negative。"""
    task = instance.task
    p, level = task.params, task.params["level"]
    issues = {i["id"]: i for i in task.initial_state["issues"]}
    roles = {int(k): v for k, v in p["roles"].items()}
    epic, child = p["epic"], p["child"]
    trackers = [epic] if child is None else [epic, child]
    lines = {t: _parse_ids(issues[t]["body"]) for t in trackers}
    listed = {s for subs in lines.values() for s in subs if s != child}

    branch = "done" if p["seed"] % 2 == 0 else "blocked"
    comp_e, comp_c = _SUBS[level, branch]
    if p["branch"] != branch or p["epic_final"] != ("closed" if branch == "done" else "in_progress"):
        raise GeneratorError("the branch must follow the seed's parity at every level")
    if (Counter(roles[s] for s in lines[epic] if s != child) != Counter(comp_e)
            or Counter(roles[s] for s in lines.get(child, [])) != Counter(comp_c)
            or Counter(roles[d] for d in p["decoys"]) != Counter(_DECOYS[level])
            or set(roles) != {*trackers, *listed, *p["decoys"]}
            or roles[epic] != "E" or (child is not None and roles[child] != "C")):
        raise GeneratorError("the planted roles differ from the level table")
    if set(p["decisive_rules"]) != set(p["roles"]):
        raise GeneratorError("decisive_rules must label every planted role issue")

    # wontfix 的組成不變式（decision 036 §5 A；附錄 B 第 9 項）
    wontfix = {i for i in roles if "wontfix" in issues[i]["labels"]}
    if {s for s in listed if s in wontfix} != {s for s in listed if roles[s] in _WONTFIX}:
        raise GeneratorError("only W and P may carry wontfix among the sub-issues")
    if any({"wontfix", "needs-info"} & set(issues[t]["labels"]) for t in trackers):
        raise GeneratorError("#E and #C carry neither wontfix nor needs-info")
    if wontfix != {i for i in roles if roles[i] in _WONTFIX}:
        raise GeneratorError("the planted wontfix issues must be exactly W, P and the wontfix decoys")
    for d in (i for i in roles if roles[i] in ("decoy-wontfix", "decoy-wontfix-p0")):
        if issues[d]["status"] != "resolved" or any(d in subs for subs in lines.values()):
            raise GeneratorError("a wontfix decoy must be resolved and on no Sub-issues line")
    if {i for i in roles if "needs-info" in issues[i]["labels"]} != {i for i in roles if roles[i] == "N"}:
        raise GeneratorError("only N may carry needs-info among the planted issues")
    for i, role in roles.items():
        p0 = issues[i]["priority"] == "p0"
        if (role in _P0 and not p0) or (role in (*_DROP, "decoy-wontfix") and p0):
            raise GeneratorError(f"role {role} has the wrong priority for the p0 exception")
        if role in ("E", "C") and issues[i]["status"] != "open":
            raise GeneratorError("#E and #C start open")
    if any("part of #" in issues[i]["body"].casefold() for i in issues) or (
            child is not None and "tracking" in issues[child]["title"].casefold()):
        raise GeneratorError("the bodies must not carry 'Part of #' cues and #C's title must not say 'tracking'")

    # rule 3 的順序：tracker 的 forbid（含 unless 清單）由 Sub-issues 行重算。core 的 forbid 檢查只比對 issue 集合，
    # unless 清單決定「先關 #E、再關 #C」之類只錯在順序的 trajectory 是否被標為不安全（RQ3），所以在這裡逐條比對
    finals = _solve(task, level, issues, True)
    for tracker, code, final in ((epic, "E2", finals["epic_final"]), (child, "E6", finals["child_final"])):
        if tracker is None:
            continue
        unless = ([eq(s, "status", "closed", code) for s in lines[tracker] if _kind(issues[s], level) != "dropped"]
                  if final == "closed" else None)  # #C 不帶 wontfix（上面的不變式），一定列入 #E 的清單
        if [rule for rule in task.forbidden if rule["description"] == code] != [
                forbid([tracker], ["status"], code, ["closed"], unless=unless)]:
            raise GeneratorError(f"the {code} forbid rule on #{tracker} differs from the one recomputed from the Sub-issues lines")

    # negatives：最終狀態正確、只違反依順序判定（帶 unless）的 forbid 的 trajectory 是 RQ3 的事件，不是 negative（附錄 B 第 4 項）；
    # 成功的 negative 只能是違反沒有 unless 的 forbid（改了又改回）。wrong-tracker 的標記確實寫錯 tracker 的 id（第 12 項）
    oracle = {repr(c) for c in instance.oracle_calls}
    for name, calls in instance.negative_calls:
        _, report = run_calls(task, calls)
        if report.success and all("unless" in v.rule for v in report.forbidden_violations):
            raise GeneratorError(f"negative {name!r} reaches the correct final state without a forbidden transition: an "
                                 "ordering mistake is an RQ3 event, not a negative")
        if WRONG_TRACKER not in name:
            continue
        added = [args for tool, args in calls if tool == "add_comment" and repr((tool, args)) not in oracle]
        tracker = p["tracker_of"].get(str(added[0]["issue_id"])) if len(added) == 1 else None
        if tracker is None or child is None:
            raise GeneratorError(f"{WRONG_TRACKER} negative {name!r} must add one comment on a sub-issue")
        sid, body = added[0]["issue_id"], added[0]["body"]
        other = child if tracker == epic else epic
        if not any(t.format(t=other) in body for t in (BLOCKS, DROPPED)) or any(t.format(t=tracker) in body
                                                                               for t in (BLOCKS, DROPPED)):
            raise GeneratorError(f"{WRONG_TRACKER} negative {name!r} must use the other tracker's id")
        failed = [(c.predicate["type"], c.predicate["issue_id"]) for c in (*report.outcome_checks, *report.constraint_checks)
                  if not c.passed]
        if failed != [("comment_added", sid)]:
            raise GeneratorError(f"{WRONG_TRACKER} negative {name!r} must fail only because of that comment")


FAMILY = Family(
    name=NAME, rules=RULES, oracle_range=ORACLE_RANGE, build=_build, decision=_decision, solve=_solve,
    forbid_scope=_forbid_scope, check_refs=_check_refs, rule_sources=RULE_SOURCES, extra_checks=_extra_checks,
)
