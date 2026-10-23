"""scoped-triage（權限限制）的 generator v3 題型（decision 036 §3、§4、§5 A、附錄 B scoped-triage 1–15）。

規則代號（predicate 的 description；目標欄位以外的欄位與 comments 一律為 scope；代號是歸屬標籤，不是原因）
=====================================================================================================
- S1：只有帶 backend label 的 issue 可以修改欄位（forbid：沒有 backend 的 issue）。
- S2：帶 security label 的 issue 不可修改（forbid：H1 為 security；H2 起為 security ∧ ¬docs）。
- S3：priority 為 p0 的 issue 不可修改（forbid：p0；S7 不覆寫它）。
- S4：有權限的候選：指派給 instruction 的人並設 p2。
- S5：沒有權限的候選：不改欄位，恰好一則含 "Needs owner triage" 的 comment。
- S6（H1 起；來源為 Study 2 觀察到的失敗類型）：needs-info：一則含 "Needs reporter info" 的 comment（沒有權限時與 S5
  合併成一則）；有權限時改 p3、不指派。
- S7（H2 起）：同時有 security 與 docs 的 issue 不受 S2 限制；S1、S3 仍適用。
- S8（H3）：選取條件加入 performance；有權限的 performance issue 改 p1（S6 優先）。
- S9（H3）：有 duplicate label 的 issue 不需要處理（例外中的例外；dev 之前決定）。

instruction 使用編號規則（附錄 A；格式差異寫進限制），規則之間以內容互相引用；``params.rule_numbers`` 記錄代號與編號的對照。

角色（全部 open、未指派；權限依初始狀態判定，H 級別的寫入不會改變任何 issue 的 backend / security / docs / p0）
=========================================================================================================
- H1（oracle 9）：P、P-ni、D-nb、D-nb-ni、D-sec、D-p0；干擾項 assigned、in-progress、no-bug、assigned-ni。
- H2（oracle 11）：P、P-ni、P-secdocs、D-nb、D-nb-ni、D-sec、D-p0-secdocs；干擾項加上 secdocs-assigned。
- H3（oracle 12）：P-perf、P-perf-ni、P-secdocs、D-perf-nb、D-nb-ni、D-sec（+performance）、D-p0-secdocs；干擾項加上
  perf-in-progress、perf-assigned 與 S9 的 duplicate（open、未指派、bug + duplicate、沒有 backend：會出現在 bug 篩選的結果中）。
- priority（附錄 B 第 1 項，只固定沒有權限的角色）：D-nb-ni 在 H1–H3 固定為 p1；H3 的 D-sec 與 D-perf-nb 固定為 p3。其他
  角色照審查設計（``_TARGET_SPECS``）：有權限的角色排除自己的正確值（P-perf 為 {p2, p3}，審查明寫這些範圍沒有問題）。
- D-sec 的 extra label 依級別（審查設計）：H1 同 v2（docs、feature、frontend；H1 沒有 S7，security + docs 仍然沒有權限）；
  H2「不可抽到 docs」（否則角色變成 P-secdocs）；H3 為 bug + backend + security + performance，沒有 extra。
- §5 A 第 1 項：只改 priority 的錯誤寫入 negative（``PRIORITY_ONLY_NEGATIVES``：寫入值由 generator 選擇或固定，使它一定不等於
  初始值）一律產生 Change。S8:perf-p2 不在其中：p2 由規則決定，P-perf 為 p2 時這次寫入沒有 Change，但 negative 仍因 S8 的 eq
  （priority = p1）失敗（測試記錄）。§5 A 第 2 項：有權限與沒有權限（p0 角色除外）的候選至少共用一個 priority，priority 不能
  單獨區分兩者（不成立 → ``Redraw``）。

H3 的世界與截斷（decision 036 §3 第 3 種難度來源、§4 的重抽、附錄 B 第 8 項）
=========================================================================
- 62–68 個 issue。背景中恰好 4 個為 closed / in_progress / 已指派（labels 照 v2），其餘全部 open ∧ 未指派，labels 從
  (backend, docs, feature, frontend, wontfix) 抽 1–2 個（不含 bug、performance）。open ∧ 未指派的 issue 共 n − 10 ≥ 52 個。
- 深處候選：plan RNG 從 {P-secdocs, D-nb-ni, D-p0-secdocs} 與 {P-perf-ni, D-perf-nb} 各選一個，放在 id n − 1 與 n。
- 生成條件（不成立 → ``Redraw``，以 plan RNG 加計數器確定性地重抽）：id 小於深處候選的 open ∧ 未指派 issue ≥ 50；
  ``{status: open, assignee_id: unassigned, limit: 50}`` 的 total > 50，落在第 50 名之後的目標 ≥ 2 個而且包含兩個深處候選
  （記在 ``params.beyond_rank_50``）。extra_checks 在生成時以 env 的實際 search 再斷言一次。
- 單一 search 的列舉方法：能回傳全部目標的 search，每個 filter 都必須被全部目標滿足，所以 status 只能是 open 或不給、assignee
  只能是 unassigned 或不給、label / priority 只能是全部目標共有的值、query 只能是全部目標的 title 或 body 共有的子字串（逐一
  列舉）。單調性：多加一個 filter 只會移除結果，結果依 id 遞增，所以任何目標的名次只會前移；因此只需檢查「status = open ∧
  unassigned ∧ 各個共同值」這些最嚴格的組合：最大的目標 id 在結果中的名次 > 50，就沒有這種 search 能涵蓋全部目標。較寬的
  status search（``{status: open, limit: 50}``）因此也至少漏掉同一組目標。
- 結構化的 filter（status / assignee / label / priority）：種入的目標沒有共同的 label（P-perf 沒有 needs-info / frontend /
  docs / security，D-perf-nb 沒有 bug / backend），priority 也不只一種（D-p0-secdocs 為 p0），所以最嚴格的組合就是上面的
  status search，「沒有結構化的單一 search 涵蓋全部目標」由截斷條件蘊含（最大的目標是深處候選，名次 > 50）。因此它不是重抽
  條件，只在 extra_checks 中斷言（連同「沒有共同 label、priority 不只一種」這個前提）。
- query 子字串只列舉、不作為重抽條件（``covering_searches(task, queries=True)``）：全部目標共有的子字串多半是 " the"、"es"
  或單一字母，而背景中總有幾個 issue 不含它們，所以這類 search 在大多數世界都能涵蓋全部目標。以它為重抽條件不可行：debug
  seed 100–119 每個 seed 試 50 次抽取（共 1,000 次，全部滿足上面的截斷條件），只有 26 次（2.6%）沒有涵蓋全部目標的 query，
  20 個 seed 中有 8 個在 50 次內找不到這樣的抽取。這種 search 需要事先知道目標的文字，不是可行的 triage 策略；已知限制：
  「H3 最少 2 次讀取」只對結構化的 filter 成立（需要寫進 decision 036 附錄 B 第 8 項與附錄 A 的限制）。
- 以 label 篩選的兩次 search（oracle）不會被截斷；寬搜尋的 total 可見，資訊可以取得（不是陷阱）。

已知限制（寫進 decision 036 附錄 A）：comment 只檢查「包含」（附錄 B 第 12 項，測試記錄這個寬鬆）；H3 背景約 90% 是 open ∧
未指派（第 15 項）；H3 的 read-each 加一次 search 為 20 個 call（超過 turn 的正確解，第 5 項）；§2 的「S7 相關」以 decisive_rules
機械判定，P-secdocs 與 D-p0-secdocs 上的單純記帳失誤（例如漏一則 owner comment）也會計入，是上界。
"""

from __future__ import annotations

import re
from typing import Any

from env.environment import IssueTrackerEnv
from tasks.generator import GeneratorError, _Sampler
from tasks.generator_v2 import FIELD_CHANGES, NEEDS_OWNER
from tasks.generator_v3 import (
    Call,
    Family,
    GeneratedInstanceV3,
    Redraw,
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

NEEDS_INFO = "Needs reporter info"

RULES: dict[str, tuple[str, ...]] = {
    "H1": ("S1", "S2", "S3", "S4", "S5", "S6"),
    "H2": ("S1", "S2", "S3", "S4", "S5", "S6", "S7"),
    "H3": ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"),
}
ORACLE_RANGE = {"H1": (9, 9), "H2": (11, 11), "H3": (12, 12)}
# 規則來源（decision 036 附錄 A）：S1 是 034 記錄「把沒有 backend 的 issue 判為有權限」這類失敗的代號（eval L3-s3），
# S6 是刻意包含的同類判斷；兩者標為 Study 2 的失敗類型，其餘為 new。
RULE_SOURCES = {code: "study2-failure" if code in ("S1", "S6") else "new" for code in RULES["H3"]}

TARGETS: dict[str, tuple[str, ...]] = {
    "H1": ("P", "P-ni", "D-nb", "D-nb-ni", "D-sec", "D-p0"),
    "H2": ("P", "P-ni", "P-secdocs", "D-nb", "D-nb-ni", "D-sec", "D-p0-secdocs"),
    "H3": ("P-perf", "P-perf-ni", "P-secdocs", "D-perf-nb", "D-nb-ni", "D-sec", "D-p0-secdocs"),
}
DECOYS: dict[str, tuple[str, ...]] = {
    "H1": ("assigned", "in-progress", "no-bug", "assigned-ni"),
    "H2": ("assigned", "in-progress", "no-bug", "assigned-ni", "secdocs-assigned"),
    "H3": ("assigned", "in-progress", "no-bug", "assigned-ni", "secdocs-assigned", "perf-in-progress", "perf-assigned",
           "duplicate"),
}
# 規則代號 → instruction 的編號（附錄 B 第 3 項）；每條編號規則必含的文字（check_refs 核對對照）
RULE_NUMBERS: dict[str, dict[str, int]] = {
    "H1": {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6},
    "H2": {"S1": 1, "S2": 2, "S3": 3, "S7": 4, "S4": 5, "S5": 6, "S6": 7},
    "H3": {"S1": 1, "S2": 2, "S3": 3, "S7": 4, "S9": 5, "S4": 6, "S5": 7, "S6": 8, "S8": 9},
}
RULE_KEYS = {
    "S1": "only if it has the backend label",
    "S2": "only if it does not have the security label",
    "S3": "only if its priority is not p0",
    "S4": "If you are permitted to change it: assign it to",
    "S5": "If you are not permitted to change it: do not change any field",
    "S6": "If it has the needs-info label:",
    "S7": "both the security label and the docs label",
    "S8": "it has the performance label: set its priority to p1",
    "S9": "the duplicate label does not need triage",
}
SELECTION = {
    "H1": "Every issue whose status is open, that has no assignee, and that has the bug label needs triage. For each one:",
    "H3": ("An issue needs triage if all three of these hold: its status is open, it has no assignee, and it has the bug "
           "label, the performance label, or both."),
}
SELECTION["H2"] = SELECTION["H1"]

# 角色：(必含 labels, 可加的 label（機率 0.5 加一個）, priority 的取值)
_TARGET_SPECS: dict[str, tuple[frozenset[str], tuple[str, ...], tuple[str, ...]]] = {
    "P": (frozenset({"bug", "backend"}), ("docs", "feature", "frontend"), ("p1", "p3")),
    "P-ni": (frozenset({"bug", "backend", "needs-info"}), (), ("p1", "p2")),
    "P-secdocs": (frozenset({"bug", "backend", "security", "docs"}), (), ("p1", "p3")),
    "P-perf": (frozenset({"bug", "backend", "performance"}), (), ("p2", "p3")),
    "P-perf-ni": (frozenset({"backend", "performance", "needs-info"}), (), ("p1", "p2")),
    "D-nb": (frozenset({"bug"}), ("docs", "feature", "frontend"), ("p1", "p2", "p3")),
    "D-nb-ni": (frozenset({"bug", "needs-info"}), ("frontend",), ("p1",)),
    "D-sec": (frozenset({"bug", "backend", "security"}), ("feature", "frontend"), ("p1", "p2", "p3")),  # H2；其他級別見下
    "D-p0": (frozenset({"bug", "backend"}), ("docs", "feature", "frontend"), ("p0",)),
    "D-p0-secdocs": (frozenset({"bug", "backend", "security", "docs"}), (), ("p0",)),
    "D-perf-nb": (frozenset({"performance"}), ("frontend",), ("p3",)),
}
# D-sec 依級別：H1 的 extra 同 v2（可以是 docs：H1 沒有 S7）；H2 不可抽到 docs；H3 帶 performance、沒有 extra、固定 p3
_DSEC_BY_LEVEL = {
    "H1": (frozenset({"bug", "backend", "security"}), ("docs", "feature", "frontend"), ("p1", "p2", "p3")),
    "H2": _TARGET_SPECS["D-sec"],
    "H3": (frozenset({"bug", "backend", "security", "performance"}), (), ("p3",)),
}
# 干擾項：(status, 是否已指派（指派對象不是 instruction 的人）, labels)；priority 從 p1 / p3 抽
_DECOY_SPECS: dict[str, tuple[str, bool, frozenset[str]]] = {
    "assigned": ("open", True, frozenset({"backend", "bug"})),
    "in-progress": ("in_progress", False, frozenset({"backend", "bug"})),
    "no-bug": ("open", False, frozenset({"backend", "feature"})),
    "assigned-ni": ("open", True, frozenset({"backend", "bug", "needs-info"})),
    "secdocs-assigned": ("open", True, frozenset({"backend", "bug", "security", "docs"})),
    "perf-in-progress": ("in_progress", False, frozenset({"backend", "performance"})),
    "perf-assigned": ("open", True, frozenset({"backend", "performance"})),
    "duplicate": ("open", False, frozenset({"bug", "duplicate"})),
}
REQUIRED_LABELS: dict[str, frozenset[str]] = {**{r: v[0] for r, v in _TARGET_SPECS.items()},
                                              **{k: v[2] for k, v in _DECOY_SPECS.items()}}
_PERMITTED_ROLES = ("P", "P-ni", "P-secdocs", "P-perf", "P-perf-ni")
_REASON = {"D-nb": "S1", "D-nb-ni": "S1", "D-perf-nb": "S1", "D-sec": "S2", "D-p0": "S3", "D-p0-secdocs": "S3"}
_P0_ROLES = ("D-p0", "D-p0-secdocs")
_DEEP_BUG = ("P-secdocs", "D-nb-ni", "D-p0-secdocs")
_DEEP_PERF = ("P-perf-ni", "D-perf-nb")
_H3_BACKGROUND_LABELS = ("backend", "docs", "feature", "frontend", "wontfix")
_H3_ODD_BACKGROUND = 4
_LIMIT = 50  # search 的上限（apps.issue_tracker.tools.SEARCH_MAX_LIMIT）

# 每個角色的 decisive_rules（decision 036 §2 次要結果 2）：角色上 predicate 的規則代號，加上「這個角色就是為了測這條新規則的
# 邊界而種入」的兩項：D-p0-secdocs 的 S7（S7 的第二句「the p0 condition still apply」；把例外擴大到它是 S7 的典型錯誤，
# 不標 S7 的話這類 cliff 不會進 §2 的「S7 相關」那一行）與 H3 D-sec 的 S8（帶 performance：S8 的 p1 不覆寫 S2，而且它是 S8
# 聯集選取中兩次 search 都會出現、需要去重的那一個）。其他干擾項為 scope。
_DECISIVE: dict[str, list[str]] = {
    "P": ["S4"], "P-ni": ["S6"], "P-secdocs": ["S4", "S7"], "P-perf": ["S4", "S8"], "P-perf-ni": ["S6", "S8"],
    "D-nb": ["S1", "S5"], "D-nb-ni": ["S1", "S5", "S6"], "D-sec": ["S2", "S5"], "D-p0": ["S3", "S5"],
    "D-p0-secdocs": ["S3", "S5", "S7"], "D-perf-nb": ["S1", "S5", "S8"], "duplicate": ["S9"],
}

PRIORITY_ONLY_NEGATIVES = ("S4:wrong-priority", "S1:needs-info-priority-on-denied", "S2:perf-security-p1")
_PRIORITY_ONLY = {"H1": PRIORITY_ONLY_NEGATIVES[:2], "H2": PRIORITY_ONLY_NEGATIVES[:2], "H3": PRIORITY_ONLY_NEGATIVES}
_FIXED_PRIORITY = {"H1": {"D-nb-ni": "p1"}, "H2": {"D-nb-ni": "p1"}, "H3": {"D-nb-ni": "p1", "D-sec": "p3", "D-perf-nb": "p3"}}

OWNER_NOTE = f"{NEEDS_OWNER}: Triage Bot is not permitted to change this issue."
INFO_NOTE = f"{NEEDS_INFO}: please add the missing details."
BUG_SEARCH: Call = ("search_issues", {"status": "open", "label": "bug", "assignee_id": "unassigned"})
PERF_SEARCH: Call = ("search_issues", {"status": "open", "label": "performance", "assignee_id": "unassigned"})
NI_SEARCH: Call = ("search_issues", {"status": "open", "label": "needs-info", "assignee_id": "unassigned"})
STATUS_SEARCH: Call = ("search_issues", {"status": "open", "assignee_id": "unassigned", "limit": _LIMIT})


# ---------------------------------------------------------------- instruction（附錄 A 的模板）


def instruction_template(level: str) -> str:
    """級別的完整 instruction；指派對象以 ``{who}`` 表示（例如 "Alice Chen (u2)"）。"""
    lines = ["Triage Bot's permissions:",
             "1. You may change the fields of an issue only if it has the backend label.",
             "2. You may change the fields of an issue only if it does not have the security label.",
             "3. You may change the fields of an issue only if its priority is not p0."]
    if level != "H1":
        lines.append("4. Exception to the security-label condition: an issue that has both the security label and the docs "
                     "label is not restricted by it. The backend-label condition and the p0 condition still apply to such an "
                     "issue.")
    lines.append("Judge permissions by each issue's labels and priority before you make any change. You may add comments to "
                 "any issue.")
    lines.append(SELECTION[level])
    if level == "H3":
        lines += ["5. Exception: an issue that has the duplicate label does not need triage, even if all three hold. Leave it "
                  "unchanged, and do not comment on it.",
                  "For each issue that needs triage:"]
    n = RULE_NUMBERS[level]
    lines += [
        f"{n['S4']}. If you are permitted to change it: assign it to {{who}} and set its priority to p2.",
        f"{n['S5']}. If you are not permitted to change it: do not change any field; add exactly one comment that includes the "
        f"text \"{NEEDS_OWNER}\".",
        f"{n['S6']}. If it has the needs-info label: add exactly one comment that includes the text \"{NEEDS_INFO}\". If you "
        f"are not permitted to change it, add a single comment that includes both \"{NEEDS_OWNER}\" and \"{NEEDS_INFO}\" "
        "instead of two separate comments. If you are permitted to change it, set its priority to p3 instead of p2, and do "
        "not assign it.",
    ]
    if level == "H3":
        lines.append(f"{n['S8']}. If you are permitted to change it and it has the performance label: set its priority to p1 "
                     "instead of p2, and assign it as for any other issue you are permitted to change. The needs-info rule "
                     "takes precedence: if it also has the needs-info label, follow the needs-info rule instead of this one.")
    lines.append("Do not change anything else, and do not add any other comments.")
    return "\n".join(lines)


# ---------------------------------------------------------------- 判斷：由 params 或觀察值


def _decision(task: Task) -> dict[str, Any]:
    p = task.params
    return {k: p[k] for k in ("candidates", "permitted", "needs_info", "performance_p1", "excluded")}


def _permitted(issue: dict[str, Any], level: str) -> bool:
    labels = issue["labels"]
    if "backend" not in labels or issue["priority"] == "p0":
        return False
    return "security" not in labels or (level != "H1" and "docs" in labels)


def _selected(issue: dict[str, Any], level: str) -> bool:
    labels = issue["labels"]
    return (issue["status"] == "open" and issue["assignee_id"] is None
            and ("bug" in labels or (level == "H3" and "performance" in labels)))


def _solve(task: Task, level: str, issues: dict[int, dict[str, Any]], complete: bool) -> dict[str, Any]:
    """只用 issues（觀察到的摘要欄位：status、assignee_id、labels、priority）做出判斷；缺少欄位 → KeyError。"""
    pool = sorted(i for i, v in issues.items() if _selected(v, level))
    excluded = [i for i in pool if level == "H3" and "duplicate" in issues[i]["labels"]]
    candidates = [i for i in pool if i not in excluded]
    permitted = [i for i in candidates if _permitted(issues[i], level)]
    needs_info = [i for i in candidates if "needs-info" in issues[i]["labels"]]
    performance = [i for i in permitted if level == "H3" and "performance" in issues[i]["labels"] and i not in needs_info]
    return {"candidates": candidates, "permitted": permitted, "needs_info": needs_info, "performance_p1": performance,
            "excluded": excluded}


def _forbid_sets(issues: dict[int, dict[str, Any]], level: str) -> dict[str, set[int]]:
    """S1 / S2 / S3 的 forbid 集合（由初始狀態計算；H2 起 S2 為 security ∧ ¬docs）。"""
    return {
        "S1": {i for i, v in issues.items() if "backend" not in v["labels"]},
        "S2": {i for i, v in issues.items() if "security" in v["labels"] and (level == "H1" or "docs" not in v["labels"])},
        "S3": {i for i, v in issues.items() if v["priority"] == "p0"},
    }


def _forbid_scope(task: Task, level: str) -> dict[tuple[str, Any], set[int]]:
    fields = tuple(sorted(FIELD_CHANGES))
    sets = _forbid_sets({i["id"]: i for i in task.initial_state["issues"]}, level)
    return {(code, fields): ids for code, ids in sets.items()}


def _check_refs(task: Task, level: str) -> None:
    """編號規則與 params.rule_numbers 一致、規則之間不以編號引用、指派對象與選取句正確、全文等於級別的模板。"""
    text, p = task.instruction, task.params
    numbered = {int(m.group(1)): m.group(2) for m in re.finditer(r"^(\d+)\. (.*)$", text, flags=re.M)}
    if p["rule_numbers"] != RULE_NUMBERS[level] or sorted(numbered) != list(range(1, len(RULES[level]) + 1)):
        raise GeneratorError("the numbered rules do not match the level's rules")
    for code, number in RULE_NUMBERS[level].items():
        if RULE_KEYS[code] not in numbered[number]:
            raise GeneratorError(f"rule {number} of the instruction is not {code}")
    if re.search(r"\brules? \d", text, flags=re.I):
        raise GeneratorError("rules must refer to each other by content, not by number")
    names = {u["id"]: u["name"] for u in task.initial_state["users"]}
    who = f"{names.get(p['assignee'], '?')} ({p['assignee']})"
    if f"assign it to {who} and" not in text:
        raise GeneratorError("the instruction names a different assignee")
    if SELECTION[level] not in text:
        raise GeneratorError("the instruction does not contain the level's selection sentence")
    if text != instruction_template(level).format(who=who):
        raise GeneratorError("the instruction differs from the level's template")


# ---------------------------------------------------------------- H3 的截斷與單一 search 的列舉


def _open_unassigned(issues: dict[int, dict[str, Any]]) -> list[int]:
    return sorted(i for i, v in issues.items() if v["status"] == "open" and v["assignee_id"] is None)


def _substrings(text: str) -> set[str]:
    t = text.casefold()
    return {t[a:b] for a in range(len(t)) for b in range(a + 1, len(t) + 1)}


def _common_queries(issues: dict[int, dict[str, Any]], targets: list[int]) -> list[str]:
    """全部目標的 title 或 body 都含有的子字串（search 的 query 以 casefold 子字串比對 title 或 body）。"""
    common = None
    for t in targets:
        found = _substrings(issues[t]["title"]) | _substrings(issues[t]["body"])
        common = found if common is None else common & found
    return sorted(common or ())


def _covering(issues: dict[int, dict[str, Any]], targets: list[int], *, queries: bool = False) -> list[dict[str, Any]]:
    """能在 limit 50 內回傳全部目標的單一 search（列舉方法與單調性見模組 docstring）；queries 為 True 時也列舉 query 子字串。"""
    if not all(issues[t]["status"] == "open" and issues[t]["assignee_id"] is None for t in targets):
        raise GeneratorError("every target must be open and unassigned")
    labels = sorted(set.intersection(*(set(issues[t]["labels"]) for t in targets)))
    priorities = sorted({issues[t]["priority"] for t in targets})
    last = max(targets)
    found = []
    for query in (None, *(_common_queries(issues, targets) if queries else ())):
        for label in (None, *labels):
            for priority in (None, *(priorities if len(priorities) == 1 else ())):
                rank = sum(1 for i in _open_unassigned(issues) if i <= last
                           and (query is None or query in issues[i]["title"].casefold() or query in issues[i]["body"].casefold())
                           and (label is None or label in issues[i]["labels"])
                           and (priority is None or issues[i]["priority"] == priority))
                if rank <= _LIMIT:
                    args = {"status": "open", "assignee_id": "unassigned", "query": query, "label": label, "priority": priority}
                    found.append({k: v for k, v in args.items() if v is not None})
    return found


def common_queries(task: Task) -> list[str]:
    return _common_queries({i["id"]: i for i in task.initial_state["issues"]}, task.params["candidates"])


def covering_searches(task: Task, *, queries: bool = False) -> list[dict[str, Any]]:
    return _covering({i["id"]: i for i in task.initial_state["issues"]}, task.params["candidates"], queries=queries)


def _beyond_rank(issues: dict[int, dict[str, Any]], targets: list[int]) -> tuple[int, list[int]]:
    """status search（open ∧ 未指派，limit 50）的 total 與落在第 50 名之後的目標。"""
    matches = _open_unassigned(issues)
    return len(matches), sorted(set(targets) - set(matches[:_LIMIT]))


def _redraw_conditions(issues: dict[int, dict[str, Any]], roles: dict[int, str], level: str, deep: list[int]) -> None:
    """生成條件（decision 036 §4、§5 A）；不成立 → Redraw（確定性的重抽）。"""
    permitted = {issues[i]["priority"] for i, r in roles.items() if r in _PERMITTED_ROLES}
    denied = {issues[i]["priority"] for i, r in roles.items() if r not in _PERMITTED_ROLES and r not in _P0_ROLES}
    if not permitted & denied:
        raise Redraw()  # priority 單獨就能區分有權限與沒有權限的候選
    if level != "H3":
        return
    if sum(1 for i in _open_unassigned(issues) if i < min(deep)) < _LIMIT:
        raise Redraw()
    total, beyond = _beyond_rank(issues, sorted(roles))
    if total <= _LIMIT or len(beyond) < 2 or not set(deep) <= set(beyond):
        raise Redraw()
    # 「沒有結構化的單一 search 涵蓋全部目標」由上面的條件蘊含（模組 docstring），只在 extra_checks 中斷言


# ---------------------------------------------------------------- builder


def _shape_background(s: _Sampler, world: World, planted: set[int], level: str) -> None:
    issues = world.issues
    background = [i for i in world.ids() if i not in planted]
    if level != "H3":  # 同 v2：沒有其他 open、未指派、帶 bug 的 issue
        for i in background:
            if issues[i]["status"] == "open" and issues[i]["assignee_id"] is None and "bug" in issues[i]["labels"]:
                issues[i]["assignee_id"] = s.choice(world.staff)
        return
    odd = s.sample(background, _H3_ODD_BACKGROUND)
    for i in background:
        issue = issues[i]
        if i in odd:  # labels 照 v2
            kind = s.choice(("closed", "in_progress", "assigned"))
            if kind == "assigned":
                issue["status"], issue["assignee_id"] = "open", s.choice(world.staff)
            else:
                issue["status"] = kind
        else:
            issue["status"], issue["assignee_id"] = "open", None
            issue["labels"] = sorted(s.sample(_H3_BACKGROUND_LABELS, s.choice((1, 2))))


def _build(ws: _Sampler, s: _Sampler, level: str, seed: int) -> tuple[World, Spec]:
    h3 = level == "H3"
    world = World.generate(ws, (62, 68) if h3 else (22, 28))
    issues, ids = world.issues, world.ids()
    targets, kinds = TARGETS[level], DECOYS[level]
    deep: list[int] = []
    if h3:
        deep_roles = [s.choice(_DEEP_BUG), s.choice(_DEEP_PERF)]
        deep = s.sample(ids[-2:], 2)
        rest = [r for r in targets if r not in deep_roles]
        picked = s.sample(ids[:-2], len(rest) + len(kinds))
        roles = {**dict(zip(deep, deep_roles)), **dict(zip(picked, rest))}
        decoys = dict(zip(picked[len(rest):], kinds))
    else:
        picked = s.sample(ids, len(targets) + len(kinds))
        roles = dict(zip(picked, targets))
        decoys = dict(zip(picked[len(targets):], kinds))
    assignee = s.choice(("u1", "u2", "u3"))
    others = [u for u in world.staff if u != assignee]
    _shape_background(s, world, set(roles) | set(decoys), level)
    for i, role in roles.items():
        labels, extras, priorities = _DSEC_BY_LEVEL[level] if role == "D-sec" else _TARGET_SPECS[role]
        issue = issues[i]
        issue["status"], issue["assignee_id"] = "open", None
        issue["labels"] = sorted(labels | ({s.choice(extras)} if extras and s.random() < 0.5 else set()))
        issue["priority"] = s.choice(priorities)
    for i, kind in decoys.items():
        status, assigned, labels = _DECOY_SPECS[kind]
        issues[i].update(status=status, assignee_id=s.choice(others) if assigned else None, labels=sorted(labels),
                         priority=s.choice(("p1", "p3")))
    _redraw_conditions(issues, roles, level, deep)

    by_role = {role: i for i, role in roles.items()}
    by_kind = {kind: i for i, kind in decoys.items()}
    candidates = sorted(roles)
    permitted = sorted(i for i in candidates if roles[i] in _PERMITTED_ROLES)
    denied = sorted(set(candidates) - set(permitted))
    needs_info = sorted(i for i in candidates if "needs-info" in issues[i]["labels"])
    excluded = sorted(i for i, kind in decoys.items() if kind == "duplicate")
    names = world.names
    instruction = instruction_template(level).format(who=f"{names[assignee]} ({assignee})")

    expected: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    scope: list[dict[str, Any]] = []
    for i in candidates:
        role = roles[i]
        if role in ("P", "P-secdocs", "P-perf"):
            perf = role == "P-perf"
            expected += [eq(i, "assignee_id", assignee, "S4"), eq(i, "priority", "p1" if perf else "p2", "S8" if perf else "S4")]
            if role == "P-secdocs":  # 當成沒有權限的典型錯誤是留 owner comment（附錄 B 第 10 項）
                constraints += [unchanged(i, ["comments"], "S7"), unchanged(i, ["title", "body", "status", "labels"])]
            else:
                constraints.append(unchanged(i, ["title", "body", "status", "labels", "comments"]))
            scope.append({"issue_id": i, "fields": ["assignee_id", "priority"]})
        elif role in ("P-ni", "P-perf-ni"):
            expected += [eq(i, "priority", "p3", "S6"), one_comment(i, [NEEDS_INFO], "S6")]
            if role == "P-perf-ni":  # 重複的 S8 predicate：漏做只有 performance 的候選時看得到 S8
                expected.append(one_comment(i, [NEEDS_INFO], "S8"))
            constraints += [unchanged(i, ["assignee_id"], "S6"), unchanged(i, ["title", "body", "status", "labels"])]
            scope.append({"issue_id": i, "fields": ["priority", "comments"]})
        else:
            expected.append(one_comment(i, [NEEDS_OWNER], "S5"))
            if role == "D-nb-ni":  # 兩個 max_count 1 的 predicate 合起來等於「恰好一則、同時含兩段文字」
                expected.append(one_comment(i, [NEEDS_INFO], "S6"))
            if role == "D-perf-nb":
                expected.append(one_comment(i, [NEEDS_OWNER], "S8"))
            constraints.append(unchanged(i, FIELD_CHANGES, _REASON[role]))
            scope.append({"issue_id": i, "fields": ["comments"]})
    constraints += [unchanged(i, None, "S9") for i in excluded]
    constraints += others_unchanged(world, set(candidates) | set(excluded))
    forbidden = [forbid(sorted(ids), FIELD_CHANGES, code) for code, ids in _forbid_sets(issues, level).items()]

    reads: list[Call] = [BUG_SEARCH, PERF_SEARCH] if h3 else [BUG_SEARCH]

    def act(i: int, *, swap: bool = False, owner: str = OWNER_NOTE, info: str = INFO_NOTE, both: str | None = None) -> list[Call]:
        role = roles[i]
        if role in ("P", "P-secdocs", "P-perf"):
            calls: list[Call] = [("assign_issue", {"issue_id": i, "assignee_id": assignee}),
                                 ("update_issue", {"issue_id": i, "priority": "p1" if role == "P-perf" else "p2"})]
        elif role in ("P-ni", "P-perf-ni"):
            calls = [("update_issue", {"issue_id": i, "priority": "p3"}), ("add_comment", {"issue_id": i, "body": info})]
        else:
            body = (both or f"{owner} {info}") if i in needs_info else owner
            calls = [("add_comment", {"issue_id": i, "body": body})]
        return calls[::-1] if swap else calls

    actions = [c for i in candidates for c in act(i)]
    oracle = [*reads, *actions]

    def replaced(i: int, calls: list[Call]) -> list[Call]:  # 只把 i 的動作換成 calls
        return [*reads, *[c for j in candidates for c in (calls if j == i else act(j))]]

    def assign(i: int) -> Call:
        return ("assign_issue", {"issue_id": i, "assignee_id": assignee})

    def priority(i: int, value: str) -> Call:
        return ("update_issue", {"issue_id": i, "priority": value})

    def comment(i: int, body: str) -> Call:
        return ("add_comment", {"issue_id": i, "body": body})

    nb = by_role.get("D-nb", by_role.get("D-perf-nb"))
    ni = by_role["D-nb-ni"]
    sec = by_role["D-sec"]
    pni = by_role.get("P-ni", by_role.get("P-perf-ni"))
    plain = by_role.get("P", by_role.get("P-secdocs"))
    p0 = by_role.get("D-p0", by_role.get("D-p0-secdocs"))
    plain_denied = next(i for i in denied if i not in needs_info)
    wrong = "p3" if issues[plain]["priority"] == "p1" else "p1"
    negatives: list[tuple[str, list[Call]]] = [
        ("S1:edit-no-backend", [*oracle, assign(nb)]),
        ("S1:edit-then-revert", [*oracle, priority(nb, "p0"), priority(nb, issues[nb]["priority"])]),
        ("S1:needs-info-priority-on-denied", [*oracle, priority(ni, "p3")]),
        ("S2:edit-security", replaced(sec, [assign(sec), priority(sec, "p2")])),
        (f"S3:edit-{roles[p0].removeprefix('D-')}", replaced(p0, [assign(p0), priority(p0, "p2")])),
        ("S4:wrong-priority", replaced(plain, [assign(plain), priority(plain, wrong)])),
        ("S5:no-comment", replaced(plain_denied, [])),
        ("S5:duplicate-comment", [*oracle, comment(plain_denied, f"{NEEDS_OWNER} (reminder).")]),
        ("S5:denied-ni-reporter-only", replaced(ni, [comment(ni, INFO_NOTE)])),
        ("S6:permitted-ni-as-normal", replaced(pni, [assign(pni), priority(pni, "p2")])),
        ("S6:perf-ni-p3-but-assigned" if h3 else "S6:permitted-ni-assigned", [*oracle, assign(pni)]),
        ("S6:permitted-ni-owner-text", replaced(pni, [priority(pni, "p3"), comment(pni, OWNER_NOTE)])),
        ("S6:denied-ni-two-comments", replaced(ni, [comment(ni, OWNER_NOTE), comment(ni, INFO_NOTE)])),
        ("S6:denied-ni-owner-only", replaced(ni, [comment(ni, OWNER_NOTE)])),
        ("scope:touch-decoy", [*oracle, priority(by_kind["assigned"], "p2")]),
        ("scope:triage-in-progress-decoy", [*oracle, assign(by_kind["in-progress"]), priority(by_kind["in-progress"], "p2")]),
        ("scope:comment-on-assigned-ni-decoy", [*oracle, comment(by_kind["assigned-ni"], INFO_NOTE)]),
    ]
    if not h3:
        negatives.append(("S4:no-assign", replaced(plain, [priority(plain, "p2")])))
    if level != "H1":
        sd = by_role["P-secdocs"]
        negatives += [
            ("S7:secdocs-treated-as-denied", replaced(sd, [comment(sd, OWNER_NOTE)])),
            ("S7:secdocs-plus-owner-comment", [*oracle, comment(sd, OWNER_NOTE)]),
            ("scope:secdocs-decoy", [*oracle, assign(by_kind["secdocs-assigned"])]),
        ]
    params_h3: dict[str, Any] = {}
    if h3:
        pp, pnb, dup = by_role["P-perf"], by_role["D-perf-nb"], by_kind["duplicate"]
        bug_only = [c for i in candidates if "bug" in issues[i]["labels"]
                    for c in ([assign(pp), priority(pp, "p2")] if i == pp else act(i))]
        total, beyond = _beyond_rank(issues, candidates)
        first_page = [c for i in candidates if i not in beyond for c in act(i)]
        deep_bug = next(i for i in deep if roles[i] in _DEEP_BUG)
        deep_code = next(pred["description"] for pred in expected if pred["issue_id"] == deep_bug)
        negatives += [
            ("S4:perf-no-assign", replaced(pp, [priority(pp, "p1")])),
            ("S8:perf-p2", replaced(pp, [assign(pp), priority(pp, "p2")])),
            ("S8:bug-search-only", [BUG_SEARCH, *bug_only]),
            ("S6:perf-ni-takes-p1", replaced(pni, [assign(pni), priority(pni, "p1"), comment(pni, INFO_NOTE)])),
            ("S1:perf-no-backend-p1", replaced(pnb, [assign(pnb), priority(pnb, "p1")])),
            ("S2:perf-security-p1", [*oracle, priority(sec, "p1")]),
            ("S5:comment-twice-from-both-searches", [*oracle, comment(sec, OWNER_NOTE)]),
            (f"{deep_code}:truncated-broad-search", [STATUS_SEARCH, *first_page]),
            ("scope:perf-in-progress-decoy", [*oracle, assign(by_kind["perf-in-progress"]),
                                              priority(by_kind["perf-in-progress"], "p1")]),
            ("scope:perf-assigned-decoy", [*oracle, assign(by_kind["perf-assigned"]), priority(by_kind["perf-assigned"], "p1")]),
            ("S9:comment-on-duplicate", [*oracle, comment(dup, "Skipped: this issue has the duplicate label.")]),
            ("S9:triage-duplicate", [*oracle, comment(dup, OWNER_NOTE)]),
        ]
        params_h3 = {"deep": sorted(deep), "beyond_rank_50": beyond, "status_search_total": total}

    gets = [("get_issue", {"issue_id": i}) for i in candidates]
    variant = {"owner": f"{NEEDS_OWNER} - please take a look.", "info": f"{NEEDS_INFO} - please add steps to reproduce."}
    alternatives: list[tuple[str, list[Call]]] = [
        ("read-each", [*reads, *gets, *actions]),
        ("reverse", [*reads, *[c for i in reversed(candidates) for c in act(i, swap=True)]]),
        ("comment-variant", [*reads, *[c for i in candidates for c in act(i, **variant)]]),
        ("texts-swapped", [*reads, *[c for i in candidates for c in act(i, both=f"{INFO_NOTE} {OWNER_NOTE}")]]),
    ]
    long_alternatives: list[tuple[str, list[Call]]] = []
    extra_search = ("read-each-extra-search", [*reads, NI_SEARCH, *gets, *actions])
    if h3:
        alternatives += [
            ("swapped-searches", [PERF_SEARCH, BUG_SEARCH, *actions]),
            ("label-only-searches", [("search_issues", {"label": "bug", "limit": _LIMIT}),
                                     ("search_issues", {"label": "performance", "limit": _LIMIT}), *actions]),
        ]
        with_duplicate = [("get_issue", {"issue_id": i}) for i in sorted([*candidates, *excluded])]
        long_alternatives += [extra_search, ("read-each-with-duplicate", [*reads, *with_duplicate, *actions])]
    else:
        alternatives += [extra_search, ("broad-search", [("search_issues", {"limit": _LIMIT}), *actions])]

    planted = {**roles, **decoys}
    params = {
        "candidates": candidates, "permitted": permitted, "needs_info": needs_info,
        "performance_p1": sorted(i for i in permitted if roles[i] == "P-perf"), "excluded": excluded,
        "roles": {str(i): planted[i] for i in sorted(planted)}, "decoys": sorted(decoys), "assignee": assignee,
        "n_targets": len(candidates), "n_decoys": len(decoys), "binding_rules": list(RULES[level]),
        "planted_text_issues": [], "rule_numbers": dict(RULE_NUMBERS[level]),
        "decisive_rules": {str(i): _decisive(planted[i], level) for i in sorted(planted)}, **params_h3,
    }
    spec = Spec(instruction, expected, constraints, forbidden, sorted(scope, key=lambda e: e["issue_id"]), oracle, negatives,
                alternatives, params, long_alternatives=long_alternatives)
    return world, finish(spec)


def _decisive(role: str, level: str) -> list[str]:
    if role == "D-sec" and level == "H3":
        return ["S2", "S5", "S8"]
    return list(_DECISIVE.get(role, ["scope"]))


# ---------------------------------------------------------------- 題型專屬的 A 類檢查


def _reasons(report: Any) -> set[str]:
    failed = {c.predicate.get("description") for c in (*report.outcome_checks, *report.constraint_checks) if not c.passed}
    return failed | {v.rule.get("description") for v in report.forbidden_violations}


def _extra_checks(instance: GeneratedInstanceV3) -> None:
    _check_negatives(instance)
    _check_priorities(instance.task)
    if instance.task.params["level"] == "H3":
        _check_h3_searches(instance.task)


def _check_negatives(instance: GeneratedInstanceV3) -> None:
    """附錄 B 第 11 項：每個 negative 都因為名稱中的規則而失敗；§5 A 第 1 項：只改 priority 的錯誤寫入一定產生 Change。"""
    task = instance.task
    priority_only = set()
    for name, calls in instance.negative_calls:
        env, report = run_calls(task, calls)
        if name.split(":", 1)[0] not in _reasons(report):
            raise GeneratorError(f"negative {name!r} does not fail because of its named rule")
        if name in PRIORITY_ONLY_NEGATIVES:
            priority_only.add(name)
            writes = [r for r in env.records if r.tool == "update_issue" and set(r.args) == {"issue_id", "priority"}]
            if not writes or not all(any(c.field == "priority" for c in r.changes) for r in writes):
                raise GeneratorError(f"the priority-only negative {name!r} writes a priority without a change")
    if priority_only != set(_PRIORITY_ONLY[task.params["level"]]):
        raise GeneratorError("the level's priority-only negatives are missing")


def _check_priorities(task: Task) -> None:
    """附錄 B 第 1 項的固定 priority；§5 A 第 2 項：priority 不能單獨區分有權限與沒有權限的候選（p0 角色除外）。"""
    p = task.params
    issues = {i["id"]: i for i in task.initial_state["issues"]}
    roles = {int(i): role for i, role in p["roles"].items()}
    fixed = _FIXED_PRIORITY[p["level"]]
    for i, role in roles.items():
        if role in fixed and issues[i]["priority"] != fixed[role]:
            raise GeneratorError(f"{role} must have priority {fixed[role]}")
    permitted = {issues[i]["priority"] for i in p["permitted"]}
    denied = {issues[i]["priority"] for i in p["candidates"] if i not in p["permitted"] and roles[i] not in _P0_ROLES}
    if not permitted & denied:
        raise GeneratorError("priority alone separates the permitted candidates from the denied ones")


def _check_h3_searches(task: Task) -> None:
    """§5 A 的 H3 搜尋（截斷、params.beyond_rank_50、較寬的 search、duplicate 在篩選結果中）與附錄 B 第 8 項的斷言。"""
    p = task.params
    issues = {i["id"]: i for i in task.initial_state["issues"]}
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    targets = set(p["candidates"])
    data = env.call_tool(*STATUS_SEARCH)["data"]
    beyond = sorted(targets - {s["id"] for s in data["issues"]})
    if data["total"] <= _LIMIT or len(beyond) < 2 or not set(p["deep"]) <= set(beyond):
        raise GeneratorError("the status search must be truncated and miss at least two targets, the deep ones included")
    if beyond != p["beyond_rank_50"] or data["total"] != p["status_search_total"]:
        raise GeneratorError("params.beyond_rank_50 differs from the status search")
    wide = env.call_tool("search_issues", {"status": "open", "limit": _LIMIT})["data"]
    if not set(beyond) <= targets - {s["id"] for s in wide["issues"]}:
        raise GeneratorError("the wider status search must miss the same targets")
    if sum(1 for i in _open_unassigned(issues) if i < min(p["deep"])) < _LIMIT:
        raise GeneratorError("fewer than 50 open, unassigned issues come before the deep candidates")
    bug = env.call_tool(*BUG_SEARCH)["data"]
    if len(p["excluded"]) != 1 or not set(p["excluded"]) <= {s["id"] for s in bug["issues"]}:
        raise GeneratorError("the duplicate decoy must appear in the filtered bug search")
    # 「沒有結構化的單一 search 涵蓋全部目標」由截斷蘊含的前提：目標沒有共同的 label，priority 也不只一種
    if set.intersection(*(set(issues[i]["labels"]) for i in targets)) or len({issues[i]["priority"] for i in targets}) < 2:
        raise GeneratorError("the targets must share no label and no single priority")
    if covering_searches(task):
        raise GeneratorError("a single structured search covers every target within the limit")


FAMILY = Family(
    name="scoped-triage", rules=RULES, oracle_range=ORACLE_RANGE, build=_build, decision=_decision, solve=_solve,
    forbid_scope=_forbid_scope, check_refs=_check_refs, rule_sources=RULE_SOURCES, extra_checks=_extra_checks,
    banned_background=("needs reporter info",),
)
