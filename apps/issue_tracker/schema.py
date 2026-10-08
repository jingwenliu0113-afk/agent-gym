"""Issue Tracker 資料 schema（decision log 001）。

四種 entity：users、labels、issues、comments。
State 是純資料，所有 entity 為 frozen dataclass；tool 只會替換 entity，不會原地修改。

序列化格式（``IssueTrackerState.to_dict``）是 canonical 形式：
- 每個 collection 是 list，依主鍵排序（users/issues/comments 依 id，labels 依 name）。
- issue 的 labels 是排序後、不重複的 list。
- 只含 JSON 原生型別（str / int / bool / None / list / dict），不含 tuple，
  因此 ``json.loads(json.dumps(x)) == x``。

next comment id 不存成欄位，而是由現有 comments 推導（max id + 1，沒有 comment 時為 1），
因此 snapshot / restore 不可能出現 counter 與資料不一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "issue_tracker/v1"

UNASSIGNED = "unassigned"  # 保留字：不可作為 user id

ID_LIMIT = 2**63  # id / issue_id 必須落在 int64 範圍 [-2**63, 2**63)

STATUSES: tuple[str, ...] = ("open", "in_progress", "resolved", "closed")
PRIORITIES: tuple[str, ...] = ("p0", "p1", "p2", "p3")  # p0 最緊急

# 可被 tool 修改的 issue 欄位，順序即 diff 輸出順序。
# reporter_id 與 created_at 永遠不可被 tool 修改。
MUTABLE_ISSUE_FIELDS: tuple[str, ...] = (
    "title",
    "body",
    "status",
    "priority",
    "assignee_id",
    "labels",
)


USER_FIELDS: tuple[str, ...] = ("id", "name", "team", "active")
LABEL_FIELDS: tuple[str, ...] = ("name", "description")
ISSUE_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "body",
    "status",
    "priority",
    "assignee_id",
    "reporter_id",
    "labels",
    "created_at",
)
COMMENT_FIELDS: tuple[str, ...] = ("id", "issue_id", "author_id", "body")
COLLECTIONS: tuple[str, ...] = ("users", "labels", "issues", "comments")


class SchemaError(ValueError):
    """State 資料不合法（缺欄位、型別錯、enum 錯、重複主鍵、參照不存在、多餘欄位）。"""


@dataclass(frozen=True)
class User:
    id: str
    name: str
    team: str
    active: bool  # False = 已離職 / 停用；tool 不阻擋指派給停用者（由 task 的 forbidden 規則判定）


@dataclass(frozen=True)
class Label:
    name: str  # 主鍵
    description: str


@dataclass(frozen=True)
class Issue:
    id: int
    title: str
    body: str
    status: str  # STATUSES 之一
    priority: str  # PRIORITIES 之一
    assignee_id: str | None
    reporter_id: str
    labels: tuple[str, ...]  # 排序、不重複
    created_at: str  # 種子資料給定的 ISO 日期字串；tool 不會修改，env 不使用 wall clock


@dataclass(frozen=True)
class Comment:
    id: int
    issue_id: int
    author_id: str
    body: str


@dataclass
class IssueTrackerState:
    users: dict[str, User]
    labels: dict[str, Label]
    issues: dict[int, Issue]
    comments: dict[int, Comment]

    def next_comment_id(self) -> int:
        """回傳下一個 comment id：現有最大 id + 1；沒有 comment 時為 1。"""
        return max((c.id for c in self.comments.values()), default=0) + 1

    def to_dict(self) -> dict[str, Any]:
        """回傳 canonical、JSON-compatible 的 dict（見模組 docstring）。

        形狀::

            {
              "users":    [{"id", "name", "team", "active"}, ...],          # 依 id 排序
              "labels":   [{"name", "description"}, ...],                   # 依 name 排序
              "issues":   [{"id", "title", "body", "status", "priority",
                            "assignee_id", "reporter_id", "labels", "created_at"}, ...],  # 依 id 排序
              "comments": [{"id", "issue_id", "author_id", "body"}, ...],   # 依 id 排序
            }

        回傳值與 state 不共用任何可變物件。
        """
        return {
            "users": [
                {"id": u.id, "name": u.name, "team": u.team, "active": u.active}
                for u in sorted(self.users.values(), key=lambda u: u.id)
            ],
            "labels": [
                {"name": lb.name, "description": lb.description}
                for lb in sorted(self.labels.values(), key=lambda lb: lb.name)
            ],
            "issues": [_issue_to_dict(i) for i in sorted(self.issues.values(), key=lambda i: i.id)],
            "comments": [
                {"id": c.id, "issue_id": c.issue_id, "author_id": c.author_id, "body": c.body}
                for c in sorted(self.comments.values(), key=lambda c: c.id)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IssueTrackerState:
        """從 dict 建立 state，並做嚴格驗證；不合法時 raise SchemaError。

        驗證項目：
        - 頂層必須剛好有 users / labels / issues / comments 四個 key。
        - 每個 entity 必須剛好有 schema 定義的欄位（缺少或多餘都算錯）。
        - 型別：id / issue_id 為 int64 範圍內的 int（bool 不算 int）；字串欄位為可 UTF-8 編碼的 str
          （不可含 lone surrogate）；active 為 bool；assignee_id 為 str 或 None；labels 為 list[str]。
        - status ∈ STATUSES，priority ∈ PRIORITIES。
        - 主鍵不可重複。
        - 參照完整性：assignee_id / reporter_id / comment.author_id 必須是既有 user；
          issue.labels 必須是既有 label；comment.issue_id 必須是既有 issue。
        - user id 不可為保留字 ``"unassigned"``（search_issues 用它表示未指派）。
        - ``data`` 本身或任何 entity 不是 dict 時同樣 raise SchemaError（不得漏出 TypeError / KeyError）。

        issue.labels 輸入可未排序，會被正規化為排序、去重後的 tuple。
        不會修改 ``data``，回傳的 state 不與 ``data`` 共用可變物件。
        """
        _require(isinstance(data, dict), "state must be an object")
        _require_keys(data, COLLECTIONS, "state")
        for name in COLLECTIONS:
            _require(isinstance(data[name], list), f"{name} must be a list")

        users: dict[str, User] = {}
        for raw in data["users"]:
            _require_keys(raw, USER_FIELDS, "user")
            _require(_is_str(raw["id"]), "user.id must be a string")
            _require(raw["id"] != UNASSIGNED, f"user id {UNASSIGNED!r} is reserved")
            _require(_is_str(raw["name"]), "user.name must be a string")
            _require(_is_str(raw["team"]), "user.team must be a string")
            _require(isinstance(raw["active"], bool), "user.active must be a boolean")
            _require(raw["id"] not in users, f"duplicate user id {raw['id']!r}")
            users[raw["id"]] = User(raw["id"], raw["name"], raw["team"], raw["active"])

        labels: dict[str, Label] = {}
        for raw in data["labels"]:
            _require_keys(raw, LABEL_FIELDS, "label")
            _require(_is_str(raw["name"]), "label.name must be a string")
            _require(_is_str(raw["description"]), "label.description must be a string")
            _require(raw["name"] not in labels, f"duplicate label {raw['name']!r}")
            labels[raw["name"]] = Label(raw["name"], raw["description"])

        issues: dict[int, Issue] = {}
        for raw in data["issues"]:
            _require_keys(raw, ISSUE_FIELDS, "issue")
            _require(_is_int(raw["id"]), "issue.id must be an integer")
            for field in ("title", "body", "reporter_id", "created_at"):
                _require(_is_str(raw[field]), f"issue.{field} must be a string")
            _require(isinstance(raw["status"], str) and raw["status"] in STATUSES, "invalid issue.status")
            _require(isinstance(raw["priority"], str) and raw["priority"] in PRIORITIES, "invalid issue.priority")
            assignee = raw["assignee_id"]
            _require(assignee is None or _is_str(assignee), "issue.assignee_id must be a string or null")
            _require(assignee is None or assignee in users, f"unknown assignee {assignee!r}")
            _require(raw["reporter_id"] in users, f"unknown reporter {raw['reporter_id']!r}")
            raw_labels = raw["labels"]
            _require(isinstance(raw_labels, list), "issue.labels must be a list")
            for name in raw_labels:
                _require(_is_str(name), "issue.labels must contain strings")
                _require(name in labels, f"unknown label {name!r}")
            _require(raw["id"] not in issues, f"duplicate issue id {raw['id']}")
            issues[raw["id"]] = Issue(
                id=raw["id"],
                title=raw["title"],
                body=raw["body"],
                status=raw["status"],
                priority=raw["priority"],
                assignee_id=assignee,
                reporter_id=raw["reporter_id"],
                labels=tuple(sorted(set(raw_labels))),
                created_at=raw["created_at"],
            )

        comments: dict[int, Comment] = {}
        for raw in data["comments"]:
            _require_keys(raw, COMMENT_FIELDS, "comment")
            _require(_is_int(raw["id"]), "comment.id must be an integer")
            _require(_is_int(raw["issue_id"]), "comment.issue_id must be an integer")
            _require(_is_str(raw["author_id"]), "comment.author_id must be a string")
            _require(_is_str(raw["body"]), "comment.body must be a string")
            _require(raw["issue_id"] in issues, f"comment refers to unknown issue {raw['issue_id']}")
            _require(raw["author_id"] in users, f"unknown comment author {raw['author_id']!r}")
            _require(raw["id"] not in comments, f"duplicate comment id {raw['id']}")
            comments[raw["id"]] = Comment(raw["id"], raw["issue_id"], raw["author_id"], raw["body"])

        return cls(users=users, labels=labels, issues=issues, comments=comments)

    def copy(self) -> IssueTrackerState:
        """深拷貝，與原 state 不共用可變容器。"""
        # entity 都是 frozen dataclass（labels 為 tuple），只需複製外層 dict。
        return IssueTrackerState(
            users=dict(self.users),
            labels=dict(self.labels),
            issues=dict(self.issues),
            comments=dict(self.comments),
        )


def _is_int(value: Any) -> bool:
    return type(value) is int and -ID_LIMIT <= value < ID_LIMIT


def _is_str(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _require_keys(obj: Any, fields: tuple[str, ...], what: str) -> None:
    _require(isinstance(obj, dict), f"{what} must be an object")
    keys = set(obj)
    missing = [f for f in fields if f not in keys]
    _require(not missing, f"{what} is missing field(s): {', '.join(missing)}")
    extra = sorted(str(k) for k in keys if k not in fields)
    _require(not extra, f"{what} has unexpected field(s): {', '.join(extra)}")


def _issue_to_dict(issue: Issue) -> dict[str, Any]:
    return {
        "id": issue.id,
        "title": issue.title,
        "body": issue.body,
        "status": issue.status,
        "priority": issue.priority,
        "assignee_id": issue.assignee_id,
        "reporter_id": issue.reporter_id,
        "labels": sorted(set(issue.labels)),
        "created_at": issue.created_at,
    }
