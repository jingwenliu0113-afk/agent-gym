"""Issue Tracker environment：reset / snapshot / restore / 逐步記錄（decision log 002）。

核心 invariant（SPEC §5）：restore 後，Agent 與 environment 看到的 observable state
必須與原 checkpoint 一致。可執行的檢驗程序：

1. env A 執行 tool call 序列 P，取 ``snap = A.snapshot()``。
2. 新的 env B 執行 ``B.restore(json.loads(json.dumps(snap)))``，斷言 ``B.snapshot() == snap``。
3. A、B 執行相同的序列 S：每一步 observation 相同，``A.records[len(P):] == B.records``
   （含 step 欄位），最後 ``A.snapshot() == B.snapshot()``。

State hash 的涵蓋範圍：``{"schema_version", "actor_id", "state": state.to_dict()}``。
``step`` 不納入 hash（它不影響任何 tool 的 observation；納入會讓 read-only call 也改變 hash）。
已知限制（decision 004）：step 不受 hash 保護。只有當被跳過或重複的前綴區段有淨狀態變化時，
grader 的重放檢查才擋得下竄改 step 或錯配前綴；若該區段淨變化為零（read-only call、失敗的 call、
改了又改回），拼接結果與一條可真實執行的 trajectory 完全相同，只拿 records 的 grader 無法分辨。
前綴來源的綁定屬於 Week 1 Recorder checkpoint 的責任。

Step 語義：``env.step`` 是自 reset 起累計的 tool call 數（restore 時取 ``snapshot["step"]``）。
每次 call 的 ``record.step`` 等於該 call 之前的 ``env.step``；call 之後 ``env.step == record.step + 1``。
從 checkpoint rerun 一律使用 ``replay.checkpoint.rerun``（前綴由 checkpoint 保存並重放驗證，decision 006）；
不得自行切 ``original_records[:k]``。

Env 不保存 wall clock，ToolCallRecord 也不含時間。latency 等非決定性量測屬於 agent runner，
在 grader 的 Usage 中傳入。

Env 不阻擋任何業務規則上「危險」的操作；forbidden 由 grader 依 task 規則判定。

所有 ``to_dict`` 的輸出都是 JSON-native（不含 tuple），滿足
``X.from_dict(json.loads(canonical_json(x.to_dict()))) == x``。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from apps.issue_tracker.schema import (
    MUTABLE_ISSUE_FIELDS,
    SCHEMA_VERSION,
    IssueTrackerState,
    SchemaError,
)
from apps.issue_tracker.tools import TOOL_SPECS, ToolError, execute_tool
from env.hashing import canonical_json, sha256_of

SNAPSHOT_FORMAT = "agent-gym/env-snapshot/v1"

# call_tool 前置條件：args 的 JSON 巢狀深度上限（decision 004）。
MAX_ARGS_DEPTH = 64

SNAPSHOT_KEYS: frozenset[str] = frozenset(
    {"format", "schema_version", "actor_id", "step", "state", "state_hash"}
)

# Change.field 的可能值：MUTABLE_ISSUE_FIELDS 之一，或 "comments"（新增 comment）。
COMMENTS_FIELD = "comments"
CHANGE_FIELDS: tuple[str, ...] = (*MUTABLE_ISSUE_FIELDS, COMMENTS_FIELD)

CHANGE_KEYS: frozenset[str] = frozenset({"issue_id", "field", "before", "after"})
RECORD_KEYS: frozenset[str] = frozenset(
    {
        "step",
        "tool",
        "args",
        "ok",
        "result",
        "error",
        "state_hash_before",
        "state_hash_after",
        "changes",
    }
)


class SnapshotError(ValueError):
    """Snapshot 格式錯、版本不符、內容不合法，或 hash 與內容不一致。"""


@dataclass(frozen=True)
class Change:
    """一筆以 issue 為中心的狀態變更。

    - 修改 issue 欄位：``field`` ∈ MUTABLE_ISSUE_FIELDS，``before`` / ``after`` 為 JSON-native 值
      （labels 為排序後的 list）。
    - 新增 comment：``field == "comments"``，``before is None``，``after`` 為 comment dict
      ``{"id", "issue_id", "author_id", "body"}``，``issue_id`` 為該 comment 所屬 issue。
    """

    issue_id: int
    field: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        """剛好回傳 ``{"issue_id", "field", "before", "after"}``；before / after 為深拷貝。"""
        return {
            "issue_id": self.issue_id,
            "field": self.field,
            "before": copy.deepcopy(self.before),
            "after": copy.deepcopy(self.after),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Change:
        """key 集合必須剛好是上述四個；issue_id 為 ``type(x) is int``；
        field ∈ MUTABLE_ISSUE_FIELDS ∪ {"comments"}；field 為 "comments" 時 before 必須是 None。
        data 不是 dict、或不符合上述條件時 raise ValueError。before / after 深拷貝。
        """
        if not isinstance(data, dict) or set(data) != CHANGE_KEYS:
            raise ValueError("change must be an object with keys issue_id, field, before, after")
        if type(data["issue_id"]) is not int:
            raise ValueError("change.issue_id must be an integer")
        if not isinstance(data["field"], str) or data["field"] not in CHANGE_FIELDS:
            raise ValueError(f"invalid change.field {data['field']!r}")
        if data["field"] == COMMENTS_FIELD and data["before"] is not None:
            raise ValueError("a comments change must have before = null")
        return cls(
            issue_id=data["issue_id"],
            field=data["field"],
            before=copy.deepcopy(data["before"]),
            after=copy.deepcopy(data["after"]),
        )


def diff_states(before: IssueTrackerState, after: IssueTrackerState) -> tuple[Change, ...]:
    """計算以 issue 為中心的 diff。

    輸出順序：先依 issue_id 遞增；同一 issue 內先依 MUTABLE_ISSUE_FIELDS 的順序列出欄位變更，
    再依 comment id 遞增列出新增的 comments。

    以下變化不可能由 tools 產生，出現時代表資料損毀，raise ValueError：
    users / labels 有任何差異；issue 集合不同；reporter_id 或 created_at 改變；
    既有 comment 被刪除或修改。
    """
    if before.users != after.users:
        raise ValueError("users changed; tools cannot modify users")
    if before.labels != after.labels:
        raise ValueError("labels changed; tools cannot modify labels")
    if set(before.issues) != set(after.issues):
        raise ValueError("issue set changed; tools cannot create or delete issues")
    for comment_id, comment in before.comments.items():
        if after.comments.get(comment_id) != comment:
            raise ValueError(f"comment {comment_id} was modified or deleted")

    new_comments: dict[int, list[dict[str, Any]]] = {}
    for comment_id in sorted(set(after.comments) - set(before.comments)):
        comment = after.comments[comment_id]
        if comment.issue_id not in after.issues:
            raise ValueError(f"comment {comment_id} refers to unknown issue {comment.issue_id}")
        new_comments.setdefault(comment.issue_id, []).append(
            {"id": comment.id, "issue_id": comment.issue_id, "author_id": comment.author_id, "body": comment.body}
        )

    changes: list[Change] = []
    for issue_id in sorted(before.issues):
        old, new = before.issues[issue_id], after.issues[issue_id]
        if old.id != new.id or old.reporter_id != new.reporter_id or old.created_at != new.created_at:
            raise ValueError(f"immutable field of issue {issue_id} changed")
        for field in MUTABLE_ISSUE_FIELDS:
            old_value, new_value = getattr(old, field), getattr(new, field)
            if field == "labels":
                old_value, new_value = sorted(set(old_value)), sorted(set(new_value))
            if old_value != new_value:
                changes.append(Change(issue_id, field, old_value, new_value))
        for comment in new_comments.get(issue_id, []):
            changes.append(Change(issue_id, COMMENTS_FIELD, None, comment))
    return tuple(changes)


def state_hash_of(state: IssueTrackerState, actor_id: str) -> str:
    """``sha256_of({"schema_version": SCHEMA_VERSION, "actor_id": actor_id, "state": state.to_dict()})``。"""
    return sha256_of({"schema_version": SCHEMA_VERSION, "actor_id": actor_id, "state": state.to_dict()})


@dataclass(frozen=True)
class ToolCallRecord:
    """Env 對每一次 tool call 的決定性紀錄（Recorder 在 Week 1 會在外面再包 model message 等）。"""

    step: int  # 該 call 之前的 env.step
    tool: str
    args: Any  # 呼叫時參數的深拷貝（任意 JSON 值；不是 dict 的也照存，該 call 為 invalid_arguments）
    ok: bool
    result: dict[str, Any] | None  # 成功時的 data；失敗為 None
    error: dict[str, str] | None  # 失敗時剛好 {"code", "message"}；成功為 None
    state_hash_before: str
    state_hash_after: str
    changes: tuple[Change, ...]  # read-only call 與失敗的 call 一定是空 tuple

    def observation(self) -> dict[str, Any]:
        """重建 agent 當時看到的 observation（等於當時 ``call_tool`` 的回傳值）：
        ok 時 ``{"ok": True, "data": deepcopy(result)}``，否則 ``{"ok": False, "error": deepcopy(error)}``。
        """
        if self.ok:
            return {"ok": True, "data": copy.deepcopy(self.result)}
        return {"ok": False, "error": copy.deepcopy(self.error)}

    def to_dict(self) -> dict[str, Any]:
        """剛好回傳 9 個 key（即 dataclass 欄位名）；changes 為 ``[Change.to_dict(), ...]``；
        回傳深拷貝，可通過 canonical_json。不含 observation。
        """
        return {
            "step": self.step,
            "tool": self.tool,
            "args": copy.deepcopy(self.args),
            "ok": self.ok,
            "result": copy.deepcopy(self.result),
            "error": copy.deepcopy(self.error),
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "changes": [c.to_dict() for c in self.changes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallRecord:
        """key 集合必須剛好是 9 個欄位名；step 為非 bool 的 int；
        ok=True ⇔ result 是 dict 且 error 是 None；ok=False ⇔ result 是 None 且 error 剛好有
        {"code", "message"}；changes 必須是 list，每個元素都能被 Change.from_dict 解析，轉回
        ``tuple[Change, ...]``。data 不是 dict 或不符合上述條件時 raise ValueError。
        """
        if not isinstance(data, dict) or set(data) != RECORD_KEYS:
            raise ValueError("record must be an object with exactly the ToolCallRecord fields")
        if type(data["step"]) is not int:
            raise ValueError("record.step must be an integer")
        ok, result, error = data["ok"], data["result"], data["error"]
        if ok is True:
            if not isinstance(result, dict) or error is not None:
                raise ValueError("ok record must have a result object and no error")
        elif ok is False:
            if result is not None or not isinstance(error, dict) or set(error) != {"code", "message"}:
                raise ValueError("failed record must have no result and an error with code and message")
        else:
            raise ValueError("record.ok must be a boolean")
        if not isinstance(data["changes"], list):
            raise ValueError("record.changes must be a list")
        return cls(
            step=data["step"],
            tool=data["tool"],
            args=copy.deepcopy(data["args"]),
            ok=ok,
            result=copy.deepcopy(result),
            error=copy.deepcopy(error),
            state_hash_before=data["state_hash_before"],
            state_hash_after=data["state_hash_after"],
            changes=tuple(Change.from_dict(c) for c in data["changes"]),
        )


class IssueTrackerEnv:
    """單一 Issue Tracker 的 deterministic environment。

    未初始化（尚未成功 reset / restore）時，``call_tool``、``snapshot``、``state_hash``、
    ``current_state`` 以及屬性 ``step``、``actor_id``、``records`` 都 raise RuntimeError；
    ``tool_specs``（staticmethod）不受影響；``reset`` 與 ``restore`` 可直接呼叫。
    """

    def __init__(self) -> None:
        self._state: IssueTrackerState | None = None
        self._actor_id: str = ""
        self._step = 0
        self._records: list[ToolCallRecord] = []

    def _require_initialized(self) -> IssueTrackerState:
        if self._state is None:
            raise RuntimeError("environment is not initialized; call reset() or restore() first")
        return self._state

    @staticmethod
    def tool_specs() -> tuple[dict[str, Any], ...]:
        """回傳 TOOL_SPECS 的深拷貝。"""
        return copy.deepcopy(TOOL_SPECS)

    def reset(self, initial_state: IssueTrackerState | dict[str, Any], actor_id: str) -> None:
        """載入初始狀態，step 歸零，清空 records。

        initial_state 不論是 dict 或 IssueTrackerState，一律以
        ``IssueTrackerState.from_dict(x if isinstance(x, dict) else x.to_dict())`` 驗證並正規化
        （SchemaError 原樣拋出）；actor_id 必須是既有 user，否則 raise ValueError。
        失敗時 env 維持呼叫前的狀態。不修改、也不與 initial_state 共用可變物件。
        """
        data = initial_state.to_dict() if isinstance(initial_state, IssueTrackerState) else initial_state
        state = IssueTrackerState.from_dict(data)
        if not isinstance(actor_id, str) or actor_id not in state.users:
            raise ValueError(f"actor {_describe(actor_id)} is not a user in the initial state")
        self._state = state
        self._actor_id = actor_id
        self._step = 0
        self._records = []

    def call_tool(self, name: str, args: Any) -> dict[str, Any]:
        """執行 tool 並記錄 ToolCallRecord，回傳 observation。

        前置條件（違反屬於 runner 的錯誤，不是 agent 行為）：name 必須是 str（否則 TypeError）且可 UTF-8
        編碼（否則 ValueError）；args 必須是 JSON 值——``canonical_json(args)`` 不 raise、可 UTF-8 編碼、
        不含 tuple、巢狀深度不超過 MAX_ARGS_DEPTH——否則 raise TypeError 或 ValueError。
        違反前置條件時不記錄、step 不變、state 不變。

        args 是 JSON 值但不是 dict（list / str / int / None 等）時，照常記錄一筆
        ok=False、code 為 invalid_arguments 的 record，args 原樣保存。

        observation 形狀：成功 ``{"ok": True, "data": {...}}``；失敗
        ``{"ok": False, "error": {"code": ..., "message": ...}}``。
        回傳值是深拷貝；呼叫端修改它不會影響 env 或 records。
        不論成功或失敗（ToolError），step 都 +1。

        tool 在 state 的拷貝上執行，hash 與 diff_states 都成功後才提交；若出現 ToolError 以外的例外，
        例外照常往外拋，env 的 state、step、records 都不變。
        """
        state = self._require_initialized()
        check_call_preconditions(name, args)

        recorded_args = copy.deepcopy(args)
        hash_before = state_hash_of(state, self._actor_id)
        work = state.copy()
        try:
            data = execute_tool(work, self._actor_id, name, copy.deepcopy(args))
        except ToolError as exc:
            record = ToolCallRecord(
                step=self._step,
                tool=name,
                args=recorded_args,
                ok=False,
                result=None,
                error={"code": exc.code, "message": exc.message},
                state_hash_before=hash_before,
                state_hash_after=hash_before,
                changes=(),
            )
            new_state = state
        else:
            changes = diff_states(state, work)
            record = ToolCallRecord(
                step=self._step,
                tool=name,
                args=recorded_args,
                ok=True,
                result=data,
                error=None,
                state_hash_before=hash_before,
                state_hash_after=state_hash_of(work, self._actor_id),
                changes=changes,
            )
            new_state = work

        self._state = new_state
        self._records.append(record)
        self._step += 1
        return record.observation()

    def snapshot(self) -> dict[str, Any]:
        """回傳 JSON-native 的 snapshot::

            {"format": SNAPSHOT_FORMAT, "schema_version": SCHEMA_VERSION,
             "actor_id": ..., "step": env.step, "state": state.to_dict(), "state_hash": env.state_hash()}

        ``snapshot["step"]`` 即下一筆 record 的 step。
        Snapshot 只包含 environment 狀態，不包含 records（trajectory 屬於 Recorder）。
        回傳值與 env 不共用可變物件。
        """
        state = self._require_initialized()
        return {
            "format": SNAPSHOT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "actor_id": self._actor_id,
            "step": self._step,
            "state": state.to_dict(),
            "state_hash": state_hash_of(state, self._actor_id),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """從 snapshot 還原 environment。尚未初始化的 env 也可以直接 restore。

        所有驗證都在修改 env 之前完成，驗證項目：
        - snapshot 是 dict，key 集合剛好是 SNAPSHOT_KEYS；
        - format == SNAPSHOT_FORMAT；schema_version == SCHEMA_VERSION；
        - step 是非 bool 的 int 且 >= 0；
        - state 可被 from_dict 解析，且是 canonical 形式（``from_dict(s).to_dict() == s``）；
        - actor_id 是 state 中既有的 user；
        - ``state_hash_of(parsed, actor_id) == snapshot["state_hash"]``。
        任何失敗都 raise SnapshotError（SchemaError 以 ``raise SnapshotError(...) from e`` 包裝），
        且 env 的 state / step / actor_id / records 完全不變（未初始化的仍維持未初始化）。

        成功後 ``env.snapshot() == snapshot``、``env.records == ()``，且 env 不與 snapshot 共用可變物件
        （之後修改傳入的 snapshot 不影響 env）。
        restore 與之後的 call_tool 都不修改傳入的 snapshot；同一份 snapshot 重複 restore 後
        執行相同的 call 序列，得到的 records 與 hash 完全相同。
        """
        if not isinstance(snapshot, dict):
            raise SnapshotError("snapshot must be an object")
        if set(snapshot) != SNAPSHOT_KEYS:
            raise SnapshotError("snapshot keys must be exactly: " + ", ".join(sorted(SNAPSHOT_KEYS)))
        if snapshot["format"] != SNAPSHOT_FORMAT:
            raise SnapshotError(f"unsupported snapshot format {_describe(snapshot['format'])}")
        if snapshot["schema_version"] != SCHEMA_VERSION:
            raise SnapshotError(f"unsupported schema version {_describe(snapshot['schema_version'])}")
        step = snapshot["step"]
        if type(step) is not int or step < 0:
            raise SnapshotError("snapshot step must be a non-negative integer")
        try:
            state = IssueTrackerState.from_dict(snapshot["state"])
        except SchemaError as exc:
            raise SnapshotError(f"invalid snapshot state: {exc}") from exc
        if state.to_dict() != snapshot["state"]:
            raise SnapshotError("snapshot state is not in canonical form")
        actor_id = snapshot["actor_id"]
        if not isinstance(actor_id, str) or actor_id not in state.users:
            raise SnapshotError(f"snapshot actor {_describe(actor_id)} is not a user in the state")
        try:
            computed_hash = state_hash_of(state, actor_id)
        except ValueError as exc:  # 例如 actor_id 含 lone surrogate，無法以 UTF-8 編碼
            raise SnapshotError(f"snapshot state cannot be hashed: {exc}") from exc
        if computed_hash != snapshot["state_hash"]:
            raise SnapshotError("snapshot state_hash does not match its content")
        self._state = state
        self._actor_id = actor_id
        self._step = step
        self._records = []

    def state_hash(self) -> str:
        """``state_hash_of(目前 state, actor_id)``。"""
        return state_hash_of(self._require_initialized(), self._actor_id)

    @property
    def step(self) -> int:
        self._require_initialized()
        return self._step

    @property
    def actor_id(self) -> str:
        self._require_initialized()
        return self._actor_id

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        """自上次 reset / restore 以來的紀錄。回傳深拷貝：修改其內容不影響 env。"""
        self._require_initialized()
        return copy.deepcopy(tuple(self._records))

    def last_records(self, count: int) -> tuple[ToolCallRecord, ...]:
        """自上次 reset / restore 以來最後 count 筆紀錄的深拷貝（依時間順序）。

        ``type(count) is int`` 且 ``0 <= count <= len(records)``，否則 ValueError；未初始化 → RuntimeError。
        供 runner 每個 turn 取得新紀錄，不必深拷貝整份 records。
        """
        self._require_initialized()
        if type(count) is not int or not 0 <= count <= len(self._records):
            raise ValueError(f"count must be an integer between 0 and {len(self._records)}")
        if count == 0:
            return ()
        return copy.deepcopy(tuple(self._records[-count:]))

    def current_state(self) -> IssueTrackerState:
        """目前 state 的深拷貝。"""
        return self._require_initialized().copy()


def check_call_preconditions(name: Any, args: Any) -> None:
    """``call_tool`` 的前置條件（見其 docstring）；違反時 raise TypeError 或 ValueError。

    公開給 agent 端（例如 ToolCall 驗證）使用，確保 policy 產生的 tool call 一定能被 env 記錄。
    """
    if not isinstance(name, str):
        raise TypeError(f"tool name must be a str, got {type(name).__name__}")
    name.encode("utf-8")
    _require_json_value(args)
    canonical_json(args).encode("utf-8")


def _describe(value: Any) -> str:
    """錯誤訊息用：str 顯示 repr，其他型別只顯示型別名（避免超大 int 的 repr 失敗）。"""
    return repr(value) if isinstance(value, str) else f"<{type(value).__name__}>"


def _require_json_value(value: Any, depth: int = 0) -> None:
    """call_tool 前置條件：只接受 JSON-native 值（不含 tuple、key 為 str、float 有限、深度有上限）。"""
    if depth > MAX_ARGS_DEPTH:
        raise ValueError(f"tool arguments must not be nested deeper than {MAX_ARGS_DEPTH} levels")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tool arguments must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("tool argument object keys must be str")
            _require_json_value(item, depth + 1)
        return
    raise TypeError(f"tool arguments must be JSON values, got {type(value).__name__}")
