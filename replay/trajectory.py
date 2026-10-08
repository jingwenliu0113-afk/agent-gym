"""Recorder 的 trajectory 格式（decision log 006）。

一個 turn = 一次 model 呼叫（``message``）+ 該訊息每個 tool call 的 env 紀錄（``tool_steps``，與
``message.tool_calls`` 一一對應）+ 該次呼叫的量測（``usage``，model_turns 固定為 1；``meta``，provider 回報的量測，
例如 response.model、request id、stop_details，None 或 JSON 物件）。
``message`` 與 ``tool_steps`` 是決定性內容；``usage`` 與 ``meta`` 是量測，不進 conversation。

本模組所有「相同 / 相符」一律以 ``canonical_json`` 逐 byte 比較（不用 Python ``==``）；所有「int」一律 ``type(x) is int``。

Trajectory（``agent-gym/trajectory/v2``）的結構規則（``validate``，違反 raise TrajectoryError）：
1. task 可被 ``Task.from_dict`` 解析，且 ``canonical_json(Task.from_dict(task).to_dict()) == canonical_json(task)``；
   agent 為 AgentConfig；max_turns 為 int >= 1。
2. 1 <= len(turns) <= max_turns；每個元素是 Turn 且 ``turns[i].index == i``（int）。
3. 每個 turn：len(tool_steps) == len(message.tool_calls)；第 j 個 step 的 tool == 第 j 個 call 的 name，
   且 ``canonical_json(step.args) == canonical_json(call.args)``；``usage.model_turns == 1``。
4. 所有 tool steps 的 step 編號跨 turn 從 0 連續遞增。
5. 除最後一個 turn 外，每個 turn 都至少有一個 tool call。
6. termination：最後一個 turn 沒有 tool call ⇔ "final_answer"；否則必須 len(turns) == max_turns 且為 "max_turns"。
7. final_snapshot 可被全新的 ``IssueTrackerEnv().restore`` 接受（SnapshotError 包成 TrajectoryError），
   其 step（int）== tool steps 總數、actor_id == task 的 actor_id。
8. parent 為 None 或剛好 ``{"trajectory_id": str, "checkpoint_id": str, "turn": int}``，且 0 <= turn < len(turns)。
9. ``canonical_json(self.to_dict()).encode("utf-8")`` 成功（所有字串可 UTF-8 編碼）。
10. task、final_snapshot、parent、每個 turn 的 meta 與每個 tool step 的 args / result / error / changes 都是 JSON 文件：容器只能是
    ``list`` / ``dict``（不含 tuple）、key 為 str、葉節點為 None / str / bool / int / float，巢狀深度不超過
    MAX_DOCUMENT_DEPTH（decision 007）。``from_dict`` / ``load`` 在任何深拷貝之前先以同一規則檢查整份輸入。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from agents.policy import AgentConfig, ModelMessage
from env.environment import IssueTrackerEnv, SnapshotError, ToolCallRecord
from env.hashing import canonical_json, sha256_of
from grader.grader import grade as grade_records
from grader.schema import GradeReport, Usage
from tasks.schema import Task

TRAJECTORY_FORMAT = "agent-gym/trajectory/v2"  # v2：ModelMessage.provider_content（decision 009）
TERMINATIONS: tuple[str, ...] = ("final_answer", "max_turns")

TRAJECTORY_KEYS: tuple[str, ...] = (
    "format",
    "task",
    "agent",
    "max_turns",
    "turns",
    "termination",
    "final_snapshot",
    "parent",
)
PARENT_KEYS: tuple[str, ...] = ("trajectory_id", "checkpoint_id", "turn")
TURN_KEYS: tuple[str, ...] = ("index", "message", "tool_steps", "usage", "meta")

# trajectory / checkpoint 文件的巢狀深度上限（以文件根為第 0 層；decision 007）。合法內容最深約為
# agent.model_config + MAX_MODEL_CONFIG_DEPTH（約 130 層），且 256 層的 deepcopy 遠低於 Python 預設的遞迴上限。
MAX_DOCUMENT_DEPTH = 256


class TrajectoryError(ValueError):
    """Trajectory / turn 資料不合法，或重放結果與紀錄不符。"""


@dataclass(frozen=True)
class Turn:
    index: int
    message: ModelMessage
    tool_steps: tuple[ToolCallRecord, ...]
    usage: Usage
    meta: dict[str, Any] | None = None  # 量測（decision 009）；None 或 JSON 物件

    def to_dict(self) -> dict[str, Any]:
        """剛好 ``{"index", "message", "tool_steps", "usage", "meta"}``，巢狀物件用各自的 to_dict，meta 為深拷貝。"""
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "tool_steps": [step.to_dict() for step in self.tool_steps],
            "usage": self.usage.to_dict(),
            "meta": copy.deepcopy(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        """data 必須是 dict 且 key 剛好為上述五個、index 為 int、tool_steps 為 list、meta 為 None 或 dict；
        巢狀物件以各自的 from_dict 解析。任何錯誤一律 raise TrajectoryError（以 ``from e`` 包裝內層例外）。
        """
        check_json_document(data, error=TrajectoryError)
        if not isinstance(data, dict) or set(data) != set(TURN_KEYS):
            raise TrajectoryError("turn must be an object with keys: " + ", ".join(TURN_KEYS))
        if type(data["index"]) is not int:
            raise TrajectoryError("turn.index must be an integer")
        if not isinstance(data["tool_steps"], list):
            raise TrajectoryError("turn.tool_steps must be a list")
        if data["meta"] is not None and type(data["meta"]) is not dict:
            raise TrajectoryError("turn.meta must be null or an object")
        try:
            message = ModelMessage.from_dict(data["message"])
            steps = tuple(ToolCallRecord.from_dict(step) for step in data["tool_steps"])
            usage = Usage.from_dict(data["usage"])
        except (TypeError, ValueError) as exc:
            raise TrajectoryError(f"invalid turn: {exc}") from exc
        return cls(index=data["index"], message=message, tool_steps=steps, usage=usage, meta=copy.deepcopy(data["meta"]))


def render_messages(instruction: str, agent: AgentConfig, turns: Sequence[Turn]) -> list[dict[str, Any]]:
    """Agent 在這些 turns 之後看到的 conversation（形狀見 ``agents/policy.py``）。

    順序：``{"role": "system", "content": agent.system_prompt}`` → ``{"role": "developer", "content": agent.developer_prompt}``
    → ``{"role": "user", "content": instruction}`` → 每個 turn：
    ``{"role": "assistant", "content": message.text, "tool_calls": [call.to_dict(), ...],
    "provider_content": message.provider_content 的深拷貝或 None}`` → 每個 tool call 一則
    ``{"role": "tool", "tool_call_id": call.id, "content": canonical_json(step.observation())}``。
    每個 dict 的 key 依上列順序建立；即使 prompt 為空字串也輸出該訊息。回傳 JSON-native 的新物件。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "developer", "content": agent.developer_prompt},
        {"role": "user", "content": instruction},
    ]
    for turn in turns:
        calls = turn.message.tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": turn.message.text,
                "tool_calls": [call.to_dict() for call in calls],
                "provider_content": copy.deepcopy(turn.message.provider_content),
            }
        )
        for call, step in zip(calls, turn.tool_steps):
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": canonical_json(step.observation())}
            )
    return messages


def check_turn_sequence(turns: Sequence[Any], *, error: type[Exception], all_have_tool_calls: bool) -> int:
    """Trajectory 規則 2–5 中與 turns 本身有關的部分（不含長度上限與 termination）；回傳 tool steps 總數。

    - 每個元素是 Turn，各欄位型別正確，``type(index) is int`` 且 ``turns[i].index == i``；
    - 規則 3：tool steps 與 tool calls 一一對應、名稱相同、args 逐 byte 相同、``usage.model_turns == 1``；
    - 規則 4：tool steps 的 step 從 0 連續；
    - 規則 5：all_have_tool_calls 為 True 時每個 turn 都要有 tool call，否則除最後一個以外都要有。
    違反時 raise ``error``。供 Trajectory.validate、Checkpoint.verify 與 runner 的入口檢查共用。
    """
    if not isinstance(turns, (list, tuple)):
        raise error("turns must be a list or tuple")
    next_step = 0
    for i, turn in enumerate(turns):
        if not isinstance(turn, Turn):
            raise error(f"turns[{i}] is not a Turn")
        if type(turn.index) is not int or turn.index != i:
            raise error(f"turns[{i}].index must be {i}")
        if not isinstance(turn.message, ModelMessage) or not isinstance(turn.usage, Usage):
            raise error(f"turns[{i}] has an invalid message or usage")
        if turn.meta is not None and type(turn.meta) is not dict:
            raise error(f"turns[{i}].meta must be None or a dict")
        if not isinstance(turn.tool_steps, (list, tuple)) or not all(
            isinstance(step, ToolCallRecord) for step in turn.tool_steps
        ):
            raise error(f"turns[{i}].tool_steps must contain ToolCallRecord objects")
        calls = turn.message.tool_calls
        if len(turn.tool_steps) != len(calls):
            raise error(f"turns[{i}] has {len(calls)} tool calls but {len(turn.tool_steps)} tool steps")
        for j, (call, step) in enumerate(zip(calls, turn.tool_steps)):
            if step.tool != call.name or not _same_json(step.args, call.args):
                raise error(f"turns[{i}].tool_steps[{j}] does not match its tool call")
            if type(step.step) is not int or step.step != next_step:
                raise error(f"turns[{i}].tool_steps[{j}].step must be {next_step}")
            next_step += 1
        if type(turn.usage.model_turns) is not int or turn.usage.model_turns != 1:
            raise error(f"turns[{i}].usage.model_turns must be 1")
        is_last = i == len(turns) - 1
        if not calls and (all_have_tool_calls or not is_last):
            raise error(f"turns[{i}] has no tool calls but is not the final turn")
    return next_step


def check_json_document(value: Any, *, error: type[Exception], max_depth: int = MAX_DOCUMENT_DEPTH) -> None:
    """以迭代（非遞迴）方式檢查 JSON 文件：容器只能是 list / dict、key 為 str、葉節點為 None / str / bool / int / float，
    巢狀深度不超過 max_depth；違反時 raise ``error``。在任何深拷貝之前呼叫，避免 RecursionError 漏出。
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > max_depth:
            raise error(f"document must not be nested deeper than {max_depth} levels")
        if type(item) is dict:
            for key, child in item.items():
                if not isinstance(key, str):
                    raise error("document object keys must be strings")
                stack.append((child, depth + 1))
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif not (item is None or type(item) in (str, bool, int, float)):
            raise error(f"document contains a non-JSON value of type {type(item).__name__}")


def check_turn_payloads(turns: Any, *, error: type[Exception], turns_depth: int = 1) -> None:
    """對 turns 中每個 turn 的 meta 與每個 tool step 的 args / result / error / changes 執行 check_json_document
    （非 Turn 元素略過，交給 check_turn_sequence 處理）。必須在任何對這些內容做遞迴處理（canonical_json、deepcopy）之前呼叫。
    深度上限以**文件根**計算（與 from_dict / load 一致）：turns list 位於文件第 ``turns_depth`` 層，因此 meta 在
    turns_depth + 2 層、tool step 的 args / result / error 在 + 4 層、change 的 before / after 在 + 6 層。
    """
    if not isinstance(turns, (list, tuple)):
        return
    for turn in turns:
        if not isinstance(turn, Turn):
            continue
        check_json_document(turn.meta, error=error, max_depth=MAX_DOCUMENT_DEPTH - turns_depth - 2)
        if not isinstance(turn.tool_steps, (list, tuple)):
            continue
        payload_limit = MAX_DOCUMENT_DEPTH - turns_depth - 4
        change_limit = MAX_DOCUMENT_DEPTH - turns_depth - 6
        for step in turn.tool_steps:
            if not isinstance(step, ToolCallRecord):
                continue
            check_json_document(step.args, error=error, max_depth=payload_limit)
            check_json_document(step.result, error=error, max_depth=payload_limit)
            check_json_document(step.error, error=error, max_depth=payload_limit)
            if isinstance(step.changes, (list, tuple)):
                for change in step.changes:
                    check_json_document(getattr(change, "before", None), error=error, max_depth=change_limit)
                    check_json_document(getattr(change, "after", None), error=error, max_depth=change_limit)


def same_canonical(a: Any, b: Any) -> bool:
    """canonical_json 逐 byte 相同；任一方無法序列化視為不同。"""
    return _same_json(a, b)


def _same_json(a: Any, b: Any) -> bool:
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):
        return False


def replay_tool_steps(task: Task, steps: Sequence[ToolCallRecord], *, error: type[Exception]) -> IssueTrackerEnv:
    """從 task 初始狀態以新 env 重放 steps；每筆 record 必須與保存的逐 byte 相同，否則 raise ``error``。回傳重放後的 env。"""
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    for index, step in enumerate(steps):
        try:
            env.call_tool(step.tool, step.args)
        except (TypeError, ValueError) as exc:
            raise error(f"tool step {index} cannot be replayed: {exc}") from exc
    for index, (step, replayed) in enumerate(zip(steps, env.records)):
        try:
            saved = step.to_dict()
        except Exception as exc:  # 保存的 record 不可信：任何無法序列化的內容都代表與重放不同
            raise error(f"tool step {index} cannot be serialized: {exc}") from exc
        if not _same_json(saved, replayed.to_dict()):
            raise error(f"tool step {index} does not match its deterministic replay")
    return env


@dataclass(frozen=True)
class Trajectory:
    task: dict[str, Any]  # Task.to_dict()
    agent: AgentConfig
    max_turns: int
    turns: tuple[Turn, ...]
    termination: str
    final_snapshot: dict[str, Any]  # 最後的 env snapshot
    parent: dict[str, Any] | None  # rerun 時：{"trajectory_id", "checkpoint_id", "turn"}

    def validate(self) -> None:
        """結構規則（模組 docstring 1–10）；違反 raise TrajectoryError。不做重放。"""
        # 深度以文件根計算：task / final_snapshot / parent 位於 to_dict() 的第 1 層（與 from_dict / load 的檢查一致）
        check_json_document(self.task, error=TrajectoryError, max_depth=MAX_DOCUMENT_DEPTH - 1)
        check_json_document(self.final_snapshot, error=TrajectoryError, max_depth=MAX_DOCUMENT_DEPTH - 1)
        check_json_document(self.parent, error=TrajectoryError, max_depth=MAX_DOCUMENT_DEPTH - 1)
        check_turn_payloads(self.turns, error=TrajectoryError)
        task = parse_canonical_task(self.task, error=TrajectoryError)
        if not isinstance(self.agent, AgentConfig):
            raise TrajectoryError("agent must be an AgentConfig")
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise TrajectoryError("max_turns must be an integer >= 1")
        if not isinstance(self.turns, (list, tuple)) or not 1 <= len(self.turns) <= self.max_turns:
            raise TrajectoryError("a trajectory must have between 1 and max_turns turns")
        total_steps = check_turn_sequence(self.turns, error=TrajectoryError, all_have_tool_calls=False)

        if self.termination not in TERMINATIONS:
            raise TrajectoryError(f"termination must be one of: {', '.join(TERMINATIONS)}")
        if self.turns[-1].message.tool_calls:
            if self.termination != "max_turns" or len(self.turns) != self.max_turns:
                raise TrajectoryError("a trajectory whose last turn has tool calls must end by max_turns")
        elif self.termination != "final_answer":
            raise TrajectoryError("a trajectory whose last turn has no tool calls must end by final_answer")

        if not isinstance(self.final_snapshot, dict):
            raise TrajectoryError("final_snapshot must be a dict")
        try:
            IssueTrackerEnv().restore(copy.deepcopy(self.final_snapshot))
        except (SnapshotError, TypeError, ValueError) as exc:
            raise TrajectoryError(f"final_snapshot cannot be restored: {exc}") from exc
        if self.final_snapshot["step"] != total_steps:
            raise TrajectoryError("final_snapshot step does not match the number of tool steps")
        if self.final_snapshot["actor_id"] != task.actor_id:
            raise TrajectoryError("final_snapshot actor_id does not match the task actor")

        if self.parent is not None:
            check_parent(self.parent, error=TrajectoryError)
            if not 0 <= self.parent["turn"] < len(self.turns):
                raise TrajectoryError("parent.turn must be in [0, len(turns))")
        try:
            canonical_json(self.to_dict()).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TrajectoryError(f"trajectory is not serializable as UTF-8 JSON: {exc}") from exc

    def verify(self) -> None:
        """validate() 之後，從 task 初始狀態重放所有 tool steps：每筆 record 須與紀錄逐 byte 相同
        （canonical_json），且 ``canonical_json(重放後 env.snapshot()) == canonical_json(final_snapshot)``。
        不符 raise TrajectoryError。
        """
        self.validate()
        env = replay_tool_steps(Task.from_dict(self.task), self.records(), error=TrajectoryError)
        if not _same_json(env.snapshot(), self.final_snapshot):
            raise TrajectoryError("final_snapshot does not match the replayed environment")

    def records(self) -> tuple[ToolCallRecord, ...]:
        """依序攤平所有 turns 的 tool steps（深拷貝）。"""
        return copy.deepcopy(tuple(step for turn in self.turns for step in turn.tool_steps))

    def messages(self) -> list[dict[str, Any]]:
        """``render_messages(task instruction, agent, turns)``，即最後一次 model 呼叫之後的完整 conversation。"""
        return render_messages(self.task["instruction"], self.agent, self.turns)

    def usage(self) -> Usage:
        """episode 用量：input_tokens / output_tokens / latency_s / cost_usd 依 turn 順序以內建 ``sum``（start=0）加總，
        任一 turn 為 None 則該欄為 None（全部為 int 時結果為 int）；model_turns = len(turns)。
        例：ScriptedPolicy（tokens 0、cost 0.0）+ ``clock=lambda: 0.0`` → ``Usage(0, 0, 0.0, 0.0, len(turns))``。
        """

        def total(field: str) -> Any:
            values = [getattr(turn.usage, field) for turn in self.turns]
            return None if any(v is None for v in values) else sum(values)

        return Usage(
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            latency_s=total("latency_s"),
            cost_usd=total("cost_usd"),
            model_turns=len(self.turns),
        )

    def grade(self) -> GradeReport:
        """``grade(Task.from_dict(task), records(), final_snapshot["state"], usage())``。
        從 checkpoint rerun 的評分一律用 rerun 回傳之 trajectory 的 grade()（前綴已由 checkpoint 綁定）。
        """
        return grade_records(Task.from_dict(self.task), self.records(), self.final_snapshot["state"], self.usage())

    @property
    def trajectory_id(self) -> str:
        """``sha256_of(self.to_dict())``。這是**紀錄 id**（包含 usage / latency 等量測），不是行為 id；
        實驗層不得以它去重或當作 rollout 的 key。
        """
        return sha256_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """剛好 TRAJECTORY_KEYS（``format`` = TRAJECTORY_FORMAT），JSON-native、深拷貝。"""
        return {
            "format": TRAJECTORY_FORMAT,
            "task": copy.deepcopy(self.task),
            "agent": self.agent.to_dict(),
            "max_turns": self.max_turns,
            "turns": [turn.to_dict() for turn in self.turns],
            "termination": self.termination,
            "final_snapshot": copy.deepcopy(self.final_snapshot),
            "parent": copy.deepcopy(self.parent),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        """data 必須是 dict、key 剛好為 TRAJECTORY_KEYS、format 相符；解析巢狀物件後呼叫 validate()。
        任何錯誤一律 raise TrajectoryError。保證 ``Trajectory.from_dict(t.to_dict()) == t``。
        """
        check_json_document(data, error=TrajectoryError)
        if not isinstance(data, dict) or set(data) != set(TRAJECTORY_KEYS):
            raise TrajectoryError("trajectory must be an object with keys: " + ", ".join(TRAJECTORY_KEYS))
        if data["format"] != TRAJECTORY_FORMAT:
            raise TrajectoryError("unsupported trajectory format")
        try:
            agent = AgentConfig.from_dict(data["agent"])
        except ValueError as exc:
            raise TrajectoryError(f"invalid agent: {exc}") from exc
        if not isinstance(data["turns"], list):
            raise TrajectoryError("trajectory turns must be a list")
        turns = tuple(Turn.from_dict(turn) for turn in data["turns"])
        try:
            trajectory = cls(
                task=copy.deepcopy(data["task"]),
                agent=agent,
                max_turns=data["max_turns"],
                turns=turns,
                termination=data["termination"],
                final_snapshot=copy.deepcopy(data["final_snapshot"]),
                parent=copy.deepcopy(data["parent"]),
            )
        except TypeError as exc:
            raise TrajectoryError(f"invalid trajectory: {exc}") from exc
        trajectory.validate()
        return trajectory

    def save(self, path: str | Path) -> None:
        """寫入 ``canonical_json(self.to_dict()) + "\\n"``（UTF-8）。"""
        Path(path).write_bytes((canonical_json(self.to_dict()) + "\n").encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        """以 strict UTF-8 讀取，``json.loads`` 拒絕 NaN / Infinity 字面量與重複 key；解碼 / 解析錯誤包成
        TrajectoryError，OSError 原樣拋出；結尾換行非必要。保證 ``Trajectory.load(p) == t``（p 為 ``t.save(p)`` 的輸出）。
        """
        raw = Path(path).read_bytes()
        try:
            data = load_strict_json(raw)
        except ValueError as exc:
            raise TrajectoryError(f"cannot parse trajectory file: {exc}") from exc
        return cls.from_dict(data)


def load_strict_json(text_bytes: bytes) -> Any:
    """strict UTF-8 解碼 + ``json.loads``（NaN / Infinity / -Infinity 與重複 key 都 raise ValueError）。
    供 Trajectory.load 與 Checkpoint.load 共用；UnicodeDecodeError / JSONDecodeError 都是 ValueError。
    """
    text = text_bytes.decode("utf-8")
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys)
    except RecursionError as exc:
        raise ValueError("JSON is nested too deeply to parse") from exc


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def parse_canonical_task(task: Any, *, error: type[Exception]) -> Task:
    """task dict 必須可被 Task.from_dict 解析且為 canonical 形式，否則 raise ``error``。"""
    if not isinstance(task, dict):
        raise error("task must be a dict")
    try:
        parsed = Task.from_dict(task)
    except ValueError as exc:
        raise error(f"invalid task: {exc}") from exc
    if not _same_json(parsed.to_dict(), task):
        raise error("task must be in canonical form")
    return parsed


def check_parent(parent: Any, *, error: type[Exception]) -> None:
    """parent 必須剛好有 PARENT_KEYS：trajectory_id / checkpoint_id 為非空 str，turn 為 int。"""
    if not isinstance(parent, dict) or set(parent) != set(PARENT_KEYS):
        raise error("parent must be an object with keys: " + ", ".join(PARENT_KEYS))
    for key in ("trajectory_id", "checkpoint_id"):
        if not isinstance(parent[key], str) or parent[key] == "":
            raise error(f"parent.{key} must be a non-empty string")
    if type(parent["turn"]) is not int:
        raise error("parent.turn must be an integer")
