"""Checkpoint、restore、rerun 與 lineage 驗證（decision log 006）。

Checkpoint k = trajectory 完成前 k 個 turns 之後、下一次 model 呼叫之前的決策點（0 <= k < max_turns）。
它自帶完整前綴（conversation + tool calls / results）與 env snapshot，載入時從 task 初始狀態重放前綴驗證，
因此呼叫端不需要、也不可以自行切 ``records[:k]``（decision 004 的前綴來源綁定）。

本模組所有「相同 / 相符」一律以 ``canonical_json`` 逐 byte 比較；所有「int」一律 ``type(x) is int``。

核心 invariant（SPEC §5）：令 M_k 為原 episode 中 policy 第 k+1 次 ``next_message`` 實際收到的 messages。
trajectory 經 save / load、checkpoint 經 save / load 之後：
- ``json.dumps(restore(cp).messages, ensure_ascii=False)``（不排序 key，對順序敏感）與 ``json.dumps(M_k, ensure_ascii=False)``
  逐 byte 相同；
- ``canonical_json(restore(cp).env.snapshot()) == canonical_json(原 episode 在該點的 env.snapshot())``；
- 原 episode 與 rerun 都使用回傳常數的 clock（例如 ``lambda: 0.0``）且 policy 為決定性時，rerun 的 turns / termination /
  final_snapshot 與原 trajectory 逐 byte 相同；一般情況只比較每個 turn 的 index、message、tool_steps。

Checkpoint 的結構與重放規則（``verify``，違反 raise CheckpointError）：
1. task 可被 Task.from_dict 解析且為 canonical 形式；agent 為 AgentConfig；max_turns 為 int >= 1；
   turn 為 int 且 0 <= turn < max_turns、turn == len(turns)。
2. 每個元素是 Turn、``turns[i].index == i``（int）；每個前綴 turn 都至少有一個 tool call，且完整符合 Trajectory 規則 3
   （tool steps 與 tool calls 一一對應、名稱與 args 逐 byte 相同、``usage.model_turns == 1``）。
3. 前綴 tool steps 的 step 從 0 連續。
4. 從 task 初始狀態重放前綴所有 tool calls：每筆 record 與保存的逐 byte 相同，且
   ``canonical_json(重放後 env.snapshot()) == canonical_json(env_snapshot)``（因此 env_snapshot["step"] == 前綴 tool steps 數）。
5. source_trajectory_id 為非空 str；``canonical_json(to_dict()).encode("utf-8")`` 成功。
6. env_snapshot 可被全新的 ``IssueTrackerEnv().restore`` 接受（因此通過 verify 的 checkpoint 一定能 restore / rerun）；
   task、env_snapshot、前綴 turns 的 meta 與 tool steps 的內容都符合 Trajectory 規則 10（JSON 文件、不含 tuple、深度上限；
   decision 007 / 009）。
   ``from_dict`` / ``load`` 在任何深拷貝之前先以同一規則檢查整份輸入。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.policy import AgentConfig, Policy
from env.environment import IssueTrackerEnv, SnapshotError
from env.hashing import canonical_json, sha256_of
from replay.runner import Clock, _continue_episode
from replay.trajectory import (
    Trajectory,
    TrajectoryError,
    Turn,
    MAX_DOCUMENT_DEPTH,
    check_json_document,
    check_turn_payloads,
    check_turn_sequence,
    load_strict_json,
    parse_canonical_task,
    render_messages,
    replay_tool_steps,
    same_canonical,
)
from tasks.schema import Task

CHECKPOINT_FORMAT = "agent-gym/checkpoint/v2"  # v2：turns 中的 ModelMessage 含 provider_content（decision 009）

CHECKPOINT_KEYS: tuple[str, ...] = (
    "format",
    "task",
    "agent",
    "max_turns",
    "turn",
    "turns",
    "env_snapshot",
    "source_trajectory_id",
    "checkpoint_id",
)


class CheckpointError(ValueError):
    """Checkpoint 不合法、重放不符、checkpoint_id 不符、lineage 不符，或 rerun 的 policy 與 checkpoint 不一致。"""


@dataclass(frozen=True)
class Checkpoint:
    task: dict[str, Any]
    agent: AgentConfig
    max_turns: int
    turn: int
    turns: tuple[Turn, ...]
    env_snapshot: dict[str, Any]
    source_trajectory_id: str

    def verify(self) -> None:
        """模組 docstring 的規則 1–6；違反 raise CheckpointError。"""
        # 深度以文件根計算（task / env_snapshot / turns 位於 to_dict() 的第 1 層），與 from_dict / load 一致
        check_json_document(self.task, error=CheckpointError, max_depth=MAX_DOCUMENT_DEPTH - 1)
        check_json_document(self.env_snapshot, error=CheckpointError, max_depth=MAX_DOCUMENT_DEPTH - 1)
        check_turn_payloads(self.turns, error=CheckpointError)
        task = parse_canonical_task(self.task, error=CheckpointError)
        if not isinstance(self.agent, AgentConfig):
            raise CheckpointError("agent must be an AgentConfig")
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise CheckpointError("max_turns must be an integer >= 1")
        if type(self.turn) is not int or not 0 <= self.turn < self.max_turns:
            raise CheckpointError("turn must be an integer in [0, max_turns)")
        if not isinstance(self.turns, (list, tuple)) or len(self.turns) != self.turn:
            raise CheckpointError("turn must equal the number of prefix turns")
        check_turn_sequence(self.turns, error=CheckpointError, all_have_tool_calls=True)
        steps = [step for turn in self.turns for step in turn.tool_steps]
        env = replay_tool_steps(task, steps, error=CheckpointError)
        if not same_canonical(env.snapshot(), self.env_snapshot):
            raise CheckpointError("env_snapshot does not match the replayed prefix")
        try:
            IssueTrackerEnv().restore(copy.deepcopy(self.env_snapshot))
        except (SnapshotError, TypeError, ValueError) as exc:
            raise CheckpointError(f"env_snapshot cannot be restored: {exc}") from exc
        if not isinstance(self.source_trajectory_id, str) or self.source_trajectory_id == "":
            raise CheckpointError("source_trajectory_id must be a non-empty string")
        try:
            canonical_json(self.to_dict()).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint is not serializable as UTF-8 JSON: {exc}") from exc

    def messages(self) -> list[dict[str, Any]]:
        """``render_messages(task instruction, agent, turns)``。"""
        return render_messages(self.task["instruction"], self.agent, self.turns)

    @property
    def checkpoint_id(self) -> str:
        """``sha256_of(to_dict() 去掉 "checkpoint_id" 這個 key)``。"""
        return sha256_of(self._content())

    def _content(self) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "task": copy.deepcopy(self.task),
            "agent": self.agent.to_dict(),
            "max_turns": self.max_turns,
            "turn": self.turn,
            "turns": [turn.to_dict() for turn in self.turns],
            "env_snapshot": copy.deepcopy(self.env_snapshot),
            "source_trajectory_id": self.source_trajectory_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """剛好 CHECKPOINT_KEYS（``format`` = CHECKPOINT_FORMAT、含 checkpoint_id），JSON-native、深拷貝；
        agent 為 ``agent.to_dict()``、turns 為 ``[turn.to_dict(), ...]``（與 Trajectory 相同的編碼）。
        """
        content = self._content()
        content["checkpoint_id"] = sha256_of(content)
        return content

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """data 必須是 dict、key 剛好為 CHECKPOINT_KEYS、format 相符；解析後呼叫 verify()，並要求
        ``data["checkpoint_id"] == checkpoint_id``。任何錯誤一律 raise CheckpointError。
        保證 ``Checkpoint.from_dict(c.to_dict()) == c``。
        """
        check_json_document(data, error=CheckpointError)
        if not isinstance(data, dict) or set(data) != set(CHECKPOINT_KEYS):
            raise CheckpointError("checkpoint must be an object with keys: " + ", ".join(CHECKPOINT_KEYS))
        if data["format"] != CHECKPOINT_FORMAT:
            raise CheckpointError("unsupported checkpoint format")
        try:
            agent = AgentConfig.from_dict(data["agent"])
            if not isinstance(data["turns"], list):
                raise CheckpointError("checkpoint turns must be a list")
            turns = tuple(Turn.from_dict(turn) for turn in data["turns"])
        except (TrajectoryError, ValueError) as exc:
            if isinstance(exc, CheckpointError):
                raise
            raise CheckpointError(f"invalid checkpoint: {exc}") from exc
        try:
            checkpoint = cls(
                task=copy.deepcopy(data["task"]),
                agent=agent,
                max_turns=data["max_turns"],
                turn=data["turn"],
                turns=turns,
                env_snapshot=copy.deepcopy(data["env_snapshot"]),
                source_trajectory_id=data["source_trajectory_id"],
            )
        except TypeError as exc:
            raise CheckpointError(f"invalid checkpoint: {exc}") from exc
        checkpoint.verify()
        if data["checkpoint_id"] != checkpoint.checkpoint_id:
            raise CheckpointError("checkpoint_id does not match the checkpoint content")
        return checkpoint

    def save(self, path: str | Path) -> None:
        """寫入 ``canonical_json(self.to_dict()) + "\\n"``（UTF-8）。"""
        Path(path).write_bytes((canonical_json(self.to_dict()) + "\n").encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        """以 ``replay.trajectory.load_strict_json`` 解析後 from_dict；解碼 / 解析錯誤包成 CheckpointError，OSError 原樣拋出。"""
        raw = Path(path).read_bytes()
        try:
            data = load_strict_json(raw)
        except ValueError as exc:
            raise CheckpointError(f"cannot parse checkpoint file: {exc}") from exc
        return cls.from_dict(data)


def make_checkpoint(trajectory: Trajectory, turn: int) -> Checkpoint:
    """在 trajectory 的第 turn 個 turn 之後建立 checkpoint。

    trajectory 先 verify（TrajectoryError 原樣拋出）。turn 必須是 int、``0 <= turn < trajectory.max_turns``、
    ``turn <= len(turns)``，且 turns[:turn] 每個都有 tool call（否則 CheckpointError；因此 final_answer 的最後一個 turn
    之後、以及 max_turns 用盡之後都不能建立 checkpoint）。
    env_snapshot 由重放 turns[:turn] 的 tool steps 取得；source_trajectory_id = trajectory.trajectory_id。
    回傳的 checkpoint 已通過 verify()，且不與 trajectory 共用可變物件。
    """
    trajectory.verify()
    if type(turn) is not int or not 0 <= turn < trajectory.max_turns or turn > len(trajectory.turns):
        raise CheckpointError("turn must be an integer with 0 <= turn < max_turns and turn <= len(turns)")
    prefix = trajectory.turns[:turn]
    if any(not t.message.tool_calls for t in prefix):
        raise CheckpointError("cannot checkpoint after a turn without tool calls")
    task = Task.from_dict(trajectory.task)
    env = replay_tool_steps(task, [step for t in prefix for step in t.tool_steps], error=CheckpointError)
    checkpoint = Checkpoint(
        task=copy.deepcopy(trajectory.task),
        agent=copy.deepcopy(trajectory.agent),
        max_turns=trajectory.max_turns,
        turn=turn,
        turns=copy.deepcopy(tuple(prefix)),
        env_snapshot=env.snapshot(),
        source_trajectory_id=trajectory.trajectory_id,
    )
    checkpoint.verify()
    return checkpoint


@dataclass
class RestoredEpisode:
    env: IssueTrackerEnv
    messages: list[dict[str, Any]]


def restore(checkpoint: Checkpoint) -> RestoredEpisode:
    """先 verify；回傳以 env_snapshot restore 的新 env，以及 checkpoint 點的 conversation（新物件）。"""
    checkpoint.verify()
    env = IssueTrackerEnv()
    env.restore(copy.deepcopy(checkpoint.env_snapshot))
    return RestoredEpisode(env=env, messages=checkpoint.messages())


def rerun(checkpoint: Checkpoint, policy: Policy, *, clock: Clock = time.perf_counter) -> Trajectory:
    """從 checkpoint 以 policy 繼續 episode。

    先 verify；``policy.agent_config.config_id != checkpoint.agent.config_id`` → CheckpointError（不同 agent 接續不叫 rerun）。
    以 restore 的 env 呼叫 ``replay.runner._continue_episode(task, policy, max_turns=checkpoint.max_turns, env=...,
    prefix_turns=checkpoint.turns, parent={"trajectory_id": checkpoint.source_trajectory_id,
    "checkpoint_id": checkpoint.checkpoint_id, "turn": checkpoint.turn}, clock=clock)``。
    回傳前再確認 ``trajectory.agent.config_id == checkpoint.agent.config_id``（policy 的 agent_config 中途改變 → CheckpointError）。
    回傳的 trajectory 含完整前綴 turns、至少一個新 turn，可直接 ``grade()``；不與 checkpoint 共用可變物件。
    """
    restored = restore(checkpoint)
    config = policy.agent_config
    if not isinstance(config, AgentConfig) or config.config_id != checkpoint.agent.config_id:
        raise CheckpointError("policy agent_config does not match the checkpoint agent")
    trajectory = _continue_episode(
        Task.from_dict(checkpoint.task),
        policy,
        max_turns=checkpoint.max_turns,
        env=restored.env,
        prefix_turns=checkpoint.turns,
        parent={
            "trajectory_id": checkpoint.source_trajectory_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "turn": checkpoint.turn,
        },
        clock=clock,
    )
    if trajectory.agent.config_id != checkpoint.agent.config_id:
        raise CheckpointError("policy agent_config changed during the rerun")
    return trajectory


def verify_lineage(child: Checkpoint | Trajectory, source: Trajectory, *, checkpoint: Checkpoint | None = None) -> None:
    """以呼叫端持有的資料為錨點，驗證 child 的前綴未被竄改；不符 raise CheckpointError。

    先 ``source.verify()``（TrajectoryError 原樣拋出）。child 不是 Checkpoint / Trajectory → TypeError。
    - child 為 Checkpoint：``child.verify()``；``child.source_trajectory_id == source.trajectory_id``；task、agent、max_turns
      與 source 相同；``child.turns`` 與 ``source.turns[:child.turn]`` 逐 byte 相同。
    - child 為 Trajectory（rerun 的結果）：必須提供 checkpoint（呼叫端自己用來 rerun 的那一個；None → CheckpointError）。
      依序：``verify_lineage(checkpoint, source)``；``child.verify()``（TrajectoryError 以 ``from e`` 包成 CheckpointError）；
      ``child.parent`` 與 ``{"trajectory_id": checkpoint.source_trajectory_id, "checkpoint_id": checkpoint.checkpoint_id,
      "turn": checkpoint.turn}`` 逐 byte 相同；task、agent、max_turns 與 checkpoint 相同；
      ``child.turns[:checkpoint.turn]`` 與 ``checkpoint.turns`` 逐 byte 相同。
    只有內部 digest 擋不住刻意重算；對刻意竄改的偵測依賴呼叫端持有自己發出的 checkpoint 與來源 trajectory。
    """
    source.verify()
    if isinstance(child, Checkpoint):
        child.verify()
        if child.source_trajectory_id != source.trajectory_id:
            raise CheckpointError("checkpoint does not come from this source trajectory")
        _require_same_header(child, source.task, source.agent, source.max_turns)
        if not same_canonical([t.to_dict() for t in child.turns], [t.to_dict() for t in source.turns[: child.turn]]):
            raise CheckpointError("checkpoint prefix differs from the source trajectory")
    elif isinstance(child, Trajectory):
        if checkpoint is None:
            raise CheckpointError("verifying a rerun trajectory requires the checkpoint it was rerun from")
        verify_lineage(checkpoint, source)
        try:
            child.verify()
        except TrajectoryError as exc:
            raise CheckpointError(f"rerun trajectory is invalid: {exc}") from exc
        expected_parent = {
            "trajectory_id": checkpoint.source_trajectory_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "turn": checkpoint.turn,
        }
        if not same_canonical(child.parent, expected_parent):
            raise CheckpointError("rerun parent does not point to the given checkpoint")
        _require_same_header(child, checkpoint.task, checkpoint.agent, checkpoint.max_turns)
        prefix = [t.to_dict() for t in child.turns[: checkpoint.turn]]
        if not same_canonical(prefix, [t.to_dict() for t in checkpoint.turns]):
            raise CheckpointError("rerun prefix differs from the checkpoint")
    else:
        raise TypeError("child must be a Checkpoint or a Trajectory")


def _require_same_header(child: Checkpoint | Trajectory, task: Any, agent: AgentConfig, max_turns: int) -> None:
    if not same_canonical(child.task, task):
        raise CheckpointError("task differs")
    if not same_canonical(child.agent.to_dict(), agent.to_dict()):
        raise CheckpointError("agent differs")
    if not same_canonical(child.max_turns, max_turns):
        raise CheckpointError("max_turns differs")
