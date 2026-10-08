"""已知真實 recoverability 的 synthetic policy（decision log 010），只用於驗證 failure localization 流程。

行為：依 task 的 oracle 行動；每個 turn 以機率 q 改為發出不可逆的錯誤 ``add_comment(protected_issue, MISTAKE_BODY)``，
錯誤之後照常繼續 oracle。對 conversation 無狀態：

- assistant 訊息中，tool_calls 剛好是一個且（name、canonical args）等於錯誤 call 者計為錯誤，其餘每則計為一個已完成的 oracle call；
  r = 已完成的 oracle call 數。
- 決策隨機數 u = ``int.from_bytes(sha256(f"{rollout_seed}|{canonical_json(messages)}".encode()).digest()[:8], "big") / 2**64``。
- u < q → 錯誤 call；否則 r < n → oracle[r]；否則結束訊息（text = "Done."，stop_reason "end_turn"）。
- 第 k 則 assistant 訊息的 tool call id 為 ``f"call_{k}_0"``；用量 0 / 0 / 0.0。

解析解（``success_probability``）：前綴已有錯誤 → 0；否則 (1 − q)^(n − r + 1)。
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Sequence

from agents.policy import AgentConfig, ModelMessage, PolicyStep, ToolCall
from env.environment import check_call_preconditions
from env.hashing import canonical_json
from replay.trajectory import Trajectory
from tasks.schema import COMMENTS_FIELD, Task

SYNTHETIC_POLICY = "synthetic"
SYNTHETIC_PROMPT_VERSION = "synthetic/v1"
MISTAKE_BODY = "SYNTHETIC-MISTAKE"


def protected_issue(task: Task) -> int:
    """success_constraints 中 ``issue_unchanged`` 且涵蓋 comments（fields 省略或含 "comments"）的 issue 裡 id 最小者。
    沒有任何這樣的 issue → ValueError。
    """
    candidates = [
        constraint["issue_id"]
        for constraint in task.success_constraints
        if constraint.get("type") == "issue_unchanged"
        and ("fields" not in constraint or COMMENTS_FIELD in constraint["fields"])
    ]
    if not candidates:
        raise ValueError(f"task {task.task_id!r} has no issue_unchanged constraint that covers comments")
    return min(candidates)


def mistake_call(task: Task) -> tuple[str, dict[str, Any]]:
    """``("add_comment", {"issue_id": protected_issue(task), "body": MISTAKE_BODY})``（新物件）。"""
    return ("add_comment", {"issue_id": protected_issue(task), "body": MISTAKE_BODY})


def synthetic_agent_config(
    oracle_calls: Sequence[tuple[str, Any]], mistake: tuple[str, Any], mistake_rate: float
) -> AgentConfig:
    """policy=SYNTHETIC_POLICY、prompt_version=SYNTHETIC_PROMPT_VERSION、prompt 為空字串、
    model_config = ``{"oracle": [{"name", "args"}, ...], "mistake": {"name", "args"}, "mistake_rate": float(q), "final_text": "Done."}``
    （decision 013 §4：以 float 保存，q = 0 與 q = 0.0 是同一個 agent）。
    mistake_rate 必須是 0 <= q < 1 的 float / int（bool 不算），否則 ValueError；oracle_calls 不可為空。
    oracle_calls 必須是 list / tuple，每個 call 與 mistake 都是長度 2 的 list / tuple ``(name, args)``，且滿足
    ``check_call_preconditions``（否則 TypeError / ValueError）。
    """
    oracle = _calls(oracle_calls)
    name, args = _call(mistake)
    if isinstance(mistake_rate, bool) or not isinstance(mistake_rate, (int, float)):
        raise ValueError("mistake_rate must be a number")
    if not math.isfinite(mistake_rate) or not 0 <= mistake_rate < 1:
        raise ValueError("mistake_rate must satisfy 0 <= q < 1")
    return AgentConfig(
        policy=SYNTHETIC_POLICY,
        prompt_version=SYNTHETIC_PROMPT_VERSION,
        system_prompt="",
        developer_prompt="",
        model_config={
            "oracle": [{"name": call_name, "args": call_args} for call_name, call_args in oracle],
            "mistake": {"name": name, "args": args},
            "mistake_rate": float(mistake_rate),
            "final_text": FINAL_TEXT,
        },
    )


FINAL_TEXT = "Done."


def _call(call: Any) -> tuple[str, Any]:
    if not isinstance(call, (list, tuple)):
        raise TypeError("each call must be a (name, args) list or tuple")
    if len(call) != 2:
        raise ValueError("each call must have exactly two elements: (name, args)")
    name, args = call
    check_call_preconditions(name, args)
    return name, copy.deepcopy(args)


def _calls(calls: Any) -> list[tuple[str, Any]]:
    if not isinstance(calls, (list, tuple)):
        raise TypeError("oracle_calls must be a list or tuple")
    if len(calls) == 0:
        raise ValueError("oracle_calls must not be empty")
    return [_call(call) for call in calls]


def _is_mistake(message: dict[str, Any], mistake_key: str) -> bool:
    """assistant 訊息的 tool_calls 剛好一個且（name、canonical args）等於錯誤 call。"""
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return False
    return _call_key(calls[0].get("name"), calls[0].get("args")) == mistake_key


def _call_key(name: Any, args: Any) -> str:
    return canonical_json([name, args])


def _progress(assistant_messages: Sequence[dict[str, Any]], mistake_key: str) -> tuple[int, bool]:
    """(已完成的 oracle call 數 r, 是否已有錯誤)：每則非錯誤的 assistant 訊息計為一個已完成的 oracle call。"""
    done = 0
    mistaken = False
    for message in assistant_messages:
        if _is_mistake(message, mistake_key):
            mistaken = True
        else:
            done += 1
    return done, mistaken


class SyntheticPolicy:
    """``SyntheticPolicy(oracle_calls, *, mistake, mistake_rate, rollout_seed)``；rollout_seed 為 int（bool 不算）。
    ``agent_config`` = synthetic_agent_config(...)（每次新物件；不含 rollout_seed）。messages 驗證同 ScriptedPolicy。
    """

    def __init__(
        self,
        oracle_calls: Sequence[tuple[str, Any]],
        *,
        mistake: tuple[str, Any],
        mistake_rate: float,
        rollout_seed: int,
    ) -> None:
        if type(rollout_seed) is not int:
            raise TypeError("rollout_seed must be an int")
        self._config = synthetic_agent_config(oracle_calls, mistake, mistake_rate)
        self._oracle = _calls(oracle_calls)
        self._mistake = _call(mistake)
        self._mistake_key = _call_key(*self._mistake)
        self._q = mistake_rate
        self._seed = rollout_seed

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise TypeError("each message must be a dict with a str 'role'")
        assistant = [message for message in messages if message["role"] == "assistant"]
        k = len(assistant)
        done, _ = _progress(assistant, self._mistake_key)
        digest = hashlib.sha256(f"{self._seed}|{canonical_json(messages)}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        if u < self._q:
            name, args = self._mistake
        elif done < len(self._oracle):
            name, args = self._oracle[done]
        else:
            message = ModelMessage(text=FINAL_TEXT, tool_calls=(), stop_reason="end_turn")
            return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)
        call = ToolCall(id=f"call_{k}_0", name=name, args=copy.deepcopy(args))
        message = ModelMessage(text="", tool_calls=(call,), stop_reason="tool_use")
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


def success_probability(n_oracle: int, oracle_done: int, mistake_in_prefix: bool, mistake_rate: float) -> float:
    """解析解：mistake_in_prefix → 0.0；否則 (1 − q)^(n_oracle − oracle_done + 1)。
    n_oracle >= 1、0 <= oracle_done <= n_oracle（int，bool 不算）、mistake_in_prefix 為 bool、0 <= q < 1；否則 ValueError。
    """
    if type(n_oracle) is not int or type(oracle_done) is not int or n_oracle < 1 or not 0 <= oracle_done <= n_oracle:
        raise ValueError("require ints with n_oracle >= 1 and 0 <= oracle_done <= n_oracle")
    if type(mistake_in_prefix) is not bool:
        raise ValueError("mistake_in_prefix must be a bool")
    if isinstance(mistake_rate, bool) or not isinstance(mistake_rate, (int, float)) or not 0 <= mistake_rate < 1:
        raise ValueError("mistake_rate must satisfy 0 <= q < 1")
    if mistake_in_prefix:
        return 0.0
    return float((1.0 - mistake_rate) ** (n_oracle - oracle_done + 1))


def checkpoint_truth(trajectory: Trajectory, k: int, mistake: tuple[str, Any], mistake_rate: float) -> float:
    """P(success | checkpoint k) 的解析解（decision 010 §1、013 §4），只對 synthetic policy 自己產生的（on-policy）軌跡有定義。

    - 0 <= k < len(turns)（int，bool 不算；與合法 checkpoint 一致）；trajectory.agent.policy 必須是 SYNTHETIC_POLICY；
      mistake 必須與 ``model_config["mistake"]`` 逐 byte 相同、mistake_rate 必須與 ``model_config["mistake_rate"]`` 數值相等；
    - 前 k 個 turns 每個都必須剛好一個 tool call，且（canonical 比較）是錯誤 call，或是下一個 oracle call（oracle[r]，
      r = 之前非錯誤 turn 的數目）；
    違反任何一項 → ValueError。n_oracle 取自 ``model_config["oracle"]`` 的長度；回傳 success_probability(n, r, 是否已有錯誤, q)。
    """
    if type(k) is not int or not 0 <= k < len(trajectory.turns):
        raise ValueError("k must be an int with 0 <= k < len(turns)")
    config = trajectory.agent.model_config
    if trajectory.agent.policy != SYNTHETIC_POLICY:
        raise ValueError("checkpoint_truth is defined only for synthetic-policy trajectories")
    name, args = _call(mistake)
    if canonical_json({"name": name, "args": args}) != canonical_json(config["mistake"]):
        raise ValueError("mistake does not match the trajectory's synthetic agent config")
    if isinstance(mistake_rate, bool) or not isinstance(mistake_rate, (int, float)) or float(mistake_rate) != float(
        config["mistake_rate"]
    ):
        raise ValueError("mistake_rate does not match the trajectory's synthetic agent config")
    oracle = [_call_key(call["name"], call["args"]) for call in config["oracle"]]
    mistake_key = _call_key(name, args)
    done = 0
    mistaken = False
    for turn in trajectory.turns[:k]:
        calls = turn.message.tool_calls
        key = _call_key(calls[0].name, calls[0].args) if len(calls) == 1 else None
        if key == mistake_key:
            mistaken = True
        elif key is not None and done < len(oracle) and key == oracle[done]:
            done += 1
        else:
            raise ValueError(f"turn {turn.index} is not an on-policy synthetic turn: no analytic truth")
    return success_probability(len(oracle), done, mistaken, mistake_rate)
