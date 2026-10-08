"""Episode runner：驅動 policy 與 env，並記錄完整 trajectory（decision log 006）。

公開入口只有 ``run_episode``（從頭執行）與 ``replay.checkpoint.rerun``（從 checkpoint 續跑）；
``_continue_episode`` 是兩者共用的內部實作，呼叫端不得自行組前綴（decision 004 / 006）。

例外中止：policy 或 env 拋出的例外原樣往外拋，不產生 trajectory。實驗腳本必須記錄每次中止
（task_id、checkpoint_id 或 None、attempt、例外型別、已完成 turn 數），在報告中列出中止數，不得默默重抽；
暫時性 API 錯誤的重試屬於 policy adapter 內部，重試上限必須在看結果前固定（decision 006）。
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Sequence

from agents.policy import AgentConfig, Policy, PolicyStep
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from grader.schema import Usage
from replay.trajectory import (
    Trajectory,
    Turn,
    check_parent,
    check_turn_sequence,
    render_messages,
    replay_tool_steps,
)
from tasks.schema import Task

Clock = Callable[[], float]


def run_episode(task: Task, policy: Policy, *, max_turns: int, clock: Clock = time.perf_counter) -> Trajectory:
    """從 task 初始狀態執行一個 episode。

    先檢查 ``type(max_turns) is int and max_turns >= 1``（否則 ValueError），再 ``task.validate()``（TaskError 原樣拋出），
    然後以新 env（``reset(task.initial_state, task.actor_id)``）、空前綴、parent=None 呼叫 ``_continue_episode``。
    """
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be an integer >= 1")
    task.validate()
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    return _continue_episode(task, policy, max_turns=max_turns, env=env, prefix_turns=(), parent=None, clock=clock)


def _continue_episode(
    task: Task,
    policy: Policy,
    *,
    max_turns: int,
    env: IssueTrackerEnv,
    prefix_turns: Sequence[Turn],
    parent: dict[str, Any] | None,
    clock: Clock,
) -> Trajectory:
    """從已完成的 prefix_turns 與對應的 env 狀態繼續執行。

    入口檢查（在第一次呼叫 policy 之前；違反 raise ValueError，env 不變、policy 不被呼叫）：
    - ``type(max_turns) is int``、``max_turns >= 1``、``len(prefix_turns) < max_turns``；
    - 每個前綴元素是 Turn，``type(index) is int`` 且 ``prefix_turns[i].index == i``；每個前綴 turn 至少一個 tool call，
      且符合 Trajectory 規則 3（tool steps 與 tool calls 一一對應、名稱與 args 逐 byte 相同、usage.model_turns == 1）；
      前綴 tool steps 的 step 從 0 連續；
    - env 已初始化（未初始化 → ValueError）且 ``env.actor_id == task.actor_id``；
    - 從 task 初始狀態以新 env 重放前綴所有 tool calls：每筆 record 與前綴保存的逐 byte 相同，且
      ``canonical_json(重放後 snapshot) == canonical_json(env.snapshot())``；
    - parent 為 None ⇒ 前綴為空（rerun 從 k = 0 出發時前綴為空但 parent 不為 None，合法）；parent 不為 None 時必須是
      剛好有 PARENT_KEYS 的 dict，trajectory_id / checkpoint_id 為非空 str，``type(parent["turn"]) is int`` 且
      ``parent["turn"] == len(prefix_turns)``。
    另外 ``agent = policy.agent_config`` 只在開始時讀一次（不是 AgentConfig → TypeError），整個 episode 都用這一份；
    policy 的 agent_config 在其生命週期內必須不變。

    迴圈（當 len(turns) < max_turns）：
    1. ``messages = render_messages(task.instruction, agent, turns)``，``tools = IssueTrackerEnv.tool_specs()``；
       恰好執行 ``t0 = clock(); step = policy.next_message(messages, tools); t1 = clock()``，``latency_s = t1 - t0``；
       runner 不在其他任何地方呼叫 clock。step 不是 PolicyStep → TypeError。
    2. 依序對 message 的每個 tool call 執行 ``env.call_tool(call.name, call.args)``；之後以
       ``env.last_records(len(message.tool_calls))`` 取得該 turn 的 tool_steps（不讀整份 ``env.records``，避免 O(n²) 深拷貝）。
    3. 加入 ``Turn(index, message, tool_steps, Usage(input_tokens, output_tokens, latency_s, cost_usd, model_turns=1))``。
    4. message 沒有 tool call → termination "final_answer"，結束。
    迴圈因 len(turns) == max_turns 結束 → termination "max_turns"。
    回傳 ``Trajectory(task=task.to_dict(), agent=agent, max_turns, turns=prefix + 新 turns, termination,
    final_snapshot=env.snapshot(), parent)``，已通過 validate()。

    傳給 policy 的 messages / tools 是新物件，policy 修改它們不影響紀錄。
    """
    prefix = _check_entry(task, max_turns=max_turns, env=env, prefix_turns=prefix_turns, parent=parent)
    agent = policy.agent_config
    if not isinstance(agent, AgentConfig):
        raise TypeError("policy.agent_config must be an AgentConfig")

    turns = list(prefix)
    termination = "max_turns"
    while len(turns) < max_turns:
        messages = render_messages(task.instruction, agent, turns)
        tools = IssueTrackerEnv.tool_specs()
        t0 = clock()
        step = policy.next_message(messages, tools)
        t1 = clock()
        if not isinstance(step, PolicyStep):
            raise TypeError("policy.next_message must return a PolicyStep")
        message = step.message
        for call in message.tool_calls:
            env.call_tool(call.name, call.args)
        tool_steps = env.last_records(len(message.tool_calls))
        usage = Usage(
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            latency_s=t1 - t0,
            cost_usd=step.cost_usd,
            model_turns=1,
        )
        turns.append(Turn(index=len(turns), message=message, tool_steps=tool_steps, usage=usage))
        if not message.tool_calls:
            termination = "final_answer"
            break

    trajectory = Trajectory(
        task=task.to_dict(),
        agent=agent,
        max_turns=max_turns,
        turns=tuple(turns),
        termination=termination,
        final_snapshot=env.snapshot(),
        parent=copy.deepcopy(parent),
    )
    trajectory.validate()
    return trajectory


def _check_entry(
    task: Task,
    *,
    max_turns: Any,
    env: IssueTrackerEnv,
    prefix_turns: Sequence[Turn],
    parent: dict[str, Any] | None,
) -> tuple[Turn, ...]:
    """``_continue_episode`` 的入口檢查；違反 raise ValueError。回傳前綴的深拷貝。"""
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be an integer >= 1")
    if not isinstance(prefix_turns, (list, tuple)):
        raise ValueError("prefix_turns must be a list or tuple")
    if len(prefix_turns) >= max_turns:
        raise ValueError("the prefix must be shorter than max_turns")
    check_turn_sequence(prefix_turns, error=ValueError, all_have_tool_calls=True)
    try:
        actor_id = env.actor_id
    except RuntimeError as exc:
        raise ValueError("env must be initialized") from exc
    if actor_id != task.actor_id:
        raise ValueError("env actor does not match the task actor")
    steps = [step for turn in prefix_turns for step in turn.tool_steps]
    replayed = replay_tool_steps(task, steps, error=ValueError)
    if canonical_json(replayed.snapshot()) != canonical_json(env.snapshot()):
        raise ValueError("env state does not match the replayed prefix")
    if parent is None:
        if prefix_turns:
            raise ValueError("a non-empty prefix requires a parent")
    else:
        check_parent(parent, error=ValueError)
        if parent["turn"] != len(prefix_turns):
            raise ValueError("parent.turn must equal the prefix length")
    return copy.deepcopy(tuple(prefix_turns))
