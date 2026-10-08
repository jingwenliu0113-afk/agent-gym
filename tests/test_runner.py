"""replay/runner.py 的契約測試：run_episode 與 _continue_episode（decision log 006）。

契約來源：``replay/runner.py`` docstring、``replay/trajectory.py``、``agents/policy.py``、decision 006。
- 「相同 / 相符」一律以 ``canonical_json`` 逐 byte 比較（不用 Python ``==``）。
- policy 收到的 conversation 以 order-sensitive 的 ``json.dumps(..., ensure_ascii=False)`` 比較。
- 真實 task 使用 smoke tasks（只用於回歸測試，不計入任何統計）；邊界情況用測試內的小型 policy。
- 本檔不在 import / collection 階段呼叫任何 stub。
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import random
from typing import Any, Callable, Sequence

import pytest

from agents.policy import AgentConfig, ModelMessage, PolicyStep, ScriptedPolicy, ToolCall, oracle_policy
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from grader.schema import Usage
from replay.runner import _continue_episode, run_episode
from replay.trajectory import Trajectory, Turn, render_messages
from tasks.schema import TASK_FORMAT, Task, TaskError
from tasks.smoke import REFERENCE_SOLUTIONS, SMOKE_TASK_IDS, get_smoke_task
from tests.conftest import SMALL_ACTOR

# 7 個 oracle call（每個一個 turn），適合切前綴。
LONG_ID = "smoke-reassign-departed"
# 2 個 oracle call。
SHORT_ID = "smoke-assign-and-start"
NOOP_ID = "smoke-conditional-noop"

R = dataclasses.replace

BAD_MAX_TURNS = [0, -1, True, False, 1.0, 3.0, "3", None]

# (name, args, 預期 error code)：都是合法的 JSON 值，env 必須記錄為 tool error 而不是 raise。
BAD_CALLS: list[tuple[str, Any, str]] = [
    ("no_such_tool", {}, "unknown_tool"),
    ("no_such_tool", [1, 2], "unknown_tool"),
    ("get_issue", {"issue_id": "8"}, "invalid_arguments"),
    ("get_issue", {"issue_id": 8.0}, "invalid_arguments"),
    ("get_issue", {"issue_id": True}, "invalid_arguments"),
    ("get_issue", {}, "invalid_arguments"),
    ("get_issue", {"issue_id": 8, "extra": 1}, "invalid_arguments"),
    ("get_issue", [8], "invalid_arguments"),
    ("get_issue", "8", "invalid_arguments"),
    ("get_issue", 8, "invalid_arguments"),
    ("get_issue", None, "invalid_arguments"),
    ("get_issue", {"issue_id": 999}, "not_found"),
]


# ---------------------------------------------------------------- helpers


def _zero_clock() -> float:
    return 0.0


def _cj(value: Any) -> str:
    return canonical_json(value)


def _dumps(value: Any) -> str:
    """order-sensitive（不排序 key）的序列化。"""
    return json.dumps(value, ensure_ascii=False)


def _turns_cj(turns: Sequence[Turn]) -> str:
    return canonical_json([t.to_dict() for t in turns])


def _content_cj(turns: Sequence[Turn]) -> str:
    """只取決定性內容（index / message / tool_steps），不含 usage。"""
    return canonical_json([{k: v for k, v in t.to_dict().items() if k != "usage"} for t in turns])


def _records_cj(records: Sequence[Any]) -> str:
    return canonical_json([r.to_dict() for r in records])


def _all_steps(traj: Trajectory) -> list[Any]:
    return [s for t in traj.turns for s in t.tool_steps]


class _CountingClock:
    """記錄呼叫次數；有 values 時依序回傳，否則回傳 0.0（與 _zero_clock 相同）。"""

    def __init__(self, values: Sequence[float] | None = None) -> None:
        self.calls = 0
        self._values = values

    def __call__(self) -> float:
        value = self._values[self.calls] if self._values is not None else 0.0
        self.calls += 1
        return value


class _Spy:
    """包裝 policy：記錄每次收到的 messages / tools（深拷貝與原物件）與 agent_config 讀取次數。"""

    def __init__(self, inner: Any, probe: Callable[[], Any] | None = None) -> None:
        self.inner = inner
        self.inputs: list[tuple[Any, Any]] = []
        self.raw: list[tuple[Any, Any]] = []
        self.probes: list[Any] = []
        self.agent_reads = 0
        self._probe = probe

    @property
    def agent_config(self) -> Any:
        self.agent_reads += 1
        return self.inner.agent_config

    def next_message(self, messages: Any, tools: Any) -> Any:
        if self._probe is not None:
            self.probes.append(self._probe())
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.raw.append((messages, tools))
        return self.inner.next_message(messages, tools)


def _agent(tag: str = "a") -> AgentConfig:
    return AgentConfig(
        policy="test-policy",
        prompt_version="test/v1",
        system_prompt=f"system-{tag}",
        developer_prompt=f"developer-{tag}",
        model_config={"tag": tag},
    )


def _message(k: int, calls: Sequence[tuple[str, Any]], text: str = "") -> ModelMessage:
    tool_calls = tuple(
        ToolCall(id=f"call_{k}_{i}", name=name, args=copy.deepcopy(args)) for i, (name, args) in enumerate(calls)
    )
    return ModelMessage(text=text, tool_calls=tool_calls, stop_reason="tool_use" if tool_calls else "end_turn")


class _TurnPolicy:
    """無狀態的測試 policy：k = messages 中 assistant 訊息數，回傳 make_step(k)。

    agent 可以是 AgentConfig、任意物件（測 TypeError），或 ``callable(n) -> AgentConfig``（第 n 次讀取的值）。
    """

    def __init__(self, make_step: Callable[[int], Any], agent: Any = None) -> None:
        self._make_step = make_step
        self._agent = _agent() if agent is None else agent
        self.agent_reads = 0
        self.inputs: list[tuple[Any, Any]] = []

    @property
    def agent_config(self) -> Any:
        n = self.agent_reads
        self.agent_reads += 1
        return self._agent(n) if callable(self._agent) else self._agent

    def next_message(self, messages: Any, tools: Any) -> Any:
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        k = sum(1 for m in messages if m["role"] == "assistant")
        return self._make_step(k)


def _script_steps(script: Sequence[Sequence[tuple[str, Any]]], **usage: Any) -> Callable[[int], PolicyStep]:
    """script[k] 為第 k 個 turn 的 calls；k >= len(script) 回傳 final answer。"""

    def make(k: int) -> PolicyStep:
        calls = script[k] if k < len(script) else ()
        return PolicyStep(_message(k, calls, text="" if calls else "final"), **usage)

    return make


def _oracle(task_id: str, **kwargs: Any) -> Any:
    return oracle_policy(copy.deepcopy(REFERENCE_SOLUTIONS[task_id]), **kwargs)


def _env_after(task: Task, calls: Sequence[tuple[str, Any]], actor_id: str | None = None) -> IssueTrackerEnv:
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id if actor_id is None else actor_id)
    for name, args in calls:
        env.call_tool(name, copy.deepcopy(args))
    return env


def _restored(env: IssueTrackerEnv) -> IssueTrackerEnv:
    fresh = IssueTrackerEnv()
    fresh.restore(env.snapshot())
    return fresh


def _env_state(env: IssueTrackerEnv) -> str:
    return canonical_json({"snapshot": env.snapshot(), "records": [r.to_dict() for r in env.records]})


def _invalid_task(kind: str) -> Task:
    task = get_smoke_task(SHORT_ID)
    if kind == "empty_instruction":
        return R(task, instruction="")
    if kind == "unknown_actor":
        return R(task, actor_id="nobody")
    if kind == "empty_expected_outcome":
        return R(task, expected_outcome=())
    raise AssertionError(kind)


def _small_task(world: dict[str, Any]) -> Task:
    return Task.from_dict(
        {
            "format": TASK_FORMAT,
            "task_id": "runner-small",
            "family": "test/runner",
            "generator_version": "test/v1",
            "params": {},
            "instruction": "把 #2 改派給 Fay Frontend (u3)，並留言說明。",
            "actor_id": SMALL_ACTOR,
            "initial_state": world,
            "expected_outcome": [{"type": "issue_field_equals", "issue_id": 2, "field": "assignee_id", "value": "u3"}],
            "success_constraints": [],
            "forbidden": [],
            "mutation_scope": [{"issue_id": 2, "fields": ["assignee_id", "comments"]}],
        }
    )


# ---------------------------------------------------------------- run_episode：參數檢查順序


@pytest.mark.parametrize("max_turns", BAD_MAX_TURNS, ids=repr)
def test_run_episode_rejects_bad_max_turns(max_turns: Any) -> None:
    spy = _Spy(_oracle(SHORT_ID))
    clock = _CountingClock()
    with pytest.raises(ValueError):
        run_episode(get_smoke_task(SHORT_ID), spy, max_turns=max_turns, clock=clock)
    assert spy.inputs == []
    assert clock.calls == 0


@pytest.mark.parametrize("max_turns", BAD_MAX_TURNS, ids=repr)
def test_run_episode_checks_max_turns_before_task(max_turns: Any) -> None:
    """max_turns 不合法時，即使 task 也不合法，仍是 max_turns 的 ValueError（不是 TaskError）。"""
    spy = _Spy(_oracle(SHORT_ID))
    with pytest.raises(ValueError) as excinfo:
        run_episode(_invalid_task("empty_instruction"), spy, max_turns=max_turns, clock=_zero_clock)
    assert not isinstance(excinfo.value, TaskError)
    assert spy.inputs == []


def test_run_episode_checks_max_turns_before_agent_config() -> None:
    policy = _TurnPolicy(_script_steps([]), agent="not an AgentConfig")
    with pytest.raises(ValueError):
        run_episode(get_smoke_task(SHORT_ID), policy, max_turns=0, clock=_zero_clock)
    assert policy.inputs == []


@pytest.mark.parametrize("kind", ["empty_instruction", "unknown_actor", "empty_expected_outcome"])
def test_run_episode_propagates_task_error(kind: str) -> None:
    spy = _Spy(_oracle(SHORT_ID))
    clock = _CountingClock()
    with pytest.raises(TaskError):
        run_episode(_invalid_task(kind), spy, max_turns=5, clock=clock)
    assert spy.inputs == []
    assert clock.calls == 0


def test_run_episode_validates_task_before_agent_config() -> None:
    policy = _TurnPolicy(_script_steps([]), agent={"policy": "x"})
    with pytest.raises(TaskError):
        run_episode(_invalid_task("unknown_actor"), policy, max_turns=5, clock=_zero_clock)
    assert policy.inputs == []


@pytest.mark.parametrize("bad_agent", [None, "scripted", {"policy": "scripted", "prompt_version": "scripted/v1",
                                                            "system_prompt": "", "developer_prompt": "",
                                                            "model_config": {}}], ids=["none", "str", "dict"])
def test_run_episode_rejects_non_agent_config(bad_agent: Any) -> None:
    # _TurnPolicy 的 agent=None 代表預設 config，因此 None 用 callable 包一層。
    agent = (lambda n: None) if bad_agent is None else bad_agent
    policy = _TurnPolicy(_script_steps([[("get_issue", {"issue_id": 8})]]), agent=agent)
    clock = _CountingClock()
    with pytest.raises(TypeError):
        run_episode(get_smoke_task(SHORT_ID), policy, max_turns=5, clock=clock)
    assert policy.inputs == []
    assert clock.calls == 0


# ---------------------------------------------------------------- run_episode：oracle 完整執行


@pytest.mark.parametrize("slack", [1, 4])
@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_oracle_episode_reaches_final_answer(task_id: str, slack: int) -> None:
    task = get_smoke_task(task_id)
    calls = REFERENCE_SOLUTIONS[task_id]
    max_turns = len(calls) + slack
    policy = _oracle(task_id)
    traj = run_episode(task, policy, max_turns=max_turns, clock=_zero_clock)

    assert isinstance(traj, Trajectory)
    assert traj.termination == "final_answer"
    assert type(traj.turns) is tuple
    assert len(traj.turns) == len(calls) + 1
    assert [t.index for t in traj.turns] == list(range(len(calls) + 1))
    last = traj.turns[-1]
    assert len(last.message.tool_calls) == 0
    assert len(last.tool_steps) == 0
    for turn, (name, args) in zip(traj.turns[:-1], calls):
        assert [c.name for c in turn.message.tool_calls] == [name]
        assert _cj(turn.message.tool_calls[0].args) == _cj(args)
        assert len(turn.tool_steps) == 1
        assert turn.tool_steps[0].tool == name
        assert _cj(turn.tool_steps[0].args) == _cj(args)

    expected_env = _env_after(task, calls)
    assert _cj(traj.final_snapshot) == _cj(expected_env.snapshot())
    assert _records_cj(_all_steps(traj)) == _records_cj(expected_env.records)

    assert type(traj.max_turns) is int and traj.max_turns == max_turns
    assert traj.parent is None
    assert _cj(traj.task) == _cj(task.to_dict())
    assert _cj(traj.agent.to_dict()) == _cj(policy.agent_config.to_dict())

    traj.validate()
    traj.verify()
    report = traj.grade()
    assert report.success is True
    assert report.trajectory_safe is True


def test_run_episode_does_not_modify_task() -> None:
    task = get_smoke_task(LONG_ID)
    before = _cj(task.to_dict())
    run_episode(task, _oracle(LONG_ID), max_turns=20, clock=_zero_clock)
    assert _cj(task.to_dict()) == before


def test_run_episode_is_deterministic_with_constant_clock() -> None:
    a = run_episode(get_smoke_task(LONG_ID), _oracle(LONG_ID), max_turns=10, clock=_zero_clock)
    b = run_episode(get_smoke_task(LONG_ID), _oracle(LONG_ID), max_turns=10, clock=_zero_clock)
    assert _cj(a.to_dict()) == _cj(b.to_dict())


def test_immediate_final_answer_with_max_turns_one() -> None:
    task = get_smoke_task(NOOP_ID)
    traj = run_episode(task, _TurnPolicy(_script_steps([])), max_turns=1, clock=_zero_clock)
    assert traj.termination == "final_answer"
    assert len(traj.turns) == 1
    assert len(traj.turns[0].tool_steps) == 0
    assert _cj(traj.final_snapshot) == _cj(_env_after(task, []).snapshot())
    traj.validate()
    traj.verify()
    report = traj.grade()
    assert report.success is True and report.trajectory_safe is True


def test_multi_call_turns_record_steps_in_call_order() -> None:
    task = get_smoke_task(LONG_ID)
    script = [
        [("search_issues", {"assignee_id": "u1"}), ("get_issue", {"issue_id": 2})],
        [("assign_issue", {"issue_id": 2, "assignee_id": "u2"})],
        [
            ("add_comment", {"issue_id": 2, "body": "Now owned by Bob Li."}),
            ("get_issue", {"issue_id": 2}),
            ("get_issue", {"issue_id": 4}),
        ],
    ]
    traj = run_episode(task, ScriptedPolicy(script), max_turns=10, clock=_zero_clock)
    assert traj.termination == "final_answer"
    assert len(traj.turns) == len(script) + 1
    for turn, turn_calls in zip(traj.turns, script):
        assert len(turn.tool_steps) == len(turn_calls) == len(turn.message.tool_calls)
        for step, call, (name, args) in zip(turn.tool_steps, turn.message.tool_calls, turn_calls):
            assert step.tool == call.name == name
            assert _cj(step.args) == _cj(call.args) == _cj(args)
    flat = [c for turn_calls in script for c in turn_calls]
    assert [s.step for s in _all_steps(traj)] == list(range(len(flat)))
    expected_env = _env_after(task, flat)
    assert _records_cj(_all_steps(traj)) == _records_cj(expected_env.records)
    assert _cj(traj.final_snapshot) == _cj(expected_env.snapshot())
    traj.validate()
    traj.verify()


# ---------------------------------------------------------------- run_episode：max_turns 截斷


@pytest.mark.parametrize("max_turns", [1, 2, 6, 7])
def test_max_turns_truncates_oracle(max_turns: int) -> None:
    task = get_smoke_task(LONG_ID)
    calls = REFERENCE_SOLUTIONS[LONG_ID]
    assert max_turns <= len(calls)
    clock = _CountingClock()
    traj = run_episode(task, _oracle(LONG_ID), max_turns=max_turns, clock=clock)
    assert traj.termination == "max_turns"
    assert len(traj.turns) == max_turns
    assert len(traj.turns[-1].message.tool_calls) > 0
    assert clock.calls == 2 * max_turns
    expected_env = _env_after(task, calls[:max_turns])
    assert _cj(traj.final_snapshot) == _cj(expected_env.snapshot())
    assert _records_cj(_all_steps(traj)) == _records_cj(expected_env.records)
    assert traj.max_turns == max_turns
    traj.validate()
    traj.verify()


def test_max_turns_one_with_multiple_tool_calls() -> None:
    task = get_smoke_task(SHORT_ID)
    calls = [
        ("get_issue", {"issue_id": 8}),
        ("search_issues", {"label": "security"}),
        ("assign_issue", {"issue_id": 8, "assignee_id": "u4"}),
    ]
    traj = run_episode(task, ScriptedPolicy([calls]), max_turns=1, clock=_zero_clock)
    assert traj.termination == "max_turns"
    assert len(traj.turns) == 1
    turn = traj.turns[0]
    assert len(turn.message.tool_calls) == 3
    assert [s.step for s in turn.tool_steps] == [0, 1, 2]
    assert [s.tool for s in turn.tool_steps] == [name for name, _ in calls]
    expected_env = _env_after(task, calls)
    assert _cj(traj.final_snapshot) == _cj(expected_env.snapshot())
    traj.validate()
    traj.verify()


# ---------------------------------------------------------------- clock 與 usage


def test_clock_called_exactly_twice_per_turn_around_next_message() -> None:
    task = get_smoke_task(LONG_ID)
    n_turns = len(REFERENCE_SOLUTIONS[LONG_ID]) + 1
    rng = random.Random(20261008)
    values: list[float] = []
    t = 0.0
    for _ in range(2 * n_turns):
        t += rng.randint(1, 64) / 8  # 二進位可精確表示，差值無捨入誤差
        values.append(t)
    clock = _CountingClock(values)
    spy = _Spy(_oracle(LONG_ID), probe=lambda: clock.calls)
    traj = run_episode(task, spy, max_turns=n_turns + 3, clock=clock)

    assert len(traj.turns) == n_turns
    assert clock.calls == 2 * n_turns
    # 第 i 次呼叫 policy 時，clock 剛好已被呼叫 2i+1 次（t0 在 next_message 之前）。
    assert spy.probes == [2 * i + 1 for i in range(n_turns)]
    for i, turn in enumerate(traj.turns):
        assert _cj(turn.usage.latency_s) == _cj(values[2 * i + 1] - values[2 * i])


def test_integer_counter_clock_gives_unit_latency_each_turn() -> None:
    counter = itertools.count()
    traj = run_episode(get_smoke_task(SHORT_ID), _oracle(SHORT_ID), max_turns=5, clock=lambda: next(counter))
    assert next(counter) == 2 * len(traj.turns)
    for turn in traj.turns:
        assert turn.usage.latency_s == 1
        assert _cj(turn.usage.latency_s) == "1"  # int 差值原樣保存，不是 1.0


@pytest.mark.parametrize("clock_value", [0, 0.0, 2.5], ids=repr)
def test_constant_clock_latency_is_exact_difference(clock_value: float) -> None:
    """0 與 0.0 在 Python == 下相同，但 canonical 不同。"""
    traj = run_episode(get_smoke_task(SHORT_ID), _oracle(SHORT_ID), max_turns=5, clock=lambda: clock_value)
    for turn in traj.turns:
        assert _cj(turn.usage.latency_s) == _cj(clock_value - clock_value)


def test_scripted_turn_usage_with_zero_clock() -> None:
    traj = run_episode(get_smoke_task(LONG_ID), _oracle(LONG_ID), max_turns=10, clock=_zero_clock)
    expected = _cj(Usage(0, 0, 0.0, 0.0, 1).to_dict())
    for turn in traj.turns:
        assert _cj(turn.usage.to_dict()) == expected


def test_turn_usage_comes_from_policy_step_fields() -> None:
    task = get_smoke_task(SHORT_ID)
    script = [[("get_issue", {"issue_id": 8})], [("get_issue", {"issue_id": 1})]]
    # (input_tokens, output_tokens, cost_usd)；cost 1（int）與 1.0 在 Python == 下相同。
    usages = [(11, 3, 0.25), (None, 7, None), (5, None, 1)]

    def make(k: int) -> PolicyStep:
        calls = script[k] if k < len(script) else ()
        i, o, c = usages[k]
        return PolicyStep(_message(k, calls), input_tokens=i, output_tokens=o, cost_usd=c)

    counter = itertools.count()
    traj = run_episode(task, _TurnPolicy(make), max_turns=5, clock=lambda: next(counter))
    assert len(traj.turns) == 3
    for turn, (i, o, c) in zip(traj.turns, usages):
        assert isinstance(turn.usage, Usage)
        assert _cj(turn.usage.to_dict()) == _cj(Usage(i, o, 1, c, 1).to_dict())
    traj.validate()


# ---------------------------------------------------------------- PolicyStep 型別與例外


@pytest.mark.parametrize("kind", ["model_message", "none", "dict", "tuple"])
def test_non_policy_step_raises_type_error(kind: str) -> None:
    def make(k: int) -> Any:
        msg = _message(k, [("get_issue", {"issue_id": 8})])
        return {"model_message": msg, "none": None, "dict": {"message": msg}, "tuple": (msg, 0, 0, 0.0)}[kind]

    policy = _TurnPolicy(make)
    with pytest.raises(TypeError):
        run_episode(get_smoke_task(SHORT_ID), policy, max_turns=3, clock=_zero_clock)
    assert len(policy.inputs) == 1


def test_non_policy_step_on_later_turn_does_not_execute_its_calls() -> None:
    """型別檢查在執行 tool calls 之前：第二個 turn 回傳非 PolicyStep 時，env 只有第一個 turn 的 call。"""
    task = get_smoke_task(SHORT_ID)

    def make(k: int) -> Any:
        msg = _message(k, [("assign_issue", {"issue_id": 8, "assignee_id": "u4"})])
        return PolicyStep(msg) if k == 0 else msg

    env = _restored(_env_after(task, []))
    policy = _TurnPolicy(make)
    with pytest.raises(TypeError):
        _continue_episode(task, policy, max_turns=5, env=env, prefix_turns=(), parent=None, clock=_zero_clock)
    assert len(policy.inputs) == 2
    expected = _env_after(task, [("assign_issue", {"issue_id": 8, "assignee_id": "u4"})])
    assert _cj(env.snapshot()) == _cj(expected.snapshot())


class _PolicyFailure(Exception):
    pass


@pytest.mark.parametrize("fail_at", [0, 1, 2])
# 不用 RuntimeError：NotImplementedError 是它的子類，紅燈階段會混淆。
@pytest.mark.parametrize("exc_type", [_PolicyFailure, ValueError, KeyError])
def test_policy_exception_propagates_unchanged(fail_at: int, exc_type: type[Exception]) -> None:
    error = exc_type("policy failed")
    inner = _script_steps([[("get_issue", {"issue_id": i})] for i in (8, 1, 2)])

    def make(k: int) -> PolicyStep:
        if k == fail_at:
            raise error
        return inner(k)

    policy = _TurnPolicy(make)
    with pytest.raises(exc_type) as excinfo:
        run_episode(get_smoke_task(SHORT_ID), policy, max_turns=10, clock=_zero_clock)
    assert excinfo.value is error
    assert len(policy.inputs) == fail_at + 1


# ---------------------------------------------------------------- agent_config 與 conversation


def test_agent_config_is_read_once_and_used_for_whole_episode() -> None:
    task = get_smoke_task(SHORT_ID)
    policy = _TurnPolicy(
        _script_steps([[("get_issue", {"issue_id": 8})], [("get_issue", {"issue_id": 1})]]),
        agent=lambda n: _agent(f"v{n}"),
    )
    traj = run_episode(task, policy, max_turns=5, clock=_zero_clock)
    assert policy.agent_reads == 1
    first = _agent("v0")
    assert _cj(traj.agent.to_dict()) == _cj(first.to_dict())
    assert traj.agent.config_id == first.config_id
    assert len(policy.inputs) == 3
    for i, (messages, _) in enumerate(policy.inputs):
        assert messages[0] == {"role": "system", "content": "system-v0"}
        assert messages[1] == {"role": "developer", "content": "developer-v0"}
        assert _dumps(messages) == _dumps(render_messages(task.instruction, first, traj.turns[:i]))
    traj.validate()


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_policy_sees_rendered_conversation_and_tool_specs(task_id: str) -> None:
    task = get_smoke_task(task_id)
    spy = _Spy(_oracle(task_id, system_prompt="You triage issues.", developer_prompt="Use the tools."))
    expected_tools = _dumps(IssueTrackerEnv.tool_specs())
    traj = run_episode(task, spy, max_turns=20, clock=_zero_clock)

    assert len(spy.inputs) == len(traj.turns)
    for i, (messages, tools) in enumerate(spy.inputs):
        assert _dumps(messages) == _dumps(render_messages(task.instruction, traj.agent, traj.turns[:i]))
        assert _dumps(tools) == expected_tools
    for messages, tools in spy.raw:
        assert type(messages) is list
        assert isinstance(tools, tuple)
    # 每次傳入的都是新物件（spy.raw 持有參照，id 不會被重用）。
    assert len({id(m) for m, _ in spy.raw}) == len(spy.raw)
    assert len({id(t) for _, t in spy.raw}) == len(spy.raw)
    assert _dumps(traj.messages()) == _dumps(render_messages(task.instruction, traj.agent, traj.turns))


def test_non_ascii_conversation_round_trips(small_world: dict[str, Any]) -> None:
    task = _small_task(small_world)
    script = [
        [("get_issue", {"issue_id": 2})],
        [
            ("assign_issue", {"issue_id": 2, "assignee_id": "u3"}),
            ("add_comment", {"issue_id": 2, "body": "已改派給 Fay Frontend（u3）。"}),
        ],
    ]
    spy = _Spy(ScriptedPolicy(script, system_prompt="你是分派機器人。", final_text="完成。"))
    traj = run_episode(task, spy, max_turns=5, clock=_zero_clock)
    assert traj.termination == "final_answer"
    for i, (messages, _) in enumerate(spy.inputs):
        assert _dumps(messages) == _dumps(render_messages(task.instruction, traj.agent, traj.turns[:i]))
    last_messages = spy.inputs[-1][0]
    assert any("登入頁面字型錯誤" in m["content"] for m in last_messages if m["role"] == "tool")
    traj.validate()
    traj.verify()
    assert traj.grade().success is True


class _MutatingPolicy:
    """先以輸入的深拷貝產生回應，再徹底破壞收到的 messages / tools。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.inputs: list[tuple[Any, Any]] = []

    @property
    def agent_config(self) -> Any:
        return self.inner.agent_config

    def next_message(self, messages: Any, tools: Any) -> Any:
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        step = self.inner.next_message(copy.deepcopy(messages), copy.deepcopy(tools))
        for m in messages:
            m["content"] = "tampered"
            for call in m.get("tool_calls", []):
                call["name"] = "tampered"
                call["id"] = "tampered"
                if isinstance(call.get("args"), dict):
                    call["args"]["tampered"] = True
                    call["args"].pop("issue_id", None)
        messages.append({"role": "assistant", "content": "injected", "tool_calls": []})
        messages.insert(0, {"role": "user", "content": "injected"})
        for spec in tools:
            spec["description"] = "tampered"
            spec["input_schema"]["properties"].clear()
            spec["name"] = "tampered"
        return step


@pytest.mark.parametrize("task_id", [LONG_ID, "smoke-close-duplicate"])
def test_policy_mutating_inputs_does_not_affect_trajectory(task_id: str) -> None:
    task = get_smoke_task(task_id)
    specs_before = _dumps(IssueTrackerEnv.tool_specs())
    clean = run_episode(task, _oracle(task_id), max_turns=20, clock=_zero_clock)
    policy = _MutatingPolicy(_oracle(task_id))
    traj = run_episode(task, policy, max_turns=20, clock=_zero_clock)

    assert _cj(traj.to_dict()) == _cj(clean.to_dict())
    for i, (messages, tools) in enumerate(policy.inputs):
        assert _dumps(messages) == _dumps(render_messages(task.instruction, traj.agent, traj.turns[:i]))
        assert _dumps(tools) == specs_before
    assert _dumps(IssueTrackerEnv.tool_specs()) == specs_before
    traj.validate()
    traj.verify()


# ---------------------------------------------------------------- tool error 不是例外


def test_bad_tool_calls_are_recorded_as_tool_errors() -> None:
    task = get_smoke_task(SHORT_ID)
    half = len(BAD_CALLS) // 2
    script = [
        [(name, args) for name, args, _ in BAD_CALLS[:half]],
        [(name, args) for name, args, _ in BAD_CALLS[half:]],
    ]
    policy = _TurnPolicy(_script_steps(script))
    traj = run_episode(task, policy, max_turns=5, clock=_zero_clock)

    assert traj.termination == "final_answer"
    assert len(traj.turns) == 3
    steps = _all_steps(traj)
    codes = [code for _, _, code in BAD_CALLS]
    assert [s.ok for s in steps] == [False] * len(BAD_CALLS)
    assert [s.error["code"] for s in steps] == codes
    assert [s.step for s in steps] == list(range(len(BAD_CALLS)))
    for step, (name, args, _) in zip(steps, BAD_CALLS):
        assert step.tool == name
        assert _cj(step.args) == _cj(args)  # 8.0 / True 不可被當成 8
    assert _cj(traj.final_snapshot["state"]) == _cj(task.initial_state)
    assert traj.final_snapshot["step"] == len(BAD_CALLS)
    # policy 看到的 tool 訊息就是這些錯誤 observation。
    last_messages = policy.inputs[-1][0]
    assert _dumps(last_messages) == _dumps(render_messages(task.instruction, traj.agent, traj.turns[:2]))
    traj.validate()
    traj.verify()
    report = traj.grade()
    assert [e.code for e in report.tool_errors] == codes
    assert report.mutating_steps == 0


# ---------------------------------------------------------------- _continue_episode：合法前綴


@dataclasses.dataclass
class _Setup:
    task_id: str
    task: Task
    calls: tuple[tuple[str, Any], ...]
    full: Trajectory
    k: int
    prefix: tuple[Turn, ...]
    env: IssueTrackerEnv
    parent: dict[str, Any] | None
    max_turns: int


def _setup(k: int, *, task_id: str = LONG_ID, env_kind: str = "restored") -> _Setup:
    """真實 oracle run 的前 k 個 turns，env 位於對應的決策點。"""
    task = get_smoke_task(task_id)
    calls = tuple(copy.deepcopy(REFERENCE_SOLUTIONS[task_id]))
    max_turns = len(calls) + 1
    full = run_episode(task, _oracle(task_id), max_turns=max_turns, clock=_zero_clock)
    prefix = tuple(full.turns[:k])
    replayed = _env_after(task, calls[:k])
    env = _restored(replayed) if env_kind == "restored" else replayed
    parent = None if k == 0 else {"trajectory_id": "source-trajectory", "checkpoint_id": "source-checkpoint", "turn": k}
    return _Setup(task_id, task, calls, full, k, prefix, env, parent, max_turns)


def _continue(s: _Setup, *, policy: Any = None, clock: Any = None, **overrides: Any) -> Trajectory:
    kwargs: dict[str, Any] = {
        "max_turns": s.max_turns,
        "env": s.env,
        "prefix_turns": s.prefix,
        "parent": s.parent,
        "clock": _zero_clock if clock is None else clock,
    }
    kwargs.update(overrides)
    return _continue_episode(s.task, _oracle(s.task_id) if policy is None else policy, **kwargs)


def _assert_rejected(
    s: _Setup, exc_type: type[Exception] = ValueError, *, policy: Any = None, **overrides: Any
) -> None:
    """入口檢查失敗：raise exc_type、policy 與 clock 都沒被呼叫、env 不變。"""
    env = overrides.get("env", s.env)
    before = _env_state(env)
    spy = _Spy(_oracle(s.task_id) if policy is None else policy)
    clock = _CountingClock()
    with pytest.raises(exc_type):
        _continue(s, policy=spy, clock=clock, **overrides)
    assert spy.inputs == []
    assert clock.calls == 0
    assert _env_state(env) == before


@pytest.mark.parametrize("env_kind", ["restored", "replayed"])
@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_continue_from_valid_prefix_reproduces_full_run(k: int, env_kind: str) -> None:
    s = _setup(k, env_kind=env_kind)
    spy = _Spy(_oracle(LONG_ID))
    clock = _CountingClock()
    result = _continue(s, policy=spy, clock=clock)

    assert isinstance(result, Trajectory)
    result.validate()
    result.verify()
    # ScriptedPolicy + 常數 clock：續跑結果與原 episode 逐 byte 相同。
    assert _turns_cj(result.turns) == _turns_cj(s.full.turns)
    assert _turns_cj(result.turns[:k]) == _turns_cj(s.prefix)
    assert result.termination == "final_answer"
    assert _cj(result.final_snapshot) == _cj(s.full.final_snapshot)
    assert _cj(result.final_snapshot) == _cj(s.env.snapshot())
    assert _cj(result.parent) == _cj(s.parent)
    assert _cj(result.task) == _cj(s.task.to_dict())
    assert result.max_turns == s.max_turns
    assert [st.step for st in _all_steps(result)] == list(range(len(s.calls)))

    new_turns = len(s.full.turns) - k
    assert len(spy.inputs) == new_turns
    assert clock.calls == 2 * new_turns
    assert spy.agent_reads == 1
    assert _dumps(spy.inputs[0][0]) == _dumps(render_messages(s.task.instruction, result.agent, s.prefix))
    assert result.grade().success is True


def test_continue_accepts_list_prefix() -> None:
    s = _setup(3)
    result = _continue(s, prefix_turns=list(s.prefix))
    result.validate()
    assert _turns_cj(result.turns) == _turns_cj(s.full.turns)


@pytest.mark.parametrize("max_turns", [3, 5])
def test_continue_counts_prefix_toward_max_turns(max_turns: int) -> None:
    s = _setup(2)
    clock = _CountingClock()
    result = _continue(s, max_turns=max_turns, clock=clock)
    assert result.termination == "max_turns"
    assert len(result.turns) == max_turns
    assert result.max_turns == max_turns
    assert clock.calls == 2 * (max_turns - 2)
    assert _turns_cj(result.turns) == _turns_cj(s.full.turns[:max_turns])
    assert _cj(result.final_snapshot) == _cj(_env_after(s.task, s.calls[:max_turns]).snapshot())
    result.validate()
    result.verify()


def test_continue_keeps_prefix_turn_usage_unchanged() -> None:
    """decision 006：rerun trajectory 的 usage 包含前綴 turns 的原始量測。重放只比 records，不看 usage，
    因此前綴 turn 必須原樣保留（int 1 / 0 不可變成 1.0 / 0.0）。"""
    s = _setup(3)
    prefix_usages = [Usage(5, 7, 1.25, 0.5, 1), Usage(5, 7, 1, 0, 1), Usage(5, 7, 1.25, 0.5, 1)]
    assert prefix_usages[1] == Usage(5, 7, 1.0, 0.0, 1)  # Python == 會被騙
    prefix = tuple(R(t, usage=u) for t, u in zip(s.prefix, prefix_usages))
    result = _continue(s, prefix_turns=prefix)

    result.validate()
    result.verify()
    assert _turns_cj(result.turns[:3]) == _turns_cj(prefix)
    zero = _cj(Usage(0, 0, 0.0, 0.0, 1).to_dict())
    for turn in result.turns[3:]:
        assert _cj(turn.usage.to_dict()) == zero
    # 決定性內容（index / message / tool_steps）與原 episode 相同。
    assert _content_cj(result.turns) == _content_cj(s.full.turns)

    usages = [t.usage for t in prefix] + [Usage(0, 0, 0.0, 0.0, 1)] * (len(result.turns) - 3)
    expected = Usage(
        sum((u.input_tokens for u in usages), 0),
        sum((u.output_tokens for u in usages), 0),
        sum((u.latency_s for u in usages), 0),
        sum((u.cost_usd for u in usages), 0),
        len(result.turns),
    )
    assert _cj(result.usage().to_dict()) == _cj(expected.to_dict())


class _EnvFailure(Exception):
    pass


class _FailingEnv(IssueTrackerEnv):
    """本實例第 fail_on 次 call_tool 時 raise error（不委派給 env）；fail_on 為 None 時行為與 IssueTrackerEnv 相同。

    設定放在實例屬性，因此 runner 為了重放而另建的 env（即使是 ``type(env)()``）不會失敗。
    """

    def __init__(self) -> None:
        super().__init__()
        self.fail_on: int | None = None
        self.error: BaseException | None = None
        self.tool_calls = 0

    def call_tool(self, name: str, args: Any) -> dict[str, Any]:
        if self.fail_on is not None:
            self.tool_calls += 1
            if self.tool_calls == self.fail_on:
                assert self.error is not None
                raise self.error
        return super().call_tool(name, args)


def _failing_env(snapshot: dict[str, Any], fail_on: int, error: BaseException) -> _FailingEnv:
    env = _FailingEnv()
    env.restore(snapshot)
    env.fail_on = fail_on
    env.error = error
    return env


@pytest.mark.parametrize("exc_type", [_EnvFailure, ValueError])
@pytest.mark.parametrize("fail_on", [1, 3])
@pytest.mark.parametrize("k", [0, 2])
def test_env_exception_propagates_unchanged(k: int, fail_on: int, exc_type: type[Exception]) -> None:
    """policy 或 env 拋出的例外原樣往外拋（不包裝、不產生 trajectory）。oracle 每個 turn 一個 call。"""
    s = _setup(k)
    error = exc_type("env failed")
    env = _failing_env(s.env.snapshot(), fail_on, error)
    spy = _Spy(_oracle(LONG_ID))
    with pytest.raises(exc_type) as excinfo:
        _continue(s, policy=spy, env=env)
    assert excinfo.value is error
    assert env.tool_calls == fail_on
    assert len(spy.inputs) == fail_on


def test_env_exception_inside_multi_call_turn_propagates_unchanged() -> None:
    s = _setup(0, task_id=SHORT_ID)
    error = _EnvFailure("env failed")
    env = _failing_env(s.env.snapshot(), 2, error)
    policy = _TurnPolicy(_script_steps([[("get_issue", {"issue_id": i}) for i in (8, 1, 2)]]))
    with pytest.raises(_EnvFailure) as excinfo:
        _continue(s, policy=policy, env=env)
    assert excinfo.value is error
    assert env.tool_calls == 2
    assert len(policy.inputs) == 1


# ---------------------------------------------------------------- _continue_episode：入口檢查


@pytest.mark.parametrize("max_turns", [True, 0, -1, 8.0, "8", None], ids=repr)
def test_continue_rejects_bad_max_turns(max_turns: Any) -> None:
    """空前綴時 True（== 1）也必須被拒絕。"""
    _assert_rejected(_setup(0), max_turns=max_turns)


@pytest.mark.parametrize("delta", [0, -1, -3])
def test_continue_rejects_prefix_not_shorter_than_max_turns(delta: int) -> None:
    _assert_rejected(_setup(3), max_turns=3 + delta)


def test_continue_rejects_shifted_turn_indexes() -> None:
    """index 整體 +1：steps 連續、重放也一致，只有 index 不符。"""
    s = _setup(2)
    prefix = (R(s.prefix[0], index=1), R(s.prefix[1], index=2))
    _assert_rejected(s, prefix_turns=prefix)


@pytest.mark.parametrize("kind", ["swapped", "duplicate"])
def test_continue_rejects_bad_turn_order(kind: str) -> None:
    s = _setup(2)
    if kind == "swapped":
        prefix = (s.prefix[1], s.prefix[0])
    else:
        prefix = (s.prefix[0], R(s.prefix[1], index=0))
    _assert_rejected(s, prefix_turns=prefix)


@pytest.mark.parametrize("k", [0, 2])
def test_continue_rejects_prefix_turn_without_tool_calls(k: int) -> None:
    s = _setup(k)
    final_turn = s.full.turns[-1]
    assert len(final_turn.message.tool_calls) == 0
    prefix = s.prefix + (R(final_turn, index=k),)
    parent = {"trajectory_id": "source-trajectory", "checkpoint_id": "source-checkpoint", "turn": k + 1}
    _assert_rejected(s, prefix_turns=prefix, parent=parent)


def _shift_steps(turn: Turn, delta: int) -> Turn:
    return R(turn, tool_steps=tuple(R(r, step=r.step + delta) for r in turn.tool_steps))


@pytest.mark.parametrize("kind", ["all_shifted", "gap", "repeat"])
def test_continue_rejects_non_contiguous_steps(kind: str) -> None:
    s = _setup(3)
    p = s.prefix
    if kind == "all_shifted":
        prefix = tuple(_shift_steps(t, 1) for t in p)
    elif kind == "gap":
        prefix = (p[0], _shift_steps(p[1], 1), _shift_steps(p[2], 1))
    else:
        prefix = (p[0], _shift_steps(p[1], -1), p[2])
    _assert_rejected(s, prefix_turns=prefix)


@pytest.mark.parametrize("k", [0, 2])
def test_continue_rejects_env_actor_mismatch(k: int) -> None:
    s = _setup(k)
    env = _env_after(s.task, s.calls[:k], actor_id="u2")
    _assert_rejected(s, env=env)


def _env_initial(s: _Setup) -> IssueTrackerEnv:
    return _restored(_env_after(s.task, []))


def _env_behind(s: _Setup) -> IssueTrackerEnv:
    return _restored(_env_after(s.task, s.calls[: s.k - 1]))


def _env_ahead(s: _Setup) -> IssueTrackerEnv:
    return _restored(_env_after(s.task, s.calls[: s.k + 1]))


def _env_extra_read_only_call(s: _Setup) -> IssueTrackerEnv:
    s.env.call_tool("get_issue", {"issue_id": 1})
    return s.env


def _env_extra_failed_call(s: _Setup) -> IssueTrackerEnv:
    s.env.call_tool("no_such_tool", {})
    return s.env


def _env_other_mutation(s: _Setup) -> IssueTrackerEnv:
    # 同樣的 call 數，但第 2 個 call 指派給別人：step 相同、state 不同。
    assert s.calls[1] == ("assign_issue", {"issue_id": 2, "assignee_id": "u2"})
    return _restored(_env_after(s.task, [s.calls[0], ("assign_issue", {"issue_id": 2, "assignee_id": "u3"})]))


def _env_step_shifted(s: _Setup) -> IssueTrackerEnv:
    # state 與 hash 相同，只有 step 不同（step 不在 hash 內）。
    snap = s.env.snapshot()
    snap["step"] += 3
    env = IssueTrackerEnv()
    env.restore(snap)
    return env


ENV_MISMATCH_K2 = [
    _env_initial,
    _env_behind,
    _env_ahead,
    _env_extra_read_only_call,
    _env_extra_failed_call,
    _env_other_mutation,
    _env_step_shifted,
]


@pytest.mark.parametrize("make_env", ENV_MISMATCH_K2, ids=lambda f: f.__name__)
def test_continue_rejects_env_not_matching_prefix(make_env: Callable[[_Setup], IssueTrackerEnv]) -> None:
    s = _setup(2)
    _assert_rejected(s, env=make_env(s))


@pytest.mark.parametrize(
    "make_env", [_env_ahead, _env_extra_read_only_call, _env_extra_failed_call, _env_step_shifted],
    ids=lambda f: f.__name__,
)
def test_continue_rejects_env_not_at_initial_state_for_empty_prefix(
    make_env: Callable[[_Setup], IssueTrackerEnv],
) -> None:
    s = _setup(0)
    _assert_rejected(s, env=make_env(s))


def _replace_step(prefix: tuple[Turn, ...], turn_i: int, **changes: Any) -> tuple[tuple[Turn, ...], Any, Any]:
    turn = prefix[turn_i]
    original = turn.tool_steps[0]
    tampered = R(original, **changes)
    new_turn = R(turn, tool_steps=(tampered,) + tuple(turn.tool_steps[1:]))
    return prefix[:turn_i] + (new_turn,) + prefix[turn_i + 1 :], original, tampered


@pytest.mark.parametrize(
    "changes",
    [
        {"args": {"issue_id": 2.0, "assignee_id": "u2"}},
        {"ok": 1},
    ],
    ids=["args_int_to_float", "ok_true_to_1"],
)
def test_continue_rejects_prefix_record_equal_only_under_python_eq(changes: dict[str, Any]) -> None:
    """竄改後的 record 在 Python == 下與原 record 相同，但 canonical 不同 → 必須被拒絕。"""
    s = _setup(2)
    prefix, original, tampered = _replace_step(s.prefix, 1, **changes)
    assert tampered == original
    assert _cj(tampered.to_dict()) != _cj(original.to_dict())
    _assert_rejected(s, prefix_turns=prefix)


@pytest.mark.parametrize(
    "changes",
    [
        {"state_hash_after": "sha256:" + "0" * 64},
        {"changes": ()},
        {"result": {"tampered": True}},
        {"args": {"issue_id": 4, "assignee_id": "u2"}},
    ],
    ids=["hash_after", "changes_dropped", "result", "args_other_issue"],
)
def test_continue_rejects_tampered_prefix_record(changes: dict[str, Any]) -> None:
    s = _setup(2)
    prefix, _, _ = _replace_step(s.prefix, 1, **changes)
    _assert_rejected(s, prefix_turns=prefix)


def test_continue_rejects_prefix_message_args_equal_only_under_python_eq() -> None:
    """message 的 call args 2 → 2.0（Python == 相同）：重放會得到不同的 record。"""
    s = _setup(2)
    turn = s.prefix[1]
    call = turn.message.tool_calls[0]
    new_args = {**call.args, "issue_id": float(call.args["issue_id"])}
    new_call = ToolCall(id=call.id, name=call.name, args=new_args)
    assert new_call == call
    new_message = ModelMessage(text=turn.message.text, tool_calls=(new_call,), stop_reason=turn.message.stop_reason)
    prefix = (s.prefix[0], R(turn, message=new_message))
    _assert_rejected(s, prefix_turns=prefix)


def test_continue_rejects_prefix_with_missing_tool_step() -> None:
    s = _setup(2)
    prefix = (s.prefix[0], R(s.prefix[1], tool_steps=()))
    _assert_rejected(s, prefix_turns=prefix)


@pytest.mark.parametrize("kind", ["steps_lag_calls", "steps_lead_calls"])
def test_continue_rejects_prefix_with_misaligned_steps_and_calls(kind: str) -> None:
    """攤平後 calls 與 records 完全一致（index、每 turn 至少一個 call、step 連續、攤平重放都通過），
    但某個 turn 的 tool_steps 與 message.tool_calls 不一一對應。

    依 Turn 的定義（trajectory 模組 docstring：tool_steps 與 message.tool_calls 一一對應），重放第 i 個 turn
    第 j 個 call 所得的 record 必須等於 turns[i].tool_steps[j]；不對應即入口檢查失敗，policy 不能看到錯位的 conversation。
    """
    s = _setup(3)
    a, b, c = (t.message.tool_calls[0] for t in s.prefix)
    ra, rb, rc = (t.tool_steps[0] for t in s.prefix)
    if kind == "steps_lag_calls":
        layout = [((a, b), (ra,)), ((c,), (rb, rc))]
    else:
        layout = [((a,), (ra, rb)), ((b, c), (rc,))]
    prefix = tuple(
        Turn(i, ModelMessage(text="", tool_calls=calls, stop_reason="tool_use"), steps, s.prefix[i].usage)
        for i, (calls, steps) in enumerate(layout)
    )
    flat_calls = [call for t in prefix for call in t.message.tool_calls]
    flat_steps = [step for t in prefix for step in t.tool_steps]
    assert [call.name for call in flat_calls] == [step.tool for step in flat_steps]
    assert [step.step for step in flat_steps] == [0, 1, 2]
    parent = {"trajectory_id": "source-trajectory", "checkpoint_id": "source-checkpoint", "turn": 2}
    _assert_rejected(s, prefix_turns=prefix, parent=parent)


@pytest.mark.parametrize("kind", ["turn_dict", "none", "index_false"])
def test_continue_rejects_prefix_element_not_turn_or_bool_index(kind: str) -> None:
    """前綴元素必須是 Turn、index 必須是 int（decision 006：所有 int 都排除 bool；False == 0 不可通過）。"""
    s = _setup(1)
    if kind == "turn_dict":
        element: Any = s.prefix[0].to_dict()
    elif kind == "none":
        element = None
    else:
        element = R(s.prefix[0], index=False)
        assert element.index == 0  # Python == 會被騙
    _assert_rejected(s, prefix_turns=(element,))


def test_continue_rejects_missing_parent_for_non_empty_prefix() -> None:
    _assert_rejected(_setup(2), parent=None)


def test_continue_accepts_parent_turn_zero_for_empty_prefix() -> None:
    """checkpoint k = 0 合法（decision 006：0 <= k < max_turns），rerun 一律傳非 None 的 parent，
    Trajectory 規則 8 也允許 parent.turn == 0；因此空前綴 + parent.turn == 0 必須被接受。
    （runner docstring 的「parent 為 None ⇔ 前綴為空」與此矛盾，已回報需修訂為「parent 為 None ⇒ 前綴為空」。）
    """
    s = _setup(0)
    parent = {"trajectory_id": "source-trajectory", "checkpoint_id": "source-checkpoint", "turn": 0}
    spy = _Spy(_oracle(LONG_ID))
    result = _continue(s, policy=spy, parent=parent)
    result.validate()
    result.verify()
    assert _cj(result.parent) == _cj(parent)
    assert _turns_cj(result.turns) == _turns_cj(s.full.turns)
    assert _cj(result.final_snapshot) == _cj(s.full.final_snapshot)
    assert len(spy.inputs) == len(s.full.turns)


@pytest.mark.parametrize("turn", [1, -1, 7])
def test_continue_rejects_parent_turn_nonzero_for_empty_prefix(turn: int) -> None:
    parent = {"trajectory_id": "source-trajectory", "checkpoint_id": "source-checkpoint", "turn": turn}
    _assert_rejected(_setup(0), parent=parent)


@pytest.mark.parametrize("turn", [False, 0.0], ids=repr)
def test_continue_never_returns_trajectory_with_ill_typed_parent_turn_for_empty_prefix(turn: Any) -> None:
    """False == 0 == 0.0（Python ==），但 parent.turn 必須是 int（Trajectory 規則 8）；至少回傳前要被擋下。"""
    assert turn == 0
    parent = {"trajectory_id": "source-trajectory", "checkpoint_id": "source-checkpoint", "turn": turn}
    with pytest.raises(ValueError):
        _continue(_setup(0), parent=parent)


@pytest.mark.parametrize(
    "parent",
    [
        {"trajectory_id": "t", "turn": 2},
        {"checkpoint_id": "c", "turn": 2},
        {"trajectory_id": "t", "checkpoint_id": "c"},
        {"trajectory_id": "t", "checkpoint_id": "c", "turn": 2, "extra": "x"},
        {"trajectory": "t", "checkpoint_id": "c", "turn": 2},
        {},
    ],
    ids=["no_checkpoint_id", "no_trajectory_id", "no_turn", "extra_key", "renamed_key", "empty"],
)
def test_continue_rejects_parent_with_wrong_keys(parent: dict[str, Any]) -> None:
    _assert_rejected(_setup(2), parent=parent)


@pytest.mark.parametrize("turn", [0, 1, 3, -1, 7])
def test_continue_rejects_parent_turn_not_prefix_length(turn: int) -> None:
    parent = {"trajectory_id": "t", "checkpoint_id": "c", "turn": turn}
    _assert_rejected(_setup(2), parent=parent)


@pytest.mark.parametrize(
    "parent",
    [
        {"trajectory_id": "t", "checkpoint_id": "c", "turn": True},
        {"trajectory_id": "t", "checkpoint_id": "c", "turn": 1.0},
        {"trajectory_id": 123, "checkpoint_id": "c", "turn": 1},
        {"trajectory_id": "t", "checkpoint_id": None, "turn": 1},
    ],
    ids=["turn_bool", "turn_float", "trajectory_id_int", "checkpoint_id_none"],
)
def test_continue_never_returns_trajectory_with_ill_typed_parent(parent: dict[str, Any]) -> None:
    """runner 入口只明列 key 與 turn 值；型別錯誤至少要在回傳前被 validate() 擋下（TrajectoryError 是 ValueError）。"""
    s = _setup(1)
    with pytest.raises(ValueError):
        _continue(s, parent=parent)


@pytest.mark.parametrize("bad_agent", ["str", "dict", "none"])
def test_continue_rejects_non_agent_config(bad_agent: str) -> None:
    s = _setup(2)
    agent_value = {
        "str": "scripted",
        "dict": {"policy": "scripted", "prompt_version": "scripted/v1", "system_prompt": "", "developer_prompt": "",
                 "model_config": {}},
        "none": None,
    }[bad_agent]
    policy = _TurnPolicy(_script_steps([[("get_issue", {"issue_id": 1})]]), agent=lambda n: agent_value)
    before = _env_state(s.env)
    clock = _CountingClock()
    with pytest.raises(TypeError):
        _continue(s, policy=policy, clock=clock)
    assert policy.inputs == []
    assert clock.calls == 0
    assert _env_state(s.env) == before
