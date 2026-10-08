"""agents/synthetic.py 的契約測試（decision log 010 §1、§5）。

斷言只取自 ``agents/synthetic.py`` 的 docstring 與 decision 010：
- 受保護 issue = success_constraints 中 ``issue_unchanged`` 且涵蓋 comments（fields 省略或含 "comments"）者 id 最小；
- 錯誤 call = ``("add_comment", {"issue_id": protected, "body": MISTAKE_BODY})``；
- 決策隨機數 u = sha256(f"{seed}|{canonical_json(messages)}") 前 8 bytes / 2**64；
  u < q → 錯誤，否則 oracle[r]（r < n）或結束訊息；
- r = 非錯誤（「剛好一個 tool call 且 (name, canonical args) 等於錯誤 call」以外）的 assistant 訊息數；
- 解析解 P(success | k) = 0（前綴有錯誤）或 (1 − q)^(n − r + 1)。

未規定的細節（tool-call 訊息的 text / stop_reason、rollout_seed 型別錯的例外類別等）不斷言。
checkpoint_truth 目前未規定、因此不斷言的行為：k = True 是否算 int；非 synthetic policy 的 trajectory；
傳入的 mistake / mistake_rate 與 trajectory.agent.model_config 不一致時；max_turns 截斷的 trajectory 在 k = len(turns)
（docstring 允許，但 decision 010 §3 的合法範圍是 [0, len(turns) − 1]）。

真值語義的範圍：
- checkpoint_truth 只在 synthetic policy 自己產生的 trajectory（每個 turn 剛好一個 call：依序的 oracle call 或錯誤）上
  才是真正的 P(success | k)；本檔以 checkpoint 的 env 狀態（受保護 issue 是否已有 MISTAKE_BODY comment）與
  「無錯誤續跑」驗證這一點。
- 同一 turn 含多個 call、且其中之一是錯誤的歷史：docstring 的「剛好一個」規則給出 > 0，但錯誤已實際寫入受保護 issue
  （真正的機率為 0）。這是契約層的爭議（已回報），本檔對 policy 與 checkpoint_truth 都**不**斷言這種歷史的結果。
- 呼叫端在建構後修改 oracle_calls / mistake 不得改變 config_id 或行為：依據 decision 006「是否為同一個 agent 一律以
  config_id 判定」（config 在建構時固定，行為必須與之一致）以及 AgentConfig / ToolCall 的正規化深拷貝契約。
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import pytest

from agents.policy import AgentConfig, ModelMessage, PolicyStep, ToolCall, oracle_policy
from agents.synthetic import (
    MISTAKE_BODY,
    SYNTHETIC_POLICY,
    SYNTHETIC_PROMPT_VERSION,
    SyntheticPolicy,
    checkpoint_truth,
    mistake_call,
    protected_issue,
    success_probability,
    synthetic_agent_config,
)
from env.environment import IssueTrackerEnv
from env.hashing import canonical_json
from replay.checkpoint import make_checkpoint, rerun
from replay.runner import run_episode
from replay.trajectory import Trajectory, render_messages
from tasks.generator import week1_pool
from tasks.schema import Task
from tasks.smoke import REFERENCE_SOLUTIONS, SMOKE_TASK_IDS, get_smoke_task

# ---------------------------------------------------------------- 共用資料與 helper

_POOL = week1_pool("all")
_POOL_TASKS: dict[str, Task] = {inst.task.task_id: inst.task for inst in _POOL}
_ORACLES: dict[str, tuple[tuple[str, Any], ...]] = {
    **{tid: tuple(REFERENCE_SOLUTIONS[tid]) for tid in SMOKE_TASK_IDS},
    **{inst.task.task_id: tuple(inst.oracle_calls) for inst in _POOL},
}
ALL_TASK_IDS: tuple[str, ...] = (*SMOKE_TASK_IDS, *_POOL_TASKS)

# 手算的 smoke task 受保護 issue（由 tasks/smoke.py 的 success_constraints 逐條核對）。
SMOKE_PROTECTED = {
    "smoke-assign-and-start": 1,  # #8 的 fields 含 comments；其他 issue 全欄位 → 最小為 #1
    "smoke-close-duplicate": 2,  # 只有 #2 全欄位不變
    "smoke-escalate-security": 1,  # #4、#6 含 comments；其他全欄位 → #1
    "smoke-reassign-departed": 1,  # #2、#4、#10 不含 comments；#7 全欄位；#1 等含 comments → #1
    "smoke-conditional-noop": 5,
}

FINAL_TEXT = "Done."


def _clock() -> float:
    return 0.0


def _task(task_id: str) -> Task:
    """每次回傳新 Task 物件。"""
    if task_id in SMOKE_TASK_IDS:
        return get_smoke_task(task_id)
    return Task.from_dict(_POOL_TASKS[task_id].to_dict())


def _expected_protected(task: Task) -> int:
    ids = [
        p["issue_id"]
        for p in task.success_constraints
        if p["type"] == "issue_unchanged" and ("fields" not in p or "comments" in p["fields"])
    ]
    assert ids, f"{task.task_id} has no comment-covering issue_unchanged constraint"
    return min(ids)


def _expected_mistake(task: Task) -> tuple[str, dict[str, Any]]:
    return ("add_comment", {"issue_id": _expected_protected(task), "body": MISTAKE_BODY})


def _u(seed: int, messages: list[dict[str, Any]]) -> float:
    digest = hashlib.sha256(f"{seed}|{canonical_json(messages)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _is_mistake(calls: list[tuple[str, Any]], mistake: tuple[str, Any]) -> bool:
    return (
        len(calls) == 1
        and calls[0][0] == mistake[0]
        and canonical_json(calls[0][1]) == canonical_json(mistake[1])
    )


def _message_calls(message: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(call["name"], call["args"]) for call in message.get("tool_calls", [])]


def _turn_calls(turn: Any) -> list[tuple[str, Any]]:
    return [(call.name, call.args) for call in turn.message.tool_calls]


def _expected_next(
    messages: list[dict[str, Any]],
    *,
    seed: int,
    q: float,
    oracle: tuple[tuple[str, Any], ...],
    mistake: tuple[str, Any],
) -> tuple[str, Any] | None:
    """docstring 的決策規則；None 代表結束訊息。"""
    assistants = [m for m in messages if m["role"] == "assistant"]
    r = sum(1 for m in assistants if not _is_mistake(_message_calls(m), mistake))
    if _u(seed, messages) < q:
        return mistake
    if r < len(oracle):
        return oracle[r]
    return None


def _expected_truth(turns: Any, k: int, *, n: int, mistake: tuple[str, Any], q: float) -> float:
    prefix = turns[:k]
    if any(_is_mistake(_turn_calls(t), mistake) for t in prefix):
        return 0.0
    r = sum(1 for t in prefix if not _is_mistake(_turn_calls(t), mistake))
    return (1 - q) ** (n - r + 1)


def _policy(task: Task, q: float, seed: int) -> SyntheticPolicy:
    return SyntheticPolicy(_ORACLES[task.task_id], mistake=mistake_call(task), mistake_rate=q, rollout_seed=seed)


def _has_mistake(trajectory: Trajectory, mistake: tuple[str, Any]) -> bool:
    return any(_is_mistake(_turn_calls(t), mistake) for t in trajectory.turns)


def _assert_final_message(message: ModelMessage) -> None:
    assert message.tool_calls == ()
    assert message.text == FINAL_TEXT
    assert message.stop_reason == "end_turn"


def _check_decisions(
    trajectory: Trajectory, *, start: int, seed: int, q: float, oracle: tuple[tuple[str, Any], ...],
    mistake: tuple[str, Any],
) -> None:
    """index >= start 的每個 turn 都等於以「runner 傳給 policy 的 messages」重算的決策。"""
    instruction = trajectory.task["instruction"]
    for i in range(start, len(trajectory.turns)):
        messages = render_messages(instruction, trajectory.agent, trajectory.turns[:i])
        expected = _expected_next(messages, seed=seed, q=q, oracle=oracle, mistake=mistake)
        message = trajectory.turns[i].message
        if expected is None:
            _assert_final_message(message)
        else:
            assert len(message.tool_calls) == 1, f"turn {i}"
            call = message.tool_calls[0]
            assert call.id == f"call_{i}_0"
            assert call.name == expected[0], f"turn {i}"
            assert canonical_json(call.args) == canonical_json(expected[1]), f"turn {i}"


def _base_messages(instruction: str = "Do the task.") -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": ""},
        {"role": "developer", "content": ""},
        {"role": "user", "content": instruction},
    ]


def _exchange(k: int, calls: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    """第 k 則 assistant 訊息（形狀見 agents/policy.py）與其 tool 訊息。"""
    tool_calls = [{"id": f"call_{k}_{j}", "name": name, "args": args} for j, (name, args) in enumerate(calls)]
    out: list[dict[str, Any]] = [
        {"role": "assistant", "content": "", "tool_calls": tool_calls, "provider_content": None}
    ]
    out += [{"role": "tool", "tool_call_id": c["id"], "content": '{"data":{},"ok":true}'} for c in tool_calls]
    return out


def _conversation(turn_calls: list[list[tuple[str, Any]]]) -> list[dict[str, Any]]:
    messages = _base_messages()
    for k, calls in enumerate(turn_calls):
        messages += _exchange(k, calls)
    return messages


class _PlayPolicy:
    """測試用：以指定的 agent_config 播放固定劇本（每個元素是一個 turn 的 calls），之後結束。"""

    def __init__(self, config: AgentConfig, script: list[list[tuple[str, Any]]]) -> None:
        self._config = config
        self._script = script

    @property
    def agent_config(self) -> AgentConfig:
        return copy.deepcopy(self._config)

    def next_message(self, messages: list[dict[str, Any]], tools: Any) -> PolicyStep:
        k = sum(1 for m in messages if m["role"] == "assistant")
        if k < len(self._script):
            calls = tuple(
                ToolCall(id=f"call_{k}_{j}", name=name, args=args) for j, (name, args) in enumerate(self._script[k])
            )
            message = ModelMessage(text="", tool_calls=calls, stop_reason="tool_use")
        else:
            message = ModelMessage(text=FINAL_TEXT, tool_calls=(), stop_reason="end_turn")
        return PolicyStep(message=message, input_tokens=0, output_tokens=0, cost_usd=0.0)


def _with_constraints(constraints: list[dict[str, Any]]) -> Task:
    data = get_smoke_task("smoke-assign-and-start").to_dict()
    data["success_constraints"] = constraints
    return Task.from_dict(data)


def _unchanged(issue_id: int, fields: list[str] | None = None) -> dict[str, Any]:
    pred: dict[str, Any] = {"type": "issue_unchanged", "issue_id": issue_id}
    if fields is not None:
        pred["fields"] = fields
    return pred


# ---------------------------------------------------------------- protected_issue / mistake_call


def test_pool_and_smoke_cover_expected_tasks() -> None:
    assert len(_POOL) == 20
    assert len(ALL_TASK_IDS) == 25 and len(set(ALL_TASK_IDS)) == 25


@pytest.mark.parametrize("task_id", ALL_TASK_IDS)
def test_protected_issue_is_smallest_comment_covering_unchanged_issue(task_id: str) -> None:
    task = _task(task_id)
    result = protected_issue(task)
    assert type(result) is int
    assert result == _expected_protected(task)


@pytest.mark.parametrize("task_id", SMOKE_TASK_IDS)
def test_protected_issue_smoke_golden_values(task_id: str) -> None:
    assert protected_issue(get_smoke_task(task_id)) == SMOKE_PROTECTED[task_id]


def test_protected_issue_selection_ignores_constraint_order_and_non_comment_fields() -> None:
    constraints = [
        _unchanged(9),
        _unchanged(2, ["status", "labels"]),
        {"type": "issue_has_labels", "issue_id": 1, "labels": ["bug"]},
        _unchanged(7, ["status", "comments"]),
    ]
    assert protected_issue(_with_constraints(constraints)) == 7
    assert protected_issue(_with_constraints(list(reversed(constraints)))) == 7
    assert protected_issue(_with_constraints([_unchanged(9)])) == 9
    assert protected_issue(_with_constraints([_unchanged(10, ["comments"]), _unchanged(3)])) == 3


@pytest.mark.parametrize(
    "constraints",
    [
        [],
        [_unchanged(8, ["assignee_id", "status"])],
        [
            {"type": "issue_has_labels", "issue_id": 1, "labels": ["bug"]},
            _unchanged(3, ["title", "body", "status", "priority", "assignee_id", "labels"]),
        ],
    ],
    ids=["empty", "fields-without-comments", "other-predicates"],
)
def test_protected_issue_raises_when_no_constraint_covers_comments(constraints: list[dict[str, Any]]) -> None:
    task = _with_constraints(constraints)
    with pytest.raises(ValueError):
        protected_issue(task)
    with pytest.raises(ValueError):
        mistake_call(task)


@pytest.mark.parametrize("task_id", ALL_TASK_IDS)
def test_mistake_call_shape(task_id: str) -> None:
    task = _task(task_id)
    call = mistake_call(task)
    assert type(call) is tuple
    assert call == ("add_comment", {"issue_id": _expected_protected(task), "body": MISTAKE_BODY})


def test_mistake_body_constant() -> None:
    assert MISTAKE_BODY == "SYNTHETIC-MISTAKE"


def test_mistake_call_returns_new_objects() -> None:
    task = get_smoke_task("smoke-close-duplicate")
    first = mistake_call(task)
    first[1]["body"] = "tampered"
    first[1]["issue_id"] = 99
    assert mistake_call(task) == ("add_comment", {"issue_id": 2, "body": MISTAKE_BODY})


@pytest.mark.parametrize("task_id", ALL_TASK_IDS)
def test_mistake_makes_success_impossible(task_id: str) -> None:
    """錯誤（不論插在 oracle 的哪個位置）實際新增了 comment，且受保護 issue 的 constraint 失敗 → success 為 False。"""
    task = _task(task_id)
    oracle = _ORACLES[task_id]
    mistake = mistake_call(task)
    protected = _expected_protected(task)
    n = len(oracle)
    for pos in sorted({0, n // 2, n}):
        calls = (*oracle[:pos], mistake, *oracle[pos:])
        trajectory = run_episode(task, oracle_policy(list(calls)), max_turns=len(calls) + 1, clock=_clock)
        step = trajectory.turns[pos].tool_steps[0]
        assert step.tool == "add_comment" and step.ok and step.changes, f"mistake at {pos} did not mutate"
        report = trajectory.grade()
        assert report.success is False, f"mistake at {pos}"
        assert any(
            not c.passed and c.predicate["type"] == "issue_unchanged" and c.predicate["issue_id"] == protected
            for c in report.constraint_checks
        ), f"mistake at {pos}"


# ---------------------------------------------------------------- synthetic_agent_config


def test_agent_config_exact_fields() -> None:
    task = get_smoke_task("smoke-close-duplicate")
    oracle = REFERENCE_SOLUTIONS["smoke-close-duplicate"]
    mistake = mistake_call(task)
    config = synthetic_agent_config(oracle, mistake, 0.25)
    assert isinstance(config, AgentConfig)
    assert SYNTHETIC_POLICY == "synthetic"
    assert SYNTHETIC_PROMPT_VERSION == "synthetic/v1"
    assert config.policy == SYNTHETIC_POLICY
    assert config.prompt_version == SYNTHETIC_PROMPT_VERSION
    assert config.system_prompt == ""
    assert config.developer_prompt == ""
    expected = {
        "oracle": [{"name": name, "args": args} for name, args in oracle],
        "mistake": {"name": "add_comment", "args": {"issue_id": 2, "body": MISTAKE_BODY}},
        "mistake_rate": 0.25,
        "final_text": FINAL_TEXT,
    }
    assert canonical_json(config.model_config) == canonical_json(expected)


@pytest.mark.parametrize("q", [0, 0.0, 1e-9, 0.5, 0.999999], ids=repr)
def test_agent_config_accepts_rates_in_unit_interval(q: float) -> None:
    """decision 013 §4：model_config 以 ``float(q)`` 保存，因此 int 與 float 的同值 q 是同一個 agent（同一個 config_id）。"""
    task = get_smoke_task("smoke-assign-and-start")
    oracle = REFERENCE_SOLUTIONS[task.task_id]
    mistake = mistake_call(task)
    config = synthetic_agent_config(oracle, mistake, q)
    assert config.model_config["mistake_rate"] == q
    assert type(config.model_config["mistake_rate"]) is float
    assert config.config_id == synthetic_agent_config(oracle, mistake, float(q)).config_id
    policy_config = SyntheticPolicy(oracle, mistake=mistake, mistake_rate=q, rollout_seed=0).agent_config
    assert type(policy_config.model_config["mistake_rate"]) is float
    assert policy_config.config_id == config.config_id


@pytest.mark.parametrize(
    "q",
    [-0.1, -1, 1, 1.0, 1.5, True, False, float("nan"), float("inf"), float("-inf"), "0.1", None],
    ids=repr,
)
def test_invalid_mistake_rate_raises_value_error(q: Any) -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = REFERENCE_SOLUTIONS[task.task_id]
    mistake = mistake_call(task)
    with pytest.raises(ValueError):
        synthetic_agent_config(oracle, mistake, q)
    with pytest.raises(ValueError):
        policy = SyntheticPolicy(oracle, mistake=mistake, mistake_rate=q, rollout_seed=0)
        _ = policy.agent_config  # 建構或讀取 config 時必須拒絕（何時拒絕未規定）


@pytest.mark.parametrize("empty", [[], ()], ids=["list", "tuple"])
def test_empty_oracle_raises_value_error(empty: Any) -> None:
    task = get_smoke_task("smoke-assign-and-start")
    mistake = mistake_call(task)
    with pytest.raises(ValueError):
        synthetic_agent_config(empty, mistake, 0.1)
    with pytest.raises(ValueError):
        policy = SyntheticPolicy(empty, mistake=mistake, mistake_rate=0.1, rollout_seed=0)
        _ = policy.agent_config


def test_config_id_depends_on_oracle_mistake_and_rate() -> None:
    task = get_smoke_task("smoke-reassign-departed")
    oracle = REFERENCE_SOLUTIONS[task.task_id]
    mistake = mistake_call(task)
    other_mistake = ("add_comment", {"issue_id": 3, "body": MISTAKE_BODY})
    ids = {
        synthetic_agent_config(oracle, mistake, 0.2).config_id,
        synthetic_agent_config(oracle, mistake, 0.3).config_id,
        synthetic_agent_config(oracle[:-1], mistake, 0.2).config_id,
        synthetic_agent_config(oracle, other_mistake, 0.2).config_id,
    }
    assert len(ids) == 4


# docstring：oracle_calls 必須是 list / tuple；每個 call 與 mistake 都是長度 2 的 list / tuple (name, args)，且滿足
# check_call_preconditions（name 為 str；args 為 JSON 值：不含 tuple、NaN / inf）；否則 TypeError / ValueError。
_NAN = float("nan")
_BAD_CALLS: list[tuple[str, Callable[[], Any]]] = [
    ("generator", lambda: (x for x in ("add_comment", {"issue_id": 1, "body": "x"}))),
    ("dict", lambda: {"name": "add_comment", "args": {"issue_id": 1, "body": "x"}}),
    ("str", lambda: "add_comment"),
    ("str-of-length-2", lambda: "ab"),
    ("none", lambda: None),
    ("1-tuple", lambda: ("add_comment",)),
    ("3-tuple", lambda: ("add_comment", {"issue_id": 1, "body": "x"}, "extra")),
    ("int-name", lambda: (1, {})),
    ("tuple-args", lambda: ("x", (1,))),
    ("nan-args", lambda: ("x", {"a": _NAN})),
    ("inf-args", lambda: ("x", {"a": [float("inf")]})),
    ("non-str-key-args", lambda: ("x", {1: "a"})),
]
_BAD_ORACLES: list[tuple[str, Callable[[tuple[tuple[str, Any], ...]], Any]]] = [
    ("generator", lambda oracle: (call for call in oracle)),
    ("dict", lambda oracle: {name: args for name, args in oracle}),
    ("str", lambda oracle: "add_comment"),
    ("none", lambda oracle: None),
    *[(f"only-{cid}", (lambda make: lambda oracle: [make()])(make)) for cid, make in _BAD_CALLS],
    *[(f"after-valid-{cid}", (lambda make: lambda oracle: [oracle[0], make()])(make)) for cid, make in _BAD_CALLS],
]


@pytest.mark.parametrize("make_oracle", [m for _, m in _BAD_ORACLES], ids=[i for i, _ in _BAD_ORACLES])
def test_malformed_oracle_calls_are_rejected(make_oracle: Callable[[tuple[tuple[str, Any], ...]], Any]) -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    with pytest.raises((TypeError, ValueError)):
        synthetic_agent_config(make_oracle(oracle), mistake, 0.1)
    with pytest.raises((TypeError, ValueError)):
        policy = SyntheticPolicy(make_oracle(oracle), mistake=mistake, mistake_rate=0.1, rollout_seed=0)
        _ = policy.agent_config  # 建構或讀取 config 時必須拒絕（何時拒絕未規定）


@pytest.mark.parametrize("make_mistake", [m for _, m in _BAD_CALLS], ids=[i for i, _ in _BAD_CALLS])
def test_malformed_mistake_is_rejected(make_mistake: Callable[[], Any]) -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    with pytest.raises((TypeError, ValueError)):
        synthetic_agent_config(oracle, make_mistake(), 0.1)
    with pytest.raises((TypeError, ValueError)):
        policy = SyntheticPolicy(oracle, mistake=make_mistake(), mistake_rate=0.1, rollout_seed=0)
        _ = policy.agent_config


def test_list_shaped_calls_are_accepted_like_tuples() -> None:
    """docstring：call 與 mistake 可以是 list 或 tuple；兩者得到相同的 config_id。"""
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    as_tuples = synthetic_agent_config(tuple(oracle), mistake, 0.2)
    as_lists = synthetic_agent_config([list(call) for call in oracle], list(mistake), 0.2)
    assert as_lists.config_id == as_tuples.config_id
    policy = SyntheticPolicy([list(call) for call in oracle], mistake=list(mistake), mistake_rate=0.2, rollout_seed=0)
    assert policy.agent_config.config_id == as_tuples.config_id


def test_mutating_inputs_after_construction_changes_neither_config_nor_behaviour() -> None:
    """建構後修改呼叫端的 oracle list / call / args 與 mistake，或修改回傳 step 的 args：config_id 與 next_message 都不變。"""
    task = get_smoke_task("smoke-reassign-departed")
    q = 0.5
    oracle_in = [[name, copy.deepcopy(args)] for name, args in _ORACLES[task.task_id]]
    mistake_in = ["add_comment", {"issue_id": protected_issue(task), "body": MISTAKE_BODY}]
    config = synthetic_agent_config(oracle_in, mistake_in, q)
    policies = [
        SyntheticPolicy(oracle_in, mistake=mistake_in, mistake_rate=q, rollout_seed=seed) for seed in range(8)
    ]
    tools = IssueTrackerEnv.tool_specs()
    histories = [[], [[_ORACLES[task.task_id][0]]], [[c] for c in _ORACLES[task.task_id][:6]]]

    def observe() -> list[PolicyStep]:
        return [p.next_message(_conversation(h), tools) for p in policies for h in histories]

    config_before = canonical_json(config.to_dict())
    steps_before = observe()
    kinds = {
        "mistake" if s.message.tool_calls and s.message.tool_calls[0].args.get("body") == MISTAKE_BODY else "other"
        for s in steps_before
    }
    assert kinds == {"mistake", "other"}  # 兩種輸出都被觀察到，修改 oracle 與 mistake 都會被偵測到

    oracle_in[0][1]["assignee_id"] = "u99"
    oracle_in[1][0] = "close_issue"
    oracle_in[2][1].clear()
    oracle_in.append(["add_comment", {"issue_id": 3, "body": "extra"}])
    del oracle_in[3]
    mistake_in[0] = "remove_label"
    mistake_in[1]["body"] = "tampered"
    mistake_in[1]["issue_id"] = 99
    for step in steps_before:  # 回傳 step 的 args 被呼叫端改掉，也不得影響之後的輸出
        for call in step.message.tool_calls:
            if isinstance(call.args, dict):
                call.args.clear()

    assert canonical_json(config.to_dict()) == config_before
    for policy in policies:
        assert policy.agent_config.config_id == config.config_id
    expected = SyntheticPolicy(
        _ORACLES[task.task_id], mistake=mistake_call(task), mistake_rate=q, rollout_seed=0
    ).agent_config
    assert config.config_id == expected.config_id
    steps_after = observe()
    fresh = [
        SyntheticPolicy(_ORACLES[task.task_id], mistake=mistake_call(task), mistake_rate=q, rollout_seed=seed)
        .next_message(_conversation(h), tools)
        for seed in range(8)
        for h in histories
    ]
    assert steps_after == fresh


# ---------------------------------------------------------------- SyntheticPolicy：config 與輸入驗證


def test_policy_agent_config_matches_and_excludes_seed() -> None:
    task = get_smoke_task("smoke-escalate-security")
    oracle = REFERENCE_SOLUTIONS[task.task_id]
    mistake = mistake_call(task)
    expected = synthetic_agent_config(oracle, mistake, 0.3)
    for seed in (0, 1, 7, 123456789):
        config = SyntheticPolicy(oracle, mistake=mistake, mistake_rate=0.3, rollout_seed=seed).agent_config
        assert isinstance(config, AgentConfig)
        assert config.config_id == expected.config_id
        assert canonical_json(config.to_dict()) == canonical_json(expected.to_dict())


def test_policy_agent_config_is_a_new_object_each_time() -> None:
    task = get_smoke_task("smoke-escalate-security")
    policy = _policy(task, 0.3, 0)
    before = policy.agent_config.config_id
    leaked = policy.agent_config
    leaked.model_config["oracle"].clear()
    leaked.model_config["mistake_rate"] = 0.9
    assert policy.agent_config.config_id == before


@pytest.mark.parametrize("seed", [True, False, "1", 1.0, None], ids=repr)
def test_rollout_seed_must_be_int(seed: Any) -> None:
    task = get_smoke_task("smoke-assign-and-start")
    with pytest.raises((TypeError, ValueError)):
        SyntheticPolicy(
            REFERENCE_SOLUTIONS[task.task_id], mistake=mistake_call(task), mistake_rate=0.1, rollout_seed=seed
        )


@pytest.mark.parametrize(
    "messages",
    [
        tuple(_base_messages()),
        "not a list",
        [*_base_messages(), "x"],
        [*_base_messages(), {"content": "no role"}],
        [*_base_messages(), {"role": 1, "content": "bad role"}],
    ],
    ids=["tuple", "str", "non-dict", "missing-role", "non-str-role"],
)
def test_next_message_validates_messages_like_scripted_policy(messages: Any) -> None:
    policy = _policy(get_smoke_task("smoke-assign-and-start"), 0.3, 0)
    with pytest.raises(TypeError):
        policy.next_message(messages, IssueTrackerEnv.tool_specs())


def test_next_message_does_not_modify_inputs() -> None:
    task = get_smoke_task("smoke-reassign-departed")
    oracle = REFERENCE_SOLUTIONS[task.task_id]
    messages = _conversation([[oracle[0]], [mistake_call(task)]])
    tools = IssueTrackerEnv.tool_specs()
    snapshot = (copy.deepcopy(messages), copy.deepcopy(tools))
    for seed in range(4):
        _policy(task, 0.5, seed).next_message(messages, tools)
    assert canonical_json(messages) == canonical_json(snapshot[0])
    assert canonical_json(list(tools)) == canonical_json(list(snapshot[1]))


# ---------------------------------------------------------------- SyntheticPolicy：決策規則


def test_next_message_follows_documented_decision_rule() -> None:
    """u < q → 錯誤；否則 oracle[r]；r >= n → 結束訊息。以 docstring 公式逐一重算。"""
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    n = len(oracle)
    q = 0.5
    histories = [
        [],
        [[oracle[0]]],
        [[oracle[0]], [mistake]],
        [[mistake], [oracle[0]], [oracle[1]]],
        [[c] for c in oracle[:5]],
        [[c] for c in oracle],
        [[c] for c in oracle] + [[mistake]],
    ]
    tools = IssueTrackerEnv.tool_specs()
    kinds: set[str] = set()
    for history in histories:
        messages = _conversation(history)
        k = len(history)
        for seed in range(32):
            step = _policy(task, q, seed).next_message(messages, tools)
            assert isinstance(step, PolicyStep)
            expected = _expected_next(messages, seed=seed, q=q, oracle=oracle, mistake=mistake)
            if expected is None:
                _assert_final_message(step.message)
                kinds.add("final")
            else:
                assert len(step.message.tool_calls) == 1
                call = step.message.tool_calls[0]
                assert call.id == f"call_{k}_0"
                assert (call.name, canonical_json(call.args)) == (expected[0], canonical_json(expected[1]))
                kinds.add("mistake" if expected == mistake else "oracle")
    assert kinds == {"mistake", "oracle", "final"}
    assert n == 7


def test_oracle_progress_counts_only_non_mistake_assistant_messages() -> None:
    """q = 0：r = 「剛好一個 call 且 (name, canonical args) 等於錯誤 call」以外的 assistant 訊息數。"""
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    protected = mistake[1]["issue_id"]
    reordered_mistake = ("add_comment", {"body": MISTAKE_BODY, "issue_id": protected})  # key 順序不同仍是錯誤
    tools = IssueTrackerEnv.tool_specs()
    policy = _policy(task, 0.0, 0)

    def next_call(history: list[list[tuple[str, Any]]]) -> tuple[str, str]:
        step = policy.next_message(_conversation(history), tools)
        assert len(step.message.tool_calls) == 1
        call = step.message.tool_calls[0]
        assert call.id == f"call_{len(history)}_0"
        return call.name, canonical_json(call.args)

    def as_key(call: tuple[str, Any]) -> tuple[str, str]:
        return call[0], canonical_json(call[1])

    assert next_call([]) == as_key(oracle[0])
    assert next_call([[oracle[0]], [mistake], [oracle[1]]]) == as_key(oracle[2])
    assert next_call([[oracle[0]], [reordered_mistake]]) == as_key(oracle[1])
    assert next_call([[mistake], [mistake]]) == as_key(oracle[0])
    # 以下都不是「錯誤」：計為已完成的 oracle call。
    assert next_call([[("add_comment", {"issue_id": protected, "body": "not the mistake"})]]) == as_key(oracle[1])
    assert next_call([[("add_comment", {"issue_id": protected + 1, "body": MISTAKE_BODY})]]) == as_key(oracle[1])
    # 同一 turn 含錯誤與其他 call 的歷史（例如 [[mistake, oracle[0]]]）：docstring 的「剛好一個」規則與 decision 010 的
    # 真值語義衝突（錯誤已實際寫入），屬契約層爭議，刻意不斷言。


def test_final_message_when_oracle_is_exhausted() -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    policy = _policy(task, 0.0, 3)
    tools = IssueTrackerEnv.tool_specs()
    for history in ([[c] for c in oracle], [[oracle[0]], [mistake], [oracle[1]], [mistake]]):
        step = policy.next_message(_conversation(history), tools)
        _assert_final_message(step.message)


def test_steps_report_zero_usage() -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    tools = IssueTrackerEnv.tool_specs()
    for q, history in ((0.0, []), (0.0, [[c] for c in oracle]), (0.999999, [])):
        step = _policy(task, q, 0).next_message(_conversation(history), tools)
        assert step.input_tokens == 0
        assert step.output_tokens == 0
        assert step.cost_usd == 0.0


def test_same_seed_and_messages_give_identical_steps() -> None:
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    tools = IssueTrackerEnv.tool_specs()
    for history in ([], [[oracle[0]], [mistake]], [[c] for c in oracle]):
        messages = _conversation(history)
        for seed in range(8):
            policy = _policy(task, 0.5, seed)
            first = policy.next_message(copy.deepcopy(messages), tools)
            again = policy.next_message(copy.deepcopy(messages), tools)
            fresh = _policy(task, 0.5, seed).next_message(copy.deepcopy(messages), tools)
            assert first == again == fresh


def test_same_seed_episodes_are_byte_identical() -> None:
    task = get_smoke_task("smoke-reassign-departed")
    for seed in range(5):
        a = run_episode(task, _policy(task, 0.4, seed), max_turns=20, clock=_clock)
        b = run_episode(task, _policy(task, 0.4, seed), max_turns=20, clock=_clock)
        assert canonical_json(a.to_dict()) == canonical_json(b.to_dict())


def test_different_seeds_diverge() -> None:
    task = get_smoke_task("smoke-assign-and-start")
    sequences = set()
    for seed in range(20):
        trajectory = run_episode(task, _policy(task, 0.5, seed), max_turns=12, clock=_clock)
        sequences.add(canonical_json([[[c.name, c.args] for c in t.message.tool_calls] for t in trajectory.turns]))
    assert len(sequences) > 1


@pytest.mark.parametrize(
    "task_id", ["smoke-reassign-departed", "smoke-close-duplicate", *[i.task.task_id for i in week1_pool("dev")]]
)
def test_episodes_follow_decision_rule_and_success_iff_no_mistake(task_id: str) -> None:
    task = _task(task_id)
    oracle = _ORACLES[task_id]
    mistake = mistake_call(task)
    n = len(oracle)
    q = 0.4
    for seed in range(6):
        trajectory = run_episode(task, _policy(task, q, seed), max_turns=2 * n + 8, clock=_clock)
        _check_decisions(trajectory, start=0, seed=seed, q=q, oracle=oracle, mistake=mistake)
        for i, turn in enumerate(trajectory.turns):
            assert [c.id for c in turn.message.tool_calls] in ([], [f"call_{i}_0"])
        if trajectory.termination == "final_answer":
            _assert_final_message(trajectory.turns[-1].message)
            assert trajectory.grade().success is (not _has_mistake(trajectory, mistake))
        else:
            assert _has_mistake(trajectory, mistake)
            assert trajectory.grade().success is False


@pytest.mark.parametrize("task_id", ALL_TASK_IDS)
def test_zero_rate_reproduces_oracle_and_succeeds(task_id: str) -> None:
    task = _task(task_id)
    oracle = _ORACLES[task_id]
    n = len(oracle)
    for seed in (0, 1):
        trajectory = run_episode(task, _policy(task, 0.0, seed), max_turns=n + 1, clock=_clock)
        assert trajectory.termination == "final_answer"
        assert len(trajectory.turns) == n + 1
        played = [_turn_calls(t) for t in trajectory.turns[:n]]
        assert canonical_json(played) == canonical_json([[call] for call in oracle])
        for i, turn in enumerate(trajectory.turns[:n]):
            assert turn.message.tool_calls[0].id == f"call_{i}_0"
        _assert_final_message(trajectory.turns[-1].message)
        assert trajectory.grade().success is True


# ---------------------------------------------------------------- 無狀態：restore 後接續


def _source_with_mistake(task: Task, q: float, max_turns: int) -> tuple[int, Trajectory]:
    mistake = mistake_call(task)
    for seed in range(200):
        trajectory = run_episode(task, _policy(task, q, seed), max_turns=max_turns, clock=_clock)
        if trajectory.termination == "final_answer" and _has_mistake(trajectory, mistake):
            return seed, trajectory
    raise AssertionError("no seed in range(200) produced a source with a mistake")


def test_rerun_from_every_checkpoint_with_same_seed_reproduces_source() -> None:
    task = get_smoke_task("smoke-reassign-departed")
    seed, source = _source_with_mistake(task, 0.35, 20)
    expected_turns = [canonical_json(t.to_dict()) for t in source.turns]
    for k in range(len(source.turns)):
        child = rerun(make_checkpoint(source, k), _policy(task, 0.35, seed), clock=_clock)
        assert [canonical_json(t.to_dict()) for t in child.turns] == expected_turns, f"k={k}"
        assert child.termination == source.termination
        assert canonical_json(child.final_snapshot) == canonical_json(source.final_snapshot)


def test_rerun_with_other_seed_continues_from_prefix_state() -> None:
    """不同 seed 的 policy 與 source 共用 config_id（rerun 接受），且續跑的每個 turn 都依前綴算出的 r 決策。"""
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    q = 0.35
    _, source = _source_with_mistake(task, q, 20)
    for k in range(len(source.turns)):
        checkpoint = make_checkpoint(source, k)
        for seed in (1000 + k, 2000 + k):
            child = rerun(checkpoint, _policy(task, q, seed), clock=_clock)
            assert [canonical_json(t.to_dict()) for t in child.turns[:k]] == [
                canonical_json(t.to_dict()) for t in source.turns[:k]
            ]
            _check_decisions(child, start=k, seed=seed, q=q, oracle=oracle, mistake=mistake)
            if child.termination == "final_answer":
                assert child.grade().success is (not _has_mistake(child, mistake))


# ---------------------------------------------------------------- success_probability / checkpoint_truth


@pytest.mark.parametrize(
    ("n", "r", "has_mistake", "q", "expected"),
    [
        (3, 0, False, 0.2, 0.8**4),
        (3, 1, False, 0.2, 0.8**3),
        (3, 3, False, 0.2, 0.8),
        (7, 2, False, 0.3, 0.7**6),
        (1, 0, False, 0.5, 0.25),
        (1, 0, False, 0.0, 1.0),
        (4, 4, False, 0.0, 1.0),
        (3, 1, True, 0.2, 0.0),
        (3, 0, True, 0.0, 0.0),
        (5, 5, True, 0.9, 0.0),
    ],
)
def test_success_probability_formula(n: int, r: int, has_mistake: bool, q: float, expected: float) -> None:
    result = success_probability(n, r, has_mistake, q)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    ("n", "r", "has_mistake", "q"),
    [
        (0, 0, False, 0.2),  # n_oracle >= 1
        (-1, 0, False, 0.2),
        (3, -1, False, 0.2),  # 0 <= oracle_done
        (3, 4, False, 0.2),  # oracle_done <= n_oracle
        (3, 4, True, 0.2),  # 範圍檢查不因前綴已有錯誤而豁免（checkpoint_truth 在 k = len(turns) 時走到這裡）
        (True, 0, False, 0.2),  # int，bool 不算
        (3, True, False, 0.2),
        (3, False, False, 0.2),
        (3.0, 0, False, 0.2),
        (3, 1.0, False, 0.2),
        (3, 1, 0, 0.2),  # mistake_in_prefix 為 bool
        (3, 1, 1, 0.2),
        (3, 1, None, 0.2),
        (3, 1, "False", 0.2),
        (3, 1, False, 1),  # 0 <= q < 1
        (3, 1, False, 1.0),
        (3, 1, False, 1.5),
        (3, 1, False, -0.1),
        (3, 1, False, True),
        (3, 1, False, float("nan")),
        (3, 1, False, "0.5"),
        (3, 1, False, None),
        (3, 1, True, 1.0),  # q 的檢查不因前綴已有錯誤而豁免
    ],
    ids=repr,
)
def test_success_probability_rejects_invalid_inputs(n: Any, r: Any, has_mistake: Any, q: Any) -> None:
    with pytest.raises(ValueError):
        success_probability(n, r, has_mistake, q)


def _play(task: Task, config: AgentConfig, script: list[list[tuple[str, Any]]]) -> Trajectory:
    return run_episode(task, _PlayPolicy(config, script), max_turns=len(script) + 1, clock=_clock)


def _protected_comments(snapshot: dict[str, Any], protected: int) -> list[dict[str, Any]]:
    return [c for c in snapshot["state"]["comments"] if c["issue_id"] == protected]


def _assert_truth_tracks_protected_issue(
    trajectory: Trajectory, task: Task, q: float, *, on_policy: bool
) -> tuple[int, int]:
    """每個合法 checkpoint k ∈ [0, len(turns) − 1]（decision 010 §3）：

    - checkpoint_truth > 0 ⇔ checkpoint 的 env 中受保護 issue 沒有 MISTAKE_BODY comment（錯誤不可逆 → 真正的機率為 0）；
    - on_policy（synthetic policy 產生的 trajectory）時另要求：> 0 ⇒ 受保護 issue 的 comments 與初始狀態逐 byte 相同。
    回傳 (truth > 0 的 checkpoint 數, truth == 0 的 checkpoint 數)。
    """
    mistake = mistake_call(task)
    protected = _expected_protected(task)
    initial = [c for c in task.initial_state["comments"] if c["issue_id"] == protected]
    positive = zero = 0
    for k in range(len(trajectory.turns)):
        truth = checkpoint_truth(trajectory, k, mistake, q)
        comments = _protected_comments(make_checkpoint(trajectory, k).env_snapshot, protected)
        posted = any(c["body"] == MISTAKE_BODY for c in comments)
        assert (truth > 0) == (not posted), f"{task.task_id} k={k} truth={truth} mistake_posted={posted}"
        if truth > 0:
            positive += 1
            if on_policy:
                assert canonical_json(comments) == canonical_json(initial), f"{task.task_id} k={k}"
        else:
            zero += 1
    return positive, zero


@pytest.mark.parametrize("task_id", ALL_TASK_IDS)
def test_checkpoint_truth_positive_iff_protected_issue_has_no_mistake_comment(task_id: str) -> None:
    """synthetic policy 產生的 source（含 max_turns 截斷者）：真值 > 0 恰好在錯誤尚未寫入受保護 issue 的 checkpoint。"""
    task = _task(task_id)
    n = len(_ORACLES[task_id])
    q = 0.35
    positive = zero = 0
    for seed in range(20):  # 至少 2 個 seed；之後直到兩種 checkpoint 都出現為止（固定 seed 序列，決定性）
        if seed >= 2 and positive > 0 and zero > 0:
            break
        for max_turns in (2 * n + 8, max(1, n // 2 + 1)):
            trajectory = run_episode(task, _policy(task, q, seed), max_turns=max_turns, clock=_clock)
            p, z = _assert_truth_tracks_protected_issue(trajectory, task, q, on_policy=True)
            positive += p
            zero += z
    assert positive > 0 and zero > 0, f"{task_id}: positive={positive} zero={zero}"


@pytest.mark.parametrize(
    "task_id", ["smoke-reassign-departed", "smoke-close-duplicate", *[i.task.task_id for i in week1_pool("dev")]]
)
def test_mistake_free_continuation_succeeds_iff_truth_positive(task_id: str) -> None:
    """decision 010 §1：成功的續跑恰好是剩餘的 oracle[r:] 加上結束 turn（共 n − r + 1 個 turn）。

    以同 config_id 的劇本 policy 從每個合法 checkpoint 播放「沒有錯誤的續跑」：真值 > 0 ⇔ 該續跑成功，且成功時恰好
    n − r + 1 個新 turn、以 final_answer 結束（r 依 docstring 規則由前綴計算）。
    """
    task = _task(task_id)
    oracle = _ORACLES[task_id]
    mistake = mistake_call(task)
    n = len(oracle)
    q = 0.35
    positive = zero = 0
    for seed in range(3):
        source = run_episode(task, _policy(task, q, seed), max_turns=2 * n + 8, clock=_clock)
        for k in range(len(source.turns)):
            truth = checkpoint_truth(source, k, mistake, q)
            r = sum(1 for t in source.turns[:k] if not _is_mistake(_turn_calls(t), mistake))
            script: list[list[tuple[str, Any]]] = [[] for _ in range(k)] + [[c] for c in oracle[r:]]
            child = rerun(make_checkpoint(source, k), _PlayPolicy(source.agent, script), clock=_clock)
            success = child.grade().success
            if truth > 0:
                positive += 1
                assert success is True, f"seed={seed} k={k}"
                assert child.termination == "final_answer"
                assert len(child.turns) - k == n - r + 1, f"seed={seed} k={k}"
            else:
                zero += 1
                assert success is False, f"seed={seed} k={k}"
    assert positive > 0 and zero > 0, f"{task_id}: positive={positive} zero={zero}"


def test_checkpoint_truth_without_mistake() -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    q = 0.3
    trajectory = _play(task, synthetic_agent_config(oracle, mistake, q), [[c] for c in oracle])
    assert len(trajectory.turns) == 3
    for k, expected in enumerate([0.7**3, 0.7**2, 0.7]):
        assert checkpoint_truth(trajectory, k, mistake, q) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("position", "expected"),
    [(0, [0.7**3, 0.0, 0.0, 0.0]), (1, [0.7**3, 0.7**2, 0.0, 0.0]), (2, [0.7**3, 0.7**2, 0.7, 0.0])],
)
def test_checkpoint_truth_with_mistake_in_prefix(position: int, expected: list[float]) -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    q = 0.3
    script = [[c] for c in oracle]
    script.insert(position, [mistake])
    trajectory = _play(task, synthetic_agent_config(oracle, mistake, q), script)
    assert len(trajectory.turns) == 4
    for k, value in enumerate(expected):
        result = checkpoint_truth(trajectory, k, mistake, q)
        assert result == pytest.approx(value, rel=1e-12, abs=1e-15), f"k={k}"
        assert result == pytest.approx(
            _expected_truth(trajectory.turns, k, n=2, mistake=mistake, q=q), rel=1e-12, abs=1e-15
        )
    assert _assert_truth_tracks_protected_issue(trajectory, task, q, on_policy=True) == (position + 1, 3 - position)


def test_checkpoint_truth_takes_n_from_agent_config_oracle() -> None:
    """n_oracle 取自 model_config["oracle"] 的長度，而不是 trajectory 實際播放的 call 數。"""
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    q = 0.3
    trajectory = _play(task, synthetic_agent_config(oracle, mistake, q), [[oracle[0]], [oracle[1]]])
    for k in range(3):
        assert checkpoint_truth(trajectory, k, mistake, q) == pytest.approx(0.7 ** (7 - k + 1), rel=1e-12)


def test_checkpoint_truth_rejects_off_policy_prefixes() -> None:
    """decision 013 §4：前綴每個 turn 必須剛好一個 call，且是錯誤 call 或下一個 oracle call；否則沒有解析解 → ValueError。
    近似錯誤（不同 body、不同 issue）與同一 turn 含錯誤加 oracle 的歷史都是 off-policy。"""
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    protected = mistake[1]["issue_id"]
    q = 0.25
    config = synthetic_agent_config(oracle, mistake, q)
    for off_policy in (
        [("add_comment", {"issue_id": protected, "body": "not the mistake"})],
        [("add_comment", {"issue_id": protected + 1, "body": MISTAKE_BODY})],
        [mistake, oracle[1]],
        [oracle[2]],  # 跳過 oracle[1]
    ):
        trajectory = _play(task, config, [[oracle[0]], off_policy, [oracle[1]]])
        assert checkpoint_truth(trajectory, 1, mistake, q) == pytest.approx((1 - q) ** 7, rel=1e-12)
        for k in (2, 3):
            with pytest.raises(ValueError):
                checkpoint_truth(trajectory, k, mistake, q)


def test_checkpoint_truth_validates_arguments_against_the_trajectory() -> None:
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    q = 0.3
    trajectory = _play(task, synthetic_agent_config(oracle, mistake, q), [[c] for c in oracle])
    assert checkpoint_truth(trajectory, 0, mistake, 0.3) == pytest.approx(0.7**3)
    for bad_q in (0.2, 0.31):
        with pytest.raises(ValueError):
            checkpoint_truth(trajectory, 0, mistake, bad_q)
    other = ("add_comment", {"issue_id": mistake[1]["issue_id"], "body": "different"})
    with pytest.raises(ValueError):
        checkpoint_truth(trajectory, 0, other, q)
    for bad_k in (-1, len(trajectory.turns), len(trajectory.turns) + 1, True, 1.0):
        with pytest.raises(ValueError):
            checkpoint_truth(trajectory, bad_k, mistake, q)  # type: ignore[arg-type]
    truncated = run_episode(task, _policy(task, 0.0, 0), max_turns=2, clock=_clock)
    assert truncated.termination == "max_turns"
    with pytest.raises(ValueError):
        checkpoint_truth(truncated, len(truncated.turns), mistake, 0.0)
    scripted = run_episode(task, oracle_policy(oracle), max_turns=5, clock=_clock)
    with pytest.raises(ValueError):
        checkpoint_truth(scripted, 0, mistake, q)


def test_checkpoint_truth_uses_the_policy_mistake_rule() -> None:
    """key 順序不同的錯誤 call 仍是錯誤（canonical 比較）。"""
    task = get_smoke_task("smoke-reassign-departed")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    protected = mistake[1]["issue_id"]
    q = 0.25
    config = synthetic_agent_config(oracle, mistake, q)

    reordered = ("add_comment", {"body": MISTAKE_BODY, "issue_id": protected})
    trajectory = _play(task, config, [[oracle[0]], [reordered], [oracle[1]]])
    assert checkpoint_truth(trajectory, 1, mistake, q) == pytest.approx(0.75**7, rel=1e-12)
    assert checkpoint_truth(trajectory, 2, mistake, q) == 0.0
    assert checkpoint_truth(trajectory, 3, mistake, q) == 0.0
    assert _assert_truth_tracks_protected_issue(trajectory, task, q, on_policy=True) == (2, 2)


def test_checkpoint_truth_matches_direct_computation_on_synthetic_sources() -> None:
    q = 0.35
    checked_mistake = checked_clean = 0
    for task_id in ("smoke-reassign-departed", "smoke-escalate-security"):
        task = get_smoke_task(task_id)
        mistake = mistake_call(task)
        n = len(_ORACLES[task_id])
        for seed in range(8):
            trajectory = run_episode(task, _policy(task, q, seed), max_turns=20, clock=_clock)
            if _has_mistake(trajectory, mistake):
                checked_mistake += 1
            else:
                checked_clean += 1
            for k in range(len(trajectory.turns)):
                assert checkpoint_truth(trajectory, k, mistake, q) == pytest.approx(
                    _expected_truth(trajectory.turns, k, n=n, mistake=mistake, q=q), rel=1e-12, abs=1e-15
                ), f"{task_id} seed={seed} k={k}"
    assert checked_mistake > 0 and checked_clean > 0


def test_checkpoint_truth_defined_on_every_legal_checkpoint_of_truncated_sources() -> None:
    """decision 010 §3：合法 checkpoint k ∈ [0, len(turns) − 1]；max_turns 截斷（沒有結束訊息）的 source 也在每個合法 k 上
    有定義，且等於直接計算的解析解。"""
    task = get_smoke_task("smoke-reassign-departed")
    mistake = mistake_call(task)
    n = len(_ORACLES[task.task_id])
    q = 0.35
    checked_mistake = checked_clean = 0
    for max_turns in (3, 5):
        for seed in range(6):
            trajectory = run_episode(task, _policy(task, q, seed), max_turns=max_turns, clock=_clock)
            assert trajectory.termination == "max_turns"  # n + 1 = 8 > max_turns：必定截斷
            if _has_mistake(trajectory, mistake):
                checked_mistake += 1
            else:
                checked_clean += 1
            for k in range(len(trajectory.turns)):
                result = checkpoint_truth(trajectory, k, mistake, q)
                assert 0.0 <= result <= 1.0
                assert result == pytest.approx(
                    _expected_truth(trajectory.turns, k, n=n, mistake=mistake, q=q), rel=1e-12, abs=1e-15
                ), f"max_turns={max_turns} seed={seed} k={k}"
    assert checked_mistake > 0 and checked_clean > 0


def _final_answer_source(mistake_position: int | None) -> tuple[Trajectory, tuple[str, Any], float]:
    """smoke-assign-and-start（n = 2）的 final_answer trajectory；mistake_position 為 None 時沒有錯誤。"""
    task = get_smoke_task("smoke-assign-and-start")
    oracle = _ORACLES[task.task_id]
    mistake = mistake_call(task)
    q = 0.3
    script = [[c] for c in oracle]
    if mistake_position is not None:
        script.insert(mistake_position, [mistake])
    trajectory = _play(task, synthetic_agent_config(oracle, mistake, q), script)
    assert trajectory.termination == "final_answer"
    assert len(trajectory.turns) == len(script) + 1
    _assert_final_message(trajectory.turns[-1].message)
    return trajectory, mistake, q


@pytest.mark.parametrize("mistake_position", [None, 0, 2], ids=["clean", "mistake-first", "mistake-last"])
def test_checkpoint_truth_rejects_k_after_the_final_message(mistake_position: int | None) -> None:
    """k = len(turns) 的前綴包含結束訊息；依 policy 的判定規則它也計為一個已完成的 oracle call → oracle_done = n + 1 > n，
    success_probability 依契約 ValueError（不論前綴是否已有錯誤）。這也與 decision 010 §3 的合法範圍 [0, len(turns) − 1] 一致。
    最後一個合法 checkpoint k = len(turns) − 1 仍有定義。"""
    trajectory, mistake, q = _final_answer_source(mistake_position)
    last = len(trajectory.turns) - 1
    expected_last = 0.7 if mistake_position is None else 0.0
    assert checkpoint_truth(trajectory, last, mistake, q) == pytest.approx(expected_last, rel=1e-12, abs=1e-15)
    with pytest.raises(ValueError):
        checkpoint_truth(trajectory, len(trajectory.turns), mistake, q)


@pytest.mark.parametrize(
    "k_kind", ["minus-one", "very-negative", "len-plus-one", "len-plus-five", "float", "float-zero", "str", "none"]
)
@pytest.mark.parametrize("mistake_position", [None, 1], ids=["clean", "mistake"])
def test_checkpoint_truth_rejects_out_of_range_or_non_int_k(k_kind: str, mistake_position: int | None) -> None:
    """docstring：k 為 int 且 0 <= k <= len(turns)，否則 ValueError。（k = True 未規定是否算 int，不斷言。）"""
    trajectory, mistake, q = _final_answer_source(mistake_position)
    length = len(trajectory.turns)
    k = {
        "minus-one": -1,
        "very-negative": -length - 1,
        "len-plus-one": length + 1,
        "len-plus-five": length + 5,
        "float": 1.0,
        "float-zero": 0.0,
        "str": "1",
        "none": None,
    }[k_kind]
    with pytest.raises(ValueError):
        checkpoint_truth(trajectory, k, mistake, q)


# ---------------------------------------------------------------- 經驗頻率 vs 解析解


def test_empirical_success_rate_from_scratch_matches_formula() -> None:
    """smoke-assign-and-start（n = 2）、q = 0.3：P(success) = 0.7^3 = 0.343。

    600 個固定 seed，容差 0.07（約 3.6 SE）。
    """
    task = get_smoke_task("smoke-assign-and-start")
    mistake = mistake_call(task)
    n = len(_ORACLES[task.task_id])
    q = 0.3
    rollouts = 600
    successes = 0
    for seed in range(rollouts):
        # max_turns = n + 1：decision 010 說成功的續跑恰好 n − r + 1 個 turn，不受此上限影響。
        trajectory = run_episode(task, _policy(task, q, seed), max_turns=n + 1, clock=_clock)
        success = trajectory.grade().success
        assert success is (not _has_mistake(trajectory, mistake))
        successes += success
    truth = success_probability(n, 0, False, q)
    assert truth == pytest.approx(0.7**3)
    observed = successes / rollouts
    assert abs(observed - truth) <= 0.07
    # 指數差一的替代公式 (1 − q)^n = 0.49、(1 − q)^(n + 2) ≈ 0.240 都必須比真值更遠。
    assert abs(observed - truth) < min(abs(observed - (1 - q) ** n), abs(observed - (1 - q) ** (n + 2)))


def test_empirical_success_rate_from_checkpoint_matches_truth() -> None:
    """smoke-reassign-departed（n = 7）、q = 0.5、k = 5、前綴無錯誤（r = 5）：P = 0.5^(n − r + 1) = 0.125。

    指數差一的替代公式為 0.5^2 = 0.25（續跑晚一個 turn 開始）與 0.5^4 = 0.0625（早一個 turn）；400 個固定 seed
    （SE ≈ 0.017），容差 0.05，且觀察值必須比兩個替代值都更接近真值。
    """
    task = get_smoke_task("smoke-reassign-departed")
    mistake = mistake_call(task)
    n = len(_ORACLES[task.task_id])
    q = 0.5
    k = 5
    source = None
    for seed in range(400):
        candidate = run_episode(task, _policy(task, q, seed), max_turns=20, clock=_clock)
        if len(candidate.turns) > k and not any(_is_mistake(_turn_calls(t), mistake) for t in candidate.turns[:k]):
            source = candidate
            break
    assert source is not None, "no seed in range(400) produced a source with 5 mistake-free turns"
    assert n == 7
    truth = checkpoint_truth(source, k, mistake, q)
    assert truth == pytest.approx(0.5**3, rel=1e-12)
    checkpoint = make_checkpoint(source, k)
    rollouts = 400
    successes = sum(
        rerun(checkpoint, _policy(task, q, 10_000 + i), clock=_clock).grade().success for i in range(rollouts)
    )
    observed = successes / rollouts
    assert abs(observed - truth) <= 0.05
    late, early = (1 - q) ** (n - k), (1 - q) ** (n - k + 2)
    assert abs(observed - truth) < min(abs(observed - late), abs(observed - early))
