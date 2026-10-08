"""Week 1 #5 實作後對抗式審查確認的缺陷之回歸測試（decision 009「審查修正」）。

1. parse_response：模型回傳極深的 tool_use input → 必須是 AdapterError（agent_output），不得漏出 RecursionError。
2. build_request：provider_content block 的 type 為 list / dict、或歷史內容極深 → ValueError（不得漏出 TypeError / RecursionError）。
3. Trajectory.validate / Checkpoint.verify 的深度上限必須以文件根計算，與 from_dict / load 一致（通過 validate 的一定能 load）。
4. run_logged：例外的 request_id 含無法 UTF-8 編碼的字元、或例外類別的 __module__ 不是 str → 仍寫入一列並維持原本的回傳 / 拋出。
5. run_logged：manifest 所在目錄不存在 → 執行前 ValueError（不執行、不花費）。
6. run_logged：完成的 trajectory 必須與 task / key / max_turns / policy 相符，否則記為 harness 並拋出。
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import BASELINE_MODEL_CONFIG, AdapterError, AnthropicPolicy, build_request, parse_response
from agents.policy import oracle_policy
from apps.issue_tracker.tools import TOOL_SPECS
from replay.aborts import RolloutKey, classify_exception, read_manifest, run_logged
from replay.checkpoint import Checkpoint, CheckpointError, make_checkpoint
from replay.runner import run_episode
from replay.trajectory import MAX_DOCUMENT_DEPTH, Trajectory, TrajectoryError
from tasks.smoke import REFERENCE_SOLUTIONS, get_smoke_task
from tests.fake_anthropic import FakeClient, FakeResponse, FakeUsage, thinking_block, tool_use_block

TASK_ID = "smoke-assign-and-start"
MC = copy.deepcopy(BASELINE_MODEL_CONFIG)


def _nested(depth: int) -> Any:
    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


# ---------------------------------------------------------------- 1. parse_response


@pytest.mark.parametrize(
    "stop_reason, depth",
    [("tool_use", 70), ("tool_use", 600), ("tool_use", 3000), ("refusal", 600), ("refusal", 3000)],
)
def test_parse_response_deep_tool_input_is_adapter_error(stop_reason: str, depth: int) -> None:
    """tool_use：超過 ToolCall 的 MAX_ARGS_DEPTH 即 AdapterError；refusal 不建立 ToolCall，但超過 provider_content 的深度上限
    （MAX_MODEL_CONFIG_DEPTH）同樣是 AdapterError（深度 70 的 refusal 合法，見下一個測試）。"""
    response = FakeResponse(
        content=[thinking_block("sig"), tool_use_block("toolu_1", "get_issue", {"issue_id": _nested(depth)})],
        stop_reason=stop_reason,
        usage=FakeUsage(10, 5),
    )
    with pytest.raises(AdapterError) as info:
        parse_response(response, "claude-opus-5-5")
    assert classify_exception(info.value) == "agent_output"


def test_parse_response_refusal_with_moderate_depth_is_accepted() -> None:
    response = FakeResponse(
        content=[thinking_block("sig"), tool_use_block("toolu_1", "get_issue", {"issue_id": _nested(70)})],
        stop_reason="refusal",
        usage=FakeUsage(10, 5),
    )
    step = parse_response(response, "claude-opus-5-5")
    assert step.message.tool_calls == () and step.message.text == ""


# ---------------------------------------------------------------- 2. build_request


def _base() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "do it"},
    ]


def _history(blocks: list[Any], *, args: Any = None, tool_content: str = '{"data":1,"ok":true}') -> list[dict[str, Any]]:
    args = {"issue_id": 1} if args is None else args
    return _base() + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "name": "get_issue", "args": args}],
            "provider_content": blocks,
        },
        {"role": "tool", "tool_call_id": "t1", "content": tool_content},
    ]


def _good_blocks(args: Any = None) -> list[dict[str, Any]]:
    args = {"issue_id": 1} if args is None else args
    return [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "get_issue", "input": args},
    ]


@pytest.mark.parametrize("bad_type", [["text"], {"k": "v"}, None, 3])
def test_build_request_unhashable_block_type_is_value_error(bad_type: Any) -> None:
    blocks = _good_blocks()
    blocks.insert(0, {"type": bad_type, "text": "x"})
    with pytest.raises(ValueError):
        build_request(_history(blocks), TOOL_SPECS, MC)


def test_build_request_deep_tool_content_is_value_error() -> None:
    deep = '{"ok": true, "data": ' + "[" * 100000 + "]" * 100000 + "}"
    with pytest.raises(ValueError):
        build_request(_history(_good_blocks(), tool_content=deep), TOOL_SPECS, MC)


def test_build_request_deep_args_is_value_error() -> None:
    args = {"x": _nested(5000)}
    with pytest.raises(ValueError):
        build_request(_history(_good_blocks(args), args=args), TOOL_SPECS, MC)


def test_build_request_deep_tool_schema_is_value_error() -> None:
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object", "x": _nested(5000)}}
    with pytest.raises(ValueError):
        build_request(_base(), (tool,), MC)


def test_anthropic_policy_never_calls_client_for_these_histories() -> None:
    client = FakeClient([])
    blocks = _good_blocks()
    blocks.insert(0, {"type": ["text"], "text": "x"})
    with pytest.raises(ValueError):
        AnthropicPolicy(client).next_message(_history(blocks), TOOL_SPECS)
    assert client.requests == []


# ---------------------------------------------------------------- 3. 深度上限一致


def _trajectory() -> Trajectory:
    task = get_smoke_task(TASK_ID)
    return run_episode(task, oracle_policy(REFERENCE_SOLUTIONS[TASK_ID]), max_turns=10, clock=lambda: 0.0)


def _with_meta_depth(trajectory: Trajectory, depth: int) -> Trajectory:
    """turn 0 的 meta = {"x": [[...]]}，葉節點在文件中的深度 = 3（turns/[0]/meta）+ 1（"x"）+ depth。"""
    turns = list(trajectory.turns)
    turns[0] = dataclasses.replace(turns[0], meta={"x": _nested(depth)})
    return dataclasses.replace(trajectory, turns=tuple(turns))


def _accepts(fn: Any) -> bool:
    try:
        fn()
    except (TrajectoryError, CheckpointError):
        return False
    return True


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_validate_and_from_dict_agree_on_turn_meta_depth(offset: int) -> None:
    depth = MAX_DOCUMENT_DEPTH - 4 + offset  # offset 0：葉節點剛好在文件第 MAX_DOCUMENT_DEPTH 層
    trajectory = _with_meta_depth(_trajectory(), depth)
    validated = _accepts(trajectory.validate)
    loaded = _accepts(lambda: Trajectory.from_dict(trajectory.to_dict()))
    assert validated == loaded == (offset <= 0)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_validate_and_from_dict_agree_on_final_snapshot_depth(offset: int) -> None:
    trajectory = _trajectory()
    snapshot = copy.deepcopy(trajectory.final_snapshot)
    snapshot["extra"] = _nested(MAX_DOCUMENT_DEPTH - 2 + offset)  # final_snapshot(1) + extra(2) + depth
    forged = dataclasses.replace(trajectory, final_snapshot=snapshot)
    # 結構規則之外的內容檢查可能另外拒絕；這裡只要求兩條路徑在深度上的結論一致
    assert _accepts(lambda: _depth_only(forged.validate)) == _accepts(lambda: _depth_only(lambda: Trajectory.from_dict(forged.to_dict())))


def _depth_only(fn: Any) -> None:
    try:
        fn()
    except (TrajectoryError, CheckpointError) as exc:
        if "nested deeper" in str(exc):
            raise
    return None


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_checkpoint_verify_and_from_dict_agree_on_turn_meta_depth(offset: int) -> None:
    trajectory = _trajectory()
    checkpoint = make_checkpoint(trajectory, 2)
    turns = list(checkpoint.turns)
    turns[0] = dataclasses.replace(turns[0], meta={"x": _nested(MAX_DOCUMENT_DEPTH - 4 + offset)})
    forged = dataclasses.replace(checkpoint, turns=tuple(turns))
    verified = _accepts(lambda: _depth_only(forged.verify))
    loaded = _accepts(lambda: _depth_only(lambda: Checkpoint.from_dict(_checkpoint_dict(forged))))
    assert verified == loaded == (offset <= 0)


def _checkpoint_dict(checkpoint: Checkpoint) -> dict[str, Any]:
    data = checkpoint.to_dict()
    return data


# ---------------------------------------------------------------- 4–6. run_logged


def _key(rollout_index: int = 0) -> RolloutKey:
    return RolloutKey(None, None, None, rollout_index, 1)


class _Raising:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self._inner = oracle_policy(REFERENCE_SOLUTIONS[TASK_ID])

    @property
    def agent_config(self) -> Any:
        return self._inner.agent_config

    def next_message(self, messages: Any, tools: Any) -> Any:
        raise self._error


def _run(policy: Any) -> Trajectory:
    return run_episode(get_smoke_task(TASK_ID), policy, max_turns=10, clock=lambda: 0.0)


def _logged(policy: Any, log: Path, *, run: Any = _run, key: RolloutKey | None = None, max_turns: int = 10) -> Any:
    return run_logged(run, policy, task=get_smoke_task(TASK_ID), key=key or _key(), max_turns=max_turns, log_path=log)


def test_unencodable_request_id_still_writes_a_row(tmp_path: Path) -> None:
    cls = type("RateLimitError", (Exception,), {"__module__": "anthropic"})
    error = cls("rate limited")
    error.request_id = "req_\ud800"  # type: ignore[attr-defined]
    log = tmp_path / "m.jsonl"
    assert _logged(_Raising(error), log) is None
    (row,) = read_manifest(log)
    assert (row.outcome, row.error_class) == ("aborted", "infrastructure")
    assert row.request_id is None or row.request_id.encode("utf-8")


def test_non_str_module_is_harness_and_still_logged(tmp_path: Path) -> None:
    cls = type("Weird", (Exception,), {})
    cls.__module__ = None  # type: ignore[assignment]
    error = cls("weird")
    assert classify_exception(error) == "harness"
    log = tmp_path / "m.jsonl"
    with pytest.raises(cls) as info:
        _logged(_Raising(error), log)
    assert info.value is error
    (row,) = read_manifest(log)
    assert row.error_class == "harness" and isinstance(row.exception_type, str)


def test_missing_manifest_directory_is_rejected_before_running(tmp_path: Path) -> None:
    called: list[Any] = []

    def run(policy: Any) -> Trajectory:
        called.append(policy)
        return _run(policy)

    with pytest.raises(ValueError):
        _logged(oracle_policy(REFERENCE_SOLUTIONS[TASK_ID]), tmp_path / "missing" / "m.jsonl", run=run)
    assert called == []


def test_completed_trajectory_must_match_max_turns(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    with pytest.raises(ValueError):
        _logged(oracle_policy(REFERENCE_SOLUTIONS[TASK_ID]), log, max_turns=12)  # run 使用 10
    (row,) = read_manifest(log)
    assert (row.outcome, row.error_class) == ("aborted", "harness")


def test_completed_trajectory_must_match_task(tmp_path: Path) -> None:
    other = "smoke-close-duplicate"

    def run(policy: Any) -> Trajectory:
        return run_episode(get_smoke_task(other), oracle_policy(REFERENCE_SOLUTIONS[other]), max_turns=10, clock=lambda: 0.0)

    log = tmp_path / "m.jsonl"
    with pytest.raises(ValueError):
        _logged(oracle_policy(REFERENCE_SOLUTIONS[TASK_ID]), log, run=run)
    assert read_manifest(log)[0].error_class == "harness"


def test_completed_trajectory_must_match_checkpoint_key(tmp_path: Path) -> None:
    """key 宣稱從頭執行（checkpoint_id None），但 run 回傳的是 rerun（parent 不為 None）→ harness。"""
    from replay.checkpoint import rerun

    source = _trajectory()
    checkpoint = make_checkpoint(source, 1)
    policy = oracle_policy(REFERENCE_SOLUTIONS[TASK_ID])
    log = tmp_path / "m.jsonl"
    with pytest.raises(ValueError):
        _logged(policy, log, run=lambda p: rerun(checkpoint, p, clock=lambda: 0.0))
    assert read_manifest(log)[0].error_class == "harness"
    good = RolloutKey(checkpoint.checkpoint_id, 1, source.trajectory_id, 0, 1)
    assert _logged(policy, log, run=lambda p: rerun(checkpoint, p, clock=lambda: 0.0), key=good) is not None
