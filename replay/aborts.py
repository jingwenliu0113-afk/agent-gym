"""嘗試紀錄（attempt manifest）與例外分類（decision 009 第 5 點）。

實驗腳本必須透過 ``run_logged`` 執行**每一次**嘗試；成功與中止都寫一列 ``AttemptRecord``（JSONL，一行一筆 canonical_json）。
報告的分母只從 manifest 計算；唯一鍵為 ``(task_id, config_id, checkpoint_id, rollout_index, attempt)``。

例外分類（``classify_exception``）：
- ``"agent_output"``：``agents.baseline.AdapterError``。
- ``"infrastructure"``：例外類別的模組為 ``anthropic`` 或 ``anthropic.*``，且（類別名稱為 APIConnectionError / APITimeoutError /
  RateLimitError / InternalServerError / OverloadedError / ServiceUnavailableError，或帶 int 的 ``status_code`` 屬性且為 429 或
  >= 500）。非 anthropic 模組的例外即使帶 status_code 也不算（避免把 harness 自己的錯誤當暫時性錯誤重試）。
- ``"harness"``：其他所有例外（含 ReplayIntegrityError、anthropic 的 4xx、TrajectoryError、CheckpointError、TypeError…）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents.baseline import MAX_EPISODE_ATTEMPTS, AdapterError
from agents.policy import Policy, PolicyStep
from apps.issue_tracker.schema import SCHEMA_VERSION
from env.hashing import canonical_json
from grader.grader import grade
from grader.schema import GRADER_VERSION
from replay.trajectory import Trajectory, load_strict_json
from tasks.schema import Task

ATTEMPT_RECORD_FORMAT = "agent-gym/attempt/v1"
ERROR_CLASSES: tuple[str, ...] = ("infrastructure", "agent_output", "harness")
OUTCOMES: tuple[str, ...] = ("completed", "aborted")
MESSAGE_LIMIT = 1000


@dataclass(frozen=True)
class RolloutKey:
    """一次嘗試在實驗中的位置（由實驗腳本給定）。"""

    checkpoint_id: str | None  # 從頭執行時為 None
    checkpoint_turn: int | None  # 從頭執行時為 None
    source_trajectory_id: str | None  # 從頭執行時為 None
    rollout_index: int  # >= 0
    attempt: int  # 1..MAX_EPISODE_ATTEMPTS


@dataclass(frozen=True)
class AttemptRecord:
    task_id: str
    generator_version: str
    config_id: str
    grader_version: str
    schema_version: str
    max_turns: int
    checkpoint_id: str | None
    checkpoint_turn: int | None
    source_trajectory_id: str | None
    rollout_index: int
    attempt: int
    outcome: str  # OUTCOMES
    trajectory_id: str | None  # completed 時
    termination: str | None  # completed 時
    error_class: str | None  # aborted 時：ERROR_CLASSES
    exception_type: str | None  # aborted 時：f"{module}.{qualname}"
    message: str | None  # aborted 時：str(exception) 截斷至 MESSAGE_LIMIT 字元
    http_status: int | None  # 例外帶 int 的 status_code 時
    request_id: str | None  # 例外帶 str 的 request_id 時
    model_calls_completed: int  # 本次嘗試中成功回傳的 next_message 次數（不含 checkpoint 前綴）
    partial_steps: int | None  # aborted 且有 partial_episode 時：部分軌跡的 tool step 總數（含前綴）
    partial_forbidden_action_count: int | None
    partial_forbidden_attempt_count: int | None
    sdk_version: str | None  # anthropic SDK 版本（可 import 時）
    utc_time: str  # 嘗試結束時的 UTC ISO 8601 時間（量測）

    def __post_init__(self) -> None:
        """欄位型別與取值檢查（違反 → ValueError）；int 一律 ``type(x) is int``。"""
        for name in _STR_FIELDS:
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"AttemptRecord.{name} must be a str")
        for name in _OPTIONAL_STR_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"AttemptRecord.{name} must be a str or None")
        for name in _INT_FIELDS:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"AttemptRecord.{name} must be an int >= 0")
        for name in _OPTIONAL_INT_FIELDS:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"AttemptRecord.{name} must be an int >= 0 or None")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"AttemptRecord.outcome must be one of {OUTCOMES}")
        if self.error_class is not None and self.error_class not in ERROR_CLASSES:
            raise ValueError(f"AttemptRecord.error_class must be one of {ERROR_CLASSES} or None")
        if (self.outcome == "aborted") != (self.error_class is not None):
            raise ValueError("AttemptRecord.error_class must be set exactly when the outcome is aborted")

    def to_dict(self) -> dict[str, Any]:
        """``{"format": ATTEMPT_RECORD_FORMAT, ...所有欄位}``。"""
        return {"format": ATTEMPT_RECORD_FORMAT, **{name: getattr(self, name) for name in ATTEMPT_RECORD_FIELDS}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptRecord:
        """key 集合必須剛好是 format + 所有欄位、format 相符；否則 ValueError。"""
        if not isinstance(data, dict) or set(data) != {"format", *ATTEMPT_RECORD_FIELDS}:
            raise ValueError("attempt record must be an object with keys: format, " + ", ".join(ATTEMPT_RECORD_FIELDS))
        if data["format"] != ATTEMPT_RECORD_FORMAT:
            raise ValueError(f"unsupported attempt record format {data['format']!r}")
        return cls(**{name: data[name] for name in ATTEMPT_RECORD_FIELDS})


ATTEMPT_RECORD_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(AttemptRecord))
_STR_FIELDS = ("task_id", "generator_version", "config_id", "grader_version", "schema_version", "outcome", "utc_time")
_OPTIONAL_STR_FIELDS = (
    "checkpoint_id",
    "source_trajectory_id",
    "trajectory_id",
    "termination",
    "error_class",
    "exception_type",
    "message",
    "request_id",
    "sdk_version",
)
_INT_FIELDS = ("max_turns", "rollout_index", "attempt", "model_calls_completed")
_OPTIONAL_INT_FIELDS = (
    "checkpoint_turn",
    "http_status",
    "partial_steps",
    "partial_forbidden_action_count",
    "partial_forbidden_attempt_count",
)

_INFRASTRUCTURE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)


def classify_exception(exc: BaseException) -> str:
    """依模組 docstring 回傳 ERROR_CLASSES 之一。"""
    if isinstance(exc, AdapterError):
        return "agent_output"
    module = getattr(type(exc), "__module__", None)
    if isinstance(module, str) and (module == "anthropic" or module.startswith("anthropic.")):
        if type(exc).__name__ in _INFRASTRUCTURE_NAMES:
            return "infrastructure"
        status = _status_code(exc)
        if status is not None and (status == 429 or status >= 500):
            return "infrastructure"
    return "harness"


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if type(status) is int else None


class CountingPolicy:
    """包住另一個 policy，計算成功回傳的 ``next_message`` 次數；``agent_config`` 原樣委派。"""

    def __init__(self, inner: Policy) -> None:
        self._inner = inner
        self._calls = 0

    @property
    def calls_completed(self) -> int:
        return self._calls

    @property
    def agent_config(self) -> Any:
        return self._inner.agent_config

    def next_message(self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]) -> PolicyStep:
        step = self._inner.next_message(messages, tools)
        self._calls += 1
        return step


def run_logged(
    run: Callable[[Policy], Trajectory],
    policy: Policy,
    *,
    task: Task,
    key: RolloutKey,
    max_turns: int,
    log_path: str | Path,
) -> Trajectory | None:
    """以 ``CountingPolicy(policy)`` 呼叫 ``run``，並把結果寫成一列 AttemptRecord（附加到 log_path）。

    執行前檢查（違反 → ValueError，不執行 run、不寫紀錄）：rollout_index 為 int >= 0；attempt 為 int 且
    ``1 <= attempt <= MAX_EPISODE_ATTEMPTS``；manifest 中沒有相同唯一鍵；attempt > 1 時同鍵的 attempt - 1 必須存在且為
    ``outcome == "aborted"``、``error_class == "infrastructure"``。
    - 成功：寫 outcome="completed"（trajectory_id、termination），回傳 trajectory。
    - ``Exception``：依 classify_exception 分類；若例外帶 ``partial_episode``（replay.runner.PartialEpisode），以 grader 對其
      records 與 env 狀態計算 partial_* 欄位；寫 outcome="aborted"。infrastructure / agent_output → 回傳 None；
      harness → 寫完紀錄後原樣往外拋。
    - ``KeyboardInterrupt`` 等非 Exception 不攔截、不寫紀錄。
    另外 key 必須是 RolloutKey，checkpoint_id / checkpoint_turn / source_trajectory_id 三者同時為 None（從頭執行）或同時有值
    （str、int >= 0、str）；max_turns 為 int >= 1；log_path 的上層目錄必須存在（避免跑完、花了錢才發現無法記錄）。
    config_id 取自 ``policy.agent_config``（實際使用的 policy）。
    - 完成的 trajectory 必須與這次嘗試相符（task 逐 byte 相同、max_turns、agent 的 config_id、parent 與 key 一致：
      從頭執行時 parent 為 None，否則為 ``{"trajectory_id": source_trajectory_id, "checkpoint_id", "turn": checkpoint_turn}``）；
      不符 → 以 ValueError 記為 harness 中止並拋出。
    - 例外帶來的自由字串（request_id、類別的模組 / 名稱）無法 UTF-8 編碼時以替代字元寫入（request_id 改為 None），
      不得讓記錄本身失敗。
    """
    _check_key(key)
    if not Path(log_path).parent.is_dir():
        raise ValueError(f"the manifest directory {Path(log_path).parent} does not exist")
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be an int >= 1")
    if not isinstance(task, Task):
        raise ValueError("task must be a Task")
    config_id = policy.agent_config.config_id
    unique = (task.task_id, config_id, key.checkpoint_id, key.rollout_index)
    previous = {
        (row.task_id, row.config_id, row.checkpoint_id, row.rollout_index, row.attempt): row
        for row in read_manifest(log_path)
    }
    if (*unique, key.attempt) in previous:
        raise ValueError("this attempt is already in the manifest")
    if key.attempt > 1:
        prior = previous.get((*unique, key.attempt - 1))
        if prior is None or prior.outcome != "aborted" or prior.error_class != "infrastructure":
            raise ValueError("a retry requires the previous attempt to be an infrastructure abort")

    counting = CountingPolicy(policy)
    base = {
        "task_id": task.task_id,
        "generator_version": task.generator_version,
        "config_id": config_id,
        "grader_version": GRADER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "max_turns": max_turns,
        "checkpoint_id": key.checkpoint_id,
        "checkpoint_turn": key.checkpoint_turn,
        "source_trajectory_id": key.source_trajectory_id,
        "rollout_index": key.rollout_index,
        "attempt": key.attempt,
        "sdk_version": _sdk_version(),
    }
    try:
        trajectory = run(counting)
        _check_completed(trajectory, task=task, key=key, max_turns=max_turns, config_id=config_id)
    except Exception as exc:
        error_class = classify_exception(exc)
        steps, actions, attempts = _partial_metrics(task, getattr(exc, "partial_episode", None))
        request_id = getattr(exc, "request_id", None)
        _append(
            log_path,
            AttemptRecord(
                **base,
                outcome="aborted",
                trajectory_id=None,
                termination=None,
                error_class=error_class,
                exception_type=_exception_type(exc),
                message=_message(exc),
                http_status=_status_code(exc),
                request_id=request_id if isinstance(request_id, str) and _encodable(request_id) else None,
                model_calls_completed=counting.calls_completed,
                partial_steps=steps,
                partial_forbidden_action_count=actions,
                partial_forbidden_attempt_count=attempts,
                utc_time=_utc_now(),
            ),
        )
        if error_class == "harness":
            raise
        return None
    _append(
        log_path,
        AttemptRecord(
            **base,
            outcome="completed",
            trajectory_id=trajectory.trajectory_id,
            termination=trajectory.termination,
            error_class=None,
            exception_type=None,
            message=None,
            http_status=None,
            request_id=None,
            model_calls_completed=counting.calls_completed,
            partial_steps=None,
            partial_forbidden_action_count=None,
            partial_forbidden_attempt_count=None,
            utc_time=_utc_now(),
        ),
    )
    return trajectory


@dataclass(frozen=True)
class RetryResult:
    trajectory: Trajectory | None  # 完成時的 trajectory
    attempts: int  # 用掉的嘗試次數（>= 1）
    error_class: str | None  # 最後一次嘗試的 error_class；完成時 None


def run_with_retries(
    run: Callable[[Policy], Trajectory],
    policy_factory: Callable[[int], Policy],
    *,
    task: Task,
    key: RolloutKey,
    max_turns: int,
    log_path: str | Path,
) -> RetryResult:
    """decision 009 的重試規則：key.attempt 必須是 1（否則 ValueError）；attempt = 1, 2, … 依序
    ``run_logged(run, policy_factory(attempt), key=replace(key, attempt=attempt), ...)``：
    - 回傳 trajectory → ``RetryResult(trajectory, attempt, None)``；
    - 回傳 None → 取 manifest 中這次嘗試的 error_class；infrastructure 且 attempt < MAX_EPISODE_ATTEMPTS → 再試；
      否則 ``RetryResult(None, attempt, error_class)``。manifest 找不到這一列 → RuntimeError。
    run_logged 的事前檢查錯誤與 harness 例外原樣往外拋。
    """
    if not isinstance(key, RolloutKey) or key.attempt != 1 or type(key.attempt) is not int:
        raise ValueError("run_with_retries starts from a key with attempt == 1")
    attempt = 0
    while True:
        attempt += 1
        attempt_key = replace(key, attempt=attempt)
        policy = policy_factory(attempt)
        trajectory = run_logged(run, policy, task=task, key=attempt_key, max_turns=max_turns, log_path=log_path)
        if trajectory is not None:
            return RetryResult(trajectory, attempt, None)
        error_class = _logged_error_class(log_path, task.task_id, policy.agent_config.config_id, attempt_key)
        if error_class != "infrastructure" or attempt >= MAX_EPISODE_ATTEMPTS:
            return RetryResult(None, attempt, error_class)


def _logged_error_class(log_path: str | Path, task_id: str, config_id: str, key: RolloutKey) -> str | None:
    """run_logged 剛寫入的那一列的 error_class；找不到代表 manifest 與執行不一致 → RuntimeError（harness 問題）。"""
    wanted = (task_id, config_id, key.checkpoint_id, key.rollout_index, key.attempt)
    for row in reversed(read_manifest(log_path)):
        if (row.task_id, row.config_id, row.checkpoint_id, row.rollout_index, row.attempt) == wanted:
            return row.error_class
    raise RuntimeError("run_logged returned None but the attempt is missing from the manifest")


def read_manifest(log_path: str | Path) -> tuple[AttemptRecord, ...]:
    """讀回 run_logged 寫入的紀錄（依寫入順序）；檔案不存在 → ()。任何一行格式不符 → ValueError。"""
    path = Path(log_path)
    if not path.exists():
        return ()
    rows: list[AttemptRecord] = []
    for number, line in enumerate(path.read_bytes().split(b"\n"), start=1):
        if line == b"":
            continue
        try:
            data = load_strict_json(line)
            rows.append(AttemptRecord.from_dict(data))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: line {number} is not a valid attempt record: {exc}") from exc
    return tuple(rows)


def _check_key(key: Any) -> None:
    if not isinstance(key, RolloutKey):
        raise ValueError("key must be a RolloutKey")
    if type(key.rollout_index) is not int or key.rollout_index < 0:
        raise ValueError("rollout_index must be an int >= 0")
    if type(key.attempt) is not int or not 1 <= key.attempt <= MAX_EPISODE_ATTEMPTS:
        raise ValueError(f"attempt must be an int between 1 and {MAX_EPISODE_ATTEMPTS}")
    origin = (key.checkpoint_id, key.checkpoint_turn, key.source_trajectory_id)
    if origin == (None, None, None):
        return
    if (
        not isinstance(key.checkpoint_id, str)
        or type(key.checkpoint_turn) is not int
        or key.checkpoint_turn < 0
        or not isinstance(key.source_trajectory_id, str)
    ):
        raise ValueError("checkpoint_id, checkpoint_turn and source_trajectory_id must be all None or all set")


def _check_completed(trajectory: Any, *, task: Task, key: RolloutKey, max_turns: int, config_id: str) -> None:
    """完成的 trajectory 必須與這次嘗試的 task / key / max_turns / policy 相符；否則 ValueError（harness）。"""
    if not isinstance(trajectory, Trajectory):
        raise TypeError("run must return a Trajectory")
    if canonical_json(trajectory.task) != canonical_json(task.to_dict()):
        raise ValueError("the trajectory was produced for a different task")
    if trajectory.max_turns != max_turns:
        raise ValueError(f"the trajectory used max_turns={trajectory.max_turns}, not {max_turns}")
    if trajectory.agent.config_id != config_id:
        raise ValueError("the trajectory was produced by a different agent config")
    if key.checkpoint_id is None:
        expected_parent = None
    else:
        expected_parent = {
            "trajectory_id": key.source_trajectory_id,
            "checkpoint_id": key.checkpoint_id,
            "turn": key.checkpoint_turn,
        }
    if canonical_json(trajectory.parent) != canonical_json(expected_parent):
        raise ValueError("the trajectory's parent does not match the rollout key")


def _partial_metrics(task: Task, partial: Any) -> tuple[int | None, int | None, int | None]:
    """中止前已發生步驟的 (tool step 數, forbidden_action_count, forbidden_attempt_count)；沒有 partial_episode → None。
    以部分軌跡自己的 task（``partial.task``）評分。"""
    # 延遲 import：replay.runner 不依賴本模組，避免日後循環
    from replay.runner import PartialEpisode

    if not isinstance(partial, PartialEpisode):
        return None, None, None
    records = partial.records()
    try:
        report = grade(Task.from_dict(partial.task), records, partial.env_snapshot["state"])
    except Exception:  # 評分失敗也必須留下這一列（計數記為 None），不得讓 manifest 漏掉嘗試
        return len(records), None, None
    return len(records), report.forbidden_action_count, report.forbidden_attempt_count


def _encodable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _exception_type(exc: BaseException) -> str:
    cls = type(exc)
    text = f"{getattr(cls, '__module__', None)}.{getattr(cls, '__qualname__', cls.__name__)}"
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _message(exc: BaseException) -> str:
    try:
        text = str(exc)
    except Exception:  # __str__ 本身壞掉時仍要留下紀錄
        text = f"<unprintable {type(exc).__name__}>"
    return text[:MESSAGE_LIMIT].encode("utf-8", errors="replace").decode("utf-8")


def _sdk_version() -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    version = getattr(anthropic, "__version__", None)
    return version if isinstance(version, str) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(log_path: str | Path, record: AttemptRecord) -> None:
    """append-only：一行一筆 canonical_json，寫入後 flush + fsync。"""
    line = canonical_json(record.to_dict()) + "\n"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
