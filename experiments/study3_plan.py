"""Study 3 的 sweep 計畫（decision 036 §6）：固定的排列、只取失敗的 source、每個 agent 最多 25 條、source 與 source 之間的預留額。

- **排列**：eval 96 題的 task_id 依字典序排序，以 ``random.Random(PERMUTATION_KEY).shuffle`` 打亂；兩個 agent 共用，在任何 eval
  推論之前寫進凍結紀錄（``FREEZE_RECORD`` 的 ``sweep_permutation``）。執行時只讀凍結紀錄，並核對它等於重算的排列。
- **順序**：排列中屬於失敗 source 的題（source 存在而且 ``grade().success`` 為 False）；source 為 None 的題略過，不計入條數。
- **上限**：完成 ``MAX_SOURCES`` 條就停止（跨段累計）。
- **預留額**：開始每一條**新的** source 之前，剩下的累計上限必須 ≥ 凍結紀錄中的預留額（本機以呼叫數、Haiku 以美元）；接續的
  source 不再檢查。沒有凍結紀錄 → 拒絕執行。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from replay.trajectory import Trajectory

ROOT = Path(__file__).resolve().parents[1]
FREEZE_RECORD = ROOT / "docs" / "decisions" / "036-freeze-record.json"
FREEZE_FORMAT = "agent-gym/study3-freeze/v1"
PERMUTATION_KEY = "study3/sweep-order"
MAX_SOURCES = 25
RESERVE_KEYS = ("local_calls", "haiku_usd")


def sweep_permutation(task_ids: Sequence[str]) -> list[str]:
    """依字典序排序後以固定的 key 打亂（decision 036 §6）。"""
    order = sorted(task_ids)
    random.Random(PERMUTATION_KEY).shuffle(order)
    return order


def read_freeze(path: str | Path | None = None) -> dict[str, Any]:
    """凍結紀錄（JSON 物件）。不存在或格式不符 → ValueError。"""
    path = Path(FREEZE_RECORD if path is None else path)
    if not path.is_file():
        raise ValueError(f"the Study 3 freeze record {path} does not exist; eval sweeps wait for the freeze (decision 036 §9)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != FREEZE_FORMAT:
        raise ValueError(f"{path} is not a Study 3 freeze record ({FREEZE_FORMAT})")
    return data


def frozen_permutation(eval_task_ids: Sequence[str], record: Mapping[str, Any]) -> list[str]:
    """凍結紀錄中的排列；必須等於由 eval pool 重算的排列（否則 ValueError）。"""
    permutation = record.get("sweep_permutation")
    if permutation != sweep_permutation(eval_task_ids):
        raise ValueError("the freeze record's sweep permutation differs from the one computed from the eval pool")
    return list(permutation)


def reserve(record: Mapping[str, Any], kind: str) -> float:
    """凍結紀錄中的預留額：kind = "local_calls" 或 "haiku_usd"。"""
    if kind not in RESERVE_KEYS:
        raise ValueError(f"kind must be one of {RESERVE_KEYS}")
    value = (record.get("reserve") or {}).get(kind)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"the freeze record has no positive reserve {kind!r}")
    return value


def failed_order(sources: Mapping[str, Trajectory | None], permutation: Sequence[str]) -> list[str]:
    """排列中屬於失敗 source 的題（source 為 None 或成功者略過）。"""
    return [t for t in permutation if sources.get(t) is not None and not sources[t].grade().success]
