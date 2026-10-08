"""Canonical JSON 與 state hash（decision log 002）。"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_json(obj: Any) -> str:
    """決定性的 JSON 序列化：key 排序、``separators=(",", ":")``、保留非 ASCII（``ensure_ascii=False``）。

    - dict 的 key 必須是 str，否則 raise TypeError（必須自行遞迴檢查，不可依賴
      json.dumps 默默把 int / bool / None key 轉成字串，否則 ``{1: "a"}`` 與 ``{"1": "a"}`` 會碰撞）。
    - float 的 NaN / ±Infinity → ValueError。
    - tuple 序列化成 list；bool 輸出 true / false（不與 int 混淆）。
    - set、bytes、自訂物件等其他非 JSON 型別 → TypeError。
    """
    _check_json(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_of(obj: Any) -> str:
    """回傳 ``"sha256:" + hex``，對 ``canonical_json(obj)`` 的 UTF-8 bytes 取 SHA-256。

    字串無法以 UTF-8 編碼（例如含 lone surrogate）時 raise ValueError（UnicodeEncodeError 亦可，它是 ValueError 子類）。
    """
    return HASH_PREFIX + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _check_json(obj: Any) -> None:
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"non-finite float is not JSON: {obj!r}")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be str, got {type(key).__name__}")
            _check_json(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _check_json(value)
        return
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")
