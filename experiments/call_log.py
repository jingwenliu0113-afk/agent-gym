"""逐次模型呼叫的稽核紀錄（decision 032）：本機模型 run 目錄的 ``calls.jsonl``，每次 chat 一行。

trajectory 只保存完成的嘗試；中止、被拒絕或中斷的嘗試裡的呼叫（與它們的 seed）只在這裡。每一行（canonical JSON）::

    {"format": "agent-gym/call/v1", "call": <這個目錄的第幾次呼叫，從 1 開始>, "utc_time": <送出時間>,
     "task_id", "rollout_index", "attempt", "k": <sweep 的 checkpoint；baseline 為 null>,
     "assistant_turn": <request 中已有的 assistant 訊息數>, "seed": <request 的 options.seed>,
     "request_sha256": "sha256:<canonical JSON 的雜湊>", "seconds": <這次呼叫的秒數>, "ok": true | false,
     "response": {"model", "done_reason", "prompt_eval_count", "eval_count", "total_duration", "message"}   # ok 時
     "error_type", "message"}                                                                             # 失敗時

每一行寫入後 flush + fsync（中斷時已發生的呼叫不會遺失）。檔案在 run 開始時建立（空的），不會覆寫既有的檔案。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

from env.hashing import canonical_json
from replay.trajectory import load_strict_json

CALL_LOG = "calls.jsonl"
CALL_LOG_FORMAT = "agent-gym/call/v1"
RESPONSE_FIELDS = ("model", "done_reason", "prompt_eval_count", "eval_count", "total_duration")


class CallLog:
    """包住一個 Ollama client，記錄每次 ``chat``；``bind(...)`` 回傳附上 task / rollout / attempt / k 的 client。"""

    def __init__(self, client: Any, path: Path) -> None:
        self._client = client
        self._path = Path(path)
        self._count = 0
        self._lock = threading.Lock()
        with open(self._path, "xb"):  # 建立空的紀錄；已存在 → FileExistsError（不覆寫）
            pass

    @property
    def client(self) -> Any:
        return self._client

    def bind(self, *, task_id: str, rollout_index: int, attempt: int, k: int | None) -> "BoundClient":
        return BoundClient(self, {"task_id": task_id, "rollout_index": rollout_index, "attempt": attempt, "k": k})

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._count += 1
            line = (canonical_json({"format": CALL_LOG_FORMAT, "call": self._count, **record}) + "\n").encode("utf-8")
            with open(self._path, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


class BoundClient:
    """CallLog.bind 的結果：``chat`` 經過紀錄，其他屬性（version / tags / show、consecutive_chat_failures）直接取自原本的 client。"""

    def __init__(self, log: CallLog, context: dict[str, Any]) -> None:
        self._log = log
        self._context = context

    def chat(self, request: dict[str, Any]) -> Any:
        options = request.get("options") if isinstance(request, dict) else None
        messages = request.get("messages") if isinstance(request, dict) else None
        record: dict[str, Any] = {
            "utc_time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            **self._context,
            "assistant_turn": sum(1 for m in messages or [] if isinstance(m, dict) and m.get("role") == "assistant"),
            "seed": options.get("seed") if isinstance(options, dict) else None,
            "request_sha256": "sha256:" + hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest(),
        }
        started = time.perf_counter()
        try:
            response = self._log.client.chat(request)
        except BaseException as exc:
            self._log._write({**record, "seconds": time.perf_counter() - started, "ok": False, "error_type": type(exc).__name__,
                              "message": str(exc)[:500]})
            raise
        summary = {name: response.get(name) for name in RESPONSE_FIELDS} if isinstance(response, dict) else {}
        if isinstance(response, dict):
            summary["message"] = response.get("message")
        try:
            canonical_json(summary)
        except (TypeError, ValueError):  # 無法記錄的回覆（adapter 會另外拒絕它）：只記型別
            summary = {"unrecordable": type(response).__name__}
        self._log._write({**record, "seconds": time.perf_counter() - started, "ok": True, "response": summary})
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._log.client, name)


def read_call_log(path: Path) -> list[dict[str, Any]]:
    """讀取 calls.jsonl（strict JSON）；每一行必須是 CALL_LOG_FORMAT 的物件，否則 ValueError（訊息含路徑與行號）。"""
    records = []
    for number, raw in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            record = load_strict_json(raw)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON: {exc}") from exc
        if not isinstance(record, dict) or record.get("format") != CALL_LOG_FORMAT:
            raise ValueError(f"{path}:{number}: not a {CALL_LOG_FORMAT} record")
        records.append(record)
    return records
