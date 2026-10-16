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
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Sequence

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
            entry = {"format": CALL_LOG_FORMAT, "call": self._count + 1, **record}
            try:
                line = (canonical_json(entry) + "\n").encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError):  # 例如單獨的 surrogate：以 ASCII 跳脫寫入，seed 照樣留下
                line = (json.dumps(entry, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=repr) + "\n").encode()
            with open(self._path, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._count += 1


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


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def interrupt_on_termination() -> None:
    """本機模型的正式 run：SIGTERM / SIGHUP 改為 KeyboardInterrupt（與 Ctrl-C 相同），正在進行的呼叫因此照樣記進紀錄。
    tools/with-gpu-lock 以 SIGTERM 結束指令（指令在自己的 process group，收不到終端機的 Ctrl-C；decision 032 第二輪）。"""
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _interrupt)


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


def count_calls(run_dirs: "Sequence[str | Path]") -> int:
    """這些目錄 ``calls.jsonl`` 的呼叫總數（累計呼叫上限用；decision 034 §8）。任一目錄沒有紀錄 → ValueError。"""
    total = 0
    for run_dir in run_dirs:
        path = Path(run_dir) / CALL_LOG
        if not path.exists():
            raise ValueError(f"{run_dir} has no {CALL_LOG}; the cumulative call count cannot be computed")
        total += len(read_call_log(path))
    return total


def reconcile(run_dirs: "list[Path] | tuple[Path, ...]") -> dict[str, Any] | None:
    """calls.jsonl 與 manifest **逐次嘗試**核對（decision 032 第二輪）。任一目錄沒有紀錄 → None。

    每次嘗試以 (目錄, task_id, k, rollout_index, attempt) 識別（manifest 的 checkpoint_turn 就是 k）：
    - matched：紀錄中成功的呼叫數等於 manifest 該列的 billed_calls；
    - interrupted：紀錄中有、manifest 沒有那一列（嘗試被中斷，manifest 只記結束的嘗試）；
    - mismatched：兩邊都有但數目不同，或 manifest 有計費的呼叫、紀錄中卻沒有（例如 server 回覆後被 adapter 以 harness 例外拒絕）。
    complete ⇔ 每次呼叫都有 seed，而且每一列 manifest 都 matched。"""
    totals = {"calls": 0, "failed": 0, "seeds": 0, "attempts": 0, "matched_attempts": 0, "interrupted_attempts": 0,
              "mismatched_attempts": 0}
    from replay.aborts import read_manifest

    for index, run_dir in enumerate(run_dirs):
        path = Path(run_dir) / CALL_LOG
        if not path.exists():
            return None
        records = read_call_log(path)
        totals["calls"] += len(records)
        totals["failed"] += sum(1 for record in records if not record["ok"])
        totals["seeds"] += sum(1 for record in records if isinstance(record.get("seed"), int))
        ok: dict[tuple[Any, ...], int] = {}
        for record in records:
            key = (index, record.get("task_id"), record.get("k"), record.get("rollout_index"), record.get("attempt"))
            ok[key] = ok.get(key, 0) + (1 if record["ok"] else 0)
        rows = {(index, row.task_id, row.checkpoint_turn, row.rollout_index, row.attempt): row.billed_calls
                for row in read_manifest(Path(run_dir) / "manifest.jsonl")}
        totals["attempts"] += len(rows)
        totals["matched_attempts"] += sum(1 for key, billed in rows.items() if ok.get(key, 0) == billed)
        totals["mismatched_attempts"] += sum(1 for key, billed in rows.items() if ok.get(key, 0) != billed)
        totals["interrupted_attempts"] += sum(1 for key in ok if key not in rows)
    totals["complete"] = totals["seeds"] == totals["calls"] and totals["matched_attempts"] == totals["attempts"]
    return totals

