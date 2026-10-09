"""atomic 檔案寫入（decision 023 §2 / 024 §3）：會被後續指令讀取的輸出不得因寫入失敗而只寫一半。"""

from __future__ import annotations

import os
from pathlib import Path


def write_atomic(path: str | Path, data: bytes) -> None:
    """把 data 寫入 path：先寫同目錄的暫存檔 ``.<name>.tmp`` 並 fsync，再以 os.replace 換成目標檔，最後盡量 fsync 目錄。
    任何失敗（含 KeyboardInterrupt）都刪除暫存檔並原樣往外拋；目標檔維持原狀（不存在或先前的內容）。"""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    try:
        descriptor = os.open(target.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
