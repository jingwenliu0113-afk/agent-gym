"""agent-gym 在 gpu-server 上自己的 Ollama server：取得 GPU lock、啟動、確認、停止（decision 029）。

只用標準函式庫（WSL Ubuntu-22.04 的 python3 3.10），不 import 本 repo 的其他模組：tools/with-gpu-lock 把這個檔案經 ssh 的 stdin
傳到 gpu-server，以 ``python3 -c <bootstrap> <bytes> agent_gym_serve <args>`` 執行。agents/ollama.py 從這裡取 SERVER_ENV。

模型檔案與 incident-investigator 共用、各自管理 server（使用者決定，2026-10-11）：
- 共用的 models 目錄（SOURCE_MODELS）**只讀**：不寫入、不刪除、不在裡面建立任何東西。
- 本 Project 的 models 目錄（STATE_DIR/models）：manifest 從共用目錄複製（< 1 KB），blob 以 symlink 指向共用目錄的檔案（不複製、
  不下載）。Ollama 執行時寫的快取（metadata/）因此只會寫到本 Project 的目錄。server 的 HOME 也是本 Project 的（STATE_DIR/home）。
  本 Project 的每一層目錄都必須是真正的目錄、在共用目錄之外；blobs/ 裡只動 symlink（decision 031 §1）。
- server 以 STATE_DIR/bin/agent-gym-ollama（指向 OLLAMA_BIN 的 symlink）啟動：命令列不是 incident-investigator 用來停止自己
  server 的「<binary> serve」。

serve 模式（預設）的 stdout 每一行都是給 wrapper 的協定（``LOG `` 開頭的是 server 的 log）：

    LOCKED                         取得 GPU lock（flock(2)，與 ``flock -n`` 是同一把鎖）
    BUSY <原因>                     lock 被占用、已有 Ollama 程序、port 被占用或 GPU 已在使用：什麼都不啟動、不停止任何程序（exit 75）
    FAILED <原因>                   無法啟動（exit 69；已啟動的 server 照常停止）
    NOT_STARTED <原因>              server 啟動前 stdin 就結束（wrapper 已結束）：不啟動（exit 69）
    SNAPSHOT <sha256>              共用 models 目錄的快照（啟動前）
    STARTING pid=… token=… helper=… helper_token=… port=…   已啟動 server（token：程序開始時間的雜湊，避免 pid 重複使用）
    READY <json>                   server 回應：{"pid", "port", "version", "gpu", "env", "models"}
    VERIFIED <nonce> / UNVERIFIED <nonce>   回覆 stdin 的 ``VERIFY <nonce>``：這個 query 是否出現在這次 server 的 log
    SERVER_EXITED <code>           server 的輸出結束（它自己結束了）
    STOPPED / STOP_FAILED <原因>
    MODELS_UNCHANGED / MODELS_CHANGED <差異>   共用 models 目錄與啟動前的快照比較（不同 → exit 70）
    MODELS_NOT_COMPARED <原因>      （只在 stop 模式）沒有這個 session 的結論、也無法確認 lock 一直屬於它（exit 70）

stdin 結束、SIGHUP / SIGTERM / SIGINT（只設定旗標，不中斷停止的步驟）→ 停止：SIGTERM 整個 process group，stop_grace_s 秒後 SIGKILL；
確認沒有留下這個 session 的程序（命令列參照本 Project 目錄者）、port 不再回應。server 繼承 lock 的 fd（helper 意外消失時
lock 仍隨 server 持有），Linux 上並在 helper 消失時收到 SIGTERM。結束時以 os._exit 結束（見 ``_run``）。

stop 模式（``--stop-pid``，wrapper 的第二條連線）：只停止 STARTING 回報的那個 server（pid、身分 token、process group 都要相符）
與它的 helper，每次送信號前重新確認身分。共用目錄以 helper 自己記下的結論為準（STATE_DIR/sessions/<helper_token>.json）；
見 ``stop_session``。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

MODEL = "qwen3.5:9b-q4_K_M"
OLLAMA_BIN = "~/opt/ollama-0.35.0/bin/ollama"  # incident-investigator 部署的 Ollama 0.35.0（共用、不修改）
SOURCE_MODELS = "~/ii-ollama-models"  # incident-investigator 的 models 目錄：只讀
STATE_DIR = "~/agent-gym-ollama"  # 本 Project：models/（manifest 複本、指向共用 blob 的 symlink、Ollama 的快取）、home/
LOCK_FILE = "/tmp/ai-portfolio-gpu.lock"
PORT = 11436  # WSL loopback；incident-investigator 用 11434
SESSIONS = "sessions"  # STATE_DIR 之下：每個 session 的結論（stop 模式讀取；decision 031 第二輪）
LAUNCHER = os.path.join("bin", "agent-gym-ollama")  # STATE_DIR 之下、指向 OLLAMA_BIN 的 symlink：server 的命令列不是「<binary> serve」
OTHER_PROCESS_NAMES = ("ollama", "llama-server")  # Ollama 0.35 的推論程序是 <install>/lib/ollama/llama-server
GPU_BUSY_MIB = 4096  # 已用的 GPU 記憶體達到這個值 → 其他 Project 可能已載入模型（16 GB 容不下兩個）
READY_TIMEOUT_S = 60.0
STOP_GRACE_S = 30.0
VERIFY_TIMEOUT_S = 10.0
HASHED_FILE_MAX = 1 << 20  # 快照中 <= 1 MiB 的檔案比對內容（manifest、metadata）；blob 比對大小、時間、inode
EXIT_BUSY, EXIT_FAILED, EXIT_DIRTY = 75, 69, 70

# server 的固定環境（decision 029 §2）：屬於模型設定（agents/ollama.py 的 model_config["server_env"]）。OLLAMA_HOST、OLLAMA_MODELS、
# HOME 是部署細節，由 helper 另外設定；呼叫端環境裡的其他 OLLAMA_* 一律不傳給 server。
SERVER_ENV = {
    "OLLAMA_KEEP_ALIVE": "-1",  # session 期間模型一直留在 GPU；結束時整個 server 停止
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_NOPRUNE": "1",  # 啟動時不清理 models 目錄
    "OLLAMA_NO_CLOUD": "1",
    "OLLAMA_NUM_PARALLEL": "1",
}

NONCE = re.compile(r"[0-9a-f]{32}")
HEX = re.compile(r"[0-9a-f]{8,64}")
NONCE_IN_LOG = re.compile(r"agent_gym_session=([0-9a-f]{32})")
TAG_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DIGEST = re.compile(r"sha256:([0-9a-f]{64})")

_output = threading.Lock()


def emit(line: str) -> None:
    """一行協定輸出；wrapper 已經不在（broken pipe）時忽略。"""
    with _output:
        try:
            sys.stdout.write(line.replace("\r", " ").replace("\n", " ") + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass


STOP = threading.Event()  # SIGHUP / SIGTERM / SIGINT 只設定這個旗標（不丟例外：停止的步驟不會被中途打斷）
STDIN_CLOSED = threading.Event()  # wrapper 關閉了 stdin（指令結束、wrapper 被中斷或連線中斷）


def _request_stop(signum: int, frame: object) -> None:
    STOP.set()


def _handle_signals(handler: object) -> None:
    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------- 快照（共用 models 目錄只讀的證據）


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root: str) -> dict[str, list[object]]:
    """root 底下每個項目（含 root）的 [類型, 權限, 大小, mtime, ctime, inode, link 數, uid, (symlink 目標 | 小檔案的 sha256)]。
    不含 atime（讀取不算改動）；不跟隨 symlink。"""
    entries: dict[str, list[object]] = {}

    def record(path: str, rel: str) -> None:
        st = os.lstat(path)
        item: list[object] = [stat.S_IFMT(st.st_mode), stat.S_IMODE(st.st_mode), st.st_size, st.st_mtime_ns, st.st_ctime_ns,
                              st.st_ino, st.st_nlink, st.st_uid]
        if stat.S_ISLNK(st.st_mode):
            item.append(os.readlink(path))
        elif stat.S_ISREG(st.st_mode) and st.st_size <= HASHED_FILE_MAX:
            item.append(_sha256_file(path))
        entries[rel] = item

    record(root, ".")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(dirnames) + sorted(filenames):
            path = os.path.join(dirpath, name)
            record(path, os.path.relpath(path, root))
    return entries


def snapshot_digest(entries: dict[str, list[object]]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def diff_snapshots(before: dict[str, list[object]], after: dict[str, list[object]]) -> list[str]:
    changes = [f"removed {rel}" for rel in sorted(set(before) - set(after))]
    changes += [f"added {rel}" for rel in sorted(set(after) - set(before))]
    changes += [f"changed {rel}" for rel in sorted(set(before) & set(after)) if before[rel] != after[rel]]
    return changes


# ---------------------------------------------------------------- 本 Project 的 models 目錄


def manifest_path(model: str) -> str:
    """``name:tag`` → ``manifests/registry.ollama.ai/library/<name>/<tag>``（只接受單純的名稱，不含路徑）。"""
    name, sep, tag = model.partition(":") if isinstance(model, str) else ("", "", "")
    if not sep or not TAG_PART.fullmatch(name) or not TAG_PART.fullmatch(tag) or ".." in (name + tag):
        raise ValueError(f"{model!r} is not a plain Ollama model tag (name:tag)")
    return os.path.join("manifests", "registry.ollama.ai", "library", name, tag)


def _nested(a: str, b: str) -> bool:
    a, b = os.path.realpath(a), os.path.realpath(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def _real_dirs(root: str, rel: str, source: str) -> str:
    """root/rel 的每一層都必須是真正的目錄（lstat，不是 symlink），不存在就建立；每一層的 realpath 都不可與 source 相同或互相包含。
    檢查在建立之前，所以不會經過 symlink 在 source 裡建立任何東西。回傳 root/rel；違反 → ValueError。"""
    path = root
    for part in [item for item in rel.split(os.sep) if item]:
        path = os.path.join(path, part)
        if _nested(path, source):
            raise ValueError(f"{path} must be outside the shared models directory")
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            os.mkdir(path)
            st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"{path} must be a real directory (not a symlink or a file)")
    return path


def _link(link: str, target: str) -> None:
    """link（本 Project 的真正目錄裡）成為指向 target 的 symlink。已是正確的 symlink → 不動；其他 symlink → 換掉；
    一般檔案或目錄 → ValueError（不刪除：它可能是共用檔案的 hard link）。"""
    try:
        st = os.lstat(link)
    except FileNotFoundError:
        os.symlink(target, link)
        return
    if not stat.S_ISLNK(st.st_mode):
        raise ValueError(f"{link} must be a symlink; it is left as it is")
    if os.readlink(link) != target:
        os.unlink(link)  # 只刪除 symlink 本身
        os.symlink(target, link)


def _root(path: str, source: str) -> str:
    """建立 path（本 Project 的根目錄）；它的 realpath 不可與 source 相同或互相包含（在建立之前與之後都檢查）。"""
    if _nested(path, source):
        raise ValueError("this project's directories must be outside the shared models directory (and not contain it)")
    os.makedirs(path, exist_ok=True)
    if _nested(path, source) or not os.path.isdir(path):
        raise ValueError("this project's directories must be outside the shared models directory (and not contain it)")
    return path


def prepare_models(source: str, models: str, model: str) -> dict[str, object]:
    """在 models（本 Project）建立 model 的 manifest 複本，與指向 source（共用、只讀）每個 blob 的 symlink（decision 029 §1、031 §1）。
    - source 的 manifest 必須存在；每個 blob 必須是大小相符的一般檔案。
    - models 與它之下會寫入的每一層目錄（blobs、manifest 的上層）都必須是真正的目錄、在 source 之外（不是就建立）。
    - blobs/ 裡只建立或替換 symlink；一般檔案 → ValueError（不刪除）。manifest 以 mkstemp + os.replace 寫入（不經過任何 symlink）。
    只讀取 source；違反 → ValueError。"""
    rel = manifest_path(model)
    src = os.path.join(source, rel)
    try:
        if not stat.S_ISREG(os.lstat(src).st_mode):
            raise ValueError(f"the shared manifest {rel} is not a regular file")
        with open(src, "rb") as handle:
            data = handle.read()
        manifest = json.loads(data)
        blobs = [manifest["config"], *manifest["layers"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"the shared models directory has no usable manifest for {model}: {exc}") from exc
    targets = []
    for blob in blobs:
        match = DIGEST.fullmatch(blob.get("digest", "")) if isinstance(blob, dict) else None
        if match is None or not isinstance(blob.get("size"), int):
            raise ValueError(f"the manifest of {model} lists an invalid blob: {blob!r}")
        name = f"sha256-{match.group(1)}"
        target = os.path.join(source, "blobs", name)
        try:
            st = os.lstat(target)
        except OSError as exc:
            raise ValueError(f"the shared blob {name} is missing: {exc}") from exc
        if not stat.S_ISREG(st.st_mode) or st.st_size != blob["size"]:
            raise ValueError(f"the shared blob {name} is not a regular file of {blob['size']} bytes")
        targets.append((name, target))
    _root(models, source)
    blob_dir = _real_dirs(models, "blobs", source)
    manifest_dir = _real_dirs(models, os.path.dirname(rel), source)
    metadata_dir = _real_dirs(models, "metadata", source)  # Ollama 執行時寫快取的地方（decision 031 第二輪）
    for name in os.listdir(metadata_dir):
        st = os.lstat(os.path.join(metadata_dir, name))
        if stat.S_ISLNK(st.st_mode) or (stat.S_ISREG(st.st_mode) and st.st_nlink > 1):
            raise ValueError(f"{os.path.join(metadata_dir, name)} is linked to another file; it is left as it is")
    for name, target in targets:
        _link(os.path.join(blob_dir, name), target)
    dst = os.path.join(manifest_dir, os.path.basename(rel))
    current = None
    try:
        st = os.lstat(dst)
        if stat.S_ISREG(st.st_mode):
            with open(dst, "rb") as handle:
                current = handle.read()
    except FileNotFoundError:
        pass
    if current != data:
        fd, tmp = tempfile.mkstemp(dir=manifest_dir, prefix=".manifest-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, dst)  # 換掉的是目錄項目（含 symlink 本身），不會寫進它指向的檔案
        except BaseException:
            if os.path.lexists(tmp):
                os.unlink(tmp)
            raise
    return {"source": SOURCE_MODELS, "blobs": len(targets), "manifest_sha256": hashlib.sha256(data).hexdigest()}


def prepare_state(source: str, state: str, model: str, binary: str) -> dict[str, object]:
    """本 Project 的目錄（decision 029 §1、031 §1）：state/models（``prepare_models``）、state/home（server 的 HOME，含 .ollama）、
    state/bin/agent-gym-ollama（指向 binary 的 symlink：server 以它啟動）、state/sessions（每個 session 的結論）。
    全部在真正的目錄裡、在 source 之外。回傳 {"models", "home", "launcher", "sessions"}；違反 → ValueError。"""
    _root(state, source)
    models = os.path.join(_real_dirs(state, "models", source))
    info = prepare_models(source, models, model)
    home = _real_dirs(state, "home", source)
    _real_dirs(home, ".ollama", source)  # Ollama 在 HOME/.ollama 寫 identity key
    launcher = os.path.join(_real_dirs(state, os.path.dirname(LAUNCHER), source), os.path.basename(LAUNCHER))
    _link(launcher, binary)
    sessions = _real_dirs(state, SESSIONS, source)
    return {"models": info, "home": home, "launcher": launcher, "sessions": sessions}


# ---------------------------------------------------------------- 程序


def _processes() -> list[tuple[int, int, int, str]]:
    """(pid, ppid, pgid, 命令列) 的清單。"""
    out = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,pgid=,args="], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else ""))
    return rows


def _is_helper_args(tokens: list[str]) -> bool:
    return any(item == "agent_gym_serve" or os.path.basename(item) == "ollama_serve.py" for item in tokens)


def ollama_processes(binary: str, launcher: str, names: tuple[str, ...]) -> list[tuple[int, str]]:
    """看起來是 Ollama 的程序（不含自己與 helper）：命令列中有 binary 或 launcher、有 Ollama 安裝目錄（binary 的上上層）之下的
    路徑，或程序名稱屬於 names（例如 ollama、llama-server）。啟動前的檢查用（寧可多擋）。"""
    install = os.path.dirname(os.path.dirname(binary)) + os.sep
    found = []
    for pid, _, _, args in _processes():
        tokens = args.split()
        if pid == os.getpid() or not tokens or _is_helper_args(tokens):
            continue
        if (binary in tokens or launcher in tokens or any(item.startswith(install) for item in tokens)
                or os.path.basename(tokens[0]) in names):
            found.append((pid, args))
    return found


def session_processes(state: str) -> list[tuple[int, str]]:
    """這個 session 的程序：命令列中有本 Project 目錄（state）之下的路徑——server（以 launcher 啟動）與它的推論程序
    （``--model <state>/models/...``）；不含 helper。其他 Project 的程序不會參照這個目錄。"""
    prefix = state.rstrip(os.sep) + os.sep
    found = []
    for pid, _, _, args in _processes():
        tokens = args.split()
        if pid != os.getpid() and tokens and not _is_helper_args(tokens) and any(item.startswith(prefix) for item in tokens):
            found.append((pid, args))
    return found


def _args_of(pid: int) -> str | None:
    for row in _processes():
        if row[0] == pid:
            return row[3]
    return None


def start_token(pid: int, proc_root: str = "/proc") -> str | None:
    """程序的身分 token（避免 pid 被重複使用）；程序不存在 → None。
    Linux：``/proc/<pid>/stat`` 第 22 欄（開機後的 clock ticks）加上 boot_id 的雜湊，不受牆上時鐘調整影響（WSL2 在主機休眠後
    會調整時鐘，``ps -o lstart`` 會因此改變；decision 031 §1）。其他平台（測試用的 macOS）：``ps -o lstart=`` 的雜湊。"""
    boot_id = os.path.join(proc_root, "sys", "kernel", "random", "boot_id")
    if os.path.isfile(boot_id):
        try:
            with open(os.path.join(proc_root, str(pid), "stat"), "rb") as handle:
                data = handle.read().decode("ascii", "replace")
            with open(boot_id, encoding="ascii") as handle:
                boot = handle.read().strip()
            starttime = data[data.rindex(")") + 2:].split()[19]  # ")" 之後從第 3 欄開始：第 22 欄是 index 19
        except (OSError, ValueError, IndexError):
            return None
        return hashlib.sha256(f"{boot}:{starttime}".encode("ascii")).hexdigest()[:16]
    result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def signal_process(pid: int, signum: int, still_ours) -> bool:  # type: ignore[no-untyped-def]
    """``still_ours()`` 為真才送出信號（每次送之前都重新確認身分）；程序已經不在 → False（不是錯誤）。回傳是否送出。"""
    if not still_ours():
        return False
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        return False
    return True


def _group_alive(pgid: int) -> bool:
    """group 裡還有不是 zombie 的程序。macOS 對只剩 zombie 的 group 回 EPERM，所以 EPERM 時也逐一檢查。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return any(group == pgid and _state(pid) not in ("", "Z") for pid, _, group, _ in _processes())


def _state(pid: int) -> str:
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.stdout.strip()[:1]


def _wait(condition, timeout: float) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while True:
        if condition():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


def exit_status(proc: subprocess.Popen[bytes]) -> int | None:
    """子程序結束了 → exit status，還在 → None。**不回收**它：尚未回收的 server 讓 pid / pgid 保持被占用，停止 group 的期間不會被
    別的程序重複使用（decision 031 第二輪）。已經回收過 → Popen 記下的 returncode。"""
    if proc.returncode is not None:
        return proc.returncode
    waitid = getattr(os, "waitid", None)
    if waitid is None:  # 沒有 waitid 的平台（測試用的 macOS Python）：zombie = 已結束，結束碼未知
        return -1 if _state(proc.pid) in ("", "Z") else None
    try:
        info = waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return -1
    if info is None or info.si_pid == 0:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def stop_group(pgid: int, grace: float, reap=None, still_ours=None) -> bool:  # type: ignore[no-untyped-def]
    """SIGTERM 整個 process group，grace 秒後仍有程序 → SIGKILL；回傳是否已全部結束（只剩 zombie 也算結束）。
    still_ours：每次送信號之前確認這個 group 仍屬於這次 session，否則不送、回傳 False。全部結束後才呼叫 reap（回收自己的子程序）。"""

    def gone() -> bool:
        return not _group_alive(pgid)

    result = None
    for signum, timeout in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        if gone():
            result = True
            break
        if still_ours is not None and not still_ours():
            result = False
            break
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            result = True
            break
        except PermissionError:  # 只剩 zombie（macOS），或不是自己的程序
            result = gone()
            break
        if _wait(gone, timeout):
            result = True
            break
    if result is None:
        result = gone()
    if result and reap is not None:
        reap()
    return result


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def gpu_memory() -> dict[str, int]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited with {result.returncode}: {result.stderr.strip()[:200]}")
    used, total = (int(value.strip()) for value in result.stdout.strip().splitlines()[0].split(","))
    return {"used_mib": used, "total_mib": total}


def _die_with_parent() -> None:  # pragma: no cover - 只在 Linux 的子程序中執行
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def server_environment(port: int, models: str, home: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("OLLAMA_")}
    env.update(SERVER_ENV)
    env.update(HOME=home, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=models)
    return env


class LogReader(threading.Thread):
    """把 server 的輸出轉成 ``LOG `` 行，並記錄出現過的 agent_gym_session 值。"""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.nonces: set[str] = set()
        self.changed = threading.Condition()

    def run(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            emit("LOG " + line)
            found = NONCE_IN_LOG.findall(line)
            if found:
                with self.changed:
                    self.nonces.update(found)
                    self.changed.notify_all()
        code = exit_status(self.proc)
        emit(f"SERVER_EXITED {'' if code is None else code}".rstrip())

    def seen(self, nonce: str, timeout: float) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: nonce in self.nonces, timeout)


def wait_ready(proc: subprocess.Popen[bytes], port: int, timeout: float) -> str | None:
    """``/api/version`` 回應且 server 程序仍在 → 版本字串；server 結束、逾時、要求停止或 stdin 結束 → None。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and exit_status(proc) is None and not STOP.is_set() and not STDIN_CLOSED.is_set():
        try:
            with opener.open(f"http://127.0.0.1:{port}/api/version", timeout=2) as response:
                body = json.loads(response.read())
            if isinstance(body, dict) and isinstance(body.get("version"), str) and exit_status(proc) is None:
                return str(body["version"])
        except (OSError, ValueError):
            pass
        time.sleep(0.3)
    return None


# ---------------------------------------------------------------- serve 模式


def _acquire_lock(path: str) -> int | None:
    fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _stdin_lines() -> "queue.Queue[bytes]":
    """在另一個 thread 讀 stdin 的每一行；結束時設定 STDIN_CLOSED。主 thread 因此可以同時檢查停止的旗標，
    也能在 server 啟動前發現 wrapper 已經結束（decision 031 §1）。"""
    lines: "queue.Queue[bytes]" = queue.Queue()

    def read() -> None:
        for raw in iter(sys.stdin.buffer.readline, b""):
            lines.put(raw)
        STDIN_CLOSED.set()

    threading.Thread(target=read, daemon=True).start()
    return lines


def _ended() -> bool:
    return STOP.is_set() or STDIN_CLOSED.is_set()


def _precheck(args: argparse.Namespace, binary: str, launcher: str) -> tuple[int, str, dict[str, int] | None]:
    """啟動前的檢查：(0, "", gpu) 或 (exit code, 協定行, None)。不啟動、不停止任何程序。"""
    names = tuple(name for name in args.process_names.split(",") if name)
    others = ollama_processes(binary, launcher, names)
    if others:
        pids = ", ".join(str(pid) for pid, _ in others)
        return EXIT_BUSY, f"BUSY Ollama processes are already running (pids {pids}); none of them is used or stopped", None
    if port_open(args.port):
        return EXIT_BUSY, f"BUSY something already listens on 127.0.0.1:{args.port}; it is neither used nor stopped", None
    try:
        gpu = gpu_memory()
    except (OSError, RuntimeError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return EXIT_FAILED, f"FAILED the GPU could not be checked: {exc}", None
    if gpu["used_mib"] >= args.gpu_busy_mib:
        return (EXIT_BUSY, f"BUSY the GPU already uses {gpu['used_mib']} MiB (limit {args.gpu_busy_mib}): "
                           "another workload may hold a model", None)
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        return EXIT_FAILED, f"FAILED the Ollama binary {args.binary} is missing", None
    return 0, "", gpu


def serve(args: argparse.Namespace) -> int:
    _handle_signals(_request_stop)
    lock_fd = _acquire_lock(args.lock_file)
    if lock_fd is None:
        emit(f"BUSY the GPU lock {args.lock_file} is held by another workload")
        return EXIT_BUSY
    emit("LOCKED")
    lines = _stdin_lines()
    binary = os.path.expanduser(args.binary)
    source = os.path.expanduser(args.source_models)
    state = os.path.expanduser(args.state_dir)
    launcher = os.path.join(state, LAUNCHER)
    status, line, gpu = _precheck(args, binary, launcher)
    if status:
        emit(line)
        return status
    try:
        before = snapshot(source)
        prepared = prepare_state(source, state, args.model, binary)
    except (OSError, ValueError) as exc:
        emit(f"FAILED {exc}")
        return EXIT_FAILED
    if _ended():
        emit("NOT_STARTED the wrapper ended before the server was started")
        return EXIT_FAILED
    emit(f"SNAPSHOT {snapshot_digest(before)}")
    home = str(prepared["home"])
    proc = subprocess.Popen(
        [launcher, "serve"], env=server_environment(args.port, os.path.join(state, "models"), home), cwd=home,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        pass_fds=(lock_fd,), preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
    helper_token = start_token(os.getpid()) or "-"
    emit(f"STARTING pid={proc.pid} token={start_token(proc.pid) or '-'} helper={os.getpid()} "
         f"helper_token={helper_token} port={args.port}")
    reader = LogReader(proc)
    reader.start()
    version = wait_ready(proc, args.port, args.ready_timeout_s)
    if version is None:
        reason = "the wrapper ended" if _ended() else f"the server did not answer on 127.0.0.1:{args.port}"
        emit(f"FAILED {reason} (exit status {exit_status(proc)})")
        status = EXIT_FAILED
    else:
        emit("READY " + json.dumps({"pid": proc.pid, "port": args.port, "version": version, "gpu": gpu, "env": SERVER_ENV,
                                    "models": prepared["models"]}, sort_keys=True))
        while not STOP.is_set():
            try:
                raw = lines.get(timeout=0.5)
            except queue.Empty:
                if STDIN_CLOSED.is_set():
                    break
                continue
            words = raw.decode("utf-8", "replace").split()
            if len(words) == 2 and words[0] == "VERIFY" and NONCE.fullmatch(words[1]):
                emit(f"{'VERIFIED' if reader.seen(words[1], VERIFY_TIMEOUT_S) else 'UNVERIFIED'} {words[1]}")
    stopped = stop_group(proc.pid, args.stop_grace_s, reap=proc.poll)
    remaining = session_processes(state)  # 只看這個 session 的程序（含離開了 process group 的推論程序）
    if stopped and not remaining and not port_open(args.port):
        stopped_line = "STOPPED"
    else:
        detail = f"processes {', '.join(str(pid) for pid, _ in remaining)} remain" if remaining else "the server is still up"
        stopped_line = f"STOP_FAILED {detail}"
        status = status or EXIT_DIRTY
    emit(stopped_line)
    reader.join(timeout=5)
    changes = diff_snapshots(before, snapshot(source))
    models_line = f"MODELS_CHANGED {'; '.join(changes[:10])}" if changes else "MODELS_UNCHANGED"
    if changes:
        status = status or EXIT_DIRTY
    _record_verdict(str(prepared["sessions"]), helper_token, stopped_line, models_line)  # 仍持有 lock 時記下
    emit(models_line)
    return status


def _record_verdict(sessions: str, helper_token: str, stopped: str, models: str) -> None:
    """這個 session 自己的結論（在仍持有 lock 時比較的結果），給第二條連線讀取；寫入本 Project 的目錄（mkstemp + replace）。"""
    if not HEX.fullmatch(helper_token):
        return
    try:
        fd, tmp = tempfile.mkstemp(dir=sessions, prefix=".verdict-")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"stopped": stopped, "models": models}, handle)
        os.replace(tmp, os.path.join(sessions, helper_token + ".json"))
    except OSError:
        pass


def _read_verdict(sessions: str, helper_token: str) -> dict[str, str] | None:
    if not HEX.fullmatch(helper_token or ""):
        return None
    path = os.path.join(sessions, helper_token + ".json")
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not all(isinstance(record.get(key), str) for key in ("stopped", "models")):
        return None
    return record


# ---------------------------------------------------------------- stop 模式（wrapper 的第二條連線）


def _alive(pid: int | None) -> bool:
    """程序存在且不是 zombie。"""
    return pid is not None and _state(pid) not in ("", "Z")


def _is_server(pid: int, token: str, launcher: str) -> bool:
    """命令列是「<launcher> serve」、身分 token 相符、而且自己是 process group leader（由 helper 以新的 session 啟動）。"""
    args = _args_of(pid)
    if args is None or start_token(pid) != token:
        return False
    tokens = args.split()
    index = tokens.index(launcher) if launcher in tokens else -1
    if index < 0 or tokens[index + 1:index + 2] != ["serve"]:
        return False
    return any(row[0] == pid and row[2] == pid for row in _processes())


def _is_helper(pid: int, token: str) -> bool:
    """命令列是這個 helper（bootstrap 的 agent_gym_serve，或直接執行的 ollama_serve.py），而且身分 token 相符。"""
    args = _args_of(pid)
    return args is not None and start_token(pid) == token and _is_helper_args(args.split())


def _group_is_ours(pgid: int, state: str) -> bool:
    """這個 process group 裡有命令列參照本 Project 目錄的程序（server 或它的推論程序）。"""
    prefix = state.rstrip(os.sep) + os.sep
    return any(group == pgid and _state(pid) not in ("", "Z") and any(item.startswith(prefix) for item in args.split())
               for pid, _, group, args in _processes())


def stop_session(args: argparse.Namespace) -> int:
    """wrapper 的第二條連線：只停止這次 session 的 helper 與 server（decision 029 §4、031）。
    - 先讓身分相符的 helper 自己停止 server（它在仍持有 lock 時比較快照、記下結論）；每次送信號之前都重新確認身分；
      程序已經結束不是錯誤。身分不符、而且 5 秒內沒有消失的程序 → 什麼都不停止。
    - STOPPED 只依身分確認過的程序判定：server 的 process group 與 helper 都已結束，沒有參照本 Project 目錄的程序。
    - 共用目錄：helper 記下的結論優先；沒有結論、而 helper 或 server 在這次停止之前仍持有 lock → 剛結束它們之後取得 lock 比較；
      都已不在（lock 在不知道的時間點釋放）→ MODELS_NOT_COMPARED。"""
    _handle_signals(signal.SIG_IGN)
    state = os.path.expanduser(args.state_dir)
    launcher = os.path.join(state, LAUNCHER)
    pid, token = args.stop_pid, args.stop_token or ""
    helper, helper_token = args.helper_pid, args.helper_token or ""
    helper_ours = helper is not None and _alive(helper) and _is_helper(helper, helper_token)
    server_ours = _alive(pid) and _is_server(pid, token, launcher)
    if helper_ours and signal_process(helper, signal.SIGTERM, lambda: _is_helper(helper, helper_token)):
        _wait(lambda: not _alive(pid) and not _alive(helper), args.stop_grace_s + 20)
    if _alive(pid) and not _is_server(pid, token, launcher):
        if not _wait(lambda: not _alive(pid), 5):  # 正在結束的程序（命令列已改變）：等它消失
            emit(f"STOP_FAILED pid {pid} is not the server this session started; nothing was stopped")
            return EXIT_DIRTY
    stopped = stop_group(pid, args.stop_grace_s, still_ours=lambda: _group_is_ours(pid, state))
    if helper_ours and _alive(helper) and signal_process(helper, signal.SIGKILL, lambda: _is_helper(helper, helper_token)):
        _wait(lambda: not _alive(helper), 10)
    remaining = session_processes(state)
    helper_left = helper_ours and _alive(helper) and _is_helper(helper, helper_token)
    status = 0
    if stopped and not remaining and not helper_left:
        emit("STOPPED")
    else:
        left = [str(item) for item, _ in remaining] + ([str(helper)] if helper_left else [])
        emit(f"STOP_FAILED processes {', '.join(left) or pid} remain")
        status = EXIT_DIRTY
    verdict = _read_verdict(os.path.join(state, SESSIONS), helper_token)
    if verdict is not None:
        emit(verdict["models"])
        if not verdict["models"].startswith("MODELS_UNCHANGED"):
            status = EXIT_DIRTY
    elif args.expect_snapshot and (helper_ours or server_ours):
        lock_fd = _acquire_lock(args.lock_file)  # 剛結束這個 session 的 lock 持有者：lock 仍空著就表示沒有別人介入
        if lock_fd is None:
            emit("MODELS_NOT_COMPARED the GPU lock was taken by another workload right after this session ended")
            status = EXIT_DIRTY
        else:
            try:
                same = snapshot_digest(snapshot(os.path.expanduser(args.source_models))) == args.expect_snapshot
            finally:
                os.close(lock_fd)  # 只在比較的這一瞬間持有
            if same:
                emit("MODELS_UNCHANGED")
            else:
                emit("MODELS_CHANGED the shared models directory differs from the snapshot taken before the server started")
                status = EXIT_DIRTY
    elif args.expect_snapshot:
        emit("MODELS_NOT_COMPARED this session's processes had already ended without a recorded verdict; the GPU lock was "
             "released at an unknown time")
        status = EXIT_DIRTY
    return status


# ---------------------------------------------------------------- 入口


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"invalid port {value}")
    return port


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="agent_gym_serve", description="agent-gym's own Ollama server (decision 029)")
    parser.add_argument("--lock-file", default=LOCK_FILE)
    parser.add_argument("--port", type=_port, default=PORT)
    parser.add_argument("--binary", default=OLLAMA_BIN)
    parser.add_argument("--source-models", default=SOURCE_MODELS)
    parser.add_argument("--state-dir", default=STATE_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--process-names", default=",".join(OTHER_PROCESS_NAMES))
    parser.add_argument("--gpu-busy-mib", type=int, default=GPU_BUSY_MIB)
    parser.add_argument("--ready-timeout-s", type=float, default=READY_TIMEOUT_S)
    parser.add_argument("--stop-grace-s", type=float, default=STOP_GRACE_S)
    parser.add_argument("--stop-pid", type=int, default=None)
    parser.add_argument("--stop-token", default=None)
    parser.add_argument("--helper-pid", type=int, default=None)
    parser.add_argument("--helper-token", default=None)
    parser.add_argument("--expect-snapshot", default=None)
    args = parser.parse_args(argv)
    if not os.path.isabs(args.lock_file):
        parser.error("--lock-file must be an absolute path")
    if args.stop_pid is not None:
        return stop_session(args)
    return serve(args)


def _run(argv: list[str]) -> None:
    """執行後以 os._exit 結束：讀 stdin 的 daemon thread 持有 stdin 的鎖，正常的直譯器結束會因此 abort（decision 031 §1）。
    lock 的 fd 隨程序結束釋放。"""
    try:
        status = main(argv)
    except SystemExit as exc:  # argparse 的用法錯誤
        status = exc.code if isinstance(exc.code, int) else 2
    except BaseException:  # noqa: BLE001 - 未預期的錯誤：印出後照樣以 os._exit 結束（server 由 PDEATHSIG / 第二條連線處理）
        import traceback

        traceback.print_exc()
        status = EXIT_DIRTY
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(status)


if __name__ == "__main__":
    _run(sys.argv[1:])
elif __name__ == "agent_gym_serve":  # tools/with-gpu-lock 的 bootstrap：python3 -c <bootstrap> <bytes> agent_gym_serve <args>
    _run(sys.argv[3:])
