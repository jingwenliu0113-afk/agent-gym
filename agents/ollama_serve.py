"""agent-gym 在 gpu-server 上自己的 Ollama server：取得 GPU lock、啟動、確認、停止（decision 029）。

只用標準函式庫（WSL Ubuntu-22.04 的 python3 3.10），不 import 本 repo 的其他模組：tools/with-gpu-lock 把這個檔案經 ssh 的 stdin
傳到 gpu-server，以 ``python3 -c <bootstrap> <bytes> agent_gym_serve <args>`` 執行。agents/ollama.py 從這裡取 SERVER_ENV。

模型檔案與 incident-investigator 共用、各自管理 server（使用者決定，2026-10-11）：
- 共用的 models 目錄（SOURCE_MODELS）**只讀**：不寫入、不刪除、不在裡面建立任何東西。
- 本 Project 的 models 目錄（STATE_DIR/models）：manifest 從共用目錄複製（< 1 KB），blob 以 symlink 指向共用目錄的檔案（不複製、
  不下載）。Ollama 執行時寫的快取（metadata/）因此只會寫到本 Project 的目錄。server 的 HOME 也是本 Project 的（STATE_DIR/home）。

serve 模式（預設）的 stdout 每一行都是給 wrapper 的協定（``LOG `` 開頭的是 server 的 log）：

    LOCKED                         取得 GPU lock（flock(2)，與 ``flock -n`` 是同一把鎖）
    BUSY <原因>                     lock 被占用、已有 Ollama 程序、port 被占用或 GPU 已在使用：什麼都不啟動、不停止任何程序（exit 75）
    FAILED <原因>                   無法啟動（exit 69；已啟動的 server 照常停止）
    SNAPSHOT <sha256>              共用 models 目錄的快照（啟動前）
    STARTING pid=… token=… helper=… helper_token=… port=…   已啟動 server（token：程序開始時間的雜湊，避免 pid 重複使用）
    READY <json>                   server 回應：{"pid", "port", "version", "gpu", "env", "models"}
    VERIFIED <nonce> / UNVERIFIED <nonce>   回覆 stdin 的 ``VERIFY <nonce>``：這個 query 是否出現在這次 server 的 log
    SERVER_EXITED <code>           server 的輸出結束（它自己結束了）
    STOPPED / STOP_FAILED <原因>
    MODELS_UNCHANGED / MODELS_CHANGED <差異>   共用 models 目錄與啟動前的快照比較（不同 → exit 70）

stdin 結束、SIGHUP / SIGTERM / SIGINT（只設定旗標，不中斷停止的步驟）→ 停止：SIGTERM 整個 process group，stop_grace_s 秒後 SIGKILL；確認沒有留下這個執行檔的程序、
port 不再回應。server 繼承 lock 的 fd（helper 意外消失時 lock 仍隨 server 持有），Linux 上並在 helper 消失時收到 SIGTERM。

stop 模式（``--stop-pid``，wrapper 的第二條連線）：只停止 STARTING 回報的那個 server（pid、開始時間、process group 都要相符）與
它的 helper；``--expect-snapshot`` 給定時比較共用 models 目錄。
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
import threading
import time
import urllib.request

MODEL = "qwen3.5:9b-q4_K_M"
OLLAMA_BIN = "~/opt/ollama-0.35.0/bin/ollama"  # incident-investigator 部署的 Ollama 0.35.0（共用、不修改）
SOURCE_MODELS = "~/ii-ollama-models"  # incident-investigator 的 models 目錄：只讀
STATE_DIR = "~/agent-gym-ollama"  # 本 Project：models/（manifest 複本、指向共用 blob 的 symlink、Ollama 的快取）、home/
LOCK_FILE = "/tmp/ai-portfolio-gpu.lock"
PORT = 11436  # WSL loopback；incident-investigator 用 11434
OTHER_PROCESS_NAMES = ("ollama",)
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


def prepare_models(source: str, models: str, model: str) -> dict[str, object]:
    """在 models（本 Project）建立 model 的 manifest 複本，與指向 source（共用、只讀）每個 blob 的 symlink。
    source 的 manifest 必須存在；每個 blob 必須是大小相符的一般檔案；兩個目錄不可相同或互相包含。違反 → ValueError。
    只在 models 之內寫入（錯誤的 symlink 會被換掉）。"""
    if _nested(source, models):
        raise ValueError("this project's models directory must be outside the shared one (and not contain it)")
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
    os.makedirs(os.path.join(models, "blobs"), exist_ok=True)
    for name, target in targets:
        link = os.path.join(models, "blobs", name)
        if os.path.islink(link) and os.readlink(link) == target:
            continue
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(target, link)
    dst = os.path.join(models, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    current = None
    if os.path.isfile(dst) and not os.path.islink(dst):
        with open(dst, "rb") as handle:
            current = handle.read()
    if current != data:
        tmp = dst + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, dst)
    return {"source": SOURCE_MODELS, "blobs": len(targets), "manifest_sha256": hashlib.sha256(data).hexdigest()}


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


def ollama_processes(binary: str, names: tuple[str, ...]) -> list[tuple[int, str]]:
    """執行 binary（命令列中有這個路徑）或程序名稱屬於 names 的程序（不含自己）。"""
    found = []
    for pid, _, _, args in _processes():
        tokens = args.split()
        if pid != os.getpid() and tokens and (binary in tokens or os.path.basename(tokens[0]) in names):
            found.append((pid, args))
    return found


def _args_of(pid: int) -> str | None:
    for row in _processes():
        if row[0] == pid:
            return row[3]
    return None


def start_token(pid: int) -> str | None:
    """程序開始時間（``ps -o lstart=``）的雜湊；程序不存在 → None。"""
    result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    for pid, _, group, _ in _processes():  # 只剩下尚未回收的 zombie 時不算
        if group == pgid and _state(pid) not in ("", "Z"):
            return True
    return False


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


def stop_group(pgid: int, grace: float, reap=None) -> bool:  # type: ignore[no-untyped-def]
    """SIGTERM 整個 process group，grace 秒後仍有程序 → SIGKILL；回傳是否已全部結束。reap：回收自己的子程序。"""

    def gone() -> bool:
        if reap is not None:
            reap()
        return not _group_alive(pgid)

    for signum, timeout in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        if gone():
            return True
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            return True
        if _wait(gone, timeout):
            return True
    return gone()


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
        code = self.proc.poll()
        emit(f"SERVER_EXITED {'' if code is None else code}".rstrip())

    def seen(self, nonce: str, timeout: float) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: nonce in self.nonces, timeout)


def wait_ready(proc: subprocess.Popen[bytes], port: int, timeout: float) -> str | None:
    """``/api/version`` 回應且 server 程序仍在 → 版本字串；server 結束、逾時或要求停止 → None。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None and not STOP.is_set():
        try:
            with opener.open(f"http://127.0.0.1:{port}/api/version", timeout=2) as response:
                body = json.loads(response.read())
            if isinstance(body, dict) and isinstance(body.get("version"), str) and proc.poll() is None:
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


def _stdin_lines() -> "queue.Queue[bytes | None]":
    """在另一個 thread 讀 stdin 的每一行（結束時放入 None），主 thread 才能同時檢查停止的旗標。"""
    lines: "queue.Queue[bytes | None]" = queue.Queue()

    def read() -> None:
        for raw in iter(sys.stdin.buffer.readline, b""):
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    return lines


def _precheck(args: argparse.Namespace, binary: str) -> tuple[int, str, dict[str, int] | None]:
    """啟動前的檢查：(0, "", gpu) 或 (exit code, 協定行, None)。不啟動、不停止任何程序。"""
    names = tuple(name for name in args.process_names.split(",") if name)
    others = ollama_processes(binary, names)
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
    binary = os.path.expanduser(args.binary)
    source = os.path.expanduser(args.source_models)
    state = os.path.expanduser(args.state_dir)
    status, line, gpu = _precheck(args, binary)
    if status:
        emit(line)
        return status
    models = os.path.join(state, "models")
    home = os.path.join(state, "home")
    try:
        before = snapshot(source)
        models_info = prepare_models(source, models, args.model)
        os.makedirs(home, exist_ok=True)
    except (OSError, ValueError) as exc:
        emit(f"FAILED {exc}")
        return EXIT_FAILED
    if STOP.is_set():
        return EXIT_FAILED
    emit(f"SNAPSHOT {snapshot_digest(before)}")
    proc = subprocess.Popen(
        [binary, "serve"], env=server_environment(args.port, models, home), cwd=home, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,),
        preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
    emit(f"STARTING pid={proc.pid} token={start_token(proc.pid) or '-'} helper={os.getpid()} "
         f"helper_token={start_token(os.getpid()) or '-'} port={args.port}")
    reader = LogReader(proc)
    reader.start()
    version = wait_ready(proc, args.port, args.ready_timeout_s)
    if version is None:
        emit(f"FAILED the server did not answer on 127.0.0.1:{args.port} (exit status {proc.poll()})")
        status = EXIT_FAILED
    else:
        emit("READY " + json.dumps({"pid": proc.pid, "port": args.port, "version": version, "gpu": gpu, "env": SERVER_ENV,
                                    "models": models_info}, sort_keys=True))
        lines = _stdin_lines()
        while not STOP.is_set():
            try:
                raw = lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if raw is None:
                break
            words = raw.decode("utf-8", "replace").split()
            if len(words) == 2 and words[0] == "VERIFY" and NONCE.fullmatch(words[1]):
                emit(f"{'VERIFIED' if reader.seen(words[1], VERIFY_TIMEOUT_S) else 'UNVERIFIED'} {words[1]}")
    stopped = stop_group(proc.pid, args.stop_grace_s, reap=proc.poll)
    remaining = ollama_processes(binary, ()) if stopped else []
    if stopped and not remaining and not port_open(args.port):
        emit("STOPPED")
    else:
        detail = f"processes {', '.join(str(pid) for pid, _ in remaining)} remain" if remaining else "the server is still up"
        emit(f"STOP_FAILED {detail}")
        status = status or EXIT_DIRTY
    reader.join(timeout=5)
    changes = diff_snapshots(before, snapshot(source))
    if changes:
        emit(f"MODELS_CHANGED {'; '.join(changes[:10])}")
        status = status or EXIT_DIRTY
    else:
        emit("MODELS_UNCHANGED")
    return status


# ---------------------------------------------------------------- stop 模式（wrapper 的第二條連線）


def _alive(pid: int | None) -> bool:
    """程序存在且不是 zombie。"""
    return pid is not None and _state(pid) not in ("", "Z")


def _is_server(pid: int, token: str, binary: str) -> bool:
    """命令列是「<binary> serve」、開始時間相符、而且自己是 process group leader（由 helper 以新的 session 啟動）。"""
    args = _args_of(pid)
    if args is None or start_token(pid) != token:
        return False
    tokens = args.split()
    index = tokens.index(binary) if binary in tokens else -1
    if index < 0 or tokens[index + 1:index + 2] != ["serve"]:
        return False
    return any(row[0] == pid and row[2] == pid for row in _processes())


def _is_helper(pid: int, token: str) -> bool:
    """命令列是這個 helper（bootstrap 的 agent_gym_serve，或直接執行的 ollama_serve.py），而且開始時間相符。"""
    args = _args_of(pid)
    if args is None or start_token(pid) != token:
        return False
    return any(item == "agent_gym_serve" or os.path.basename(item) == "ollama_serve.py" for item in args.split())


def stop_session(args: argparse.Namespace) -> int:
    """只停止這次 session 的 helper 與 server（身分不符 → 什麼都不停止）。"""
    _handle_signals(signal.SIG_IGN)
    binary = os.path.expanduser(args.binary)
    pid = args.stop_pid
    if _alive(pid) and not _is_server(pid, args.stop_token or "", binary):
        emit(f"STOP_FAILED pid {pid} is not the server this session started; nothing was stopped")
        return EXIT_DIRTY
    helper = args.helper_pid if _alive(args.helper_pid) and _is_helper(args.helper_pid, args.helper_token or "") else None
    if helper is not None:
        os.kill(helper, signal.SIGTERM)  # helper 自己停止 server、比對快照、釋放 lock
        _wait(lambda: not _alive(pid) and not _alive(helper), args.stop_grace_s + 20)
    stopped = stop_group(pid, args.stop_grace_s) if _alive(pid) else True
    if helper is not None and _alive(helper):
        os.kill(helper, signal.SIGKILL)
        _wait(lambda: not _alive(helper), 10)
    remaining = ollama_processes(binary, ())
    status = 0
    if stopped and not remaining and not (helper is not None and _alive(helper)):
        emit("STOPPED")
    else:
        emit(f"STOP_FAILED processes {', '.join(str(item) for item, _ in remaining) or pid} remain")
        status = EXIT_DIRTY
    if args.expect_snapshot:
        if snapshot_digest(snapshot(os.path.expanduser(args.source_models))) == args.expect_snapshot:
            emit("MODELS_UNCHANGED")
        else:
            emit("MODELS_CHANGED the shared models directory differs from the snapshot taken before the server started")
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
elif __name__ == "agent_gym_serve":  # tools/with-gpu-lock 的 bootstrap：python3 -c <bootstrap> <bytes> agent_gym_serve <args>
    sys.exit(main(sys.argv[3:]))
