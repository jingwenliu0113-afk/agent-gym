"""agents/ollama_serve.py 的契約測試（decision 029）：gpu-server 上 agent-gym 自己的 Ollama server。

helper 直接在本機以子程序執行（HOME 換成假的 WSL home、假的 Ollama 與 nvidia-smi，見 tests/remote_fakes.py）。
不連任何遠端、不使用 GPU。
"""

from __future__ import annotations

import ast
import fcntl
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import pytest

from agents import ollama_serve
from agents.ollama import LOCAL_MODEL_CONFIG
from agents.ollama_serve import (
    EXIT_BUSY,
    EXIT_DIRTY,
    EXIT_FAILED,
    MODEL,
    SERVER_ENV,
    diff_snapshots,
    manifest_path,
    prepare_models,
    prepare_state,
    snapshot,
    snapshot_digest,
)
from tests.remote_fakes import HELPER, Remote, alive, free_port, make_remote

# ---------------------------------------------------------------- 模組本身


def test_the_helper_is_stdlib_only_and_runs_on_the_remote_python() -> None:
    source = HELPER.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 10))  # WSL Ubuntu-22.04 的 python3 是 3.10
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported - set(sys.stdlib_module_names)


def test_the_helper_and_the_adapter_share_the_model_and_server_settings() -> None:
    assert MODEL == LOCAL_MODEL_CONFIG["model"]
    assert LOCAL_MODEL_CONFIG["server_env"] == SERVER_ENV
    assert SERVER_ENV["OLLAMA_NOPRUNE"] == "1" and SERVER_ENV["OLLAMA_NO_CLOUD"] == "1"
    assert SERVER_ENV["OLLAMA_NUM_PARALLEL"] == "1" and SERVER_ENV["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in SERVER_ENV.items())
    assert not {"OLLAMA_HOST", "OLLAMA_MODELS", "HOME"} & set(SERVER_ENV)  # 部署細節，不屬於設定


def test_remote_paths_do_not_name_a_user() -> None:
    for value in (ollama_serve.OLLAMA_BIN, ollama_serve.SOURCE_MODELS, ollama_serve.STATE_DIR):
        assert value.startswith("~/")
    assert "/home/" not in HELPER.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 快照


def test_snapshot_detects_every_kind_of_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "models"
    (root / "blobs").mkdir(parents=True)
    small = root / "small.json"
    small.write_text("{}", encoding="utf-8")
    big = root / "blobs" / "big"
    big.write_bytes(b"x" * 100)
    monkeypatch.setattr(ollama_serve, "HASHED_FILE_MAX", 10)  # big 只比大小 / 時間 / inode
    before = snapshot(str(root))
    assert diff_snapshots(before, snapshot(str(root))) == []
    small.read_text(encoding="utf-8")  # 讀取（atime）不算改動
    assert diff_snapshots(before, snapshot(str(root))) == []

    stat = small.stat()
    small.write_text("[]", encoding="utf-8")  # 同大小、內容不同；時間也還原
    os.utime(small, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert any("small.json" in item for item in diff_snapshots(before, snapshot(str(root))))
    small.write_text("{}", encoding="utf-8")
    os.utime(small, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    big.chmod(0o600)
    assert any("blobs/big" in item for item in diff_snapshots(before, snapshot(str(root))))
    (root / "new").write_text("", encoding="utf-8")
    changes = diff_snapshots(before, snapshot(str(root)))
    assert any(item.startswith("added") and "new" in item for item in changes)
    big.unlink()
    assert any(item.startswith("removed") and "blobs/big" in item for item in diff_snapshots(before, snapshot(str(root))))
    assert snapshot_digest(before) != snapshot_digest(snapshot(str(root)))
    assert len(snapshot_digest(before)) == 64


# ---------------------------------------------------------------- 本 Project 的 models 目錄


def test_prepare_models_links_the_shared_blobs_and_copies_the_manifest(tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    models = remote.state / "models"
    before = snapshot(str(remote.source))
    info = prepare_models(str(remote.source), str(models), MODEL)
    assert info["blobs"] == 4
    manifest = json.loads(remote.manifest.read_text(encoding="utf-8"))
    for blob in [manifest["config"], *manifest["layers"]]:
        name = "sha256-" + blob["digest"].split(":")[1]
        link = models / "blobs" / name
        assert link.is_symlink() and os.readlink(link) == str(remote.source / "blobs" / name)
    copy = models / manifest_path(MODEL)
    assert not copy.is_symlink() and copy.read_bytes() == remote.manifest.read_bytes()
    assert diff_snapshots(before, snapshot(str(remote.source))) == []  # 共用目錄沒有任何改動
    prepare_models(str(remote.source), str(models), MODEL)  # 重複準備：不變
    assert diff_snapshots(before, snapshot(str(remote.source))) == []


def test_prepare_models_replaces_a_wrong_link_in_its_own_directory(tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    models = remote.state / "models"
    prepare_models(str(remote.source), str(models), MODEL)
    link = next((models / "blobs").iterdir())
    target = os.readlink(link)
    link.unlink()
    link.symlink_to(tmp_path / "elsewhere")
    prepare_models(str(remote.source), str(models), MODEL)
    assert os.readlink(link) == target


@pytest.mark.parametrize("problem", ["missing-blob", "wrong-size", "linked-blob", "missing-manifest"])
def test_prepare_models_refuses_an_incomplete_shared_directory(problem: str, tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    manifest = json.loads(remote.manifest.read_text(encoding="utf-8"))
    blob = remote.source / "blobs" / ("sha256-" + manifest["layers"][0]["digest"].split(":")[1])
    if problem == "missing-blob":
        blob.unlink()
    elif problem == "wrong-size":
        blob.write_bytes(b"short")
    elif problem == "linked-blob":
        data = blob.read_bytes()
        blob.unlink()
        (tmp_path / "real").write_bytes(data)
        blob.symlink_to(tmp_path / "real")
    else:
        remote.manifest.unlink()
    with pytest.raises(ValueError):
        prepare_models(str(remote.source), str(remote.state / "models"), MODEL)


@pytest.mark.parametrize("layout", ["inside-source", "contains-source", "same"])
def test_prepare_models_never_writes_into_the_shared_directory(layout: str, tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    models = {"inside-source": remote.source / "agent-gym", "contains-source": remote.home, "same": remote.source}[layout]
    before = snapshot(str(remote.source))
    with pytest.raises(ValueError):
        prepare_models(str(remote.source), str(models), MODEL)
    assert diff_snapshots(before, snapshot(str(remote.source))) == []


@pytest.mark.parametrize("model", ["qwen3.5", "../x:y", "a:b/c", "a:", ":b", "a:..", "A b:c"])
def test_model_tags_must_be_plain(model: str) -> None:
    with pytest.raises(ValueError):
        manifest_path(model)


# ---------------------------------------------------------------- 整個 session（helper 以子程序執行）


_SESSIONS: list["Session"] = []


class Session:
    """以子程序執行 helper：stdin / stdout 是協定。stdout 由背景 thread 讀進 queue，所以每個等待都有真正的期限
    （helper 回歸時測試會失敗，而不是卡住）。"""

    def __init__(self, remote: Remote, tmp_path: Path, *args: str, **env: str) -> None:
        self.lock = tmp_path / "gpu.lock"
        self.port = free_port()
        self.proc = subprocess.Popen(
            [sys.executable, str(HELPER), "--lock-file", str(self.lock), "--port", str(self.port), "--process-names=",
             "--stop-grace-s", "2", *args],
            env=remote.env(**env), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines: list[str] = []
        self.queue: "queue.Queue[str | None]" = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        _SESSIONS.append(self)

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.queue.put(line.rstrip("\n"))
        self.queue.put(None)

    def _next(self, deadline: float, waiting_for: object) -> str | None:
        try:
            return self.queue.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise AssertionError(f"no {waiting_for} before the deadline: {self.lines}") from None

    def until(self, *prefixes: str, timeout: float = 20) -> str:
        deadline = time.monotonic() + timeout
        while True:
            line = self._next(deadline, prefixes)
            if line is None:
                raise AssertionError(f"the helper ended: {self.lines}")
            self.lines.append(line)
            if line.startswith(prefixes):
                return line

    def rest(self, timeout: float = 60) -> None:
        """讀到 helper 的輸出結束。"""
        deadline = time.monotonic() + timeout
        while (line := self._next(deadline, "the end of the output")) is not None:
            self.lines.append(line)

    def send(self, text: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def finish(self, timeout: float = 60) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.rest(timeout)
        return self.proc.wait(timeout=timeout)

    def protocol(self) -> list[str]:
        return [line for line in self.lines if not line.startswith("LOG ")]

    def starting(self) -> dict[str, str]:
        line = next(line for line in self.lines if line.startswith("STARTING "))
        return dict(item.split("=", 1) for item in line.split()[1:])


def _get(port: int, path: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        data: dict[str, Any] = json.loads(response.read())
        return data


def _lock_is_free(lock: Path) -> bool:
    fd = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


@pytest.fixture
def remote(tmp_path: Path) -> Iterator[Remote]:
    fake = make_remote(tmp_path)
    yield fake
    for session in _SESSIONS:  # 失敗的測試也不遺留 helper 或假 server
        if session.proc.poll() is None:
            session.proc.kill()
            session.proc.wait()
    _SESSIONS.clear()
    fake.kill_all()


def test_a_session_starts_verifies_and_stops_its_own_server(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, "--source-models", str(remote.source))
    assert session.until("LOCKED", "BUSY", "FAILED") == "LOCKED"
    ready = session.until("READY", "BUSY", "FAILED")
    assert ready.startswith("READY "), session.lines
    info = json.loads(ready[len("READY "):])
    assert info["version"] == "0.35.0" and info["port"] == session.port and info["env"] == SERVER_ENV
    assert info["gpu"] == {"used_mib": 675, "total_mib": 16303} and info["models"]["blobs"] == 4
    assert not _lock_is_free(session.lock)  # 整個 session 持有 GPU lock

    [server] = remote.servers()
    assert server["pid"] == info["pid"] and server["pgid"] == info["pid"]  # 自己的 process group
    env = server["env"]
    assert isinstance(env, dict)
    assert {k: env[k] for k in SERVER_ENV} == SERVER_ENV
    assert env["OLLAMA_HOST"] == f"127.0.0.1:{session.port}"
    assert env["OLLAMA_MODELS"] == str(remote.state / "models") and env["HOME"] == str(remote.state / "home")

    nonce = "0123456789abcdef" * 2
    assert _get(session.port, f"/api/version?agent_gym_session={nonce}") == {"version": "0.35.0"}
    session.send(f"VERIFY {nonce}\n")
    assert session.until("VERIFIED", "UNVERIFIED") == f"VERIFIED {nonce}"
    other = "f" * 32  # 沒有送過的請求
    session.send(f"VERIFY {other}\n")
    assert session.until("VERIFIED", "UNVERIFIED", timeout=30) == f"UNVERIFIED {other}"

    assert session.finish() == 0
    protocol = session.protocol()
    assert "STOPPED" in protocol and "MODELS_UNCHANGED" in protocol, protocol
    assert any(line.startswith("STARTING pid=") for line in protocol)
    assert any(line.startswith("SNAPSHOT ") for line in protocol)
    assert not alive(int(info["pid"])) and _lock_is_free(session.lock)
    assert any("agent_gym_session=" + nonce in line for line in session.lines if line.startswith("LOG "))


def test_the_callers_ollama_settings_do_not_leak_into_the_server(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, OLLAMA_FLASH_ATTENTION="1", OLLAMA_HOST="0.0.0.0:1", OLLAMA_MODELS="/elsewhere")
    assert session.until("READY", "BUSY", "FAILED").startswith("READY")
    assert session.finish() == 0
    env = remote.servers()[0]["env"]
    assert isinstance(env, dict) and "OLLAMA_FLASH_ATTENTION" not in env and env["OLLAMA_HOST"].startswith("127.0.0.1:")


def test_the_server_writes_only_into_this_projects_directory(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="write_models")
    assert session.until("READY", "BUSY", "FAILED").startswith("READY")
    assert session.finish() == 0 and "MODELS_UNCHANGED" in session.protocol()
    assert (remote.state / "models" / "metadata" / "cache.json").exists()


def test_a_change_to_the_shared_directory_is_reported(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="touch_source")
    assert session.until("READY", "BUSY", "FAILED").startswith("READY")
    assert session.finish() == EXIT_DIRTY
    changed = [line for line in session.protocol() if line.startswith("MODELS_CHANGED")]
    assert changed and "metadata" in changed[0] and "STOPPED" in session.protocol()


def test_a_busy_lock_starts_nothing(remote: Remote, tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    fd = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        session = Session(remote, tmp_path)
        assert session.until("LOCKED", "BUSY", "FAILED").startswith("BUSY")
        assert session.finish() == EXIT_BUSY
    finally:
        os.close(fd)
    assert remote.servers() == [] and not (remote.state / "models").exists()


def test_a_running_ollama_process_starts_nothing_and_is_left_alone(remote: Remote, tmp_path: Path) -> None:
    other = subprocess.Popen([str(remote.binary), "runner"])
    try:
        session = Session(remote, tmp_path)
        assert session.until("LOCKED") == "LOCKED"
        busy = session.until("READY", "BUSY", "FAILED")
        assert busy.startswith("BUSY") and str(other.pid) in busy
        assert session.finish() == EXIT_BUSY and remote.servers() == []
        assert alive(other.pid)
    finally:
        other.kill()
        other.wait()


def test_an_ollama_process_name_starts_nothing(remote: Remote, tmp_path: Path) -> None:
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        session = Session(remote, tmp_path, "--process-names", "sleep")
        assert session.until("READY", "BUSY", "FAILED").startswith("BUSY")
        assert session.finish() == EXIT_BUSY and alive(sleeper.pid)
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_taken_port_starts_nothing(remote: Remote, tmp_path: Path) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        port = listener.getsockname()[1]
        session = Session(remote, tmp_path, "--port", str(port))
        busy = session.until("READY", "BUSY", "FAILED")
        assert busy.startswith("BUSY") and str(port) in busy
        assert session.finish() == EXIT_BUSY and remote.servers() == []
    finally:
        listener.close()


def test_a_busy_gpu_starts_nothing(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_GPU_USED="9000")
    busy = session.until("READY", "BUSY", "FAILED")
    assert busy.startswith("BUSY") and "9000" in busy
    assert session.finish() == EXIT_BUSY and remote.servers() == []


def test_a_failing_gpu_check_starts_nothing(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_GPU_FAIL="1")
    assert session.until("READY", "BUSY", "FAILED").startswith("FAILED")
    assert session.finish() == EXIT_FAILED and remote.servers() == []


def test_an_incomplete_shared_directory_starts_nothing(remote: Remote, tmp_path: Path) -> None:
    remote.manifest.unlink()
    session = Session(remote, tmp_path)
    assert session.until("READY", "BUSY", "FAILED").startswith("FAILED")
    assert session.finish() == EXIT_FAILED and remote.servers() == []


def test_a_server_that_dies_at_start_is_a_failure(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="crash")
    failed = session.until("READY", "BUSY", "FAILED")
    assert failed.startswith("FAILED")
    assert session.finish() == EXIT_FAILED
    assert "STOPPED" in session.protocol() and "MODELS_UNCHANGED" in session.protocol()
    assert any("fake crash" in line for line in session.lines)


def test_a_server_that_ignores_sigterm_is_killed(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="ignore_term")
    ready = session.until("READY", "BUSY", "FAILED")
    pid = json.loads(ready[len("READY "):])["pid"]
    assert session.finish() == 0 and "STOPPED" in session.protocol()
    assert not alive(pid)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_a_signal_stops_the_server(signum: int, remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path)
    ready = session.until("READY", "BUSY", "FAILED")
    pid = json.loads(ready[len("READY "):])["pid"]
    session.proc.send_signal(signum)  # stdin 仍開著（讀 stdin 的 thread 還在等）
    session.rest()
    assert session.proc.wait(timeout=30) == 0  # 不是 Python 結束時的 abort（decision 031 §1.5）
    assert "STOPPED" in session.protocol() and "MODELS_UNCHANGED" in session.protocol()
    assert not alive(pid) and _lock_is_free(session.lock)


def _stop_mode(remote: Remote, session: Session, *extra: str, **fields: str) -> subprocess.CompletedProcess[str]:
    values = {**session.starting(), **fields}
    return subprocess.run(
        [sys.executable, str(HELPER), "--stop-pid", values["pid"], "--stop-token", values["token"], "--helper-pid",
         values["helper"], "--helper-token", values["helper_token"], "--port", str(session.port), "--stop-grace-s", "2",
         "--lock-file", str(session.lock), *extra],
        env=remote.env(), capture_output=True, text=True, timeout=120)


def test_the_server_keeps_the_lock_if_the_helper_is_killed_and_the_stop_mode_ends_it(remote: Remote, tmp_path: Path) -> None:
    # ignore_term：Linux 上 PDEATHSIG 送來的 SIGTERM 也結束不了它，兩個平台都走到 stop 模式
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="ignore_term")
    session.until("READY", "BUSY", "FAILED")
    fields = session.starting()
    snapshot_line = next(line for line in session.lines if line.startswith("SNAPSHOT "))
    pid = int(fields["pid"])
    session.proc.kill()  # helper 突然消失（SIGKILL）
    session.proc.wait()
    time.sleep(0.5)
    assert alive(pid) and not _lock_is_free(session.lock)  # server 繼承 lock 的 fd：它還在，lock 就還被持有

    wrong = _stop_mode(remote, session, token="0" * 16)
    assert wrong.returncode == EXIT_DIRTY and "STOP_FAILED" in wrong.stdout and alive(pid)  # 身分不符：什麼都不停止

    stop = _stop_mode(remote, session, "--expect-snapshot", snapshot_line.split()[1])
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert "STOPPED" in stop.stdout.splitlines() and "MODELS_UNCHANGED" in stop.stdout.splitlines()
    assert not alive(pid) and _lock_is_free(session.lock)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_on_linux_the_server_ends_with_its_helper(remote: Remote, tmp_path: Path) -> None:  # pragma: no cover - Linux
    session = Session(remote, tmp_path)
    ready = session.until("READY", "BUSY", "FAILED")
    pid = json.loads(ready[len("READY "):])["pid"]
    session.proc.kill()
    session.proc.wait()
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(pid) and _lock_is_free(session.lock)


def test_the_stop_mode_reports_a_change_the_session_saw(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="touch_source")  # server 執行期間共用目錄改變
    session.until("READY", "BUSY", "FAILED")
    stop = _stop_mode(remote, session, "--expect-snapshot", next(
        line for line in session.lines if line.startswith("SNAPSHOT ")).split()[1])
    assert stop.returncode == EXIT_DIRTY and "MODELS_CHANGED" in stop.stdout, stop.stdout
    assert session.proc.wait(timeout=30) == EXIT_DIRTY  # helper 收到 SIGTERM 後自己停止 server、比對、結束
    assert not alive(int(session.starting()["pid"]))


@pytest.mark.parametrize("touch_source", [False, True], ids=["unchanged", "changed-while-held"])
def test_the_stop_mode_compares_right_after_ending_the_lock_holder(touch_source: bool, remote: Remote, tmp_path: Path) -> None:
    """helper 已消失、只剩 server 持有 lock：stop 模式結束它之後立刻取得 lock 比較（lock 一直屬於這個 session）。"""
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="ignore_term")
    session.until("READY", "BUSY", "FAILED")
    snapshot_line = next(line for line in session.lines if line.startswith("SNAPSHOT "))
    session.proc.kill()
    session.proc.wait()
    if touch_source:
        remote.metadata.write_text('{"changed": true}', encoding="utf-8")
    stop = _stop_mode(remote, session, "--expect-snapshot", snapshot_line.split()[1])
    lines = stop.stdout.splitlines()
    assert "STOPPED" in lines
    if touch_source:
        assert stop.returncode == EXIT_DIRTY and any(line.startswith("MODELS_CHANGED") for line in lines)
    else:
        assert stop.returncode == 0 and "MODELS_UNCHANGED" in lines
    assert _lock_is_free(session.lock)


# ---------------------------------------------------------------- decision 031：第 029 輪審查的回歸測試


def test_the_server_runs_under_this_projects_launcher(remote: Remote, tmp_path: Path) -> None:
    """命令列不是「<binary> serve」：incident-investigator 以 `pkill -x -f '<binary> serve'` 停止自己的 server，不會比對到本 Project 的。"""
    session = Session(remote, tmp_path)
    session.until("READY", "BUSY", "FAILED")
    [server] = remote.servers()
    assert server["argv0"] == str(remote.launcher)
    assert remote.launcher.is_symlink() and os.readlink(remote.launcher) == str(remote.binary)
    args = subprocess.run(["ps", "-ww", "-o", "args=", "-p", str(server["pid"])], capture_output=True, text=True).stdout
    assert str(remote.binary) not in args.split() and str(remote.launcher) in args.split()
    assert session.finish() == 0


def test_the_lock_is_held_until_the_server_is_gone(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, "--stop-grace-s", "3", FAKE_OLLAMA_MODE="ignore_term")
    ready = session.until("READY", "BUSY", "FAILED")
    pid = json.loads(ready[len("READY "):])["pid"]
    assert session.proc.stdin is not None
    session.proc.stdin.close()  # 開始停止：TERM 被忽略，3 秒後才 KILL
    checks = 0
    while alive(pid):
        assert not _lock_is_free(session.lock), "the lock was released while the server still ran"
        checks += 1
        time.sleep(0.1)
    assert checks >= 10  # 確實在寬限期內檢查過
    session.rest()
    assert session.proc.wait(timeout=30) == 0 and _lock_is_free(session.lock)


def test_a_process_of_this_session_that_survives_the_stop_is_reported(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="detached_runner")
    session.until("READY", "BUSY", "FAILED")
    deadline = time.monotonic() + 10
    while not remote.runners() and time.monotonic() < deadline:
        time.sleep(0.05)
    [runner] = remote.runners()
    assert session.finish() == EXIT_DIRTY
    failed = [line for line in session.protocol() if line.startswith("STOP_FAILED")]
    assert failed and str(runner) in failed[0] and "STOPPED" not in session.protocol()
    assert alive(runner)  # 不在 process group 內的程序不會被猜測後停止；只回報


def test_other_projects_processes_are_never_stopped(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path)
    session.until("READY", "BUSY", "FAILED")
    other = subprocess.Popen([str(remote.binary), "runner"], start_new_session=True)  # 同一個執行檔，不屬於這次 session
    try:
        assert session.finish() == 0 and "STOPPED" in session.protocol()
        assert alive(other.pid)
    finally:
        other.kill()
        other.wait()


def test_the_stop_mode_never_signals_a_process_that_is_not_the_helper(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="ignore_term")
    session.until("READY", "BUSY", "FAILED")
    session.proc.kill()
    session.proc.wait()
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        token = ollama_serve.start_token(sleeper.pid)
        assert token is not None
        stop = _stop_mode(remote, session, helper=str(sleeper.pid), helper_token=token)  # 身分 token 正確，但不是 helper
        assert stop.returncode == 0 and "STOPPED" in stop.stdout.splitlines(), stop.stdout + stop.stderr
        assert alive(sleeper.pid)
    finally:
        sleeper.kill()
        sleeper.wait()


@pytest.mark.parametrize("other_holds_lock", [False, True], ids=["lock-free", "lock-held"])
def test_the_stop_mode_does_not_blame_another_workload(other_holds_lock: bool, remote: Remote, tmp_path: Path) -> None:
    """session 已正常結束（helper 在持有 lock 時比較過、記下結論），之後別的 Project 以同一個執行檔執行並寫入共用目錄：
    第二條連線報告 session 自己的結論，不把之後的改動算在本 session 上。"""
    session = Session(remote, tmp_path)
    session.until("READY", "BUSY", "FAILED")
    snapshot_line = next(line for line in session.lines if line.startswith("SNAPSHOT "))
    assert session.finish() == 0
    fd = os.open(session.lock, os.O_RDONLY | os.O_CREAT, 0o666)
    if other_holds_lock:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    other = subprocess.Popen([str(remote.binary), "runner"], start_new_session=True)
    try:
        remote.metadata.write_text('{"ollama_version": "0.35.0", "other": 1}', encoding="utf-8")
        stop = _stop_mode(remote, session, "--expect-snapshot", snapshot_line.split()[1])
        lines = stop.stdout.splitlines()
        assert stop.returncode == 0 and "STOPPED" in lines and "MODELS_UNCHANGED" in lines, stop.stdout
        assert alive(other.pid)
    finally:
        other.kill()
        other.wait()
        os.close(fd)


def test_without_the_sessions_own_verdict_nothing_is_claimed(remote: Remote, tmp_path: Path) -> None:
    """helper 與 server 都已結束、也沒有它們的結論（lock 在不知道的時間點釋放）：無法比較，不猜。"""
    session = Session(remote, tmp_path)
    session.until("READY", "BUSY", "FAILED")
    snapshot_line = next(line for line in session.lines if line.startswith("SNAPSHOT "))
    assert session.finish() == 0
    for verdict in (remote.state / "sessions").iterdir():
        verdict.unlink()
    stop = _stop_mode(remote, session, "--expect-snapshot", snapshot_line.split()[1])
    lines = stop.stdout.splitlines()
    assert stop.returncode == EXIT_DIRTY and "STOPPED" in lines and any(line.startswith("MODELS_NOT_COMPARED") for line in lines)


def test_the_helper_records_its_verdict_for_the_second_connection(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path, FAKE_OLLAMA_MODE="touch_source")
    session.until("READY", "BUSY", "FAILED")
    assert session.finish() == EXIT_DIRTY
    [verdict] = (remote.state / "sessions").iterdir()
    assert verdict.name == session.starting()["helper_token"] + ".json"
    record = json.loads(verdict.read_text(encoding="utf-8"))
    assert record["stopped"] == "STOPPED" and record["models"].startswith("MODELS_CHANGED")


def test_the_server_is_not_reaped_before_its_group_is_stopped() -> None:
    """結束了但尚未回收的 server 讓 pid / pgid 保持被占用：停止 group 的期間不會被別的程序重複使用（decision 031 第二輪）。"""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    deadline = time.monotonic() + 10
    while ollama_serve.exit_status(proc) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ollama_serve.exit_status(proc) == (0 if hasattr(os, "waitid") else -1)  # 沒有 waitid：已結束、結束碼未知
    assert os.waitpid(proc.pid, os.WNOHANG)[0] == proc.pid  # 還沒被回收：現在才回收


def test_stop_group_signals_only_while_the_group_is_still_ours() -> None:
    sleeper = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        assert ollama_serve.stop_group(sleeper.pid, 1, still_ours=lambda: False) is False
        assert alive(sleeper.pid)  # 不再確認是自己的 → 不送信號
        assert ollama_serve.stop_group(sleeper.pid, 1, reap=sleeper.poll, still_ours=lambda: True) is True
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()


def test_signalling_a_process_that_already_exited_is_not_an_error() -> None:
    gone = subprocess.Popen(["true"])
    gone.wait()
    assert ollama_serve.signal_process(gone.pid, signal.SIGTERM, lambda: True) is False
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        assert ollama_serve.signal_process(sleeper.pid, signal.SIGTERM, lambda: False) is False  # 身分已不符：不送
        assert alive(sleeper.pid)
        assert ollama_serve.signal_process(sleeper.pid, signal.SIGTERM, lambda: True) is True
        sleeper.wait(timeout=10)
    finally:
        if sleeper.poll() is None:
            sleeper.kill()


def test_start_tokens_on_linux_come_from_procfs_not_the_wall_clock(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "sys" / "kernel" / "random").mkdir(parents=True)
    (proc_root / "sys" / "kernel" / "random" / "boot_id").write_text("1b2c3d\n", encoding="utf-8")
    (proc_root / "42").mkdir()
    fields = ["S"] + [str(n) for n in range(4, 22)] + ["987654"] + ["0"] * 30  # 第 22 欄 = starttime
    (proc_root / "42" / "stat").write_text("42 (a) b) " + " ".join(fields) + "\n", encoding="utf-8")
    token = ollama_serve.start_token(42, proc_root=str(proc_root))
    assert token is not None and len(token) == 16
    (proc_root / "42" / "stat").write_text("42 (a) b) " + " ".join(fields).replace("987654", "987655") + "\n", encoding="utf-8")
    assert ollama_serve.start_token(42, proc_root=str(proc_root)) != token
    assert ollama_serve.start_token(43, proc_root=str(proc_root)) is None  # 不存在的程序


def test_stdin_closing_before_the_start_means_no_server(remote: Remote, tmp_path: Path) -> None:
    """wrapper 在 server 啟動前就結束（中斷或逾時）：helper 不啟動 server。"""
    session = Session(remote, tmp_path, FAKE_GPU_SLEEP="2")
    assert session.until("LOCKED", "BUSY", "FAILED") == "LOCKED"
    assert session.finish() == EXIT_FAILED
    assert any(line.startswith("NOT_STARTED") for line in session.protocol()) and remote.servers() == []
    assert _lock_is_free(session.lock)


@pytest.mark.parametrize("layout", ["blobs", "manifest-parent", "models", "state", "metadata"])
def test_a_symlinked_directory_into_the_shared_models_is_refused(layout: str, tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    before = snapshot(str(remote.source))
    models = remote.state / "models"
    if layout == "metadata":  # Ollama 執行時寫快取的目錄
        models.mkdir(parents=True)
        (models / "metadata").symlink_to(remote.source / "metadata")
    elif layout == "blobs":
        models.mkdir(parents=True)
        (models / "blobs").symlink_to(remote.source / "blobs")
    elif layout == "manifest-parent":
        models.mkdir(parents=True)
        (models / "manifests").symlink_to(remote.source / "manifests")
    elif layout == "models":
        remote.state.mkdir()
        models.symlink_to(remote.source)
    else:
        remote.state.symlink_to(remote.source / "metadata")
    with pytest.raises(ValueError):
        prepare_state(str(remote.source), str(remote.state), MODEL, str(remote.binary))
    assert diff_snapshots(before, snapshot(str(remote.source))) == []


def test_a_regular_file_in_this_projects_blobs_is_refused_not_deleted(tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    models = remote.state / "models"
    prepare_models(str(remote.source), str(models), MODEL)
    link = next((models / "blobs").iterdir())
    target = Path(os.readlink(link))
    link.unlink()
    os.link(target, link)  # hard link 到共用的 blob（029 不採用的做法）
    before = snapshot(str(remote.source))
    with pytest.raises(ValueError):
        prepare_models(str(remote.source), str(models), MODEL)
    assert link.exists() and not link.is_symlink() and diff_snapshots(before, snapshot(str(remote.source))) == []


def test_the_manifest_copy_never_writes_through_a_planted_symlink(tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    models = remote.state / "models"
    copy_path = models / manifest_path(MODEL)
    copy_path.parent.mkdir(parents=True)
    for name in (copy_path.name + ".tmp", copy_path.name):
        (copy_path.parent / name).symlink_to(remote.metadata)  # 指向共用目錄的檔案
    before = snapshot(str(remote.source))
    prepare_models(str(remote.source), str(models), MODEL)
    assert diff_snapshots(before, snapshot(str(remote.source))) == []
    assert not copy_path.is_symlink() and copy_path.read_bytes() == remote.manifest.read_bytes()


def test_prepare_state_builds_models_home_and_launcher(tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    before = snapshot(str(remote.source))
    info = prepare_state(str(remote.source), str(remote.state), MODEL, str(remote.binary))
    assert info["launcher"] == str(remote.launcher) and os.readlink(remote.launcher) == str(remote.binary)
    assert (remote.state / "home").is_dir() and not (remote.state / "home").is_symlink()
    assert info["models"]["blobs"] == 4  # type: ignore[index]
    remote.launcher.unlink()
    remote.launcher.write_text("", encoding="utf-8")  # 一般檔案：拒絕，不刪除
    with pytest.raises(ValueError):
        prepare_state(str(remote.source), str(remote.state), MODEL, str(remote.binary))
    assert diff_snapshots(before, snapshot(str(remote.source))) == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_a_linked_entry_in_this_projects_metadata_is_refused(kind: str, tmp_path: Path) -> None:
    remote = make_remote(tmp_path)
    metadata = remote.state / "models" / "metadata"
    metadata.mkdir(parents=True)
    entry = metadata / remote.metadata.name
    if kind == "symlink":
        entry.symlink_to(remote.metadata)
    else:
        os.link(remote.metadata, entry)
    before = snapshot(str(remote.source)) if kind == "symlink" else None
    with pytest.raises(ValueError):
        prepare_state(str(remote.source), str(remote.state), MODEL, str(remote.binary))
    assert os.path.lexists(entry)  # 不刪除
    if before is not None:
        assert diff_snapshots(before, snapshot(str(remote.source))) == []

