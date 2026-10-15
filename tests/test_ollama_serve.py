"""agents/ollama_serve.py 的契約測試（decision 029）：gpu-server 上 agent-gym 自己的 Ollama server。

helper 直接在本機以子程序執行（HOME 換成假的 WSL home、假的 Ollama 與 nvidia-smi，見 tests/remote_fakes.py）。
不連任何遠端、不使用 GPU。
"""

from __future__ import annotations

import ast
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import IO, Any

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


class Session:
    """以子程序執行 helper：stdin / stdout 是協定。"""

    def __init__(self, remote: Remote, tmp_path: Path, *args: str, **env: str) -> None:
        self.lock = tmp_path / "gpu.lock"
        self.port = free_port()
        self.proc = subprocess.Popen(
            [sys.executable, str(HELPER), "--lock-file", str(self.lock), "--port", str(self.port), "--process-names=",
             "--stop-grace-s", "2", *args],
            env=remote.env(**env), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines: list[str] = []

    @property
    def stdout(self) -> IO[str]:
        assert self.proc.stdout is not None
        return self.proc.stdout

    def until(self, *prefixes: str, timeout: float = 20) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.stdout.readline()
            if not line:
                raise AssertionError(f"the helper ended: {self.lines}")
            line = line.rstrip("\n")
            self.lines.append(line)
            if line.startswith(prefixes):
                return line
        raise AssertionError(f"no {prefixes} in {self.lines}")

    def send(self, text: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def finish(self, timeout: float = 30) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.lines.extend(line.rstrip("\n") for line in self.stdout.read().splitlines())
        return self.proc.wait(timeout=timeout)

    def protocol(self) -> list[str]:
        return [line for line in self.lines if not line.startswith("LOG ")]


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
def remote(tmp_path: Path) -> Remote:
    return make_remote(tmp_path)


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
    session.proc.send_signal(signum)
    session.lines.extend(line.rstrip("\n") for line in session.stdout.read().splitlines())
    session.proc.wait(timeout=30)
    assert "STOPPED" in session.protocol() and "MODELS_UNCHANGED" in session.protocol()
    assert not alive(pid) and _lock_is_free(session.lock)


def test_the_server_keeps_the_lock_if_the_helper_is_killed_and_the_stop_mode_ends_it(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path)
    session.until("READY", "BUSY", "FAILED")
    starting = next(line for line in session.lines if line.startswith("STARTING "))
    fields = dict(item.split("=", 1) for item in starting.split()[1:])
    snapshot_line = next(line for line in session.lines if line.startswith("SNAPSHOT "))
    pid = int(fields["pid"])
    session.proc.kill()  # helper 突然消失（SIGKILL）：Linux 上 server 會收到 SIGTERM（PDEATHSIG）；其他平台 server 留著並持有 lock
    session.proc.wait()
    if sys.platform.startswith("linux"):
        pytest.skip("PDEATHSIG ends the server on Linux")
    assert alive(pid) and not _lock_is_free(session.lock)

    wrong = subprocess.run(
        [sys.executable, str(HELPER), "--stop-pid", str(pid), "--stop-token", "0" * 16, "--helper-pid", fields["helper"],
         "--helper-token", fields["helper_token"], "--port", str(session.port), "--stop-grace-s", "2"],
        env=remote.env(), capture_output=True, text=True, timeout=60)
    assert wrong.returncode == EXIT_DIRTY and "STOP_FAILED" in wrong.stdout and alive(pid)  # 身分不符：什麼都不停止

    stop = subprocess.run(
        [sys.executable, str(HELPER), "--stop-pid", str(pid), "--stop-token", fields["token"], "--helper-pid", fields["helper"],
         "--helper-token", fields["helper_token"], "--port", str(session.port), "--stop-grace-s", "2",
         "--expect-snapshot", snapshot_line.split()[1]],
        env=remote.env(), capture_output=True, text=True, timeout=60)
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert "STOPPED" in stop.stdout.splitlines() and "MODELS_UNCHANGED" in stop.stdout.splitlines()
    assert not alive(pid) and _lock_is_free(session.lock)


def test_the_stop_mode_reports_a_changed_shared_directory(remote: Remote, tmp_path: Path) -> None:
    session = Session(remote, tmp_path)
    session.until("READY", "BUSY", "FAILED")
    fields = dict(item.split("=", 1) for item in next(line for line in session.lines if line.startswith("STARTING ")).split()[1:])
    stop = subprocess.run(
        [sys.executable, str(HELPER), "--stop-pid", fields["pid"], "--stop-token", fields["token"], "--helper-pid",
         fields["helper"], "--helper-token", fields["helper_token"], "--port", str(session.port), "--stop-grace-s", "2",
         "--expect-snapshot", "0" * 64],
        env=remote.env(), capture_output=True, text=True, timeout=60)
    assert stop.returncode == EXIT_DIRTY and "MODELS_CHANGED" in stop.stdout
    session.proc.wait(timeout=30)  # helper 收到 SIGTERM 後自己停止 server 並結束
    assert not alive(int(fields["pid"]))
