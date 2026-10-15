"""tools/with-gpu-lock 的契約測試（decision 026 §6、028 §4、029）。

PATH 最前面的假 ``ssh`` 在本機轉送 -L 的 port，並把遠端命令（``wsl.exe ... bash -lc "<bootstrap>"``）交給本機的 bash：
真正的 helper（agents/ollama_serve.py，經 stdin 傳過去）以假的 WSL home 與假的 Ollama 執行（tests/remote_fakes.py）。
不連任何遠端、不使用 GPU。
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agents.ollama_serve import EXIT_BUSY, EXIT_DIRTY, EXIT_FAILED, SERVER_ENV, diff_snapshots, snapshot
from tests.remote_fakes import WRAPPER, Remote, alive, free_port, make_remote

PRINT_ENV = [sys.executable, "-c", "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
             "('AI_PORTFOLIO_GPU_LOCK', 'AGENT_GYM_OLLAMA_URL', 'AGENT_GYM_OLLAMA_SERVER')}))"]


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.remote: Remote = make_remote(tmp_path)
        self.lock = tmp_path / "gpu.lock"
        self.local_port = free_port()
        self.remote_port = free_port()
        self.ssh_log = tmp_path / "ssh.log"
        self.env = self.remote.env(FAKE_SSH_LOG=str(self.ssh_log), FAKE_SSH_EXTRA_ARGS="--process-names= --stop-grace-s 2",
                                   WITH_GPU_LOCK_TIMEOUT_S="10", WITH_GPU_LOCK_STOP_TIMEOUT_S="30")
        self.env["HOME"] = os.environ.get("HOME", str(tmp_path))  # Mac 端的 HOME 不變；假的 ssh 把遠端的 HOME 換掉

    def args(self, *extra: str) -> list[str]:
        return [str(WRAPPER), "--lock-file", str(self.lock), "--local-port", str(self.local_port), "--remote-port",
                str(self.remote_port), *extra]

    def run(self, *command: str, options: tuple[str, ...] = (), **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([*self.args(*options), "--", *command], env={**self.env, **env}, capture_output=True, text=True,
                              timeout=120)

    def ssh_calls(self) -> list[list[str]]:
        if not self.ssh_log.exists():
            return []
        return [line.split("\x1f") for line in self.ssh_log.read_text(encoding="utf-8").splitlines()]

    def lock_is_free(self) -> bool:
        fd = os.open(self.lock, os.O_RDONLY | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            os.close(fd)
        return True


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def _servers_stopped(setup: Setup) -> bool:
    return all(not alive(int(server["pid"])) for server in setup.remote.servers())  # type: ignore[arg-type]


def test_the_wrapper_is_executable() -> None:
    assert WRAPPER.exists() and os.access(WRAPPER, os.X_OK)


def test_the_command_runs_against_this_sessions_own_server(setup: Setup) -> None:
    source_before = snapshot(str(setup.remote.source))
    result = setup.run(*PRINT_ENV)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env["AI_PORTFOLIO_GPU_LOCK"] == "held" and env["AGENT_GYM_OLLAMA_URL"] == f"http://127.0.0.1:{setup.local_port}"
    server = json.loads(env["AGENT_GYM_OLLAMA_SERVER"])
    assert server["env"] == SERVER_ENV and server["version"] == "0.35.0" and server["port"] == setup.remote_port
    [record] = setup.remote.servers()
    assert record["pid"] == server["pid"] and not alive(int(record["pid"]))  # type: ignore[arg-type]
    assert setup.lock_is_free()
    assert diff_snapshots(source_before, snapshot(str(setup.remote.source))) == []
    assert "stopped" in result.stderr.lower() and "unchanged" in result.stderr.lower()

    [call] = setup.ssh_calls()
    assert "-L" in call and f"127.0.0.1:{setup.local_port}:127.0.0.1:{setup.remote_port}" in call
    assert "ExitOnForwardFailure=yes" in call and "BatchMode=yes" in call and call[-2] == "gpu-server"
    remote = call[-1]
    assert remote.startswith('wsl.exe -d Ubuntu-22.04 -- bash -lc "python3 -c ') and remote.endswith('"')
    assert remote.count('"') == 2 and not set("%^!$`\\") & set(remote)  # cmd.exe / 外層 shell 不會改寫
    assert f"--lock-file {setup.lock}" in remote and f"--port {setup.remote_port}" in remote


def test_the_lock_is_held_while_the_command_runs(setup: Setup) -> None:
    check = (f"import fcntl, os; fd = os.open({str(setup.lock)!r}, os.O_RDONLY)\n"
             "try:\n    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); print('free')\nexcept BlockingIOError:\n    print('held')")
    result = setup.run(sys.executable, "-c", check)
    assert result.returncode == 0 and result.stdout.strip() == "held"


def test_the_exit_code_of_the_command_is_returned(setup: Setup) -> None:
    result = setup.run(sys.executable, "-c", "raise SystemExit(7)")
    assert result.returncode == 7 and _servers_stopped(setup) and setup.lock_is_free()


def test_the_server_log_can_be_kept(setup: Setup) -> None:
    log = setup.tmp_path / "server.log"
    result = setup.run("true", options=("--server-log", str(log)))
    assert result.returncode == 0, result.stderr
    text = log.read_text(encoding="utf-8")
    assert "[GIN]" in text and "agent_gym_session=" in text and not any(line.startswith("LOG ") for line in text.splitlines())


def _not_run(setup: Setup, *, mode: str = "", options: tuple[str, ...] = (), **env: str) -> subprocess.CompletedProcess[str]:
    marker = setup.tmp_path / "ran"
    result = setup.run(sys.executable, "-c", f"open({str(marker)!r}, 'w')", options=options,
                       **({"FAKE_SSH_MODE": mode} if mode else {}), **env)
    assert not marker.exists(), result.stderr
    return result


def test_a_busy_lock_is_not_taken(setup: Setup) -> None:
    fd = os.open(setup.lock, os.O_RDONLY | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = _not_run(setup)
    finally:
        os.close(fd)
    assert result.returncode == EXIT_BUSY and "busy" in result.stderr.lower() and setup.remote.servers() == []


def test_a_running_ollama_process_is_left_alone(setup: Setup) -> None:
    other = subprocess.Popen([str(setup.remote.binary), "runner"])
    try:
        result = _not_run(setup)
        assert result.returncode == EXIT_BUSY and str(other.pid) in result.stderr and alive(other.pid)
    finally:
        other.kill()
        other.wait()


def test_a_busy_gpu_is_not_used(setup: Setup) -> None:
    result = _not_run(setup, FAKE_GPU_USED="9000")
    assert result.returncode == EXIT_BUSY and setup.remote.servers() == []


def test_a_taken_remote_port_is_not_used(setup: Setup) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", setup.remote_port))
        listener.listen(1)
        result = _not_run(setup)
    assert result.returncode == EXIT_BUSY and setup.remote.servers() == []


def test_a_taken_local_port_is_busy_before_any_connection(setup: Setup) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", setup.local_port))
        listener.listen(1)
        result = _not_run(setup)
    assert result.returncode == EXIT_BUSY and "port" in result.stderr.lower() and setup.ssh_calls() == []


def test_a_server_that_fails_to_start_is_reported(setup: Setup) -> None:
    result = _not_run(setup, FAKE_OLLAMA_MODE="crash")
    assert result.returncode == EXIT_FAILED and "fake crash" in result.stderr and _servers_stopped(setup)


def test_a_tunnel_that_reaches_another_server_is_refused(setup: Setup) -> None:
    """Windows 那一側有別的程式在聽同一個 port：通道接到的不是這次啟動的 server（它的 log 沒有這次的請求）。"""
    binary = setup.tmp_path / "windows-side" / "ollama"  # 另一個執行檔：不是 WSL 內的 Ollama 程序
    binary.parent.mkdir()
    binary.write_bytes(setup.remote.binary.read_bytes())
    binary.chmod(0o755)
    impostor_port = free_port()
    impostor = subprocess.Popen([str(binary), "serve"], env={**setup.remote.env(), "OLLAMA_HOST": f"127.0.0.1:{impostor_port}",
                                                            "FAKE_OLLAMA_RECORD": str(setup.tmp_path / "impostor")},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(impostor_port)
        result = _not_run(setup, FAKE_SSH_FORWARD_PORT=str(impostor_port))
        assert result.returncode == EXIT_FAILED and "not the server this session started" in result.stderr
        assert _servers_stopped(setup) and setup.lock_is_free() and alive(impostor.pid)
    finally:
        impostor.terminate()
        impostor.wait()


def _wait_for_port(port: int, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"nothing listens on {port}")


def test_a_change_to_the_shared_models_is_a_failure_after_the_command(setup: Setup) -> None:
    result = setup.run("true", FAKE_OLLAMA_MODE="touch_source")
    assert result.returncode == EXIT_DIRTY and "changed" in result.stderr.lower() and _servers_stopped(setup)


def test_a_failing_command_keeps_its_exit_code_even_when_the_models_changed(setup: Setup) -> None:
    result = setup.run(sys.executable, "-c", "raise SystemExit(3)", FAKE_OLLAMA_MODE="touch_source")
    assert result.returncode == 3 and "changed" in result.stderr.lower()


def test_a_server_that_ignores_sigterm_is_still_stopped(setup: Setup) -> None:
    result = setup.run("true", FAKE_OLLAMA_MODE="ignore_term")
    assert result.returncode == 0, result.stderr
    assert _servers_stopped(setup) and setup.lock_is_free()


def test_an_interrupted_wrapper_stops_the_server(setup: Setup) -> None:
    marker = setup.tmp_path / "started"
    proc = subprocess.Popen([*setup.args(), "--", sys.executable, "-c",
                             f"import time; open({str(marker)!r}, 'w'); time.sleep(60)"],
                            env=setup.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert _wait_for_file(marker)
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143, stderr
    assert _servers_stopped(setup) and setup.lock_is_free()


def _wait_for_file(path: Path, timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def test_a_remote_that_never_sees_the_end_is_stopped_over_a_second_connection(setup: Setup) -> None:
    result = setup.run("true", FAKE_SSH_MODE="hang", WITH_GPU_LOCK_STOP_TIMEOUT_S="3")
    holder = setup.tmp_path / "ssh.log.holder"
    try:
        assert result.returncode == 0, result.stderr
        assert "second connection" in result.stderr.lower()
        assert len(setup.ssh_calls()) == 2 and "-L" not in setup.ssh_calls()[1]
        assert "--stop-pid" in setup.ssh_calls()[1][-1]
        assert _servers_stopped(setup)
        deadline = time.monotonic() + 20
        while not setup.lock_is_free() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert setup.lock_is_free()
    finally:
        if holder.exists():
            try:
                os.kill(int(holder.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_an_unreachable_remote_is_reported(setup: Setup) -> None:
    assert _not_run(setup, mode="down").returncode == EXIT_FAILED


def test_a_remote_that_never_confirms_the_lock_times_out(setup: Setup) -> None:
    result = _not_run(setup, mode="silent", WITH_GPU_LOCK_TIMEOUT_S="2")
    assert result.returncode == EXIT_FAILED


@pytest.mark.parametrize(
    "args",
    [[], ["--"], ["--lock-file"], ["--local-port", "abc", "--", "true"], ["--lock-file", "relative.lock", "--", "true"],
     ["--lock-file", "/tmp/x y.lock", "--", "true"], ["--bogus", "--", "true"], ["--remote-port", "0", "--", "true"],
     ["--server-log", "", "--", "true"]],
    ids=["no-command", "empty-command", "missing-value", "bad-port", "relative-lock", "space-in-lock", "unknown-option",
         "zero-port", "empty-log"],
)
def test_usage_errors(args: list[str], setup: Setup) -> None:
    result = subprocess.run([str(WRAPPER), *args], env=setup.env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 64 and setup.ssh_calls() == []


def test_nested_use_is_refused(setup: Setup) -> None:
    result = subprocess.run([*setup.args(), "--", "true"], env={**setup.env, "AI_PORTFOLIO_GPU_LOCK": "held"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 64 and setup.ssh_calls() == []


def test_the_cli_takes_the_tunnel_url_from_the_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import run_baseline

    monkeypatch.setenv("AGENT_GYM_OLLAMA_URL", "http://127.0.0.1:11435")
    assert run_baseline.ollama_url_default() == "http://127.0.0.1:11435"
    monkeypatch.delenv("AGENT_GYM_OLLAMA_URL")
    assert run_baseline.ollama_url_default() == "http://127.0.0.1:11434"


def test_defaults_are_the_shared_lock_and_this_projects_ports() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "LOCK_FILE=/tmp/ai-portfolio-gpu.lock" in text and "LOCAL_PORT=11435" in text and "REMOTE_PORT=11436" in text
    assert "/home/" not in text
