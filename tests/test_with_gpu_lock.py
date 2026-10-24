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
from typing import Callable, Iterator

import pytest

from agents.ollama_serve import EXIT_BUSY, EXIT_DIRTY, EXIT_FAILED, SERVER_ENV, diff_snapshots, snapshot
from tests.remote_fakes import WRAPPER, Remote, alive, free_port, make_remote

PRINT_ENV = [sys.executable, "-c", "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
             "('AI_PORTFOLIO_GPU_LOCK', 'AGENT_GYM_OLLAMA_URL', 'AGENT_GYM_OLLAMA_SERVER')}))"]


class Setup:
    def __init__(self, tmp_path: Path, wrapper: Path = WRAPPER) -> None:
        self.tmp_path = tmp_path
        self.wrapper = wrapper
        self.remote: Remote = make_remote(tmp_path)
        self.lock = tmp_path / "gpu.lock"
        self.local_port = free_port()
        self.remote_port = free_port()
        self.ssh_log = tmp_path / "ssh.log"
        self.env = self.remote.env(FAKE_SSH_LOG=str(self.ssh_log), FAKE_SSH_EXTRA_ARGS="--process-names= --stop-grace-s 2",
                                   WITH_GPU_LOCK_TIMEOUT_S="10", WITH_GPU_LOCK_STOP_TIMEOUT_S="30")
        self.env["HOME"] = os.environ.get("HOME", str(tmp_path))  # Mac 端的 HOME 不變；假的 ssh 把遠端的 HOME 換掉
        (tmp_path / "tmp").mkdir()
        self.env["TMPDIR"] = str(tmp_path / "tmp")  # wrapper 的工作目錄（警告時保留）留在測試的暫存目錄裡

    def args(self, *extra: str) -> list[str]:
        return [str(self.wrapper), "--lock-file", str(self.lock), "--local-port", str(self.local_port), "--remote-port",
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
def setup(tmp_path: Path) -> Iterator[Setup]:
    fake = Setup(tmp_path)
    yield fake
    _clean(fake)


def _clean(setup: Setup) -> None:
    """失敗的測試也不遺留假 server、runner、helper 或 hang 模式的 stdin holder：holder 結束後 helper 會收到 stdin 的結束；
    之後仍在的、命令列含這個測試暫存目錄的程序（全部是測試自己的）一律結束。"""
    holder = setup.tmp_path / "ssh.log.holder"
    if holder.exists():
        try:
            os.kill(int(holder.read_text()), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    time.sleep(0.5)
    for _ in range(2):
        subprocess.run(["pkill", "-KILL", "-f", str(setup.tmp_path)], capture_output=True)
        setup.remote.kill_all()
        time.sleep(0.2)


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
    assert result.returncode == EXIT_BUSY and "held by another workload" in result.stderr and setup.remote.servers() == []


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
    status, command, stderr = _interrupt(setup, _sleeper(marker), marker, signal.SIGTERM)
    assert status == 143, stderr
    assert not alive(command) and _servers_stopped(setup) and setup.lock_is_free()


LONG_S = 600  # 指令睡得比任何等待都久：wrapper 沒有結束它時，測試以「指令還在」失敗，而不是指令剛好在逾時前自己結束


def _sleeper(marker: Path) -> list[str]:
    """寫下自己的 pid、之後睡 LONG_S 秒的指令。"""
    return [sys.executable, "-c", f"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep({LONG_S})"]


def _wait_for_pid(marker: Path, timeout: float = 30) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if text:
            return int(text)
        time.sleep(0.05)
    raise AssertionError(f"{marker} was not written within {timeout}s")


def _interrupt(setup: Setup, command: list[str], marker: Path, signum: int,
               then: Callable[[subprocess.Popen[bytes], int], object] | None = None, **env: str) -> tuple[int, int, str]:
    """執行指令；它寫下 pid（marker）之後送 signum 給 wrapper、呼叫 then(proc, pid)，等 wrapper 結束。
    回傳 wrapper 的 exit code、指令的 pid 與 wrapper 的 stderr。輸出寫到檔案而不是 pipe：遺留的程序持有輸出時，等待不會因此逾時，
    而是由「指令還在」的檢查失敗（2026-10-24 完整測試中的逾時）。"""
    stderr_path = setup.tmp_path / "wrapper.stderr"
    with open(stderr_path, "w", encoding="utf-8") as stderr:
        proc = subprocess.Popen([*setup.args(), "--", *command], env={**setup.env, **env}, stdout=subprocess.DEVNULL,
                                stderr=stderr)
    try:
        pid = _wait_for_pid(marker)
        proc.send_signal(signum)
        if then is not None:
            then(proc, pid)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return proc.returncode, pid, stderr_path.read_text(encoding="utf-8")


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
    result = _not_run(setup, mode="silent", WITH_GPU_LOCK_TIMEOUT_S="2", WITH_GPU_LOCK_STOP_TIMEOUT_S="2")
    assert result.returncode == EXIT_DIRTY  # 連線被結束、遠端沒有回答：無法證明什麼都沒啟動 → 回報使用者（MACHINE_POLICY）
    assert "could not confirm that nothing was started" in result.stderr


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


# ---------------------------------------------------------------- decision 031：第 029 輪審查的回歸測試


def test_the_remote_server_runs_under_the_launcher(setup: Setup) -> None:
    assert setup.run("true").returncode == 0
    [record] = setup.remote.servers()
    assert record["argv0"] == str(setup.remote.launcher)


def test_an_interrupt_before_the_server_starts_starts_nothing(setup: Setup) -> None:
    """wrapper 在 server 啟動前被中斷：遠端收到 stdin 的結束，不啟動 server；wrapper 有證據（NOT_STARTED），不必警告。"""
    proc = subprocess.Popen([*setup.args(), "--", "true"], env={**setup.env, "FAKE_GPU_SLEEP": "3"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.5)  # 已取得 lock，正在檢查 GPU
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143, stderr
    time.sleep(4)  # 若 helper 仍在跑，它在 GPU 檢查之後就會啟動 server
    assert setup.remote.servers() == [] and setup.lock_is_free() and "WARNING" not in stderr


def test_an_interrupt_before_the_start_on_a_remote_that_never_sees_the_end_is_warned(setup: Setup) -> None:
    proc = subprocess.Popen([*setup.args(), "--", "true"],
                            env={**setup.env, "FAKE_GPU_SLEEP": "6", "FAKE_SSH_MODE": "hang", "WITH_GPU_LOCK_STOP_TIMEOUT_S": "2"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.5)
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == EXIT_DIRTY  # 指令還沒執行、又無法證明什麼都沒啟動：70，不是 143
    assert "could not confirm that nothing was started" in stderr and "pgrep" in stderr and "diagnostics kept in" in stderr


FIFO_FDS = [sys.executable, "-c", (
    "import json, os, stat\n"
    "fifos = []\n"
    "for fd in range(3, 256):\n"
    "    try:\n"
    "        st = os.fstat(fd)\n"
    "    except OSError:\n"
    "        continue\n"
    "    if stat.S_ISFIFO(st.st_mode):\n"
    "        fifos.append(fd)\n"
    "print(json.dumps(fifos))\n")]


def test_the_command_does_not_inherit_the_remote_stdin(setup: Setup) -> None:
    """bash 3.2 的 `exec cmd <&8 8<&- 9>&-` 會把 fd 8 / 9 複製到 >= 10 的 fd 並傳給 cmd：指令不可持有任何 FIFO。"""
    result = setup.run(*FIFO_FDS)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_a_daemon_left_by_the_command_does_not_keep_the_remote_running(setup: Setup) -> None:
    program = setup.tmp_path / "daemon.py"
    # 真正的 daemon：離開 session、0 / 1 / 2 改到 /dev/null（不持有測試擷取輸出的 pipe），其他 fd 照樣繼承
    program.write_text("import os, time\nif os.fork() == 0:\n    os.setsid()\n    null = os.open(os.devnull, os.O_RDWR)\n"
                       "    for fd in (0, 1, 2):\n        os.dup2(null, fd)\n    time.sleep(40)\n", encoding="utf-8")
    started = time.monotonic()
    result = setup.run(sys.executable, str(program), WITH_GPU_LOCK_STOP_TIMEOUT_S="20")
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert "second connection" not in result.stderr, result.stderr  # daemon 沒有持有遠端 stdin 的 FIFO
    assert elapsed < 15 and _servers_stopped(setup)


def test_a_command_that_stops_is_ended(setup: Setup) -> None:
    """指令被停住（例如在背景 process group 讀取終端機：SIGTTIN）：wrapper 不會無限期持有 lock。"""
    result = setup.run(sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSTOP)")
    assert result.returncode != 0 and "stopped" in result.stderr.lower()
    assert _servers_stopped(setup) and setup.lock_is_free()


def test_a_terminal_stdin_is_not_given_to_the_command(setup: Setup) -> None:
    """從終端機執行：指令在自己的 process group，讀終端機會被停住；改給它 /dev/null（讀到結束），wrapper 正常結束。"""
    import pty

    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - 子程序
        os.environ.clear()
        os.environ.update(setup.env)
        os.execv(str(setup.wrapper), [*setup.args(), "--", sys.executable, "-c",
                                      "import sys; print('read=%r' % sys.stdin.read())"])
    output = b""
    deadline = time.monotonic() + 90
    status = None
    while time.monotonic() < deadline:
        try:
            chunk = os.read(master, 4096)
        except OSError:
            chunk = b""
        if chunk:
            output += chunk
        done, code = os.waitpid(pid, os.WNOHANG)
        if done:
            status = code
            break
        if not chunk:
            time.sleep(0.1)
    if status is None:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    os.close(master)
    assert status is not None and os.waitstatus_to_exitcode(status) == 0, output.decode(errors="replace")
    assert b"read=''" in output


def test_a_connection_lost_after_locked_is_not_taken_as_proof_that_nothing_started(setup: Setup) -> None:
    result = _not_run(setup, mode="drop_after_locked", WITH_GPU_LOCK_STOP_TIMEOUT_S="2")
    assert result.returncode == EXIT_DIRTY and "could not confirm" in result.stderr, result.stderr


def test_messages_before_the_session_survive_a_closed_stderr(setup: Setup) -> None:
    def run_closed(*args: str) -> int:
        proc = subprocess.Popen([str(WRAPPER), *args], env=setup.env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stderr is not None
        proc.stderr.close()
        return proc.wait(timeout=60)

    assert run_closed() == 64  # 用法錯誤：不是 SIGPIPE（141）
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", setup.local_port))
        listener.listen(1)
        assert run_closed(*setup.args()[1:], "--", "true") == EXIT_BUSY


def test_sigquit_still_cleans_up(setup: Setup) -> None:
    marker = setup.tmp_path / "started"
    status, command, stderr = _interrupt(setup, _sleeper(marker), marker, signal.SIGQUIT)
    assert status == 131, stderr
    assert not alive(command) and _servers_stopped(setup) and setup.lock_is_free()


def test_a_closed_stdin_is_not_an_error(setup: Setup) -> None:
    result = subprocess.run(["bash", "-c", 'exec "$@" <&-', "_", *setup.args(), "--", "true"], env=setup.env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert list((setup.tmp_path / "tmp").iterdir()) == []  # 沒有遺留的工作目錄


def test_an_interrupt_ends_the_commands_whole_process_group(setup: Setup) -> None:
    marker = setup.tmp_path / "grandchild"
    program = setup.tmp_path / "grandchild.py"
    program.write_text(f"import os, time\nopen({str(marker)!r}, 'w').write(str(os.getpid()))\ntime.sleep({LONG_S})\n",
                       encoding="utf-8")
    script = f"python3 {program}; echo done"  # bash 不轉送 TERM：只對 bash 送信號會留下 python
    status, grandchild, stderr = _interrupt(setup, ["bash", "-c", script], marker, signal.SIGTERM)
    assert status == 143, stderr
    assert not alive(grandchild) and _servers_stopped(setup)


def test_background_processes_left_by_the_command_are_ended(setup: Setup) -> None:
    marker = setup.tmp_path / "left"
    program = setup.tmp_path / "left.py"
    program.write_text(f"import os, time\nopen({str(marker)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n", encoding="utf-8")
    script = f"python3 {program} & sleep 0.5"
    result = setup.run("bash", "-c", script)
    assert result.returncode == 0, result.stderr
    assert not alive(int(marker.read_text())) and "second connection" not in result.stderr


def test_the_fallback_uses_the_helper_copied_at_the_start(tmp_path: Path) -> None:
    """執行期間 agents/ollama_serve.py 被修改（例如切換分支）：第二條連線仍送出與開始時相同的 helper。"""
    wrapper, helper = _wrapper_copy(tmp_path)
    setup = Setup(tmp_path, wrapper=wrapper)
    try:
        edit = f"p = {str(helper)!r}; t = open(p).read(); open(p, 'w').write('# edited during the run\\n' + t)"
        result = setup.run(sys.executable, "-c", edit, FAKE_SSH_MODE="hang", WITH_GPU_LOCK_STOP_TIMEOUT_S="3")
        assert result.returncode == 0, result.stderr
        assert "second connection" in result.stderr and _servers_stopped(setup)
    finally:
        _clean(setup)


def test_a_closed_stderr_does_not_change_the_exit_code(setup: Setup) -> None:
    log = setup.tmp_path / "server.log"
    proc = subprocess.Popen([*setup.args("--server-log", str(log)), "--", "true"], env=setup.env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stderr is not None
    proc.stderr.close()  # 讀取端已不在：wrapper 寫 stderr 會失敗
    assert proc.wait(timeout=120) == 0
    assert log.exists() and _servers_stopped(setup) and setup.lock_is_free()


def test_a_connection_that_ends_during_the_tunnel_check_is_not_blamed_on_windows(setup: Setup) -> None:
    result = _not_run(setup, mode="drop_at_verify")
    assert result.returncode == EXIT_FAILED
    assert "ended during the tunnel check" in result.stderr and "not the server this session started" not in result.stderr
    deadline = time.monotonic() + 20
    while not _servers_stopped(setup) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _servers_stopped(setup)


def test_a_warning_keeps_the_diagnostics(setup: Setup) -> None:
    result = setup.run("true", FAKE_OLLAMA_MODE="touch_source")
    assert result.returncode == EXIT_DIRTY
    kept = [line for line in result.stderr.splitlines() if "diagnostics kept in" in line]
    assert kept
    path = Path(kept[0].split("diagnostics kept in ", 1)[1].split(" ", 1)[0])
    assert path.parent == setup.tmp_path / "tmp"
    assert (path / "out").exists() and "MODELS_CHANGED" in (path / "out").read_text(encoding="utf-8")


def test_a_surviving_process_of_the_session_is_a_failure(setup: Setup) -> None:
    result = setup.run("true", FAKE_OLLAMA_MODE="detached_runner")
    assert result.returncode == EXIT_DIRTY and "WARNING" in result.stderr and "STOP_FAILED" in result.stderr


# ---------------------------------------------------------------- 2026-10-24：信號的時間窗（完整測試中 test_sigquit_still_cleans_up 逾時）


def _wrapper_copy(tmp_path: Path, edit: Callable[[str], str] | None = None) -> tuple[Path, Path]:
    """wrapper 與 helper 的複本（同樣的目錄結構）；edit 改寫 wrapper 的內容。回傳 (wrapper, helper)。"""
    tree = tmp_path / "tree"
    (tree / "tools").mkdir(parents=True)
    (tree / "agents").mkdir()
    wrapper = tree / "tools" / "with-gpu-lock"
    text = WRAPPER.read_text(encoding="utf-8")
    wrapper.write_text(edit(text) if edit else text, encoding="utf-8")
    wrapper.chmod(0o755)
    helper = tree / "agents" / "ollama_serve.py"
    helper.write_bytes((WRAPPER.parents[1] / "agents" / "ollama_serve.py").read_bytes())
    return wrapper, helper


def test_a_signal_while_the_command_is_being_started_still_ends_it(tmp_path: Path) -> None:
    """wrapper 剛 fork 出指令、還沒記下它的 process group 時收到信號（完整測試的負載下 wrapper 在這裡被延遲）：指令仍被結束。
    原本 trap 立刻結束 wrapper，cleanup 不知道指令：server 停了、lock 釋放了，指令卻繼續執行，並持有呼叫者的輸出 pipe
    （test_sigquit_still_cleans_up 因此等到指令自己的 60 秒結束，剛好碰上 60 秒的逾時）。
    複本在 fork 之後等 gate：信號一定在 wrapper 記下 process group 之前送達，不靠負載。"""
    gate = tmp_path / "gate"
    fork_line = 'exec "$@" ) &\n'

    def delay_after_the_fork(text: str) -> str:
        assert text.count(fork_line) == 1
        return text.replace(fork_line, fork_line + f"until [[ -e '{gate}' ]]; do sleep 0.05; done\n")

    wrapper, _ = _wrapper_copy(tmp_path, delay_after_the_fork)
    setup = Setup(tmp_path, wrapper=wrapper)
    marker = tmp_path / "started"
    try:
        status, command, stderr = _interrupt(setup, _sleeper(marker), marker, signal.SIGQUIT, then=lambda *_: gate.touch())
        assert status == 131, stderr
        assert not alive(command) and _servers_stopped(setup) and setup.lock_is_free()
    finally:
        _clean(setup)


def test_a_second_sigquit_does_not_cut_the_cleanup_short(setup: Setup) -> None:
    """cleanup 期間再收到一次 QUIT（再按一次 Ctrl-\\）：cleanup 照樣等 server 停止、確認之後才結束（decision 028 §4）。
    原本 cleanup 只忽略 INT / TERM / HUP：第二次 QUIT 讓 wrapper 在 server 停止之前就結束，沒有確認也沒有警告。
    server 忽略 SIGTERM：停止至少要 grace 的 2 秒，第二次 QUIT 一定落在 cleanup 之中。"""
    marker = setup.tmp_path / "started"

    def again(proc: subprocess.Popen[bytes], command: int) -> None:
        deadline = time.monotonic() + 30
        while alive(command) and time.monotonic() < deadline:  # cleanup 已開始：它先結束指令
            time.sleep(0.02)
        proc.send_signal(signal.SIGQUIT)

    status, _, stderr = _interrupt(setup, _sleeper(marker), marker, signal.SIGQUIT, then=again, FAKE_OLLAMA_MODE="ignore_term")
    assert status == 131, stderr
    assert _servers_stopped(setup) and setup.lock_is_free() and "GPU lock released" in stderr, stderr

