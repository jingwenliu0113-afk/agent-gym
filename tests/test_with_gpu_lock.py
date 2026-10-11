"""tools/with-gpu-lock 的契約測試（decision 026 §3、MACHINE_POLICY）。

以 PATH 最前面的假 ``ssh`` 模擬遠端：取得 lock（印出 LOCKED、等 stdin 關閉）、lock 被占用（exit 1）、遠端無法連線（exit 255）。
真正的拓撲（Mac → SSH 通道 → Windows localhost → WSL loopback，以及經 stdin 持有 / 釋放 flock）已在 2026-10-11 以測試用的 lock 檔
與臨時 HTTP 服務實測（NEXT）。這裡不連到任何遠端。
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "tools" / "with-gpu-lock"

FAKE_SSH = r"""#!/usr/bin/env bash
# 記錄參數；依 FAKE_SSH_MODE 模擬遠端
printf '%s\n' "$@" > "$FAKE_SSH_LOG"
case "$FAKE_SSH_MODE" in
  locked) echo LOCKED; cat >/dev/null; echo released >> "$FAKE_SSH_LOG.released"; exit 0 ;;
  busy) exit 1 ;;
  down) echo "ssh: connect to host gpu-server: Operation timed out" >&2; exit 255 ;;
  silent) sleep 60 ;;
esac
"""


@pytest.fixture
def fake_ssh(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(FAKE_SSH, encoding="utf-8")
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k not in ("AI_PORTFOLIO_GPU_LOCK", "AGENT_GYM_OLLAMA_URL")}
    env.update(PATH=f"{bin_dir}{os.pathsep}{env['PATH']}", FAKE_SSH_LOG=str(tmp_path / "ssh.log"), WITH_GPU_LOCK_TIMEOUT_S="5")
    return env


def _run(env: dict[str, str], mode: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(WRAPPER), *args], env={**env, "FAKE_SSH_MODE": mode}, capture_output=True, text=True, timeout=30)


PRINT_ENV = [sys.executable, "-c", "import os; print(os.environ.get('AI_PORTFOLIO_GPU_LOCK'), os.environ.get('AGENT_GYM_OLLAMA_URL'))"]


def test_the_wrapper_is_executable() -> None:
    assert WRAPPER.exists() and os.access(WRAPPER, os.X_OK)


def test_the_command_runs_inside_the_lock_with_the_tunnel_url(fake_ssh: dict[str, str], tmp_path: Path) -> None:
    result = _run(fake_ssh, "locked", "--", *PRINT_ENV)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "held http://127.0.0.1:11435"
    ssh_args = (tmp_path / "ssh.log").read_text(encoding="utf-8").splitlines()
    assert "-L" in ssh_args and "127.0.0.1:11435:127.0.0.1:11434" in ssh_args and "gpu-server" in ssh_args
    assert "ExitOnForwardFailure=yes" in ssh_args and "BatchMode=yes" in ssh_args
    remote = ssh_args[-1]
    assert "wsl.exe -d Ubuntu-22.04" in remote and "flock -n /tmp/ai-portfolio-gpu.lock" in remote
    assert (tmp_path / "ssh.log.released").exists()  # 指令結束後關閉 stdin → 遠端釋放 lock


def test_the_exit_code_of_the_command_is_returned(fake_ssh: dict[str, str], tmp_path: Path) -> None:
    result = _run(fake_ssh, "locked", "--", sys.executable, "-c", "raise SystemExit(7)")
    assert result.returncode == 7 and (tmp_path / "ssh.log.released").exists()


def test_a_busy_lock_is_not_taken(fake_ssh: dict[str, str], tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    result = _run(fake_ssh, "busy", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w')")
    assert result.returncode == 75 and not marker.exists()
    assert "lock" in result.stderr.lower() and "busy" in result.stderr.lower()


def test_an_unreachable_remote_is_reported(fake_ssh: dict[str, str], tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    result = _run(fake_ssh, "down", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w')")
    assert result.returncode == 69 and not marker.exists()


def test_a_remote_that_never_confirms_the_lock_times_out(fake_ssh: dict[str, str], tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    result = _run(fake_ssh, "silent", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w')")
    assert result.returncode == 69 and not marker.exists()


def test_options_override_the_lock_file_and_ports(fake_ssh: dict[str, str], tmp_path: Path) -> None:
    result = _run(fake_ssh, "locked", "--lock-file", "/tmp/agent-gym-wrapper-test.lock", "--local-port", "18766",
                  "--remote-port", "18765", "--", *PRINT_ENV)
    assert result.returncode == 0 and result.stdout.strip() == "held http://127.0.0.1:18766"
    ssh_args = (tmp_path / "ssh.log").read_text(encoding="utf-8").splitlines()
    assert "127.0.0.1:18766:127.0.0.1:18765" in ssh_args and "flock -n /tmp/agent-gym-wrapper-test.lock" in ssh_args[-1]


@pytest.mark.parametrize(
    "args",
    [[], ["--"], ["--lock-file"], ["--local-port", "abc", "--", "true"], ["--lock-file", "relative.lock", "--", "true"],
     ["--lock-file", "/tmp/x y.lock", "--", "true"], ["--bogus", "--", "true"]],
    ids=["no-command", "empty-command", "missing-value", "bad-port", "relative-lock", "space-in-lock", "unknown-option"],
)
def test_usage_errors(args: list[str], fake_ssh: dict[str, str]) -> None:
    assert _run(fake_ssh, "locked", *args).returncode == 64


def test_nested_use_is_refused(fake_ssh: dict[str, str]) -> None:
    result = subprocess.run([str(WRAPPER), "--", "true"], env={**fake_ssh, "FAKE_SSH_MODE": "locked", "AI_PORTFOLIO_GPU_LOCK": "held"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 64


def test_the_cli_takes_the_tunnel_url_from_the_wrapper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import run_baseline

    monkeypatch.setenv("AGENT_GYM_OLLAMA_URL", "http://127.0.0.1:11435")
    assert run_baseline.ollama_url_default() == "http://127.0.0.1:11435"
    monkeypatch.delenv("AGENT_GYM_OLLAMA_URL")
    assert run_baseline.ollama_url_default() == "http://127.0.0.1:11434"
