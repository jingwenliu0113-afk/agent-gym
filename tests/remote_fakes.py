"""agents/ollama_serve.py 與 tools/with-gpu-lock 測試用的假遠端（decision 029）。不連任何遠端、不使用 GPU。

- ``make_remote(tmp_path)``：假的 WSL home（``opt/ollama-0.35.0/bin/ollama`` 是假的 Ollama、``ii-ollama-models`` 是共用的
  models 目錄：manifest、四個 blob、metadata 快取）與 PATH 上的假 ``nvidia-smi`` / ``python3`` / ``ssh``。
- 假的 Ollama（``FAKE_OLLAMA`` 的內容）：``serve`` 以 OLLAMA_HOST 提供 ``/api/version``，每個請求先寫一行 gin 形式的 log（含 query
  string）再回應；啟動一個同 process group 的 ``runner`` 子程序；把 argv[0] 與收到的環境記到 ``FAKE_OLLAMA_RECORD``
  （runner 的 pid 記到 ``<record>.runners``）。``FAKE_OLLAMA_MODE``（逗號分隔）：crash、ignore_term、touch_source、write_models、
  no_query_log、detached_runner（runner 離開 server 的 process group）。
- 假的 nvidia-smi：``FAKE_GPU_USED``、``FAKE_GPU_FAIL``、``FAKE_GPU_SLEEP``（先等幾秒）。
- 假的 ssh（``FAKE_SSH`` 的內容）：解析 ``-L``，在本機轉送（``FAKE_SSH_FORWARD_PORT`` 可改轉送終點，模擬 Windows 那一側的
  其他程式）；把 ``wsl.exe -d Ubuntu-22.04 -- bash -lc "<cmd>"`` 換成本機的 ``bash -c "<cmd> $FAKE_SSH_EXTRA_ARGS"``，HOME 換成
  假的 home。遠端在自己的 session 執行，輸出經過假 ssh 轉寫：假 ssh 被殺掉後遠端再也寫不到 Mac（與真正斷線相同）。
  ``FAKE_SSH_MODE``：down（無法連線）、silent（連上但遠端不回應）、hang（有 -L 的連線：遠端收不到 stdin 的結束）、
  drop_at_verify（有 -L 的連線：wrapper 送出 VERIFY 時連線中斷）、drop_after_locked（有 -L 的連線：LOCKED 之後中斷，
  遠端收不到 stdin 的結束）。ssh 自己的訊息與真正的 OpenSSH 一樣：有 ``-E <file>`` 時寫到那個檔案。
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "agents" / "ollama_serve.py"
WRAPPER = ROOT / "tools" / "with-gpu-lock"

FAKE_OLLAMA = r'''
import http.server, json, os, signal, subprocess, sys, time

MODES = set(filter(None, os.environ.get("FAKE_OLLAMA_MODE", "").split(",")))

record = os.environ.get("FAKE_OLLAMA_RECORD")
if sys.argv[1:2] == ["runner"]:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if "detached_runner" in MODES:  # runner 離開 server 的 process group（停止整個 group 也殺不到它）
        os.setsid()
    if record:
        with open(record + ".runners", "a", encoding="utf-8") as handle:
            handle.write("%d\n" % os.getpid())
    time.sleep(600)
    sys.exit(0)
assert sys.argv[1:] == ["serve"], sys.argv
if record:
    with open(record, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0), "cwd": os.getcwd(), "argv0": sys.argv[0],
                                 "env": {k: v for k, v in os.environ.items() if k.startswith("OLLAMA_") or k == "HOME"}}) + "\n")
if "crash" in MODES:
    print("Error: listen tcp: fake crash", file=sys.stderr, flush=True)
    sys.exit(1)
if "touch_source" in MODES:
    with open(os.environ["FAKE_SOURCE_TOUCH"], "ab") as handle:
        handle.write(b" ")
if "write_models" in MODES:  # 真正的 Ollama 第一次使用模型時會在 OLLAMA_MODELS/metadata 寫快取
    os.makedirs(os.path.join(os.environ["OLLAMA_MODELS"], "metadata"), exist_ok=True)
    with open(os.path.join(os.environ["OLLAMA_MODELS"], "metadata", "cache.json"), "w", encoding="utf-8") as handle:
        handle.write("{}")
runner = subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0]), "runner"])

def stop(signum, frame):
    if "detached_runner" not in MODES:  # detached_runner：server 結束時沒有停止它的推論程序（殘留的情況）
        runner.terminate()
        runner.wait()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal.SIG_IGN if "ignore_term" in MODES else stop)
print("time=2026-10-11 level=INFO msg=\"server config\" env=\"fake\"", file=sys.stderr, flush=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0] if "no_query_log" in MODES else self.path
        sys.stderr.write('[GIN] 2026/10/11 - 12:00:00 | 200 | 1ms | 127.0.0.1 | GET "%s"\n' % path)
        sys.stderr.flush()
        body = json.dumps({"version": "0.35.0"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
'''

FAKE_SSH = r'''
import os, re, select, socket, subprocess, sys, threading, time

args = sys.argv[1:]
with open(os.environ["FAKE_SSH_LOG"], "a", encoding="utf-8") as handle:
    handle.write("\x1f".join(args) + "\n")
mode = os.environ.get("FAKE_SSH_MODE", "")
forward, ssh_log, rest, i = None, None, [], 0
while i < len(args):
    if args[i] in ("-o", "-L", "-E"):
        if args[i] == "-L":
            forward = args[i + 1]
        if args[i] == "-E":
            ssh_log = args[i + 1]
        i += 2
        continue
    if args[i] in ("-n", "-T"):
        i += 1
        continue
    rest.append(args[i])
    i += 1
host, command = rest
assert host == "gpu-server", host


def ssh_says(message):  # ssh 自己的訊息：有 -E 時寫到那個檔案（與真正的 OpenSSH 相同），否則寫到 stderr
    if ssh_log:
        with open(ssh_log, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    else:
        print(message, file=sys.stderr, flush=True)


if mode == "down":
    ssh_says("ssh: connect to host gpu-server port 22: Operation timed out")
    sys.exit(255)
if mode == "silent":
    time.sleep(60)
    sys.exit(255)

def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

def accept(listener, port):
    while True:
        client, _ = listener.accept()
        try:
            upstream = socket.create_connection(("127.0.0.1", port))
        except OSError:
            client.close()
            continue
        threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
        threading.Thread(target=pump, args=(upstream, client), daemon=True).start()

if forward:
    _, local, _, remote = forward.split(":")
    target = int(os.environ.get("FAKE_SSH_FORWARD_PORT") or remote)
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", int(local)))
    except OSError:
        ssh_says("bind [127.0.0.1]:%s: Address already in use" % local)
        ssh_says("Could not request local forwarding.")
        sys.exit(255)
    listener.listen(16)
    threading.Thread(target=accept, args=(listener, target), daemon=True).start()

m = re.fullmatch(r'wsl\.exe -d Ubuntu-22\.04 -- bash -lc "(.*)"', command, re.S)
assert m, command
inner = m.group(1)
if os.environ.get("FAKE_SSH_EXTRA_ARGS"):
    inner += " " + os.environ["FAKE_SSH_EXTRA_ARGS"]
env = {**os.environ, "HOME": os.environ["FAKE_REMOTE_HOME"]}

def relay(src, fd):  # 遠端的輸出經過 ssh 本身：ssh 被殺掉後，遠端再也寫不到 Mac（與真正斷線相同）
    while True:
        data = os.read(src, 65536)
        if not data:
            break
        try:
            os.write(fd, data)
        except OSError:
            pass
        if mode == "drop_after_locked" and forward and fd == 1 and b"LOCKED\n" in data:
            # 連線在 LOCKED 之後中斷：Mac 收不到之後的輸出，遠端也收不到 stdin 的結束（Windows sshd 還沒察覺）
            ssh_says("Read from remote host gpu-server: Operation timed out")
            os._exit(255)

def start(stdin):
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    proc = subprocess.Popen(["bash", "-c", inner], env=env, stdin=stdin, stdout=out_w, stderr=err_w,
                            start_new_session=True)  # 遠端不在 Mac 的 process group（送給 wrapper 的信號不會直接打到它）
    os.close(out_w)
    os.close(err_w)
    threads = [threading.Thread(target=relay, args=(out_r, 1), daemon=True),
               threading.Thread(target=relay, args=(err_r, 2), daemon=True)]
    for thread in threads:
        thread.start()
    return proc, threads

if mode == "drop_at_verify" and forward:  # wrapper 送出 VERIFY 時連線中斷：遠端收到 stdin 的結束，Mac 收不到遠端的輸出
    read_end, write_end = os.pipe()
    proc, threads = start(read_end)
    os.close(read_end)
    while True:
        data = os.read(0, 65536)
        if not data or re.search(rb"(^|\n)VERIFY [0-9a-f]{32}\n", data):  # helper 的原始碼本身也含 "VERIFY"
            break
        os.write(write_end, data)
    os.close(write_end)
    os._exit(255)
if mode in ("hang", "drop_after_locked") and forward:  # 遠端收不到 stdin 的結束：另一個程序一直開著寫入端
    read_end, write_end = os.pipe()
    holder = subprocess.Popen(["sleep", "600"], pass_fds=(write_end,))
    with open(os.environ["FAKE_SSH_LOG"] + ".holder", "w", encoding="utf-8") as handle:
        handle.write(str(holder.pid))
    proc, threads = start(read_end)
    os.close(read_end)
    while True:
        data = os.read(0, 65536)
        if not data:
            break
        os.write(write_end, data)
else:
    proc, threads = start(None)
code = proc.wait()
for thread in threads:
    thread.join(timeout=5)
sys.exit(code)
'''


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@dataclass
class Remote:
    home: Path
    bin_dir: Path
    binary: Path
    source: Path
    state: Path
    record: Path
    manifest: Path
    metadata: Path

    def env(self, **extra: str) -> dict[str, str]:
        """helper 直接在本機執行時的環境（HOME 換成假的 home、PATH 最前面是假的工具）。"""
        env = {k: v for k, v in os.environ.items() if k not in ("AI_PORTFOLIO_GPU_LOCK", "AGENT_GYM_OLLAMA_URL",
                                                                 "AGENT_GYM_OLLAMA_SERVER")}
        env.update(HOME=str(self.home), PATH=f"{self.bin_dir}{os.pathsep}{env['PATH']}", FAKE_OLLAMA_RECORD=str(self.record),
                   FAKE_SOURCE_TOUCH=str(self.metadata), FAKE_REMOTE_HOME=str(self.home), **extra)
        return env

    @property
    def launcher(self) -> Path:
        """server 以這個 symlink 啟動（decision 031 §3）：命令列不是「<binary> serve」。"""
        return self.state / "bin" / "agent-gym-ollama"

    def servers(self) -> list[dict[str, object]]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]

    def runners(self) -> list[int]:
        path = Path(str(self.record) + ".runners")
        return [int(line) for line in path.read_text(encoding="utf-8").split()] if path.exists() else []

    def kill_all(self) -> None:
        """測試結束時清掉這個假遠端留下的任何程序（失敗的測試不會遺留假 server）。"""
        for server in self.servers():
            try:
                os.killpg(int(server["pgid"]), signal.SIGKILL)  # type: ignore[call-overload]
            except (ProcessLookupError, PermissionError):
                pass
        for pid in self.runners():
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def make_remote(tmp_path: Path) -> Remote:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    binary = home / "opt" / "ollama-0.35.0" / "bin" / "ollama"
    _executable(binary, f"#!{sys.executable}\n{FAKE_OLLAMA}")
    _executable(bin_dir / "nvidia-smi", '#!/bin/sh\n[ -n "$FAKE_GPU_SLEEP" ] && sleep "$FAKE_GPU_SLEEP"\n'
                '[ -n "$FAKE_GPU_FAIL" ] && exit 9\necho "${FAKE_GPU_USED:-675}, 16303"\n')
    _executable(bin_dir / "python3", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    _executable(bin_dir / "ssh", f"#!{sys.executable}\n{FAKE_SSH}")
    source = home / "ii-ollama-models"
    blobs = {name: f"{name} bytes".encode() * 50 for name in ("model", "license", "params", "config")}
    for data in blobs.values():
        path = source / "blobs" / f"sha256-{_sha(data)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": f"sha256:{_sha(blobs['config'])}", "size": len(blobs["config"])},
        "layers": [{"digest": f"sha256:{_sha(blobs[name])}", "size": len(blobs[name])} for name in ("model", "license", "params")],
    }
    manifest_path = source / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "9b-q4_K_M"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata = source / "metadata" / f"sha256-{_sha(blobs['model'])}.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"ollama_version": "0.35.0"}', encoding="utf-8")
    return Remote(home=home, bin_dir=bin_dir, binary=binary, source=source, state=home / "agent-gym-ollama",
                  record=tmp_path / "servers.jsonl", manifest=manifest_path, metadata=metadata)


def alive(pid: int) -> bool:
    """程序存在且不是已結束、尚未回收的 zombie。"""
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")
