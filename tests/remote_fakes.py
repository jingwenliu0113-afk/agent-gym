"""agents/ollama_serve.py 與 tools/with-gpu-lock 測試用的假遠端（decision 029）。不連任何遠端、不使用 GPU。

- ``make_remote(tmp_path)``：假的 WSL home（``opt/ollama-0.35.0/bin/ollama`` 是假的 Ollama、``ii-ollama-models`` 是共用的
  models 目錄：manifest、四個 blob、metadata 快取）與 PATH 上的假 ``nvidia-smi`` / ``python3`` / ``ssh``。
- 假的 Ollama（``FAKE_OLLAMA`` 的內容）：``serve`` 以 OLLAMA_HOST 提供 ``/api/version``，每個請求先寫一行 gin 形式的 log（含 query
  string）再回應；啟動一個同 process group 的 ``runner`` 子程序；把收到的環境記到 ``FAKE_OLLAMA_RECORD``。
  ``FAKE_OLLAMA_MODE``（逗號分隔）：crash、ignore_term、touch_source、write_models、no_query_log。
- 假的 ssh（``FAKE_SSH`` 的內容）：解析 ``-L``，在本機轉送（``FAKE_SSH_FORWARD_PORT`` 可改轉送終點，模擬 Windows 那一側的
  其他程式）；把 ``wsl.exe -d Ubuntu-22.04 -- bash -lc "<cmd>"`` 換成本機的 ``bash -c "<cmd> $FAKE_SSH_EXTRA_ARGS"``，HOME 換成
  假的 home。``FAKE_SSH_MODE``：down（無法連線）、silent（連上但遠端不回應）、hang（有 -L 的連線：遠端收不到 stdin 的結束）。
"""

from __future__ import annotations

import hashlib
import json
import os
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

if sys.argv[1:2] == ["runner"]:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    time.sleep(600)
    sys.exit(0)
assert sys.argv[1:] == ["serve"], sys.argv
record = os.environ.get("FAKE_OLLAMA_RECORD")
if record:
    with open(record, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0), "cwd": os.getcwd(),
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
if mode == "down":
    print("ssh: connect to host gpu-server port 22: Operation timed out", file=sys.stderr)
    sys.exit(255)
if mode == "silent":
    time.sleep(60)
    sys.exit(255)
forward, rest, i = None, [], 0
while i < len(args):
    if args[i] in ("-o", "-L"):
        if args[i] == "-L":
            forward = args[i + 1]
        i += 2
        continue
    if args[i] in ("-n", "-T"):
        i += 1
        continue
    rest.append(args[i])
    i += 1
host, command = rest
assert host == "gpu-server", host

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
        print("bind [127.0.0.1]:%s: Address already in use" % local, file=sys.stderr)
        print("Could not request local forwarding.", file=sys.stderr)
        sys.exit(255)
    listener.listen(16)
    threading.Thread(target=accept, args=(listener, target), daemon=True).start()

m = re.fullmatch(r'wsl\.exe -d Ubuntu-22\.04 -- bash -lc "(.*)"', command, re.S)
assert m, command
inner = m.group(1)
if os.environ.get("FAKE_SSH_EXTRA_ARGS"):
    inner += " " + os.environ["FAKE_SSH_EXTRA_ARGS"]
env = {**os.environ, "HOME": os.environ["FAKE_REMOTE_HOME"]}
if mode == "hang" and forward:  # 遠端收不到 stdin 的結束：另一個程序一直開著寫入端
    read_end, write_end = os.pipe()
    holder = subprocess.Popen(["sleep", "600"], pass_fds=(write_end,))
    with open(os.environ["FAKE_SSH_LOG"] + ".holder", "w", encoding="utf-8") as handle:
        handle.write(str(holder.pid))
    proc = subprocess.Popen(["bash", "-c", inner], env=env, stdin=read_end)
    os.close(read_end)
    while True:
        data = os.read(0, 65536)
        if not data:
            break
        os.write(write_end, data)
    proc.wait()
    sys.exit(proc.returncode)
proc = subprocess.run(["bash", "-c", inner], env=env)
sys.exit(proc.returncode)
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

    def servers(self) -> list[dict[str, object]]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]


def make_remote(tmp_path: Path) -> Remote:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    binary = home / "opt" / "ollama-0.35.0" / "bin" / "ollama"
    _executable(binary, f"#!{sys.executable}\n{FAKE_OLLAMA}")
    _executable(bin_dir / "nvidia-smi", '#!/bin/sh\n[ -n "$FAKE_GPU_FAIL" ] && exit 9\necho "${FAKE_GPU_USED:-675}, 16303"\n')
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
