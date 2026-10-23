"""付費研究（Claude Haiku 5.5：Study 2 = decision 035，Study 3 = decision 036 §7）的總帳、累計上限、每次呼叫的保險絲與程序鎖。
不連網、不讀取任何 credential。每個研究的金額與總帳由 ``tasks.suites.PaidSpec`` 決定（``spec``）。

**總帳**：每個研究一份，在任何 checkout 之外（同一台機器的所有 checkout 與 worktree 共用，不會被 ``git clean`` 刪除）：
Study 2 為 ``~/.agent-gym/study2-haiku-billing.jsonl``（``DEFAULT_JOURNAL``），Study 3 為
``~/.agent-gym/study3-haiku-billing.jsonl``（``STUDY3_JOURNAL``）。每行一個 JSON 物件（ASCII），寫入後 flush + fsync::

    {"format": "agent-gym/billing/v2", "kind": "reserve", "id", "utc_time", "phase", "run", "bound_usd"}   # 呼叫之前
    {"format": "agent-gym/billing/v2", "kind": "settle",  "id", "utc_time", "cost_usd": <數或 null>}      # 已計費
    {"format": "agent-gym/billing/v2", "kind": "void",    "id", "utc_time"}                              # 沒有 response、不計費

已花費 = 已結算的成本 + 沒有結算也沒有作廢的預留（以其上限計：程序在呼叫途中被中斷時保守地計入）。結算為 null（無法計價）→ unknown。

**上限**（``paid_session``，在程序鎖內計算）：
- 一般停止：每個 phase（dev / baseline / sweep）全部合計 ≤ 該 phase 的上限（``spec.phase_caps``，有列出時）；全部合計 ≤
  ``spec.total_usd`` − ``SAFETY_MARGIN_USD``；在每個 episode / rerun 開始前檢查（``budget_reached``）。
- 保險絲（``PaidSession.reserve``）：每次呼叫前，開始時的已花費 + 這次 session 已結算 + 這次呼叫的上限 > 總預算（``total``），或這次
  session 已有無法計價的呼叫 → ``BudgetFuse``。
  呼叫的上限 = 1.1 ×（prompt 估計 × 最貴的 input 費率 + max_tokens × 最貴的 output 費率），prompt 以每 ``CHARS_PER_TOKEN_BOUND`` 個
  字元一個 token 從 request 估計（偏高）。所以合計不會超過總預算。
- 帳上有無法計價的呼叫、或 ``runs/`` 中這個 agent、這個研究（manifest 的 generator_version）的 manifest 合計大於總帳（總帳被
  重設或換了檔案）→ 拒絕開始。

**程序鎖**：同一份總帳同時只能有一個付費程序（``fcntl.flock``，非阻塞；拿不到就拒絕，不等待）。不同研究的總帳各有自己的鎖。
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, NamedTuple

from tasks.suites import PAID_PHASES, PaidSpec

JOURNAL_FORMAT = "agent-gym/billing/v2"
DEFAULT_JOURNAL = Path.home() / ".agent-gym" / "study2-haiku-billing.jsonl"  # Study 2（decision 035）
STUDY3_JOURNAL = Path.home() / ".agent-gym" / "study3-haiku-billing.jsonl"  # Study 3（decision 036 §7）
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"  # 核對總帳沒有被重設：這裡的 Haiku 目錄的 manifest 合計
SAFETY_MARGIN_USD = 1.0
PRICE_MULTIPLIER_BOUND = 1.1  # US-only 推論（decision 035 §3）
CHARS_PER_TOKEN_BOUND = 2.0  # 估計 prompt 的 token 數時每個 token 的字元數（偏小 → token 數偏高）
PHASES = PAID_PHASES  # dev（Study 3 的 dev baseline）、baseline（eval baseline）、sweep


class BudgetFuse(BaseException):
    """下一次呼叫的最壞成本會讓總花費超過總預算，或這次 run 出現無法計價的呼叫：在呼叫之前停止（decision 035 §3）。

    是 BaseException（與 KeyboardInterrupt 相同）：``replay.aborts.run_logged`` 不把它記成 harness 中止，manifest 的狀態與
    「下一個嘗試開始前的預算停止」相同；``_run_split`` / ``_run_sweeps`` 把它當成預算停止。"""


_ISSUED = object()  # 只有 paid_session 建立的 session 帶有這個標記


class Spent(NamedTuple):
    total: float
    baseline: float
    sweep: float
    unknown: int  # 無法計價的已計費呼叫數
    dev: float = 0.0


def _write(path: Path, line: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as handle:
        handle.write(json.dumps({"format": JOURNAL_FORMAT, **line}, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _number(value: Any, what: str, *, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"billing journal: {what} must be {'null or ' if nullable else ''}a finite number >= 0")
    return float(value)


def journal_spent(path: str | Path) -> Spent:
    """總帳的合計；檔案不存在 → 全部為 0。格式不符（含寫到一半的最後一行）→ ValueError：不猜測金額。"""
    path = Path(path)
    if not path.exists():
        return Spent(0.0, 0.0, 0.0, 0)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{path}: the last line is incomplete (interrupted write); check it by hand before any paid run")
    reserves: dict[str, tuple[str, float]] = {}
    settled: dict[str, float | None] = {}
    voided: set[str] = set()
    for number, line in enumerate(raw.split(b"\n")[:-1], start=1):
        try:
            item = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{number}: not a billing line") from exc
        if not isinstance(item, dict) or item.get("format") != JOURNAL_FORMAT or not isinstance(item.get("id"), str):
            raise ValueError(f"{path}:{number}: not a billing line")
        kind, rid = item.get("kind"), item["id"]
        if kind == "reserve" and item.get("phase") in PHASES and rid not in reserves:
            reserves[rid] = (item["phase"], _number(item.get("bound_usd"), "bound_usd", nullable=False))
        elif kind == "settle" and rid in reserves and rid not in settled and rid not in voided:
            settled[rid] = _number(item.get("cost_usd"), "cost_usd", nullable=True)
        elif kind == "void" and rid in reserves and rid not in settled and rid not in voided:
            voided.add(rid)
        else:
            raise ValueError(f"{path}:{number}: not a billing line (or out of order)")
    sums: dict[str, list[float]] = {phase: [] for phase in PHASES}
    unknown = 0
    for rid, (phase, bound) in reserves.items():
        if rid in voided:
            continue
        cost = settled[rid] if rid in settled else bound  # 沒有結算：以上限計（呼叫途中被中斷）
        if cost is None:
            unknown += 1
        else:
            sums[phase].append(cost)
    dev, baseline, sweep = (math.fsum(sums[phase]) for phase in ("dev", "baseline", "sweep"))
    return Spent(math.fsum((dev, baseline, sweep)), baseline, sweep, unknown, dev)


def journal_path(spec: PaidSpec) -> Path:
    """研究的總帳路徑（``spec.journal`` 指名的本模組屬性，使用時才讀取）。"""
    return Path(globals()[spec.journal])


def manifest_floor(config_id: str, root: str | Path | None = None, *, generator: str | None = None) -> float:
    """root（預設 ``RUNS_ROOT``）下（遞迴）每個 run.json 的 config_id 等於 config_id、不是 dry-run 的目錄的 manifest cost_usd 合計；
    generator 不是 None 時只計入 generator_version 等於它的列（同一個 agent 在不同研究的花費不混在一起）。"""
    from replay.aborts import read_manifest  # 延遲 import

    root = Path(RUNS_ROOT if root is None else root)
    total = []
    if not root.is_dir():
        return 0.0
    for run_file in sorted(root.rglob("run.json")):  # 遞迴：巢狀的 run 名稱也算
        try:
            run = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(run, dict) and run.get("config_id") == config_id and run.get("dry_run") is False:
            total += [row.cost_usd for row in read_manifest(run_file.parent / "manifest.jsonl")
                      if generator is None or row.generator_version == generator]
    return math.fsum(total)


def call_bound(request_chars: int, max_tokens: int, prices: dict[str, Any]) -> float:
    """一次呼叫的最壞成本：1.1 ×（prompt 估計 × 最貴的 input 類費率 + max_tokens × 最貴的 output 費率）/ 1e6。"""
    cards = [prices, *([prices["long_context"]] if "long_context" in prices else [])]
    input_rate = max(max(card["input"], card["cache_write"], card["cache_write_1h"]) for card in cards)
    output_rate = max(card["output"] for card in cards)
    prompt = math.ceil(request_chars / CHARS_PER_TOKEN_BOUND)
    return PRICE_MULTIPLIER_BOUND * (prompt * input_rate + max_tokens * output_rate) / 1e6


@dataclass
class PaidSession:
    """在程序鎖內建立（``paid_session``）；``active`` 只在鎖內為 True。自己記帳：已結算的成本與無法計價的呼叫。"""

    path: Path
    phase: str
    run: str
    spent_before: float
    cap: float
    total: float  # 保險絲的總預算（spec.total_usd）
    active: bool = False
    used: bool = False  # 一個 session 只給一次 _run_split / _run_sweeps
    settled: float = 0.0
    unknown: int = 0
    _bounds: dict[str, float] = field(default_factory=dict, repr=False)
    _issued: Any = field(default=None, repr=False)

    def reserve(self, *, request_chars: int, max_tokens: int, prices: dict[str, Any]) -> str:
        """呼叫之前：保險絲檢查，再寫一行預留；回傳預留的 id。"""
        if not self.active:
            raise RuntimeError("the paid session is not active (the billing lock is not held)")
        if self.unknown:
            raise BudgetFuse("a billed call of this run has an unknown cost; no further calls (decision 014 / 035 §3)")
        bound = call_bound(request_chars, max_tokens, prices)
        if self.spent_before + self.settled + bound > self.total:
            raise BudgetFuse(f"the next call may cost up to ${bound:.4f}; spent ${self.spent_before + self.settled:.4f} of "
                             f"${self.total:g}")
        rid = uuid.uuid4().hex
        _write(self.path, {"kind": "reserve", "id": rid, "utc_time": _now(), "phase": self.phase, "run": self.run,
                           "bound_usd": bound})
        self._bounds[rid] = bound
        return rid

    def settle(self, rid: str, cost: float | None) -> None:
        bound = self._bounds.pop(rid)
        if cost is None:
            self.unknown += 1
            self.settled += bound  # 保守地以上限計入，之後的呼叫一律停止
        else:
            self.settled += cost
        _write(self.path, {"kind": "settle", "id": rid, "utc_time": _now(), "cost_usd": cost})

    def settle_at_bound(self, rid: str) -> None:
        """沒有 response、但伺服器可能已經計費（連線中斷、逾時）：以上限結算。"""
        bound = self._bounds[rid]
        self.settle(rid, bound)

    def void(self, rid: str) -> None:
        self._bounds.pop(rid)
        _write(self.path, {"kind": "void", "id": rid, "utc_time": _now()})


def maybe_billed(exc: BaseException) -> bool:
    """沒有 response 的 Anthropic SDK 例外中，伺服器可能已經處理並計費的：連線中斷與逾時（沒有 HTTP status）。
    帶 HTTP status 的錯誤回應（4xx / 5xx）不計費。"""
    return type(exc).__module__.startswith("anthropic") and not isinstance(getattr(exc, "status_code", None), int)


def check_session(session: Any, *, spec: PaidSpec, phase: str, max_cost_usd: Any) -> None:
    """_run_split / _run_sweeps 的付費入口：session 必須是 ``paid_session`` 發出、仍 active、這個 phase、沒用過，總帳是這個研究的
    總帳、總預算是這個研究的總預算，max_cost_usd 不超過它的上限；通過後標記為已使用。否則 ValueError。"""
    ok = (isinstance(session, PaidSession) and session._issued is _ISSUED and session.active and not session.used
          and session.phase == phase and session.path == journal_path(spec) and session.total == spec.total_usd
          and isinstance(max_cost_usd, (int, float)) and not isinstance(max_cost_usd, bool) and max_cost_usd <= session.cap)
    if not ok:
        raise ValueError(f"a paid {spec.name} run runs inside its own billing.paid_session (decision {spec.decision})")
    session.used = True


def paid_cap(spent: Spent, *, spec: PaidSpec, phase: str, requested: float) -> float:
    """一般停止的上限 = min(requested, 總預算 − 安全邊際 − 已花, [phase 有上限時] 該上限 − 該 phase 已花)。
    用完或有 unknown → ValueError。"""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of: {', '.join(PHASES)}")
    if spent.unknown:
        raise ValueError(f"the billing journal has {spent.unknown} billed call(s) of unknown cost; the budget cannot be checked "
                         "(decision 014)")
    by_phase = {"dev": spent.dev, "baseline": spent.baseline, "sweep": spent.sweep}
    limits = [requested, spec.total_usd - SAFETY_MARGIN_USD - spent.total]
    if phase in spec.phase_caps:
        limits.append(spec.phase_caps[phase] - by_phase[phase])
    cap = min(limits)
    if cap <= 0:
        caps = ", ".join(f"{name} ${by_phase[name]:.4f}" + (f" of ${spec.phase_caps[name]:g}" if name in spec.phase_caps else "")
                         for name in PHASES)
        raise ValueError(
            f"the paid {spec.name} budget is spent: ${spent.total:.4f} of ${spec.total_usd:g} in total ({caps}; "
            f"safety margin ${SAFETY_MARGIN_USD:g}; this {phase} run); decision {spec.decision}"
        )
    return cap


def check_budget(*, spec: PaidSpec, phase: str, requested: float, config_id: str) -> tuple[Spent, float]:
    """CLI 的事前檢查（不取鎖）：(總帳合計, 這一次的上限)。總帳被重設（這個研究的 manifest 合計大於總帳）→ ValueError。"""
    path = journal_path(spec)
    spent = journal_spent(path)
    floor = manifest_floor(config_id, generator=spec.generator)
    if floor > spent.total + 1e-9:
        raise ValueError(f"the runs/ manifests of this agent in {spec.name} record ${floor:.4f} but the billing journal {path} "
                         f"only ${spent.total:.4f}: the journal was reset or replaced; stop and check (decision {spec.decision})")
    return spent, paid_cap(spent, spec=spec, phase=phase, requested=requested)


@contextlib.contextmanager
def paid_lock(path: str | Path) -> Iterator[None]:
    """同一份總帳同時只能有一個付費程序；拿不到 → ValueError（不等待、不搶占）。路徑先 resolve，同一檔案的不同寫法共用一個鎖。"""
    lock_path = Path(f"{Path(path).expanduser().resolve()}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another paid process holds {lock_path}; run one at a time (decision 035 §3)") from None
        yield
    finally:
        handle.close()  # 關閉即釋放


@contextlib.contextmanager
def paid_session(*, spec: PaidSpec, phase: str, requested: float, run: str, config_id: str) -> Iterator[PaidSession]:
    """取得這個研究的總帳的程序鎖，在鎖內以 ``check_budget`` 計算上限，回傳 active 的 PaidSession；離開時釋放鎖、session 失效。
    總帳一律是 ``journal_path(spec)``（呼叫端不能指定路徑）。"""
    path = journal_path(spec)
    with paid_lock(path):
        spent, cap = check_budget(spec=spec, phase=phase, requested=requested, config_id=config_id)
        session = PaidSession(path=path, phase=phase, run=run, spent_before=spent.total, cap=cap, total=spec.total_usd,
                              active=True, _issued=_ISSUED)
        try:
            yield session
        finally:
            session.active = False


def check_run_dir(run_dir: str | Path) -> None:
    """付費 run 的目錄必須是 ``RUNS_ROOT`` 的直接子目錄（總帳的核對看得到它；decision 035 §3）。否則 ValueError。"""
    resolved = Path(run_dir).resolve()
    if resolved.parent != Path(RUNS_ROOT).resolve():
        raise ValueError(f"a paid run directory must be directly under {RUNS_ROOT} (run from the repository root, "
                         "--run-name without '/'; decision 035 §3)")
