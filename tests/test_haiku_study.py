"""decision 035：Claude Haiku 5.5 跑 Study 2 的 eval（付費 API）的契約測試。

不連網、不呼叫任何模型；總帳（``billing.DEFAULT_JOURNAL``）與核對用的 ``billing.RUNS_ROOT`` 一律換成暫存目錄，不碰使用者的總帳與 repo 的 runs/。
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from agents.baseline import (
    BASELINE_MODEL_CONFIG,
    HAIKU_MODEL,
    PRICING,
    baseline_agent_config,
    haiku_agent_config,
    parse_response,
)
from experiments import billing, run_baseline, run_sweep
from experiments import study2_report as rep
from experiments.run_baseline import OracleFakeClient, _run_split, run_split
from experiments.run_sweep import _run_sweeps, run_sweeps, select_sources
from replay.aborts import read_manifest
from replay.localization import SweepPlan
from tasks import suites
from tasks.suites import suite_pool
from tests.test_baseline_adapter import FakeUsage, _resp, text_block

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOURNAL_AT_IMPORT = billing.DEFAULT_JOURNAL  # 任何 monkeypatch 之前的預設值
EVAL = suite_pool("study2", "eval")
POOL_IDS = [i.task.task_id for i in EVAL]
SMALL = SweepPlan(coarse_stride=4, n_coarse=1, n_fine=1, drop_threshold=0.3)
FAILING = {"gen2-scoped-triage-L3-s4": "S4:no-assign", "gen2-verified-close-L2-s3": "V2:no-comment"}


def ZERO() -> float:  # noqa: N802
    return 0.0


class PaidLikeClient:
    """不是 oracle 假 client 的 client（走真正推論的檢查路徑），回覆仍由 OracleFakeClient 產生：不連網、不計費。
    usage 固定 input 1000 / output 200 → Haiku 每次呼叫 $0.0002。"""

    def __init__(self, instances) -> None:
        self.inner = OracleFakeClient(instances)
        self.beta = self.inner.beta


@contextlib.contextmanager
def journal_at(directory: Path, runs: Path | None = None):
    """總帳換到 directory、付費 run 的 runs/ 換到 runs（預設 directory/runs）；module 範圍的 fixture 也能用。"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(billing, "DEFAULT_JOURNAL", directory / "billing.jsonl")
        mp.setattr(billing, "RUNS_ROOT", runs or directory / "runs")
        yield directory / "billing.jsonl"


@pytest.fixture(scope="module", autouse=True)
def reopen_paid_study2():
    """decision 035 已結案、Study 2 的付費入口預設關閉（decision 036 §7；見 tests/test_study_registry.py）。本檔測試的是同一套
    付費機制（Study 3 共用），所以在本檔中重新打開。"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(suites, "STUDY2_PAID_CLOSED", False)
        yield


@pytest.fixture
def journal(tmp_path: Path):
    """總帳在 tmp_path/journal；付費 run 的 runs/ 是 tmp_path/runs（CLI 測試的工作目錄是 tmp_path）。"""
    with journal_at(tmp_path / "journal", runs=tmp_path / "runs") as path:
        yield path


def runs(name: str) -> Path:
    return billing.RUNS_ROOT / name


def _pool_with_failures() -> tuple:
    by_id = {i.task.task_id: i for i in EVAL}
    return tuple(dataclasses.replace(i, oracle_calls=dict(by_id[i.task.task_id].negative_calls)[FAILING[i.task.task_id]])
                 if i.task.task_id in FAILING else i for i in EVAL)


def _paid_baseline(run_dir: Path, **kwargs):
    """run_dir 必須在 billing.RUNS_ROOT 之下（decision 035 §3）。"""
    return run_split("eval", PaidLikeClient(_pool_with_failures()), run_dir=run_dir, clock=ZERO, suite="study2",
                     agent_config=haiku_agent_config(), rollouts=3, **{"max_cost_usd": 15.0, **kwargs})


def _paid_sweep(sources, run_dir: Path, **kwargs):
    return run_sweeps(sources, PaidLikeClient(_pool_with_failures()), plan=SMALL, criterion="bonferroni95", run_dir=run_dir,
                      clock=ZERO, agent_config=haiku_agent_config(), **{"max_cost_usd": 70.0, "order": "failed_first", **kwargs})


def _spend(path: Path, entries: list[tuple[str, float | None]]) -> None:
    """總帳中先前的花費：每筆一個預留與結算。"""
    for i, (phase, cost) in enumerate(entries):
        rid = f"earlier-{path.stat().st_size if path.exists() else 0}-{i}"
        billing._write(path, {"kind": "reserve", "id": rid, "utc_time": "t", "phase": phase, "run": "earlier",
                              "bound_usd": 0.0 if cost is None else cost})
        billing._write(path, {"kind": "settle", "id": rid, "utc_time": "t", "cost_usd": cost})


# ---------------------------------------------------------------- agent 設定與計價（decision 035 §2 / §3）


def test_the_haiku_agent_differs_from_decision_009_only_in_the_model() -> None:
    haiku, opus = haiku_agent_config(), baseline_agent_config()
    assert HAIKU_MODEL == "claude-haiku-5-5" and haiku.model_config["model"] == HAIKU_MODEL
    expected = copy.deepcopy(BASELINE_MODEL_CONFIG)
    expected["model"] = HAIKU_MODEL
    assert haiku.model_config == expected
    assert (haiku.policy, haiku.prompt_version, haiku.system_prompt, haiku.developer_prompt) == (
        opus.policy, opus.prompt_version, opus.system_prompt, opus.developer_prompt)
    assert haiku.config_id != opus.config_id
    assert haiku.model_config["effort"] == "medium" and haiku.model_config["fallbacks"] is None
    assert haiku.model_config["thinking"]["type"] == "adaptive"
    for key in ("temperature", "top_p", "top_k"):
        assert key not in haiku.model_config
    assert haiku_agent_config().model_config is not haiku.model_config  # 每次回傳新物件


def test_haiku_pricing_matches_the_official_page_of_2026_10_22() -> None:
    p = PRICING[HAIKU_MODEL]
    assert {k: p[k] for k in ("input", "output", "cache_write", "cache_write_1h", "cache_read")} == {
        "input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_write_1h": 0.20, "cache_read": 0.01}
    assert p["long_context"] == {"threshold": 100_000, "input": 0.50, "output": 2.50, "cache_write": 0.625,
                                 "cache_write_1h": 1.00, "cache_read": 0.05}
    assert "long_context" not in PRICING["claude-opus-5-5"]  # Opus 5.5 的 1M context 都是同一個費率


def _cost(input_tokens: int, cache_read: int = 0, output: int = 1000, **extra) -> float | None:
    usage = FakeUsage(input_tokens=input_tokens, output_tokens=output, cache_creation_input_tokens=0,
                      cache_read_input_tokens=cache_read)
    for name, value in extra.items():
        setattr(usage, name, value)
    return parse_response(_resp([text_block("x")], "end_turn", usage, model=HAIKU_MODEL), HAIKU_MODEL).cost_usd


def test_haiku_cost_uses_the_long_context_card_only_above_100k_prompt_tokens() -> None:
    # prompt = 100,000（含 cache 讀取）：一般費率；100,001：整個 request 改用長 prompt 的費率（含 cache 讀取與 output）
    assert _cost(20_000, 80_000) == pytest.approx((20_000 * 0.10 + 80_000 * 0.01 + 1000 * 0.50) / 1e6, rel=1e-12)
    assert _cost(20_001, 80_000) == pytest.approx((20_001 * 0.50 + 80_000 * 0.05 + 1000 * 2.50) / 1e6, rel=1e-12)


def test_billing_modifiers_reported_in_usage_change_or_void_the_price() -> None:
    """帳戶層級的設定可能讓同樣的 tokens 更貴：US-only 推論 × 1.1；其他無法確認的等級 → 成本未知（decision 014：停止）。"""
    base = _cost(10_000)
    assert base == pytest.approx((10_000 * 0.10 + 1000 * 0.50) / 1e6, rel=1e-12)
    for extra in ({}, {"inference_geo": "global"}, {"service_tier": "standard"}, {"speed": "standard"}, {"inference_geo": None}):
        assert _cost(10_000, **extra) == pytest.approx(base, rel=1e-12)
    assert _cost(10_000, inference_geo="us") == pytest.approx(base * 1.1, rel=1e-12)
    for extra in ({"inference_geo": "eu"}, {"service_tier": "priority"}, {"service_tier": "batch"}, {"speed": "fast"}):
        assert _cost(10_000, **extra) is None


# ---------------------------------------------------------------- 總帳、上限、保險絲與程序鎖（decision 035 §3）


def test_the_journal_is_outside_every_checkout() -> None:
    assert DEFAULT_JOURNAL_AT_IMPORT == Path.home() / ".agent-gym" / "study2-haiku-billing.jsonl"
    assert ROOT not in DEFAULT_JOURNAL_AT_IMPORT.parents


def test_reservations_count_until_settled_or_voided(journal: Path) -> None:
    session = billing.PaidSession(path=journal, phase="baseline", run="r", spent_before=0.0, cap=15.0, total=70.0, active=True)
    prices = PRICING[HAIKU_MODEL]
    a = session.reserve(request_chars=2000, max_tokens=16000, prices=prices)
    bound = billing.call_bound(2000, 16000, prices)
    assert billing.journal_spent(journal).baseline == pytest.approx(bound)  # 呼叫途中被中斷：以上限計
    session.settle(a, 0.002)
    b = session.reserve(request_chars=2000, max_tokens=16000, prices=prices)
    session.void(b)  # 沒有 response：不計費
    c = session.reserve(request_chars=2000, max_tokens=16000, prices=prices)
    session.settle(c, None)
    spent = billing.journal_spent(journal)
    assert spent.baseline == pytest.approx(0.002) and spent.unknown == 1 and spent.sweep == 0
    lines = journal.read_bytes().split(b"\n")[:-1]
    assert all(json.loads(line)["format"] == billing.JOURNAL_FORMAT for line in lines)
    session.active = False
    with pytest.raises(RuntimeError):
        session.reserve(request_chars=1, max_tokens=1, prices=prices)


def test_the_journal_fails_closed_on_damage(journal: Path) -> None:
    _spend(journal, [("baseline", 1.0)])
    with journal.open("ab") as handle:
        handle.write(b'{"format": "agent-gym/billing/v2", "kind": "reserve"')  # 寫到一半
    with pytest.raises(ValueError, match="incomplete"):
        billing.journal_spent(journal)
    other = journal.parent / "other.jsonl"
    billing._write(other, {"kind": "settle", "id": "never-reserved", "utc_time": "t", "cost_usd": 1.0})
    with pytest.raises(ValueError):
        billing.journal_spent(other)


def test_the_fuse_stops_before_a_call_that_could_cross_the_total(journal: Path) -> None:
    prices = PRICING[HAIKU_MODEL]
    bound = billing.call_bound(400_000, 16000, prices)
    assert bound > 0.25  # 長 prompt、最大輸出、1.1 倍：一次呼叫的最壞成本
    session = billing.PaidSession(path=journal, phase="sweep", run="r", spent_before=69.9, cap=0.1, total=70.0, active=True)
    with pytest.raises(billing.BudgetFuse):
        session.reserve(request_chars=400_000, max_tokens=16000, prices=prices)
    assert not journal.exists()  # 沒有預留
    session.reserve(request_chars=1000, max_tokens=100, prices=prices)


def test_paid_caps_are_cumulative(journal: Path) -> None:
    cid = haiku_agent_config().config_id
    assert billing.check_budget(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, config_id=cid)[1] == pytest.approx(15.0)
    assert billing.check_budget(spec=suites.STUDY2_PAID, phase="sweep", requested=70.0, config_id=cid)[1] == pytest.approx(70.0 - billing.SAFETY_MARGIN_USD)
    _spend(journal, [("baseline", 10.0)])
    assert billing.check_budget(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, config_id=cid)[1] == pytest.approx(5.0)
    assert billing.check_budget(spec=suites.STUDY2_PAID, phase="sweep", requested=70.0, config_id=cid)[1] == pytest.approx(60.0 - billing.SAFETY_MARGIN_USD)
    _spend(journal, [("baseline", 5.0)])
    with pytest.raises(ValueError, match="baseline"):
        billing.check_budget(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, config_id=cid)
    _spend(journal, [("sweep", 54.1)])
    with pytest.raises(ValueError, match="70"):
        billing.check_budget(spec=suites.STUDY2_PAID, phase="sweep", requested=70.0, config_id=cid)


def test_unknown_costs_and_a_reset_journal_are_refused(journal: Path, tmp_path: Path) -> None:
    cid = haiku_agent_config().config_id
    _spend(journal, [("sweep", None)])
    with pytest.raises(ValueError, match="unknown"):
        billing.check_budget(spec=suites.STUDY2_PAID, phase="sweep", requested=70.0, config_id=cid)
    journal.unlink()  # 總帳被刪除或換了檔案，但 runs/ 中有這個 agent 的花費
    _paid_baseline_dir = billing.RUNS_ROOT / "h"
    _paid_baseline_dir.mkdir(parents=True)
    (_paid_baseline_dir / "run.json").write_text(json.dumps({"config_id": cid, "dry_run": False}), encoding="utf-8")
    shutil.copy(_haiku_manifest(tmp_path), _paid_baseline_dir / "manifest.jsonl")
    with pytest.raises(ValueError, match="reset"):
        billing.check_budget(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, config_id=cid)


def _haiku_manifest(tmp_path: Path) -> Path:
    with journal_at(tmp_path / "scratch"):
        _paid_baseline(runs("scratch-run"))
    return tmp_path / "scratch" / "runs" / "scratch-run" / "manifest.jsonl"


def test_only_one_paid_process_at_a_time(journal: Path) -> None:
    with billing.paid_lock(journal):
        with pytest.raises(ValueError, match="another"):
            with billing.paid_lock(journal.parent / "." / journal.name):  # 同一個檔案的另一種寫法
                pass
        with pytest.raises(ValueError, match="another"):
            _paid_baseline(runs("b"))
    with billing.paid_lock(journal):  # 釋放後可以再取得
        pass


# ---------------------------------------------------------------- 付費 Study 2 的檢查（decision 035 §1 / §2 / §3）


def test_paid_study2_limits() -> None:
    assert suites.STUDY2_PAID_MODEL == HAIKU_MODEL
    assert suites.STUDY2_PAID_TOTAL_USD == 70.0 and suites.STUDY2_PAID_BASELINE_USD == 15.0
    suites.check_paid_run(suites.STUDY2_PAID, model=HAIKU_MODEL, split="eval", max_cost_usd=15.0, sweep=False, rollouts=3)
    suites.check_paid_run(suites.STUDY2_PAID, model=HAIKU_MODEL, split="eval", max_cost_usd=70.0, sweep=True)
    bad = [
        dict(model="claude-opus-5-5", split="eval", max_cost_usd=15.0, sweep=False, rollouts=3),
        dict(model=HAIKU_MODEL, split="dev", max_cost_usd=1.0, sweep=False, rollouts=3),  # 不跑 dev（使用者決定）
        dict(model=HAIKU_MODEL, split="eval", max_cost_usd=None, sweep=False, rollouts=3),
        dict(model=HAIKU_MODEL, split="eval", max_cost_usd=15.01, sweep=False, rollouts=3),
        dict(model=HAIKU_MODEL, split="eval", max_cost_usd=15.0, sweep=False, rollouts=1),
        dict(model=HAIKU_MODEL, split="eval", max_cost_usd=70.01, sweep=True),
        dict(model=HAIKU_MODEL, split="eval", max_cost_usd=0.0, sweep=True),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            suites.check_paid_run(suites.STUDY2_PAID, **kwargs)


def test_run_split_checks_paid_runs_before_any_call(journal: Path, tmp_path: Path) -> None:
    client = PaidLikeClient(EVAL)
    other = haiku_agent_config()
    other = dataclasses.replace(other, model_config={**other.model_config, "max_tokens": 128000})  # 不是凍結的 Haiku agent
    for kwargs in (dict(agent_config=baseline_agent_config(), max_cost_usd=15.0, rollouts=3),  # Opus 不在 035
                   dict(agent_config=other, max_cost_usd=15.0, rollouts=3),
                   dict(agent_config=haiku_agent_config(), max_cost_usd=16.0, rollouts=3),
                   dict(agent_config=haiku_agent_config(), max_cost_usd=15.0, rollouts=1),
                   dict(agent_config=haiku_agent_config(), max_cost_usd=None, rollouts=3)):
        with pytest.raises(ValueError):
            run_split("eval", client, run_dir=runs("x"), clock=ZERO, suite="study2", **kwargs)
    with pytest.raises(ValueError):
        run_split("dev", PaidLikeClient(suite_pool("study2", "dev")), run_dir=tmp_path / "d", clock=ZERO, suite="study2",
                  agent_config=haiku_agent_config(), max_cost_usd=1.0, rollouts=3)
    with pytest.raises(ValueError):  # 內部函式也不能繞過：沒有 active 的 session
        _run_split("eval", client, run_dir=runs("x"), clock=ZERO, suite="study2", agent_config=haiku_agent_config(),
                   max_cost_usd=15.0, rollouts=3)
    week1 = PaidLikeClient(suite_pool("week1", "eval"))
    for runner in (run_split, _run_split):  # Haiku 只登記給 Study 2
        with pytest.raises(ValueError):
            runner("eval", week1, run_dir=tmp_path / "w", clock=ZERO, rollouts=3, agent_config=haiku_agent_config(), max_cost_usd=5.0)
    assert client.inner.requests == [] and week1.inner.requests == [] and not journal.exists()
    assert not runs("x").exists() and not (tmp_path / "w").exists()


def test_a_paid_baseline_reserves_and_settles_every_call(journal: Path, tmp_path: Path) -> None:
    summary = _paid_baseline(runs("b"))
    assert len(summary.results) == 108 and summary.unknown_cost_calls == 0
    spent = billing.journal_spent(journal)
    assert spent.baseline == pytest.approx(summary.spent_usd) and spent.sweep == 0 and spent.unknown == 0
    kinds = [json.loads(line)["kind"] for line in journal.read_bytes().split(b"\n")[:-1]]
    assert kinds.count("reserve") == kinds.count("settle") == summary.model_calls > 0


def test_the_baseline_cap_counts_earlier_baseline_attempts(journal: Path, tmp_path: Path) -> None:
    _spend(journal, [("baseline", 15.0 - 0.001)])  # 先前的嘗試幾乎用完 15 美元
    summary = _paid_baseline(runs("b"))
    assert summary.stopped_by_budget and len(summary.results) < 108
    assert billing.journal_spent(journal).baseline <= 15.0 + 0.01  # 最多超出一個進行中的 episode
    _spend(journal, [("baseline", 1.0)])
    with pytest.raises(ValueError):
        _paid_baseline(runs("c"))


# ---------------------------------------------------------------- run_baseline 的 CLI（decision 035）


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch, journal: Path) -> list:
    """make_anthropic_client 換成不連網的 PaidLikeClient。建立之前就該拒絕的情況，made 必須是空的。"""
    created: list = []

    def fake(model_config=None):
        created.append(None if model_config is None else model_config["model"])
        return PaidLikeClient(EVAL)

    monkeypatch.setattr(run_baseline, "make_anthropic_client", fake)
    return created


PAID = ["--split", "eval", "--suite", "study2", "--provider", "anthropic", "--rollouts", "3", "--confirm-paid-run"]


def test_the_cli_runs_a_paid_haiku_study2_eval_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list,
                                                        journal: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_baseline.main([*PAID, "--model", HAIKU_MODEL, "--max-cost-usd", "15", "--run-name", "h"]) == 0
    assert made == [HAIKU_MODEL]
    summary = json.loads((tmp_path / "runs" / "h" / "summary.json").read_text(encoding="utf-8"))
    assert summary["config_id"] == haiku_agent_config().config_id and len(summary["results"]) == 108
    assert billing.journal_spent(journal).baseline == pytest.approx(summary["spent_usd"])


@pytest.mark.parametrize(
    "args",
    [
        [*PAID, "--max-cost-usd", "15"],  # --model 預設為 decision 009 的 Opus：Study 2 的付費 run 只登記 Haiku
        [*PAID, "--model", "claude-opus-5-5", "--max-cost-usd", "15"],
        [*PAID, "--model", HAIKU_MODEL, "--max-cost-usd", "16"],
        [*PAID, "--model", HAIKU_MODEL],
        ["--split", "eval", "--suite", "study2", "--provider", "anthropic", "--rollouts", "1", "--confirm-paid-run",
         "--model", HAIKU_MODEL, "--max-cost-usd", "15"],
        ["--split", "dev", "--suite", "study2", "--provider", "anthropic", "--rollouts", "3", "--confirm-paid-run",
         "--model", HAIKU_MODEL, "--max-cost-usd", "1"],
        ["--split", "eval", "--provider", "anthropic", "--rollouts", "3", "--confirm-paid-run", "--model", HAIKU_MODEL,
         "--max-cost-usd", "5"],  # 第一輪（week1）的付費 run 只登記 Opus（decision 009 / 025）
        ["--split", "dev", "--suite", "study2", "--dry-run", "--model", HAIKU_MODEL],  # --model 只給 anthropic
    ],
)
def test_the_cli_refuses_unregistered_paid_runs_before_creating_a_client(args: list, tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch, made: list) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_baseline.main([*args, "--run-name", "x"])
    assert made == [] and not (tmp_path / "runs" / "x").exists()


def test_the_cli_refuses_when_the_baseline_budget_is_spent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list,
                                                           journal: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _spend(journal, [("baseline", 15.0)])
    with pytest.raises(SystemExit):
        run_baseline.main([*PAID, "--model", HAIKU_MODEL, "--max-cost-usd", "15", "--run-name", "x"])
    assert made == [] and not (tmp_path / "runs" / "x").exists()


def test_a_haiku_dry_run_needs_no_client_and_no_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list,
                                                        journal: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_baseline.main(["--split", "eval", "--suite", "study2", "--provider", "anthropic", "--model", HAIKU_MODEL,
                              "--dry-run", "--run-name", "d"]) == 0
    assert made == [] and not journal.exists()
    run = json.loads((tmp_path / "runs" / "d" / "run.json").read_text(encoding="utf-8"))
    assert run["dry_run"] is True and run["config_id"] == haiku_agent_config().config_id


# ---------------------------------------------------------------- sweep（decision 035 §3 / §4）


@pytest.fixture(scope="module")
def haiku_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hb")
    with journal_at(root):
        _paid_baseline(root / "runs" / "b")
    return root / "runs" / "b"


def test_a_paid_study2_sweep_runs_failed_sources_first(haiku_baseline: Path, tmp_path: Path, journal: Path) -> None:
    sources = select_sources(haiku_baseline)
    assert sorted(t for t, s in sources.items() if s is not None and not s.grade().success) == sorted(FAILING)
    summary = _paid_sweep(sources, runs("s"))
    order = list(dict.fromkeys(row.task_id for row in read_manifest(runs("s") / "manifest.jsonl")))
    assert order[:2] == [t for t in POOL_IDS if t in FAILING] and order[2:] == [t for t in POOL_IDS if t not in FAILING]
    assert not summary.stopped_by_budget
    assert billing.journal_spent(journal).sweep == pytest.approx(summary.spent_usd)


def test_paid_study2_sweeps_are_checked_in_run_sweeps(haiku_baseline: Path, tmp_path: Path, journal: Path) -> None:
    sources = select_sources(haiku_baseline)
    for kwargs in (dict(order="pool"),  # 付費的 Study 2 必須先跑失敗的 source
                   dict(max_cost_usd=70.01),
                   dict(max_cost_usd=None),
                   dict(order="random")):
        with pytest.raises(ValueError):
            _paid_sweep(sources, runs("x"), **kwargs)
    with pytest.raises(ValueError):  # 內部函式也不能繞過：沒有 active 的 session
        _run_sweeps(sources, PaidLikeClient(EVAL), plan=SMALL, criterion="bonferroni95", run_dir=runs("x"), clock=ZERO,
                    agent_config=haiku_agent_config(), max_cost_usd=70.0, order="failed_first")
    with pytest.raises(ValueError):  # failed_first 只給付費的 Study 2
        run_sweeps(sources, OracleFakeClient(EVAL), plan=SMALL, criterion="bonferroni95", run_dir=tmp_path / "x", clock=ZERO,
                   agent_config=baseline_agent_config(), order="failed_first", dry_run=True)
    with pytest.raises(ValueError):  # 非 Mapping 的 sources 仍是原本的 ValueError
        run_sweeps(["a"], OracleFakeClient(()), plan=SMALL, criterion="bonferroni95", run_dir=tmp_path / "x")
    assert not runs("x").exists() and not journal.exists()


def test_the_sweep_stops_at_the_total_budget_including_the_baseline(haiku_baseline: Path, tmp_path: Path, journal: Path) -> None:
    _spend(journal, [("baseline", 8.0), ("sweep", 70.0 - billing.SAFETY_MARGIN_USD - 8.0 - 0.003)])
    summary = _paid_sweep(select_sources(haiku_baseline), runs("s"))
    assert summary.stopped_by_budget and 0 < len(summary.completed) < 36
    assert billing.journal_spent(journal).total <= 70.0
    with pytest.raises(ValueError):
        _paid_sweep(select_sources(haiku_baseline), runs("t"))


def test_a_failed_first_sweep_stopped_by_budget_passes_the_strict_checks(haiku_baseline: Path, tmp_path: Path,
                                                                        journal: Path) -> None:
    """失敗優先的順序在預算用完時，completed 是「失敗優先順序」的前綴；嚴格核對（decision 017 / 018）必須接受。"""
    from experiments.analyze_sweep import analyze_sweep_runs

    summary = _paid_sweep(select_sources(haiku_baseline), runs("s"), max_cost_usd=0.004)
    assert summary.stopped_by_budget and 0 < len(summary.completed) < 36
    assert list(summary.completed[: len(FAILING)]) == [t for t in POOL_IDS if t in FAILING][: len(summary.completed)]
    assert analyze_sweep_runs([runs("s")], [haiku_baseline], replicates=10) is not None


def test_failed_first_order_is_accepted_only_for_paid_study2_sources(haiku_baseline: Path) -> None:
    from experiments.analyze_sweep import allowed_orders

    assert allowed_orders(select_sources(haiku_baseline)) == ("pool", "failed_first")
    assert allowed_orders({i.task.task_id: None for i in suite_pool("week1", "eval")}) == ("pool",)
    assert allowed_orders({t: None for t in POOL_IDS}) == ("pool",)


SWEEP = ["--plan", "2,10,48,0.3", "--criterion", "bonferroni95", "--confirm-paid-run"]


@pytest.fixture
def sweep_cli(haiku_baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, journal: Path) -> dict:
    """CLI 的檢查與上限計算；run_sweeps 換成記錄參數後停止（完整的 registered plan 在測試中太慢），make_anthropic_client 換成假 client。"""
    (tmp_path / "runs").mkdir()
    shutil.copytree(haiku_baseline, tmp_path / "runs" / "b")
    _spend(journal, [("baseline", sum(r.cost_usd for r in read_manifest(tmp_path / "runs" / "b" / "manifest.jsonl")))])
    monkeypatch.chdir(tmp_path)
    seen: dict = {"made": [], "calls": []}
    monkeypatch.setattr(run_sweep, "make_anthropic_client",
                        lambda model_config=None: seen["made"].append(model_config["model"]) or PaidLikeClient(EVAL))

    def fake_run_sweeps(sources, client, **kwargs):
        seen["calls"].append(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(run_sweep, "run_sweeps", fake_run_sweeps)
    return seen


def test_the_paid_sweep_cli_checks_the_journal_and_runs_failed_first(sweep_cli: dict, journal: Path,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    _spend(journal, [("baseline", 12.5)])
    expected = 70.0 - billing.SAFETY_MARGIN_USD - billing.journal_spent(journal).total
    with pytest.raises(SystemExit):
        run_sweep.main(["--baseline-run", "runs/b", *SWEEP, "--max-cost-usd", "70", "--total-cost-usd", "70", "--run-name", "s1"])
    assert sweep_cli["made"] == [HAIKU_MODEL]
    kwargs = sweep_cli["calls"][0]
    assert kwargs["max_cost_usd"] == pytest.approx(70.0) and kwargs["order"] == "failed_first"
    assert f"this run may spend ${expected:.4f}" in capsys.readouterr().err  # 70 − 1 的安全邊際 − 總帳


@pytest.mark.parametrize(
    ("extra", "entries"),
    [
        (["--max-cost-usd", "70"], []),  # 缺 --total-cost-usd
        (["--max-cost-usd", "70", "--total-cost-usd", "80"], []),  # 總上限必須是登記的 70
        (["--max-cost-usd", "70", "--total-cost-usd", "70"], [("baseline", 15.0), ("sweep", 54.1)]),  # 已用完
        (["--max-cost-usd", "70", "--total-cost-usd", "70"], [("sweep", None)]),  # 有無法計價的呼叫
    ],
)
def test_the_paid_sweep_cli_refuses_before_creating_a_client(sweep_cli: dict, journal: Path, extra: list, entries: list) -> None:
    _spend(journal, entries)
    with pytest.raises(SystemExit) as exc:
        run_sweep.main(["--baseline-run", "runs/b", *SWEEP, *extra, "--run-name", "s1"])
    assert exc.value.code != 0 and sweep_cli["made"] == [] and sweep_cli["calls"] == []


def test_no_paid_sweep_when_every_adopted_episode_succeeded(tmp_path: Path, sweep_cli: dict, capsys: pytest.CaptureFixture[str]) -> None:
    """採計的 episode 全部成功 → 天花板，不跑 sweep（decision 034 §8 / 035 §4）；中斷後重跑的目錄也一樣。Study 2 的訊息不變
    （Study 3 的天花板另依 036 §6 判定，見 tests/test_study3_paid_cli.py）。"""
    for name in ("c0", "c"):  # 同一份總帳（sweep_cli 的 journal fixture）
        run_split("eval", PaidLikeClient(EVAL), run_dir=tmp_path / "runs" / name, clock=ZERO, suite="study2",
                  agent_config=haiku_agent_config(), rollouts=3, max_cost_usd=7.0)
    (tmp_path / "runs" / "c0" / "summary.json").unlink()  # 中斷的第一次（沒有 summary）
    for dirs in (["runs/c"], ["runs/c0", "runs/c"]):
        with pytest.raises(SystemExit) as exc:
            run_sweep.main(["--baseline-run", *dirs, *SWEEP, "--max-cost-usd", "70", "--total-cost-usd", "70", "--run-name", "s1"])
        assert exc.value.code != 0
        assert "every adopted eval baseline episode succeeded: ceiling, no sweep (decision 034 §8 / 035 §4)" in capsys.readouterr().err
    assert sweep_cli["made"] == []


def test_a_stray_total_cost_flag_is_refused_before_any_side_effect(sweep_cli: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_sweep, "start_local_run", lambda *a, **k: pytest.fail("must refuse before starting a local run"))
    for mode in (["--dry-run"], ["--confirm-local-run", "--max-model-calls", "3000", "--total-model-calls", "70000"]):
        with pytest.raises(SystemExit) as exc:
            run_sweep.main(["--baseline-run", "runs/b", "--plan", "2,10,48,0.3", "--criterion", "bonferroni95", *mode,
                            "--total-cost-usd", "70", "--run-name", "s1"])
        assert exc.value.code != 0
    assert sweep_cli["calls"] == []


# ---------------------------------------------------------------- 報告（decision 035 §6）


@pytest.fixture(scope="module")
def haiku_runs(haiku_baseline: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("hs")
    with journal_at(root):
        _paid_sweep(select_sources(haiku_baseline), root / "runs" / "s")
    return haiku_baseline, root / "runs" / "s"


def test_the_haiku_report_has_its_own_block_title_and_costs(haiku_runs: tuple[Path, Path]) -> None:
    baseline, sweep_dir = haiku_runs
    summary = rep.build([baseline], [sweep_dir], replicates=200)
    prov = summary["provenance"]
    assert prov["model"] == HAIKU_MODEL and prov["provider"] == "anthropic"
    assert prov["spent_usd"]["baseline"] == pytest.approx(sum(r.cost_usd for r in read_manifest(baseline / "manifest.jsonl")))
    assert prov["spent_usd"]["sweep"] == pytest.approx(sum(r.cost_usd for r in read_manifest(sweep_dir / "manifest.jsonl")))
    assert rep.variant(summary) == "study2-haiku"
    page = rep.render_page(summary)
    assert page.startswith("# Study 2 跨模型比較：Claude Haiku 5.5（decision 035")
    assert "Anthropic API" in page and "總帳" in page and "Ollama" not in page
    assert "Claude Haiku 5.5" in rep.recoverability_svg(summary)
    summary["provenance"]["dry_run"] = False  # 只測 README 段落的形狀
    block = rep.render_readme_block(summary)
    assert block.startswith("<!-- study2-haiku:start -->") and block.endswith("<!-- study2-haiku:end -->")
    assert "docs/results/study2-haiku/README.md" in block and "decision 035" in block
    readme = "# x\n<!-- study2:start -->\nlocal\n<!-- study2:end -->\n"
    embedded = rep.embed(readme, block)
    assert "<!-- study2:start -->\nlocal\n<!-- study2:end -->" in embedded and embedded.count("study2-haiku:start") == 1
    assert rep.embed(embedded, block) == embedded


def test_a_haiku_report_without_trajectories_still_belongs_to_haiku(haiku_baseline: Path, tmp_path: Path) -> None:
    """baseline 沒有任何完成的 trajectory 時，仍依 run.json 的設定歸給 Haiku，不會被當成本機 Study 2 的報告。"""
    copy_dir = tmp_path / "b"
    shutil.copytree(haiku_baseline, copy_dir)
    summary = rep._provenance([copy_dir], [], rep.analyze([copy_dir], [], replicates=10))
    shutil.rmtree(copy_dir / "trajectories")
    (copy_dir / "trajectories").mkdir()
    assert rep._run_config_id([copy_dir]) == haiku_agent_config().config_id
    assert rep.variant({"provenance": {**summary, "config_id": rep._run_config_id([copy_dir]), "model": None}}) == "study2-haiku"
    with pytest.raises(ValueError):
        rep.variant({"provenance": {"config_id": baseline_agent_config().config_id}})


def test_the_local_study2_report_is_unchanged() -> None:
    summary = json.loads((ROOT / "docs" / "results" / "study2" / "summary.json").read_text(encoding="utf-8"))
    assert rep.variant(summary) == "study2" and "provider" not in summary["provenance"]


def test_each_agent_report_goes_only_to_its_own_directory(haiku_runs: tuple[Path, Path], tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """結果目錄換成暫存目錄中的副本，所以即使檢查失效也不會寫到 repo 本身。"""
    local_dir, haiku_dir = tmp_path / "results" / "study2", tmp_path / "results" / "study2-haiku"
    monkeypatch.setattr(rep, "RESULT_DIRS", {"study2": local_dir, "study2-haiku": haiku_dir})
    baseline, sweep_dir = haiku_runs
    for out in (local_dir, local_dir / "haiku"):  # 另一個 agent 的目錄與其子目錄
        with pytest.raises(ValueError, match="study2-haiku"):
            rep.write_report([baseline], [sweep_dir], out, replicates=200)
    assert not local_dir.exists()
    rep.write_report([baseline], [sweep_dir], haiku_dir)  # 正式（非 dry-run）的報告使用登記的 B = 10,000
    assert (haiku_dir / "summary.json").exists()


# ---------------------------------------------------------------- 第三輪審查的修正（decision 035「付費前審查」）


def test_an_unknown_cost_stops_the_next_call(journal: Path) -> None:
    session = billing.PaidSession(path=journal, phase="sweep", run="r", spent_before=0.0, cap=69.0, total=70.0, active=True)
    prices = PRICING[HAIKU_MODEL]
    rid = session.reserve(request_chars=1000, max_tokens=1000, prices=prices)
    bound = billing.call_bound(1000, 1000, prices)
    session.settle(rid, None)
    assert session.unknown == 1 and session.settled == pytest.approx(bound)  # 以上限計入
    with pytest.raises(billing.BudgetFuse):
        session.reserve(request_chars=1000, max_tokens=1000, prices=prices)


def test_sessions_are_single_use_and_issued_by_paid_session(journal: Path) -> None:
    hand_made = billing.PaidSession(path=journal, phase="baseline", run="r", spent_before=0.0, cap=15.0, total=70.0, active=True)
    with pytest.raises(ValueError):
        billing.check_session(hand_made, spec=suites.STUDY2_PAID, phase="baseline", max_cost_usd=1.0)
    with billing.paid_session(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, run="r", config_id=haiku_agent_config().config_id) as session:
        with pytest.raises(ValueError):
            billing.check_session(session, spec=suites.STUDY2_PAID, phase="sweep", max_cost_usd=1.0)
        with pytest.raises(ValueError):
            billing.check_session(session, spec=suites.STUDY2_PAID, phase="baseline", max_cost_usd=15.5)
        billing.check_session(session, spec=suites.STUDY2_PAID, phase="baseline", max_cost_usd=15.0)
        with pytest.raises(ValueError):  # 已用過
            billing.check_session(session, spec=suites.STUDY2_PAID, phase="baseline", max_cost_usd=15.0)
    assert not session.active


def test_paid_runs_must_live_directly_under_runs(journal: Path, tmp_path: Path) -> None:
    billing.check_run_dir(runs("ok"))
    for bad in (tmp_path / "elsewhere", runs("nested") / "x"):
        with pytest.raises(ValueError):
            billing.check_run_dir(bad)
        with pytest.raises(ValueError):
            _paid_baseline(bad)
    assert not journal.exists()


def test_manifest_floor_scans_nested_runs(tmp_path: Path) -> None:
    cid = haiku_agent_config().config_id
    nested = tmp_path / "root" / "haiku" / "eval"
    nested.mkdir(parents=True)
    (nested / "run.json").write_text(json.dumps({"config_id": cid, "dry_run": False}), encoding="utf-8")
    shutil.copy(_haiku_manifest(tmp_path), nested / "manifest.jsonl")
    assert billing.manifest_floor(cid, tmp_path / "root") > 0
    assert billing.manifest_floor("sha256:other", tmp_path / "root") == 0


def test_the_fuse_in_the_baseline_leaves_an_interrupted_but_readable_directory(journal: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline 在 episode 中途觸發保險絲：不寫 summary（與中斷相同），目錄仍能由 manifest 讀取（decision 035 §3）。"""
    from experiments.analyze_baseline import load_runs

    real_bound = billing.call_bound
    calls = {"n": 0}

    def bound(*args, **kwargs):  # 前面的呼叫照常；第 40 次之後的呼叫最壞成本很高 → 保險絲在 episode 中途觸發
        calls["n"] += 1
        return real_bound(*args, **kwargs) if calls["n"] < 40 else 100.0

    monkeypatch.setattr(billing, "call_bound", bound)
    with pytest.raises(billing.BudgetFuse):
        _paid_baseline(runs("b"))
    assert not (runs("b") / "summary.json").exists()
    assert all(row.error_class != "harness" for row in read_manifest(runs("b") / "manifest.jsonl"))
    loaded = load_runs([runs("b")])
    assert loaded.interrupted_runs and billing.journal_spent(journal).total <= 70.0
    select_sources(runs("b"))  # sweep 的 source 選擇也能讀


def test_a_timeout_is_counted_at_its_bound(journal: Path) -> None:
    """沒有 response 的連線中斷或逾時：伺服器可能已計費 → 以上限結算；有 HTTP status 的錯誤回應 → 作廢。"""
    timeout = type("APITimeoutError", (Exception,), {"__module__": "anthropic._exceptions"})
    overloaded = type("OverloadedError", (Exception,), {"__module__": "anthropic._exceptions", "status_code": 529})
    assert billing.maybe_billed(timeout()) and not billing.maybe_billed(overloaded()) and not billing.maybe_billed(ValueError())

    class Flaky:
        def __init__(self, error) -> None:
            self.error = error
            self.inner = OracleFakeClient(EVAL)
            self.beta = self

        @property
        def messages(self):
            return self

        def create(self, **request):
            raise self.error

    from experiments.run_baseline import CostMeter, Ledger, make_policy

    with billing.paid_session(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, run="t", config_id=haiku_agent_config().config_id) as session:
        for error, expect_bound in ((timeout(), True), (overloaded(), False)):
            before = billing.journal_spent(journal).baseline
            meter = CostMeter(make_policy(Flaky(error), haiku_agent_config()), Ledger(session=session))
            instance = EVAL[0]
            messages = [{"role": "system", "content": "s"}, {"role": "developer", "content": "d"},
                        {"role": "user", "content": instance.task.instruction}]
            with pytest.raises(type(error)):
                meter.next_message(messages, ({"name": "t", "description": "d", "input_schema": {"type": "object"}},))
            added = billing.journal_spent(journal).baseline - before
            assert (added > 0) is expect_bound


def test_the_fuse_is_a_budget_stop_in_the_sweep_and_the_directory_stays_analyzable(
        haiku_baseline: Path, journal: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.analyze_sweep import analyze_sweep_runs

    real_bound = billing.call_bound
    calls = {"n": 0}

    def bound(*args, **kwargs):  # 前面的呼叫照常；第 40 次之後的呼叫最壞成本很高 → 保險絲在 rollout 中途觸發
        calls["n"] += 1
        return real_bound(*args, **kwargs) if calls["n"] < 40 else 100.0

    monkeypatch.setattr(billing, "call_bound", bound)
    summary = _paid_sweep(select_sources(haiku_baseline), runs("s"))
    assert summary.stopped_by_budget and len(summary.completed) < 36
    assert all(row.error_class != "harness" for row in read_manifest(runs("s") / "manifest.jsonl"))
    assert analyze_sweep_runs([runs("s")], [haiku_baseline], replicates=10) is not None


# ---------------------------------------------------------------- 已 commit 的正式結果（decision 035，2026-10-23）

HAIKU_RESULTS = ROOT / "docs" / "results" / "study2-haiku"
HAIKU_EVAL = ROOT / "runs" / "haiku-study2-eval-20261023-b"


def test_the_committed_haiku_results_match_the_summary() -> None:
    """正式的 Haiku 報告、曲線圖與 README 段落，必須等於從已 commit 的 summary.json 重新呈現的結果。"""
    summary = json.loads((HAIKU_RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["dry_run"] is False and summary["provenance"]["replicates"] == 10_000
    assert rep.variant(summary) == "study2-haiku"
    readme = (ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
    assert rep.embed(readme, rep.render_readme_block(summary)) == readme
    assert (HAIKU_RESULTS / "README.md").read_text(encoding="utf-8") == rep.render_page(summary)
    assert (HAIKU_RESULTS / "recoverability.svg").read_text(encoding="utf-8") == rep.recoverability_svg(summary)


@pytest.mark.skipif(not HAIKU_EVAL.is_dir(), reason="runs/ 不在 repo；只在有原始 run 目錄的機器上檢查")
def test_the_haiku_summary_regenerates_byte_identical_from_the_raw_run() -> None:
    summary = rep.build([HAIKU_EVAL], [])
    text = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    assert text == (HAIKU_RESULTS / "summary.json").read_text(encoding="utf-8")
