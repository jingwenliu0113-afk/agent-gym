"""研究登記表（decision 036 §8 第 1 步）：每個 suite 的開關、呼叫上限、source 規則與付費規格由 ``tasks.suites.STUDIES`` 提供；
沒有登記的 suite 一律拒絕模型推論（fail closed）。Study 3 的付費規格（獨立總帳、40 美元、dev ≤ 3、baseline ≤ 8）與 Study 2 的
付費入口關閉（decision 035 已結案）。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from agents.baseline import HAIKU_MODEL, PRICING, haiku_agent_config
from experiments import billing, run_baseline
from experiments.run_baseline import run_split
from tasks import suites
from tasks.suites import STUDIES, SUITES, suite_pool
from tests.test_haiku_study import PaidLikeClient, ZERO

ROOT = Path(__file__).resolve().parent.parent
STUDY3_JOURNAL_AT_IMPORT = billing.STUDY3_JOURNAL


@pytest.fixture
def journals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """兩份總帳與付費 run 的 runs/ 都換到 tmp_path。"""
    paths = {"study2": tmp_path / "j" / "study2.jsonl", "study3": tmp_path / "j" / "study3.jsonl"}
    monkeypatch.setattr(billing, "DEFAULT_JOURNAL", paths["study2"])
    monkeypatch.setattr(billing, "STUDY3_JOURNAL", paths["study3"])
    monkeypatch.setattr(billing, "RUNS_ROOT", tmp_path / "runs")
    return paths


def _spend(path: Path, entries: list[tuple[str, float | None]]) -> None:
    for i, (phase, cost) in enumerate(entries):
        rid = f"earlier-{path.stat().st_size if path.exists() else 0}-{i}"
        billing._write(path, {"kind": "reserve", "id": rid, "utc_time": "t", "phase": phase, "run": "earlier",
                              "bound_usd": 0.0 if cost is None else cost})
        billing._write(path, {"kind": "settle", "id": rid, "utc_time": "t", "cost_usd": cost})


# ---------------------------------------------------------------- 登記表


def test_every_suite_is_registered_and_only_the_first_round_is_ungated() -> None:
    assert set(STUDIES) == set(SUITES)
    assert [s for s, spec in STUDIES.items() if not spec.gated] == ["week1"]
    week1, study2 = STUDIES["week1"], STUDIES["study2"]
    assert week1.source_rule == "rollout0" and week1.paid is None
    assert study2.source_rule == "first_failure" and study2.paid is suites.STUDY2_PAID
    assert study2.baseline_caps == {"dev": 2880, "eval": 4320} and study2.segment_cap == 3000
    assert suites.sweep_cap("study2") == 70_000


def test_a_gated_study_must_register_every_limit() -> None:
    """開關存在、上限缺漏的研究無法登記：之後新增的研究不會因為少寫一個欄位而在沒有上限的情況下推論。"""
    base = STUDIES["study2"]
    for field in ("frozen_flag", "baseline_caps", "segment_cap", "sweep_cap_flag"):
        with pytest.raises(ValueError):
            dataclasses.replace(base, **{field: None})
    with pytest.raises(ValueError):
        dataclasses.replace(base, source_rule="rollout7")
    with pytest.raises(ValueError):  # 沒有開關的研究必須寫明理由
        dataclasses.replace(base, approved_flag=None, frozen_flag=None, ungated_reason=None)


def test_unregistered_suites_are_refused_everywhere() -> None:
    for check in (lambda: suites.check_model_run("study9", "dev"),
                  lambda: suites.study("study9"),
                  lambda: suites.check_limits("study9", split="dev", call_log=True, max_model_calls=1, sweep=False),
                  lambda: suites.check_limits("week1", split="dev", call_log=True, max_model_calls=1, sweep=False)):
        with pytest.raises(ValueError):
            check()


def test_the_study2_switches_and_limits_read_the_current_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    suites.check_model_run("study2", "eval")
    monkeypatch.setattr(suites, "STUDY2_FROZEN", False)
    with pytest.raises(ValueError, match="frozen"):
        suites.check_model_run("study2", "eval")
    suites.check_limits("study2", split="dev", call_log=True, max_model_calls=2880, sweep=False)
    suites.check_limits("study2", split="eval", call_log=True, max_model_calls=4320, sweep=False)
    suites.check_limits("study2", split="eval", call_log=True, max_model_calls=3000, sweep=True)
    for kwargs in (dict(split="dev", call_log=False, max_model_calls=2880, sweep=False),
                   dict(split="eval", call_log=True, max_model_calls=2880, sweep=False),
                   dict(split="eval", call_log=True, max_model_calls=3001, sweep=True),
                   dict(split="eval", call_log=True, max_model_calls=None, sweep=True)):
        with pytest.raises(ValueError):
            suites.check_limits("study2", **kwargs)
    monkeypatch.setattr(suites, "STUDY2_SWEEP_CAP", None)
    assert suites.sweep_cap("study2") is None


# ---------------------------------------------------------------- 付費規格


def test_the_study3_paid_spec_matches_decision_036() -> None:
    spec = suites.STUDY3_PAID
    assert spec.model == HAIKU_MODEL and spec.total_usd == 40.0
    assert dict(spec.phase_caps) == {"dev": 3.0, "baseline": 8.0}
    assert spec.splits == ("dev", "eval") and spec.rollouts == 3
    assert spec.generator == "issue-tracker-gen/v3" and spec.closed_flag is None
    assert STUDY3_JOURNAL_AT_IMPORT == Path.home() / ".agent-gym" / "study3-haiku-billing.jsonl"
    assert STUDY3_JOURNAL_AT_IMPORT != billing.DEFAULT_JOURNAL and ROOT not in STUDY3_JOURNAL_AT_IMPORT.parents


def test_paid_phases_follow_the_split() -> None:
    assert suites.paid_phase("dev", sweep=False) == "dev"
    assert suites.paid_phase("eval", sweep=False) == "baseline"
    assert suites.paid_phase("eval", sweep=True) == "sweep"
    with pytest.raises(ValueError):
        suites.paid_phase("dev", sweep=True)


def test_study3_paid_runs_are_checked_against_its_own_caps() -> None:
    spec = suites.STUDY3_PAID
    suites.check_paid_run(spec, model=HAIKU_MODEL, split="dev", max_cost_usd=3.0, sweep=False, rollouts=3)
    suites.check_paid_run(spec, model=HAIKU_MODEL, split="eval", max_cost_usd=8.0, sweep=False, rollouts=3)
    suites.check_paid_run(spec, model=HAIKU_MODEL, split="eval", max_cost_usd=40.0, sweep=True)
    for kwargs in (dict(model="claude-opus-5-5", split="dev", max_cost_usd=1.0, sweep=False, rollouts=3),
                   dict(model=HAIKU_MODEL, split="dev", max_cost_usd=3.01, sweep=False, rollouts=3),
                   dict(model=HAIKU_MODEL, split="dev", max_cost_usd=1.0, sweep=False, rollouts=1),
                   dict(model=HAIKU_MODEL, split="eval", max_cost_usd=8.01, sweep=False, rollouts=3),
                   dict(model=HAIKU_MODEL, split="eval", max_cost_usd=40.01, sweep=True),
                   dict(model=HAIKU_MODEL, split="dev", max_cost_usd=1.0, sweep=True),
                   dict(model=HAIKU_MODEL, split="eval", max_cost_usd=0.0, sweep=True),
                   dict(model=HAIKU_MODEL, split="eval", max_cost_usd=True, sweep=True)):
        with pytest.raises(ValueError):
            suites.check_paid_run(spec, **kwargs)


def test_the_study2_paid_entry_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """decision 035 已結案（036 §7）：Study 2 的付費 run 一律拒絕；登記的檢查本身不變（重新打開時照舊）。"""
    assert suites.STUDY2_PAID_CLOSED is True
    with pytest.raises(ValueError, match="closed"):
        suites.check_paid_run(suites.STUDY2_PAID, model=HAIKU_MODEL, split="eval", max_cost_usd=15.0, sweep=False, rollouts=3)
    monkeypatch.setattr(suites, "STUDY2_PAID_CLOSED", False)
    suites.check_paid_run(suites.STUDY2_PAID, model=HAIKU_MODEL, split="eval", max_cost_usd=15.0, sweep=False, rollouts=3)


def test_a_paid_study2_run_is_refused_before_any_call(journals: dict[str, Path], tmp_path: Path) -> None:
    client = PaidLikeClient(suite_pool("study2", "eval"))
    with pytest.raises(ValueError, match="closed"):
        run_split("eval", client, run_dir=billing.RUNS_ROOT / "x", clock=ZERO, suite="study2", agent_config=haiku_agent_config(),
                  rollouts=3, max_cost_usd=15.0)
    assert client.inner.requests == [] and not journals["study2"].exists() and not (billing.RUNS_ROOT / "x").exists()


def test_the_paid_study2_cli_is_closed(journals: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list = []
    monkeypatch.setattr(run_baseline, "make_anthropic_client", lambda model_config=None: made.append(1))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "eval", "--suite", "study2", "--provider", "anthropic", "--model", HAIKU_MODEL,
                           "--rollouts", "3", "--confirm-paid-run", "--max-cost-usd", "15", "--run-name", "x"])
    assert made == [] and not journals["study2"].exists()


# ---------------------------------------------------------------- 總帳（每個研究一份）


def test_the_dev_phase_is_counted_separately(journals: dict[str, Path]) -> None:
    _spend(journals["study3"], [("dev", 0.5), ("baseline", 2.0), ("sweep", 1.0)])
    spent = billing.journal_spent(journals["study3"])
    assert (spent.dev, spent.baseline, spent.sweep, spent.total) == pytest.approx((0.5, 2.0, 1.0, 3.5))
    assert billing.journal_spent(journals["study2"]).dev == 0.0


def test_study3_caps_are_cumulative_per_phase_and_in_total(journals: dict[str, Path]) -> None:
    spec, cid = suites.STUDY3_PAID, haiku_agent_config().config_id
    assert billing.check_budget(spec=spec, phase="dev", requested=3.0, config_id=cid)[1] == pytest.approx(3.0)
    assert billing.check_budget(spec=spec, phase="baseline", requested=8.0, config_id=cid)[1] == pytest.approx(8.0)
    assert billing.check_budget(spec=spec, phase="sweep", requested=40.0, config_id=cid)[1] == pytest.approx(39.0)
    _spend(journals["study3"], [("dev", 2.5)])
    assert billing.check_budget(spec=spec, phase="dev", requested=3.0, config_id=cid)[1] == pytest.approx(0.5)
    assert billing.check_budget(spec=spec, phase="sweep", requested=40.0, config_id=cid)[1] == pytest.approx(36.5)
    _spend(journals["study3"], [("baseline", 8.0)])
    with pytest.raises(ValueError, match="baseline"):
        billing.check_budget(spec=spec, phase="baseline", requested=8.0, config_id=cid)
    _spend(journals["study3"], [("sweep", 28.5)])
    with pytest.raises(ValueError, match="40"):
        billing.check_budget(spec=spec, phase="sweep", requested=40.0, config_id=cid)


def test_the_two_journals_do_not_mix(journals: dict[str, Path]) -> None:
    """Study 2 的花費不計入 Study 3（各自的總帳與上限）；Study 3 的 session 不能當成 Study 2 的 session 用，反之亦然。"""
    cid = haiku_agent_config().config_id
    _spend(journals["study2"], [("baseline", 0.117), ("sweep", 60.0)])
    assert billing.check_budget(spec=suites.STUDY3_PAID, phase="sweep", requested=40.0, config_id=cid)[1] == pytest.approx(39.0)
    with billing.paid_session(spec=suites.STUDY3_PAID, phase="dev", requested=3.0, run="r", config_id=cid) as session:
        assert session.path == journals["study3"] and session.total == 40.0
        with pytest.raises(ValueError):
            billing.check_session(session, spec=suites.STUDY2_PAID, phase="dev", max_cost_usd=1.0)
        billing.check_session(session, spec=suites.STUDY3_PAID, phase="dev", max_cost_usd=3.0)
    with billing.paid_lock(journals["study2"]):  # 各自的程序鎖
        with billing.paid_session(spec=suites.STUDY3_PAID, phase="dev", requested=3.0, run="r", config_id=cid):
            pass


def test_the_study3_fuse_is_at_40_dollars(journals: dict[str, Path]) -> None:
    prices = PRICING[HAIKU_MODEL]
    bound = billing.call_bound(400_000, 16000, prices)
    session = billing.PaidSession(path=journals["study3"], phase="sweep", run="r", spent_before=40.0 - bound / 2, cap=0.1,
                                  total=40.0, active=True)
    with pytest.raises(billing.BudgetFuse, match="40"):
        session.reserve(request_chars=400_000, max_tokens=16000, prices=prices)
    session = billing.PaidSession(path=journals["study3"], phase="sweep", run="r", spent_before=40.0 - 2 * bound, cap=0.1,
                                  total=40.0, active=True)
    session.reserve(request_chars=400_000, max_tokens=16000, prices=prices)  # 40 以下：照常預留（70 不再適用）


def test_the_reset_check_only_counts_the_study_s_own_runs(journals: dict[str, Path], tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """總帳被重設的核對只計入同一個研究的目錄（manifest 的 generator_version）：Study 2 的 Haiku 目錄不會讓 Study 3 拒絕開始。"""
    from tests.test_haiku_study import _haiku_manifest  # 真正的 Study 2 付費 manifest（generator v2）

    cid = haiku_agent_config().config_id
    run = billing.RUNS_ROOT / "haiku-study2"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"config_id": cid, "dry_run": False}), encoding="utf-8")
    (run / "manifest.jsonl").write_text("", encoding="utf-8")
    assert billing.manifest_floor(cid, generator="issue-tracker-gen/v2") == 0  # 空的 manifest
    monkeypatch.setattr(suites, "STUDY2_PAID_CLOSED", False)  # 只為了產生一份真正的 Study 2 付費 manifest
    (run / "manifest.jsonl").write_bytes(_haiku_manifest(tmp_path).read_bytes())
    monkeypatch.setattr(suites, "STUDY2_PAID_CLOSED", True)
    assert billing.manifest_floor(cid, generator="issue-tracker-gen/v2") > 0
    assert billing.manifest_floor(cid, generator="issue-tracker-gen/v3") == 0
    assert billing.manifest_floor(cid) == billing.manifest_floor(cid, generator="issue-tracker-gen/v2")
    billing.check_budget(spec=suites.STUDY3_PAID, phase="dev", requested=3.0, config_id=cid)
    with pytest.raises(ValueError, match="reset"):
        billing.check_budget(spec=suites.STUDY2_PAID, phase="baseline", requested=15.0, config_id=cid)


# ---------------------------------------------------------------- 審查的修正（decision 036 §8 第 1 步 A 部分）


@pytest.fixture
def unpaid_study(monkeypatch: pytest.MonkeyPatch) -> str:
    """在記憶體中登記一個有開關、但沒有付費規格的研究（題目沿用 study2 的 pool）。"""
    spec = dataclasses.replace(STUDIES["study2"], suite="study9", paid=None)
    monkeypatch.setitem(STUDIES, "study9", spec)
    for module in (suites, run_baseline):
        monkeypatch.setattr(module, "SUITES", (*suites.SUITES, "study9"))
    suites.suite_pool.cache_clear()
    yield "study9"
    suites.suite_pool.cache_clear()


def test_a_gated_study_without_a_paid_spec_refuses_every_paid_run(unpaid_study: str, tmp_path: Path) -> None:
    """付費規格缺漏的研究：任何 Anthropic agent（不只 Haiku）的真正推論都在建立 run 目錄、送出任何呼叫之前拒絕，不會以本機的
    呼叫上限代替總帳。"""
    from agents.baseline import baseline_agent_config
    from experiments.run_baseline import _run_split
    from experiments.run_sweep import _run_sweeps, run_sweeps
    from replay.localization import SweepPlan

    client = PaidLikeClient(suite_pool(unpaid_study, "dev"))
    for runner in (run_split, _run_split):
        with pytest.raises(ValueError, match="paid"):
            runner("dev", client, run_dir=tmp_path / "x", clock=ZERO, suite=unpaid_study, agent_config=baseline_agent_config(),
                   rollouts=1, call_log=True, max_model_calls=2880)
    pool = suite_pool(unpaid_study, "eval")
    sources = {i.task.task_id: None for i in pool}
    plan = SweepPlan(coarse_stride=4, n_coarse=1, n_fine=1, drop_threshold=0.3)
    for runner in (run_sweeps, _run_sweeps):
        with pytest.raises(ValueError):
            runner(sources, PaidLikeClient(pool), plan=plan, criterion="bonferroni95", run_dir=tmp_path / "y",
                   agent_config=baseline_agent_config(), call_log=True, max_model_calls=3000)
    assert client.inner.requests == [] and not (tmp_path / "x").exists() and not (tmp_path / "y").exists()
    with pytest.raises(ValueError, match="paid"):
        run_baseline.check_paid_registration(client, baseline_agent_config(), unpaid_study, 1.0)


def test_registered_caps_must_be_positive_integers() -> None:
    base = STUDIES["study2"]
    for change in (dict(baseline_caps={"dev": None, "eval": 4320}), dict(baseline_caps={"dev": 0, "eval": 4320}),
                   dict(segment_cap=float("inf")), dict(segment_cap=True), dict(segment_cap=-1)):
        with pytest.raises(ValueError):
            dataclasses.replace(base, **change)
