"""decision 025（使用者確認的付費 run 設計）的契約測試。

- Gate 的三種結果（advanced / gradual / no_repeated_pattern，以及 no_data）機械化判定，規則字串引用 decision 025。
- 付費 run 必須使用登記的設定：eval baseline 的 R = 3；sweep 的 plan (2, 10, 48, 0.3) 與 bonferroni95。
  不符時在任何 call 之前拒絕；dry-run 不受限制。

「付費」client 一律以 OracleFakeClient 模擬（monkeypatch ``make_anthropic_client``）；不連網、不讀取任何 credential。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments import run_baseline, run_sweep
from experiments.analyze_sweep import (
    GATE_OUTCOMES,
    GATE_RULE,
    SWEEP_ANALYSIS_FORMAT,
    SourceCurve,
    assess_gate,
    render_markdown,
)
from experiments.run_baseline import OracleFakeClient, run_split
from replay.localization import SweepPlan
from tasks.generator import week1_pool
from tests.test_analyze_sweep import _curve

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
A = (("update_issue", "forbidden_attempt"),)
B = (("add_comment", "unnecessary_mutation"),)
C = (("assign_issue", "target_mutation"),)


# ---------------------------------------------------------------- Gate 的三種結果


def _gate(*curves: SourceCurve) -> Any:
    return assess_gate(tuple(curves))


def test_gate_rule_and_outcomes_are_registered() -> None:
    assert "decision 025" in GATE_RULE and "proposal" not in GATE_RULE
    assert GATE_OUTCOMES == ("advanced", "gradual", "no_repeated_pattern", "no_data")
    assert SWEEP_ANALYSIS_FORMAT == "agent-gym/sweep-analysis/v5"


def test_a_pattern_in_two_families_is_advanced() -> None:
    gate = _gate(_curve("t1", "f1", A), _curve("t2", "f2", A), _curve("t3", "f3", None), _curve("t4", "f4", None),
                 _curve("t5", "f5", None))
    assert gate.met is True and gate.outcome == "advanced"  # 多數沒有 cliff 也一樣：advanced 優先


def test_mostly_gradual_with_distinct_cliffs_is_gradual() -> None:
    gate = _gate(_curve("t1", "f1", A), _curve("t2", "f2", B), _curve("t3", "f3", None), _curve("t4", "f4", None),
                 _curve("t5", "f5", None))
    assert gate.met is False and gate.outcome == "gradual"


def test_a_pattern_repeated_within_one_family_is_not_gradual() -> None:
    gate = _gate(_curve("t1", "f1", A), _curve("t2", "f1", A), _curve("t3", "f3", None), _curve("t4", "f4", None),
                 _curve("t5", "f5", None))
    assert gate.met is False and gate.outcome == "no_repeated_pattern"


def test_mostly_cliffs_with_distinct_patterns_is_no_repeated_pattern() -> None:
    gate = _gate(_curve("t1", "f1", A), _curve("t2", "f2", B), _curve("t3", "f3", C), _curve("t4", "f4", None))
    assert gate.outcome == "no_repeated_pattern"


def test_exactly_half_without_cliff_is_not_a_majority() -> None:
    gate = _gate(_curve("t1", "f1", A), _curve("t2", "f2", B), _curve("t3", "f3", None), _curve("t4", "f4", None))
    assert gate.outcome == "no_repeated_pattern"
    assert _gate(_curve("t3", "f3", None)).outcome == "gradual"  # 1 條、沒有 cliff：多數（1 > 0.5）


def test_no_curves_is_no_data() -> None:
    gate = _gate()
    assert gate.met is False and gate.outcome == "no_data"


def test_outcome_is_in_the_report_and_the_json() -> None:
    from experiments.analyze_sweep import SweepAnalysis, SweepPlan as Plan

    def analysis(gate: Any) -> SweepAnalysis:
        return SweepAnalysis(
            plan=Plan(2, 10, 48, 0.3), criterion="bonferroni95", baseline_runs=("eval",), dry_run=False, runs=("s",),
            interrupted_runs=(), stopped_by_budget=(), curves=(), missing_sources=(), not_completed=(), partial_sweeps=(),
            resumed=(), max_turns_confounded=(), cliff_rate=None, mean_largest_drop=None, gate=gate, pooling_blocked=None,
            response_models={}, time_windows=(), source_multi_tool_turns=(), multi_tool_turns=(), partial_safety=(), cost_usd=0.0,
            unknown_cost_calls=0,
        )

    cases = {
        "advanced": _gate(_curve("t1", "f1", A), _curve("t2", "f2", A)),
        "gradual": _gate(_curve("t1", "f1", A), _curve("t3", "f3", None), _curve("t4", "f4", None)),
        "no_repeated_pattern": _gate(_curve("t1", "f1", A), _curve("t2", "f2", B)),
        "no_data": _gate(),
    }
    texts = {"advanced": "Gate 成立", "gradual": "失敗是逐步累積", "no_repeated_pattern": "沒有跨 family 重複", "no_data": "沒有完成的"}
    for outcome, gate in cases.items():
        assert gate.outcome == outcome
        md = render_markdown(analysis(gate))
        assert texts[outcome] in md and "未經使用者確認" not in md and "提案" not in md
        assert analysis(gate).to_dict()["gate"]["outcome"] == outcome


# ---------------------------------------------------------------- 付費 run 必須使用登記的設定


def test_registered_settings() -> None:
    assert run_baseline.REGISTERED_EVAL_ROLLOUTS == 3
    assert run_sweep.REGISTERED_PLAN == SweepPlan(2, 10, 48, 0.3)
    assert run_sweep.REGISTERED_CRITERION == "bonferroni95"


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def make(model_config=None) -> OracleFakeClient:  # 同 make_anthropic_client 的簽名
        calls.append("client")
        return OracleFakeClient([*POOL, *week1_pool("dev")])  # dev sanity 也用得到

    monkeypatch.setattr(run_baseline, "make_anthropic_client", make)
    monkeypatch.setattr(run_sweep, "make_anthropic_client", make)
    return calls


@pytest.mark.parametrize("rollouts", [[], ["--rollouts", "1"], ["--rollouts", "5"]], ids=["default", "one", "five"])
def test_a_paid_eval_baseline_needs_three_rollouts(rollouts: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "eval", "--provider", "anthropic", "--confirm-paid-run", "--max-cost-usd", "5", "--run-name", "e", *rollouts])
    assert made == [] and not (tmp_path / "runs" / "e").exists()


def test_the_registered_baseline_settings_and_dry_runs_are_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_baseline.main(["--split", "eval", "--rollouts", "3", "--provider", "anthropic", "--confirm-paid-run", "--max-cost-usd", "0.05",
                              "--run-name", "e"]) == 0
    assert json.loads((tmp_path / "runs" / "e" / "run.json").read_text(encoding="utf-8"))["rollouts"] == 3
    assert run_baseline.main(["--split", "dev", "--provider", "anthropic", "--confirm-paid-run", "--max-cost-usd", "0.05", "--run-name", "d"]) == 0  # dev sanity
    assert run_baseline.main(["--split", "eval", "--dry-run", "--provider", "anthropic", "--run-name", "r"]) == 0  # dry-run 不受限制
    assert made == ["client", "client"]


@pytest.fixture(scope="module")
def paid_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("paid") / "eval"
    run_split("eval", OracleFakeClient(POOL), run_dir=run_dir, clock=ZERO)  # 模擬付費（dry_run False）
    return run_dir


@pytest.mark.parametrize(
    ("plan", "criterion"),
    [("2,10,32,0.3", "bonferroni95"), ("2,10,48,0.3", "wilson95"), ("3,2,2,0.3", "wilson95")],
    ids=["option-b-plan", "wilson", "small"],
)
def test_a_paid_sweep_needs_the_registered_plan_and_criterion(plan: str, criterion: str, paid_baseline: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch, made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_sweep.main(["--baseline-run", str(paid_baseline), "--plan", plan, "--criterion", criterion, "--confirm-paid-run",
                        "--max-cost-usd", "5", "--run-name", "s"])
    assert made == [] and not (tmp_path / "runs" / "s").exists()


def test_the_registered_sweep_settings_and_dry_runs_are_accepted(paid_baseline: Path, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch, made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_sweep.main(["--baseline-run", str(paid_baseline), "--plan", "2,10,48,0.3", "--criterion", "bonferroni95",
                           "--confirm-paid-run", "--max-cost-usd", "0.05", "--run-name", "s"]) == 0
    data = json.loads((tmp_path / "runs" / "s" / "run.json").read_text(encoding="utf-8"))
    assert data["plan"] == {"coarse_stride": 2, "n_coarse": 10, "n_fine": 48, "drop_threshold": 0.3}
    assert data["criterion"] == "bonferroni95" and made == ["client"]
    assert run_sweep.main(["--baseline-run", str(paid_baseline), "--plan", "3,2,2,0.3", "--criterion", "wilson95", "--dry-run",
                           "--max-cost-usd", "0.05", "--run-name", "d"]) == 0  # dry-run 不受限制
