"""Study 2（decision 034）在既有 run / sweep / 分析工具上的端到端測試（dry-run 與假的本機 server；不連網）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import local_agent_config
from experiments import run_baseline, run_sweep
from experiments.analyze_baseline import analyze_runs, load_runs
from experiments.analyze_sweep import analyze_sweep_runs
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from replay.localization import SweepPlan
from tasks.suites import eval_task_ids, suite_pool
from tests.wrapper_env import WRAPPER_URL, frozen_server, inside_wrapper

SMALL_PLAN = SweepPlan(coarse_stride=4, n_coarse=1, n_fine=1, drop_threshold=0.3)
EVAL = suite_pool("study2", "eval")
DEV = suite_pool("study2", "dev")


def ZERO() -> float:  # noqa: N802
    return 0.0


@pytest.fixture(scope="module")
def study2_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("study2") / "eval"
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True,
              suite="study2")
    return run_dir


def test_a_study2_dry_run_baseline_completes_every_task(study2_baseline: Path) -> None:
    summary = json.loads((study2_baseline / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_tasks"] == 36 and len(summary["results"]) == 36
    assert all(r["outcome"] == "success" and r["trajectory_safe"] for r in summary["results"])
    loaded = load_runs([study2_baseline])
    assert loaded.suite == "study2"
    assert {r.task_id for r in loaded.records} == set(eval_task_ids("study2"))
    analysis = analyze_runs([study2_baseline], replicates=200)
    assert analysis.missing_tasks == ()


def test_the_week1_default_is_unchanged(tmp_path: Path) -> None:
    week1 = suite_pool("week1", "dev")
    summary = run_split("dev", OllamaOracleFakeClient(week1), run_dir=tmp_path / "w", clock=ZERO, agent_config=local_agent_config(),
                        dry_run=True)
    assert [r.task_id for r in summary.results] == [i.task.task_id for i in week1]


def test_run_split_rejects_an_unknown_suite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="suite"):
        run_split("dev", OllamaOracleFakeClient(DEV), run_dir=tmp_path / "x", agent_config=local_agent_config(), suite="study3")


def test_a_study2_dry_run_sweep_and_its_analysis(study2_baseline: Path, tmp_path: Path) -> None:
    sources = select_sources(study2_baseline)
    assert set(sources) == set(eval_task_ids("study2")) and all(s is not None for s in sources.values())
    summary = run_sweeps(sources, OllamaOracleFakeClient(EVAL), plan=SMALL_PLAN, criterion="bonferroni95", run_dir=tmp_path / "s",
                         clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    assert set(summary.results) == set(sources)
    analysis = analyze_sweep_runs([tmp_path / "s"], [study2_baseline], replicates=50)
    assert len(analysis.curves) == 36


def test_study2_cli_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_baseline.main(["--split", "dev", "--suite", "study2", "--dry-run"]) == 0
    summary = json.loads((tmp_path / "runs" / "study2-dry-run-dev" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_tasks"] == 24


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def make(base_url: str, model_config: Any = None) -> OllamaOracleFakeClient:
        urls.append(base_url)
        return frozen_server([*EVAL, *DEV])

    monkeypatch.setattr(run_baseline, "make_ollama_client", make)
    return urls


@pytest.mark.parametrize("split", ["dev", "eval"])
def test_study2_model_runs_are_refused_before_approval(split: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       made: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    argv = ["--split", split, "--suite", "study2", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "x"]
    if split == "eval":
        argv += ["--rollouts", "3"]
    with pytest.raises(SystemExit):
        run_baseline.main(argv)
    assert "approval of decision 034" in capsys.readouterr().err
    assert made == [] and not (tmp_path / "runs" / "x").exists()


def test_study2_dev_runs_are_allowed_after_approval_but_eval_needs_the_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                              made: list[str],
                                                                              capsys: pytest.CaptureFixture[str]) -> None:
    from tasks import suites

    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    monkeypatch.setattr(suites, "STUDY2_APPROVED", True)
    with pytest.raises(SystemExit):  # dev baseline 的上限固定為 2,880（decision 034 §8）
        run_baseline.main(["--split", "dev", "--suite", "study2", "--confirm-local-run", "--max-model-calls", "40", "--run-name", "d0"])
    assert "--max-model-calls 2880" in capsys.readouterr().err and made == []
    assert run_baseline.main(["--split", "dev", "--suite", "study2", "--confirm-local-run", "--max-model-calls", "2880",
                              "--run-name", "d"]) == 0
    assert made == [WRAPPER_URL]
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "eval", "--suite", "study2", "--rollouts", "3", "--confirm-local-run", "--max-model-calls", "4320",
                           "--run-name", "e"])
    assert "frozen record of decision 034" in capsys.readouterr().err
    assert made == [WRAPPER_URL] and not (tmp_path / "runs" / "e").exists()


def test_a_study2_formal_sweep_is_refused_before_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list[str],
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    baseline = tmp_path / "b"
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=baseline, clock=ZERO, agent_config=local_agent_config(), rollouts=1,
              suite="study2")  # 以 Python 直接建立（非 dry-run）的 baseline，只用來測 CLI 的檢查
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):
        run_sweep.main(["--baseline-run", str(baseline), "--run-name", "s", "--plan", "2,10,48,0.3", "--criterion", "bonferroni95",
                        "--confirm-local-run", "--max-model-calls", "50"])
    assert "approval of decision 034" in capsys.readouterr().err
    assert made == [] and not (tmp_path / "runs" / "s").exists()


def test_the_round1_analysis_and_report_refuse_study2(study2_baseline: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Study 2 的報告不做 Gate 判定（decision 034 §1）：第一輪的 sweep 分析 CLI 與結果報告都拒絕 Study 2 的目錄。"""
    from experiments import analyze_sweep, results_report

    sources = select_sources(study2_baseline)
    run_sweeps(sources, OllamaOracleFakeClient(EVAL), plan=SMALL_PLAN, criterion="bonferroni95", run_dir=tmp_path / "s",
               clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    with pytest.raises(SystemExit):
        analyze_sweep.main([str(tmp_path / "s"), "--baseline-run", str(study2_baseline), "--out", str(tmp_path / "a")])
    assert "Study 2 has its own analysis" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()
    with pytest.raises(ValueError, match="Study 2 has its own report"):
        results_report.write_report([study2_baseline], [tmp_path / "s"], tmp_path / "r")
    assert not (tmp_path / "r").exists()


def test_a_study2_formal_sweep_must_use_the_registered_total_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list[str],
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    from tasks import suites

    baseline = tmp_path / "b"
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=baseline, clock=ZERO, agent_config=local_agent_config(), rollouts=1,
              suite="study2")
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    monkeypatch.setattr(suites, "STUDY2_APPROVED", True)
    monkeypatch.setattr(suites, "STUDY2_FROZEN", True)
    monkeypatch.setattr(suites, "STUDY2_SWEEP_CAP", 60000)
    common = ["--baseline-run", str(baseline), "--plan", "2,10,48,0.3", "--criterion", "bonferroni95", "--confirm-local-run",
              "--max-model-calls", "30"]
    for wrong in ([], ["--total-model-calls", "100"]):
        with pytest.raises(SystemExit):
            run_sweep.main([*common, "--run-name", "w", *wrong])
        assert "must use --total-model-calls 60000" in capsys.readouterr().err
    assert made == [] and not (tmp_path / "runs" / "w").exists()
    with pytest.raises(SystemExit):  # 每段最多 3,000 次
        run_sweep.main([*common[:-1], "5000", "--run-name", "big", "--total-model-calls", "60000"])
    assert "at most 3000" in capsys.readouterr().err
    assert run_sweep.main([*common, "--run-name", "ok", "--total-model-calls", "60000"]) == 0
    assert made == [WRAPPER_URL]


class _RealLooking:
    """不是 oracle 假 client 的 client：函式層的開關必須在建立 policy 之後、任何呼叫之前拒絕。"""

    def __getattr__(self, name: str):  # noqa: ANN204
        raise AssertionError(f"the model must not be called ({name})")


def test_run_split_and_run_sweeps_check_the_locks_themselves(study2_baseline: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """decision 034 審查：開關不只在 CLI；直接以 Python 呼叫、client 不是 oracle 假 client 時也拒絕。"""
    import experiments.run_baseline as rb
    import experiments.run_sweep as rs

    monkeypatch.setattr(rb, "make_policy", lambda client, config: None)
    monkeypatch.setattr(rs, "make_policy", lambda client, config: None)
    with pytest.raises(ValueError, match="approval of decision 034"):
        run_split("dev", _RealLooking(), run_dir=tmp_path / "d", agent_config=local_agent_config(), suite="study2")
    assert not (tmp_path / "d").exists()
    with pytest.raises(ValueError, match="approval of decision 034"):
        run_sweeps(select_sources(study2_baseline), _RealLooking(), plan=SMALL_PLAN, criterion="bonferroni95", run_dir=tmp_path / "s",
                   agent_config=local_agent_config())
    assert not (tmp_path / "s").exists()


def test_function_level_limits_for_study2(study2_baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """開關打開之後，直接以 Python 呼叫也必須記錄每次呼叫、使用登記的上限（驗證 B 第二輪）。"""
    import experiments.run_baseline as rb
    import experiments.run_sweep as rs
    from tasks import suites

    monkeypatch.setattr(suites, "STUDY2_APPROVED", True)
    monkeypatch.setattr(suites, "STUDY2_FROZEN", True)
    monkeypatch.setattr(rb, "make_policy", lambda client, config: None)
    monkeypatch.setattr(rs, "make_policy", lambda client, config: None)
    with pytest.raises(ValueError, match="log every call"):
        run_split("dev", _RealLooking(), run_dir=tmp_path / "a", agent_config=local_agent_config(), suite="study2", max_model_calls=2880)
    with pytest.raises(ValueError, match="2880"):
        run_split("dev", _RealLooking(), run_dir=tmp_path / "b", agent_config=local_agent_config(), suite="study2", call_log=True,
                  max_model_calls=100)
    with pytest.raises(ValueError, match="at most 3000"):
        run_sweeps(select_sources(study2_baseline), _RealLooking(), plan=SMALL_PLAN, criterion="bonferroni95", run_dir=tmp_path / "s",
                   agent_config=local_agent_config(), call_log=True, max_model_calls=None)
    assert not any((tmp_path / d).exists() for d in ("a", "b", "s"))
