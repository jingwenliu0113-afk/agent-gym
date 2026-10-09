"""第九輪對抗式審查（decision 018 的續跑、核對與目錄識別）確認的缺陷之回歸測試（decision 019）。

- 鏈式續跑只給最近的目錄時，更早目錄已完成的 source 會被重新 sweep、再付一次費（019 §3）。
- eval baseline 跨多個目錄時，source 該取哪一條 rollout_index 0 沒有規則，工具也只接受一個 baseline 目錄（019 §1）。
- 一併處理：沒有 summary 的目錄無從核對 plan、被中斷的續跑目錄沒有略過清單（run.json，019 §2）；沒有要跑的 source 時仍建立
  無法分析的目錄；續跑時印出先前目錄的花費（019 §3）。

全部只用假的 client，不連網。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import pytest

from agents.baseline import baseline_agent_config
from env.hashing import canonical_json
from experiments import run_sweep
from experiments.analyze_sweep import SWEEP_ANALYSIS_FORMAT, analyze_sweep_runs
from experiments.run_baseline import OracleFakeClient, run_split
from experiments.run_sweep import SWEEP_SUMMARY_FORMAT, main, run_sweeps, select_sources
from replay.aborts import read_manifest
from replay.localization import SweepPlan
from tasks.generator import week1_pool
from tests.test_analyze_baseline import ScriptedClient
from tests.test_run_sweep import _FailFor, _InterruptAfter

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
EVAL_IDS = [instance.task.task_id for instance in POOL]
PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)
PLAN_TEXT = "3,2,2,0.3"


@pytest.fixture(scope="module")
def baselines(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """依序執行（manifest 的時間依此遞增）：

    - a：中斷的 eval baseline。EVAL_IDS[0] r0 完成；EVAL_IDS[1] r0 第 1 次嘗試 infrastructure 中止、第 2 次嘗試進行中被
      KeyboardInterrupt（interrupted）；其餘沒有執行。
    - b：之後以新目錄完整重跑（OracleFakeClient：與 a 的 ScriptedClient 回應不同，trajectory_id 也不同）。
    - c：EVAL_IDS[0] r0 是 agent_error，其餘完成。
    - e：c 之後的完整 run。
    """
    root = tmp_path_factory.mktemp("baselines")
    a = root / "a"
    with pytest.raises(KeyboardInterrupt):
        run_split("eval", ScriptedClient({(EVAL_IDS[1], 0): ["oracle|connerr@0", "oracle|kbint@1"]}, rollouts=1), run_dir=a,
                  clock=ZERO)
    b = root / "b"
    run_split("eval", OracleFakeClient(POOL), run_dir=b, clock=ZERO)
    c = root / "c"
    run_split("eval", _FailFor(POOL[0].task.instruction), run_dir=c, clock=ZERO)
    e = root / "e"
    run_split("eval", OracleFakeClient(POOL), run_dir=e, clock=ZERO)
    return {"a": a, "b": b, "c": c, "e": e}


def _ids(sources: dict[str, Any]) -> dict[str, str | None]:
    return {task_id: None if source is None else source.trajectory_id for task_id, source in sources.items()}


# ---------------------------------------------------------------- 019 §1：多目錄 eval baseline 的 source


def test_fixture_premises(baselines: dict[str, Path]) -> None:
    rows = read_manifest(baselines["a"] / "manifest.jsonl")
    assert [(row.task_id, row.attempt, row.outcome) for row in rows] == [(EVAL_IDS[0], 1, "completed"), (EVAL_IDS[1], 1, "aborted")]
    assert not (baselines["a"] / "summary.json").exists()
    a, b = _ids(select_sources([baselines["a"]])), _ids(select_sources([baselines["b"]]))
    assert a[EVAL_IDS[0]] is not None and a[EVAL_IDS[0]] != b[EVAL_IDS[0]]
    assert _ids(select_sources([baselines["c"]]))[EVAL_IDS[0]] is None


def test_select_sources_takes_the_earliest_finished_rollout_zero_across_directories(baselines: dict[str, Path]) -> None:
    a, b = _ids(select_sources([baselines["a"]])), _ids(select_sources([baselines["b"]]))
    expected = {task_id: a[task_id] if task_id == EVAL_IDS[0] else b[task_id] for task_id in EVAL_IDS}
    assert _ids(select_sources([baselines["a"], baselines["b"]])) == expected  # interrupted 的 EVAL_IDS[1] 不算
    assert _ids(select_sources([baselines["b"], baselines["a"]])) == expected  # 與傳入順序無關
    assert list(select_sources([baselines["b"], baselines["a"]])) == EVAL_IDS


def test_an_earlier_unfinished_rollout_zero_leaves_the_task_without_a_source(baselines: dict[str, Path]) -> None:
    sources = _ids(select_sources([baselines["e"], baselines["c"]]))
    assert sources[EVAL_IDS[0]] is None  # c 的 agent_error 較早結束；不得以 e 的 rollout 0 補上（012 §3）
    assert all(sources[task_id] is not None for task_id in EVAL_IDS[1:])
    assert _ids(select_sources([baselines["c"], baselines["b"]]))[EVAL_IDS[0]] == _ids(select_sources([baselines["b"]]))[EVAL_IDS[0]]


def test_rollout_zero_times_that_tie_are_rejected(baselines: dict[str, Path], tmp_path: Path) -> None:
    twin = tmp_path / "twin"
    shutil.copytree(baselines["b"], twin)
    with pytest.raises(ValueError):
        select_sources([baselines["b"], twin])


def test_select_sources_still_accepts_a_single_directory(baselines: dict[str, Path]) -> None:
    single = _ids(select_sources(baselines["b"]))
    assert single == _ids(select_sources(str(baselines["b"]))) == _ids(select_sources([baselines["b"]]))
    with pytest.raises(ValueError):
        select_sources([])


# ---------------------------------------------------------------- 019 §2：run.json


class _Explode:
    """第一次 call 就 KeyboardInterrupt（還沒有任何 manifest 列）。"""

    def __init__(self) -> None:
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> Any:
        raise KeyboardInterrupt


def test_run_json_is_written_before_the_first_attempt(baselines: dict[str, Path], tmp_path: Path) -> None:
    sources = select_sources([baselines["c"]])
    run_dir = tmp_path / "s"
    with pytest.raises(KeyboardInterrupt):
        run_sweeps(sources, _Explode(), plan=PLAN, criterion="bonferroni95", run_dir=run_dir, clock=ZERO, skip=(EVAL_IDS[2],))
    assert read_manifest(run_dir / "manifest.jsonl") == ()  # 空的 manifest（decision 024 §1）
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(data.pop("run_id")) == 32 and data.pop("resumed_from") == []  # decision 020 §2
    assert data.pop("dry_run") is False  # decision 021 §1：直接以 Python 呼叫時預設 False
    assert data == {
        "format": SWEEP_SUMMARY_FORMAT,
        "plan": asdict(PLAN),
        "criterion": "bonferroni95",
        "config_id": baseline_agent_config().config_id,
        "skipped": [EVAL_IDS[2]],
        "sources": _ids(sources),
    }
    assert data["sources"][EVAL_IDS[0]] is None


def test_format_versions() -> None:
    assert SWEEP_SUMMARY_FORMAT == "agent-gym/sweep-run/v7"  # v4：run.json（019 §2）；v5–v7：020 / 021 / 023
    assert SWEEP_ANALYSIS_FORMAT == "agent-gym/sweep-analysis/v4"  # v3：baseline_runs（019 §4）；v4：dry_run（021）


@pytest.fixture(scope="module")
def skipped_sweep(baselines: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("skipped") / "sweep"
    run_sweeps(select_sources([baselines["b"]]), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=run_dir,
               clock=ZERO, replicates=20, skip=tuple(EVAL_IDS[3:]))
    return run_dir


def _edit_run_json(sweep: Path, edit: Callable[[dict[str, Any]], Any]) -> None:
    data = json.loads((sweep / "run.json").read_text(encoding="utf-8"))
    edit(data)
    (sweep / "run.json").write_text(canonical_json(data) + "\n", encoding="utf-8")


RUN_JSON_CHANGES: dict[str, Callable[[Path], Any]] = {
    "missing": lambda sw: (sw / "run.json").unlink(),
    "format": lambda sw: _edit_run_json(sw, lambda d: d.update(format="agent-gym/sweep-run/v3")),
    "extra-key": lambda sw: _edit_run_json(sw, lambda d: d.update(extra=1)),
    "plan": lambda sw: _edit_run_json(sw, lambda d: d["plan"].update(n_fine=4)),
    "criterion": lambda sw: _edit_run_json(sw, lambda d: d.update(criterion="bonferroni95")),
    "config": lambda sw: _edit_run_json(sw, lambda d: d.update(config_id="f" * 64)),
    "skipped": lambda sw: _edit_run_json(sw, lambda d: d["skipped"].pop()),
    "source-id": lambda sw: _edit_run_json(sw, lambda d: d["sources"].update({EVAL_IDS[5]: "0" * 64})),
    "source-missing": lambda sw: _edit_run_json(sw, lambda d: d["sources"].pop(EVAL_IDS[5])),
    "not-an-object": lambda sw: (sw / "run.json").write_text("[]\n", encoding="utf-8"),
}


def test_the_skipped_sweep_is_analyzable(skipped_sweep: Path, baselines: dict[str, Path]) -> None:
    analysis = analyze_sweep_runs([skipped_sweep], [baselines["b"]], replicates=20)
    assert [curve.task_id for curve in analysis.curves] == EVAL_IDS[:3]
    assert analysis.resumed == ({"run": "sweep", "skipped": tuple(EVAL_IDS[3:])},)
    assert analysis.baseline_runs == ("b",)


@pytest.mark.parametrize("change", sorted(RUN_JSON_CHANGES))
def test_analysis_requires_a_run_json_that_matches(change: str, skipped_sweep: Path, baselines: dict[str, Path],
                                                    tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    shutil.copytree(skipped_sweep, sweep)
    RUN_JSON_CHANGES[change](sweep)
    with pytest.raises(ValueError):
        analyze_sweep_runs([sweep], [baselines["b"]], replicates=20)


def test_an_interrupted_directory_takes_its_plan_from_run_json(baselines: dict[str, Path], tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """審查情境 F：沒有 summary、也沒有完成 sweep 的目錄，以另一個 plan / criterion 續跑。"""
    monkeypatch.chdir(tmp_path)
    sources = select_sources([baselines["b"]])
    first = tmp_path / "runs" / "first"
    with pytest.raises(KeyboardInterrupt):
        run_sweeps(sources, _InterruptAfter(OracleFakeClient(POOL), 5), plan=SweepPlan(2, 2, 2, 0.3), criterion="bonferroni95",
                   run_dir=first, clock=ZERO, dry_run=True)  # 與下面的 --dry-run 同一種模式（decision 021 §1）
    assert read_manifest(first / "manifest.jsonl") and not list((first / "sweeps").glob("*.json"))
    with pytest.raises(ValueError):
        main(["--baseline-run", str(baselines["b"]), "--plan", PLAN_TEXT, "--criterion", "wilson95", "--dry-run",
              "--run-name", "second", "--resume-from", str(first)])
    assert not (tmp_path / "runs" / "second").exists()
    other = tmp_path / "runs" / "other"
    run_sweeps(sources, OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=other, clock=ZERO, replicates=20,
               dry_run=True)
    with pytest.raises(ValueError):
        analyze_sweep_runs([first, other], [baselines["b"]], replicates=20)


def test_an_interrupted_resume_directory_still_lists_what_it_skipped(baselines: dict[str, Path], tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    with pytest.raises(KeyboardInterrupt):
        run_sweeps(select_sources([baselines["b"]]), _InterruptAfter(OracleFakeClient(POOL), 3), plan=PLAN, criterion="wilson95",
                   run_dir=sweep, clock=ZERO, skip=tuple(EVAL_IDS[5:]))
    assert not (sweep / "summary.json").exists()
    analysis = analyze_sweep_runs([sweep], [baselines["b"]], replicates=20)
    assert analysis.interrupted_runs == ("sweep",)
    assert analysis.resumed == ({"run": "sweep", "skipped": tuple(EVAL_IDS[5:])},)
    assert set(EVAL_IDS[5:]) <= set(analysis.not_completed)


def test_analysis_with_a_different_set_of_baseline_directories_is_rejected(baselines: dict[str, Path], tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    both = [baselines["a"], baselines["b"]]
    run_sweeps(select_sources(both), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=sweep, clock=ZERO,
               replicates=20, skip=tuple(EVAL_IDS[1:]))  # 只 sweep EVAL_IDS[0]（source 來自 a）
    analysis = analyze_sweep_runs([sweep], both, replicates=20)
    assert analysis.baseline_runs == ("a", "b") and [curve.task_id for curve in analysis.curves] == [EVAL_IDS[0]]
    with pytest.raises(ValueError):
        analyze_sweep_runs([sweep], [baselines["b"]], replicates=20)  # 少了 a：source 不同
    with pytest.raises(ValueError):
        analyze_sweep_runs([baselines["a"]], both, replicates=20)  # sweep 目錄不得是 baseline 目錄


# ---------------------------------------------------------------- 019 §3：鏈式續跑、先前花費、沒有要跑的 source


def _cli(baseline: list[Path], *extra: str) -> int:
    return main(["--baseline-run", *(str(d) for d in baseline), "--plan", PLAN_TEXT, "--criterion", "wilson95", "--dry-run", *extra])


def _manifest_tasks(run_dir: Path) -> set[str]:
    return {row.task_id for row in read_manifest(run_dir / "manifest.jsonl")}


def _completed(run_dir: Path) -> list[str]:
    return sorted(path.stem for path in (run_dir / "sweeps").glob("*.json"))


def test_chained_resume_must_name_every_earlier_directory(baselines: dict[str, Path], tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """審查情境 E：e1 因預算停止（完成 4 個）；e2 從 e1 續跑也因預算停止（沒有完成）；e3 只給 e2 → 會重做 e1 的 source。"""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    base = [baselines["b"]]
    assert _cli(base, "--run-name", "e1", "--max-cost-usd", "0.3") == 0
    done = _completed(runs / "e1")
    assert 0 < len(done) < 15
    assert _cli(base, "--run-name", "e2", "--max-cost-usd", "0.001", "--resume-from", str(runs / "e1")) == 0
    assert _completed(runs / "e2") == [] and read_manifest(runs / "e2" / "manifest.jsonl")
    with pytest.raises(SystemExit):
        _cli(base, "--run-name", "e3", "--resume-from", str(runs / "e2"))
    assert not (runs / "e3").exists()  # 沒有任何 call
    assert _cli(base, "--run-name", "e3", "--resume-from", str(runs / "e1"), str(runs / "e2")) == 0
    assert not (_manifest_tasks(runs / "e3") & set(done))  # e1 完成的不重複付費
    analysis = analyze_sweep_runs([runs / "e1", runs / "e2", runs / "e3"], base, replicates=20)
    assert len(analysis.curves) == 15 and analysis.not_completed == ()
    with pytest.raises(ValueError):  # 漏給 e1：已知的祖先目錄沒有傳入 → 拒絕（decision 021 §2 取代原本的「標示為未完成」）
        analyze_sweep_runs([runs / "e2", runs / "e3"], base, replicates=20)


def test_chained_resume_after_an_interrupted_resume(baselines: dict[str, Path], tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """審查情境 H：h2（從 h1 續跑）被中斷、沒有 summary；它略過的 source 仍由 run.json 得知。"""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    base = [baselines["b"]]
    assert _cli(base, "--run-name", "h1", "--max-cost-usd", "0.3") == 0
    done = _completed(runs / "h1")
    monkeypatch.setattr(run_sweep, "OracleFakeClient", lambda pool: _InterruptAfter(OracleFakeClient(pool), 20))
    with pytest.raises(KeyboardInterrupt):
        _cli(base, "--run-name", "h2", "--resume-from", str(runs / "h1"))
    assert not (runs / "h2" / "summary.json").exists()
    monkeypatch.setattr(run_sweep, "OracleFakeClient", OracleFakeClient)
    with pytest.raises(SystemExit):
        _cli(base, "--run-name", "h3", "--resume-from", str(runs / "h2"))
    assert not (runs / "h3").exists()
    assert _cli(base, "--run-name", "h3", "--resume-from", str(runs / "h1"), str(runs / "h2")) == 0
    assert not (_manifest_tasks(runs / "h3") & set(done))
    analysis = analyze_sweep_runs([runs / "h1", runs / "h2", runs / "h3"], base, replicates=20)
    assert len(analysis.curves) == 15 and analysis.not_completed == ()


def test_resume_prints_what_the_earlier_directories_spent(baselines: dict[str, Path], tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    base = [baselines["b"]]
    assert _cli(base, "--run-name", "first", "--max-cost-usd", "0.3") == 0
    spent = sum(row.cost_usd for row in read_manifest(tmp_path / "runs" / "first" / "manifest.jsonl"))
    capsys.readouterr()
    assert _cli(base, "--run-name", "second", "--max-cost-usd", "0.2", "--resume-from", str(tmp_path / "runs" / "first")) == 0
    err = capsys.readouterr().err
    assert f"${spent:.4f}" in err and "first" in err


def test_run_sweeps_refuses_when_nothing_is_left_to_run(baselines: dict[str, Path], tmp_path: Path) -> None:
    sources = select_sources([baselines["b"]])
    with pytest.raises(ValueError):
        run_sweeps(sources, OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "all-skipped", clock=ZERO,
                   skip=tuple(EVAL_IDS))
    with pytest.raises(ValueError):
        run_sweeps(dict.fromkeys(EVAL_IDS), OracleFakeClient(POOL), plan=PLAN, criterion="wilson95", run_dir=tmp_path / "no-sources",
                   clock=ZERO)
    assert not (tmp_path / "all-skipped").exists() and not (tmp_path / "no-sources").exists()


def test_cli_resume_with_nothing_left_creates_no_directory(baselines: dict[str, Path], tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    base = [baselines["b"]]
    assert _cli(base, "--run-name", "done") == 0
    capsys.readouterr()
    assert _cli(base, "--run-name", "again", "--resume-from", str(tmp_path / "runs" / "done")) == 0
    assert not (tmp_path / "runs" / "again").exists()
    assert "nothing" in capsys.readouterr().err


# ---------------------------------------------------------------- 019 §1 / §4：CLI 接受多個 baseline 目錄


@pytest.mark.parametrize("form", ["repeated", "list"])
def test_the_cli_keeps_every_baseline_directory(form: str, baselines: dict[str, Path], tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = str(baselines["a"]), str(baselines["b"])
    given = ["--baseline-run", a, "--baseline-run", b] if form == "repeated" else ["--baseline-run", a, b]
    assert main([*given, "--plan", PLAN_TEXT, "--criterion", "wilson95", "--dry-run", "--run-name", "s"]) == 0
    data = json.loads((tmp_path / "runs" / "s" / "run.json").read_text(encoding="utf-8"))
    assert data["sources"] == _ids(select_sources([baselines["a"], baselines["b"]]))
    out = tmp_path / "analysis"
    from experiments.analyze_sweep import main as analyze_main

    assert analyze_main([str(tmp_path / "runs" / "s"), *given, "--out", str(out)]) == 0
    assert json.loads((out / "analysis.json").read_text(encoding="utf-8"))["baseline_runs"] == ["a", "b"]
