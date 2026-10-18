"""experiments/results_report.py（正式結果頁與 README 的結果段落；decision 033）的契約測試。

以假的 client 跑出「正式形狀」的 run 目錄（eval baseline R = 3 + 登記的 sweep plan，非 dry-run、有 calls.jsonl），再產生報告。
不連網、不使用 GPU。
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import local_agent_config
from experiments import results_report
from experiments.results_report import (
    README_END,
    README_START,
    build_summary,
    embed_results,
    recoverability_svg,
    render_page,
    render_readme_block,
    write_report,
)
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import REGISTERED_CRITERION, REGISTERED_PLAN, run_sweeps, select_sources
from tasks.generator import week1_pool

ROOT = Path(__file__).resolve().parents[1]
ZERO = lambda: 0.0  # noqa: E731
EVAL = week1_pool("eval")


class _SometimesQuits(OllamaOracleFakeClient):
    """每 7 次「已經有 assistant turn」的呼叫中，有一次直接回答 Done.（沒有做完）→ 部分 rerun 失敗。"""

    def __init__(self) -> None:
        super().__init__(EVAL)
        self.count = 0

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        response = super().chat(request)
        if any(m["role"] == "assistant" for m in request["messages"]):
            self.count += 1
            if self.count % 7 == 0:
                response["message"] = {"role": "assistant", "content": "Done."}
        return response


def _runs(base: Path, client_for_sweep: Any) -> tuple[Path, Path]:
    baseline = base / "eval"
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=baseline, clock=ZERO, agent_config=local_agent_config(), rollouts=3,
              call_log=True)
    sweep = base / "sweep"
    run_sweeps(select_sources(baseline), client_for_sweep, plan=REGISTERED_PLAN, criterion=REGISTERED_CRITERION, run_dir=sweep,
               clock=ZERO, call_log=True, replicates=200)
    return baseline, sweep


@pytest.fixture(scope="module")
def clean_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return _runs(tmp_path_factory.mktemp("clean"), OllamaOracleFakeClient(EVAL))


@pytest.fixture(scope="module")
def failing_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return _runs(tmp_path_factory.mktemp("failing"), _SometimesQuits())


@pytest.fixture(scope="module")
def clean_summary(clean_runs: tuple[Path, Path]) -> dict[str, Any]:
    baseline, sweep = clean_runs
    return build_summary([baseline], [sweep], replicates=200)


def test_the_summary_records_provenance_and_every_seed(clean_summary: dict[str, Any]) -> None:
    provenance = clean_summary["provenance"]
    assert provenance["config_id"] == local_agent_config().config_id
    assert provenance["model"] == "qwen3.5:9b-q4_K_M" and provenance["dry_run"] is False
    calls = clean_summary["calls"]
    for part in ("baseline", "sweep"):
        stats = calls[part]
        assert stats["calls"] > 0 and stats["seeds"] == stats["calls"] and stats["complete"] is True
        assert stats["matched_attempts"] == stats["attempts"] > 0
    json.dumps(clean_summary)  # JSON-native


def test_the_summary_has_the_baseline_metrics_and_every_source(clean_summary: dict[str, Any]) -> None:
    metrics = {m["name"]: m for m in clean_summary["baseline"]["metrics"]}
    assert metrics["success_rate"]["value"] == 1.0 and metrics["trajectory_safe_rate"]["value"] == 1.0
    assert clean_summary["baseline"]["rollouts"] == 45
    sources = clean_summary["sweep"]["sources"]
    assert len(sources) == 15 and {s["task_id"] for s in sources} == {i.task.task_id for i in EVAL}
    for source in sources:
        assert source["points"] and all(0 <= p["ci_low"] <= p["p_hat"] <= p["ci_high"] <= 1 for p in source["points"])
    assert clean_summary["sweep"]["gate"]["outcome"] in ("advanced", "gradual", "no_repeated_pattern", "no_data")


def test_failed_reruns_become_failure_cases(failing_runs: tuple[Path, Path]) -> None:
    baseline, sweep = failing_runs
    summary = build_summary([baseline], [sweep], replicates=200)
    cases = summary["failure_cases"]["reruns"]
    assert cases and summary["failure_cases"]["rerun_failures"] >= len(cases)
    case = cases[0]
    assert {"task_id", "k", "rollout_index", "outcome", "termination", "failed_checks", "last_message"} <= set(case)
    assert case["outcome"] != "success"
    assert any(item.startswith(("outcome #", "constraint #")) for c in cases for item in c["failed_checks"])  # grader 的檢查
    assert any(c["last_message"] == "Done." for c in cases)  # 提早回答的那一則
    page = render_page(summary)
    assert case["task_id"] in page and "失敗" in page


def test_the_page_states_the_gate_and_does_not_overclaim(clean_summary: dict[str, Any]) -> None:
    page = render_page(clean_summary)
    gate = clean_summary["sweep"]["gate"]["outcome"]
    assert f"`{gate}`" in page and "decision 025" in page
    if gate != "advanced":
        assert "不進 Advanced" in page
    assert "15 題" in page and clean_summary["provenance"]["config_id"] in page


def test_the_chart_is_well_formed_and_accessible(clean_summary: dict[str, Any]) -> None:
    svg = recoverability_svg(clean_summary)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert root.find(f"{ns}title") is not None and root.find(f"{ns}desc") is not None
    panels = [g for g in root.iter(f"{ns}g") if g.get("class") == "panel"]
    assert len(panels) == 15
    assert "prefers-color-scheme: dark" in svg  # 深色模式有自己的設定
    assert svg.count("<circle") == sum(len(s["points"]) for s in clean_summary["sweep"]["sources"])


def test_the_report_is_deterministic(clean_runs: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, sweep = clean_runs
    first, second = tmp_path / "a", tmp_path / "b"
    write_report([baseline], [sweep], first, replicates=200)
    write_report([baseline], [sweep], second, replicates=200)
    for name in ("README.md", "recoverability.svg", "summary.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_dry_run_inputs_never_reach_the_readme(tmp_path: Path) -> None:
    baseline = tmp_path / "eval"
    run_split("eval", OllamaOracleFakeClient(EVAL), run_dir=baseline, clock=ZERO, agent_config=local_agent_config(),
              dry_run=True)
    sweep = tmp_path / "sweep"
    run_sweeps(select_sources(baseline), OllamaOracleFakeClient(EVAL), plan=REGISTERED_PLAN, criterion=REGISTERED_CRITERION,
               run_dir=sweep, clock=ZERO, dry_run=True, replicates=200)
    summary = build_summary([baseline], [sweep], replicates=200)
    assert summary["provenance"]["dry_run"] is True and "不是研究結果" in render_page(summary)
    with pytest.raises(ValueError):
        render_readme_block(summary)
    out = tmp_path / "out"  # CLI：沒有 --allow-dry-run → 什麼都不寫
    with pytest.raises(SystemExit):
        results_report.main(["--baseline-run", str(baseline), "--sweep-run", str(sweep), "--out", str(out), "--replicates", "200"])
    assert not out.exists()


def test_the_readme_block_is_embedded_between_its_markers(clean_summary: dict[str, Any]) -> None:
    block = render_readme_block(clean_summary)
    text = f"intro\n{README_START}\nold\n{README_END}\noutro\n"
    once = embed_results(text, block)
    assert once == f"intro\n{README_START}\n{block}\n{README_END}\noutro\n" and embed_results(once, block) == once
    with pytest.raises(ValueError):
        embed_results("no markers", block)


def test_the_committed_results_match_the_readme() -> None:
    summary_path = ROOT / "docs" / "results" / "summary.json"
    if not summary_path.exists():
        pytest.skip("no formal results yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert embed_results(readme, render_readme_block(summary)) == readme
    assert (ROOT / "docs" / "results" / "README.md").read_text(encoding="utf-8") == render_page(summary)
    assert (ROOT / "docs" / "results" / "recoverability.svg").read_text(encoding="utf-8") == recoverability_svg(summary)


def test_the_cli_writes_the_report(clean_runs: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, sweep = clean_runs
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{README_START}\n{README_END}\n", encoding="utf-8")
    out = tmp_path / "results"
    assert results_report.main(["--baseline-run", str(baseline), "--sweep-run", str(sweep), "--out", str(out),
                                "--readme", str(readme), "--replicates", "200"]) == 0
    assert (out / "README.md").exists() and README_START in readme.read_text(encoding="utf-8")
    assert "gate" in json.loads((out / "summary.json").read_text(encoding="utf-8"))["sweep"]


def test_tool_errors_the_model_fixed_are_shown_with_the_next_call(tmp_path: Path) -> None:
    """模型先送出不合法的參數（limit 999）、下一步照 oracle 走：以 ScriptedPolicy 在真正的 env 上產生 trajectory。"""
    from agents.policy import ScriptedPolicy
    from replay.runner import run_episode

    run_dir = tmp_path / "eval"
    (run_dir / "trajectories").mkdir(parents=True)
    for instance in EVAL[:2]:
        script = [[("search_issues", {"limit": 999})], *[[call] for call in instance.oracle_calls]]
        trajectory = run_episode(instance.task, ScriptedPolicy(script), max_turns=20, clock=ZERO)
        trajectory.save(run_dir / "trajectories" / f"{instance.task.task_id}-r0.json")
    counts, examples = results_report._tool_error_examples(results_report._trajectories([run_dir]))
    assert counts == {"search_issues/invalid_arguments": 2}
    [example] = examples  # 同一種錯誤只舉一個例子
    assert example["call"] == {"tool": "search_issues", "args": {"limit": 999}} and example["error"]
    assert example["next_call"]["tool"] == EVAL[0].oracle_calls[0][0] and example["next_call"]["ok"] is True
    assert example["fixed_next"] is (EVAL[0].oracle_calls[0][0] == "search_issues")  # 同一個 tool 而且成功才算修正
    page = render_page({**_minimal_summary(), "failure_cases": {"rerun_failures": 0, "reruns": [], "baseline": [],
                                                                "tool_error_examples": examples}})
    assert "limit" in page and "999" in page


def _minimal_summary() -> dict[str, Any]:
    return {
        "provenance": {"dry_run": False, "model": "m", "model_digest": "d", "ollama_version": "v", "think": False,
                       "config_id": "c", "options": {}, "baseline_runs": ["b"], "sweep_runs": ["s"], "dates": ["2026-10-16"]},
        "calls": {"baseline": None, "sweep": None},
        "baseline": {"rollouts": 0, "metrics": [], "families": [], "outcome_counts": {}, "attempt_counts": {}, "missing_tasks": [],
                     "interrupted_runs": [], "success_but_unsafe": 0, "partial_safety": 0, "tool_error_counts": {}},
        "sweep": {"plan": {}, "criterion": "bonferroni95", "gate": {"outcome": "no_data", "data_complete": True}, "cliff_rate": {},
                  "mean_largest_drop": {}, "reruns": 0, "rerun_outcomes": {}, "not_completed": [], "missing_sources": [],
                  "partial_sweeps": [], "partial_safety": 0, "interrupted_runs": [], "stopped_by_budget": [], "resumed": [],
                  "sources": []},
        "walkthrough": None,
        "failure_cases": {"rerun_failures": 0, "reruns": [], "baseline": [], "tool_error_examples": []},
    }


def test_the_walkthrough_shows_a_formal_trajectory_step_by_step(clean_summary: dict[str, Any]) -> None:
    walk = clean_summary["walkthrough"]
    longest = max(clean_summary["sweep"]["sources"], key=lambda s: (s["n_turns"], s["task_id"]))
    assert walk["task_id"] == longest["task_id"] and len(walk["turns"]) == longest["n_turns"]
    assert walk["instruction"] and walk["grade"]["success"] is True
    calls = [call for turn in walk["turns"] for call in turn["calls"]]
    assert calls and all({"tool", "args", "ok", "changes"} <= set(call) for call in calls)
    assert any(call["changes"] for call in calls)  # 至少一步改變了狀態
    page = render_page(clean_summary)
    assert "一個正式 trajectory" in page and walk["task_id"] in page


def _hand_summary(sources: list[dict[str, Any]], *, gate: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = _minimal_summary()
    summary["sweep"].update(sources=sources, gate=gate if gate is not None else {"outcome": "no_repeated_pattern",
                                                                                 "data_complete": True})
    return summary


def _source(task_id: str, family: str, ks: list[int], p: list[float], *, cliff_k: int | None = None) -> dict[str, Any]:
    return {"task_id": task_id, "family": family, "n_turns": max(ks) + 1, "is_cliff": cliff_k is not None,
            "largest_drop": 0.8 if cliff_k is not None else 0.0, "largest_drop_k": cliff_k if cliff_k is not None else 0,
            "max_turns_confounded": False, "points": [{"k": k, "n": 10, "successes": int(v * 10), "p_hat": v,
                                                      "ci_low": max(0.0, v - 0.2), "ci_high": min(1.0, v + 0.1)}
                                                     for k, v in zip(ks, p)]}


def test_the_chart_marks_a_cliff_between_its_two_checkpoints() -> None:
    sources = [_source("gen-a-s1", "gen/a", [0, 1, 2, 3, 4, 5], [1, 1, 1, 0.9, 0.1, 0.1], cliff_k=3),
               _source("gen-b-s1<&>", "gen/b", [0], [1.0])]
    svg = recoverability_svg(_hand_summary(sources))
    root = ET.fromstring(svg)  # 特殊字元已跳脫
    ns = "{http://www.w3.org/2000/svg}"
    panel = next(g for g in root.iter(f"{ns}g") if g.get("aria-label") == "gen-a-s1")
    circles = {round(float(c.get("cx")), 1): c for c in panel.iter(f"{ns}circle")}
    xs = sorted(circles)
    cliff = next(line for line in panel.iter(f"{ns}line") if line.get("class") == "cliff")
    assert xs[3] < float(cliff.get("x1")) < xs[4]  # 虛線在 k = 3 與 k = 4 之間
    assert any("（cliff）" in (t.text or "") for t in panel.iter(f"{ns}text"))
    whiskers = [line for line in panel.iter(f"{ns}line") if line.get("class") == "ci"]
    grid = [float(line.get("y1")) for line in panel.iter(f"{ns}line") if line.get("class") == "grid"]
    y0, y1 = max(grid), min(grid)  # p = 0 與 p = 1 的位置
    for whisker, point in zip(whiskers, sources[0]["points"]):
        assert float(whisker.get("y1")) == pytest.approx(y0 - (y0 - y1) * point["ci_low"], abs=0.11)
        assert float(whisker.get("y2")) == pytest.approx(y0 - (y0 - y1) * point["ci_high"], abs=0.11)


def test_a_small_chart_keeps_its_header_inside_the_canvas() -> None:
    svg = recoverability_svg(_hand_summary([_source("gen-a-s1", "gen/a", [0], [1.0])]))
    root = ET.fromstring(svg)
    assert float(root.get("width")) >= results_report.MIN_SVG_WIDTH


def test_incomplete_or_unevaluated_results_never_reach_the_readme() -> None:
    incomplete = _hand_summary([_source("gen-a-s1", "gen/a", [0], [1.0])], gate={"outcome": "gradual", "data_complete": False})
    assert "資料不完整" in render_page(incomplete) and "資料不完整" in recoverability_svg(incomplete)
    with pytest.raises(ValueError):
        render_readme_block(incomplete)
    blocked = _hand_summary([_source("gen-a-s1", "gen/a", [0], [1.0])])
    blocked["sweep"]["gate"] = None
    assert "Gate 不評估" in render_page(blocked) and "不評估" in recoverability_svg(blocked)
    with pytest.raises(ValueError):
        render_readme_block(blocked)


def test_the_page_attaches_the_preregistered_analyses(clean_runs: tuple[Path, Path], tmp_path: Path) -> None:
    from experiments.analyze_baseline import analyze_runs, render_markdown as baseline_markdown
    from experiments.analyze_sweep import analyze_sweep_runs, render_markdown as sweep_markdown

    baseline, sweep = clean_runs
    write_report([baseline], [sweep], tmp_path / "out", replicates=200)
    assert (tmp_path / "out" / "baseline-analysis.md").read_text(encoding="utf-8") == baseline_markdown(
        analyze_runs([baseline], replicates=200))
    assert (tmp_path / "out" / "sweep-analysis.md").read_text(encoding="utf-8") == sweep_markdown(
        analyze_sweep_runs([sweep], [baseline], replicates=200))
    page = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "baseline-analysis.md" in page and "sweep-analysis.md" in page and "Wilson" in page and "Bonferroni）" not in page




def test_the_cli_never_overwrites_the_sealed_round1_outputs(clean_runs: tuple[Path, Path], tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """第一輪已封存（docs/results/round1-seal.json）：CLI 拒絕寫進封存的 docs/results 與 README，而且在寫任何檔案之前就拒絕（結案稽核）。

    封存的路徑換成暫存目錄中的副本，所以即使檢查失效，也不會寫到 repo 本身。"""
    assert results_report.SEALED_RESULTS == ROOT / "docs" / "results" and results_report.SEALED_README == ROOT / "README.md"
    sealed = tmp_path / "repo" / "docs" / "results"
    sealed.mkdir(parents=True)
    (sealed / "README.md").write_text("sealed page\n", encoding="utf-8")
    readme = tmp_path / "repo" / "README.md"
    original = f"# x\n{README_START}\nsealed\n{README_END}\n"
    readme.write_text(original, encoding="utf-8")
    monkeypatch.setattr(results_report, "SEALED_RESULTS", sealed)
    monkeypatch.setattr(results_report, "SEALED_README", readme)
    baseline, sweep = clean_runs
    args = ["--baseline-run", str(baseline), "--sweep-run", str(sweep), "--replicates", "200"]
    monkeypatch.chdir(tmp_path / "repo")  # 封存頁「重現」中的相對路徑寫法
    for out in (str(sealed), str(sealed) + "/.", str(sealed / ".." / "results"), str(tmp_path / "repo" / "docs" / "RESULTS"), "docs/results"):
        with pytest.raises(SystemExit):
            results_report.main([*args, "--out", out])
        assert "sealed" in capsys.readouterr().err
    for readme_arg in (str(readme), "README.md", "./README.md"):
        with pytest.raises(SystemExit):
            results_report.main([*args, "--out", str(tmp_path / "r"), "--readme", readme_arg])
        assert "sealed" in capsys.readouterr().err and not (tmp_path / "r").exists()
    assert [p.name for p in sealed.iterdir()] == ["README.md"] and (sealed / "README.md").read_text(encoding="utf-8") == "sealed page\n"
    assert readme.read_text(encoding="utf-8") == original
