"""experiments/study3_report.py（decision 036 §6「分析與報告」）的契約測試。

測試資料：oracle 假 client 在 study3 eval pool 上的 dry-run 輸出（部分題目改用失敗的 negative 當 oracle），以及在它的分析摘要上
改寫 provenance、RQ1、RQ2 與 sweep 狀態的合成摘要（呈現只依摘要，不重算判定）。所有輸出目錄、結果目錄（本模組的 RESULT_DIRS、
study2_report 的 ROUND1_RESULTS / RESULT_DIRS / STUDY3_RESULT_DIRS）與 README 一律換成 tmp 路徑：測試不寫 repo 的 docs/results/ 或
README.md（只讀 README.md 當副本的來源），不讀 repo 的 runs/，也不輸出任何 eval 的內容（decision 036 §4）。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from agents.ollama import local_agent_config
from experiments import study2_report as rep2
from experiments import study3_analysis as s3
from experiments import study3_plan as plan
from experiments import study3_report as rep
from experiments.run_baseline import OllamaOracleFakeClient, run_split
from experiments.run_sweep import run_sweeps, select_sources
from experiments.study3_dev import agent_config_id
from replay.localization import SweepPlan
from tasks.generator_v3 import FAMILIES, LEVELS
from tasks.suites import eval_task_ids, suite_pool

ROOT = Path(__file__).resolve().parent.parent
SMALL = SweepPlan(coarse_stride=6, n_coarse=1, n_fine=1, drop_threshold=0.3)
FAILING = {  # task → 改用的失敗 negative（只以程式判定，不輸出內容）
    "gen3-epic-closeout-H2-s2": "E5:wrong-tracker-text",
    "gen3-epic-closeout-H1-s2": "E5:blocks-instead-of-dropped",
    "gen3-scoped-triage-H2-s2": "S7:secdocs-treated-as-denied",
    "gen3-verified-close-H1-s2": "V3:close-qa-broken",
}
LABEL_WORDS = ("以驟降為主", "以逐步為主", "無法判斷")
FREEZE = "sha256:" + "0" * 64


def ZERO() -> float:  # noqa: N802
    return 0.0


def _with(pool: tuple, replaced: dict[str, str]) -> tuple:
    return tuple(dataclasses.replace(i, oracle_calls=dict(i.negative_calls)[replaced[i.task.task_id]]) if i.task.task_id in replaced else i
                 for i in pool)


def _record(path: Path) -> Path:
    path.write_text(json.dumps({"format": plan.FREEZE_FORMAT, "sweep_permutation": plan.sweep_permutation(eval_task_ids("study3")),
                                "reserve": {"local_calls": 1, "haiku_usd": 1.0}}), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """(runs 根目錄, sweep 目錄, 凍結紀錄)：dry-run 的 eval baseline（FAILING 的題目失敗，R = 1），以及只 sweep 前 2 條失敗 source 的 sweep。"""
    base = tmp_path_factory.mktemp("s3r")
    root = base / "runs"
    pool = _with(suite_pool("study3", "eval"), FAILING)
    run_split("eval", OllamaOracleFakeClient(pool), run_dir=root / "b", clock=ZERO, suite="study3",
              agent_config=local_agent_config(), dry_run=True, rollouts=1)
    record = _record(base / "record.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plan, "FREEZE_RECORD", record)
        mp.setattr(plan, "MAX_SOURCES", 2)
        run_sweeps(select_sources(root / "b"), OllamaOracleFakeClient(pool), plan=SMALL, criterion="bonferroni95", run_dir=base / "sweep",
                   clock=ZERO, agent_config=local_agent_config(), dry_run=True, order="frozen_permutation", call_log=True)
    return root, base / "sweep", record


@pytest.fixture(scope="module")
def dry(runs: tuple[Path, Path, Path]) -> dict[str, Any]:
    root, sweep, record = runs
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plan, "FREEZE_RECORD", record)
        mp.setattr(plan, "MAX_SOURCES", 2)
        summary = rep.build("local", root, [sweep], dry_run=True, replicates=50)
    assert summary["analysis"]["sweep"]["state"] == "complete_full" and len(summary["analysis"]["curves"]) == 2
    return summary


def _formal(summary: dict[str, Any], agent: str = "local") -> dict[str, Any]:
    """合成的正式摘要（只用於呈現的測試）：provenance 改成非 dry-run、B = 10,000、有凍結紀錄。"""
    out = copy.deepcopy(summary)
    out["analysis"]["provenance"].update(dry_run=False, replicates=10_000, freeze_record_sha256=FREEZE, agent=agent,
                                         config_id=agent_config_id(agent), runs_root=str(s3.RUNS_ROOT))  # 正式分析一律列舉 repo 的 runs/
    if agent == "haiku":
        out["records"] = {"model": "claude-haiku-5-5", "model_digest": None, "ollama_version": None, "calls_baseline": None,
                          "calls_sweep": None, "spent_usd": {"baseline": 1.25, "sweep": 0.0}}
    return out


@pytest.fixture
def local(dry: dict[str, Any]) -> dict[str, Any]:
    return _formal(dry)


@pytest.fixture
def haiku(dry: dict[str, Any]) -> dict[str, Any]:
    return _formal(dry, "haiku")


def _curve(cls: str, i: int) -> dict[str, Any]:
    return {"task_id": f"t{i}", "class": cls, "classes": {"round1": cls, "excluding": cls}, "largest_drop_k": 1,
            "first_irreversible_turn": None, "source_termination": "final_answer", "qualifying_drop_below_decidable": False,
            "source_success": False, "sensitivity": {"decidable": cls != "undecidable", "is_cliff": cls.startswith("cliff")},
            "agent_error_sensitivity": {"class": "cliff" if cls.startswith("cliff") else cls}}


def _state(summary: dict[str, Any], classes: list[str], *, state: str = "complete_full", target: int | None = None,
           end_reason: str | None = None, not_started: int = 0, partial: dict[str, Any] | None = None,
           blocked: str | None = None) -> dict[str, Any]:
    """以 study3_analysis 的 rq1 / secondary / rq1_sensitivity 在合成的曲線類別上重算 RQ1、次要結果與敏感度，放進摘要（其他部分不變）。"""
    out = copy.deepcopy(summary)
    a = out["analysis"]
    curves = [_curve(c, i) for i, c in enumerate(classes)]
    target = len(classes) if target is None else target
    r1 = s3.rq1(curves, state=state, target=target, end_reason=end_reason, blocked=blocked)
    r1.update(n_failed_sources=max(target, len(classes)), baseline_incomplete=False, baseline_gaps=None)
    a["rq1"], a["secondary"] = r1, s3.secondary(curves, r1)
    a["sensitivity"].update(s3.rq1_sensitivity(curves, r1), cap_range=r1["cap_range"])
    a["sweep"].update(state=state, end_reason=end_reason, target=target, partial=partial,
                      not_started=[f"n{i}" for i in range(not_started)])
    return out


def _rq1_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("- RQ1："))


def _patch_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """封存的 docs/results、Study 2 / 035 與 Study 3 的結果目錄全部換成 tmp 中的路徑（即使守衛失效也不會寫到 repo）。"""
    results = tmp_path / "results"
    monkeypatch.setattr(rep2, "ROUND1_RESULTS", results)
    monkeypatch.setattr(rep2, "RESULT_DIRS", {"study2": results / "study2", "study2-haiku": results / "study2-haiku"})
    study3 = {"study3": results / "study3", "study3-haiku": results / "study3-haiku"}
    monkeypatch.setattr(rep2, "STUDY3_RESULT_DIRS", study3)
    monkeypatch.setattr(rep, "RESULT_DIRS", dict(study3))
    return results


def _readme(tmp_path: Path) -> Path:
    """repo 的 README，去掉已 commit 的 Study 3 段落（正式結果寫入之後也從「還沒有 Study 3 段落」的 README 開始）。"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for start, end in [(v["start"], v["end"]) for v in rep.VARIANTS.values()] + [(rep.COMPARE["start"], rep.COMPARE["end"])]:
        text = re.sub(r"\n\n" + re.escape(start) + r".*?" + re.escape(end), "", text, flags=re.S)
    path = tmp_path / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ 歸屬：(suite, agent)


def test_the_variant_comes_from_suite_and_agent(dry: dict[str, Any], local: dict[str, Any], haiku: dict[str, Any]) -> None:
    assert rep.variant(dry) == rep.variant(local) == "study3" and rep.variant(haiku) == "study3-haiku"
    swapped = copy.deepcopy(local)
    swapped["analysis"]["provenance"]["config_id"] = agent_config_id("haiku")  # config_id 不能單獨決定歸屬
    with pytest.raises(ValueError, match="config_id"):
        rep.variant(swapped)
    study2 = copy.deepcopy(local)
    study2["analysis"]["provenance"]["suite"] = "study2"
    with pytest.raises(ValueError, match="study2"):
        rep.variant(study2)
    with pytest.raises(ValueError):
        rep.variant({**local, "format": "agent-gym/study2-report/v1"})
    assert rep.RESULT_DIRS == {"study3": rep2.ROUND1_RESULTS / "study3", "study3-haiku": rep2.ROUND1_RESULTS / "study3-haiku"}
    assert rep2.STUDY3_RESULT_DIRS == rep.RESULT_DIRS  # study2_report 的反向守衛使用同一組目錄
    assert {v["agent"] for v in rep.VARIANTS.values()} == {"local", "haiku"}


# ------------------------------------------------------------------ 檔案與 dry-run


def test_a_dry_run_report_writes_three_files_and_never_the_readme(dry: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dirs(monkeypatch, tmp_path)
    readme = _readme(tmp_path)
    before = readme.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="dry-run"):
        rep.write(dry, tmp_path / "out", readme=readme)
    assert readme.read_text(encoding="utf-8") == before and not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="dry-run"):  # dry-run 不寫進任何一個結果目錄
        rep.write(dry, rep.RESULT_DIRS["study3"])
    rep.write(dry, tmp_path / "out")
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == ["README.md", "recoverability.svg", "summary.json"]
    saved = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(dry)) and saved["format"] == rep.REPORT_FORMAT
    page = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    svg = (tmp_path / "out" / "recoverability.svg").read_text(encoding="utf-8")
    assert page == rep.render_page(saved) and svg == rep.recoverability_svg(saved)  # 由 summary.json 逐 byte 重現
    assert "> **dry-run" in page and "[dry-run]" in svg
    assert svg.count('class="panel"') == len(dry["analysis"]["curves"])
    for text in (page, svg, json.dumps(saved, ensure_ascii=False)):
        assert "Advanced" not in text and "advanced" not in text


def test_the_cli_builds_the_same_summary(runs: tuple[Path, Path, Path], dry: dict[str, Any], tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    root, sweep, record = runs
    _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(plan, "FREEZE_RECORD", record)
    monkeypatch.setattr(plan, "MAX_SOURCES", 2)
    assert rep.main(["--agent", "local", "--runs-root", str(root), "--sweep-run", str(sweep), "--out", str(tmp_path / "o"), "--dry-run",
                     "--replicates", "50"]) == 0
    assert json.loads((tmp_path / "o" / "summary.json").read_text(encoding="utf-8")) == json.loads(json.dumps(dry))  # 決定性


# ------------------------------------------------------------------ 結果頁的結構


def test_the_page_structure(local: dict[str, Any]) -> None:
    page = rep.render_page(local)
    assert page.startswith("# Study 3：本機模型") and "不做 Core Gate 的判定" in page and "dry-run" not in page
    headings = [line for line in page.splitlines() if line.startswith("## ")]
    names = [h.split("（")[0].split("：")[0] for h in headings]
    assert names == ["## 結論", "## 目錄", "## eval baseline", "## comment 稽核", "## source", "## sweep", "## RQ1", "## RQ1 的次要結果", "## RQ3", "## 曲線",
                     "## 分母的組成", "## 已知限制", "## 設定與紀錄"]
    assert "analysis.json" not in page and "[recoverability.svg](recoverability.svg)" in page and "[summary.json](summary.json)" in page
    assert rep.COMPARE_NOTE in page and local["analysis"]["provenance"]["config_id"] in page
    for lv in LEVELS:
        assert f"| {lv} |" in page
    haiku_page = rep.render_page(_formal(local, "haiku"))
    assert haiku_page.startswith("# Study 3：Claude Haiku 5.5") and "$1.2500" in haiku_page


def test_the_known_limitations_are_on_both_pages(local: dict[str, Any]) -> None:
    """dev 審查（eval 之前，decision 036「實作紀錄」）判定不是缺陷、但寫進報告的 6 項候選疑慮：兩個 agent 的結果頁都列出，固定文字。"""
    for page in (rep.render_page(local), rep.render_page(_formal(local, "haiku"))):
        section = page[page.index("## 已知限制"):page.index("## 設定與紀錄")]
        assert [line[:3] for line in section.splitlines() if line[:1].isdigit()] == ["1. ", "2. ", "3. ", "4. ", "5. ", "6. "]
        for phrase in ("limit 100", "get_issue", "qa-broken", "rule 2", "S7", "status"):
            assert phrase in section
        assert "不是題目缺陷" in section and "decision 036「實作紀錄」" in section and "不改變評分" in section


# ------------------------------------------------------------------ RQ1 的措辭（§2、§6）


def test_three_way_labels_and_small_n_wording(local: dict[str, Any]) -> None:
    nine = _state(local, ["cliff_behavioral"] * 8 + ["gradual"])
    block, page = rep.render_readme_block(nine), rep.render_page(nine)
    for text in (block, page):
        line = _rq1_line(text)
        assert line.startswith("- RQ1：以驟降為主（cliff 比例的 Wilson 95% CI 下限 > 0.5）：可判定的失敗曲線中 cliff 8 / 9，Wilson 95% CI [")
        assert "樣本少（n = 9），證據弱" in text and "n ≤ 5" not in text
    assert "樣本少（n = 9），證據弱" in rep.render_compare_block(nine, None)
    four = rep.render_readme_block(_state(local, ["cliff_mechanical"] * 4))
    assert "樣本少（n = 4），證據弱" in four and "n ≤ 5：只有全部都是 cliff 才可能判為以驟降為主" in four
    even = _rq1_line(rep.render_readme_block(_state(local, ["cliff_behavioral"] * 5 + ["gradual"] * 5)))
    assert even.startswith("- RQ1：無法判斷（Wilson 95% CI 包含 0.5）：")
    gradual = rep.render_readme_block(_state(local, ["gradual"] * 10 + ["undecidable", "max_turns_confounded"]))
    assert _rq1_line(gradual).startswith("- RQ1：以逐步為主（cliff 比例的 Wilson 95% CI 上限 < 0.5）：可判定的失敗曲線中 cliff 0 / 10")
    assert "無法判定 1 條、max_turns 混淆 1 條" in gradual and "樣本少" not in gradual
    none = rep.render_readme_block(_state(local, ["undecidable"]))
    assert _rq1_line(none).startswith("- RQ1：沒有可判定的曲線") and "樣本少（n = 0），證據弱" in none


def test_the_interpretation_sentence_is_filled_mechanically(local: dict[str, Any]) -> None:
    mostly_mechanical = _state(local, ["cliff_mechanical"] * 9 + ["cliff_behavioral"] * 2)  # m > k
    for text in (rep.render_readme_block(mostly_mechanical), rep.render_page(mostly_mechanical)):
        assert "排除 9 條機械性 cliff（不可撤回的 comment，題目設計造成）之後，行為性 cliff 為 2 / 2（Wilson 95% CI [" in text
        assert "以驟降為主的判定依賴這 9 條機械性 cliff。" in text and "行為性 cliff 占全部 cliff 的比例為 2 / 11。" in text
        assert "驟降的曲線大多是機械性 cliff，驟降的判定主要來自不可撤回的 comment。" in text
    balanced = rep.render_readme_block(_state(local, ["cliff_mechanical"] * 4 + ["cliff_behavioral"] * 5 + ["gradual"]))  # m ≤ k
    assert "排除 4 條機械性 cliff" in balanced and "5 / 6（Wilson 95% CI [" in balanced and "主要來自" not in balanced
    behavioral = rep.render_readme_block(_state(local, ["cliff_behavioral"] * 10))  # 行為性比例本身已超過 0.5：不加解讀句
    assert "排除" not in behavioral and "行為性 cliff 比例（次要結果，不取代主要判定）：10 / 10" in behavioral


def test_the_stop_states(local: dict[str, Any]) -> None:
    capped = rep.render_readme_block(_state(local, ["cliff_behavioral"] * 6 + ["gradual"] * 2, state="complete_cap_mid_source", target=25,
                                            end_reason="cap_mid_source", not_started=3,
                                            partial={"task_id": "gen3-sync-repair-H3-s5", "rollouts": 4, "points": []}))
    assert _rq1_line(capped).startswith("- RQ1：無法判斷（Wilson 95% CI 包含 0.5）（因上限中途停止）：可判定的失敗曲線中 cliff 6 / 8")
    assert "當成 cliff 7 / 9" in capped and "當成非 cliff 6 / 9" in capped and "之後沒有開始的 source 3 條" in capped
    assert "**資料不完整" not in capped  # 因上限中途停止是完整的一種（§6）
    early = rep.render_readme_block(_state(local, ["cliff_behavioral"] * 3, state="incomplete_early_end", target=10, end_reason="early_end"))
    line = _rq1_line(early)
    assert line.startswith("- RQ1：資料不完整（提前結束）：不輸出三分法的標籤。") and not any(w in line for w in LABEL_WORDS)
    assert "全部當成 cliff 10 / 10" in early and "全部當成非 cliff 3 / 10" in early and "**資料不完整。**" in early
    ceiling = _state(local, [], state="ceiling", target=0)
    ceiling["analysis"]["baseline"]["levels"]["H2"]["agent_error"] = 2
    text = rep.render_readme_block(ceiling)
    assert _rq1_line(text).startswith("- RQ1：天花板：沒有任何失敗的 source，而且 baseline 完整") and "agent_error 共 2 個 episode" in text
    assert not any(w in _rq1_line(text) for w in LABEL_WORDS)
    no_curve = rep.render_readme_block(_state(local, [], state="incomplete_no_curve", target=4))
    assert _rq1_line(no_curve).startswith("- RQ1：資料不完整：有失敗的 source（4 條），但前綴中沒有完成的曲線") and "**資料不完整。**" in no_curve
    blocked = copy.deepcopy(local)
    blocked["analysis"]["rq1"] = {**blocked["analysis"]["rq1"], "outcome": "blocked", "label": None, "small_n": None, "small_n_rule": None}
    assert _rq1_line(rep.render_readme_block(blocked)).startswith("- RQ1：不合併估計")


RAW_LABELS = (*LABEL_WORDS, *s3.LABELS)


def test_an_early_end_outputs_no_three_way_label_anywhere(local: dict[str, Any]) -> None:
    """§6：提前結束時「不論已完成的曲線有多少，都不輸出三分法的標籤」：次要結果與敏感度也只報 k / n 與 CI。"""
    early = _state(local, ["cliff_behavioral"] * 7 + ["gradual"], state="incomplete_early_end", target=25, end_reason="early_end")
    a = early["analysis"]
    assert a["secondary"]["behavioral"]["label"] is None and a["sensitivity"]["agent_error"]["label"] is None
    assert a["secondary"]["round1_rule"]["label"] is None and "label" not in a["sensitivity"]["max_turns"]
    for text in (rep.render_readme_block(early), rep.render_page(early), s3.render_markdown(a)):
        assert not any(w in text for w in RAW_LABELS)
    assert "行為性 cliff 比例（次要結果，不取代主要判定）：7 / 8（Wilson 95% CI [" in rep.render_readme_block(early)


def test_blocked_pooling_prints_no_pooled_estimate(local: dict[str, Any]) -> None:
    blocked = _state(local, ["cliff_behavioral"] * 8 + ["gradual"] * 2, state="complete_cap_mid_source", target=25,
                     end_reason="cap_mid_source", blocked="more than one response model: ['a', 'b']")
    a = blocked["analysis"]
    assert a["rq1"]["cap_range"] is None and a["secondary"] is None
    assert a["sensitivity"]["max_turns"] is None and a["sensitivity"]["agent_error"] is None
    for text in (rep.render_readme_block(blocked), rep.render_page(blocked), s3.render_markdown(a)):
        assert not any(w in text for w in RAW_LABELS) and "不合併估計" in text
    assert "不合併估計：response model 不只一種（decision 009 §4）" in s3.render_markdown(a).split("## RQ1 的次要結果")[1]


def test_the_secondary_bullets_and_rows(local: dict[str, Any]) -> None:
    out = _state(local, ["cliff_behavioral"] * 9 + ["gradual"])
    line = next(x for x in rep.render_readme_block(out).splitlines() if x.startswith("- 第一輪的規則"))
    assert line.startswith("- 第一輪的規則（decision 017 §1；次要結果）：9 / 10（Wilson 95% CI [") and "→ 以驟降為主" in line
    page = rep.render_page(out)
    assert "第一輪的規則（decision 017 §1）：9 / 10（Wilson 95% CI [" in page
    assert "verified-close 與 qa 角色相關（含 H1 / H3 的 qa-confirmed 干擾項）" in page
    assert "查找與去重（verified-close H3，附錄 A.5 / A.6" in page
    header = next(line for line in page.splitlines() if line.startswith("| 級別 | 題數 | episode |"))
    assert "最大 request（bytes；上限 57,344）" in header  # §3：逐級別報告最大的 request 大小
    assert "最大 request（bytes）" in next(line for line in rep.render_page(_formal(local, "haiku")).splitlines()
                                         if line.startswith("| 級別 | 題數 | episode |"))


def test_formal_reports_come_from_the_repository_runs_root(local: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    """§6：正式報告的分析必須列舉 repo 的 runs/（從磁碟讀回的摘要也一樣）；寫任何檔案之前拒絕。"""
    _patch_dirs(monkeypatch, tmp_path)
    elsewhere = copy.deepcopy(local)
    elsewhere["analysis"]["provenance"]["runs_root"] = str(tmp_path / "runs-copy")
    with pytest.raises(ValueError, match="enumerates every directory"):
        rep.write(elsewhere, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):  # 總帳一律是登記的那一份：沒有 --haiku-journal
        rep.main(["--agent", "haiku", "--out", str(tmp_path / "o"), "--haiku-journal", str(tmp_path / "j.jsonl")])
    assert "--haiku-journal" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_baseline_gaps_are_marked_and_listed(local: dict[str, Any]) -> None:
    gaps = copy.deepcopy(local)
    b = gaps["analysis"]["baseline"]
    b.update(complete=False, missing=[["gen3-sync-repair-H1-s4", 2]], aborted=[["gen3-verified-close-H3-s9", 0]])
    gaps["analysis"]["rq1"].update(baseline_incomplete=True, baseline_gaps={"missing": b["missing"], "aborted": b["aborted"]})
    for text in (rep.render_readme_block(gaps), rep.render_page(gaps)):
        assert "baseline 不完整" in text and "gen3-sync-repair-H1-s4 r2" in text and "gen3-verified-close-H3-s9 r0" in text


# ------------------------------------------------------------------ RQ2 的措辭（§2）


def _levels(values: dict[str, list[int]]) -> dict[str, dict[str, Fraction]]:
    return {lv: {f"{lv}-{i}": Fraction(v, 3) for i, v in enumerate(vs)} for lv, vs in values.items()}


def test_rq2_states_the_point_estimate_the_ci_and_what_valid_means(local: dict[str, Any]) -> None:
    def with_rq2(rq2: dict[str, Any]) -> str:
        out = copy.deepcopy(local)
        out["analysis"]["baseline"]["rq2"] = json.loads(json.dumps(rq2))
        return rep.render_readme_block(out)

    valid = with_rq2(s3.rq2(_levels({"H1": [3] * 32, "H2": [2] * 32, "H3": [0] * 16 + [3] * 16}), complete=True, replicates=200))
    line = next(x for x in valid.splitlines() if x.startswith("- 分級（RQ2）"))
    assert line.startswith("- 分級（RQ2）：有效。H1 − H3 = 0.500，bootstrap 95% CI [") and "精確置換檢定 p = " in line
    assert "「有效」只表示 H1 > H3，不代表差距大小" in line
    incomplete = with_rq2(s3.rq2(_levels({"H1": [3] * 32, "H2": [2] * 32, "H3": [0] * 16 + [3] * 16}), complete=False, replicates=200))
    assert "- 分級（RQ2）：有效（資料不完整）。" in incomplete
    same = with_rq2(s3.rq2(_levels({"H1": [3] * 32, "H2": [3] * 32, "H3": [3] * 32}), complete=True, replicates=200))
    assert "- 分級（RQ2）：無效：不下難度的結論。H1 − H3 = 0.000" in same and "不成立" in same
    short = with_rq2(s3.rq2(_levels({"H1": [3] * 31, "H2": [2] * 32, "H3": [0] * 32}), complete=False, replicates=200))
    assert "- 分級（RQ2）：無法計算" in short and "H1 − H3 的點估計 1.000" in short and "不代表差距大小" in short


# ------------------------------------------------------------------ RQ1-X 並列表

def test_the_compare_block(local: dict[str, Any], haiku: dict[str, Any]) -> None:
    block = rep.render_compare_block(local, haiku)
    lines = block.splitlines()
    assert lines[0] == rep.COMPARE["start"] and lines[-1] == rep.COMPARE["end"]
    assert rep.COMPARE_NOTE == "兩個 agent 的失敗來自不同的題（預期本機主要在 H1 / H2，Haiku 主要在 H3），所以 cliff 比例的差異不能歸因於 agent。"
    assert rep.COMPARE_NOTE in block and "decision 009 §4" in block and "Advanced" not in block
    x = s3.rq1x_inputs(local["analysis"])
    for group, key in [(lv, "by_level") for lv in LEVELS] + [(f, "by_family") for f in FAMILIES]:
        c = x[key][group]
        row = f"| {group} | 本機模型 | {c['failed_sources']} | {c['curves']} | {c['decidable']} | {c['undecidable']} | {c['confounded']} | {c['cliffs']} |"
        assert row in lines and row.replace("本機模型", "Claude Haiku 5.5") in lines
    alone = rep.render_compare_block(local, None)
    assert "尚未有正式結果" in alone and "| H1 | Claude Haiku 5.5 | — |" in alone
    with pytest.raises(ValueError, match="dry-run"):
        rep.render_compare_block(local, {**haiku, "analysis": {**haiku["analysis"], "provenance": {**haiku["analysis"]["provenance"],
                                                                                                    "dry_run": True}}})
    with pytest.raises(ValueError):
        rep.render_compare_block(haiku, local)  # 欄位依 (suite, agent)，不能對調
    with pytest.raises(ValueError):
        rep.render_compare_block(None, None)
    other = copy.deepcopy(haiku)
    other["analysis"]["provenance"]["freeze_record_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="freeze"):
        rep.render_compare_block(local, other)


# ------------------------------------------------------------------ README（tmp 副本）


def test_readme_blocks_go_in_order_and_leave_everything_else(local: dict[str, Any], haiku: dict[str, Any], tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dirs(monkeypatch, tmp_path)
    readme = _readme(tmp_path)
    original = readme.read_text(encoding="utf-8")
    anchor = "<!-- study2-haiku:end -->"
    rep.write(local, rep.RESULT_DIRS["study3"], readme=readme)
    once = readme.read_text(encoding="utf-8")
    head, tail = original.split(anchor)
    assert once.startswith(head + anchor + "\n\n" + rep.render_readme_block(local)) and once.endswith(tail)
    assert once.index("<!-- study3:end -->") < once.index(rep.COMPARE["start"]) and "尚未有正式結果" in once
    study2 = json.loads((ROOT / "docs" / "results" / "study2" / "summary.json").read_text(encoding="utf-8"))
    assert rep2.embed(once, rep2.render_readme_block(study2)) == once  # Study 2 的段落不變
    rep.write(haiku, rep.RESULT_DIRS["study3-haiku"], readme=readme)
    twice = readme.read_text(encoding="utf-8")
    order = [twice.index(m) for m in ("<!-- study3:start -->", "<!-- study3-haiku:start -->", rep.COMPARE["start"])]
    assert order == sorted(order) and twice.count(rep.COMPARE["start"]) == 1 and "尚未有正式結果" not in twice
    assert twice.endswith(tail) and twice.startswith(head + anchor)
    rep.write(local, rep.RESULT_DIRS["study3"], readme=readme)
    assert readme.read_text(encoding="utf-8") == twice  # 重寫同一份：不變
    assert sorted(p.name for p in rep.RESULT_DIRS["study3-haiku"].iterdir()) == ["README.md", "recoverability.svg", "summary.json"]


def test_readme_writes_need_the_own_result_directory_and_a_clean_other_side(local: dict[str, Any], dry: dict[str, Any], tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dirs(monkeypatch, tmp_path)
    readme = _readme(tmp_path)
    before = readme.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="study3"):  # README 連到 docs/results/study3：正式報告寫進自己的結果目錄
        rep.write(local, tmp_path / "elsewhere", readme=readme)
    other = rep.RESULT_DIRS["study3-haiku"]
    other.mkdir(parents=True)
    stale = _formal(dry, "haiku")
    stale["analysis"]["provenance"]["dry_run"] = True
    (other / "summary.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="dry-run"):
        rep.write(local, rep.RESULT_DIRS["study3"], readme=readme)
    with pytest.raises(ValueError, match="does not exist"):
        rep.write(local, rep.RESULT_DIRS["study3"], readme=tmp_path / "missing.md")
    assert readme.read_text(encoding="utf-8") == before and not rep.RESULT_DIRS["study3"].exists() and not (tmp_path / "elsewhere").exists()


def test_formal_reports_use_the_registered_replicates_and_the_freeze_record(local: dict[str, Any], tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_dirs(monkeypatch, tmp_path)
    fewer = copy.deepcopy(local)
    fewer["analysis"]["provenance"]["replicates"] = 200
    with pytest.raises(ValueError, match="B = 10000"):
        rep.write(fewer, tmp_path / "out")
    unfrozen = copy.deepcopy(local)
    unfrozen["analysis"]["provenance"]["freeze_record_sha256"] = None
    with pytest.raises(ValueError, match="freeze record"):
        rep.write(unfrozen, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------ 目錄的守衛（雙向）


def test_study3_reports_never_go_into_study2_or_round1_directories(local: dict[str, Any], haiku: dict[str, Any], tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    results = _patch_dirs(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="round-1"):
        rep.write(local, results)
    for out in (results / "study2", results / "study2-haiku", results / "study2" / "sub"):
        with pytest.raises(ValueError, match="Study 2"):
            rep.write(local, out)
    with pytest.raises(ValueError, match="study3-haiku"):
        rep.write(local, results / "study3-haiku")
    with pytest.raises(ValueError, match="study3"):
        rep.write(haiku, results / "study3" / "x")
    with pytest.raises(ValueError, match="study3-haiku"):  # 分析之前就拒絕
        rep.write_report("local", tmp_path / "no-runs", [], results / "study3-haiku")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "summary.json").write_text(json.dumps({"format": "agent-gym/study2-report/v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="summary.json"):
        rep.write(local, foreign)
    with pytest.raises(ValueError, match="never modified"):
        rep.write(local, results / "study3", readme=results / "README.md")
    assert not results.exists() and sorted(p.name for p in foreign.iterdir()) == ["summary.json"]


def test_study2_reports_never_go_into_study3_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _patch_dirs(monkeypatch, tmp_path)
    for out in (results / "study3", results / "study3-haiku" / "sub"):
        with pytest.raises(ValueError, match="study3"):
            rep2.write_report([tmp_path / "no-baseline"], [], out)  # 在分析之前拒絕
    holder = tmp_path / "holder"
    holder.mkdir()
    (holder / "summary.json").write_text(json.dumps({"format": rep.REPORT_FORMAT}), encoding="utf-8")
    with pytest.raises(ValueError, match="Study 3"):
        rep2.write_report([tmp_path / "no-baseline"], [], holder)
    assert not results.exists()


# ------------------------------------------------------------------ 曲線圖


def test_the_svg_has_one_panel_per_swept_source(local: dict[str, Any]) -> None:
    out = copy.deepcopy(local)
    a = out["analysis"]
    pending = next(t for t in a["sources"]["failed"] if t not in a["sweep"]["completed"])
    a["sweep"]["partial"] = {"task_id": pending, "rollouts": 3, "points": [{"k": 0, "n": 2, "successes": 2, "agent_errors": 0},
                                                                            {"k": 6, "n": 1, "successes": 0, "agent_errors": 1}]}
    svg = rep.recoverability_svg(out)
    assert svg.count('class="panel"') == len(a["curves"]) + 1 and "未完成" in svg and "[dry-run]" not in svg
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>") and "prefers-color-scheme: dark" in svg
    ceiling = _state(local, [], state="ceiling", target=0)
    ceiling["analysis"]["curves"] = []
    empty = rep.recoverability_svg(ceiling)
    assert 'class="panel"' not in empty and empty.count("沒有完成的曲線") == len(LEVELS) and "天花板" in empty
    assert re.search(r'<svg[^>]*viewBox="0 0 \d+ \d+"', empty)


# ------------------------------------------------------------------ 已 commit 的正式結果（之後才會有）


@pytest.mark.parametrize("name", ["study3", "study3-haiku"])
def test_the_committed_study3_results_match_their_summary(name: str) -> None:
    directory = ROOT / "docs" / "results" / name
    if not (directory / "summary.json").is_file():
        pytest.skip("Study 3 的正式結果尚未產生")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert rep.variant(summary) == name and summary["analysis"]["provenance"]["dry_run"] is False
    assert (directory / "README.md").read_text(encoding="utf-8") == rep.render_page(summary)
    assert (directory / "recoverability.svg").read_text(encoding="utf-8") == rep.recoverability_svg(summary)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert rep.embed_block(readme, rep.render_readme_block(summary)) == readme
