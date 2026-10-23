"""experiments/study3_freeze.py（decision 036 §9 的凍結紀錄）的契約測試。只用 dev instance 與合成資料；eval 只經過 hash。"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agents.baseline import haiku_agent_config
from agents.ollama import local_agent_config
from env.hashing import canonical_json
from experiments import billing, study3_plan, study3_stats
from experiments import study3_freeze as fz
from experiments.run_baseline import OllamaOracleFakeClient, OracleFakeClient, run_split
from replay.aborts import read_manifest
from tasks import generator_v3, suites

ROOT = Path(__file__).resolve().parent.parent
DEV = suites.suite_pool("study3", "dev")


def ZERO() -> float:  # noqa: N802
    return 0.0


@pytest.fixture(scope="module")
def dev_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("freeze3")
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=root / "local", clock=ZERO, suite="study3",
              agent_config=local_agent_config(), dry_run=True, rollouts=3)
    run_split("dev", OracleFakeClient(DEV), run_dir=root / "haiku", clock=ZERO, suite="study3",
              agent_config=haiku_agent_config(), dry_run=True, rollouts=3)
    return root / "local", root / "haiku"


@pytest.fixture(scope="module")
def built(dev_runs: tuple[Path, Path]) -> dict:
    """真正由 build_record 建出的紀錄（dry-run 的 dev；次數縮短），經過 JSON 來回（同 CLI 寫出的檔案）。"""
    record = fz.build_record([dev_runs[0]], [dev_runs[1]], allow_dry_run=True, reps=200, power_reps=2, rq1_power_reps=1,
                             yield_reps=50, commit="0" * 40)
    return json.loads(json.dumps(record, ensure_ascii=False, sort_keys=True))


def _copy(record: dict) -> dict:
    return json.loads(json.dumps(record))


# ---------------------------------------------------------------- 協定


def test_protocol_values(built: dict) -> None:
    agents = built["agents"]
    assert agents["local"]["config_id"] == local_agent_config().config_id
    assert agents["haiku"]["config_id"] == haiku_agent_config().config_id != agents["local"]["config_id"]
    assert agents["haiku"]["model"] == "claude-haiku-5-5" and agents["local"]["max_turns"] == agents["haiku"]["max_turns"] == 20
    tasks = built["tasks"]
    assert tasks["generator_version"] == "issue-tracker-gen/v3"
    assert tasks["golden_pool_sha256"] == generator_v3.GOLDEN_POOL_SHA256 and tasks["golden_rng_sha256"] == generator_v3.GOLDEN_RNG_SHA256
    assert len(tasks["eval"]) == 96 and len({t["sha256"] for t in tasks["eval"]}) == 96
    assert [t["task_id"] for t in tasks["eval"]] == suites.eval_task_ids("study3")
    assert all(tasks[k].startswith("sha256:") and len(tasks[k]) == 71 for k in ("appendix_a_sha256", "appendix_b_sha256"))
    p = built["protocol"]
    assert p["proposal_sha256"] == fz.doc_sha256("proposal")
    assert p["eval_rollouts"] == 3 and p["primary_rule"] == "excluding" and p["criterion"] == "bonferroni95"
    assert p["plan"] == {"coarse_stride": 2, "n_coarse": 10, "n_fine": 48, "drop_threshold": 0.3}
    assert p["source_rule"] == "first_failure" and p["decidable"] == "2/5" and p["conclusion_line"] == 0.5
    caps = p["caps"]
    assert caps["baseline_calls"] == {"dev": 2880, "eval": 2880} and caps["eval_segments"] == [[2, 3], [4, 5], [6, 7], [8, 9]]
    assert caps["sweep_segment_calls"] == 3000 and caps["sweep_calls"] == 80_000 and caps["max_sources"] == 25
    assert caps["not_raisable"] is True
    assert p["sweep"] == {"order": "frozen_permutation", "permutation_key": "study3/sweep-order", "reserve_length": 20,
                          "reserve_quantile": 0.99, "reserve_reps": 4000, "reserve_seed": 0}
    billing_ = built["billing"]
    assert billing_["journals"] == {"study2": "~/.agent-gym/study2-haiku-billing.jsonl", "study3": "~/.agent-gym/study3-haiku-billing.jsonl"}
    assert billing_["total_usd"] == 40.0 and billing_["safety_margin_usd"] == 1.0 and billing_["phase_caps"] == {"dev": 3.0, "baseline": 8.0}
    # §7 的估算用 study3_stats 的常數：必須等於登記的上限（提高上限時兩邊一起改，否則這裡失敗）
    est = p["estimate"]
    assert est["sweep_cap_calls"] == suites.sweep_cap("study3") == caps["sweep_calls"]
    assert est["paid_total_usd"] == suites.STUDY3_PAID.total_usd and est["safety_margin_usd"] == billing.SAFETY_MARGIN_USD
    assert est["ask_below"] == 15 and est["max_turns"] == 20
    stats = built["statistics"]
    assert stats["bootstrap_replicates"] == 10_000 and stats["bootstrap_seed"] == 0 and stats["tasks_per_level"] == 32
    assert stats["alpha"] == 0.025
    assert stats["power"]["rq2"] == {"p3": [0.3, 0.6, 0.8], "delta": [0.1, 0.25, 0.4], "kappa": [2, 10], "reps": 1000, "seed": 0}
    assert stats["power"]["rq1"]["lengths"] == [16, 18, 20] and stats["power"]["rq1"]["reps"] == 200
    assert stats["expected_yield"] == {"reps": 4000, "seed": 0, "decidable_share": "4/5", "true_cliff": [0.8, 0.65]}
    assert set(stats["sensitivity"]) == {"max_turns", "agent_error", "rq2_without_agent_error", "cap_mid_source"}
    assert "不做檢定" in stats["sensitivity"]["rq2_without_agent_error"]
    assert built["sweep_permutation"] == study3_plan.sweep_permutation(suites.eval_task_ids("study3"))
    assert str(Path.home()) not in canonical_json(built)  # 不把使用者的路徑寫進 repo


def test_document_hashes_cover_only_the_frozen_parts() -> None:
    text = (ROOT / "docs/decisions/036-study3-proposal.md").read_text(encoding="utf-8")
    parts = ("proposal", "appendix_a", "appendix_b")
    base = {part: fz.doc_sha256(part, text) for part in parts}
    assert base["appendix_a"] == fz.appendix_a_sha256() == fz.appendix_a_sha256(text)

    def changed(new_text: str) -> set[str]:
        return {part for part in parts if fz.doc_sha256(part, new_text) != base[part]}

    assert changed(text.replace("## 實作紀錄\n", "## 實作紀錄\n\n- 新的一筆\n")) == set()  # 實作紀錄可以繼續寫
    assert changed(text.replace("### 完整性與天花板", "### 完整性與天花板 ")) == {"proposal"}  # §6
    assert changed(text.replace("### A.2 instruction 逐字稿", "### A.2 instruction 逐字稿 ")) == {"appendix_a"}
    assert changed(text.replace("### epic-closeout（13 項）", "### epic-closeout（13 項） ")) == {"appendix_b"}
    with pytest.raises(ValueError):
        fz.appendix_a_sha256(text.replace("## 附錄 A", "## 附錄 X"))
    with pytest.raises(ValueError):
        fz.doc_sha256("proposal", text + "\n## 背景\n")


def test_check_record_detects_drift(built: dict) -> None:
    assert fz.check_record(built) == []
    tampered = _copy(built)
    tampered["tasks"]["eval"][0]["sha256"] = "sha256:" + "0" * 64
    tampered["protocol"]["caps"]["sweep_calls"] = 90_000
    perm = tampered["sweep_permutation"]
    perm[0], perm[1] = perm[1], perm[0]
    assert fz.check_record(tampered) == ["tasks", "protocol", "sweep_permutation"]
    for section in ("agents", "billing", "statistics"):
        other = _copy(built)
        other[section]["extra"] = 1
        assert fz.check_record(other) == [section]
    nan = _copy(built)
    nan["statistics"]["alpha"] = math.nan  # 不 raise：回報不一致
    assert fz.check_record(nan) == ["statistics"]
    p90 = built["dev"]["haiku"]["call_usd_p90"]
    for change in ({"reserve": {"local_calls": built["reserve"]["local_calls"] + 1}},
                   {"reserve": {"haiku_usd": built["reserve"]["haiku_usd"] * 2}},
                   # 預留額與 dev 的 P90 一起改：仍要等於估算中使用的 P90
                   {"reserve": {"haiku_usd": built["reserve"]["local_calls"] * p90 * 2}, "dev": {"call_usd_p90": p90 * 2}},
                   {"reserve": {"haiku_usd": math.inf}, "dev": {"call_usd_p90": math.inf}}):
        other = _copy(built)
        other["reserve"].update(change["reserve"])
        other["dev"]["haiku"].update(change.get("dev", {}))
        assert fz.check_record(other) == ["reserve"], change
    assert fz.check_record({**built, "format": "x"}) == ["format"]


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("code_commit"),
    lambda r: r.update(code_commit="HEAD"),
    lambda r: r.update(decision="034"),
    lambda r: r["expected"].pop("haiku"),
    lambda r: r.pop("power"),
    lambda r: r["power"].pop("rq1"),
    lambda r: r["estimate"]["haiku"].update(ask_user=True),  # 估算要求問使用者，但紀錄中沒有使用者的決定
    lambda r: r.update(user_decision={"response": "accept_fewer_curves", "agents": ["haiku"], "note": "x"}),  # 沒有問卻有決定
])
def test_check_record_requires_the_other_section_9_items(built: dict, mutate: Any) -> None:
    other = _copy(built)
    mutate(other)
    assert fz.check_record(other) == ["contents"]


def test_a_recorded_user_decision_covers_the_agents_that_asked(built: dict) -> None:
    other = _copy(built)
    other["estimate"]["haiku"]["ask_user"] = True
    other["user_decision"] = {"response": "accept_fewer_curves", "agents": ["haiku"], "note": "2026-10-30 使用者：接受較少的曲線"}
    assert fz.check_record(other) == []
    for wrong in ({"agents": ["local"]}, {"response": "raise_cap"}, {"note": "  "}):
        bad = _copy(other)
        bad["user_decision"].update(wrong)
        assert fz.check_record(bad) == ["contents"]


def test_the_written_record_feeds_the_sweep_plan(built: dict, tmp_path: Path) -> None:
    """build_record 的輸出（照 CLI 的寫法）就是 study3_plan 讀取的格式：排列與兩種預留額。"""
    path = tmp_path / "record.json"
    path.write_text(json.dumps(built, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    record = study3_plan.read_freeze(path)
    assert study3_plan.frozen_permutation(suites.eval_task_ids("study3"), record) == built["sweep_permutation"]
    assert study3_plan.reserve(record, "local_calls") == study3_stats.reserve_calls() == 6510
    assert study3_plan.reserve(record, "haiku_usd") == 6510 * built["dev"]["haiku"]["call_usd_p90"]


# ---------------------------------------------------------------- 預期的 n 與偵測力


def test_mostly_cliff_probability_matches_the_registered_table() -> None:
    """decision 036 §2「偵測力與預期產量」：判為以驟降為主所需的 k 與機率（二項分布精確計算）。"""
    assert [fz.cliff_threshold(n) for n in (10, 15, 20, 25)] == [9, 12, 15, 18]
    assert fz.cliff_threshold(0) is None
    assert fz.mostly_cliff_probability(10, 0.8) == pytest.approx(0.8 ** 10 + 10 * 0.8 ** 9 * 0.2)
    table = {(0.8, 10): 0.38, (0.8, 15): 0.65, (0.8, 20): 0.80, (0.8, 25): 0.89,
             (0.65, 10): 0.09, (0.65, 15): 0.17, (0.65, 20): 0.25, (0.65, 25): 0.31}
    for (p, n), value in table.items():
        assert abs(fz.mostly_cliff_probability(n, p) - value) < 0.01
    assert fz.mostly_cliff_probability(0, 0.8) == 0.0


def _review(outcomes: dict[str, str]) -> dict:
    """合成的 dev 審查：每級 8 題，每題 3 個 rollout；outcomes[level] = 三個 rollout 的結果（S 成功、F 失敗、A agent_error：沒有
    trajectory）。"""
    episodes, unfinished = [], []
    for level, pattern in outcomes.items():
        for t in range(8):
            for i, outcome in enumerate(pattern):
                row = {"task_id": f"{level}-{t}", "level": level, "rollout_index": i}
                if outcome == "A":
                    unfinished.append({**row, "outcome": "agent_error"})
                else:
                    episodes.append({**row, "success": outcome == "S"})
    return {"episodes": episodes, "unfinished": unfinished}


def test_failed_source_probability_does_not_count_agent_error_as_a_source() -> None:
    """agent_error 沒有 trajectory，不是 source（規則 B）：只計入分母。SFA → f = 1/3（不是 2/3，也不是略過時的 1/2）。"""
    p = fz.failed_source_probability(_review({"H1": "SSS", "H2": "SFA", "H3": "FFF"}))
    assert p == pytest.approx({"H1": 0.0, "H2": 1 - (2 / 3) ** 3, "H3": 1.0})
    assert fz.failed_source_probability(_review({"H1": "SAA"})) == {"H1": 0.0}


def test_expected_yield_simulates_the_section_6_stopping_rule() -> None:
    review = _review({"H1": "SSS", "H2": "SFA", "H3": "FFF"})
    free = fz.expected_yield(review, length=20, budget=10 ** 9, reserve=0, reps=100)
    assert free["expected_failed_sources"] == pytest.approx(32 * (1 - (2 / 3) ** 3) + 32)
    assert free["expected_swept"] == 25 and free["p_full"] == 1 and free["expected_n"] == pytest.approx(20) and free["n_used"] == 20
    exact = math.fsum(math.comb(25, n) * 0.8 ** n * 0.2 ** (25 - n) * fz.mostly_cliff_probability(n, 0.8) for n in range(26))
    assert free["p_mostly_cliff"]["0.8"] == pytest.approx(exact) and free["decidable_share"] == "4/5"
    # 本機：80,000 次、預留額 6,510、L = 20 → 約一半做完 25 條，其餘多半停在 21–24 條（decision 036 §7「估計依據」）
    local = fz.expected_yield(review, length=20, budget=80_000, reserve=6510, reps=1000)
    assert 23.5 < local["expected_swept"] < 25 and 0.3 < local["p_full"] < 0.8
    assert fz.expected_yield(review, length=20, budget=80_000, reserve=80_001, reps=20)["expected_swept"] == 0  # 預留額不足：不開始
    # Haiku：以美元計（每次呼叫成本 × 呼叫數）
    haiku = fz.expected_yield(review, length=20, budget=30.0, reserve=6510 * 0.0006, unit_cost=0.0006, reps=300)
    assert 10 < haiku["expected_swept"] < 20 and haiku["p_full"] == 0
    none = fz.expected_yield(_review({"H1": "SSS", "H2": "SSS", "H3": "SSS"}), length=20, budget=80_000, reserve=6510, reps=50)
    assert none["expected_failed_sources"] == 0 and none["expected_swept"] == 0 and none["n_used"] == 0
    assert none["p_mostly_cliff"] == {"0.8": 0.0, "0.65": 0.0}
    assert fz.expected_yield(review, length=20, budget=80_000, reserve=6510, reps=50) == \
        fz.expected_yield(review, length=20, budget=80_000, reserve=6510, reps=50)  # 決定性


# ---------------------------------------------------------------- build_record


def test_build_record(built: dict, dev_runs: tuple[Path, Path]) -> None:
    assert built["format"] == fz.FREEZE_FORMAT and built["decision"] == "036" and built["code_commit"] == "0" * 40
    for agent in ("local", "haiku"):
        dev = built["dev"][agent]
        assert dev["runs"] == [agent] and dev["config_id"] == built["agents"][agent]["config_id"]
        assert all(level["episodes"] == 24 for level in dev["levels"].values())
        assert dev["suspected_grader_issues"] == 0 and dev["unfinished"] == 0 and dev["stopped_by_budget"] == []
        assert built["estimate"][agent]["agent"] == agent and built["estimate"][agent]["reps"] == 200
        expected = built["expected"][agent]
        assert expected["length"] == built["estimate"][agent]["length"] and expected["reps"] == 50
        assert expected["expected_failed_sources"] == 0  # oracle 的 dev：沒有失敗
    assert built["expected"]["local"]["budget"] == 80_000 and built["expected"]["local"]["reserve"] == 6510
    haiku = built["dev"]["haiku"]
    manifest = math.fsum(row.cost_usd for row in read_manifest(dev_runs[1] / "manifest.jsonl"))
    assert haiku["spent"]["spent_usd"] == manifest > 0 and haiku["spent"]["source"].startswith("dry-run")
    assert built["estimate"]["haiku"]["sweep_usd_left"] == pytest.approx(
        40 - 1 - manifest - built["estimate"]["haiku"]["haiku_baseline_p90_usd"])
    assert haiku["call_usd_p90"] > 0 and 0 < haiku["call_usd_mean"] <= max(haiku["call_usd_p90"] * 10, 1)
    assert built["expected"]["haiku"]["budget"] == built["estimate"]["haiku"]["sweep_usd_left"]
    assert built["expected"]["haiku"]["reserve"] == built["reserve"]["haiku_usd"]
    assert built["expected"]["haiku"]["unit_cost"] == haiku["call_usd_mean"]
    assert built["power"]["rq2"] and all(row["reps"] == 2 for row in built["power"]["rq2"])
    assert built["power"]["rq1"]["reps"] == 1 and {row["length"] for row in built["power"]["rq1"]["rows"]} == {16, 18, 20}
    assert str(dev_runs[0].parent) not in canonical_json(built)


def _journal(path: Path, entries: list[tuple[str, float | None]]) -> None:
    lines = []
    for i, (phase, cost) in enumerate(entries):
        lines += [{"format": billing.JOURNAL_FORMAT, "kind": "reserve", "id": f"r{i}", "utc_time": "t", "phase": phase, "run": "x",
                   "bound_usd": 1.0},
                  {"format": billing.JOURNAL_FORMAT, "kind": "settle", "id": f"r{i}", "utc_time": "t", "cost_usd": cost}]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="ascii")


def test_haiku_dev_spent_reads_the_registered_journal(dev_runs: tuple[Path, Path], tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """§7 的「dev 的實際花費」：正式的 dev 一律讀登記的 Study 3 總帳；沒有總帳、無法計價、凍結前已有 eval 的花費、總帳被重設 → 拒絕。"""
    runs = tmp_path / "runs"
    formal = Path(shutil.copytree(dev_runs[1], runs / "haiku-dev"))
    run = json.loads((formal / "run.json").read_text(encoding="utf-8"))
    (formal / "run.json").write_text(json.dumps({**run, "dry_run": False}), encoding="utf-8")
    journal = tmp_path / "study3.jsonl"
    monkeypatch.setattr(billing, "RUNS_ROOT", runs)
    monkeypatch.setattr(billing, "STUDY3_JOURNAL", journal)
    with pytest.raises(ValueError, match="does not exist"):
        fz.haiku_dev_spent([formal])
    manifest = billing.manifest_floor(haiku_agent_config().config_id, generator=generator_v3.GENERATOR_VERSION)
    assert manifest > 0
    _journal(journal, [("dev", manifest + 0.1)])  # 總帳把中斷的呼叫以上限計：可以大於 manifest 合計
    spent = fz.haiku_dev_spent([formal])
    assert spent["spent_usd"] == pytest.approx(manifest + 0.1) and spent["manifest_usd"] == manifest
    assert str(tmp_path) not in json.dumps(spent)
    for entries, message in (([("dev", manifest / 2)], "reset"), ([("dev", manifest + 0.1), ("dev", None)], "unknown cost"),
                             ([("dev", manifest + 0.1), ("baseline", 0.01)], "eval spending")):
        _journal(journal, entries)
        with pytest.raises(ValueError, match=message):
            fz.haiku_dev_spent([formal])
    with pytest.raises(ValueError, match="mix"):
        fz.haiku_dev_spent([dev_runs[1], formal])


def test_build_record_refuses_an_incomplete_or_swapped_dev(dev_runs: tuple[Path, Path], tmp_path: Path) -> None:
    local, haiku = dev_runs
    with pytest.raises(ValueError):  # 兩個 agent 對調
        fz.build_record([haiku], [local], allow_dry_run=True, reps=200, power_reps=1, rq1_power_reps=1, yield_reps=1, commit="0")
    with pytest.raises(ValueError, match="dry-run"):
        fz.build_record([local], [haiku], reps=200, power_reps=1, rq1_power_reps=1, yield_reps=1, commit="0")
    one = tmp_path / "one"
    run_split("dev", OllamaOracleFakeClient(DEV), run_dir=one, clock=ZERO, suite="study3", agent_config=local_agent_config(),
              dry_run=True, rollouts=1)
    with pytest.raises(ValueError, match="24"):  # dev 必須是 24 題 × R = 3（decision 036 §5 C）
        fz.build_record([one], [haiku], allow_dry_run=True, reps=200, power_reps=1, rq1_power_reps=1, yield_reps=1, commit="0")


# ---------------------------------------------------------------- code_commit 與 CLI


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_code_commit_requires_a_clean_tree(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("runs/\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py", ".gitignore")
    _git(tmp_path, "commit", "-q", "-m", "a")
    head = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "r.json").write_text("{}\n", encoding="utf-8")  # 忽略的目錄（runs/）不影響
    assert fz.code_commit(tmp_path) == head
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")  # 沒有 commit 的新程式：HEAD 不是寫紀錄的程式
    with pytest.raises(ValueError, match="commit"):
        fz.code_commit(tmp_path)
    (tmp_path / "b.py").unlink()
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="commit"):
        fz.code_commit(tmp_path)


def _fake_record(ask: bool) -> dict:
    return {"format": fz.FREEZE_FORMAT, "estimate": {"local": {"n": 25, "ask_user": False}, "haiku": {"n": 13 if ask else 20, "ask_user": ask}},
            "expected": {"local": {"expected_swept": 24.2, "n_used": 19}, "haiku": {"expected_swept": 5.0, "n_used": 4}}}


def test_cli_writes_once_and_stops_when_the_estimate_asks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(fz, "build_record", lambda *a, **k: _fake_record(ask=True))
    out = tmp_path / "record.json"
    args = ["--local-dev-run", "runs/a", "--haiku-dev-run", "runs/b", "--out", str(out)]
    with pytest.raises(SystemExit):  # 能 sweep 的失敗 source < 15：回頭問使用者（decision 036 §7），不寫紀錄
        fz.main(args)
    assert not out.exists() and "ask the user" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # 空白的決定不算
        fz.main([*args, "--user-accepts-fewer-curves", "  "])
    assert fz.main([*args, "--user-accepts-fewer-curves", "2026-10-30 使用者：接受較少的曲線"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["user_decision"] == {"response": "accept_fewer_curves", "agents": ["haiku"],
                                        "note": "2026-10-30 使用者：接受較少的曲線"}
    with pytest.raises(SystemExit):  # 只寫一次
        fz.main([*args, "--user-accepts-fewer-curves", "x"])
    monkeypatch.setattr(fz, "build_record", lambda *a, **k: _fake_record(ask=False))
    other = tmp_path / "other.json"
    assert fz.main(["--local-dev-run", "a", "--haiku-dev-run", "b", "--out", str(other)]) == 0
    assert "user_decision" not in json.loads(other.read_text(encoding="utf-8"))
    with pytest.raises(SystemExit):  # 不需要問時不接受決定（避免紀錄中出現沒有發生的決定）
        fz.main(["--local-dev-run", "a", "--haiku-dev-run", "b", "--out", str(tmp_path / "third.json"),
                 "--user-accepts-fewer-curves", "x"])
    with pytest.raises(SystemExit):  # 總帳的路徑不能另給（decision 036 §7：一律讀登記的 Study 3 總帳）
        fz.main(["--local-dev-run", "a", "--haiku-dev-run", "b", "--out", str(tmp_path / "x.json"), "--haiku-journal", "j"])


def test_cli_reports_a_missing_dev_directory_as_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        fz.main(["--local-dev-run", str(tmp_path / "nope"), "--haiku-dev-run", str(tmp_path / "nope2"), "--out", str(tmp_path / "r.json")])
    assert exc.value.code == 2 and "nope" in capsys.readouterr().err and not (tmp_path / "r.json").exists()


# ---------------------------------------------------------------- 凍結之後


def test_the_freeze_switch_follows_the_committed_record() -> None:
    """凍結紀錄 commit 之後，題目、agent、協定、統計與預留額都不得再改動，§9 的其他項目也必須存在（decision 036 §9：測試每次都
    比對紀錄與程式）；eval 的開關只在紀錄存在時打開。正式紀錄以預設的次數寫。"""
    path = ROOT / fz.RECORD_PATH
    assert suites.STUDY3_FROZEN is path.is_file()
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    assert fz.check_record(record) == []
    assert study3_plan.read_freeze() == record
    assert all(row["reps"] == fz.POWER_REPS for row in record["power"]["rq2"])
    assert record["power"]["rq1"]["reps"] == fz.RQ1_POWER_REPS
    for agent in ("local", "haiku"):
        assert record["estimate"][agent]["reps"] == study3_stats.COST_REPS
        assert record["expected"][agent]["reps"] == fz.YIELD_REPS and record["expected"][agent]["seed"] == fz.YIELD_SEED
        assert all(level["episodes"] == 24 for level in record["dev"][agent]["levels"].values())
    assert not record["dev"]["haiku"]["spent"]["source"].startswith("dry-run")
