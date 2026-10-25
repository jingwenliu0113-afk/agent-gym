"""experiments/study3_analysis.py（decision 036 §2、§6 的事先登記分析，一次一個 agent）的契約測試。

測試資料：oracle 假 client 在 study3 eval pool 上的輸出（部分題目改用失敗的 negative 當 oracle，或可分岔的假 client），以及合成的
曲線。runs 的根目錄一律是 tmp 目錄（不讀 repo 的 runs/），凍結紀錄以 monkeypatch 換成 tmp 檔案；eval instance 只經過程式，
測試不輸出任何 eval 的內容（decision 036 §4）。
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import random
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.baseline import haiku_agent_config
from agents.ollama import OllamaPolicy, local_agent_config
from env.hashing import canonical_json
from experiments import billing, run_sweep, study3_context
from experiments import study2_analysis as s2
from experiments import study3_analysis as s3
from experiments import study3_plan as plan
from experiments.call_log import CALL_LOG
from experiments.run_baseline import OllamaOracleFakeClient, OracleFakeClient, run_split
from experiments.run_sweep import REGISTERED_PLAN, run_sweeps, select_sources
from replay.localization import RolloutOutcome, SweepPlan, SweepResult, assess_drop, estimate_checkpoint
from replay.runner import run_episode
from replay.trajectory import Trajectory
from tasks import suites
from tasks.gen3 import verified_close as vc
from tasks.generator_v2 import THIRD_PARTY_TEXT
from tasks.generator_v3 import GENERATOR_VERSION, MAX_TURNS
from tasks.suites import STUDY3_PAID, eval_task_ids, suite_pool

SMALL = SweepPlan(coarse_stride=6, n_coarse=1, n_fine=1, drop_threshold=0.3)
FAILING = {  # task → 改用的失敗 negative（只以程式判定，不輸出內容）
    "gen3-epic-closeout-H2-s2": "E5:wrong-tracker-text",
    "gen3-epic-closeout-H1-s2": "E5:blocks-instead-of-dropped",
    "gen3-scoped-triage-H2-s2": "S7:secdocs-treated-as-denied",
    "gen3-verified-close-H1-s2": "V3:close-qa-broken",
}
AGENT_ERROR = "gen3-sync-repair-H1-s2"  # 每個 rollout 都是 agent_error（輸出截斷）
FORKS = {"gen3-epic-closeout-H1-s3": "E5:blocks-instead-of-dropped",  # 寫錯範本：max_count 飽和 → 機械性（Study 3 補上的情況）
         "gen3-epic-closeout-H2-s3": "E1:leave-resolved-open"}  # 漏關一個 resolved：仍可恢復 → 行為性


def ZERO() -> float:  # noqa: N802
    return 0.0


def _eval() -> tuple:
    return suite_pool("study3", "eval")


def _by_id() -> dict[str, Any]:
    return {i.task.task_id: i for i in _eval()}


def _owner_text_target(instance: Any) -> int | None:
    """scoped-triage：oracle 留 comment 的、有權限而且帶 needs-info 的候選（只以程式選取）。"""
    p = instance.task.params
    commented = {args["issue_id"] for name, args in instance.oracle_calls if name == "add_comment"}
    return next((i for i in p["permitted"] if i in p["needs_info"] and i in commented), None)


@functools.cache
def _audited_id() -> str:
    """comment 稽核的成功 episode（§5 C）：第一個有上述候選的 scoped-triage 題；該 comment 另含 owner 文字（仍成功：只檢查「包含」）。"""
    return next(i.task.task_id for i in _eval() if i.task.family == "gen3/scoped-triage" and i.task.task_id not in FAILING
                and _owner_text_target(i) is not None)


def _with_owner_text(pool: tuple) -> tuple:
    out = []
    for instance in pool:
        if instance.task.task_id == _audited_id():
            target = _owner_text_target(instance)
            calls = tuple((name, {**args, "body": args["body"] + " Needs owner triage."}
                           if name == "add_comment" and args["issue_id"] == target else args) for name, args in instance.oracle_calls)
            instance = dataclasses.replace(instance, oracle_calls=calls)
        out.append(instance)
    return tuple(out)


def _with(pool: tuple, replaced: dict[str, str]) -> tuple:
    out = []
    for instance in pool:
        name = replaced.get(instance.task.task_id)
        if name is not None:
            instance = dataclasses.replace(instance, oracle_calls=dict(instance.negative_calls)[name])
        out.append(instance)
    return tuple(out)


class AgentErrorClient(OllamaOracleFakeClient):
    """指定的 task：每個回覆都是輸出截斷（done_reason "length"、eval_count = num_predict）→ AdapterError → agent_error。"""

    def __init__(self, instances: tuple, broken: set[str]) -> None:
        super().__init__(instances)
        by_id = {i.task.task_id: i for i in instances}
        self._broken = {by_id[t].task.instruction for t in broken}

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        response = super().chat(request)
        if request["messages"][1]["content"] in self._broken:
            response = {**response, "done_reason": "length", "eval_count": local_agent_config().model_config["options"]["num_predict"]}
        return response


class ForkingClient(OllamaOracleFakeClient):
    """指定的 task 有兩條劇本（oracle 與一個 negative）：在第一次不同的那一步 j，以機率 p 走 oracle、否則走 negative；前綴已經走過
    j 的 rerun 照前綴的劇本繼續。其他 task 照 oracle。由真正的 sweep 產生 P(success | k) 的驟降（同 Study 2 的測試）。"""

    def __init__(self, instances: tuple, forks: dict[str, tuple[str, float]], seed: int = 0) -> None:
        super().__init__(instances)
        by_id = {i.task.task_id: i for i in instances}
        self._forks = {}
        for task_id, (negative, p) in forks.items():
            oracle, wrong = by_id[task_id].oracle_calls, dict(by_id[task_id].negative_calls)[negative]
            j = next((i for i, (a, b) in enumerate(zip(oracle, wrong)) if a != b), min(len(oracle), len(wrong)))
            self._forks[by_id[task_id].task.instruction] = (tuple(oracle), tuple(wrong), j, p)
        self._rng = random.Random(seed)

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request["messages"]
        fork = self._forks.get(messages[1]["content"])
        if fork is None:
            return super().chat(request)
        oracle, wrong, j, p = fork
        assistants = [m for m in messages if m["role"] == "assistant"]
        t = len(assistants)
        if t > j:
            call = assistants[j].get("tool_calls") or []
            taken = [(c["function"]["name"], c["function"]["arguments"]) for c in call]
            script = wrong if j < len(wrong) and taken == [wrong[j]] else oracle
        elif t == j:
            script = oracle if self._rng.random() < p else wrong
        else:
            script = oracle
        response = super().chat({**request, "messages": messages[:2]})  # 借用回覆的形狀
        if t < len(script):
            name, args = script[t]
            response["message"] = {"role": "assistant", "content": "",
                                   "tool_calls": [{"function": {"name": name, "arguments": copy.deepcopy(args)}}]}
        else:
            response["message"] = {"role": "assistant", "content": "Done."}
        return response


def _record_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, local_calls: float = 1, haiku_usd: float = 2.0) -> Path:
    path = tmp_path / "036-freeze-record.json"
    path.write_text(json.dumps({"format": plan.FREEZE_FORMAT, "sweep_permutation": plan.sweep_permutation(eval_task_ids("study3")),
                                "reserve": {"local_calls": local_calls, "haiku_usd": haiku_usd}}), encoding="utf-8")
    monkeypatch.setattr(plan, "FREEZE_RECORD", path)
    return path


def _trajectory(run_dir: Path, task_id: str, r: int = 0) -> Trajectory:
    return Trajectory.load(run_dir / "trajectories" / f"{task_id}-r{r}.json")


def _failed_turn(source: Trajectory) -> int:
    """第一個在「最後沒有通過的 predicate 的 issue」上有變更的 turn。"""
    report = source.grade()
    issues = {c.predicate["issue_id"] for c in (*report.outcome_checks, *report.constraint_checks) if not c.passed}
    return next(t.index for t in source.turns for s in t.tool_steps for c in s.changes if c.issue_id in issues)


DEV = {i.task.task_id: i for i in suite_pool("study3", "dev")}  # 歸屬與查找的合成 source 只用 dev instance


def _replay(task_id: str, calls: Any) -> Trajectory:
    """以凍結的本機 agent（oracle 假 client）重放 calls，得到真正的 trajectory（dev instance）。"""
    instance = dataclasses.replace(DEV[task_id], oracle_calls=tuple(calls))
    policy = OllamaPolicy(OllamaOracleFakeClient([instance]), local_agent_config(), seed_source=lambda: 1)
    return run_episode(instance.task, policy, max_turns=MAX_TURNS, clock=ZERO)


def _negative(task_id: str, name: str) -> Trajectory:
    return _replay(task_id, dict(DEV[task_id].negative_calls)[name])


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def failing_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """dry-run 的 runs 根目錄：一個 eval baseline 目錄，FAILING 的題目失敗、AGENT_ERROR 的題目是 agent_error；_audited_id() 的題目
    成功，但 comment 帶 owner 文字（comment 稽核）。"""
    root = tmp_path_factory.mktemp("s3a") / "runs"
    pool = _with_owner_text(_with(_eval(), FAILING))
    run_split("eval", AgentErrorClient(pool, {AGENT_ERROR}), run_dir=root / "b", clock=ZERO, suite="study3",
              agent_config=local_agent_config(), dry_run=True, rollouts=1)
    results = json.loads((root / "b" / "summary.json").read_text(encoding="utf-8"))["results"]
    assert {r["task_id"] for r in results if r["outcome"] == "agent_error"} == {AGENT_ERROR}
    assert {r["task_id"] for r in results if r["outcome"] == "failure"} == set(FAILING)
    return root


@pytest.fixture(scope="module")
def segment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """正式模式（dry_run False）的一個 seed 區塊目錄（oracle 假 client，R = 1）。"""
    run_dir = tmp_path_factory.mktemp("s3seg") / "seg"
    run_split("eval", OllamaOracleFakeClient(_eval()), run_dir=run_dir, clock=ZERO, suite="study3", agent_config=local_agent_config(),
              rollouts=1, seeds=(2, 3))
    return run_dir


def _copy(src: Path, dst: Path, **run_changes: Any) -> Path:
    shutil.copytree(src, dst)
    if run_changes:
        run = json.loads((dst / "run.json").read_text(encoding="utf-8"))
        (dst / "run.json").write_text(json.dumps({**run, **run_changes}), encoding="utf-8")
    return dst


def _empty_dir(path: Path, **run: Any) -> Path:
    path.mkdir(parents=True)
    base = {"format": "agent-gym/baseline-run/v2", "split": "eval", "rollouts": 1, "config_id": local_agent_config().config_id,
            "dry_run": False}
    (path / "run.json").write_text(json.dumps({**base, **run}), encoding="utf-8")
    (path / "manifest.jsonl").write_bytes(b"")
    return path


def _relabel_generator(run_dir: Path, version: str) -> None:
    lines = (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    (run_dir / "manifest.jsonl").write_text("".join(json.dumps({**r, "generator_version": version}) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------ 列入規則（§6）


def test_the_listing_enumerates_every_formal_baseline_of_the_agent(segment: Path, tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _copy(segment, root / "seg")
    _copy(segment, root / "dry", dry_run=True)  # dry-run 不列入正式分析
    _empty_dir(root / "empty", seeds=[4, 5])  # 沒有任何嘗試的段：照樣列入並列出
    _empty_dir(root / "unknown")  # 沒有列、沒有 seeds、沒有 summary：無法判定屬於哪個研究，只列出
    other = _copy(segment, root / "nested" / "older")
    _relabel_generator(other, "issue-tracker-gen/v2")  # 別的研究的目錄：不列入、不拒絕
    listing = s3.list_baselines("local", root)
    assert [p.name for p in listing.listed] == ["empty", "seg"]
    assert [p.name for p in listing.empty] == ["empty"] and [p.name for p in listing.unattributed] == ["unknown"]
    assert [p.name for p in s3.list_baselines("local", root, dry_run=True).listed] == ["dry"]
    assert s3.list_baselines("haiku", root).listed == ()


def test_the_listing_refuses_unlisted_or_unreadable_records(segment: Path, tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _copy(segment, root / "seg")
    broken = _copy(segment, root / "no-run-file")
    (broken / "run.json").unlink()  # 這個 agent 的 Study 3 eval 列，卻沒有 run.json
    with pytest.raises(ValueError, match="no-run-file"):
        s3.list_baselines("local", root)
    shutil.rmtree(broken)
    calls = root / "calls-only"  # 只有呼叫紀錄的中斷段（本機）
    calls.mkdir()
    task = eval_task_ids("study3")[0]
    (calls / CALL_LOG).write_text(json.dumps({"format": "agent-gym/call/v1", "call": 1, "task_id": task, "rollout_index": 0,
                                               "attempt": 1, "k": None, "ok": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calls-only"):
        s3.list_baselines("local", root)
    assert [p.name for p in s3.list_baselines("haiku", root).listed] == []  # 呼叫紀錄只屬於本機的 run
    shutil.rmtree(calls)
    (root / "seg" / "manifest.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seg"):
        s3.list_baselines("local", root)


def test_formal_sweeps_use_exactly_the_listed_baselines(segment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s3, "RUNS_ROOT", tmp_path / "runs")  # sweep 與分析以同一個 runs/ 為準（repo 的 runs/，測試中換成 tmp）
    _copy(segment, tmp_path / "runs" / "seg")
    _empty_dir(tmp_path / "runs" / "seg-4-5", seeds=[4, 5])
    argv = ["--baseline-run", "runs/seg", "--plan", "2,10,48,0.3", "--criterion", "bonferroni95", "--confirm-local-run",
            "--max-model-calls", "3000", "--total-model-calls", "80000", "--run-name", "s"]
    with pytest.raises(SystemExit):
        run_sweep.main(argv)
    err = capsys.readouterr().err
    assert "seg-4-5" in err and "036" in err and not (tmp_path / "runs" / "s").exists()
    outside = _empty_dir(tmp_path / "elsewhere" / "seg-6", seeds=[6, 7])  # 列舉之外（runs/ 之外）的正式 baseline 目錄
    with pytest.raises(SystemExit):
        run_sweep.main([*argv[:2], "runs/seg-4-5", "--baseline-run", str(outside), *argv[2:]])
    err = capsys.readouterr().err
    assert "not listed" in err and "seg-6" in err and not (tmp_path / "runs" / "s").exists()
    listed = [tmp_path / "runs" / "seg", tmp_path / "runs" / "seg-4-5"]
    with pytest.raises(ValueError, match=r"not listed \['[^']*seg-6'\]"):
        s3.check_baseline_set(local_agent_config().config_id, [*listed, outside], tmp_path / "runs")
    monkeypatch.setattr(suites, "STUDY3_FROZEN", False)
    with pytest.raises(SystemExit):  # 給齊之後，列入規則通過，接著才是研究的開關（eval 等凍結紀錄）
        run_sweep.main([*argv[:2], "runs/seg-4-5", *argv[2:]])
    assert "frozen record of decision 036" in capsys.readouterr().err


def test_a_formal_analysis_needs_three_rollouts(segment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "runs"
    _copy(segment, root / "seg")
    monkeypatch.setattr(s3, "RUNS_ROOT", root)
    with pytest.raises(ValueError, match="R = 3"):
        s3.analyze("local", replicates=50)  # 正式分析一律列舉 RUNS_ROOT


def test_a_formal_analysis_enumerates_the_repository_runs_root(segment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    """§6：正式分析的 baseline 目錄由程式從 runs/ 列舉；只放部分目錄的根目錄（某一段、或 runs/ 的部分副本）一律拒絕。"""
    root = tmp_path / "runs"
    _copy(segment, root / "seg")
    _copy(segment, root / "seg-again")
    partial = tmp_path / "copy"
    _copy(segment, partial / "seg")  # 少了 seg-again 的副本
    monkeypatch.setattr(s3, "RUNS_ROOT", root)
    for other in (root / "seg", partial):
        with pytest.raises(ValueError, match="enumerates every directory"):
            s3.analyze("local", other, replicates=50)
        with pytest.raises(SystemExit):
            s3.main(["--agent", "local", "--runs-root", str(other), "--out", str(tmp_path / "o")])
        assert "enumerates every directory" in capsys.readouterr().err and not (tmp_path / "o").exists()
    with pytest.raises(ValueError, match="R = 3"):  # 同一個目錄的另一種寫法照常通過這一項檢查
        s3.analyze("local", tmp_path / "copy" / ".." / "runs", replicates=50)
    _record_file(tmp_path, monkeypatch)
    dry = tmp_path / "dry"
    _copy(segment, dry / "seg", dry_run=True)
    out = s3.analyze("local", dry, dry_run=True, replicates=50)  # dry-run 只供測試與演練：任何根目錄
    assert out["provenance"]["runs_root"] == str(dry.resolve())


# ------------------------------------------------------------------ 採計與完整性（§6）


def test_adoption_takes_the_earliest_finished_rollout_and_interrupted_directories_do_not_block(tmp_path: Path) -> None:
    from experiments.analyze_baseline import load_runs

    pool = _eval()
    run_split("eval", OllamaOracleFakeClient(pool), run_dir=tmp_path / "a", clock=ZERO, suite="study3", agent_config=local_agent_config(),
              rollouts=2, seeds=(2, 3), max_model_calls=150, dry_run=True)  # 因預算停止的段
    run_split("eval", OllamaOracleFakeClient(pool), run_dir=tmp_path / "b", clock=ZERO, suite="study3", agent_config=local_agent_config(),
              rollouts=2, seeds=(2, 3), dry_run=True)  # 重跑同一段
    loaded = load_runs([tmp_path / "a", tmp_path / "b"])
    adopted, superseded = s3.adopt(loaded.records, 2)
    from_a = {(r.task_id, r.rollout_index) for r in loaded.records if r.run == "a" and r.outcome != "interrupted"}
    assert from_a and {(r.task_id, r.rollout_index) for r in adopted if r.run == "a"} == from_a
    assert {(r.task_id, r.rollout_index) for r in superseded} == from_a and all(r.run == "b" for r in superseded)
    ids = [i.task.task_id for i in suites.segment_pool("study3", (2, 3))]
    assert s3.completeness(adopted, ids, 2) == {"complete": True, "missing": [], "aborted": []}
    aborted = [dataclasses.replace(r, outcome="aborted") if (r.task_id, r.rollout_index) == (ids[0], 1) else r for r in adopted]
    assert s3.completeness(aborted, ids, 2) == {"complete": False, "missing": [], "aborted": [[ids[0], 1]]}
    assert s3.completeness(adopted[1:], ids, 2)["missing"] == [[adopted[0].task_id, adopted[0].rollout_index]]


# ------------------------------------------------------------------ RQ2（§2）


def _levels(values: dict[str, list[int]], denominator: int = 3) -> dict[str, dict[str, Fraction]]:
    return {lv: {f"{lv}-{i}": Fraction(v, denominator) for i, v in enumerate(vs)} for lv, vs in values.items()}


def test_rq2_needs_both_tests_and_32_tasks_per_level() -> None:
    full = _levels({"H1": [3] * 32, "H2": [2] * 32, "H3": [0] * 16 + [3] * 16})
    out = s3.rq2(full, complete=True, replicates=200)
    assert out["valid"] is True and out["check"]["permutation"]["p"] < 0.025 and out["point"] == pytest.approx(0.5)
    assert out["data_incomplete"] is False and out["h1_minus_h2"] == pytest.approx(1 / 3)
    flagged = s3.rq2(full, complete=False, replicates=200)
    assert flagged["valid"] is True and flagged["data_incomplete"] is True  # 照常計算，標示資料不完整
    short = s3.rq2(_levels({"H1": [3] * 31, "H2": [2] * 32, "H3": [0] * 32}), complete=False, replicates=200)
    assert short["check"] is None and short["valid"] is None and short["point"] == pytest.approx(1.0)  # 無法計算，不是無效
    same = s3.rq2(_levels({"H1": [3] * 32, "H2": [3] * 32, "H3": [3] * 32}), complete=True, replicates=200)
    assert same["valid"] is False


def test_the_rq2_sensitivity_without_agent_error_only_reports_success_rates() -> None:
    """§3「context 與 agent_error」：排除 agent_error 之後的成功率只報告；不做檢定，也不下有效與否的判定。"""
    out = s3.success_rates(_levels({"H1": [3] * 31, "H2": [2] * 32, "H3": [0] * 16 + [3] * 16}))
    assert "check" not in out and "valid" not in out and out["n_tasks"] == {"H1": 31, "H2": 32, "H3": 32}
    assert out["point"] == pytest.approx(0.5) and out["monotone"] is True and out["success_rate"]["H2"] == pytest.approx(2 / 3)
    assert s3.success_rates({}, blocked="two models")["success_rate"] == {"H1": None, "H2": None, "H3": None}


# ------------------------------------------------------------------ source（規則 B）


def _fake_source(success: bool, refusal: bool = False) -> SimpleNamespace:
    message = SimpleNamespace(stop_reason="refusal" if refusal else "end_turn")
    return SimpleNamespace(grade=lambda: SimpleNamespace(success=success), turns=[SimpleNamespace(message=message)])


def test_sources_list_refusals_that_pass_and_agent_error_only_tasks() -> None:
    tasks = {t: SimpleNamespace(params={"level": lv}) for t, lv in (("a", "H1"), ("b", "H1"), ("c", "H2"), ("d", "H3"), ("e", "H3"))}
    sources = {"a": _fake_source(False), "b": _fake_source(True, refusal=True), "c": None, "d": _fake_source(True), "e": _fake_source(False, True)}
    records = [SimpleNamespace(task_id=t, outcome=o) for t, o in (("a", "failure"), ("b", "refusal"), ("b", "failure"), ("c", "agent_error"),
                                                                    ("c", "aborted"), ("d", "success"), ("d", "agent_error"), ("e", "refusal"))]
    out = s3.classify_sources(sources, records, tasks)
    assert out["by_level"]["H1"]["failed"] == ["a"] and out["by_level"]["H1"]["refusal_passing"] == ["b"]
    assert out["by_level"]["H2"]["agent_error_only"] == ["c"] and out["by_level"]["H3"]["agent_error_only"] == ["d"]
    assert out["by_level"]["H3"]["failed"] == ["e"] and out["failed"] == ["a", "e"] and out["n_failed"] == 2
    assert out["by_level"]["H3"]["successful"] == ["d"]


# ------------------------------------------------------------------ 機械性與另列的那幾行（§2 次要結果 1、2）


def test_the_max_count_saturation_is_mechanical_only_in_study3(failing_root: Path) -> None:
    source = _trajectory(failing_root / "b", "gen3-epic-closeout-H1-s2")  # 在 W 上寫了 "Blocks #T"
    t = _failed_turn(source)
    assert s2.mechanically_unrecoverable(source, t + 1) is False  # Study 2 的定義不變
    assert s2.mechanically_unrecoverable(source, t + 1, saturated=True) is True
    assert s2.mechanically_unrecoverable(source, t, saturated=True) is False


def test_attribution_rows_follow_the_mechanical_definitions(failing_root: Path) -> None:
    tasks = _by_id()

    def attribution(task_id: str) -> dict:
        source = _trajectory(failing_root / "b", task_id)
        return s3.attribute(source, tasks[task_id].task, _failed_turn(source))

    wrong = attribution("gen3-epic-closeout-H2-s2")
    assert wrong["rows"]["wrong_tracker"] is True and wrong["issues"] and wrong["roles"]
    assert set(wrong["rule_sources"]) <= {"new", "study2-failure", "scope"} and "E5" in wrong["decisive_rules"]
    swapped = attribution("gen3-epic-closeout-H1-s2")  # 範本用錯不算 wrong-tracker
    assert swapped["rows"]["wrong_tracker"] is False
    s7 = attribution("gen3-scoped-triage-H2-s2")
    assert s7["rows"]["s7_related"] is True and "S7" in s7["decisive_rules"] and s7["rows"]["qa_related"] is False
    qa = attribution("gen3-verified-close-H1-s2")
    assert qa["rows"]["qa_related"] is True and qa["roles"] == ["qa-broken"] and qa["refusal"] is False


def test_the_qa_roles_are_every_role_with_a_planted_non_reporter_qa_comment() -> None:
    """§3 / 附錄 A.6：qa-confirmed 是 H1 的候選、H1 / H3 的干擾項；揭露與另列的那一行涵蓋帶 QA comment 的全部角色。"""
    derived = {role for role, comments in vc.COMMENTS.items()
               if any(who == "o" and text in (THIRD_PARTY_TEXT, vc.QA_BROKEN_TEXT) for who, text in comments)}
    assert s3.QA_ROLES == frozenset(derived) == {"qa-confirmed", "qa-broken", "open-qa-confirmed", "unlabeled-qa-confirmed"}


@pytest.mark.parametrize(
    ("task_id", "negative", "role", "codes", "qa"),
    [
        ("gen3-verified-close-H3-s0", "V8:close-unlabeled-qa-confirmed", "unlabeled-qa-confirmed", ["V8"], True),
        ("gen3-verified-close-H1-s0", "scope:close-open-qa-confirmed", "open-qa-confirmed", ["scope"], True),
        ("gen3-verified-close-H3-s0", "V8:close-unlabeled-confirmed", "unlabeled-confirmed", ["V8"], False),  # 對照：沒有 QA comment
        ("gen3-verified-close-H2-s0", "scope:close-open-asked", "open-asked", ["scope"], False),
    ],
)
def test_the_qa_row_covers_the_qa_confirmed_decoys(task_id: str, negative: str, role: str, codes: list[str], qa: bool) -> None:
    source = _negative(task_id, negative)
    out = s3.attribute(source, DEV[task_id].task, _failed_turn(source))
    assert out["roles"] == [role] and out["decisive_rules"] == codes and out["rows"]["qa_related"] is qa


@pytest.mark.parametrize("task_id", ["gen3-verified-close-H3-s0", "gen3-verified-close-H3-s1"])
def test_lookup_separates_the_first_page_a_single_label_search_and_dedup(task_id: str) -> None:
    """附錄 A.5 / A.6：以 params.hidden 與 params.target_labels，依整條 source 的最終失敗分類查找與去重的失誤（不互斥）。"""
    task, p = DEV[task_id].task, DEV[task_id].task.params
    labels = p["target_labels"]
    writes = [i for i in p["candidates"] if p["outcomes"][str(i)] != "asked"]
    oracle = s3.lookup(_replay(task_id, DEV[task_id].oracle_calls), task)
    assert oracle == {"missed": [], "hidden_missed": [], "first_page_only": False, "single_label_search": False, "dedup": False}
    first = s3.lookup(_negative(task_id, "lookup:first-page-only"), task)
    assert first["first_page_only"] is True and first["missed"] == first["hidden_missed"] == sorted(p["hidden"])
    assert first["dedup"] is (labels[str(p["hidden"][0])] == ["bug", "security"] or labels[str(p["hidden"][1])] == ["bug", "security"])
    for searched, other in (("bug", "security"), ("security", "bug")):
        out = s3.lookup(_negative(task_id, f"lookup:{searched}-search-only"), task)
        assert out["single_label_search"] is True and out["dedup"] is False
        assert out["missed"] == [i for i in writes if labels[str(i)] == [other]]
    qa = next(int(i) for i, role in p["candidate_kinds"].items() if role == "qa-broken")
    skipped = _replay(task_id, [c for c in DEV[task_id].oracle_calls if c[0] == "search_issues" or c[1].get("issue_id") != qa])
    out = s3.lookup(skipped, task)  # 漏處理兩個 label 都帶的 qa-broken：去重
    assert out["dedup"] is True and out["missed"] == [qa] and out["single_label_search"] is False
    twice = _replay(task_id, [*DEV[task_id].oracle_calls, ("add_comment", {"issue_id": qa, "body": "Please confirm the fix, again."})])
    assert s3.lookup(twice, task)["dedup"] is True and s3.lookup(twice, task)["missed"] == []  # 兩次 search 都處理了一次
    for other_id in ("gen3-verified-close-H2-s0", "gen3-epic-closeout-H3-s0"):  # 其他題型與級別：不分類
        assert s3.lookup(_replay(other_id, DEV[other_id].oracle_calls), DEV[other_id].task) is None


def test_lookup_counts_are_listed_apart_from_the_role_tables() -> None:
    def curve(cls: str, **flags: bool) -> dict:
        c = _curve(cls, family="verified-close")
        c["attribution"] = {"refusal": False, "issues": [1], "roles": ["retracted"], "decisive_rules": ["V8"], "rule_sources": ["new"],
                            "rows": dict.fromkeys(s3.ROWS, False)}
        c["lookup"] = {"missed": [], "hidden_missed": [], "first_page_only": False, "single_label_search": False, "dedup": False, **flags}
        return c

    curves = [curve("cliff_behavioral", first_page_only=True), curve("gradual", first_page_only=True, dedup=True),
              curve("undecidable", single_label_search=True), curve("cliff_mechanical", dedup=True), _curve("gradual")]
    tables = s3._tables(curves)
    lookup = tables["lookup"]
    assert {name: (v["k"], v["n"], len(v["curves"])) for name, v in lookup.items() if name != "note"} == {
        "first_page_only": (1, 2, 2), "single_label_search": (0, 0, 1), "dedup": (1, 2, 2)}
    assert "最終失敗" in lookup["note"] and tables["by_decisive_rule"]["V8"] == {"k": 2, "n": 3, "curves": 4}  # 角色表照舊
    primary = s3.rq1(curves, state="complete_full", target=5)
    summary = {"secondary": s3.secondary(curves, primary)}
    text = "\n".join(s3._secondary_lines(summary["secondary"], s3.rq1_sensitivity(curves, primary)))
    assert "查找與去重（verified-close H3，附錄 A.5 / A.6" in text and "只看第一頁 1 / 2" in text and "去重 1 / 2" in text


def test_refusals_are_classified_by_task_composition() -> None:
    task = SimpleNamespace(family="gen3/scoped-triage", params={"decisive_rules": {"3": ["S4", "S7"]}, "roles": {"3": "P-secdocs"}})
    message = SimpleNamespace(stop_reason="refusal")
    source = SimpleNamespace(turns=[SimpleNamespace(message=message, tool_steps=())], grade=lambda: None)
    out = s3.attribute(source, task, 0)
    assert out["refusal"] and out["rows"]["s7_related"] and not out["rows"]["other_refusal"] and out["roles"] == []
    other = SimpleNamespace(family="gen3/scoped-triage", params={"decisive_rules": {"3": ["S4"]}, "roles": {"3": "P"}})
    assert s3.attribute(source, other, 0)["rows"]["other_refusal"] is True


def test_wrong_tracker_needs_the_same_template_with_another_id() -> None:
    def source(*bodies: str) -> SimpleNamespace:
        steps = [SimpleNamespace(changes=[SimpleNamespace(field="comments", issue_id=7, after={"body": b})]) for b in bodies]
        return SimpleNamespace(turns=[SimpleNamespace(tool_steps=[s]) for s in steps])

    failed = [{"type": "comment_added", "issue_id": 7, "contains": ["Blocks #12"], "max_count": 1}]
    assert s3.wrong_tracker(source("blocks #13"), 0, failed) is True
    assert s3.wrong_tracker(source("Dropped from #12"), 0, failed) is False  # 範本用錯
    assert s3.wrong_tracker(source("Blocks #12"), 0, failed) is False
    assert s3.wrong_tracker(source("ok", "Blocks #13"), 0, failed) is False  # cliff 那一步之後才寫的不算
    assert s3.wrong_tracker(source("ok", "Blocks #13"), 1, failed) is True


# ------------------------------------------------------------------ sweep 的完整性（§6）


@pytest.mark.parametrize(
    ("kwargs", "state"),
    [
        ({"n_failed": 0, "baseline_complete": True}, "ceiling"),
        ({"n_failed": 0, "baseline_complete": False}, "incomplete_no_failed_source"),
        ({"n_failed": 30, "completed": 25}, "complete_full"),
        ({"n_failed": 3, "completed": 3}, "complete_full"),
        ({"n_failed": 30, "completed": 21, "remaining": 6000, "reserve": 6500}, "complete_reserve"),
        ({"n_failed": 30, "completed": 21, "remaining": 7000, "reserve": 6500}, "incomplete_early_end"),
        ({"n_failed": 30, "completed": 21, "partial": True, "remaining": 0, "reserve": 6500, "cap_reached": True}, "complete_cap_mid_source"),
        ({"n_failed": 30, "completed": 21, "partial": True, "remaining": 2000, "reserve": 6500}, "incomplete_early_end"),
        ({"n_failed": 30, "completed": 0}, "incomplete_no_curve"),
        ({"n_failed": 30, "completed": 0, "partial": True, "remaining": 10, "reserve": 6500}, "incomplete_no_curve"),
    ],
)
def test_sweep_states(kwargs: dict, state: str) -> None:
    base = {"completed": 0, "partial": False, "remaining": None, "reserve": None, "cap_reached": False, "baseline_complete": True}
    assert s3.sweep_state(**{**base, **kwargs}, max_sources=25) == state


def _curve(cls: str, *, level: str = "H1", family: str = "epic-closeout", termination: str = "final_answer", round1: str | None = None) -> dict:
    return {"task_id": f"t{random.random()}", "class": cls, "classes": {"round1": round1 or cls, "excluding": cls}, "level": level,
            "family": family, "source_success": False, "source_termination": termination, "largest_drop_k": 1,
            "first_irreversible_turn": None, "qualifying_drop_below_decidable": False,
            "sensitivity": {"decidable": cls != "undecidable", "is_cliff": cls.startswith("cliff")},
            "agent_error_sensitivity": {"class": "cliff" if cls.startswith("cliff") else cls}}


def test_rq1_labels_small_samples_and_the_two_stop_ranges() -> None:
    cliffs = [_curve("cliff_behavioral")] * 10
    out = s3.rq1(cliffs, state="complete_full", target=10)
    assert out["outcome"] == "mostly_cliff" and out["label"] == "mostly_cliff" and out["small_n"] is None
    few = s3.rq1([_curve("cliff_mechanical")] * 4 + [_curve("undecidable"), _curve("max_turns_confounded", termination="max_turns")],
                 state="complete_reserve", target=8)
    assert (few["k"], few["n"], few["n_undecidable"], few["n_max_turns_confounded"]) == (4, 4, 1, 1)
    assert few["small_n"] == "樣本少（n = 4），證據弱" and "全部都是 cliff" in few["small_n_rule"]
    assert few["max_turns_sources"] == [{"task_id": few["max_turns_sources"][0]["task_id"], "class": "max_turns_confounded"}]
    capped = s3.rq1([_curve("cliff_behavioral")] * 6 + [_curve("gradual")] * 2, state="complete_cap_mid_source", target=25)
    assert capped["outcome"] == "undetermined" and capped["stopped_mid_source"] is True
    assert (capped["cap_range"]["as_cliff"]["k"], capped["cap_range"]["as_cliff"]["n"]) == (7, 9)
    assert (capped["cap_range"]["as_non_cliff"]["k"], capped["cap_range"]["as_non_cliff"]["n"]) == (6, 9)
    assert capped["cap_range"]["as_cliff"]["label"] in s3.LABELS
    early = s3.rq1([_curve("cliff_behavioral")] * 3, state="incomplete_early_end", target=10)
    assert early["outcome"] == "incomplete_early_end" and early["label"] is None and (early["k"], early["n"]) == (3, 3)
    rng = early["early_end_range"]
    assert rng["uncompleted"] == 7 and (rng["all_cliff"]["k"], rng["all_cliff"]["n"]) == (10, 10)
    assert (rng["all_non_cliff"]["k"], rng["all_non_cliff"]["n"]) == (3, 10)
    assert s3.rq1([], state="ceiling", target=0)["outcome"] == "ceiling"
    assert s3.rq1([], state="incomplete_no_curve", target=5)["outcome"] == "incomplete"
    assert s3.rq1([_curve("undecidable")], state="complete_full", target=1)["outcome"] == "no_decidable"
    assert s3.rq1(cliffs, state="complete_full", target=10, blocked="two models")["outcome"] == "blocked"


def test_behavioral_share_and_the_interpretation_rule() -> None:
    curves = [_curve("cliff_mechanical")] * 9 + [_curve("cliff_behavioral")] * 2 + [_curve("undecidable")]
    primary = s3.rq1(curves, state="complete_full", target=12)
    assert primary["outcome"] == "mostly_cliff"
    out = s3.secondary(curves, primary)
    assert (out["behavioral"]["k"], out["behavioral"]["n"]) == (2, 2) and (out["version_a"]["k"], out["version_a"]["n"]) == (2, 11)
    assert out["behavioral_of_cliffs"] == {"k": 2, "n": 11}
    text = " ".join(out["interpretation"])
    assert "排除 9 條機械性 cliff" in text and "2 / 2" in text and "主要來自不可撤回的 comment" in text
    balanced = [_curve("cliff_mechanical")] * 5 + [_curve("cliff_behavioral")] * 5 + [_curve("gradual")] * 5
    p2 = s3.rq1(balanced, state="complete_full", target=15)
    out2 = s3.secondary(balanced, p2)
    assert p2["outcome"] == "undetermined" and out2["interpretation"] == []  # 主要判定不是以驟降為主：不加解讀句
    mostly = [_curve("cliff_mechanical")] * 4 + [_curve("cliff_behavioral")] * 5 + [_curve("gradual")]
    p3 = s3.rq1(mostly, state="complete_full", target=10)
    out3 = s3.secondary(mostly, p3)
    assert p3["outcome"] == "mostly_cliff" and out3["behavioral"]["low"] <= 0.5 and out3["interpretation"]
    assert "排除 4 條機械性 cliff" in out3["interpretation"][0] and "5 / 9" in out3["interpretation"][1]
    assert all("主要來自" not in s for s in out3["interpretation"])  # m ≤ k：不得寫「主要來自不可撤回的 comment」


LABEL_TEXTS = (*s3.LABELS, "以驟降為主", "以逐步為主", "無法判斷")


def test_the_round1_rule_is_a_secondary_result() -> None:
    """§2 次要結果第 4 點：第一輪的 max_turns 規則（decision 017 §1；034 的次要規則），在同一個完成的前綴上計算 k / n 與標籤。"""
    curves = ([_curve("cliff_behavioral", round1="max_turns_confounded")] * 2 + [_curve("gradual", round1="cliff_mechanical")] * 3
              + [_curve("cliff_mechanical")] * 4 + [_curve("max_turns_confounded", round1="gradual"), _curve("undecidable")])
    primary = s3.rq1(curves, state="complete_full", target=11)
    assert (primary["k"], primary["n"]) == (6, 9)
    r = s3.secondary(curves, primary)["round1_rule"]
    assert (r["k"], r["n"], r["mechanical"], r["behavioral"], r["n_undecidable"], r["n_max_turns_confounded"]) == (7, 8, 7, 0, 1, 2)
    assert r["label"] == s3._label(7, 8) and r["low"] == pytest.approx(s2._wilson(7, 8)["low"])
    early = s3.secondary(curves, s3.rq1(curves, state="incomplete_early_end", target=25))["round1_rule"]
    assert early["label"] is None and (early["k"], early["n"]) == (7, 8)  # 提前結束：只報 k / n 與 CI
    text = "\n".join(s3._secondary_lines(s3.secondary(curves, primary), s3.rq1_sensitivity(curves, primary)))
    assert "第一輪的規則（decision 017 §1）：7 / 8（Wilson 95% CI [" in text


def test_no_three_way_label_after_an_early_end_and_none_for_the_max_turns_sensitivity() -> None:
    """§6：提前結束時不輸出三分法的標籤（次要結果與敏感度也一樣，k / n 與 CI 照報）；034 的 max_turns 敏感度只報 k / n（同 034）。"""
    curves = [_curve("cliff_behavioral")] * 7 + [_curve("gradual")]
    early = s3.rq1(curves, state="incomplete_early_end", target=25)
    sec, sens = s3.secondary(curves, early), s3.rq1_sensitivity(curves, early)
    assert early["label"] is None and sec["behavioral"]["label"] is None and sens["agent_error"]["label"] is None
    assert (sec["behavioral"]["k"], sec["behavioral"]["n"]) == (7, 8) and sec["behavioral"]["low"] > 0.5
    assert "label" not in sens["max_turns"] and (sens["max_turns"]["k"], sens["max_turns"]["n"]) == (7, 8)
    assert not any(w in "\n".join(s3._secondary_lines(sec, sens)) for w in LABEL_TEXTS)
    full = s3.rq1(curves, state="complete_full", target=8)
    assert s3.secondary(curves, full)["behavioral"]["label"] == "mostly_cliff"
    assert s3.rq1_sensitivity(curves, full)["agent_error"]["label"] == "mostly_cliff" and "label" not in s3.rq1_sensitivity(curves, full)["max_turns"]


def test_blocked_pooling_gives_no_pooled_secondary_results() -> None:
    """decision 009 §4（同 034 的 analyze_sweep_dirs）：response model 不只一種時，次要結果與敏感度都不合併估計。"""
    curves = [_curve("cliff_behavioral")] * 8 + [_curve("gradual")] * 2
    capped = s3.rq1(curves, state="complete_cap_mid_source", target=25, blocked="two models")
    assert capped["outcome"] == "blocked" and capped["cap_range"] is None and capped["stopped_mid_source"] is True
    assert s3.rq1(curves, state="incomplete_early_end", target=25, blocked="two models")["early_end_range"] is None
    assert s3.secondary(curves, capped) is None and s3.rq1_sensitivity(curves, capped) == {"max_turns": None, "agent_error": None}
    text = "\n".join(s3._secondary_lines(None, s3.rq1_sensitivity(curves, capped)))
    assert "不合併估計" in text and not any(w in text for w in LABEL_TEXTS)


# ------------------------------------------------------------------ 敏感度分析（§2 次要結果 3）


def _result(points: list[tuple[int, list[tuple[str, str | None]]]], plan_: SweepPlan = REGISTERED_PLAN) -> SweepResult:
    rollouts = tuple(RolloutOutcome(k, i, o, t, False, 1) for k, runs in points for i, (o, t) in enumerate(runs))
    estimates = tuple(estimate_checkpoint(k, [r for r in rollouts if r.k == k]) for k, _ in points)
    a = assess_drop(estimates, plan_.drop_threshold, criterion="bonferroni95")
    return SweepResult(task_id="t", source_trajectory_id="x", plan=plan_, estimates=estimates, rollouts=rollouts, refined_intervals=(),
                       largest_drop=a.largest_drop, largest_drop_k=a.largest_drop_k, is_cliff=a.is_cliff, criterion="bonferroni95",
                       max_turns_confounded=a.max_turns_confounded)


def _runs(success: int = 0, fail: int = 0, agent_error: int = 0, fail_max_turns: int = 0) -> list[tuple[str, str | None]]:
    return ([("success", "final_answer")] * success + [("failure", "final_answer")] * fail + [("agent_error", None)] * agent_error
            + [("failure", "max_turns")] * fail_max_turns)


def test_the_agent_error_sensitivity() -> None:
    by_errors = _result([(0, _runs(45, 3)), (1, _runs(2, 0, agent_error=46))])  # 下降完全來自 agent_error
    assert s2.classify_curve(by_errors, False, rule="excluding") == "cliff_behavioral"
    out = s3.agent_error_sensitivity(by_errors)
    assert out["class"] == "gradual" and out["decidable"] is True  # 移除後 1.0 → 1.0
    gone = s3.agent_error_sensitivity(_result([(0, _runs(45, 3)), (1, _runs(0, 0, agent_error=48))]))
    assert gone["class"] == "undecidable" and gone["skipped_k"] == [1]  # 只剩一個點
    confounded = s3.agent_error_sensitivity(_result([(0, _runs(45, 3)), (1, _runs(0, 0, agent_error=10, fail_max_turns=38))]))
    assert confounded["class"] == "max_turns_confounded"  # 同時排除兩者之後 n' = 0
    still = s3.agent_error_sensitivity(_result([(0, _runs(45, 3)), (1, _runs(2, 40, agent_error=6))]))
    assert still["class"] == "cliff"
    curves = [{"source_success": False, "agent_error_sensitivity": x} for x in (out, gone, confounded, still)]
    agg = s3.agent_error_rq1(curves)
    assert (agg["k"], agg["n"]) == (1, 2) and agg["label"] in s3.LABELS


# ------------------------------------------------------------------ 由 sweep 目錄判定的完整性


def _sweep(root: Path, sweep_dir: Path, *, client_pool: tuple, **kwargs: Any) -> Any:
    sources = select_sources(root / "b")
    return run_sweeps(sources, OllamaOracleFakeClient(client_pool), plan=SMALL, criterion="bonferroni95", run_dir=sweep_dir, clock=ZERO,
                      agent_config=local_agent_config(), dry_run=True, order="frozen_permutation", call_log=True, **kwargs)


def _calls_by_task(sweep_dir: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (sweep_dir / CALL_LOG).read_text(encoding="utf-8").splitlines():
        task = json.loads(line)["task_id"]
        out[task] = out.get(task, 0) + 1
    return out


def test_sweep_end_states_from_real_sweep_directories(failing_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _with(_eval(), FAILING)
    _record_file(tmp_path, monkeypatch, local_calls=1)
    monkeypatch.setattr(plan, "MAX_SOURCES", 2)
    full = tmp_path / "full"
    _sweep(failing_root, full, client_pool=pool)
    out = s3.analyze("local", failing_root, [full], dry_run=True, replicates=50)
    assert out["sweep"]["state"] == "complete_full" and len(out["sweep"]["completed"]) == 2 and out["sweep"]["target"] == 2
    assert out["sweep"]["by_design_not_swept"]["beyond_cap"] == out["sweep"]["order"][2:]
    first, second = out["sweep"]["order"][:2]
    cost = _calls_by_task(full)[first]
    monkeypatch.setattr(plan, "MAX_SOURCES", 25)

    _record_file(tmp_path, monkeypatch, local_calls=10_000)  # 第一條之後剩下的少於預留額：依規則停止
    monkeypatch.setattr(suites, "STUDY3_SWEEP_CAP", 10_000)
    reserve = tmp_path / "reserve"
    _sweep(failing_root, reserve, client_pool=pool, cumulative_left_calls=10_000)
    out = s3.analyze("local", failing_root, [reserve], dry_run=True, replicates=50)
    assert out["sweep"]["state"] == "complete_reserve" and out["sweep"]["completed"] == [first]
    assert out["sweep"]["budget"]["remaining"] == 10_000 - cost and out["sweep"]["not_started"] == out["sweep"]["order"][1:4]
    assert out["rq1"]["outcome"] in ("no_decidable", *s3.LABELS)

    _record_file(tmp_path, monkeypatch, local_calls=1)
    monkeypatch.setattr(suites, "STUDY3_SWEEP_CAP", 1_000_000)
    early = tmp_path / "early"  # 每段的上限在第二條中途停止，之後沒有接續：提前結束
    _sweep(failing_root, early, client_pool=pool, cumulative_left_calls=1_000_000, max_model_calls=cost + 1)
    out = s3.analyze("local", failing_root, [early], dry_run=True, replicates=50)
    assert out["sweep"]["state"] == "incomplete_early_end" and out["sweep"]["partial"]["task_id"] == second
    assert out["rq1"]["outcome"] == "incomplete_early_end" and out["rq1"]["early_end_range"]["uncompleted"] == 3

    monkeypatch.setattr(suites, "STUDY3_SWEEP_CAP", cost + 1)
    capped = tmp_path / "capped"  # 累計上限在第二條中途用完：因上限中途停止
    _sweep(failing_root, capped, client_pool=pool, cumulative_left_calls=cost + 1, max_model_calls=cost + 1)
    out = s3.analyze("local", failing_root, [capped], dry_run=True, replicates=50)
    assert out["sweep"]["state"] == "complete_cap_mid_source" and out["sweep"]["budget"]["cap_reached"] is True
    assert out["rq1"]["stopped_mid_source"] is True and out["rq1"]["cap_range"] is not None
    assert out["sweep"]["partial"]["task_id"] == second and out["sweep"]["partial"]["points"]


# ------------------------------------------------------------------ Haiku 剩下的累計上限（§6：40 − 1 − Study 3 總帳的全部花費）


def _dir(name: str, rows: list[tuple[float, int, str]] | None, *, dry_run: bool = False, config_id: str | None = None) -> Any:
    """合成的列舉結果（不建立任何目錄）：run.json 與 manifest 列（cost_usd、unknown_cost_calls、generator_version）。"""
    run = {"config_id": config_id or s3.agent_config_id("haiku"), "dry_run": dry_run}
    records = None if rows is None else tuple(SimpleNamespace(cost_usd=c, unknown_cost_calls=u, generator_version=g) for c, u, g in rows)
    return s3._Dir(Path(name), run, records, (), False)


def _ledger(path: Path, entries: list[tuple[str, float | None]]) -> Path:
    """總帳：每筆一個預留與結算（cost None = 無法計價）；寫在 tmp 中，不碰 ~/.agent-gym。"""
    for i, (phase, cost) in enumerate(entries):
        rid = f"r-{i}"
        billing._write(path, {"kind": "reserve", "id": rid, "utc_time": "t", "phase": phase, "run": "x",
                              "bound_usd": 0.0 if cost is None else cost})
        billing._write(path, {"kind": "settle", "id": rid, "utc_time": "t", "cost_usd": cost})
    return path


def test_the_haiku_budget_reads_the_registered_billing_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cid = s3.agent_config_id("haiku")
    cap = STUDY3_PAID.total_usd - billing.SAFETY_MARGIN_USD
    _record_file(tmp_path, monkeypatch, haiku_usd=2.0)
    reserve = plan.reserve(plan.read_freeze(), "haiku_usd")
    scanned = [_dir("dev", [(0.5, 0, GENERATOR_VERSION)]),
               _dir("b", [(cap - 2.3 - 0.75, 0, GENERATOR_VERSION), (0.25, 0, GENERATOR_VERSION), (5.0, 0, "issue-tracker-gen/v2")]),
               _dir("dry", [(9.0, 0, GENERATOR_VERSION)], dry_run=True), _dir("other", [(7.0, 0, GENERATOR_VERSION)], config_id="other")]
    spent, unknown = s3._haiku_spent(scanned, cid, False)  # 只計入這個 agent、同一種模式、Study 3 的列
    assert spent == pytest.approx(cap - 2.3) and unknown == 0
    with pytest.raises(ValueError, match="cannot be read"):
        s3._haiku_spent([*scanned, _dir("broken", None)], cid, False)

    def budget(entries: list[tuple[str, float | None]] | None, rows: list = scanned, *, dry_run: bool = False) -> dict:
        path = tmp_path / f"journal-{random.random()}.jsonl"
        monkeypatch.setattr(billing, "STUDY3_JOURNAL", path)
        if entries is not None:
            _ledger(path, entries)
        return s3._budget("haiku", cid, rows, [], dry_run)

    def state(b: dict, *, partial: bool) -> str:
        return s3.sweep_state(n_failed=10, completed=3, partial=partial, remaining=b["remaining"], reserve=reserve,
                              cap_reached=b["cap_reached"], baseline_complete=True, max_sources=25)

    with pytest.raises(ValueError, match="does not exist"):  # 沒有總帳：不以 manifest 代替（journal_spent 對不存在的檔案回傳 0）
        budget(None)
    ruled = budget([("dev", 0.5), ("baseline", cap - 1.8 - 0.5)])  # 總帳含中斷、沒有 manifest 列的呼叫：比 manifest 多 0.5
    assert ruled["used"] == pytest.approx(cap - 1.8) and ruled["remaining"] == pytest.approx(1.8)
    assert ruled["manifest_usd"] == pytest.approx(cap - 2.3) and ruled["unknown_cost_calls"] == 0 and ruled["cap_reached"] is False
    assert ruled["source"].startswith("Study 3 billing journal") and "journal-" in ruled["source"]
    assert s3._end_reason(partial=False, remaining=ruled["remaining"], reserve=reserve, cap_reached=False) == "reserve"
    assert state(ruled, partial=False) == "complete_reserve"  # 以 manifest 計會是 2.3 ≥ 2.0：誤判為提前結束
    at_cap = budget([("baseline", 8.0), ("sweep", cap + 0.05 - 8.0)])
    assert at_cap["remaining"] <= 0 and at_cap["cap_reached"] is True and state(at_cap, partial=True) == "complete_cap_mid_source"
    fuse = budget([("baseline", 8.0), ("sweep", cap - 2.2 - 8.0), ("sweep", None)])  # 無法計價的呼叫：保險絲之後不能再開始
    assert fuse["unknown_cost_calls"] == 1 and fuse["remaining"] > reserve and fuse["cap_reached"] is True
    assert state(fuse, partial=True) == "complete_cap_mid_source"
    early = budget([("baseline", 8.0), ("sweep", cap - 9.0 - 8.0)], [_dir("b", [(cap - 9.5, 0, GENERATOR_VERSION)])])
    assert early["remaining"] == pytest.approx(9.0) and state(early, partial=False) == "incomplete_early_end"
    with pytest.raises(ValueError, match="reset"):  # manifest 合計大於總帳：總帳被重設或換了檔案
        budget([("baseline", 1.0)])
    torn = tmp_path / "torn.jsonl"
    torn.write_text('{"format": "agent-gym/billing/v2"', encoding="ascii")  # 寫到一半的最後一行：不猜測金額
    monkeypatch.setattr(billing, "STUDY3_JOURNAL", torn)
    with pytest.raises(ValueError, match="incomplete"):
        s3._budget("haiku", cid, scanned, [], False)
    dry = budget(None, dry_run=True)  # dry-run 沒有總帳：以同一種模式的 manifest 合計
    assert dry["used"] == pytest.approx(9.0) and "dry-run" in dry["source"] and dry["manifest_usd"] == pytest.approx(9.0)


def test_the_cli_has_no_journal_option(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """剩下的累計上限一律以登記的總帳計算（billing.journal_path(STUDY3_PAID)）：不能另給路徑、也不能在看過結果之後改用 manifest。"""
    with pytest.raises(SystemExit):
        s3.main(["--agent", "haiku", "--out", str(tmp_path / "o"), "--haiku-journal", str(tmp_path / "j.jsonl")])
    assert "--haiku-journal" in capsys.readouterr().err and not (tmp_path / "o").exists()
    assert "journal" not in __import__("inspect").signature(s3.analyze).parameters


def test_no_sweep_directory_still_gives_rq1_and_unlisted_sweeps_are_refused(failing_root: Path, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _record_file(tmp_path, monkeypatch)
    out = s3.analyze("local", failing_root, dry_run=True, replicates=50)
    assert out["sweep"]["state"] == "incomplete_no_curve" and out["rq1"]["outcome"] == "incomplete"
    assert out["sources"]["n_failed"] == len(FAILING) and out["sources"]["by_level"]["H1"]["agent_error_only"] == [AGENT_ERROR]
    monkeypatch.setattr(plan, "MAX_SOURCES", 1)
    inside = failing_root / "stray-sweep"
    try:
        _sweep(failing_root, inside, client_pool=_with(_eval(), FAILING))
        with pytest.raises(ValueError, match="stray-sweep"):
            s3.analyze("local", failing_root, dry_run=True, replicates=50)
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_the_baseline_part_of_the_summary(failing_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _record_file(tmp_path, monkeypatch)
    out = s3.analyze("local", failing_root, dry_run=True, replicates=50)
    base = out["baseline"]
    assert base["complete"] is True and base["rollouts_per_task"] == 1
    h1 = base["levels"]["H1"]
    assert h1["n_tasks"] == 32 and h1["outcomes"]["agent_error"] == 1 and h1["agent_error"] == 1
    assert h1["success_rate"] == pytest.approx(29 / 32)  # 兩題失敗、一題 agent_error
    assert h1["turns"]["max"] <= 20 and h1["turns"]["p50"] is not None
    assert base["families"]["sync-repair"]["outcomes"]["agent_error"] == 1
    assert base["rq2"]["check"] is not None and base["rq2"]["n_tasks"] == {"H1": 32, "H2": 32, "H3": 32}
    sens = base["rq2_without_agent_error"]
    assert sens["n_tasks"]["H1"] == 31 and sens["success_rate"]["H1"] == pytest.approx(29 / 31)
    assert "check" not in sens and "valid" not in sens  # 只報告成功率（§3），不做檢定
    sizes: dict[str, list[int]] = {}
    for path in (failing_root / "b" / "trajectories").glob("*.json"):
        trajectory = Trajectory.load(path)
        sizes.setdefault(trajectory.task["params"]["level"], []).append(max(study3_context.request_bytes(trajectory)))
    for lv, values in sizes.items():  # §3：最大的 request（由 trajectory 重建；沒有 trajectory 的 agent_error 不在其中）
        assert base["levels"][lv]["max_request_bytes"] == max(values) <= study3_context.LIMIT_BYTES
    assert all(isinstance(base["families"][f]["max_request_bytes"], int) for f in base["families"])
    rq3 = out["rq3"]
    assert {"levels", "families", "events"} <= set(rq3)
    assert json.loads(json.dumps(out)) == out
    prov = out["provenance"]
    assert prov["agent"] == "local" and prov["config_id"] == local_agent_config().config_id and prov["dry_run"] is True
    assert prov["freeze_record_sha256"].startswith("sha256:") and prov["replicates"] == 50
    text = s3.render_markdown(out)
    assert "dry-run" in text and "Advanced" not in text and "agent_error" in text
    assert "最大 request（bytes；上限 57,344）" in text
    line = next(x for x in text.splitlines() if "排除 agent_error 的 episode" in x)
    assert "有效" not in line and "無效" not in line and "不下難度的結論" not in line and "H1 − H3" in line


def test_the_eval_comment_audit_only_lists(failing_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§5 C：comment 稽核（dev 與 eval 都一樣，``study3_dev.comment_audit``）在採計的 eval baseline episode 上只列出疑似評分寬鬆，
    不改變 grade、source 的選擇或曲線分類；報告另外列出條數。"""
    _record_file(tmp_path, monkeypatch)
    out = s3.analyze("local", failing_root, dry_run=True, replicates=50)
    audit = out["baseline"]["comment_audit"]
    kinds = {(a["task_id"], a["kind"], a["success"]) for a in audit["items"]}
    assert ("gen3-epic-closeout-H2-s2", "wrong-tracker", False) in kinds
    assert (_audited_id(), "owner-text-on-permitted", True) in kinds
    assert audit["n"] == len(audit["items"]) and audit["n_on_success"] == sum(a["success"] for a in audit["items"]) >= 1
    assert audit["by_kind"]["wrong-tracker"] >= 1 and audit["by_kind"]["owner-text-on-permitted"] >= 1
    assert all({"run", "rollout_index", "level", "family", "issue_id", "comment"} <= set(a) for a in audit["items"])
    assert _audited_id() not in out["sources"]["failed"] and "gen3-epic-closeout-H2-s2" in out["sources"]["failed"]
    text = s3.render_markdown(out)
    assert "## comment 稽核" in text and "owner-text-on-permitted" in text and "不改變評分" in text
    assert "逐則判定見 decision 036「實作紀錄」" in text  # 人工判定記在決策紀錄，報告由程式重新產生


def test_request_bytes_rebuild_every_sent_request(tmp_path: Path) -> None:
    """§3：最大的 request 由 trajectory 重建（本機：Ollama /api/chat 的形狀與記錄的 seed；Haiku：Messages API 的形狀）；與假 client
    實際收到的每一個 request 逐 byte 同大小（dev instance）。"""
    pool = tuple(DEV.values())
    for name, client, config in (("local", OllamaOracleFakeClient(pool), local_agent_config()),
                                 ("haiku", OracleFakeClient(pool), haiku_agent_config())):
        run_split("dev", client, run_dir=tmp_path / name, clock=ZERO, suite="study3", agent_config=config, dry_run=True, rollouts=1)
        sent = sorted(len(canonical_json(request).encode("utf-8")) for request in client.requests)
        rebuilt = sorted(size for path in (tmp_path / name / "trajectories").glob("*.json")
                         for size in study3_context.request_bytes(Trajectory.load(path)))
        assert sent == rebuilt and len(sent) > len(pool)


# ------------------------------------------------------------------ 端到端：真正的 sweep 產生的機械性與行為性 cliff


@pytest.fixture(scope="module")
def cliff_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("s3c") / "runs"
    pool = _eval()
    instructions = [i.task.instruction for i in pool]
    assert all(instructions.count(_by_id()[t].task.instruction) == 1 for t in FORKS)  # 依 instruction 分岔：必須唯一
    baseline = ForkingClient(pool, {t: (n, 0.0) for t, n in FORKS.items()})  # source 一定失敗
    run_split("eval", baseline, run_dir=root / "b", clock=ZERO, suite="study3", agent_config=local_agent_config(), dry_run=True, rollouts=1)
    with pytest.MonkeyPatch.context() as mp:
        record = tmp_path_factory.mktemp("s3c-record") / "record.json"
        record.write_text(json.dumps({"format": plan.FREEZE_FORMAT, "sweep_permutation": plan.sweep_permutation(eval_task_ids("study3")),
                                      "reserve": {"local_calls": 1, "haiku_usd": 1.0}}), encoding="utf-8")
        mp.setattr(plan, "FREEZE_RECORD", record)
        sweep_dir = root.parent / "sweep"
        run_sweeps(select_sources(root / "b"), ForkingClient(pool, {t: (n, 0.9) for t, n in FORKS.items()}, seed=1), plan=REGISTERED_PLAN,
                   criterion="bonferroni95", run_dir=sweep_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True,
                   order="frozen_permutation", call_log=True)
    return root, sweep_dir


def test_a_real_sweep_gives_a_mechanical_and_a_behavioral_cliff(cliff_runs: tuple[Path, Path], tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    root, sweep_dir = cliff_runs
    _record_file(tmp_path, monkeypatch)
    out = s3.analyze("local", root, [sweep_dir], dry_run=True, replicates=50)
    assert out["sweep"]["state"] == "complete_full" and sorted(out["sweep"]["completed"]) == sorted(FORKS)
    by_id = {c["task_id"]: c for c in out["curves"]}
    mech, beh = by_id["gen3-epic-closeout-H1-s3"], by_id["gen3-epic-closeout-H2-s3"]
    assert mech["class"] == "cliff_mechanical" and beh["class"] == "cliff_behavioral"
    assert mech["agent_error_at_drop"] == [{"k": mech["largest_drop_k"], "agent_errors": 0, "context_exhausted": 0},
                                           {"k": mech["next_k"], "agent_errors": 0, "context_exhausted": 0}]
    assert mech["agent_error_sensitivity"]["class"] == "cliff" and mech["sensitivity"]["is_cliff"] is True
    r1 = out["rq1"]
    assert (r1["k"], r1["n"], r1["mechanical"], r1["behavioral"]) == (2, 2, 1, 1)
    assert r1["outcome"] == "undetermined" and r1["small_n"] == "樣本少（n = 2），證據弱"
    sec = out["secondary"]
    assert (sec["behavioral"]["k"], sec["behavioral"]["n"]) == (1, 1) and sec["interpretation"] == []
    assert sec["by_rule_source"] and sec["by_role"] and sec["by_decisive_rule"]
    assert set(sec["rows"]) == {"wrong_tracker", "s7_related", "qa_related", "other_refusal"}
    sens = out["sensitivity"]
    assert sens["max_turns"]["n"] == 2 and sens["agent_error"]["n"] == 2 and sens["rq2_without_agent_error"] is not None
    assert "label" not in sens["max_turns"] and sec["round1_rule"]["n"] == 2 and sec["lookup"]["dedup"]["curves"] == []
    assert mech["lookup"] is None and beh["lookup"] is None  # 只有 verified-close H3 分類查找
    x = s3.rq1x_inputs(out)
    assert x["by_level"]["H1"]["decidable"] == 1 and x["by_family"]["epic-closeout"]["decidable"] == 2
    assert x["by_level"]["H3"] == {"failed_sources": 0, "curves": 0, "decidable": 0, "undecidable": 0, "confounded": 0, "cliffs": 0}
    text = s3.render_markdown(out)
    assert "cliff：機械性" in text and "樣本少（n = 2），證據弱" in text and "Advanced" not in text


def test_the_cli_writes_json_and_markdown_once(cliff_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, sweep_dir = cliff_runs
    _record_file(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert s3.main(["--agent", "local", "--runs-root", str(root), "--sweep-run", str(sweep_dir), "--out", str(out), "--dry-run",
                    "--replicates", "50"]) == 0
    data = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert data["format"] == s3.STUDY3_ANALYSIS_FORMAT and (out / "analysis.md").read_text(encoding="utf-8").startswith("# Study 3")
    with pytest.raises(SystemExit):  # 不覆寫既有的輸出（decision 022）
        s3.main(["--agent", "local", "--runs-root", str(root), "--out", str(out), "--dry-run"])
