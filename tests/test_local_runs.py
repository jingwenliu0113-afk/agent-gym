"""本機模型（Ollama）接進實驗流程的契約測試（decision 026 §2–§3）。

- ``make_policy`` 依 agent_config.policy 選擇 adapter；run_sweeps 沿用 source 自己的 agent 設定。
- ``OllamaOracleFakeClient``：Ollama 形狀的 oracle 假 client（dry-run 與測試用）。
- 以呼叫次數計的預算（``max_model_calls``）；成本上限不適用於本機模型。
- CLI：``--provider``、``--confirm-local-run``、GPU lock 環境變數、正式 run 要求凍結的設定、server 檢查。

全部只用假的 client，不連網、不使用 GPU。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from agents import ollama
from agents.baseline import ANTHROPIC_POLICY, AnthropicPolicy, baseline_agent_config
from agents.ollama import FROZEN_MODEL_DIGEST, OLLAMA_POLICY, OllamaPolicy, local_agent_config
from experiments import run_baseline, run_sweep
from experiments.analyze_baseline import analyze_runs
from experiments.analyze_sweep import analyze_sweep_runs
from experiments.run_baseline import (
    FAKE_OLLAMA_VERSION,
    OllamaOracleFakeClient,
    OracleFakeClient,
    check_budget_args,
    make_policy,
    run_split,
)
from experiments.run_sweep import run_sweeps, select_sources
from replay.checkpoint import make_checkpoint, rerun
from replay.localization import SweepPlan
from replay.trajectory import Trajectory
from tasks.generator import week1_pool
from tests.wrapper_env import WRAPPER_URL, frozen_server, inside_wrapper, outside_wrapper

ZERO = lambda: 0.0  # noqa: E731
POOL = week1_pool("eval")
EVAL_IDS = [instance.task.task_id for instance in POOL]
SMALL_PLAN = SweepPlan(coarse_stride=3, n_coarse=2, n_fine=2, drop_threshold=0.3)


@pytest.fixture(scope="module")
def local_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("local") / "eval"
    run_split("eval", OllamaOracleFakeClient(POOL), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config(), dry_run=True)
    return run_dir


# ---------------------------------------------------------------- dispatch 與 dry-run


def test_make_policy_dispatches_on_the_config_policy() -> None:
    assert isinstance(make_policy(OllamaOracleFakeClient(POOL), local_agent_config()), OllamaPolicy)
    assert isinstance(make_policy(OracleFakeClient(POOL), baseline_agent_config()), AnthropicPolicy)
    config = local_agent_config()
    with pytest.raises(ValueError):
        make_policy(OllamaOracleFakeClient(POOL), type(config)("unknown", config.prompt_version, "s", "", {}))


def test_a_local_dry_run_baseline_completes_every_task(local_baseline: Path) -> None:
    summary = json.loads((local_baseline / "summary.json").read_text(encoding="utf-8"))
    assert summary["config_id"] == local_agent_config().config_id
    assert all(result["outcome"] == "success" and result["cost_usd"] == 0.0 for result in summary["results"])
    trajectory = Trajectory.load(local_baseline / "trajectories" / f"{EVAL_IDS[0]}-r0.json")
    trajectory.verify()
    assert trajectory.agent.policy == OLLAMA_POLICY
    assert all(turn.meta["response_model"] == ollama.LOCAL_MODEL_CONFIG["model"] for turn in trajectory.turns)
    analysis = analyze_runs([local_baseline], replicates=20)
    assert analysis.dry_run is True and analysis.n_rollouts == 15


def test_the_fake_receives_valid_ollama_requests() -> None:
    from replay.runner import run_episode

    client = OllamaOracleFakeClient(POOL)

    run_episode(POOL[0].task, OllamaPolicy(client, local_agent_config()), max_turns=20, clock=ZERO)
    assert client.requests and all(request["stream"] is False and request["truncate"] is False for request in client.requests)
    assert client.requests[0]["messages"][1] == {"role": "user", "content": POOL[0].task.instruction}


def test_restore_resends_the_same_conversation() -> None:
    """decision 009 的不變式：restore 後送出的 messages / tools 與原 episode 相同（seed 是抽樣的隨機性，不在比較之內）。"""
    from replay.runner import run_episode

    original = OllamaOracleFakeClient(POOL)
    trajectory = run_episode(POOL[0].task, OllamaPolicy(original, local_agent_config()), max_turns=20, clock=ZERO)
    assert len(trajectory.turns) >= 2
    checkpoint = make_checkpoint(trajectory, 1)
    replayed = OllamaOracleFakeClient(POOL)
    rerun(checkpoint, OllamaPolicy(replayed, local_agent_config()), clock=ZERO)

    def without_seed(request: dict[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(request)
        del request["options"]["seed"]
        return request

    assert json.dumps(without_seed(replayed.requests[0]), sort_keys=True) == json.dumps(without_seed(original.requests[1]), sort_keys=True)


def test_a_local_dry_run_sweep_uses_the_sources_config(local_baseline: Path, tmp_path: Path) -> None:
    sources = select_sources(local_baseline)
    summary = run_sweeps(sources, OllamaOracleFakeClient(POOL), plan=SMALL_PLAN, criterion="wilson95", run_dir=tmp_path / "s",
                         clock=ZERO, replicates=20, dry_run=True)
    assert summary.config_id == local_agent_config().config_id and len(summary.completed) == 15
    analysis = analyze_sweep_runs([tmp_path / "s"], local_baseline, replicates=20)
    assert len(analysis.curves) == 15 and analysis.response_models == {ollama.LOCAL_MODEL_CONFIG["model"]: sum(
        len(result.rollouts) for result in summary.results.values())}
    with pytest.raises(ValueError):  # source 是本機 agent，不能用另一個設定 sweep
        run_sweeps(sources, OracleFakeClient(POOL), plan=SMALL_PLAN, criterion="wilson95", run_dir=tmp_path / "other", clock=ZERO,
                   agent_config=baseline_agent_config())


# ---------------------------------------------------------------- 預算：呼叫次數


def test_a_model_call_budget_stops_the_baseline(tmp_path: Path) -> None:
    summary = run_split("eval", OllamaOracleFakeClient(POOL), run_dir=tmp_path / "e", clock=ZERO, agent_config=local_agent_config(),
                        max_model_calls=10)
    assert summary.stopped_by_budget is True and 10 <= summary.model_calls < 10 + 20  # 在下一次嘗試之前停止
    assert len(summary.results) < 15


def test_a_model_call_budget_stops_the_sweep(local_baseline: Path, tmp_path: Path) -> None:
    summary = run_sweeps(select_sources(local_baseline), OllamaOracleFakeClient(POOL), plan=SMALL_PLAN, criterion="wilson95",
                         run_dir=tmp_path / "s", clock=ZERO, replicates=20, max_model_calls=25)
    assert summary.stopped_by_budget is True and 25 <= summary.model_calls < 25 + 20


@pytest.mark.parametrize("cap", [0, -1, 1.5, True, "10"])
def test_invalid_model_call_caps_are_rejected(cap: Any) -> None:
    with pytest.raises(ValueError):
        check_budget_args(None, local_agent_config(), cap)


def test_cost_caps_do_not_apply_to_local_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_split("eval", OllamaOracleFakeClient(POOL), run_dir=tmp_path / "e", clock=ZERO, agent_config=local_agent_config(),
                  max_cost_usd=5.0)


# ---------------------------------------------------------------- CLI：run_baseline


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``make_ollama_client`` 換成 Ollama 形狀的 oracle 假 client（模擬遠端 server）；記錄被建立的 URL。"""
    urls: list[str] = []

    def make(base_url: str, model_config: Any = None) -> OllamaOracleFakeClient:
        urls.append(base_url)
        return frozen_server([*POOL, *week1_pool("dev")])  # 模擬凍結的 server（decision 030）

    monkeypatch.setattr(run_baseline, "make_ollama_client", make)  # run_sweep 也經 run_baseline.start_local_run 建立 client
    return urls


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """decision 030 之後預設的設定就是凍結的設定。"""
    assert ollama.is_frozen(local_agent_config())


@pytest.fixture
def draft(monkeypatch: pytest.MonkeyPatch) -> None:
    """模擬尚未凍結的設定（decision 030 之前）。"""
    monkeypatch.setitem(ollama.LOCAL_MODEL_CONFIG, "model_digest", None)
    monkeypatch.setitem(ollama.LOCAL_MODEL_CONFIG, "ollama_version", None)


def _summary(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return data


def test_the_default_provider_is_the_local_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_baseline.main(["--split", "dev", "--dry-run"]) == 0
    assert _summary(tmp_path / "runs" / "dry-run-dev")["config_id"] == local_agent_config().config_id
    assert run_baseline.main(["--split", "dev", "--dry-run", "--provider", "anthropic", "--run-name", "a"]) == 0
    assert _summary(tmp_path / "runs" / "a")["config_id"] == baseline_agent_config().config_id


@pytest.mark.parametrize(
    "argv",
    [
        ["--confirm-local-run", "--max-model-calls", "50"],  # 沒有 GPU lock
        ["--confirm-local-run", "--lock", "--max-cost-usd", "5"],  # 本機不用成本上限
        ["--confirm-local-run", "--lock"],  # 沒有呼叫次數上限
        ["--confirm-paid-run", "--max-cost-usd", "5"],  # 付費模式只限 anthropic
        ["--confirm-local-run", "--lock", "--max-model-calls", "50", "--provider", "anthropic"],
        ["--confirm-local-run", "--lock", "--max-model-calls", "50", "--ollama-url", "http://gpu-server:11434"],
    ],
    ids=["no-lock", "cost-cap", "no-call-cap", "paid-with-ollama", "local-with-anthropic", "remote-url"],
)
def test_local_baseline_runs_are_refused_before_any_call(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    outside_wrapper(monkeypatch)
    if "--lock" in argv:
        argv = [item for item in argv if item != "--lock"]
        inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "dev", "--run-name", "x", *argv])
    assert made == [] and not (tmp_path / "runs" / "x").exists()


def test_a_local_dev_run_with_the_frozen_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list[str],
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    assert run_baseline.main(["--split", "dev", "--confirm-local-run", "--max-model-calls", "200", "--run-name", "smoke"]) == 0
    assert made == [WRAPPER_URL]
    summary = _summary(tmp_path / "runs" / "smoke")
    assert summary["config_id"] == local_agent_config().config_id and summary["results"]
    run_json = json.loads((tmp_path / "runs" / "smoke" / "run.json").read_text(encoding="utf-8"))
    assert run_json["dry_run"] is False
    assert FROZEN_MODEL_DIGEST in capsys.readouterr().err  # server 檢查的結果會印出


def test_a_formal_local_eval_baseline_needs_the_frozen_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              made: list[str], draft: None) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):  # 草案設定
        run_baseline.main(["--split", "eval", "--rollouts", "3", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "e"])
    assert made == [] and not (tmp_path / "runs" / "e").exists()


def test_a_formal_local_eval_baseline_with_the_frozen_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, made: list[str],
                                                             frozen: None) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):  # R 必須是 3（decision 025 §2）
        run_baseline.main(["--split", "eval", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "r1"])
    assert run_baseline.main(["--split", "eval", "--rollouts", "3", "--confirm-local-run", "--max-model-calls", "30",
                              "--run-name", "e"]) == 0
    summary = _summary(tmp_path / "runs" / "e")
    assert summary["stopped_by_budget"] is True and summary["config_id"] == local_agent_config().config_id


def test_a_frozen_config_that_does_not_match_the_server_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    monkeypatch.setitem(ollama.LOCAL_MODEL_CONFIG, "model_digest", "sha256:" + "f" * 64)
    monkeypatch.setitem(ollama.LOCAL_MODEL_CONFIG, "ollama_version", FAKE_OLLAMA_VERSION)
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "eval", "--rollouts", "3", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "e"])
    assert not (tmp_path / "runs" / "e").exists()


# ---------------------------------------------------------------- CLI：run_sweep


def _sweep(baseline: Path, *extra: str) -> int:
    return run_sweep.main(["--baseline-run", str(baseline), "--run-name", "s", *extra])


def test_a_dry_run_sweep_follows_the_baseline_provider(local_baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _sweep(local_baseline, "--plan", "3,2,2,0.3", "--criterion", "wilson95", "--dry-run") == 0
    assert _summary(tmp_path / "runs" / "s")["config_id"] == local_agent_config().config_id


@pytest.fixture(scope="module")
def frozen_local_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("frozen") / "eval"
    run_split("eval", frozen_server(POOL), run_dir=run_dir, clock=ZERO, agent_config=local_agent_config())  # 模擬正式 run
    return run_dir


REGISTERED = ["--plan", "2,10,48,0.3", "--criterion", "bonferroni95"]


@pytest.mark.parametrize(
    ("extra", "lock"),
    [
        (["--confirm-local-run", "--max-model-calls", "30", *REGISTERED], False),
        (["--confirm-local-run", *REGISTERED], True),
        (["--confirm-local-run", "--max-model-calls", "30", "--plan", "3,2,2,0.3", "--criterion", "wilson95"], True),
        (["--confirm-paid-run", "--max-cost-usd", "5", *REGISTERED], True),
    ],
    ids=["no-lock", "no-call-cap", "unregistered-plan", "paid-against-local-sources"],
)
def test_local_sweeps_are_refused_before_any_call(extra: list[str], lock: bool, frozen_local_baseline: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch, made: list[str], frozen: None) -> None:
    monkeypatch.chdir(tmp_path)
    outside_wrapper(monkeypatch)
    if lock:
        inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):
        _sweep(frozen_local_baseline, *extra)
    assert made == [] and not (tmp_path / "runs" / "s").exists()


def test_a_formal_local_sweep_needs_the_frozen_config(local_baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      made: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    with pytest.raises(SystemExit):  # 草案設定的 source（dry-run baseline 也不能用於正式 sweep）
        _sweep(local_baseline, "--confirm-local-run", "--max-model-calls", "30", *REGISTERED)
    assert made == []


def test_a_formal_local_sweep_runs_with_the_frozen_config(frozen_local_baseline: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          made: list[str], frozen: None) -> None:
    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    assert _sweep(frozen_local_baseline, "--confirm-local-run", "--max-model-calls", "30", *REGISTERED) == 0
    summary = _summary(tmp_path / "runs" / "s")
    assert summary["stopped_by_budget"] is True and made == [WRAPPER_URL]
    run_json = json.loads((tmp_path / "runs" / "s" / "run.json").read_text(encoding="utf-8"))
    assert run_json["dry_run"] is False and run_json["criterion"] == "bonferroni95"
    assert ANTHROPIC_POLICY != OLLAMA_POLICY


@pytest.mark.parametrize("error", ["transport", "protocol"])
def test_an_unreachable_server_is_a_clean_refusal(error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """2026-10-11 開發題測試：server 不在時，CLI 以清楚的訊息拒絕（不是 traceback），且不建立 run 目錄。"""
    from agents.ollama import OllamaProtocolError, OllamaTransportError

    class Down(OllamaOracleFakeClient):
        def version(self) -> dict[str, Any]:
            raise (OllamaTransportError if error == "transport" else OllamaProtocolError)("GET /api/version failed")

    monkeypatch.chdir(tmp_path)
    inside_wrapper(monkeypatch)
    monkeypatch.setattr(run_baseline, "make_ollama_client", lambda url, mc=None: Down(POOL))
    with pytest.raises(SystemExit):
        run_baseline.main(["--split", "dev", "--confirm-local-run", "--max-model-calls", "50", "--run-name", "x"])
    assert "not usable" in capsys.readouterr().err and not (tmp_path / "runs" / "x").exists()
