"""第七輪對抗式審查（分析工具、README 示範、Core 完整性）確認的低層缺陷之回歸測試。

- read_manifest：manifest.jsonl 不是一般檔案時漏出 IsADirectoryError（應為 ValueError）。
- AttemptRecord：超出 float 範圍的 int cost_usd 漏出 OverflowError；utc_time 沒有時區（分析時 UTC 日期隨主機時區改變）
  或轉成 UTC 時溢位（OverflowError）。
- load_strict_json：``1e400`` 被 json 轉成 inf，繞過「不接受非有限數」的規則。
- demo 文字宣稱「每個 checkpoint k」都重新 rollout，實際只估計了 coarse + refine 的點。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from env.hashing import canonical_json
from replay.aborts import ATTEMPT_RECORD_FIELDS, ATTEMPT_RECORD_FORMAT, AttemptRecord, read_manifest
from replay.trajectory import load_strict_json


def _row(**changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "format": ATTEMPT_RECORD_FORMAT,
        "task_id": "gen-assign-and-start-s1",
        "generator_version": "g",
        "config_id": "c",
        "grader_version": "v",
        "schema_version": "s",
        "max_turns": 20,
        "checkpoint_id": None,
        "checkpoint_turn": None,
        "source_trajectory_id": None,
        "rollout_index": 0,
        "attempt": 1,
        "outcome": "completed",
        "trajectory_id": "t",
        "termination": "final_answer",
        "error_class": None,
        "exception_type": None,
        "message": None,
        "http_status": None,
        "request_id": None,
        "model_calls_completed": 1,
        "billed_calls": 1,
        "cost_usd": 0.0,
        "unknown_cost_calls": 0,
        "partial_steps": None,
        "partial_forbidden_action_count": None,
        "partial_forbidden_attempt_count": None,
        "sdk_version": None,
        "utc_time": "2026-10-08T03:00:00+00:00",
    }
    assert set(row) == {"format", *ATTEMPT_RECORD_FIELDS}
    row.update(changes)
    return row


def test_the_row_helper_is_a_valid_record() -> None:
    AttemptRecord.from_dict(_row())


def test_read_manifest_rejects_a_directory_with_value_error(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").mkdir()
    with pytest.raises(ValueError):
        read_manifest(tmp_path / "manifest.jsonl")


def test_huge_int_cost_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AttemptRecord.from_dict(_row(cost_usd=10**400))
    path = tmp_path / "manifest.jsonl"
    path.write_text(canonical_json(_row(cost_usd=10**400)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


@pytest.mark.parametrize(
    "utc_time",
    [
        "2026-10-08T03:00:00",  # 沒有時區：UTC 日期會隨分析主機的時區改變
        "0001-01-01T00:00:00+14:00",  # 轉成 UTC 時溢位
        "9999-12-31T23:59:59-14:00",
        "yesterday",
        "",
    ],
)
def test_utc_time_must_be_an_aware_iso_timestamp_convertible_to_utc(utc_time: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AttemptRecord.from_dict(_row(utc_time=utc_time))
    path = tmp_path / "manifest.jsonl"
    path.write_text(canonical_json(_row(utc_time=utc_time)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


@pytest.mark.parametrize("utc_time", ["2026-10-08T03:00:00+00:00", "2026-10-08T03:00:00Z", "2026-10-02T01:30:00+02:00"])
def test_aware_timestamps_are_accepted(utc_time: str) -> None:
    assert AttemptRecord.from_dict(_row(utc_time=utc_time)).utc_time == utc_time


@pytest.mark.parametrize("text", [b"1e400", b"-1e400", b'{"a": [1, 2e999]}', b"1E+400"])
def test_load_strict_json_rejects_floats_that_overflow_to_infinity(text: bytes) -> None:
    with pytest.raises(ValueError):
        load_strict_json(text)


@pytest.mark.parametrize("text", [b"1e308", b"-0.0", b"0.1", b"5e-324", b"12345678901234567890"])
def test_load_strict_json_keeps_finite_numbers_exactly_as_json_does(text: bytes) -> None:
    assert load_strict_json(text) == json.loads(text)


def test_demo_text_names_exactly_the_checkpoints_that_were_rerun(tmp_path: Path) -> None:
    from experiments.demo_synthetic import render_demo, run_demo

    result = run_demo(tmp_path / "demo")
    text = render_demo(result)
    ks = [e.k for e in result.sweep.estimates]
    n_checkpoints = len(result.source.turns)
    assert len(ks) < n_checkpoints  # 前提：coarse + refine 沒有估計每個 checkpoint
    assert "每個 checkpoint" not in text
    assert f"k = {', '.join(str(k) for k in ks)}" in text
    assert f"{len(ks)} / {n_checkpoints}" in text


def test_readme_does_not_claim_trajectories_are_replayed_on_load() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "trajectory / checkpoint 自帶前綴並在載入時重放驗證" not in readme
