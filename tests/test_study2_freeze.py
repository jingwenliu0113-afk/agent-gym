"""experiments/study2_freeze.py（decision 034 §6 的凍結紀錄）的契約測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import study2_freeze as fz
from tasks import suites

ROOT = Path(__file__).resolve().parent.parent


def test_protocol_values() -> None:
    p = fz.protocol()
    assert p["agent"]["config_id"] == "sha256:9feb8eb7bb109ccc0308baba1150417f9cfd8b896d3b134f75beb735fa251a87"
    assert len(p["tasks"]["eval"]) == 36 and len({t["sha256"] for t in p["tasks"]["eval"]}) == 36
    assert p["protocol"]["source_rule"] == "first_failure" and p["protocol"]["primary_rule"] == "excluding"
    assert p["protocol"]["sweep_cap"] == 70_000 and p["protocol"]["segment_cap"] == 3000
    assert p["protocol"]["baseline_caps"] == {"dev": 2880, "eval": 4320} and p["protocol"]["eval_rollouts"] == 3
    # 有作用的規則在 eval 中的題數（decision 034「驗證 B 的修正」：E3 只在 2 題、E4 只在 1 題）
    assert p["tasks"]["binding_rules_in_eval"]["E3"] == 2 and p["tasks"]["binding_rules_in_eval"]["E4"] == 1


def test_check_record_detects_drift() -> None:
    record = {"format": fz.FREEZE_FORMAT, **fz.protocol()}
    assert fz.check_record(record) == []
    tampered = json.loads(json.dumps(record))
    tampered["tasks"]["eval"][0]["sha256"] = "sha256:" + "0" * 64
    tampered["protocol"]["sweep_cap"] = 1
    assert fz.check_record(tampered) == ["tasks", "protocol"]


@pytest.mark.skipif(not (ROOT / fz.RECORD_PATH).exists(), reason="凍結紀錄在 dev 驗證完成後才寫入")
def test_the_committed_freeze_record_matches_the_code() -> None:
    """凍結之後，題目、agent 與協定都不得再改動（decision 034 §6 / §9）。"""
    record = json.loads((ROOT / fz.RECORD_PATH).read_text(encoding="utf-8"))
    assert fz.check_record(record) == []
    assert suites.STUDY2_FROZEN is True
