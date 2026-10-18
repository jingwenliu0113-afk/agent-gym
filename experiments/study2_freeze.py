"""Study 2 的凍結紀錄（decision 034 §6）：``docs/decisions/034-freeze-record.json``。不呼叫任何模型。

    uv run python -m experiments.study2_freeze --dev-run runs/<dev> [--dev-run runs/<dev-rerun> ...] --out docs/decisions/034-freeze-record.json

記錄第一次 eval-seed 推論之前必須固定的所有值：

- agent 的設定：config_id、模型、digest、Ollama 版本；
- 題目：generator 版本、golden hash 與 RNG 指紋；task format、grader 與 schema 版本；eval 36 題各自的 task hash 與 format；
- 協定：R、plan、判定、主要規則、source 規則、各項呼叫上限；
- 程式的 git commit；
- dev 驗證：各級別的成功率、turn 數、max_turns 比例（含 L3），待判定與疑似 grader 漏洞的數目；
- 估算：source 規則 B 下的 P50 / P90 與上限，以及 P90 是否超過上限；
- 每條規則在 eval 中有作用的題數（``binding_rules_in_eval``；其他已知的涵蓋限制記在 decision 034）。

``check_record(record)`` 確認紀錄與目前的程式一致：測試在凍結之後每次都檢查，題目或協定在凍結後不得再改動。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from agents.ollama import local_agent_config
from apps.issue_tracker.schema import SCHEMA_VERSION
from env.hashing import canonical_json
from experiments import study2_analysis, study2_dev
from experiments.run_sweep import REGISTERED_CRITERION, REGISTERED_PLAN
from grader.schema import GRADER_VERSION
from tasks import generator_v2, suites
from tasks.generator_v2 import GENERATOR_VERSION, RULES

FREEZE_FORMAT = "agent-gym/study2-freeze/v1"
RECORD_PATH = Path("docs/decisions/034-freeze-record.json")
EVAL_ROLLOUTS = 3


def _sha(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def protocol() -> dict[str, Any]:
    """凍結的協定（不含 dev 結果與 commit）；check_record 與目前的程式比對這一部分。"""
    config = local_agent_config()
    mc = config.model_config
    eval_pool = suites.suite_pool("study2", "eval")
    binding: dict[str, int] = {}
    for instance in eval_pool:
        for rule in instance.task.params["binding_rules"]:
            binding[rule] = binding.get(rule, 0) + 1
    return {
        "agent": {"config_id": config.config_id, "model": mc.get("model"), "model_digest": mc.get("model_digest"),
                  "ollama_version": mc.get("ollama_version"), "max_turns": 20},
        "tasks": {
            "generator_version": GENERATOR_VERSION,
            "golden_pool_sha256": generator_v2.GOLDEN_POOL_SHA256,
            "golden_rng_sha256": generator_v2.GOLDEN_RNG_SHA256,
            "grader_version": GRADER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "eval": [{"task_id": i.task.task_id, "format": i.task.format, "sha256": _sha(i.task.to_dict())} for i in eval_pool],
            "binding_rules_in_eval": dict(sorted(binding.items())),
            "rules": {family: {level: list(rules) for level, rules in levels.items()} for family, levels in RULES.items()},
        },
        "protocol": {
            "eval_rollouts": EVAL_ROLLOUTS,
            "plan": {"coarse_stride": REGISTERED_PLAN.coarse_stride, "n_coarse": REGISTERED_PLAN.n_coarse,
                     "n_fine": REGISTERED_PLAN.n_fine, "drop_threshold": REGISTERED_PLAN.drop_threshold},
            "criterion": REGISTERED_CRITERION,
            "primary_rule": study2_analysis.PRIMARY_RULE,
            "source_rule": "first_failure",
            "baseline_caps": dict(suites.STUDY2_BASELINE_CAPS),
            "sweep_cap": suites.STUDY2_SWEEP_CAP,
            "segment_cap": suites.STUDY2_SEGMENT_CAP,
            "bootstrap_replicates": study2_analysis.BOOTSTRAP_REPLICATES,
        },
    }


def build_record(dev_runs: Sequence[str | Path], *, reps: int = 2000) -> dict[str, Any]:
    review = study2_dev.review(dev_runs)
    estimate = study2_dev.estimate_sweep_calls(review["levels"], source_rule="first_failure", reps=reps)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {
        "format": FREEZE_FORMAT,
        "decision": "034",
        "code_commit": commit,
        **protocol(),
        "dev": {
            "runs": [Path(d).name for d in dev_runs],
            "levels": review["levels"],
            "to_classify": len(review["to_classify"]),
            "suspected_grader_issues": len(review["suspected_grader_issues"]),
            "unfinished": len(review["unfinished"]),
            "stopped_by_budget": review["stopped_by_budget"],
        },
        "estimate": {"p50": estimate["p50"], "p90": estimate["p90"], "mean": round(estimate["mean"]), "reps": estimate["reps"],
                     "cap": suites.STUDY2_SWEEP_CAP, "over_cap": estimate["p90"] > suites.STUDY2_SWEEP_CAP},
    }


def check_record(record: dict[str, Any]) -> list[str]:
    """紀錄中的協定與目前的程式不同之處（空 list = 一致）。"""
    current = protocol()
    problems = []
    for section in ("agent", "tasks", "protocol"):
        if canonical_json(record.get(section)) != canonical_json(current[section]):
            problems.append(section)
    if record.get("format") != FREEZE_FORMAT:
        problems.append("format")
    return problems


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-run", action="append", required=True)
    parser.add_argument("--out", default=str(RECORD_PATH))
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} already exists; the freeze record is written once")
    try:
        record = build_record(args.dev_run, reps=args.reps)
    except ValueError as exc:
        parser.error(str(exc))
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}; P90 {record['estimate']['p90']} vs cap {record['estimate']['cap']}"
          + ("  — OVER THE CAP: ask the user (decision 034 §8)" if record["estimate"]["over_cap"] else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
