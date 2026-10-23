"""Study 3 的凍結紀錄（decision 036 §9）：``docs/decisions/036-freeze-record.json``。不呼叫任何模型。

    uv run python -m experiments.study3_freeze --local-dev-run runs/<dev> [...] --haiku-dev-run runs/<dev> [...] \
        [--out docs/decisions/036-freeze-record.json] [--user-accepts-fewer-curves "<日期：使用者的決定>"]

在任何一個 agent 的第一次 eval 推論之前寫一次，兩個 agent 共用：

- **題目與評分**（``tasks``）：generator 版本、golden hash 與 RNG 指紋、grader 與 schema 版本、eval 96 題各自的 task hash 與
  format、附錄 A（逐字稿、規則來源、decisive_rules）與附錄 B 的 hash；
- **agent 與付費**（``agents``、``billing``）：兩個 agent 的 config_id、兩份總帳的路徑（以 ``~/`` 表示，不寫入使用者的路徑）、
  付費上限；
- **RQ1 與 sweep 的規則**（``protocol``）：decision 036 本文（背景到「放棄的替代方案」，不含實作紀錄）的 hash；R、plan、判定、主要
  規則、source 規則與「失敗的 source」、可判定門檻、結論規則、sweep 順序與 25 條、預留額的算法、§7 估算用的常數、各項上限（不得
  提高）、完整性與天花板的狀態、baseline 目錄的列入規則；完整的排列在頂層 ``sweep_permutation``；
- **統計**（``statistics``）：bootstrap、置換檢定、行為性 cliff 比例、機械性判定的範圍、四個敏感度分析、兩種偵測力模擬的參數、
  預期產量模擬的參數；數值一律取自執行時用的程式（例如 ``study3_analysis.rq2`` 的 replicates 與 seed）；
- **預留額**（``reserve``）：本機 = ``study3_stats.reserve_calls()``；Haiku = 這個呼叫數 × dev 實測每次呼叫成本的 P90；
- **dev 的結果與估算**（``dev``、``estimate``、``expected``、``power``）：兩個 agent 的 dev 摘要（Haiku 另含 dev 的實際花費）、
  §7 的估算、預期的 n 與判為以驟降為主的機率（``expected_yield``）、分級檢定與 RQ1（L = 16–20）的偵測力；
- ``code_commit``：寫紀錄時的程式 commit（工作區必須沒有修改、也沒有沒追蹤的檔案；忽略的 runs/ 等除外）。

``check_record(record)`` 確認紀錄與目前的程式一致：測試每次都檢查，凍結之後題目、agent、協定、統計與預留額都不得再改動，§9 的
其他項目也必須存在。§7 的估算要求回頭問使用者時（任一 agent 能 sweep 的失敗 source < 15），CLI 不寫紀錄。§7 只有兩種回應：
提高上限（程式與 decision log 的變更；之後重算的估算不再要求問）或接受較少的曲線（``--user-accepts-fewer-curves``，記進紀錄）。

**Haiku dev 的實際花費**（§7 的 N_haiku）：一律讀登記的 Study 3 總帳（``billing.journal_path(STUDY3_PAID)``；不能另給路徑）；
總帳不存在、有無法計價的呼叫、凍結之前已有 eval 的花費，或 runs/ 中 Haiku 的 Study 3 manifest 合計大於總帳（總帳被重設）→
拒絕。dry-run 的 dev 目錄（測試）以給定目錄的 manifest 合計。

**預期產量**（§2「偵測力與預期產量」，只用於報告）：每題有失敗 source 的機率以 dev 每題的失敗比例 f 計（1 − (1 − f)³；
f = 有 trajectory 而且沒有成功的 episode ÷ 全部 episode；agent_error / aborted 沒有 trajectory，不是 source，只計入分母）；
模擬 ``YIELD_REPS`` 次：每次先依各級的機率抽出失敗 source 的條數，再依 §6 的規則逐條 sweep（目標 min(25, N_failed)；開始新的
source 之前剩下的 ≥ 預留額；單條成本取自 ``study3_stats.cost_samples(L)``，Haiku 再乘上 dev 每次呼叫成本的平均；超過剩下的
上限 → 中途停止、不計入）。可判定的條數 ~ Binomial(完成條數, 4/5)（Study 2 的可判定比例）；判為以驟降為主的機率以二項分布
精確計算後平均。
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.baseline import BASELINE_MAX_TURNS, haiku_agent_config
from agents.ollama import local_agent_config
from apps.issue_tracker.schema import SCHEMA_VERSION
from env.hashing import canonical_json
from experiments import billing, study2_analysis, study2_dev, study3_analysis, study3_dev, study3_plan, study3_stats
from experiments.run_sweep import REGISTERED_CRITERION, REGISTERED_PLAN
from grader.schema import GRADER_VERSION
from replay.aborts import read_manifest
from replay.stats import wilson_interval
from replay.trajectory import load_strict_json
from tasks import generator_v3, suites

ROOT = Path(__file__).resolve().parents[1]
FREEZE_FORMAT = study3_plan.FREEZE_FORMAT
RECORD_PATH = Path("docs/decisions/036-freeze-record.json")
PROPOSAL = ROOT / "docs" / "decisions" / "036-study3-proposal.md"
DOC_PARTS = {  # decision 036 中凍結的部分：(開始的標題, 結束的標題；None = 檔尾)。實作紀錄不在其中，之後可以繼續寫
    "proposal": ("## 背景", "## 實作紀錄"),
    "appendix_a": ("## 附錄 A", "## 附錄 B"),
    "appendix_b": ("## 附錄 B", None),
}
SECTIONS = ("agents", "billing", "tasks", "protocol", "statistics", "sweep_permutation")
AGENT_KEYS = ("local", "haiku")
DEV_EPISODES_PER_LEVEL = 8 * study3_analysis.ROLLOUTS  # 4 題型 × dev seed 0–1 × R = 3（decision 036 §5 C）
POWER_REPS = 1000  # §2 RQ2 的偵測力模擬
POWER_SEED = 0
RQ1_POWER_LENGTHS = (16, 18, 20)  # §8 第 3 步：RQ1 在 L = 16–20 的偵測力
RQ1_POWER_REPS = 200
YIELD_REPS = 4000
YIELD_SEED = 0
DECIDABLE_SHARE = Fraction(4, 5)  # Study 2 的可判定比例（decision 036 §2「n 的意思」）；只用於預期的 n
TRUE_CLIFF = (0.8, 0.65)  # decision 036 §2 的表
ACCEPT_FEWER = "accept_fewer_curves"  # §7 的兩種回應中，記進紀錄的那一種
_COMMIT = re.compile(r"[0-9a-f]{40}")

DEFINITIONS = {
    "failed_source": "規則 B 的 source（每題 index 最小的 failure 或 refusal），而且 grade().success 為 False；source 為 None、"
                     "或規則 B 選到通過評分的 refusal 的題，不屬於 RQ1 的母體（decision 036 §2）",
    "conclusion": "cliff 比例的 Wilson 95% CI 下限 > 0.5 → mostly_cliff；上限 < 0.5 → mostly_gradual；其他 → undetermined",
    "small_n": "可判定的曲線 n < 10：判定照規則輸出，報告一律寫明「樣本少（n = …），證據弱」",
    "reserve": "開始每一條新的 source 之前，剩下的累計上限（本機 80,000 − 所有 sweep 段的呼叫數；Haiku 40 − 1 − Study 3 總帳的"
               "全部花費）必須 ≥ 預留額；預留額 = L = 20 時單條成本的 P99（本機以呼叫數；Haiku 再乘上 dev 每次呼叫成本的 P90）；"
               "接續的 source 不再檢查（decision 036 §6）",
    "completeness": "decision 036 §6「完整性與天花板」：前綴、依規則停止、因上限中途停止、天花板、資料不完整（含提前結束）",
    "listing_rule": "runs/ 中該 agent（config_id）、suite study3、split eval、非 dry-run 的全部 baseline 目錄由程式列舉；有 manifest "
                    "列或呼叫紀錄、卻沒有列入的目錄 → 拒絕（study3_analysis.check_baseline_set）",
    "caps": "eval baseline 開始之後不得提高任何上限（80,000 次、40 美元、25 條），也不得依中途看到的曲線提前停止或延長",
}
STATISTICS = {
    "rq2": "bootstrap（study2_analysis.bootstrap_difference：H1 − H3，B = 10,000、seed 0，2.5% 分位數 > 0）與精確置換檢定"
           "（p < 0.025）兩者都成立才有效；任一級不足 32 題 → 無法計算",
    "permutation_test": "task 層級的單側精確置換檢定（study3_stats.exact_permutation）：H1、H3 各 32 題的值（採計 rollout 的成功"
                        "比例）× 6 化為 0–6 的整數，以動態規劃求 32 / 32 所有分組下「H1 平均 − H3 平均」的精確分布；p = 統計量 ≥ "
                        "觀測值的分組比例",
    "behavioral_share": "k = 行為性 cliff 的條數；n = RQ1 的分母 − 機械性 cliff 的條數；Wilson 95% CI 與三分法標籤；(A) 版本 k 相同、"
                        "n = RQ1 的完整分母（只描述）；另列 k / (k + m)；解讀句依 decision 036 §2 次要結果第 1 點",
    "mechanical_scope": "study2_analysis.mechanically_unrecoverable(saturated=True)：034 §7 的情況，加上帶 max_count m 的 "
                        "comment_added，該 issue 新增的 comment 已達 m 則、其中符合 contains 的少於 min_count",
    "round1_rule": "§2 次要結果第 4 點：第一輪的 max_turns 規則（decision 017 §1）在主要判定的同一個完成前綴上的 k / n、Wilson 95% CI "
                   "與標籤",
    "lookup": "附錄 A.5 / A.6 的查找與去重（verified-close H3，study3_analysis.lookup）：依整條 source 的最終失敗分類，另列 k / n，"
              "不從角色表與 V8 的計數中移除；三者不互斥",
    "sensitivity": {
        "max_turns": "034 的 max_turns 敏感度分析：只移除 max_turns，agent_error 仍計為未成功；只報 k / n 與 Wilson 95% CI（沒有標籤，"
                     "同 034）",
        "agent_error": "每個 checkpoint 移除以 agent_error 結束的 rerun，以移除後的估計套用主要規則（Bonferroni 的 m 以移除後仍有"
                       "估計的 checkpoint 數計；可判定性重算）；k / n、Wilson 95% CI 與標籤，只報告",
        "rq2_without_agent_error": "排除 agent_error 之後各級的成功率與描述用的級別差（study3_analysis.success_rates）：全部 episode "
                                   "都是 agent_error 的題剔除；只報告，不做檢定，也不判定有效與否",
        "cap_mid_source": "因累計上限在 source 中途停止：只針對那一條，當成 cliff（k + 1、n + 1）與非 cliff（k、n + 1）各自計算 "
                          "k / n、Wilson 95% CI 與標籤",
    },
    "labels_suppressed": "RQ1 為天花板、資料不完整（含提前結束）或 blocked 時，次要結果與敏感度只報 k / n 與 CI，不輸出三分法的標籤；"
                         "因上限中途停止的範圍照 §6 輸出標籤",
    "blocked": "response model 不只一種 → RQ1 為 blocked，RQ1 的範圍、次要結果與敏感度都不合併估計（decision 009 §4、034）",
}


def _sha(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _home(path: Path) -> str:
    return "~/" + path.relative_to(Path.home()).as_posix()


def _default(function: Any, name: str) -> Any:
    return inspect.signature(function).parameters[name].default


def doc_sha256(part: str, text: str | None = None) -> str:
    """decision 036 中 DOC_PARTS[part] 那一段（開始的標題那一行起，到結束的標題之前或檔尾）的 sha256。標題不唯一或找不到 → ValueError。"""
    start, end = DOC_PARTS[part]
    text = PROPOSAL.read_text(encoding="utf-8") if text is None else text
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(start)]
    ends = [len(lines)] if end is None else [i for i, line in enumerate(lines) if line.startswith(end)]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError(f"decision 036 must contain exactly one {start!r} followed by {end or 'the end of the file'!r}")
    return "sha256:" + hashlib.sha256("".join(lines[starts[0]:ends[0]]).encode("utf-8")).hexdigest()


def appendix_a_sha256(text: str | None = None) -> str:
    """decision 036 附錄 A（從「## 附錄 A」到「## 附錄 B」之前）的 sha256（§9）。"""
    return doc_sha256("appendix_a", text)


def protocol() -> dict[str, Any]:
    """凍結的協定（不含 dev 結果、預留額與 commit）；check_record 與目前的程式比對這一部分。"""
    local, haiku = local_agent_config(), haiku_agent_config()
    lc = local.model_config
    spec, paid = suites.study("study3"), suites.STUDY3_PAID
    eval_pool = suites.suite_pool("study3", "eval")
    return {
        "agents": {
            "local": {"config_id": local.config_id, "model": lc.get("model"), "model_digest": lc.get("model_digest"),
                      "ollama_version": lc.get("ollama_version"), "max_turns": BASELINE_MAX_TURNS},
            "haiku": {"config_id": haiku.config_id, "model": haiku.model_config.get("model"), "max_turns": BASELINE_MAX_TURNS},
        },
        "billing": {
            "journals": {"study2": _home(billing.DEFAULT_JOURNAL), "study3": _home(billing.journal_path(paid))},
            "model": paid.model, "total_usd": paid.total_usd, "safety_margin_usd": billing.SAFETY_MARGIN_USD,
            "phase_caps": dict(paid.phase_caps), "rollouts": paid.rollouts, "sweep_order": paid.sweep_order,
        },
        "tasks": {
            "generator_version": generator_v3.GENERATOR_VERSION,
            "golden_pool_sha256": generator_v3.GOLDEN_POOL_SHA256,
            "golden_rng_sha256": generator_v3.GOLDEN_RNG_SHA256,
            "grader_version": GRADER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "eval": [{"task_id": i.task.task_id, "format": i.task.format, "sha256": _sha(i.task.to_dict())} for i in eval_pool],
            "appendix_a_sha256": doc_sha256("appendix_a"),
            "appendix_b_sha256": doc_sha256("appendix_b"),
        },
        "protocol": {
            "proposal_sha256": doc_sha256("proposal"),
            "eval_rollouts": study3_analysis.ROLLOUTS,
            "plan": {"coarse_stride": REGISTERED_PLAN.coarse_stride, "n_coarse": REGISTERED_PLAN.n_coarse,
                     "n_fine": REGISTERED_PLAN.n_fine, "drop_threshold": REGISTERED_PLAN.drop_threshold},
            "criterion": REGISTERED_CRITERION,
            "primary_rule": study2_analysis.PRIMARY_RULE,
            "source_rule": spec.source_rule,
            "decidable": str(study2_analysis.DECIDABLE),
            "conclusion_line": study2_analysis.CONCLUSION_LINE,
            "small_n": study3_analysis.SMALL_N,
            "sweep": {"order": spec.sweep_order, "permutation_key": study3_plan.PERMUTATION_KEY,
                      "reserve_length": study3_stats.RESERVE_LENGTH, "reserve_quantile": 0.99,
                      "reserve_reps": _default(study3_stats.reserve_calls, "reps"),
                      "reserve_seed": _default(study3_stats.reserve_calls, "seed")},
            "estimate": {"sweep_cap_calls": study3_stats.SWEEP_CAP_CALLS, "paid_total_usd": study3_stats.PAID_TOTAL_USD,
                         "safety_margin_usd": study3_stats.SAFETY_MARGIN_USD, "ask_below": study3_stats.ASK_BELOW,
                         "max_turns": study3_stats.MAX_TURNS, "reps": study3_stats.COST_REPS,
                         "seed": _default(study3_stats.estimate, "seed")},
            "caps": {
                "baseline_calls": dict(spec.baseline_caps), "eval_segments": [list(s) for s in spec.eval_segments],
                "sweep_segment_calls": spec.segment_cap, "sweep_calls": suites.sweep_cap("study3"),
                "max_sources": study3_plan.MAX_SOURCES, "not_raisable": True,
            },
            "states": {"sweep": list(study3_analysis.SWEEP_STATES), "end_reasons": list(study3_analysis.END_REASONS),
                       "labels": list(study3_analysis.LABELS)},
            "definitions": DEFINITIONS,
        },
        "statistics": {
            "bootstrap_replicates": _default(study3_analysis.rq2, "replicates"),
            "bootstrap_seed": _default(study3_analysis.rq2, "seed"),
            "tasks_per_level": study3_stats.TASKS_PER_LEVEL,
            "alpha": study3_stats.ALPHA,
            "power": {
                "rq2": {"p3": list(study3_stats.POWER_P3), "delta": list(study3_stats.POWER_DELTA),
                        "kappa": list(study3_stats.POWER_KAPPA), "reps": POWER_REPS, "seed": POWER_SEED},
                "rq1": {"lengths": list(RQ1_POWER_LENGTHS), "turn_out": list(study2_dev.POWER_TURN_OUT), "reps": RQ1_POWER_REPS,
                        "seed": POWER_SEED},
            },
            "expected_yield": {"reps": YIELD_REPS, "seed": YIELD_SEED, "decidable_share": str(DECIDABLE_SHARE),
                               "true_cliff": list(TRUE_CLIFF)},
            **STATISTICS,
        },
        "sweep_permutation": study3_plan.sweep_permutation(suites.eval_task_ids("study3")),
    }


def _same(a: Any, b: Any) -> bool:
    try:
        return canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError):  # NaN、Infinity 或不是 JSON 的值：不一致
        return False


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _reserve_ok(record: Mapping[str, Any]) -> bool:
    reserve, dev, estimate = (record.get(k) if isinstance(record.get(k), Mapping) else {} for k in ("reserve", "dev", "estimate"))
    haiku_dev = dev.get("haiku") if isinstance(dev.get("haiku"), Mapping) else {}
    haiku_estimate = estimate.get("haiku") if isinstance(estimate.get("haiku"), Mapping) else {}
    p90 = haiku_dev.get("call_usd_p90")
    return (reserve.get("local_calls") == study3_stats.reserve_calls() and _number(p90)
            and haiku_estimate.get("haiku_call_usd_p90") == p90 and reserve.get("haiku_usd") == reserve["local_calls"] * p90)


def _contents_ok(record: Mapping[str, Any]) -> bool:
    """§9 的其他項目都在：decision、code_commit、兩個 agent 的 dev / estimate / expected、兩種偵測力；估算要求問使用者時，紀錄中
    有使用者對這些 agent 的決定。"""
    if record.get("decision") != "036" or not isinstance(record.get("code_commit"), str) or not _COMMIT.fullmatch(record["code_commit"]):
        return False
    for key in ("dev", "estimate", "expected"):
        if not isinstance(record.get(key), Mapping) or set(record[key]) != set(AGENT_KEYS):
            return False
    power = record.get("power")
    if not isinstance(power, Mapping) or not power.get("rq2") or not isinstance(power.get("rq1"), Mapping):
        return False
    asks = sorted(a for a, e in record["estimate"].items() if not isinstance(e, Mapping) or e.get("ask_user") is not False)
    decision = record.get("user_decision")
    if not asks:
        return decision is None
    return (isinstance(decision, Mapping) and decision.get("response") == ACCEPT_FEWER and decision.get("agents") == asks
            and isinstance(decision.get("note"), str) and bool(decision["note"].strip()))


def check_record(record: Mapping[str, Any]) -> list[str]:
    """紀錄與目前的程式不同之處（空 list = 一致）：SECTIONS 依序，接著 reserve（預留額與 dev 的每次呼叫成本 P90）、contents（§9 的
    其他項目）、format。不會因為紀錄的內容而 raise。"""
    current = json.loads(canonical_json(protocol()))
    problems = [s for s in SECTIONS if not _same(record.get(s), current[s])]
    if not _reserve_ok(record):
        problems.append("reserve")
    if not _contents_ok(record):
        problems.append("contents")
    if record.get("format") != FREEZE_FORMAT:
        problems.append("format")
    return problems


# ---------------------------------------------------------------- 預期的 n 與偵測力（decision 036 §2，只用於報告）


def cliff_threshold(n: int) -> int | None:
    """判為以驟降為主所需的最小 k（Wilson 95% CI 下限 > 0.5）；n = 0 → None。"""
    return next((k for k in range(n + 1) if wilson_interval(k, n)[0] > study2_analysis.CONCLUSION_LINE), None) if n > 0 else None


def mostly_cliff_probability(n: int, p: float) -> float:
    """真實 cliff 比例 p、可判定曲線 n 條時判為以驟降為主的機率（二項分布精確計算）。"""
    k0 = cliff_threshold(n)
    if k0 is None:
        return 0.0
    return math.fsum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(k0, n + 1))


def failed_source_probability(review: Mapping[str, Any]) -> dict[str, float]:
    """各級別「每題至少有一條失敗的 source」的機率：dev 每題的失敗比例 f（有 trajectory 而且沒有成功的 episode ÷ 全部 episode；
    沒有 trajectory 的 agent_error / aborted 只計入分母：它們不是 source）→ 1 − (1 − f)^R，各題平均。"""
    counts: dict[tuple[str, str], list[int]] = {}
    for e in review["episodes"]:
        c = counts.setdefault((e["level"], e["task_id"]), [0, 0])
        c[0] += not e["success"]
        c[1] += 1
    for u in review["unfinished"]:
        counts.setdefault((u["level"], u["task_id"]), [0, 0])[1] += 1
    out = {}
    for level in sorted({lv for lv, _ in counts}):
        values = [1 - (1 - failed / total) ** study3_analysis.ROLLOUTS for (lv, _), (failed, total) in counts.items() if lv == level]
        out[level] = math.fsum(values) / len(values)
    return out


def expected_yield(review: Mapping[str, Any], *, length: int, budget: float, reserve: float, unit_cost: float = 1.0,
                   reps: int = YIELD_REPS, seed: int = YIELD_SEED) -> dict[str, Any]:
    """預期的失敗 source、完成的 sweep 條數、可判定曲線數與判為以驟降為主的機率（模組 docstring；只用於報告）。
    budget / reserve 的單位與 unit_cost × 呼叫數相同（本機：呼叫數、unit_cost = 1；Haiku：美元）。"""
    p_any = failed_source_probability(review)
    samples = study3_stats.cost_samples(length)
    rng = random.Random(f"study3/expected-yield|{seed}")
    done: list[int] = []
    for _ in range(reps):
        n_failed = sum(rng.random() < p for p in p_any.values() for _ in range(study3_stats.TASKS_PER_LEVEL))
        left, completed = budget, 0
        for _ in range(min(study3_plan.MAX_SOURCES, n_failed)):
            if left < reserve:  # §6：開始新的 source 之前
                break
            cost = rng.choice(samples) * unit_cost
            if cost > left:  # 累計上限在 source 中途用完：不計入
                break
            left -= cost
            completed += 1
        done.append(completed)
    share = DECIDABLE_SHARE.numerator / DECIDABLE_SHARE.denominator
    dist = Counter(done)

    def p_cliff(p: float) -> float:
        return math.fsum(count / reps * math.fsum(math.comb(m, n) * share ** n * (1 - share) ** (m - n) * mostly_cliff_probability(n, p)
                                                  for n in range(m + 1)) for m, count in dist.items())

    swept = math.fsum(done) / reps
    return {"p_any_failed_source": p_any, "expected_failed_sources": math.fsum(study3_stats.TASKS_PER_LEVEL * p for p in p_any.values()),
            "length": length, "budget": budget, "reserve": reserve, "unit_cost": unit_cost, "reps": reps, "seed": seed,
            "expected_swept": swept, "p_full": dist[study3_plan.MAX_SOURCES] / reps, "decidable_share": str(DECIDABLE_SHARE),
            "expected_n": swept * share, "n_used": math.floor(swept * share + 1e-9),
            "p_mostly_cliff": {str(p): p_cliff(p) for p in TRUE_CLIFF}}


# ---------------------------------------------------------------- 紀錄


def _dev_summary(review: Mapping[str, Any], runs: Sequence[str | Path], agent: str) -> dict[str, Any]:
    if review["missing"] or review["stopped_by_budget"] or any(
            level["episodes"] != DEV_EPISODES_PER_LEVEL for level in review["levels"].values()):
        raise ValueError(f"the {agent} dev is incomplete: every level needs {DEV_EPISODES_PER_LEVEL} episodes "
                         "(24 tasks × R = 3, decision 036 §5 C)")
    return {
        "runs": [Path(d).name for d in runs],
        "config_id": study3_dev.agent_config_id(agent),
        "levels": review["levels"],
        "to_classify": len(review["to_classify"]),
        "suspected_grader_issues": len(review["suspected_grader_issues"]),
        "comment_audit": len(review["comment_audit"]),
        "unfinished": len(review["unfinished"]),
        "stopped_by_budget": list(review["stopped_by_budget"]),
    }


def haiku_dev_spent(haiku_dev_runs: Sequence[str | Path]) -> dict[str, Any]:
    """§7 的「dev 的實際花費」（模組 docstring）。"""
    runs = [Path(d) for d in haiku_dev_runs]
    modes = {load_strict_json((d / "run.json").read_bytes()).get("dry_run") is True for d in runs}
    if len(modes) != 1:
        raise ValueError("the Haiku dev directories mix dry-run and formal outputs")
    if modes == {True}:
        return {"spent_usd": math.fsum(row.cost_usd for d in runs for row in read_manifest(d / "manifest.jsonl")),
                "source": "dry-run: manifests of the given dev directories"}
    path = billing.journal_path(suites.STUDY3_PAID)
    if not path.is_file():  # journal_spent 對不存在的檔案回傳 0
        raise ValueError(f"the Study 3 billing journal {path} does not exist; the Haiku dev spend is read from it (decision 036 §7)")
    spent = billing.journal_spent(path)  # 無法讀取或寫到一半的行 → ValueError
    if spent.unknown:
        raise ValueError(f"the Study 3 billing journal has {spent.unknown} call(s) of unknown cost; the dev spend cannot be computed")
    if spent.baseline or spent.sweep:
        raise ValueError("the Study 3 billing journal already has eval spending before the freeze (decision 036 §9)")
    manifest = billing.manifest_floor(haiku_agent_config().config_id, generator=generator_v3.GENERATOR_VERSION)
    if manifest > spent.total + 1e-9:
        raise ValueError(f"the Haiku Study 3 manifests record ${manifest:.4f} but the billing journal only ${spent.total:.4f}: the "
                         "journal was reset or replaced; stop and check (decision 036 §7)")
    return {"spent_usd": spent.dev, "manifest_usd": manifest, "source": "the Study 3 billing journal"}


def _call_cost_mean(review: Mapping[str, Any]) -> float:
    costs = [c for e in review["episodes"] for c in e["cost_per_turn"] if c is not None]
    if not costs:
        raise ValueError("the Haiku dev episodes carry no per-call cost")
    return math.fsum(costs) / len(costs)


def build_record(local_dev_runs: Sequence[str | Path], haiku_dev_runs: Sequence[str | Path], *, allow_dry_run: bool = False,
                 reps: int = study3_stats.COST_REPS, power_reps: int = POWER_REPS, rq1_power_reps: int = RQ1_POWER_REPS,
                 yield_reps: int = YIELD_REPS, commit: str | None = None) -> dict[str, Any]:
    """兩個 agent 的 dev 目錄 → 凍結紀錄（模組 docstring）。reps 等參數只供測試縮短；正式紀錄以預設值寫（測試核對）。"""
    reviews = {"local": study3_dev.review(local_dev_runs, agent="local", allow_dry_run=allow_dry_run),
               "haiku": study3_dev.review(haiku_dev_runs, agent="haiku", allow_dry_run=allow_dry_run)}
    dev = {"local": _dev_summary(reviews["local"], local_dev_runs, "local"),
           "haiku": _dev_summary(reviews["haiku"], haiku_dev_runs, "haiku")}
    spent = haiku_dev_spent(haiku_dev_runs)
    estimate = {"local": {**study3_dev.dev_estimate(reviews["local"], reps=reps), "reps": reps},
                "haiku": {**study3_dev.dev_estimate(reviews["haiku"], haiku_dev_spent_usd=spent["spent_usd"], reps=reps), "reps": reps}}
    call_p90, call_mean = estimate["haiku"]["haiku_call_usd_p90"], _call_cost_mean(reviews["haiku"])
    dev["haiku"].update({"spent": spent, "call_usd_p90": call_p90, "call_usd_mean": call_mean})
    local_calls = study3_stats.reserve_calls()
    reserve = {"local_calls": local_calls, "haiku_usd": local_calls * call_p90}
    expected = {
        "local": expected_yield(reviews["local"], length=estimate["local"]["length"], budget=study3_stats.SWEEP_CAP_CALLS,
                                reserve=local_calls, reps=yield_reps),
        "haiku": expected_yield(reviews["haiku"], length=estimate["haiku"]["length"], budget=estimate["haiku"]["sweep_usd_left"],
                                reserve=reserve["haiku_usd"], unit_cost=call_mean, reps=yield_reps),
    }
    rq1_rows = study2_dev.max_turns_power(reps=rq1_power_reps, seed=POWER_SEED, lengths=RQ1_POWER_LENGTHS)
    return {
        "format": FREEZE_FORMAT,
        "decision": "036",
        "code_commit": code_commit() if commit is None else commit,
        **protocol(),
        "reserve": reserve,
        "dev": dev,
        "estimate": estimate,
        "expected": expected,
        "power": {"rq2": study3_stats.rq2_power(reps=power_reps, seed=POWER_SEED),
                  "rq1": {"reps": rq1_power_reps, "rows": study2_dev.summarize_power(rq1_rows, RQ1_POWER_LENGTHS)}},
    }


def code_commit(root: str | Path | None = None) -> str:
    """HEAD 的 commit；工作區有修改、或有沒有追蹤的檔案（不在 .gitignore 中，例如還沒 commit 的程式）→ ValueError：code_commit
    必須就是寫紀錄的程式。忽略的目錄（runs/ 等）不影響。"""
    cwd = ROOT if root is None else Path(root)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        raise ValueError("the working tree has uncommitted or untracked files; commit the code before writing the freeze record")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:  # CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local-dev-run", action="append", required=True)
    parser.add_argument("--haiku-dev-run", action="append", required=True)
    parser.add_argument("--out", default=str(ROOT / RECORD_PATH))
    parser.add_argument("--user-accepts-fewer-curves", default=None, metavar="NOTE",
                        help="the user's decision (date and words) when the estimate asks (decision 036 §7)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} already exists; the freeze record is written once")
    try:
        record = build_record(args.local_dev_run, args.haiku_dev_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    for agent, e in record["estimate"].items():
        x = record["expected"][agent]
        print(f"{agent}: §7 sweepable failed sources (P90) {e['n']}; expected swept {x['expected_swept']:.1f}, expected n {x['n_used']}")
    asks = sorted(agent for agent, e in record["estimate"].items() if e["ask_user"])
    note = args.user_accepts_fewer_curves
    if asks and not (note and note.strip()):
        parser.error(f"fewer than {study3_stats.ASK_BELOW} sweepable failed sources for {', '.join(asks)}: ask the user (decision 036 "
                     "§7: raise that agent's cap, or accept fewer curves), then rerun with --user-accepts-fewer-curves")
    if note is not None and not asks:
        parser.error("--user-accepts-fewer-curves is only for an estimate that asks the user (decision 036 §7)")
    if asks:
        record["user_decision"] = {"response": ACCEPT_FEWER, "agents": asks, "note": note}
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
