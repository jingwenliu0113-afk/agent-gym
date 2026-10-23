"""題目集合（suite）與研究登記表：第一輪的 ``week1``（generator v1）與 Study 2 的 ``study2``（generator v2；decision 034）。

run 與分析工具由 suite 取得 pool；分析時由 manifest 的 generator_version 或 source 的 task id 推出 suite，所以第一輪的
run.json、manifest 與 summary 格式都不變。

**研究登記表**（decision 036 §8 第 1 步）：每個 suite 的開關、呼叫上限、source 規則與付費規格都由 ``STUDIES`` 提供，
run 工具不再寫死任何 suite 名稱。沒有登記的 suite 一律拒絕；有開關的研究必須登記全部上限（``StudySpec`` 建立時檢查），
所以之後新增的研究不會因為少寫一個欄位而在沒有上限的情況下推論（fail closed）。

Study 2 的模型推論有兩道開關（decision 034 §5 / §9）：
- ``STUDY2_APPROVED``：使用者確認研究方案之後才設為 True；在這之前不對任何 Study 2 的題目做模型推論（dev 也一樣）。
- ``STUDY2_FROZEN``：dev 驗證完成、凍結紀錄 commit 之後才設為 True；在這之前不對 eval seed 做推論。
dry-run（oracle 假 client）不受限制。開關與上限在每次檢查時才讀取（測試與凍結時可以改變）。

付費規格（``PaidSpec``）：Study 2 的 Claude Haiku 5.5（decision 035，已結案：``STUDY2_PAID_CLOSED``）；Study 3 的
``STUDY3_PAID``（decision 036 §7：獨立總帳、40 美元、dev ≤ 3、eval baseline ≤ 8；Study 3 加進 SUITES 時登記）。
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from tasks.generator import GENERATOR_VERSION as WEEK1_GENERATOR, GeneratedInstance, week1_pool
from tasks.generator_v2 import GENERATOR_VERSION as STUDY2_GENERATOR, study2_pool
from tasks.generator_v3 import GENERATOR_VERSION as STUDY3_GENERATOR, study3_pool

SUITES: tuple[str, ...] = ("week1", "study2", "study3")
_GENERATORS = {WEEK1_GENERATOR: "week1", STUDY2_GENERATOR: "study2", STUDY3_GENERATOR: "study3"}
SOURCE_RULES = ("rollout0", "first_failure")  # 第一輪：rollout 0；規則 B：index 最小的失敗 rollout（decision 034）
PAID_PHASES = ("dev", "baseline", "sweep")

STUDY2_APPROVED = True  # 使用者於 2026-10-17 核准 decision 034（source 規則 B、排除後仍合格的規則、上限 70,000）
STUDY2_FROZEN = True  # dev 驗證（2026-10-17）沒有缺陷修正、P90 26,632 ≤ 70,000；凍結紀錄 docs/decisions/034-freeze-record.json
STUDY2_SWEEP_CAP: int | None = 70_000  # 正式 sweep 的累計呼叫上限（decision 034 §8；使用者選擇 source 規則 B，2026-10-17）
STUDY2_SEGMENT_CAP = 3000  # 正式 sweep 每段的呼叫上限；段與段之間釋放 GPU lock（decision 034 §8）
STUDY2_BASELINE_CAPS = {"dev": 2880, "eval": 4320}  # 每輪 dev / eval baseline 的呼叫上限（24 或 36 題 × R 3 × max_turns 20 × 2 次嘗試）

# Study 3（decision 036，使用者 2026-10-23 核准）：§8 第 1–4 步完成之前不推論（STUDY3_APPROVED；2026-10-23 第 1–4 步完成後打開，
# 773cd1a）；凍結紀錄 commit 之前不推論 eval（STUDY3_FROZEN 與 docs/decisions/036-freeze-record.json 一起改，測試核對）。
STUDY3_APPROVED = True
STUDY3_FROZEN = True  # 2026-10-24：凍結紀錄 docs/decisions/036-freeze-record.json（code_commit b512397）
STUDY3_SWEEP_CAP: int | None = 80_000  # 本機 sweep 的累計呼叫上限（decision 036 §7）
STUDY3_SEGMENT_CAP = 3000  # 本機 sweep 每段的呼叫上限；段與段之間釋放 GPU lock
STUDY3_BASELINE_CAPS = {"dev": 2880, "eval": 2880}  # dev 一輪（24 題 × 3）與 eval 每個 seed 區塊（24 題 × 3）：× max_turns 20 × 2
STUDY3_EVAL_SEGMENTS = ((2, 3), (4, 5), (6, 7), (8, 9))  # 本機 eval baseline 的 4 段（decision 036 §6）


@dataclass(frozen=True)
class PaidSpec:
    """一個研究的付費推論（Anthropic API）規格。金額單位為美元。

    - ``journal``：``experiments.billing`` 中保存總帳路徑的屬性名稱（每次使用時才讀取）；
    - ``phase_caps``：各 phase 所有嘗試合計的上限（沒有列出的 phase 只受總上限限制）；``total_usd``：全部合計的上限；
    - ``splits``：可以付費推論的 split；baseline 必須使用 ``rollouts`` 個 rollout；
    - ``sweep_order``：付費 sweep 的執行順序；``generator``：總帳重設的核對只計入這個 generator 的 manifest 列；
    - ``closed_flag``：本模組中表示「已結案、拒絕付費 run」的屬性名稱（None = 開放）。"""

    name: str
    decision: str
    model: str
    journal: str
    total_usd: float
    phase_caps: Mapping[str, float]
    splits: tuple[str, ...]
    rollouts: int
    sweep_order: str
    generator: str
    closed_flag: str | None = None

    def __post_init__(self) -> None:
        if set(self.phase_caps) - set(PAID_PHASES) or not set(self.splits) <= {"dev", "eval"} or not self.splits:
            raise ValueError("a paid spec names unknown phases or splits")
        if not all(0 < cap <= self.total_usd for cap in self.phase_caps.values()):
            raise ValueError("every phase cap must be > 0 and at most the total")


# decision 035：Claude Haiku 5.5 跑 Study 2 的 eval（付費 API；使用者 2026-10-22 授權）。不跑 dev（使用者決定不試跑）。
STUDY2_PAID_MODEL = "claude-haiku-5-5"
STUDY2_PAID_TOTAL_USD = 70.0  # eval baseline、sweep 與所有中止的嘗試合計
STUDY2_PAID_BASELINE_USD = 15.0  # eval baseline 的上限
STUDY2_PAID_ROLLOUTS = 3
STUDY2_PAID_CLOSED = True  # decision 035 已結案（2026-10-23）；decision 036 §7：Study 2 的付費 run 一律拒絕
STUDY2_PAID = PaidSpec(
    name="Study 2", decision="035", model=STUDY2_PAID_MODEL, journal="DEFAULT_JOURNAL", total_usd=STUDY2_PAID_TOTAL_USD,
    phase_caps={"baseline": STUDY2_PAID_BASELINE_USD}, splits=("eval",), rollouts=STUDY2_PAID_ROLLOUTS,
    sweep_order="failed_first", generator=STUDY2_GENERATOR, closed_flag="STUDY2_PAID_CLOSED",
)
# decision 036 §7：Claude Haiku 5.5 跑 Study 3（使用者 2026-10-23 核准；硬上限 40 美元）。dev 只限 seed 0–1（dev split）、R = 3。
STUDY3_PAID = PaidSpec(
    name="Study 3", decision="036", model=STUDY2_PAID_MODEL, journal="STUDY3_JOURNAL", total_usd=40.0,
    phase_caps={"dev": 3.0, "baseline": 8.0}, splits=("dev", "eval"), rollouts=3,
    sweep_order="frozen_permutation", generator="issue-tracker-gen/v3",
)


@dataclass(frozen=True)
class StudySpec:
    """一個 suite 的登記內容。開關與 sweep 上限以本模組的屬性名稱登記，每次檢查時才讀取。

    - 有開關（``approved_flag``）的研究必須登記 ``frozen_flag``、``baseline_caps``（每個 split 一次 baseline run 的呼叫上限）、
      ``segment_cap``（sweep 每段的呼叫上限）與 ``sweep_cap_flag``（累計上限）；
    - 沒有開關的研究（只有第一輪）必須寫明 ``ungated_reason``；
    - ``eval_segments``：eval baseline 依 seed 區塊分段時登記的區塊（本機的 eval baseline 必須逐段執行；None = 不分段）。"""

    suite: str
    generator: str
    pool: Callable[[str], tuple[GeneratedInstance, ...]]
    decision: str
    source_rule: str
    approved_flag: str | None = None
    frozen_flag: str | None = None
    baseline_caps: Mapping[str, int] | None = None
    segment_cap: int | None = None
    sweep_cap_flag: str | None = None
    paid: PaidSpec | None = None
    ungated_reason: str | None = None
    eval_segments: tuple[tuple[int, ...], ...] | None = None
    sweep_order: str = "pool"  # 本機正式 sweep 的執行順序（Study 3：凍結紀錄中的排列，decision 036 §6）

    def __post_init__(self) -> None:
        if self.source_rule not in SOURCE_RULES:
            raise ValueError(f"source_rule must be one of {SOURCE_RULES}")
        if self.eval_segments is not None:
            seeds = [seed for segment in self.eval_segments for seed in segment]
            if not self.eval_segments or not all(segment for segment in self.eval_segments) or len(set(seeds)) != len(seeds):
                raise ValueError(f"{self.suite}: eval_segments must be non-empty, disjoint seed blocks")
        if self.approved_flag is None:
            if self.frozen_flag is not None or not self.ungated_reason:
                raise ValueError(f"{self.suite}: a study without an approval switch must say why (ungated_reason)")
            return
        missing = [name for name in ("frozen_flag", "baseline_caps", "segment_cap", "sweep_cap_flag") if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.suite}: a gated study must register {', '.join(missing)}")
        if set(self.baseline_caps) != {"dev", "eval"}:
            raise ValueError(f"{self.suite}: baseline_caps must give both dev and eval")
        caps = [*self.baseline_caps.values(), self.segment_cap]
        if not all(type(cap) is int and cap > 0 for cap in caps):
            raise ValueError(f"{self.suite}: registered call caps must be positive integers")

    @property
    def gated(self) -> bool:
        return self.approved_flag is not None


STUDIES: dict[str, StudySpec] = {
    "week1": StudySpec(
        suite="week1", generator=WEEK1_GENERATOR, pool=week1_pool, decision="025", source_rule="rollout0",
        ungated_reason="第一輪（decision 025 / 030）：開關與上限在 CLI 的 run 設計中檢查；已封存",
    ),
    "study2": StudySpec(
        suite="study2", generator=STUDY2_GENERATOR, pool=study2_pool, decision="034", source_rule="first_failure",
        approved_flag="STUDY2_APPROVED", frozen_flag="STUDY2_FROZEN", baseline_caps=STUDY2_BASELINE_CAPS,
        segment_cap=STUDY2_SEGMENT_CAP, sweep_cap_flag="STUDY2_SWEEP_CAP", paid=STUDY2_PAID,
    ),
    "study3": StudySpec(
        suite="study3", generator=STUDY3_GENERATOR, pool=study3_pool, decision="036", source_rule="first_failure",
        approved_flag="STUDY3_APPROVED", frozen_flag="STUDY3_FROZEN", baseline_caps=STUDY3_BASELINE_CAPS,
        segment_cap=STUDY3_SEGMENT_CAP, sweep_cap_flag="STUDY3_SWEEP_CAP", paid=STUDY3_PAID, eval_segments=STUDY3_EVAL_SEGMENTS,
        sweep_order="frozen_permutation",
    ),
}


def _flag(name: str) -> Any:
    return getattr(sys.modules[__name__], name)


def study(suite: str) -> StudySpec:
    """登記的研究；沒有登記 → ValueError。"""
    if suite not in SUITES or suite not in STUDIES:
        raise ValueError(f"suite must be one of: {', '.join(SUITES)}")
    return STUDIES[suite]


def source_rule(suite: str) -> str:
    return study(suite).source_rule


def sweep_cap(suite: str) -> int | None:
    """有開關的研究目前的累計 sweep 呼叫上限（凍結時才填入的值可能是 None）；沒有開關 → ValueError。"""
    spec = study(suite)
    if not spec.gated:
        raise ValueError(f"{suite} has no registered sweep cap")
    return _flag(spec.sweep_cap_flag)


def paid_spec(suite: str) -> PaidSpec | None:
    return study(suite).paid


def paid_phase(split: str, *, sweep: bool) -> str:
    """付費總帳的 phase：dev baseline → "dev"、eval baseline → "baseline"、sweep（只限 eval）→ "sweep"。"""
    if split not in ("dev", "eval") or (sweep and split != "eval"):
        raise ValueError("a paid sweep uses the eval split; a paid baseline uses dev or eval")
    return "sweep" if sweep else "dev" if split == "dev" else "baseline"


def check_paid_run(spec: PaidSpec, *, model: Any, split: str, max_cost_usd: Any, sweep: bool, rollouts: Any = None) -> None:
    """付費推論的登記（decision 035 / 036 §7）：研究沒有結案；model 是 ``spec.model``；split 在 ``spec.splits``；這一次要求的
    上限 0 < x <= 該 phase 的上限（sweep 為總上限）；baseline 的 R = ``spec.rollouts``。累計的扣除（總帳、程序鎖、保險絲）由
    ``experiments.billing.paid_session`` 在 run_split / run_sweeps 內處理。違反 → ValueError。"""
    if spec.closed_flag is not None and _flag(spec.closed_flag):
        raise ValueError(f"paid {spec.name} runs are closed (decision {spec.decision} is finished; decision 036 §7)")
    if model != spec.model:
        raise ValueError(f"paid {spec.name} runs are registered only for {spec.model} (decision {spec.decision})")
    if split not in spec.splits:
        raise ValueError(f"paid {spec.name} runs use the {' / '.join(spec.splits)} split only (decision {spec.decision})")
    phase = paid_phase(split, sweep=sweep)
    cap = spec.phase_caps.get(phase, spec.total_usd)
    if isinstance(max_cost_usd, bool) or not isinstance(max_cost_usd, (int, float)) or not 0 < max_cost_usd <= cap:
        raise ValueError(f"a paid {spec.name} {phase} run needs 0 < --max-cost-usd <= {cap:g} (decision {spec.decision})")
    if not sweep and rollouts != spec.rollouts:
        raise ValueError(f"a paid {spec.name} baseline must use --rollouts {spec.rollouts} (decision {spec.decision})")


def segment_pool(suite: str, seeds: Sequence[int]) -> tuple[GeneratedInstance, ...]:
    """登記的 eval seed 區塊的 instance（eval pool 的順序）。不是登記的區塊 → ValueError。"""
    spec = study(suite)
    block = tuple(seeds) if isinstance(seeds, (list, tuple)) else None
    if spec.eval_segments is None or block not in spec.eval_segments:
        raise ValueError(f"seeds {seeds!r} are not a registered eval segment of {suite} "
                         f"({'none' if spec.eval_segments is None else list(spec.eval_segments)})")
    return tuple(i for i in suite_pool(suite, "eval") if i.task.params["seed"] in block)


def check_limits(suite: str, *, split: str, call_log: bool, max_model_calls: int | None, sweep: bool,
                 seeds: Sequence[int] | None = None) -> None:
    """有開關的研究的真正推論（本機）：必須記錄每次呼叫（calls.jsonl），呼叫上限必須是登記的值（baseline）或不超過每段上限
    （sweep）；有登記 eval 區塊的研究，eval baseline 必須逐段執行（seeds 是登記的區塊）。沒有開關的研究沒有登記的上限
    → ValueError。run_split / run_sweeps 在 client 不是 oracle 假 client 時呼叫（不只在 CLI；decision 034 驗證 B 第二輪）。"""
    spec = study(suite)
    if not spec.gated:
        raise ValueError(f"{suite} has no registered call limits")
    name = f"{suite} (decision {spec.decision} §8)"
    if not call_log:
        raise ValueError(f"{name} model runs must log every call (calls.jsonl; decision 032)")
    if not sweep and split == "eval" and spec.eval_segments is not None:
        segment_pool(suite, seeds if seeds is not None else ())  # 本機的 eval baseline 逐段執行（decision 036 §6）
    if sweep:
        if max_model_calls is None or max_model_calls > spec.segment_cap:
            raise ValueError(f"a {name} sweep segment may use at most {spec.segment_cap} model calls")
    elif max_model_calls != spec.baseline_caps[split]:
        raise ValueError(f"a {name} {split} baseline must use --max-model-calls {spec.baseline_caps[split]}")


@functools.lru_cache(maxsize=None)
def suite_pool(suite: str, split: str) -> tuple[GeneratedInstance, ...]:
    """suite 的 dev / eval pool（快取；instance 是 frozen，可以共用）。未知 suite 或 split → ValueError。"""
    spec = study(suite)
    if split not in ("dev", "eval"):
        raise ValueError("split must be 'dev' or 'eval'")
    return spec.pool(split)


def eval_task_ids(suite: str) -> list[str]:
    return [instance.task.task_id for instance in suite_pool(suite, "eval")]


def suite_for_generator(generator_version: str) -> str:
    if generator_version not in _GENERATORS:
        raise ValueError(f"no suite uses generator {generator_version!r}")
    return _GENERATORS[generator_version]


def suite_for_task_ids(task_ids: Sequence[str]) -> str:
    """task id 的集合剛好等於某個 suite 的 eval pool → 該 suite；否則 ValueError。"""
    ids = set(task_ids)
    for suite in SUITES:
        if ids == set(eval_task_ids(suite)):
            return suite
    raise ValueError("the task ids are not exactly the eval pool of any suite")


def suite_containing(task_ids: Sequence[str]) -> str:
    """eval pool 包含所有 task id 的 suite（不必是全部）；沒有任何 suite 包含全部 → ValueError。"""
    ids = set(task_ids)
    for suite in SUITES:
        if ids and ids <= set(eval_task_ids(suite)):
            return suite
    raise ValueError("no suite's eval pool contains all of these task ids")


def check_model_run(suite: str, split: str) -> None:
    """真正的模型推論之前呼叫（dry-run 不需要）。沒有登記的 suite、未經使用者確認或未凍結（eval）的研究 → ValueError。"""
    spec = study(suite)
    if not spec.gated:
        return
    if not _flag(spec.approved_flag):
        raise ValueError(f"{suite} model runs need the user's approval of decision {spec.decision} first "
                         f"(tasks.suites.{spec.approved_flag})")
    if split == "eval" and not _flag(spec.frozen_flag):
        raise ValueError(f"{suite} eval-seed runs need the frozen record of decision {spec.decision} first "
                         f"(tasks.suites.{spec.frozen_flag})")
