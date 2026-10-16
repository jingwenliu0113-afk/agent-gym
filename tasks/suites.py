"""題目集合（suite）：第一輪的 ``week1``（generator v1）與 Study 2 的 ``study2``（generator v2；decision 034）。

run 與分析工具由 suite 取得 pool；分析時由 manifest 的 generator_version 或 source 的 task id 推出 suite，所以第一輪的
run.json、manifest 與 summary 格式都不變。

Study 2 的模型推論有兩道開關（decision 034 §5 / §9）：
- ``STUDY2_APPROVED``：使用者確認研究方案之後才設為 True；在這之前不對任何 Study 2 的題目做模型推論（dev 也一樣）。
- ``STUDY2_FROZEN``：dev 驗證完成、凍結紀錄 commit 之後才設為 True；在這之前不對 eval seed 做推論。
dry-run（oracle 假 client）不受限制。
"""

from __future__ import annotations

import functools
from typing import Sequence

from tasks.generator import GENERATOR_VERSION as WEEK1_GENERATOR, GeneratedInstance, week1_pool
from tasks.generator_v2 import GENERATOR_VERSION as STUDY2_GENERATOR, study2_pool

SUITES: tuple[str, ...] = ("week1", "study2")
_GENERATORS = {WEEK1_GENERATOR: "week1", STUDY2_GENERATOR: "study2"}

STUDY2_APPROVED = False
STUDY2_FROZEN = False
STUDY2_SWEEP_CAP: int | None = None  # 正式 sweep 的累計呼叫上限（decision 034 §8；使用者選擇 source 規則後、凍結時填入）


@functools.lru_cache(maxsize=None)
def suite_pool(suite: str, split: str) -> tuple[GeneratedInstance, ...]:
    """suite 的 dev / eval pool（快取；instance 是 frozen，可以共用）。未知 suite 或 split → ValueError。"""
    if suite not in SUITES:
        raise ValueError(f"suite must be one of: {', '.join(SUITES)}")
    if split not in ("dev", "eval"):
        raise ValueError("split must be 'dev' or 'eval'")
    return week1_pool(split) if suite == "week1" else study2_pool(split)


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
    """真正的模型推論之前呼叫（dry-run 不需要）。Study 2 未經使用者確認 / 未凍結 → ValueError。"""
    if suite not in SUITES:
        raise ValueError(f"suite must be one of: {', '.join(SUITES)}")
    if suite != "study2":
        return
    if not STUDY2_APPROVED:
        raise ValueError("Study 2 model runs need the user's approval of decision 034 first (tasks.suites.STUDY2_APPROVED)")
    if split == "eval" and not STUDY2_FROZEN:
        raise ValueError("Study 2 eval-seed runs need the frozen record of decision 034 first (tasks.suites.STUDY2_FROZEN)")
