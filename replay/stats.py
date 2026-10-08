"""failure localization 的統計工具（decision log 010）。

- ``wilson_interval``：單一 checkpoint 的 rollout 在給定 checkpoint 下獨立同分佈 → Wilson score interval。
- ``stratified_task_bootstrap``：跨 task 的彙總以 task 為 cluster、family 分層重抽（decision 005），統計量為 task 等權平均。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval；n == 0 → (0.0, 1.0)。successes / n 必須是 int（bool 不算）且 0 <= successes <= n，否則 ValueError。
    結果夾在 [0, 1]；successes == 0 時下界恰為 0.0、successes == n 時上界恰為 1.0。z 必須是有限的正數。
    """
    if type(successes) is not int or type(n) is not int:
        raise ValueError("successes and n must be ints")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a finite positive number")
    z2 = float(z) * float(z)
    if not math.isfinite(z2):
        raise ValueError("z is too large: z * z overflows")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    inv_n = 1 / n  # int / int 的真除法：任何大小的 n 都不會溢位（極大的 n 下溢為 0.0）
    denominator = 1.0 + z2 * inv_n
    center = (p + z2 * inv_n / 2) / denominator
    half = z * math.sqrt(p * (1.0 - p) * inv_n + z2 * inv_n * inv_n / 4) / denominator
    low = 0.0 if successes == 0 else min(1.0, max(0.0, center - half))
    high = 1.0 if successes == n else min(1.0, max(0.0, center + half))
    return low, high


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float  # 原樣本的 task 等權平均
    low: float
    high: float
    replicates: int  # B
    n_tasks: int


def stratified_task_bootstrap(
    values: Mapping[str, float],
    strata: Mapping[str, str],
    *,
    replicates: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """family 分層的 task-level percentile bootstrap。

    - values：task_id → 該 task 的單一數值；strata：task_id → family；兩者 key 集合必須相同且非空，否則 ValueError。
    - 每個 replicate：對每個 stratum 以 ``random.Random(seed)`` 的 ``rng.random()`` 有放回抽出與原本相同數量的 task
      （strata 依名稱排序、task 依 id 排序後抽，結果可重現），統計量為抽出 task 的等權平均。
    - low / high = replicate 統計量排序後第 ``floor(alpha/2 * B)`` 與 ``ceil((1 - alpha/2) * B) - 1`` 個（0-based）。
    - replicates 為 int >= 1、0 < alpha < 1，否則 ValueError。
    - values 的值必須是有限的 int / float（bool 不算）；strata 的值必須是 str；seed 為 int（bool 不算）；否則 ValueError。
    - 抽取：stratum 內 m 個 task 時，index = ``min(int(rng.random() * m), m - 1)``，每次抽取恰好一次 ``rng.random()``。
    """
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be an int >= 1")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    if not isinstance(values, Mapping) or not isinstance(strata, Mapping):
        raise ValueError("values and strata must be mappings")
    if len(values) == 0 or set(values) != set(strata):
        raise ValueError("values and strata must have the same non-empty set of task ids")
    for task_id, value in values.items():
        if not isinstance(task_id, str):
            raise ValueError("task ids must be str")
        if not _is_finite_number(value):
            raise ValueError(f"value for task {task_id!r} must be a finite number")
    groups: dict[str, list[str]] = {}
    for task_id, stratum in strata.items():
        if not isinstance(stratum, str):
            raise ValueError(f"stratum for task {task_id!r} must be a str")
        groups.setdefault(stratum, []).append(task_id)
    ordered = [[float(values[task_id]) for task_id in sorted(groups[name])] for name in sorted(groups)]
    n_tasks = len(values)

    # 先除以 n_tasks 再加總：接近 float 上限的值也不會在加總時溢位
    estimate = math.fsum(value / n_tasks for group in ordered for value in group)
    rng = random.Random(seed)
    statistics: list[float] = []
    for _ in range(replicates):
        drawn: list[float] = []
        for group in ordered:
            m = len(group)
            for _ in range(m):
                drawn.append(group[min(int(rng.random() * m), m - 1)])
        statistics.append(math.fsum(value / n_tasks for value in drawn))
    statistics.sort()
    low_index = math.floor(alpha / 2 * replicates)
    high_index = math.ceil((1 - alpha / 2) * replicates) - 1
    low_index = min(max(low_index, 0), replicates - 1)
    high_index = min(max(high_index, 0), replicates - 1)
    return BootstrapResult(
        estimate=estimate,
        low=statistics[low_index],
        high=statistics[high_index],
        replicates=replicates,
        n_tasks=n_tasks,
    )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:  # 超出 float 範圍的 int
        return False
