"""replay/stats.py 的契約測試（decision log 010 §4 / §5；decision 005 統計協定；EXPERIMENT_RULES Statistics）。

契約來源：``replay/stats.py`` docstring、decision 010、decision 005。

``wilson_interval``
- 期望值一律由測試內的 Wilson score 閉式解計算（再夾到 [0, 1]）；另以 Newcombe (1998) 的公開數值交叉檢查閉式解本身。
- n == 0 → (0.0, 1.0)；successes / n 必須是 int（bool 不算）且 0 <= successes <= n，否則 ValueError；結果夾在 [0, 1]。
- z 必須是有限的正數：0、負數、nan、±inf、非數值 → ValueError（docstring 的 z 句沒有逐字寫「否則 ValueError」，
  本檔沿用同一 docstring 的 ValueError 慣例）。z = True **不**測：docstring 只對 successes / n 註明「bool 不算」。

``stratified_task_bootstrap``
- 抽取對應已由 docstring 完整規定：replicate 依序產生；每個 replicate 內 strata 依名稱排序、stratum 內 task 依 id 排序，
  每次抽取恰好呼叫一次 ``rng.random()``，index = ``min(int(rng.random() * m), m - 1)``；統計量為抽出 task 的等權平均；
  low / high 為排序後第 ``floor(alpha/2 · B)`` 與 ``ceil((1 − alpha/2) · B) − 1`` 個。
  → ``_reference_bootstrap`` 直接照 docstring 逐字實作（約 15 行），在多組 seed / B / alpha / 不等大小 strata（含單一 task 的
  stratum、字典序與數值序不同的 id、非排序的插入順序）下逐值比對 estimate / low / high（abs = 1e-12）。
  另以一個不呼叫被測函式的測試確認這個比對能分辨「依 stratum 迴圈（先抽完 stratum A 的所有 replicate）」與「strata 反序」。
- 與上述細節無關的性質仍保留（可在 reference 以外再定位錯誤）：
  * 分層：以「每個 task 的值是不同位數的 10 的次方」編碼，replicates = 1 時 low == high == 該唯一 replicate 的統計量，
    乘回 task 數即可解出每個 task 被抽到的次數 → 驗證每個 stratum 的抽出數量不變、有放回、近似均勻。
  * 分位數 index：同一 seed / B 下 replicate 集合與 alpha 無關，因此依 docstring 公式落在同一組 index 的 alpha 必須給出
    完全相同的 (low, high)，落在不同 index 的 alpha 必須嚴格不同（連續值資料，order statistic 幾乎必然相異）。
  * 隨機來源：只能用 ``random.Random(seed)`` 的 ``rng.random()``：其他抽樣方法被 monkeypatch 成丟例外；
    消耗的 random() 序列必須等於全新 ``random.Random(seed)`` 序列的前綴（單一 rng、依序消耗）；
    把 random() 固定成常數 c 時所有 replicate 相同，且每個 stratum 全部抽到 index ``min(int(c · m), m − 1)``（含 c = 1.0 的夾住）。
- 驗證錯誤（docstring 明文「否則 ValueError」）：values 的值非有限 int / float 或為 bool（含超出 float 範圍的 int，
  比照 decision 007 的「超出 ±float max 的 int 不算有限」）；strata 的值非 str；seed 非 int 或為 bool。
  replicates = True **不**測：docstring 只寫「replicates 為 int >= 1」，沒有像 values / seed 那樣註明「bool 不算」。
- 涵蓋率（decision 010 §5，門檻 ≥ 0.88）。**估計目標（estimand）已由 decision 013 §6 澄清（不改協定）**：
  decision 005 寫「正式統計的母體 = eval split 的 15 個 instances」（task 固定、只有 rollout 雜訊是隨機的），這是正式的 estimand；
  decision 010 則以「task 可交換」為由採單階段 cluster bootstrap（對應 family 內的 superpopulation）。兩種讀法下
  eval 設計（5 families × 3 題）的涵蓋率不同（superpopulation 約 0.84–0.87 < 0.88；固定 15 題約 0.90–0.995，依題組而定），
  因此本檔：
  * ``test_bootstrap_coverage_large_strata_implementation_check``：16 / 20 / 24 題、superpopulation 真值，是
    **實作正確性檢查**（percentile bootstrap 近似有效的區間），**不是** eval CI 的驗證。
  * ``test_bootstrap_coverage_at_eval_design_5x3_superpopulation``：在真正的 5 × 3 設計上斷言預先登記的門檻 ≥ 0.88，
    以 ``xfail(strict=True)`` 記錄 decision 013 §6 的已知限制（superpopulation 讀法下未達門檻）；m = 3 時 percentile bootstrap
    的層內變異只有不偏估計的 (m − 1)/m = 2/3，CI 偏窄。strict：若意外通過也會失敗。
- 本檔只用 stdlib + pytest、固定 seed；不在 import / collection 階段呼叫任何被測函式。
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest

from replay.stats import (
    Z_95,
    BootstrapResult,
    stratified_task_bootstrap,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wilson_closed_form(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval 閉式解（測試內獨立計算），夾到 [0, 1]。"""
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _continuous_data(
    sizes: dict[str, int], *, data_seed: int = 0
) -> tuple[dict[str, float], dict[str, str]]:
    """連續值資料（固定 seed 的均勻亂數）：不同的抽取組合幾乎不可能給出相同的 replicate 統計量。"""
    rng = random.Random(data_seed)
    values: dict[str, float] = {}
    strata: dict[str, str] = {}
    for family, size in sizes.items():
        for i in range(size):
            task_id = f"{family}-t{i:02d}"
            values[task_id] = rng.random()
            strata[task_id] = family
    return values, strata


# 編碼資料：task 值為 10 的不同次方；N = 6、每個 task 被抽到的次數 <= 3 < 10，總和可精確解碼。
_CODE_VALUES: dict[str, float] = {
    "a": 1.0,
    "b": 10.0,
    "c": 100.0,
    "d": 1_000.0,
    "e": 10_000.0,
    "f": 100_000.0,
}
_CODE_STRATA: dict[str, str] = {
    "a": "fam-1",
    "b": "fam-1",
    "c": "fam-1",
    "d": "fam-2",
    "e": "fam-2",
    "f": "fam-3",
}
_CODE_ORDER = ["a", "b", "c", "d", "e", "f"]
_CODE_STRATUM_TASKS = {"fam-1": ["a", "b", "c"], "fam-2": ["d", "e"], "fam-3": ["f"]}


def _decode_counts(statistic: float) -> dict[str, int]:
    """把單一 replicate 的等權平均解碼成每個 task 被抽到的次數。"""
    n_tasks = len(_CODE_VALUES)
    raw = statistic * n_tasks
    total = round(raw)
    assert abs(raw - total) < 1e-6, (
        f"replicate 統計量 {statistic!r} 不是被抽到 task 的等權平均"
    )
    counts: dict[str, int] = {}
    for task_id in _CODE_ORDER:
        counts[task_id] = total % 10
        total //= 10
    assert total == 0, f"replicate 統計量 {statistic!r} 解碼後有多餘的位數"
    return counts


def _assert_stratum_counts_preserved(counts: dict[str, int]) -> None:
    for family, tasks in _CODE_STRATUM_TASKS.items():
        drawn = sum(counts[t] for t in tasks)
        assert drawn == len(tasks), (
            f"stratum {family} 抽出 {drawn} 個 task，原本有 {len(tasks)} 個：{counts}"
        )


def _single_replicate(seed: int) -> float:
    """replicates = 1：low 與 high 都是第 0 個（唯一）replicate 的統計量。"""
    result = stratified_task_bootstrap(
        _CODE_VALUES, _CODE_STRATA, replicates=1, seed=seed
    )
    assert result.low == result.high
    return result.low


# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------


def test_z95_is_the_standard_normal_975_quantile() -> None:
    assert Z_95 == pytest.approx(NormalDist().inv_cdf(0.975), abs=1e-12)


WILSON_CASES = [
    (0, 1, Z_95),
    (1, 1, Z_95),
    (0, 10, Z_95),
    (1, 10, Z_95),
    (5, 10, Z_95),
    (7, 10, Z_95),
    (10, 10, Z_95),
    (3, 8, Z_95),
    (13, 24, Z_95),
    (24, 24, Z_95),
    (81, 263, Z_95),
    (1, 10, 1.0),
    (5, 10, 1.0),
    (5, 10, 2.5758293035489004),
    (0, 3, 3.0),
    (2, 3, 0.5),
    (12, 12, 1.0),
    (40, 1000, Z_95),
]


@pytest.mark.parametrize(("successes", "n", "z"), WILSON_CASES)
def test_wilson_matches_closed_form(successes: int, n: int, z: float) -> None:
    low, high = wilson_interval(successes, n, z)
    exp_low, exp_high = _wilson_closed_form(successes, n, z)
    assert low == pytest.approx(exp_low, abs=1e-12)
    assert high == pytest.approx(exp_high, abs=1e-12)


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        # Newcombe (1998), Statistics in Medicine 17:857-872, Table I, method 3（score，無連續性校正）。
        (81, 263, (0.2553, 0.3662)),
        (15, 148, (0.0624, 0.1605)),
        (0, 20, (0.0000, 0.1611)),
        (1, 29, (0.0061, 0.1718)),
    ],
)
def test_wilson_matches_published_reference_values(
    successes: int, n: int, expected: tuple[float, float]
) -> None:
    # 公開數值四捨五入到小數 4 位 → 容許 5e-5。
    low, high = wilson_interval(successes, n)
    assert low == pytest.approx(expected[0], abs=5.0001e-5)
    assert high == pytest.approx(expected[1], abs=5.0001e-5)
    # 也確認測試內的閉式解與公開數值一致（期望值來源本身沒錯）。
    ref_low, ref_high = _wilson_closed_form(successes, n, Z_95)
    assert ref_low == pytest.approx(expected[0], abs=5.0001e-5)
    assert ref_high == pytest.approx(expected[1], abs=5.0001e-5)


@pytest.mark.parametrize("z", [Z_95, 1.0, 2.5758293035489004])
def test_wilson_n_zero_is_the_whole_unit_interval(z: float) -> None:
    assert wilson_interval(0, 0, z) == (0.0, 1.0)


def test_wilson_n_zero_with_default_z() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_returns_a_tuple_of_two_floats() -> None:
    result = wilson_interval(3, 7)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(x, float) for x in result)


def test_wilson_default_z_is_z95() -> None:
    for successes, n in [(0, 5), (2, 5), (5, 5), (17, 40)]:
        assert wilson_interval(successes, n) == wilson_interval(successes, n, Z_95)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 24, 100])
@pytest.mark.parametrize("z", [Z_95, 1.0, 3.0])
def test_wilson_zero_successes(n: int, z: float) -> None:
    low, high = wilson_interval(0, n, z)
    assert low >= 0.0
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(_wilson_closed_form(0, n, z)[1], abs=1e-12)
    assert 0.0 < high <= 1.0


@pytest.mark.parametrize("n", [1, 2, 5, 10, 24, 100])
@pytest.mark.parametrize("z", [Z_95, 1.0, 3.0])
def test_wilson_all_successes(n: int, z: float) -> None:
    low, high = wilson_interval(n, n, z)
    assert high <= 1.0
    assert high == pytest.approx(1.0, abs=1e-12)
    assert low == pytest.approx(_wilson_closed_form(n, n, z)[0], abs=1e-12)
    assert 0.0 <= low < 1.0


@pytest.mark.parametrize("n", [1, 3, 10, 27])
def test_wilson_is_symmetric_in_successes_and_failures(n: int) -> None:
    for successes in range(n + 1):
        low, high = wilson_interval(successes, n)
        mirror_low, mirror_high = wilson_interval(n - successes, n)
        assert low == pytest.approx(1.0 - mirror_high, abs=1e-12)
        assert high == pytest.approx(1.0 - mirror_low, abs=1e-12)


@pytest.mark.parametrize("z", [0.5, 1.0, Z_95, 2.5758293035489004, 5.0, 50.0])
def test_wilson_is_clamped_to_unit_interval_and_contains_point_estimate(
    z: float,
) -> None:
    # 未夾住的閉式解在浮點下會略為超出 [0, 1]（例如 z = 1、12/12 的上界 1.0000000000000002）；結果必須夾在 [0, 1]。
    for n in range(1, 61):
        for successes in range(n + 1):
            low, high = wilson_interval(successes, n, z)
            assert 0.0 <= low <= high <= 1.0, (successes, n, z, low, high)
            p_hat = successes / n
            assert low <= p_hat + 1e-12 and p_hat - 1e-12 <= high, (
                successes,
                n,
                z,
                low,
                high,
            )


def test_wilson_clamps_known_floating_point_overshoot() -> None:
    # 測試內未夾住的閉式解：12/12、z = 1 的上界 > 1。
    p, n, z = 1.0, 12, 1.0
    unclamped_high = (p + z * z / (2 * n)) / (1 + z * z / n) + (
        z / (1 + z * z / n)
    ) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    assert unclamped_high > 1.0  # 前提：此案例確實會溢出
    low, high = wilson_interval(12, 12, 1.0)
    assert high <= 1.0
    assert high == pytest.approx(1.0, abs=1e-12)
    assert low == pytest.approx(_wilson_closed_form(12, 12, 1.0)[0], abs=1e-12)


@pytest.mark.parametrize(
    ("successes", "n"), [(0, 4), (1, 4), (3, 10), (9, 10), (50, 100)]
)
def test_wilson_larger_z_gives_nested_wider_interval(successes: int, n: int) -> None:
    narrow = wilson_interval(successes, n, 1.0)
    default = wilson_interval(successes, n)
    wide = wilson_interval(successes, n, 2.5758293035489004)
    # 0 / n 的下界三者理論上都是 0（浮點誤差內）。
    assert wide[0] <= default[0] + 1e-12 and default[0] <= narrow[0] + 1e-12
    assert narrow[1] <= default[1] + 1e-12 and default[1] <= wide[1] + 1e-12
    assert (default[1] - default[0]) > (narrow[1] - narrow[0])
    assert (wide[1] - wide[0]) > (default[1] - default[0])


@pytest.mark.parametrize(
    ("successes", "n"),
    [(True, 1), (False, 1), (1, True), (0, False), (False, False), (True, True)],
)
def test_wilson_rejects_bool(successes: object, n: object) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("successes", "n"),
    [
        (1.0, 2),
        (1, 2.0),
        (0.0, 0),
        (0, 0.0),
        (0.5, 1),
        (2.0, 4.0),
        ("1", 2),
        (1, "2"),
        (None, 2),
        (1, None),
    ],
)
def test_wilson_rejects_non_int(successes: object, n: object) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("successes", "n"), [(-1, 5), (0, -1), (-3, -1), (-1, 0), (-1, -1)]
)
def test_wilson_rejects_negative(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


@pytest.mark.parametrize(("successes", "n"), [(6, 5), (1, 0), (11, 10), (2, 1)])
def test_wilson_rejects_successes_greater_than_n(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


_BAD_Z = [0, 0.0, -1.0, -Z_95, float("nan"), float("inf"), float("-inf"), "1.96", None]


@pytest.mark.parametrize("z", _BAD_Z, ids=repr)
@pytest.mark.parametrize(("successes", "n"), [(3, 10), (0, 5), (5, 5)])
def test_wilson_rejects_non_finite_or_non_positive_z(
    successes: int, n: int, z: object
) -> None:
    # 「z 必須是有限的正數」。n == 0 不在此列：docstring 沒有規定 n == 0 的特例與 z 驗證的先後。
    with pytest.raises(ValueError):
        wilson_interval(successes, n, z)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：驗證
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "strata"),
    [
        ({"a": 0.1, "b": 0.2}, {"a": "f"}),  # values 多一個 key
        ({"a": 0.1}, {"a": "f", "b": "g"}),  # strata 多一個 key
        ({"a": 0.1, "b": 0.2}, {"a": "f", "c": "f"}),  # 大小相同但 key 不同
        ({"a": 0.1}, {}),
        ({}, {"a": "f"}),
    ],
)
def test_bootstrap_rejects_key_mismatch(
    values: dict[str, float], strata: dict[str, str]
) -> None:
    with pytest.raises(ValueError):
        stratified_task_bootstrap(values, strata, replicates=10)


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        stratified_task_bootstrap({}, {}, replicates=10)


@pytest.mark.parametrize("replicates", [0, -1, -100, 2.0, 1.5])
def test_bootstrap_rejects_bad_replicates(replicates: object) -> None:
    values, strata = _continuous_data({"fam-x": 3, "fam-y": 2})
    with pytest.raises(ValueError):
        stratified_task_bootstrap(values, strata, replicates=replicates)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, float("nan")])
def test_bootstrap_rejects_bad_alpha(alpha: float) -> None:
    values, strata = _continuous_data({"fam-x": 3, "fam-y": 2})
    with pytest.raises(ValueError):
        stratified_task_bootstrap(values, strata, replicates=10, alpha=alpha)


_BAD_VALUES = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "0.5",
    None,
    10**400,  # 超出 float 範圍的 int：不是有限的數值（比照 decision 007）
    -(10**400),
]


@pytest.mark.parametrize(
    "bad", _BAD_VALUES, ids=lambda v: type(v).__name__ + ":" + repr(v)[:12]
)
def test_bootstrap_rejects_non_finite_or_non_numeric_values(bad: object) -> None:
    # 「values 的值必須是有限的 int / float（bool 不算）……否則 ValueError」：只有一個 task 的值不合法也要拒絕。
    values, strata = _continuous_data({"fam-x": 3, "fam-y": 2})
    values = dict(values)
    values[sorted(values)[-1]] = bad  # type: ignore[assignment]
    with pytest.raises(ValueError):
        stratified_task_bootstrap(values, strata, replicates=10)


def test_bootstrap_accepts_int_values() -> None:
    # 「有限的 int / float」：int 值合法，點估計照常為等權平均。
    values = {"x1": 0, "x2": 1, "x3": 1, "y1": 0.25}
    strata = {"x1": "fam-x", "x2": "fam-x", "x3": "fam-x", "y1": "fam-y"}
    result = stratified_task_bootstrap(values, strata, replicates=20, seed=0)
    assert result.estimate == pytest.approx(2.25 / 4, abs=1e-12)
    assert result.n_tasks == 4


@pytest.mark.parametrize("bad", [1, None, b"fam-x", ("fam-x",)], ids=repr)
def test_bootstrap_rejects_non_str_strata(bad: object) -> None:
    # 「strata 的值必須是 str」。
    values, strata = _continuous_data({"fam-x": 3, "fam-y": 2})
    strata = dict(strata)
    strata[sorted(strata)[0]] = bad  # type: ignore[assignment]
    with pytest.raises(ValueError):
        stratified_task_bootstrap(values, strata, replicates=10)


@pytest.mark.parametrize("bad", [True, False, 1.0, 0.0, "0", None], ids=repr)
def test_bootstrap_rejects_non_int_seed(bad: object) -> None:
    # 「seed 為 int（bool 不算）」。
    values, strata = _continuous_data({"fam-x": 3, "fam-y": 2})
    with pytest.raises(ValueError):
        stratified_task_bootstrap(values, strata, replicates=10, seed=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：結果欄位與點估計
# ---------------------------------------------------------------------------


def test_bootstrap_result_fields() -> None:
    values, strata = _continuous_data({"fam-x": 4, "fam-y": 3, "fam-z": 2})
    result = stratified_task_bootstrap(values, strata, replicates=200, seed=3)
    assert isinstance(result, BootstrapResult)
    assert result.replicates == 200
    assert result.n_tasks == 9
    assert result.estimate == pytest.approx(
        sum(values.values()) / len(values), abs=1e-12
    )
    assert result.low <= result.high


def test_bootstrap_estimate_is_task_equal_weight_not_stratum_weighted() -> None:
    # 等權平均 = 1 / 4 = 0.25；各 family 平均的平均 = 0.5。
    values = {"x1": 0.0, "x2": 0.0, "x3": 0.0, "y1": 1.0}
    strata = {"x1": "fam-x", "x2": "fam-x", "x3": "fam-x", "y1": "fam-y"}
    result = stratified_task_bootstrap(values, strata, replicates=50, seed=1)
    assert result.estimate == pytest.approx(0.25, abs=1e-12)
    assert result.n_tasks == 4


def test_bootstrap_defaults() -> None:
    values, strata = _continuous_data({"fam-x": 3, "fam-y": 3})
    default = stratified_task_bootstrap(values, strata)
    assert default.replicates == 10000
    explicit = stratified_task_bootstrap(
        values, strata, replicates=10000, seed=0, alpha=0.05
    )
    assert default == explicit


def test_bootstrap_accepts_read_only_mappings() -> None:
    from types import MappingProxyType

    values, strata = _continuous_data({"fam-x": 3, "fam-y": 4})
    plain = stratified_task_bootstrap(values, strata, replicates=300, seed=5)
    proxied = stratified_task_bootstrap(
        MappingProxyType(values), MappingProxyType(strata), replicates=300, seed=5
    )
    assert proxied == plain


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：可重現性與隨機來源
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    values, strata = _continuous_data({"fam-x": 5, "fam-y": 4, "fam-z": 3})
    first = stratified_task_bootstrap(values, strata, replicates=500, seed=17)
    second = stratified_task_bootstrap(values, strata, replicates=500, seed=17)
    assert first == second


def test_bootstrap_different_seeds_give_different_replicates() -> None:
    values, strata = _continuous_data({"fam-x": 5, "fam-y": 4, "fam-z": 3})
    results = {
        (r.low, r.high)
        for r in (
            stratified_task_bootstrap(values, strata, replicates=50, seed=s)
            for s in range(6)
        )
    }
    assert len(results) > 1


def test_bootstrap_does_not_depend_on_or_consume_global_random_state() -> None:
    values, strata = _continuous_data({"fam-x": 5, "fam-y": 4})
    random.seed(111)
    before = random.getstate()
    first = stratified_task_bootstrap(values, strata, replicates=300, seed=9)
    assert random.getstate() == before
    random.seed(999)
    second = stratified_task_bootstrap(values, strata, replicates=300, seed=9)
    assert first == second


def test_bootstrap_is_invariant_to_mapping_insertion_order() -> None:
    values, strata = _continuous_data({"fam-b": 5, "fam-a": 4, "fam-c": 3})
    keys = list(values)
    shuffled = keys[:]
    random.Random(2024).shuffle(shuffled)
    reversed_keys = keys[::-1]
    base = stratified_task_bootstrap(values, strata, replicates=400, seed=4)
    for order in (shuffled, reversed_keys):
        values_2 = {k: values[k] for k in order}
        strata_2 = {k: strata[k] for k in reversed(order)}
        assert (
            stratified_task_bootstrap(values_2, strata_2, replicates=400, seed=4)
            == base
        )


def test_bootstrap_depends_on_names_only_through_sorted_order() -> None:
    # 「strata 依名稱排序、task 依 id 排序後抽」：保持排序順序的改名不改變結果。
    values, strata = _continuous_data({"fam-a": 5, "fam-b": 4, "fam-c": 3})
    base = stratified_task_bootstrap(values, strata, replicates=400, seed=8)

    def rename_task(task_id: str) -> str:
        family, index = task_id.rsplit("-t", 1)
        return f"T{family[-1]}{int(index) * 7 + 3:03d}"

    def rename_family(family: str) -> str:
        return {"fam-a": "gen/alpha", "fam-b": "gen/beta", "fam-c": "gen/gamma"}[family]

    renamed_values = {rename_task(k): v for k, v in values.items()}
    renamed_strata = {rename_task(k): rename_family(f) for k, f in strata.items()}
    # 前提：改名保持 family 順序與 family 內 task 順序。
    for family in ("fam-a", "fam-b", "fam-c"):
        old_ids = sorted(k for k in values if strata[k] == family)
        new_ids = sorted(rename_task(k) for k in old_ids)
        assert new_ids == [rename_task(k) for k in old_ids]
    renamed = stratified_task_bootstrap(
        renamed_values, renamed_strata, replicates=400, seed=8
    )
    assert renamed == base


def test_bootstrap_draws_by_task_id_position() -> None:
    # 在 family 內重新指派值與 id 的對應（多重集合不變）：抽取依 id 排序後的位置，replicate 一般會改變。
    values, strata = _continuous_data({"fam-a": 6, "fam-b": 5})
    base = stratified_task_bootstrap(values, strata, replicates=40, alpha=0.5, seed=12)
    changed = 0
    for perm_seed in range(5):
        permuted = dict(values)
        for family in ("fam-a", "fam-b"):
            ids = sorted(k for k in values if strata[k] == family)
            shuffled_vals = [values[k] for k in ids]
            random.Random(perm_seed).shuffle(shuffled_vals)
            permuted.update(zip(ids, shuffled_vals))
        result = stratified_task_bootstrap(
            permuted, strata, replicates=40, alpha=0.5, seed=12
        )
        assert result.estimate == pytest.approx(base.estimate, abs=1e-12)
        changed += (result.low, result.high) != (base.low, base.high)
    assert changed >= 1


_FORBIDDEN_RANDOM_METHODS = (
    "getrandbits",
    "choice",
    "choices",
    "sample",
    "shuffle",
    "randrange",
    "randint",
    "_randbelow",
    "uniform",
    "randbytes",
    "binomialvariate",
)


def test_bootstrap_uses_only_rng_random(monkeypatch: pytest.MonkeyPatch) -> None:
    values, strata = _continuous_data({"fam-x": 5, "fam-y": 4, "fam-z": 1})
    expected = stratified_task_bootstrap(values, strata, replicates=300, seed=21)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stratified_task_bootstrap 只能使用 rng.random()")

    for name in _FORBIDDEN_RANDOM_METHODS:
        if hasattr(random.Random, name):
            monkeypatch.setattr(random.Random, name, forbidden)

    original_random = random.Random.random
    calls = 0

    def counting_random(self: random.Random) -> float:
        nonlocal calls
        calls += 1
        return original_random(self)

    monkeypatch.setattr(random.Random, "random", counting_random)
    result = stratified_task_bootstrap(values, strata, replicates=300, seed=21)
    assert calls > 0
    assert result == expected


@pytest.mark.parametrize("seed", [0, 7, 4242])
def test_bootstrap_consumes_the_random_stream_of_random_Random_seed(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    # 單一 random.Random(seed)、依序呼叫 rng.random()：消耗的序列必須是全新 Random(seed) 序列的前綴。
    values, strata = _continuous_data({"fam-x": 4, "fam-y": 3})
    original_random = random.Random.random
    consumed: list[float] = []

    def recording_random(self: random.Random) -> float:
        value = original_random(self)
        consumed.append(value)
        return value

    monkeypatch.setattr(random.Random, "random", recording_random)
    stratified_task_bootstrap(values, strata, replicates=25, seed=seed)
    monkeypatch.undo()

    assert consumed
    reference = random.Random(seed)
    assert consumed == [reference.random() for _ in consumed]


@pytest.mark.parametrize("constant", [0.0, 0.3, 0.5, 0.9, 0.9999999999999999, 1.0])
def test_bootstrap_randomness_flows_only_through_rng_random(
    monkeypatch: pytest.MonkeyPatch, constant: float
) -> None:
    # rng.random() 固定為常數 c → 每個 replicate 的抽取完全相同 → CI 退化；
    # 每個 stratum（task 依 id 排序）的 m 次抽取全落在 index min(int(c · m), m − 1)：c = 1.0 檢查夾住。
    monkeypatch.setattr(random.Random, "random", lambda self: constant)
    result = stratified_task_bootstrap(
        _CODE_VALUES, _CODE_STRATA, replicates=50, seed=3
    )
    assert result.low == result.high
    counts = _decode_counts(result.low)
    _assert_stratum_counts_preserved(counts)
    expected = dict.fromkeys(_CODE_ORDER, 0)
    for tasks in _CODE_STRATUM_TASKS.values():
        ordered = sorted(tasks)
        m = len(ordered)
        expected[ordered[min(int(constant * m), m - 1)]] = m
    assert counts == expected


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：逐值比對 docstring 規定的抽取（reference implementation）
# ---------------------------------------------------------------------------


def _reference_bootstrap(
    values: dict[str, float],
    strata: dict[str, str],
    *,
    replicates: int,
    seed: int,
    alpha: float,
    order: str = "docstring",
) -> tuple[float, float, float]:
    """照 ``stratified_task_bootstrap`` docstring 逐字實作；回傳 (estimate, low, high)。

    order = "docstring"：replicate 依序產生，每個 replicate 內 strata 依名稱排序、task 依 id 排序。
    order = "stratum-major" / "strata-reversed"：刻意錯誤的變體，只用來確認比對測試有鑑別力。
    """
    groups: dict[str, list[str]] = {}
    for task_id, stratum in strata.items():
        groups.setdefault(stratum, []).append(task_id)
    names = sorted(groups, reverse=(order == "strata-reversed"))
    ordered = [[values[t] for t in sorted(groups[name])] for name in names]
    n_tasks = len(values)
    rng = random.Random(seed)

    def draw(group: list[float]) -> list[float]:
        m = len(group)
        return [group[min(int(rng.random() * m), m - 1)] for _ in range(m)]

    if order == "stratum-major":
        per_stratum = [[draw(group) for _ in range(replicates)] for group in ordered]
        drawn = [
            [x for s in range(len(ordered)) for x in per_stratum[s][r]]
            for r in range(replicates)
        ]
    else:
        drawn = [
            [x for group in ordered for x in draw(group)] for _ in range(replicates)
        ]
    statistics = sorted(math.fsum(d) / n_tasks for d in drawn)
    low = statistics[math.floor(alpha / 2 * replicates)]
    high = statistics[math.ceil((1 - alpha / 2) * replicates) - 1]
    return math.fsum(values.values()) / n_tasks, low, high


def _exact_dataset(name: str) -> tuple[dict[str, float], dict[str, str]]:
    rng = random.Random(f"exact/{name}")
    if name == "uneven":
        # 不等大小（4 / 1 / 3 / 2）、含單一 task 的 stratum；插入順序不是排序順序；id 的字典序 ≠ 數值序（"z-10" < "z-2"）。
        layout = {
            "gen/zeta": ["z-10", "z-2", "z-9", "z-1"],
            "gen/alpha": ["a-only"],
            "gen/mid": ["m-3", "m-12", "m-1"],
            "gen/beta": ["b-20", "b-3"],
        }
        values = {t: rng.random() for tasks in layout.values() for t in tasks}
        strata = {t: family for family, tasks in layout.items() for t in tasks}
        return values, strata
    # "interleaved"：各 stratum 的 task 交錯插入；值混有 int 與 float。
    sizes = {"fam-c": 5, "fam-a": 3, "fam-b": 1, "fam-d": 6}
    values = {}
    strata = {}
    for i in range(max(sizes.values())):
        for family, size in sizes.items():
            if i < size:
                task_id = f"{family[-1]}{(i * 7) % 11}"
                kind = int(rng.random() * 3)
                values[task_id] = 0 if kind == 0 else 1 if kind == 1 else rng.random()
                strata[task_id] = family
    return values, strata


@pytest.mark.parametrize("dataset", ["uneven", "interleaved"])
@pytest.mark.parametrize("seed", [0, 7, 4242])
@pytest.mark.parametrize("replicates", [1, 8, 1000])
@pytest.mark.parametrize("alpha", [0.05, 0.3])
def test_bootstrap_matches_docstring_reference_exactly(
    dataset: str, seed: int, replicates: int, alpha: float
) -> None:
    values, strata = _exact_dataset(dataset)
    exp_estimate, exp_low, exp_high = _reference_bootstrap(
        values, strata, replicates=replicates, seed=seed, alpha=alpha
    )
    result = stratified_task_bootstrap(
        values, strata, replicates=replicates, seed=seed, alpha=alpha
    )
    assert result.estimate == pytest.approx(exp_estimate, abs=1e-12)
    assert result.low == pytest.approx(exp_low, abs=1e-12)
    assert result.high == pytest.approx(exp_high, abs=1e-12)
    assert result.replicates == replicates
    assert result.n_tasks == len(values)


def test_reference_comparison_distinguishes_wrong_draw_orders() -> None:
    # 不呼叫被測函式：確認上面的逐值比對能抓到「依 stratum 迴圈」與「strata 反序」這兩種只靠性質測試抓不到的錯誤。
    for wrong in ("stratum-major", "strata-reversed"):
        differs = False
        for dataset in ("uneven", "interleaved"):
            values, strata = _exact_dataset(dataset)
            for seed in (0, 7, 4242):
                for replicates in (8, 1000):
                    for alpha in (0.05, 0.3):
                        right = _reference_bootstrap(
                            values,
                            strata,
                            replicates=replicates,
                            seed=seed,
                            alpha=alpha,
                        )
                        other = _reference_bootstrap(
                            values,
                            strata,
                            replicates=replicates,
                            seed=seed,
                            alpha=alpha,
                            order=wrong,
                        )
                        differs |= (
                            abs(right[1] - other[1]) > 1e-9
                            or abs(right[2] - other[2]) > 1e-9
                        )
        assert differs, wrong


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：分層重抽
# ---------------------------------------------------------------------------


def test_bootstrap_each_replicate_preserves_per_stratum_counts() -> None:
    for seed in range(300):
        counts = _decode_counts(_single_replicate(seed))
        _assert_stratum_counts_preserved(counts)


def test_bootstrap_resamples_with_replacement_and_uniformly_within_strata() -> None:
    n_seeds = 300
    patterns: list[tuple[int, ...]] = []
    totals = dict.fromkeys(_CODE_ORDER, 0)
    for seed in range(n_seeds):
        counts = _decode_counts(_single_replicate(seed))
        patterns.append(tuple(counts[t] for t in _CODE_ORDER))
        for task_id, c in counts.items():
            totals[task_id] += c
    # 有放回：會出現同一 task 被抽到兩次以上；重抽確實會變：不只一種抽取組合。
    assert any(max(p) >= 2 for p in patterns)
    assert len(set(patterns)) >= 5
    # 每個 task 都抽得到；層內均勻 → 每個 task 的期望抽中次數 = 1。
    for task_id, total in totals.items():
        assert total > 0, task_id
        assert abs(total / n_seeds - 1.0) < 0.3, (task_id, total / n_seeds)
    # 單一 task 的 stratum 每次都恰好抽到它一次。
    assert all(p[_CODE_ORDER.index("f")] == 1 for p in patterns)


def test_bootstrap_constant_within_strata_gives_degenerate_ci() -> None:
    # 每個 family 內的值都相同：分層重抽每個 replicate 都等於原等權平均（未分層則會隨抽到的 family 組成變動）。
    # 等權平均 = (3 × 0 + 1 × 1) / 4 = 0.25（若 replicate 用 family 平均的平均則會是 0.5）。
    values = {"x1": 0.0, "x2": 0.0, "x3": 0.0, "y1": 1.0}
    strata = {"x1": "fam-x", "x2": "fam-x", "x3": "fam-x", "y1": "fam-y"}
    result = stratified_task_bootstrap(values, strata, replicates=2000, seed=0)
    assert result.estimate == pytest.approx(0.25, abs=1e-12)
    assert result.low == pytest.approx(0.25, abs=1e-12)
    assert result.high == pytest.approx(0.25, abs=1e-12)

    values = {"p1": 0.2, "p2": 0.2, "q1": 0.9, "q2": 0.9, "q3": 0.9, "r1": 0.5}
    strata = {
        "p1": "gen/p",
        "p2": "gen/p",
        "q1": "gen/q",
        "q2": "gen/q",
        "q3": "gen/q",
        "r1": "gen/r",
    }
    expected = (0.2 * 2 + 0.9 * 3 + 0.5) / 6
    result = stratified_task_bootstrap(values, strata, replicates=2000, seed=5)
    assert result.estimate == pytest.approx(expected, abs=1e-12)
    assert result.low == pytest.approx(expected, abs=1e-12)
    assert result.high == pytest.approx(expected, abs=1e-12)


def test_bootstrap_singleton_strata_give_degenerate_ci() -> None:
    values = {"t1": 0.0, "t2": 1.0, "t3": 0.5, "t4": 0.125}
    strata = {"t1": "f1", "t2": "f2", "t3": "f3", "t4": "f4"}
    result = stratified_task_bootstrap(values, strata, replicates=1000, seed=2)
    assert result.estimate == pytest.approx(0.40625, abs=1e-12)
    assert result.low == pytest.approx(0.40625, abs=1e-12)
    assert result.high == pytest.approx(0.40625, abs=1e-12)


def test_stratified_ci_is_narrower_than_pooled_when_families_differ() -> None:
    # 同一組值：依 family 分層 vs 全部放進同一層。family 平均差很大時，未分層的 CI 主要反映「抽到哪些 family」。
    noise = random.Random(77)
    values: dict[str, float] = {}
    by_family: dict[str, str] = {}
    for family, center in (("gen/low", 0.1), ("gen/high", 0.9)):
        for i in range(6):
            task_id = f"{family}-{i}"
            values[task_id] = center + noise.uniform(-0.05, 0.05)
            by_family[task_id] = family
    pooled = {task_id: "all" for task_id in values}
    stratified = stratified_task_bootstrap(values, by_family, replicates=2000, seed=1)
    unstratified = stratified_task_bootstrap(values, pooled, replicates=2000, seed=1)
    assert stratified.estimate == pytest.approx(unstratified.estimate, abs=1e-12)
    assert (stratified.high - stratified.low) < 0.5 * (
        unstratified.high - unstratified.low
    )


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：percentile 分位數
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.9, 0.99])
def test_bootstrap_single_replicate_has_low_equal_high(alpha: float) -> None:
    values, strata = _continuous_data({"fam-x": 4, "fam-y": 3})
    result = stratified_task_bootstrap(
        values, strata, replicates=1, seed=6, alpha=alpha
    )
    assert result.low == result.high
    assert result.replicates == 1


def test_bootstrap_two_replicates_low_is_min_high_is_max_for_every_alpha() -> None:
    # B = 2：任何 0 < alpha < 1 都是 index floor(alpha) = 0 與 ceil(2 − alpha) − 1 = 1 → (min, max)。
    values, strata = _continuous_data({"fam-x": 5, "fam-y": 4})
    for seed in range(10):
        results = [
            stratified_task_bootstrap(
                values, strata, replicates=2, seed=seed, alpha=alpha
            )
            for alpha in (0.01, 0.05, 0.5, 0.99)
        ]
        first = results[0]
        assert first.low < first.high
        for r in results[1:]:
            assert (r.low, r.high) == (first.low, first.high)


def test_bootstrap_quantile_indices_small_b() -> None:
    # B = 8：floor(alpha / 2 · B) 與 ceil((1 − alpha / 2) · B) − 1
    #   alpha = 0.05, 0.2         → (0, 7)
    #   alpha = 0.25, 0.3, 0.45   → (1, 6)（0.25：1.0 → 1、7.0 → 6）
    #   alpha = 0.5, 0.55         → (2, 5)（0.5：2.0 → 2、6.0 → 5）
    values, strata = _continuous_data({"fam-a": 7, "fam-b": 5, "fam-c": 4})

    def ci(alpha: float) -> tuple[float, float]:
        r = stratified_task_bootstrap(
            values, strata, replicates=8, seed=31, alpha=alpha
        )
        return r.low, r.high

    outer = ci(0.05)
    assert ci(0.2) == outer
    middle = ci(0.25)
    assert ci(0.3) == middle
    assert ci(0.45) == middle
    inner = ci(0.5)
    assert ci(0.55) == inner
    assert outer[0] < middle[0] < inner[0] <= inner[1] < middle[1] < outer[1]


def test_bootstrap_quantile_indices_default_alpha() -> None:
    # B = 1000：alpha = 0.05 → (25, 974)；0.0502 → (floor 25.1, ceil 974.9 − 1) = (25, 974)；
    # 0.0498 → (floor 24.9, ceil 975.1 − 1) = (24, 975)。
    values, strata = _continuous_data({"fam-a": 10, "fam-b": 10, "fam-c": 10})

    def ci(alpha: float) -> tuple[float, float]:
        r = stratified_task_bootstrap(
            values, strata, replicates=1000, seed=13, alpha=alpha
        )
        return r.low, r.high

    nominal = ci(0.05)
    assert ci(0.0502) == nominal
    wider = ci(0.0498)
    assert wider[0] < nominal[0]
    assert nominal[1] < wider[1]


def test_bootstrap_intervals_are_nested_in_alpha() -> None:
    values, strata = _continuous_data({"fam-a": 6, "fam-b": 6, "fam-c": 5})
    previous: tuple[float, float] | None = None
    for alpha in (0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95):
        r = stratified_task_bootstrap(
            values, strata, replicates=400, seed=2, alpha=alpha
        )
        assert r.low <= r.high
        if previous is not None:
            assert previous[0] <= r.low and r.high <= previous[1], alpha
        previous = (r.low, r.high)


def test_bootstrap_ci_brackets_estimate_on_symmetric_data() -> None:
    values, strata = _continuous_data({"fam-a": 8, "fam-b": 8, "fam-c": 8})
    for seed in range(5):
        r = stratified_task_bootstrap(values, strata, replicates=2000, seed=seed)
        assert r.low <= r.estimate <= r.high
        assert r.low < r.high


# ---------------------------------------------------------------------------
# stratified_task_bootstrap：涵蓋率模擬（decision 010 §5）
# ---------------------------------------------------------------------------

# family → (task 數, Beta a, Beta b)；task 的 P(success) ~ Beta(a, b)，task 值 = ROLLOUTS 次 rollout 的成功比例。
COVERAGE_FAMILIES = {
    "gen/easy": (16, 6.0, 2.0),
    "gen/mid": (20, 2.0, 2.0),
    "gen/hard": (24, 2.0, 5.0),
}
COVERAGE_ROLLOUTS = 10
COVERAGE_EXPERIMENTS = 300
COVERAGE_REPLICATES = 1000


def test_bootstrap_coverage_large_strata_implementation_check() -> None:
    """實作正確性檢查，**不是** eval CI 的驗證。

    每個 family 16 / 20 / 24 題、真值 = superpopulation 的 task 等權平均（每次實驗重抽 task 的 P(success)）：
    這是 percentile bootstrap 近似有效的區間，正確的實作應接近名目 0.95；用來抓分位數、分層、等權平均的實作錯誤。
    eval 設計（5 families × 3 題）見 ``test_bootstrap_coverage_at_eval_design_5x3_superpopulation``。
    """
    n_total = sum(n for n, _, _ in COVERAGE_FAMILIES.values())
    truth = sum(n * a / (a + b) for n, a, b in COVERAGE_FAMILIES.values()) / n_total
    data_rng = random.Random(20261008)
    covered = 0
    for experiment in range(COVERAGE_EXPERIMENTS):
        values: dict[str, float] = {}
        strata: dict[str, str] = {}
        for family, (n_tasks, a, b) in COVERAGE_FAMILIES.items():
            for i in range(n_tasks):
                p_task = data_rng.betavariate(a, b)
                successes = sum(
                    data_rng.random() < p_task for _ in range(COVERAGE_ROLLOUTS)
                )
                task_id = f"{family}-s{i:02d}"
                values[task_id] = successes / COVERAGE_ROLLOUTS
                strata[task_id] = family
        result = stratified_task_bootstrap(
            values, strata, replicates=COVERAGE_REPLICATES, seed=experiment
        )
        assert result.n_tasks == n_total
        covered += result.low <= truth <= result.high
    coverage = covered / COVERAGE_EXPERIMENTS
    # 名目 0.95；門檻 ≥ 0.88（decision 010 §5）。上限 0.99 抓「CI 過寬」（例如取 min / max）的錯誤。
    assert 0.88 <= coverage <= 0.99, f"coverage = {coverage:.3f}（truth = {truth:.4f}）"


# decision 005 的 eval 設計：5 個 family × 3 題（seeds 1–3）。task 的 P(success) ~ Uniform(lo, hi)（family 專屬），
# 只用 rng.random() 產生（跨 Python 版本穩定）；真值 = superpopulation 的 task 等權平均 = Σ 3 · (lo + hi)/2 / 15。
EVAL_DESIGN_FAMILIES = {
    "gen/assign-and-start": (0.5, 1.0),
    "gen/close-duplicate": (0.2, 0.8),
    "gen/conditional-assign": (0.0, 0.6),
    "gen/escalate-label": (0.3, 0.9),
    "gen/reassign-departed": (0.0, 0.5),
}
EVAL_DESIGN_TASKS_PER_FAMILY = 3
EVAL_DESIGN_EXPERIMENTS = 1000


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "已知限制（decision 013 §6）：decision 010 §5 的涵蓋率門檻 >= 0.88 在 decision 005 的 5 x 3 eval 設計、superpopulation"
        "真值下不成立（約 0.84-0.87；m = 3 時 percentile bootstrap 的層內變異只有不偏估計的 2/3）。正式 estimand 是 decision 005"
        "的『母體 = 15 個固定 instances』（decision 013 §6 澄清）；本測試以 strict xfail 記錄 superpopulation 讀法下的限制。"
    ),
)
def test_bootstrap_coverage_at_eval_design_5x3_superpopulation() -> None:
    n_total = EVAL_DESIGN_TASKS_PER_FAMILY * len(EVAL_DESIGN_FAMILIES)
    truth = (
        sum(
            EVAL_DESIGN_TASKS_PER_FAMILY * (lo + hi) / 2
            for lo, hi in EVAL_DESIGN_FAMILIES.values()
        )
        / n_total
    )
    data_rng = random.Random(20261008)
    covered = 0
    for experiment in range(EVAL_DESIGN_EXPERIMENTS):
        values: dict[str, float] = {}
        strata: dict[str, str] = {}
        for family, (lo, hi) in EVAL_DESIGN_FAMILIES.items():
            for seed in range(1, EVAL_DESIGN_TASKS_PER_FAMILY + 1):
                p_task = lo + (hi - lo) * data_rng.random()
                successes = sum(
                    data_rng.random() < p_task for _ in range(COVERAGE_ROLLOUTS)
                )
                task_id = f"{family}-s{seed}"
                values[task_id] = successes / COVERAGE_ROLLOUTS
                strata[task_id] = family
        result = stratified_task_bootstrap(
            values, strata, replicates=COVERAGE_REPLICATES, seed=experiment
        )
        assert result.n_tasks == n_total
        covered += result.low <= truth <= result.high
    coverage = covered / EVAL_DESIGN_EXPERIMENTS
    # 預先登記的門檻（decision 010 §5）原樣套用到真正的 eval 設計；不調整 family 或門檻。
    assert coverage >= 0.88, f"coverage = {coverage:.3f}（truth = {truth:.4f}）"


# ---------------------------------------------------------------- §6 審查確認的溢位邊界（回歸測試）


@pytest.mark.parametrize("z", [1.35e154, 1e200, 1.7e308])
def test_wilson_rejects_z_whose_square_overflows(z: float) -> None:
    """z*z 溢位成 inf 時原本會靜默回傳 (0.0, 0.0)（不含 p̂）。必須 ValueError。"""
    with pytest.raises(ValueError):
        wilson_interval(1, 2, z)


@pytest.mark.parametrize("n", [10**154, 10**160, 10**200])
def test_wilson_handles_huge_n(n: int) -> None:
    low, high = wilson_interval(n // 2, n)
    assert low <= 0.5 <= high
    low, high = wilson_interval(0, n)
    assert low == 0.0 and 0.0 <= high <= 1e-100


def test_bootstrap_handles_values_near_float_max() -> None:
    result = stratified_task_bootstrap({"a": 1e308, "b": 1e308}, {"a": "f", "b": "f"}, replicates=5)
    assert result.estimate == pytest.approx(1e308) and result.low == pytest.approx(1e308)
    result = stratified_task_bootstrap({"a": 1e308, "b": -1e308}, {"a": "f", "b": "f"}, replicates=50)
    assert result.estimate == 0.0 and result.low == pytest.approx(-1e308) and result.high == pytest.approx(1e308)
