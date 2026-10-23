"""generator v3 的整個 pool（decision 036 §3 / §4 / §5 A）：dev 24 題、eval 96 題，四個題型一起的級別檢查與 context 檢查。

eval seed（2–9）的 instance 只經過程式檢查：失敗時只回報 task_id 與檢查名稱，不輸出內容（decision 036 §4）。
"""

from __future__ import annotations

import pytest

from experiments import study3_context as ctx
from tasks import generator_v3 as g3


@pytest.fixture(scope="module")
def pool() -> tuple[g3.GeneratedInstanceV3, ...]:
    return g3.study3_pool("all")


def test_the_pool_has_the_registered_splits_and_order(pool) -> None:
    dev, eval_ = g3.study3_pool("dev"), g3.study3_pool("eval")
    assert len(dev) == 24 and len(eval_) == 96 and pool == dev + eval_
    expected = [g3.task_id_for(f, lv, s) for s in (*g3.DEV_SEEDS, *g3.EVAL_SEEDS) for lv in g3.LEVELS for f in g3.FAMILIES]
    assert [i.task.task_id for i in pool] == expected
    for level in g3.LEVELS:
        assert sum(1 for i in eval_ if i.task.params["level"] == level) == 32
    with pytest.raises(ValueError):
        g3.study3_pool("test")


def test_levels_are_strictly_harder_across_the_whole_pool(pool) -> None:
    g3.check_levels(pool)


def test_every_rule_has_a_source_tag_and_the_study2_failure_types_are_tagged() -> None:
    tagged = {f: sorted(c for c, s in g3.family(f).rule_sources.items() if s == "study2-failure") for f in g3.FAMILIES}
    assert tagged == {"epic-closeout": ["E2", "E6"], "scoped-triage": ["S1", "S6"], "sync-repair": ["Y4", "Y6"],
                      "verified-close": ["V6"]}


def test_every_instance_fits_the_local_context(pool) -> None:
    failing = [i.task.task_id for i in pool if not ctx.check_context(i)["ok"]]
    assert failing == []  # 只列 task_id（decision 036 §4）


def test_golden_hash_pins_every_dev_and_eval_task(pool) -> None:
    """generator v3 的輸出在 B 類審查之後、第一次 dev 模型呼叫之前以 golden hash 釘住（decision 036 §8 第 2 步）。
    之後任何改變輸出的修正都必須在 decision log 記為「重抽」。"""
    import hashlib

    from env.hashing import canonical_json

    digest = hashlib.sha256("\n".join(canonical_json(i.task.to_dict()) for i in pool).encode("utf-8")).hexdigest()
    assert digest == g3.GOLDEN_POOL_SHA256


def test_rng_fingerprints_are_pinned() -> None:
    import hashlib

    keys = [(f, lv, s) for s in (*g3.DEV_SEEDS, *g3.EVAL_SEEDS) for lv in g3.LEVELS for f in g3.FAMILIES]
    joined = "\n".join(g3.rng_fingerprint(f, lv, s) for f, lv, s in keys)
    assert hashlib.sha256(joined.encode("utf-8")).hexdigest() == g3.GOLDEN_RNG_SHA256
