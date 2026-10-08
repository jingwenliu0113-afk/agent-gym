"""env/hashing.py 的契約測試（decision log 002）。"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import decimal
import hashlib
import json
import random
import re
from typing import Any

import pytest

from env.hashing import HASH_PREFIX, canonical_json, sha256_of

HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_STR_POOL = "abcXYZ 012-_.登入頁面é🙂\"\\\n\t/"


def _reference_json(obj: Any) -> str:
    """decision 002 規定的 json.dumps 參數；只用於 JSON-native（不含 tuple）的輸入。"""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _random_str(rng: random.Random) -> str:
    return "".join(rng.choice(_STR_POOL) for _ in range(rng.randint(0, 8)))


def _random_json(rng: random.Random, depth: int = 0) -> Any:
    kinds = ["none", "bool", "int", "float", "str"]
    if depth < 4:
        kinds += ["list", "dict", "dict"]
    kind = rng.choice(kinds)
    if kind == "none":
        return None
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "int":
        return rng.randint(-(10**12), 10**12)
    if kind == "float":
        return rng.uniform(-1e6, 1e6)
    if kind == "str":
        return _random_str(rng)
    if kind == "list":
        return [_random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {_random_str(rng): _random_json(rng, depth + 1) for _ in range(rng.randint(0, 5))}


def _reorder_keys(obj: Any, rng: random.Random) -> Any:
    """重建每個 dict，key 以隨機順序插入；值不變。"""
    if isinstance(obj, dict):
        keys = list(obj)
        rng.shuffle(keys)
        return {k: _reorder_keys(obj[k], rng) for k in keys}
    if isinstance(obj, list):
        return [_reorder_keys(v, rng) for v in obj]
    return obj


def _reverse_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _reverse_keys(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_reverse_keys(v) for v in obj]
    return obj


def _expected_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class _Point:
    x: int
    y: int


class _Opaque:
    pass


# ---------------------------------------------------------------------------
# canonical_json：格式
# ---------------------------------------------------------------------------


def test_canonical_json_returns_str():
    assert type(canonical_json({"a": 1})) is str


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2, "c": 3}) == '{"a":2,"b":1,"c":3}'


def test_canonical_json_sorts_nested_keys():
    obj = {"z": {"y": 1, "x": 2}, "a": [{"d": 1, "c": 2}]}
    assert canonical_json(obj) == '{"a":[{"c":2,"d":1}],"z":{"x":2,"y":1}}'


def test_canonical_json_sorts_keys_by_code_point():
    assert canonical_json({"é": 1, "z": 2, "B": 3, "a": 4}) == '{"B":3,"a":4,"z":2,"é":1}'


def test_canonical_json_uses_compact_separators():
    assert canonical_json({"a": [1, 2, {"b": None}], "c": "x y"}) == '{"a":[1,2,{"b":null}],"c":"x y"}'


def test_canonical_json_keeps_non_ascii():
    assert canonical_json({"t": "登入頁面"}) == '{"t":"登入頁面"}'


def test_canonical_json_keeps_non_bmp_characters():
    assert canonical_json(["🙂", "é"]) == '["🙂","é"]'


def test_canonical_json_non_ascii_output_has_no_unicode_escapes():
    assert "\\u" not in canonical_json({"名稱": "字型錯誤", "e": "café"})


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (-7, "-7"),
        ("x", '"x"'),
        ("", '""'),
        ([], "[]"),
        ({}, "{}"),
        (1.5, "1.5"),
    ],
)
def test_canonical_json_scalars_and_empty_containers(obj, expected):
    assert canonical_json(obj) == expected


def test_canonical_json_escapes_quotes_and_control_characters():
    obj = {"s": 'a"b\\c\nd'}
    assert canonical_json(obj) == _reference_json(obj)
    assert json.loads(canonical_json(obj)) == obj


# ---------------------------------------------------------------------------
# canonical_json：tuple 與 bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ((1, 2), "[1,2]"),
        ((), "[]"),
        ({"a": (1, (2, 3))}, '{"a":[1,[2,3]]}'),
        ([(1,), [(2,)]], "[[1],[[2]]]"),
        (({"b": 1, "a": (True, None)},), '[{"a":[true,null],"b":1}]'),
    ],
)
def test_canonical_json_serializes_tuples_as_lists(obj, expected):
    assert canonical_json(obj) == expected


def test_canonical_json_tuple_and_list_are_identical():
    assert canonical_json({"a": (1, "x", (None,))}) == canonical_json({"a": [1, "x", [None]]})


def test_canonical_json_bools_are_true_false():
    assert canonical_json([True, False, 1, 0]) == "[true,false,1,0]"
    assert canonical_json({"a": True}) == '{"a":true}'


@pytest.mark.parametrize(("b", "i"), [(True, 1), (False, 0)])
def test_canonical_json_bool_differs_from_int(b, i):
    assert canonical_json(b) != canonical_json(i)
    assert canonical_json({"k": b}) != canonical_json({"k": i})


# ---------------------------------------------------------------------------
# canonical_json：錯誤
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param({1: "a"}, id="int_key"),
        pytest.param({0: "a"}, id="zero_key"),
        pytest.param({None: "a"}, id="none_key"),
        pytest.param({True: "a"}, id="true_key"),
        pytest.param({False: "a"}, id="false_key"),
        pytest.param({1.5: "a"}, id="float_key"),
        pytest.param({(1, 2): "a"}, id="tuple_key"),
        pytest.param({"a": 1, 2: "b"}, id="mixed_keys"),
        pytest.param({"a": {2: "b"}}, id="nested_dict"),
        pytest.param([{"a": 1}, {3: "c"}], id="nested_in_list"),
        pytest.param(({"ok": 1}, {4: "d"}), id="nested_in_tuple"),
        pytest.param({"a": [{"b": {None: 1}}]}, id="deeply_nested"),
    ],
)
def test_canonical_json_rejects_non_str_keys(obj):
    with pytest.raises(TypeError):
        canonical_json(obj)


def test_canonical_json_int_key_does_not_collide_with_str_key():
    assert canonical_json({"1": "a"}) == '{"1":"a"}'
    with pytest.raises(TypeError):
        canonical_json({1: "a"})


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="neg_inf"),
        pytest.param({"a": float("nan")}, id="nan_in_dict"),
        pytest.param([1, float("inf")], id="inf_in_list"),
        pytest.param((float("-inf"),), id="neg_inf_in_tuple"),
        pytest.param({"a": {"b": [float("nan")]}}, id="nan_deeply_nested"),
    ],
)
def test_canonical_json_rejects_nan_and_infinity(obj):
    with pytest.raises(ValueError):
        canonical_json(obj)


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param({1, 2}, id="set"),
        pytest.param(frozenset({1}), id="frozenset"),
        pytest.param(b"x", id="bytes"),
        pytest.param(bytearray(b"x"), id="bytearray"),
        pytest.param(object(), id="object"),
        pytest.param(_Opaque(), id="custom_object"),
        pytest.param(_Point(1, 2), id="dataclass_instance"),
        pytest.param(1j, id="complex"),
        pytest.param(decimal.Decimal("1.5"), id="decimal"),
        pytest.param(datetime.date(2026, 10, 8), id="date"),
        pytest.param({"a": {1, 2}}, id="set_in_dict"),
        pytest.param([b"x"], id="bytes_in_list"),
        pytest.param(({"a": object()},), id="object_in_tuple_dict"),
    ],
)
def test_canonical_json_rejects_non_json_types(obj):
    with pytest.raises(TypeError):
        canonical_json(obj)


# ---------------------------------------------------------------------------
# canonical_json：決定性
# ---------------------------------------------------------------------------


def test_canonical_json_independent_of_insertion_order():
    a = {"x": 1, "y": {"p": [1, 2], "q": None}, "z": "s"}
    b = {"z": "s", "y": {"q": None, "p": [1, 2]}, "x": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_list_order_matters():
    assert canonical_json([1, 2]) != canonical_json([2, 1])


def test_canonical_json_small_world_independent_of_key_order(small_world):
    assert canonical_json(small_world) == canonical_json(_reverse_keys(small_world))


def test_canonical_json_small_world_round_trips(small_world):
    assert json.loads(canonical_json(small_world)) == small_world


def test_canonical_json_random_objects_independent_of_insertion_order():
    rng = random.Random(42)
    for _ in range(200):
        obj = _random_json(rng)
        expected = canonical_json(obj)
        for _ in range(3):
            assert canonical_json(_reorder_keys(obj, rng)) == expected


def test_canonical_json_random_objects_match_reference_dumps():
    rng = random.Random(7)
    for _ in range(200):
        obj = _random_json(rng)
        assert canonical_json(obj) == _reference_json(obj)


def test_canonical_json_random_objects_round_trip():
    rng = random.Random(2026)
    for _ in range(200):
        obj = _random_json(rng)
        assert json.loads(canonical_json(obj)) == obj


def test_canonical_json_is_repeatable(small_world):
    assert canonical_json(small_world) == canonical_json(copy.deepcopy(small_world))


# ---------------------------------------------------------------------------
# sha256_of
# ---------------------------------------------------------------------------


def test_hash_prefix_constant():
    assert HASH_PREFIX == "sha256:"


@pytest.mark.parametrize(
    "obj",
    [None, True, 0, "x", [], {}, {"a": [1, (2, 3)]}, {"t": "登入"}],
)
def test_sha256_of_format(obj):
    h = sha256_of(obj)
    assert type(h) is str
    assert h.startswith(HASH_PREFIX)
    assert HASH_RE.match(h)


def test_sha256_of_empty_dict_known_vector():
    assert sha256_of({}) == (
        "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_sha256_of_empty_list_known_vector():
    assert sha256_of([]) == (
        "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )


def test_sha256_of_hashes_canonical_utf8_bytes():
    obj = {"d": None, "c": True, "b": [1, (2, 3)], "a": "登入"}
    assert sha256_of(obj) == _expected_hash('{"a":"登入","b":[1,[2,3]],"c":true,"d":null}')


def test_sha256_of_matches_canonical_json(small_world):
    assert sha256_of(small_world) == _expected_hash(canonical_json(small_world))


def test_sha256_of_equal_for_reordered_dicts():
    a = {"x": 1, "y": {"p": [1, 2], "q": None}}
    b = {"y": {"q": None, "p": [1, 2]}, "x": 1}
    assert sha256_of(a) == sha256_of(b)


def test_sha256_of_small_world_independent_of_key_order(small_world):
    assert sha256_of(small_world) == sha256_of(_reverse_keys(small_world))


def test_sha256_of_random_objects_independent_of_insertion_order():
    rng = random.Random(99)
    for _ in range(100):
        obj = _random_json(rng)
        assert sha256_of(obj) == sha256_of(_reorder_keys(obj, rng))


def test_sha256_of_tuple_equals_list():
    assert sha256_of({"a": (1, 2)}) == sha256_of({"a": [1, 2]})


def test_sha256_of_is_repeatable(small_world):
    assert sha256_of(small_world) == sha256_of(small_world)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param({"a": 1}, {"a": 2}, id="value"),
        pytest.param({"a": 1}, {"b": 1}, id="key"),
        pytest.param({"a": 1}, {"a": "1"}, id="int_vs_str"),
        pytest.param({"a": True}, {"a": 1}, id="bool_vs_int"),
        pytest.param({"a": False}, {"a": 0}, id="false_vs_zero"),
        pytest.param({"a": None}, {"a": "null"}, id="none_vs_str"),
        pytest.param({"a": None}, {}, id="none_vs_missing"),
        pytest.param([1, 2], [2, 1], id="list_order"),
        pytest.param([], {}, id="list_vs_dict"),
        pytest.param("登入", "登出", id="non_ascii"),
    ],
)
def test_sha256_of_differs_for_different_content(a, b):
    assert sha256_of(a) != sha256_of(b)


def test_sha256_of_changes_when_small_world_changes(small_world):
    before = sha256_of(small_world)
    small_world["issues"][0]["title"] = "Crash on login!"
    assert sha256_of(small_world) != before


def test_sha256_of_changes_when_labels_change(small_world):
    before = sha256_of(small_world)
    small_world["issues"][0]["labels"] = ["bug"]
    assert sha256_of(small_world) != before


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param("\ud800", id="high_surrogate"),
        pytest.param("\udfff", id="low_surrogate"),
        pytest.param({"a": "x\udc00y"}, id="surrogate_in_value"),
        pytest.param({"\ud83d": 1}, id="surrogate_in_key"),
        pytest.param(["ok", "\udc00"], id="surrogate_in_list"),
    ],
)
def test_sha256_of_rejects_lone_surrogates(obj):
    with pytest.raises(ValueError):
        sha256_of(obj)


@pytest.mark.parametrize(
    ("obj", "exc"),
    [
        pytest.param({1: "a"}, TypeError, id="non_str_key"),
        pytest.param({"a": {1, 2}}, TypeError, id="set"),
        pytest.param({"a": float("nan")}, ValueError, id="nan"),
        pytest.param([float("inf")], ValueError, id="inf"),
    ],
)
def test_sha256_of_propagates_canonical_json_errors(obj, exc):
    with pytest.raises(exc):
        sha256_of(obj)
