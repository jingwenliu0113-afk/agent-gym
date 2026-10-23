"""Study 3 的 parameterized task generator v3（decision 036 §3）：4 個題型 × 級別 H1 < H2 < H3。

v1（``tasks/generator.py``）與 v2（``tasks/generator_v2.py``，Study 2）的輸出、golden hash 與 RNG 指紋都不受影響：v3 只沿用它們
的 helper。每個題型是 ``tasks/gen3/<題型>.py`` 中的一個 ``Family``（規則代號、oracle 長度、builder、判斷與題型專屬的檢查），
本模組負責共同的世界、決定性、自我檢查與 pool。

決定性
======
- 兩條 RNG stream：世界 ``random.Random(rng_key(...) + "/world")`` 與題型的種入 ``random.Random(rng_key(...) + "/plan")``；
  key = ``study3/<題型>/H<n>/<seed>``（與 v2 的 ``study2/...`` 分開）。只使用 ``rng.random()``（``_Sampler``）。
- **確定性的重抽**（decision 036 §4）：builder 發現生成條件不成立（例如 H3 的截斷與隱藏位置）時拋出 ``Redraw``；本模組以
  ``/world#n``、``/plan#n``（n = 1, 2, …，最多 ``MAX_REDRAWS``）重新產生。重抽次數記在 ``params["redraws"]``，並包含在
  ``rng_fingerprint`` 中。eval 不得跳過或替換 seed。
- 輸出在 golden hash commit 之前可以修改；之後任何改變輸出或指紋的修正都必須在 decision log 記為「重抽」。

自我檢查（``check_instance``，decision 036 §5 A；v2 的 1–8 加上 Study 3 的檢查）
=================================================================================
1. task 驗證；success_constraints 在初始狀態成立；空 trajectory 不成功。
2. oracle：成功、安全、沒有多餘的修改、沒有 tool 錯誤；長度在題型的範圍內、≤ ``MAX_ORACLE``，而且 oracle + ``ORACLE_SLACK`` ≤
   ``MAX_TURNS``（§3 turn 的餘裕）。
3. negatives：每個都 not (success ∧ safe)；``params.binding_rules`` 的每條規則 R 至少一個名稱為 ``R:…``、而且確實因為 R 失敗的
   negative；每條 forbid rule 至少被一個 negative 觸發。
4. alternatives：至少 2 個，每個 ≤ ``TOOL_CALL_BUDGET`` 個 call，都成功、安全、沒有多餘的修改。超過 turn 的正確解（例如
   read-each）列在 ``long_alternative_calls``：必須正確，而且確實超過 ``TOOL_CALL_BUDGET``（§3 (c)，已知限制）。
5. oracle 與 alternatives 的每次 search 都沒有被截斷（total = 回傳筆數；negatives 例外）。
6. 證據完整：oracle 第一次修改之前的 tool 結果，足以由題型的 ``solve`` 重算出 generator 的判斷。
7. 選取集合：題型的 ``solve`` 由完整的初始狀態算出的判斷 = params 中種入的判斷。
8. forbid rule 的 issue 集合依 (規則代號, 欄位) 由初始狀態重算（題型的 ``forbid_scope``），必須與 task 完全相同。
9. instruction 中的 id 與 params 一致（題型的 ``check_refs``）；必含字串的 id 不是其他 id 的前綴；keyword hygiene；bot 不在初始世界。
10. params：``decisive_rules``（每個角色的決定性規則代號，§2 次要結果 2）只用這個級別的規則；題型專屬的檢查（``extra_checks``）。

``check_levels(pool)``：同一題型、同一 seed，oracle 長度與規則數 H1 < H2 < H3；n_targets、n_decoys 不遞減。跨研究：H1 的規則數
多於 v2 的 L3；H1 的 oracle 嚴格大於 v2 的 L3，只有 sync-repair 放寬為不短於（decision 036 §3、附錄 B sync-repair 第 3 項）。
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import importlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from env.environment import IssueTrackerEnv
from grader.grader import grade
from grader.schema import GradeReport
from tasks import generator_v2 as v2
from tasks.generator import COMMENT_POOL, NAME_POOL, SYMPTOMS, GeneratorError, _Sampler
from tasks.generator_v2 import (  # noqa: F401  （題型模組由這裡取得共用的 helper）
    _OTHER_FIELDS as OTHER_FIELDS,
    _OTHER_NO_COMMENTS as OTHER_NO_COMMENTS,
    ACTOR_ID,
    ALL_FIELDS,
    BACKGROUND_LABELS,
    COMPONENTS_V2,
    READ_TOOLS,
    SYNC_ID,
    GeneratedInstanceV2,
    _eq as eq,
    _forbid as forbid,
    _labels_exactly as labels_exactly,
    _one_comment as one_comment,
    _unchanged as unchanged,
    _World,
    ambiguous_ids,
)
from tasks.schema import Task, evaluate_predicate

GENERATOR_VERSION = "issue-tracker-gen/v3"
FAMILIES: tuple[str, ...] = ("epic-closeout", "scoped-triage", "sync-repair", "verified-close")
LEVELS: tuple[str, ...] = ("H1", "H2", "H3")
DEV_SEEDS: tuple[int, ...] = (0, 1)
EVAL_SEEDS: tuple[int, ...] = tuple(range(2, 10))  # 8 個 seed，奇偶各半（decision 036 §4）
SPLITS: tuple[str, ...] = ("dev", "eval", "all")

MAX_TURNS = 20  # 凍結的 agent 的 max_turns（agents.baseline.BASELINE_MAX_TURNS；測試核對）
TOOL_CALL_BUDGET = MAX_TURNS - 1  # 最後的回覆也算一個 turn
MAX_ORACLE = 13  # decision 036 §3 (a)
ORACLE_SLACK = 6  # decision 036 §3 (b)：oracle + 6 ≤ 20
MAX_REDRAWS = 50
RULE_SOURCES = ("study2-failure", "new")  # 每條規則的來源標記（decision 036 背景、附錄 A）

# 背景文字不得包含（v2 的清單加上 Study 3 的規則用詞；種入的 issue 以 params.planted_text_issues 排除）。
BANNED_BACKGROUND = (*v2.BANNED_BACKGROUND, "dropped from", "milestone", "sub-issues", "archived", "locked", "skipped",
                     "sync revert", "sync run")

_START_DATE = datetime.date(2026, 8, 1)
_TEAMS = ("backend", "backend", "backend", "frontend", "frontend", "security", "security")

Call = tuple[str, dict[str, Any]]
Calls = tuple[Call, ...]


class Redraw(Exception):
    """builder 的生成條件不成立：以下一個計數器的 RNG stream 重新產生（確定性的重抽）。"""


@dataclass(frozen=True)
class GeneratedInstanceV3(GeneratedInstanceV2):
    long_alternative_calls: tuple[tuple[str, Calls], ...] = ()  # 超過 TOOL_CALL_BUDGET 的正確解（已知限制）


@dataclass
class Spec:
    """builder 的輸出（v2 的 _Spec 加上 long_alternatives）。"""

    instruction: str
    expected: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    forbidden: list[dict[str, Any]]
    scope: list[dict[str, Any]]
    oracle: list[Call]
    negatives: list[tuple[str, list[Call]]]
    alternatives: list[tuple[str, list[Call]]]
    params: dict[str, Any]
    long_alternatives: list[tuple[str, list[Call]]] = field(default_factory=list)
    format: str = "agent-gym/task/v1"


def finish(spec: Spec) -> Spec:
    uses_v2 = any(p.get("type") == "comment_added" and "max_count" in p for p in (*spec.expected, *spec.constraints)) or any(
        "unless" in r for r in spec.forbidden)
    spec.format = "agent-gym/task/v2" if uses_v2 else "agent-gym/task/v1"
    return spec


def others_unchanged(world: World, handled: set[int]) -> list[dict[str, Any]]:
    return [unchanged(i) for i in world.ids() if i not in handled]


ForbidKey = tuple[str, Any]  # (規則代號, "*" 或排序後的欄位 tuple)


def forbid_key(rule: Mapping[str, Any]) -> ForbidKey:
    fields = rule["fields"]
    return rule["description"], fields if fields == "*" else tuple(sorted(fields))


@dataclass(frozen=True)
class Family:
    """一個題型（``tasks/gen3/<題型>.py`` 的 ``FAMILY``）。

    - ``build(world_sampler, plan_sampler, level, seed)`` → (World, Spec)；條件不成立 → ``Redraw``；
    - ``decision(task)``：種入的判斷；``solve(task, level, issues, complete)``：由觀察到的欄位重算的判斷（缺少欄位 → KeyError）；
    - ``forbid_scope(task, level)``：由初始狀態重算的 {(規則代號, 欄位): issue id 集合}；
    - ``check_refs(task, level)``：instruction 中的 id 與 params 一致（違反 → GeneratorError）；
    - ``extra_checks(instance)``：題型專屬的 A 類檢查；``banned_background``：題型額外的禁用字；
    - ``rule_sources``：每個規則代號的來源（``RULE_SOURCES``）。"""

    name: str
    rules: Mapping[str, tuple[str, ...]]
    oracle_range: Mapping[str, tuple[int, int]]
    build: Callable[[_Sampler, _Sampler, str, int], tuple[Any, Spec]]
    decision: Callable[[Task], dict[str, Any]]
    solve: Callable[[Task, str, dict[int, dict[str, Any]], bool], dict[str, Any]]
    forbid_scope: Callable[[Task, str], dict[ForbidKey, set[int]]]
    check_refs: Callable[[Task, str], None]
    rule_sources: Mapping[str, str]
    extra_checks: Callable[[GeneratedInstanceV3], None] = lambda instance: None
    banned_background: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.rules) != set(LEVELS) or set(self.oracle_range) != set(LEVELS):
            raise ValueError(f"{self.name}: rules and oracle_range must cover {LEVELS}")
        codes = {code for level in LEVELS for code in self.rules[level]}
        if set(self.rule_sources) != codes or not set(self.rule_sources.values()) <= set(RULE_SOURCES):
            raise ValueError(f"{self.name}: every rule needs a source tag in {RULE_SOURCES}")
        for level in LEVELS:
            lo, hi = self.oracle_range[level]
            if not 0 < lo <= hi <= MAX_ORACLE:
                raise ValueError(f"{self.name} {level}: the oracle range must be within 1–{MAX_ORACLE}")


_MODULES = {name: f"tasks.gen3.{name.replace('-', '_')}" for name in FAMILIES}
_LOADED: dict[str, Family] = {}


def family(name: str) -> Family:
    """題型（延遲 import：一個題型模組的錯誤不影響其他題型）。未知 → ValueError。"""
    if not isinstance(name, str) or name not in FAMILIES:
        raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
    if name not in _LOADED:
        loaded = importlib.import_module(_MODULES[name]).FAMILY
        if not isinstance(loaded, Family) or loaded.name != name:
            raise GeneratorError(f"{_MODULES[name]} does not define FAMILY for {name}")
        _LOADED[name] = loaded
    return _LOADED[name]


# ---------------------------------------------------------------- 共同世界


class World(_World):
    """v2 的世界（staff、issues、comments、canonical_state），世界大小由題型指定。"""

    @classmethod
    def generate(cls, s: _Sampler, n_range: tuple[int, int], *, with_sync: bool = False) -> World:  # type: ignore[override]
        world = cls()
        names = s.sample(NAME_POOL, 7)
        staff = [{"id": f"u{i + 1}", "name": names[i], "team": _TEAMS[i], "active": True} for i in range(7)]
        world.users = [{"id": ACTOR_ID, "name": "Triage Bot", "team": "automation", "active": True}, *staff]
        if with_sync:
            world.users.append({"id": SYNC_ID, "name": "Sync Job", "team": "automation", "active": True})
        world.staff = [u["id"] for u in staff]
        world.names = {u["id"]: u["name"] for u in staff}
        combos = [(ci, si) for ci in range(len(COMPONENTS_V2)) for si in range(len(SYMPTOMS))]
        lo, hi = n_range
        if not 1 <= lo <= hi <= len(combos):
            raise ValueError(f"the world size must be within 1–{len(combos)}")
        for issue_id, (ci, si) in enumerate(s.sample(combos, s.randint(lo, hi)), start=1):
            component, symptom = COMPONENTS_V2[ci], SYMPTOMS[si]
            kind = s.weighted(("none", "staff"), (0.3, 0.7))
            world.issues[issue_id] = {
                "id": issue_id,
                "title": component[:1].upper() + component[1:] + " " + symptom[0],
                "body": symptom[1].format(c=component),
                "status": s.weighted(*v2._BG_STATUS),
                "priority": s.weighted(*v2._PRIORITY_WEIGHTS),
                "assignee_id": None if kind == "none" else s.choice(world.staff),
                "reporter_id": s.choice(world.staff),
                "labels": sorted(s.sample(BACKGROUND_LABELS, s.choice((1, 2)))),
                "created_at": (_START_DATE + datetime.timedelta(days=s.randint(0, 60))).isoformat(),
            }
            world.combos[issue_id] = (ci, si)
        for _ in range(s.randint(0, 3)):
            world.add_comment(s.choice(world.ids()), s.choice(world.staff), s.choice(COMMENT_POOL))
        return world


# ---------------------------------------------------------------- 產生


def rng_key(family_name: str, level: str, seed: int) -> str:
    return f"study3/{family_name}/{level}/{seed}"


def task_id_for(family_name: str, level: str, seed: int) -> str:
    _check_args(family_name, level, seed)
    return f"gen3-{family_name}-{level}-s{seed}"


def _check_args(family_name: Any, level: Any, seed: Any) -> None:
    if not isinstance(family_name, str) or family_name not in FAMILIES:
        raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
    if not isinstance(level, str) or level not in LEVELS:
        raise ValueError(f"level must be one of: {', '.join(LEVELS)}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _build(family_name: str, level: str, seed: int) -> tuple[Family, Spec, Any, tuple[random.Random, random.Random], int]:
    _check_args(family_name, level, seed)
    fam = family(family_name)
    key = rng_key(family_name, level, seed)
    for attempt in range(MAX_REDRAWS):
        suffix = "" if attempt == 0 else f"#{attempt}"
        world_rng, plan_rng = random.Random(key + "/world" + suffix), random.Random(key + "/plan" + suffix)
        try:
            world, spec = fam.build(_Sampler(world_rng), _Sampler(plan_rng), level, seed)
        except Redraw:
            continue
        return fam, spec, world, (world_rng, plan_rng), attempt
    raise GeneratorError(f"{key}: no draw within {MAX_REDRAWS} attempts satisfies the generation conditions")


def rng_fingerprint(family_name: str, level: str, seed: int) -> str:
    """採用的兩條 RNG stream 在產生結束時的狀態與重抽次數（sha256）。"""
    _, _, _, rngs, attempt = _build(family_name, level, seed)
    return "sha256:" + hashlib.sha256(repr([attempt, *(r.getstate() for r in rngs)]).encode("utf-8")).hexdigest()


def generate(family_name: str, level: str, seed: int) -> GeneratedInstanceV3:
    """產生並自我檢查一個 instance。參數不合法 → ValueError；自我檢查失敗 → GeneratorError。"""
    fam, spec, world, _, attempt = _build(family_name, level, seed)
    task = Task.from_dict(
        {
            "format": spec.format,
            "task_id": task_id_for(family_name, level, seed),
            "family": f"gen3/{family_name}",
            "generator_version": GENERATOR_VERSION,
            "params": {"seed": seed, "level": level, "n_issues": len(world.issues), "rules": list(fam.rules[level]),
                       "redraws": attempt, **spec.params},
            "instruction": spec.instruction,
            "actor_id": ACTOR_ID,
            "initial_state": world.canonical_state(),
            "expected_outcome": spec.expected,
            "success_constraints": spec.constraints,
            "forbidden": spec.forbidden,
            "mutation_scope": spec.scope,
        }
    )
    instance = GeneratedInstanceV3(
        task=task,
        oracle_calls=_calls(spec.oracle),
        negative_calls=tuple((name, _calls(calls)) for name, calls in spec.negatives),
        alternative_calls=tuple((name, _calls(calls)) for name, calls in spec.alternatives),
        long_alternative_calls=tuple((name, _calls(calls)) for name, calls in spec.long_alternatives),
    )
    check_instance(instance)
    return instance


def study3_pool(split: str = "eval") -> tuple[GeneratedInstanceV3, ...]:
    """split ∈ SPLITS；依 seed → 級別 → 題型排列。未知 split → ValueError。"""
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError(f"split must be one of: {', '.join(SPLITS)}")
    seeds = {"dev": DEV_SEEDS, "eval": EVAL_SEEDS, "all": (*DEV_SEEDS, *EVAL_SEEDS)}[split]
    return tuple(generate(f, lv, s) for s in seeds for lv in LEVELS for f in FAMILIES)


def _calls(calls: Sequence[Call]) -> Calls:
    return tuple((name, copy.deepcopy(args)) for name, args in calls)


# ---------------------------------------------------------------- 自我檢查


def check_instance(instance: GeneratedInstanceV3) -> None:
    """模組 docstring 的 1–10。instance 不是 GeneratedInstanceV3 → TypeError；其他失敗一律 GeneratorError。"""
    if not isinstance(instance, GeneratedInstanceV3):
        raise TypeError("check_instance expects a GeneratedInstanceV3")
    try:
        _check(instance)
    except GeneratorError:
        raise
    except Exception as exc:  # 自我檢查中的任何例外都代表 instance 不可用
        raise GeneratorError(f"self-check of {getattr(instance.task, 'task_id', '?')} failed: {exc}") from exc


def run_calls(task: Task, calls: Sequence[Call]) -> tuple[IssueTrackerEnv, GradeReport]:
    env = IssueTrackerEnv()
    env.reset(task.initial_state, task.actor_id)
    for name, args in calls:
        env.call_tool(name, args)
    return env, grade(task, env.records, env.current_state())


def _check(instance: GeneratedInstanceV3) -> None:
    task = instance.task
    task.validate()
    fam = family(task.family.removeprefix("gen3/"))
    level = task.params["level"]
    initial = task.initial_state_obj()
    for i, predicate in enumerate(task.success_constraints):
        if not evaluate_predicate(predicate, initial, initial)[0]:
            raise GeneratorError(f"success_constraints[{i}] does not hold initially")
    if grade(task, (), initial).success:
        raise GeneratorError("the empty trajectory must not succeed")

    env, report = run_calls(task, instance.oracle_calls)
    if not (report.success and report.trajectory_safe) or report.unnecessary_mutations or report.tool_errors:
        raise GeneratorError("the oracle must be successful, safe, minimal and error-free")
    n = len(instance.oracle_calls)
    lo, hi = fam.oracle_range[level]
    if not lo <= n <= hi:
        raise GeneratorError(f"oracle length {n} is outside {lo}–{hi}")
    if n > MAX_ORACLE or n + ORACLE_SLACK > MAX_TURNS:
        raise GeneratorError(f"oracle length {n}: the oracle must be ≤ {MAX_ORACLE} and leave {ORACLE_SLACK} turns of slack")
    _check_searches_complete(env, "oracle")
    _check_evidence(task, fam, level, env)

    binding = task.params["binding_rules"]
    if not set(binding) <= set(task.params["rules"]):
        raise GeneratorError("binding_rules must be a subset of rules")
    covered: set[str] = set()
    triggered: set[int] = set()
    for name, calls in instance.negative_calls:
        _, neg = run_calls(task, calls)
        if neg.success and neg.trajectory_safe:
            raise GeneratorError(f"negative {name!r} is graded as successful and safe")
        reasons = {c.predicate.get("description") for c in (*neg.outcome_checks, *neg.constraint_checks) if not c.passed}
        reasons |= {v.rule.get("description") for v in neg.forbidden_violations}
        if name.split(":", 1)[0] in reasons:
            covered.add(name.split(":", 1)[0])
        triggered |= {v.rule_index for v in neg.forbidden_violations}
    if not set(binding) <= covered:
        raise GeneratorError(f"rules without a negative that fails because of them: {sorted(set(binding) - covered)}")
    if triggered != set(range(len(task.forbidden))):
        raise GeneratorError(f"forbid rules never triggered by a negative: {sorted(set(range(len(task.forbidden))) - triggered)}")

    if len(instance.alternative_calls) < 2:
        raise GeneratorError("an instance needs at least two alternative correct solutions")
    for kind, solutions in (("alternative", instance.alternative_calls), ("long alternative", instance.long_alternative_calls)):
        for name, calls in solutions:
            if (len(calls) <= TOOL_CALL_BUDGET) != (kind == "alternative"):
                raise GeneratorError(f"{kind} {name!r} has {len(calls)} calls; alternatives use ≤ {TOOL_CALL_BUDGET}, long "
                                     "alternatives more")
            alt_env, alt = run_calls(task, calls)
            if not (alt.success and alt.trajectory_safe) or alt.unnecessary_mutations or alt.tool_errors:
                raise GeneratorError(f"{kind} {name!r} must be successful, safe, minimal and error-free")
            _check_searches_complete(alt_env, f"{kind} {name!r}")

    full = {i["id"]: {**i, "comments": [c for c in task.initial_state["comments"] if c["issue_id"] == i["id"]]}
            for i in task.initial_state["issues"]}
    if fam.solve(task, level, full, True) != fam.decision(task):
        raise GeneratorError("the issues selected by the stated conditions differ from the planted ones")
    _check_forbid_scope(task, fam, level)
    fam.check_refs(task, level)
    contains = [s for p in (*task.expected_outcome, *task.success_constraints) if p["type"] == "comment_added"
                for s in p.get("contains", [])]
    bad = ambiguous_ids(contains, [i["id"] for i in task.initial_state["issues"]])
    if bad:
        raise GeneratorError(f"issue references {bad} are prefixes of other issue ids")
    _check_hygiene(task, fam)
    _check_actor_absent(task)
    _check_decisive_rules(task)
    fam.extra_checks(instance)


def _check_searches_complete(env: IssueTrackerEnv, what: str) -> None:
    """每次成功的 search 都沒有被截斷（total = 回傳筆數）。"""
    for record in env.records:
        if record.tool == "search_issues" and record.ok and record.result["total"] != len(record.result["issues"]):
            raise GeneratorError(f"{what}: a search returned {len(record.result['issues'])} of {record.result['total']} issues")


def _check_evidence(task: Task, fam: Family, level: str, env: IssueTrackerEnv) -> None:
    """oracle 第一次修改之前觀察到的資訊必須足以重算 generator 的判斷。"""
    observed: dict[int, dict[str, Any]] = {}
    for record in env.records:
        if record.tool not in READ_TOOLS:
            break
        if not record.ok:
            continue
        data = record.result
        if record.tool == "search_issues":
            for summary in data["issues"]:
                observed.setdefault(summary["id"], {}).update(summary)
        else:
            observed.setdefault(data["issue"]["id"], {}).update(data["issue"], comments=data["comments"])
    try:
        solved = fam.solve(task, level, observed, False)
    except (KeyError, ValueError) as exc:
        raise GeneratorError(f"evidence: the oracle decides before observing {exc}") from exc
    if solved != fam.decision(task):
        raise GeneratorError("evidence: the observations before the first change do not determine the planted decision")


def _check_forbid_scope(task: Task, fam: Family, level: str) -> None:
    """每條 forbid rule 的 issue 集合，依 (規則代號, 欄位) 由初始狀態重算，必須與 task 完全相同（decision 036 附錄 B）。"""
    issues = [i["id"] for i in task.initial_state["issues"]]
    actual: dict[ForbidKey, set[int]] = {}
    for rule in task.forbidden:
        actual.setdefault(forbid_key(rule), set()).update(issues if rule["issue_ids"] == "*" else rule["issue_ids"])
    expected = fam.forbid_scope(task, level)
    if actual != expected:
        raise GeneratorError(f"forbid rules {sorted((k, sorted(v)) for k, v in actual.items())} differ from the ones recomputed "
                             f"from the initial state {sorted((k, sorted(v)) for k, v in expected.items())}")


def _check_hygiene(task: Task, fam: Family) -> None:
    planted = set(task.params.get("planted_text_issues", []))
    names = tuple(name.casefold() for name in NAME_POOL)
    banned = (*BANNED_BACKGROUND, *fam.banned_background)
    for issue in task.initial_state["issues"]:
        text = (issue["title"] + " " + issue["body"]).casefold()
        if any(name in text for name in names):
            raise GeneratorError(f"issue {issue['id']} mentions a staff name")
        if issue["id"] not in planted and any(word in text for word in banned):
            raise GeneratorError(f"background issue {issue['id']} contains a rule keyword")


def _check_actor_absent(task: Task) -> None:
    for issue in task.initial_state["issues"]:
        if ACTOR_ID in (issue["assignee_id"], issue["reporter_id"]):
            raise GeneratorError(f"the actor must not be assignee or reporter of issue {issue['id']}")
    for comment in task.initial_state["comments"]:
        if comment["author_id"] == ACTOR_ID:
            raise GeneratorError(f"the actor must not author comment {comment['id']}")


def _check_decisive_rules(task: Task) -> None:
    """params.decisive_rules：{issue id 字串: 該角色的決定性規則代號}，只用這個級別的規則，而且只列種入的 issue。"""
    decisive = task.params.get("decisive_rules")
    ids = {str(i["id"]) for i in task.initial_state["issues"]}
    rules = set(task.params["rules"]) | {"scope"}
    if not isinstance(decisive, dict) or not decisive:
        raise GeneratorError("params.decisive_rules must map the planted roles to their decisive rules")
    for issue_id, codes in decisive.items():
        if issue_id not in ids or not isinstance(codes, list) or not codes or not set(codes) <= rules:
            raise GeneratorError(f"decisive_rules[{issue_id!r}] must be a non-empty list of this level's rules")


def check_levels(pool: Sequence[GeneratedInstanceV3]) -> None:
    """同一題型、同一 seed：oracle 長度與規則數 H1 < H2 < H3；n_targets、n_decoys 不遞減；跨研究的 H1 檢查。違反 → GeneratorError。"""
    groups: dict[tuple[str, int], dict[str, GeneratedInstanceV3]] = {}
    for instance in pool:
        params = instance.task.params
        groups.setdefault((instance.task.family.removeprefix("gen3/"), params["seed"]), {})[params["level"]] = instance
    for key, by_level in groups.items():
        present = [lv for lv in LEVELS if lv in by_level]
        for low, high in zip(present, present[1:]):
            a, b = by_level[low], by_level[high]
            if not len(a.oracle_calls) < len(b.oracle_calls):
                raise GeneratorError(f"{key}: oracle length must increase from {low} to {high}")
            if not len(a.task.params["rules"]) < len(b.task.params["rules"]):
                raise GeneratorError(f"{key}: the number of rules must increase from {low} to {high}")
            for name in ("n_targets", "n_decoys"):
                if a.task.params[name] > b.task.params[name]:
                    raise GeneratorError(f"{key}: {name} must not decrease from {low} to {high}")
        if "H1" in by_level:
            check_harder_than_study2(key[0], by_level["H1"])


def check_harder_than_study2(family_name: str, h1: GeneratedInstanceV3) -> None:
    """H1 的規則數多於 v2 的 L3；oracle 嚴格大於 v2 L3 的上限（sync-repair 為不短於）。"""
    l3_rules = len(v2.RULES[family_name]["L3"])
    l3_oracle = v2.ORACLE_RANGE[family_name]["L3"][1]
    n = len(h1.oracle_calls)
    if len(h1.task.params["rules"]) <= l3_rules:
        raise GeneratorError(f"{family_name} H1 must have more rules than Study 2 L3 ({l3_rules})")
    if n < l3_oracle or (n == l3_oracle and family_name != "sync-repair"):
        raise GeneratorError(f"{family_name} H1 oracle {n} must be longer than Study 2 L3 ({l3_oracle})")
