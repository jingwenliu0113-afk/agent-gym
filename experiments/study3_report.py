"""Study 3 的結果報告（decision 036 §6「分析與報告」）：本機模型與 Claude Haiku 5.5 各一份，另加一個並列的描述表（RQ1-X）。

    uv run python -m experiments.study3_report --agent local|haiku \\
        [--sweep-run runs/<sweep-1> --sweep-run runs/<sweep-2> ...] --out docs/results/study3 [--readme README.md] \\
        [--dry-run [--runs-root DIR]] [--replicates N]

- 分析一律呼叫 ``experiments.study3_analysis.analyze``（baseline 目錄由程式從 repo 的 runs/ 列舉；Haiku 的累計上限一律以登記的
  Study 3 總帳計算）；這裡不另外計算任何判定，只把 decision 036 §2、§6 事先寫定的措辭（三分法標籤、樣本少、機械性 cliff 的解讀句、
  資料不完整（提前結束）、因上限中途停止、天花板、RQ2 的「有效」）以分析的數字機械地填入（``conclusions``）。分析沒有給標籤的
  地方（提前結束、blocked 等）不印任何標籤。
- 輸出：``summary.json``（分析摘要與紀錄）、``README.md``（結果頁）、``recoverability.svg``（每條 sweep 過的 source 一格，依級別分段）。
- 報告屬於哪個 agent，由摘要 provenance 的 (suite, agent) 決定（``variant``）；config_id 只用來核對，不用來選擇：
  本機 → ``docs/results/study3/`` 與 README 的 ``study3`` 段落；Haiku → ``docs/results/study3-haiku/`` 與 ``study3-haiku`` 段落。
  給 ``--readme`` 時同時更新 ``study3-compare`` 段落（RQ1-X 並列表）；另一個 agent 的摘要讀自它自己的結果目錄，沒有時標示
  「尚未有正式結果」。新的段落放在 Study 2 的段落之後，依 study3、study3-haiku、study3-compare 的順序。
- 目錄的守衛（寫任何檔案之前）：拒絕封存的 docs/results 本身、Study 2 / decision 035 的結果目錄（或其子目錄）、另一個 agent 的
  Study 3 目錄、已有其他報告 summary.json 的目錄；dry-run 不寫進任何 Study 3 的結果目錄；--readme 不得位在 docs/results 之下；
  寫 README 的正式報告必須寫進自己的結果目錄（README 連到那裡）。反向的守衛在 ``study2_report``（``STUDY3_RESULT_DIRS``）。
- dry-run 的摘要標示「不是研究結果」，不能寫進 README；正式報告必須使用 B = 10,000、有凍結紀錄（decision 036 §9），而且分析的
  runs 根目錄是 repo 的 runs/（``study3_analysis.RUNS_ROOT``；從磁碟讀回的摘要也一樣）。
- **不做 Core Gate 的判定**，輸出中也不會出現「Advanced」（有測試）。
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import study2_analysis as s2
from experiments import study2_report as rep2
from experiments import study3_analysis as s3
from experiments.call_log import reconcile
from experiments.study3_dev import agent_config_id
from experiments.study3_stats import ALPHA, TASKS_PER_LEVEL
from replay.trajectory import Trajectory
from tasks.generator_v3 import EVAL_SEEDS, FAMILIES, LEVELS, task_id_for
from tasks.suites import STUDY3_PAID

REPORT_FORMAT = "agent-gym/study3-report/v1"
VARIANTS: dict[str, dict[str, str]] = {
    "study3": {
        "agent": "local", "start": "<!-- study3:start -->", "end": "<!-- study3:end -->",
        "heading": "### Study 3：更難的分級任務，本機模型（decision 036）",
        "page": "docs/results/study3/README.md",
    },
    "study3-haiku": {
        "agent": "haiku", "start": "<!-- study3-haiku:start -->", "end": "<!-- study3-haiku:end -->",
        "heading": "### Study 3：更難的分級任務，Claude Haiku 5.5（decision 036）",
        "page": "docs/results/study3-haiku/README.md",
    },
}
COMPARE = {"start": "<!-- study3-compare:start -->", "end": "<!-- study3-compare:end -->",
           "heading": "### Study 3：兩個 agent 並列（RQ1-X，只做描述）"}
RESULT_DIRS: dict[str, Path] = {name: rep2.ROUND1_RESULTS / Path(v["page"]).parent.name for name, v in VARIANTS.items()}
OF_AGENT = {v["agent"]: name for name, v in VARIANTS.items()}
AGENT_SHORT = {"local": "本機模型", "haiku": "Claude Haiku 5.5"}
INTRO = ("獨立的後續研究（generator v3，eval 96 題 × R = 3，只 sweep 失敗的 source）；第一輪、Study 2 與 decision 035 的結果與判定不變，"
         "本段也不做 Core Gate 的判定。")
COMPARE_NOTE = "兩個 agent 的失敗來自不同的題（預期本機主要在 H1 / H2，Haiku 主要在 H3），所以 cliff 比例的差異不能歸因於 agent。"
LABEL_TEXT = {"mostly_cliff": "以驟降為主", "mostly_gradual": "以逐步為主", "undetermined": "無法判斷"}
OUTCOME_SHORT = {**LABEL_TEXT, "no_decidable": "沒有可判定的曲線", "ceiling": "天花板", "incomplete": "資料不完整",
                 "incomplete_early_end": "資料不完整（提前結束）", "blocked": "不合併估計（response model 不只一種）"}
CAP_TEXT = "（因上限中途停止）"
MEANING = "「有效」只表示 H1 > H3，不代表差距大小。"
README_ANCHOR = "<!-- study2-haiku:end -->"  # 新的 Study 3 段落放在 Study 2 的段落之後（README 的「結果」一節）
BLOCK_ORDER = ("study3", "study3-haiku", "study3-compare")
_MARKERS = {**{name: (v["start"], v["end"]) for name, v in VARIANTS.items()}, "study3-compare": (COMPARE["start"], COMPARE["end"])}
_PARTS = {task_id_for(f, lv, s): (f, lv, s) for f in FAMILIES for lv in LEVELS for s in EVAL_SEEDS}
_REPORTED = (*s3.LABELS, "no_decidable", "incomplete_early_end")  # 有 k / n 可報告的 RQ1 結果


# ---------------------------------------------------------------- 歸屬與建立


def variant(summary: Mapping[str, Any]) -> str:
    """報告屬於哪個 agent：摘要 provenance 的 (suite, agent)。suite 不是 study3、agent 不明，或 config_id 不是該 agent 的凍結設定
    → ValueError（config_id 只核對，不單獨決定歸屬）。"""
    if not isinstance(summary, Mapping) or summary.get("format") != REPORT_FORMAT:
        raise ValueError(f"not a Study 3 report summary ({REPORT_FORMAT})")
    prov = summary["analysis"]["provenance"]
    if prov.get("suite") != s3.SUITE:
        raise ValueError(f"the summary's suite is {prov.get('suite')!r}, not study3; Study 2 and decision 035 reports belong to "
                         "experiments.study2_report")
    agent = prov.get("agent")
    if agent not in OF_AGENT:
        raise ValueError(f"unknown Study 3 agent {agent!r}")
    if prov.get("config_id") != agent_config_id(agent):
        raise ValueError(f"the summary's config_id is not the frozen config of the {agent} agent (decision 036 §1)")
    return OF_AGENT[agent]


def _records(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """結果頁「設定與紀錄」用的紀錄：模型（第一個 baseline trajectory 的設定）、呼叫的逐次核對、Haiku 的 manifest 花費。"""
    baseline_dirs = [Path(p) for p in analysis["baseline"]["paths"]]
    sweep_dirs = [Path(p) for p in analysis["sweep"]["paths"]]
    trajectories = sorted(p for d in baseline_dirs for p in (d / "trajectories").glob("*.json"))
    mc = Trajectory.load(trajectories[0]).agent.model_config if trajectories else {}
    out: dict[str, Any] = {"model": mc.get("model"), "model_digest": mc.get("model_digest"), "ollama_version": mc.get("ollama_version"),
                           "calls_baseline": reconcile(baseline_dirs), "calls_sweep": reconcile(sweep_dirs) if sweep_dirs else None}
    if analysis["provenance"]["agent"] == "haiku":
        out["spent_usd"] = {"baseline": rep2._spent(baseline_dirs), "sweep": rep2._spent(sweep_dirs)}
    return out


def build(agent: str, runs_root: str | Path | None = None, sweep_runs: Sequence[str | Path] = (), *, dry_run: bool = False,
          replicates: int = s2.BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    analysis = s3.analyze(agent, runs_root, sweep_runs, dry_run=dry_run, replicates=replicates)
    return {"format": REPORT_FORMAT, "analysis": analysis, "records": _records(analysis)}


# ---------------------------------------------------------------- 事先寫定的措辭（§2、§6）


def _label(w: Mapping[str, Any]) -> str:
    return LABEL_TEXT.get(w.get("label") or "", "—")


def _arrow(w: Mapping[str, Any]) -> str:
    """有標籤時「 → 標籤」；分析沒有給標籤（提前結束等，§6）→ 空字串。"""
    return f" → {_label(w)}" if w.get("label") else ""


def _kn_ci(w: Mapping[str, Any]) -> str:
    return f"{w['k']} / {w['n']}" + (f"，Wilson 95% CI [{w['low']:.3f}, {w['high']:.3f}]" if w["n"] else "")


def _rq1_bullets(a: Mapping[str, Any]) -> list[str]:
    r1, sw, sec = a["rq1"], a["sweep"], a["secondary"]
    outcome = r1["outcome"]
    found = (f"可判定的失敗曲線中 cliff {_kn_ci(r1)}（失敗的 source {r1['n_failed_sources']} 條、完成的曲線 {r1['n_curves']} 條；"
             f"無法判定 {r1['n_undecidable']} 條、max_turns 混淆 {r1['n_max_turns_confounded']} 條；cliff 中機械性 {r1['mechanical']}、"
             f"行為性 {r1['behavioral']}）")
    if outcome == "ceiling":
        errors = sum(a["baseline"]["levels"][lv]["agent_error"] for lv in LEVELS)
        head = f"天花板：沒有任何失敗的 source，而且 baseline 完整；不跑 sweep。agent_error 共 {errors} 個 episode（另外列出，不屬於 RQ1）。"
    elif outcome == "blocked":
        head = f"{s3.RQ1_TEXT['blocked']}。"
    elif outcome == "incomplete":
        reason = ("沒有失敗的 source，但 baseline 不完整" if r1["state"] == "incomplete_no_failed_source"
                  else f"有失敗的 source（{r1['n_failed_sources']} 條），但前綴中沒有完成的曲線")
        end = f"（sweep 的結束原因：{s3.END_TEXT[sw['end_reason']]}）" if sw["end_reason"] else ""
        head = f"資料不完整：{reason}{end}。"
    elif outcome == "incomplete_early_end":
        head = f"資料不完整（提前結束）：不輸出三分法的標籤。已完成的前綴中，{found}。"
    else:
        head = f"{s3.RQ1_TEXT[outcome]}{CAP_TEXT if r1['stopped_mid_source'] else ''}：{found}。"
    subs = []
    if r1["small_n"]:
        subs.append(f"**{r1['small_n']}。**")
    if r1["small_n_rule"]:
        subs.append(f"{r1['small_n_rule']}。")
    partial = None if sw["partial"] is None else sw["partial"]["task_id"]
    if r1["cap_range"]:
        x, y = r1["cap_range"]["as_cliff"], r1["cap_range"]["as_non_cliff"]
        subs.append(f"因上限中途停止的敏感度範圍（只針對中途停止的那一條 source {partial or '—'}）：當成 cliff {s3._ci(x)} → {_label(x)}；"
                    f"當成非 cliff {s3._ci(y)} → {_label(y)}。之後沒有開始的 source {len(sw['not_started'])} 條（與依規則停止相同，不列入範圍）。")
    if r1["early_end_range"]:
        e = r1["early_end_range"]
        subs.append(f"提前結束的範圍（順序中前 {sw['target']} 條裡沒有完成的 {e['uncompleted']} 條）：全部當成 cliff {s3._ci(e['all_cliff'])}；"
                    f"全部當成非 cliff {s3._ci(e['all_non_cliff'])}"
                    + (f"；中途停下的 {partial} 只列出已完成的點，不計入 RQ1" if partial else "") + "。")
    if sec is not None and sec["interpretation"]:  # blocked：沒有次要結果
        subs.append("".join(sec["interpretation"]))
    if r1["baseline_incomplete"]:
        subs.append("**baseline 不完整**：RQ1 照前綴規則計算並輸出判定；缺漏見下。")
    return [f"- RQ1：{head}", *[f"  - {s}" for s in subs]]


def _secondary_bullets(a: Mapping[str, Any]) -> list[str]:
    """§2 次要結果 1 與 4（第一輪的規則）；標籤只在分析有給時印出（提前結束時沒有，§6）。"""
    sec = a["secondary"]
    if sec is None:
        return []
    beh, share, r = sec["behavioral"], sec["behavioral_of_cliffs"], sec["round1_rule"]
    return [f"- 行為性 cliff 比例（次要結果，不取代主要判定）：{s3._ci(beh)}{_arrow(beh)}；(A) 版本（n = RQ1 的完整分母，只做描述）："
            f"{s3._ci(sec['version_a'])}；行為性 cliff 占全部 cliff：{share['k']} / {share['n']}。",
            f"- 第一輪的規則（decision 017 §1；次要結果）：{s3._ci(r)}{_arrow(r)}；無法判定 {r['n_undecidable']}、max_turns 混淆 "
            f"{r['n_max_turns_confounded']}。"]


def _gaps_bullet(b: Mapping[str, Any]) -> str:
    def items(pairs: Sequence[Sequence[Any]]) -> str:
        return "、".join(f"{t} r{i}" for t, i in pairs) or "—"

    return (f"- **baseline 不完整**：沒有採計 rollout 的 (task, rollout_index) {len(b['missing'])} 個：{items(b['missing'])}；"
            f"採計到 aborted 的 {len(b['aborted'])} 個：{items(b['aborted'])}。")


def _sweep_bullet(a: Mapping[str, Any]) -> str:
    sw = a["sweep"]
    end = f"（結束的原因：{s3.END_TEXT[sw['end_reason']]}）" if sw["end_reason"] else ""
    nd = sw["by_design_not_swept"]
    return (f"- sweep：{s3.STATE_TEXT[sw['state']]}{end}。完成 {len(sw['completed'])} / 目標 {sw['target']} 條（失敗的 source "
            f"{sw['n_failed']} 條）；做到一半：{'—' if sw['partial'] is None else sw['partial']['task_id']}；之後沒有開始："
            f"{len(sw['not_started'])} 條；依設計不 sweep：成功的 source {len(nd['successful_sources'])} 條、第 {sw['max_sources'] + 1} "
            f"條之後的失敗 source {len(nd['beyond_cap'])} 條。")


def _yes(value: bool) -> str:
    return "成立" if value else "不成立"


def _rq2_bullet(rq2: Mapping[str, Any]) -> str:
    """§2 RQ2：一律寫出差的點估計（與 CI，有計算時）、兩個檢定各自的結果，並註明「有效」的意思。"""
    point = s3._fmt(rq2["point"])
    check = rq2["check"]
    if check is None:
        return f"- 分級（RQ2）：無法計算（有級別不足 {TASKS_PER_LEVEL} 題，或不合併估計）；H1 − H3 的點估計 {point}。{MEANING}"
    b, p = check["bootstrap"], check["permutation"]
    verdict = "有效" if check["valid"] else "無效：不下難度的結論"
    return (f"- 分級（RQ2）：{verdict}{'（資料不完整）' if rq2['data_incomplete'] else ''}。H1 − H3 = {point}，bootstrap 95% CI "
            f"[{b['low']:.3f}, {b['high']:.3f}]（B = {b['replicates']}、seed {b['seed']}；2.5% 分位數 > 0：{_yes(b['low_positive'])}）；"
            f"精確置換檢定 p = {p['p']:.4g}（< {ALPHA}：{_yes(p['p'] < ALPHA)}）。兩者都成立才有效。{MEANING}")


def _rq3_bullet(a: Mapping[str, Any]) -> str:
    levels = a["rq3"]["levels"]
    parts = "、".join(f"{lv} {levels[lv]['success_unsafe_among_successes']['k']} / {levels[lv]['success_unsafe_among_successes']['n']}"
                     for lv in LEVELS)
    return f"- RQ3（安全）：成功中不安全的 episode {parts}；只錯在順序、最終狀態正確的 trajectory 是 success，計為 RQ3 的事件。"


def conclusions(summary: Mapping[str, Any]) -> list[str]:
    """結果頁與 README 共用的結論（markdown 的項目符號行）；措辭依 decision 036 §2、§6 事先寫定，數字機械填入。"""
    variant(summary)
    a = summary["analysis"]
    out = _rq1_bullets(a)
    if a["rq1"]["outcome"] in _REPORTED:
        out += _secondary_bullets(a)
    if not a["baseline"]["complete"]:
        out.append(_gaps_bullet(a["baseline"]))
    return [*out, _sweep_bullet(a), _rq2_bullet(a["baseline"]["rq2"]), _rq3_bullet(a)]


def _flags(a: Mapping[str, Any]) -> str:
    return (("**資料不完整。**" if a["rq1"]["outcome"] in ("incomplete", "incomplete_early_end") else "")
            + ("" if a["baseline"]["complete"] else "**baseline 不完整。**"))


# ---------------------------------------------------------------- 結果頁


def _composition(a: Mapping[str, Any]) -> list[str]:
    x = s3.rq1x_inputs(a)
    lines = ["## 分母的組成（RQ1-X 的輸入，只做描述）", "",
             "| 組 | 失敗的 source | 完成的曲線 | 可判定 | 無法判定 | max_turns 混淆 | cliff |", "|---|---|---|---|---|---|---|"]
    for key, groups in (("by_level", LEVELS), ("by_family", FAMILIES)):
        for group in groups:
            c = x[key][group]
            lines.append(f"| {group} | {c['failed_sources']} | {c['curves']} | {c['decidable']} | {c['undecidable']} | {c['confounded']} | "
                         f"{c['cliffs']} |")
    return [*lines, "", COMPARE_NOTE, ""]


# dev 審查（eval 之前）判定不是缺陷、但寫進報告的候選疑慮（decision 036「實作紀錄」，034 §5 C）；固定文字，兩個 agent 都列出
LIMITATIONS = (
    "本機 agent 每個 episode 的第一個 search 都用 limit 100，被 invalid_arguments 拒絕後改用 50，每個 episode 少一個 turn"
    "（上限 50 寫在 tool 的 input_schema；dev 中 Haiku 沒有這樣做）。",
    "epic-closeout 的 dev trajectory 都是逐一 get_issue 的讀法：H3 done 分支需要 21 個 call（結構上做不完）、blocked 分支恰好 20 個"
    "（§3 (c) 登記的已知限制）。epic-closeout H3 的失敗與依 decisive_rules 列出的歸屬，可能主要由讀法決定。",
    "verified-close：instruction 的範例 user id 可能恰好是 qa-broken 上 QA comment 的作者，「照抄範例」與「把 QA 作者誤認為 reporter」"
    "兩種錯誤無法區分（歸屬混淆；grader 對兩者都判為失敗是正確的）。",
    "sync-repair rule 2 的英文句子較拗口；只有一種合理讀法（反向讀法會使 rule 1 失去作用）。",
    "scoped-triage 的「S7 相關」一行會機械地計入 label 幻覺造成的失敗，是上界（附錄 A.5）。",
    "「不要改變 status」只以最終狀態與 forbid（只禁止 closed）判定，暫時改動再改回不會被判為失敗（全專案的慣例）。",
)


def _limitations() -> list[str]:
    return ["## 已知限制（dev 審查的候選疑慮；不是題目缺陷，不改變評分）", "",
            "以下在 eval 之前的 dev 審查中判定不是題目缺陷，依 034 §5 C 寫進報告；題目、tool 說明與 grader 都沒有修改。"
            "判定的依據見 decision 036「實作紀錄」。", "",
            *(f"{i}. {text}" for i, text in enumerate(LIMITATIONS, 1)), ""]


def _settings(summary: Mapping[str, Any]) -> list[str]:
    a, rec = summary["analysis"], summary["records"]
    prov, b, sw = a["provenance"], a["baseline"], a["sweep"]
    if prov["agent"] == "haiku":
        model = [f"- 模型：`{s3._fmt(rec['model'])}`（Anthropic API；凍結設定 decision 035）"]
        spent = rec.get("spent_usd")
        if spent:
            model.append(f"- 列入的目錄的實際花費（manifest 合計）：eval baseline ${spent['baseline']:.4f}、sweep ${spent['sweep']:.4f}，合計 "
                         f"${spent['baseline'] + spent['sweep']:.4f}。全部付費呼叫（含 dev、中止與中斷的嘗試）以 Study 3 總帳為準"
                         f"（硬上限 ${STUDY3_PAID.total_usd:g}，decision 036 §7）")
    else:
        model = [f"- 模型：`{s3._fmt(rec['model'])}`，digest `{s3._fmt(rec['model_digest'])}`，Ollama {s3._fmt(rec['ollama_version'])}"
                 "（凍結設定 decision 030）"]
    plan = prov["plan"]
    plan_text = "—" if plan is None else f"({plan['coarse_stride']}, {plan['n_coarse']}, {plan['n_fine']}, {plan['drop_threshold']})"
    lines = ["## 設定與紀錄", "", *model,
             f"- config_id：`{prov['config_id']}`",
             "- 凍結紀錄：" + (f"`{prov['freeze_record_sha256']}`" if prov["freeze_record_sha256"] else "沒有"),
             f"- runs 根目錄：`{prov['runs_root']}`；列入的 eval baseline 目錄：{', '.join(b['runs']) or '—'}；sweep 目錄：{', '.join(sw['runs']) or '—'}",
             f"- 主要規則 {prov['primary_rule']}；判定 {s3._fmt(prov['criterion'])}；plan {plan_text}；bootstrap B = {prov['replicates']}；"
             f"R = {prov['rollouts_per_task']}"]
    for key, label in (("calls_baseline", "baseline"), ("calls_sweep", "sweep")):
        calls = rec[key]
        if calls:
            lines.append(f"- {label} 的模型呼叫：{calls['calls']} 次（失敗 {calls['failed']}、有 seed {calls['seeds']}）；逐次嘗試核對："
                         f"{'完整' if calls['complete'] else '不完整'}")
    return lines


def render_page(summary: Mapping[str, Any]) -> str:
    variant(summary)
    a = summary["analysis"]
    prov = a["provenance"]
    lines = [f"# Study 3：{s3.AGENT_TEXT[prov['agent']]}（decision 036）", ""]
    if prov["dry_run"]:
        lines += ["> **dry-run 的輸出（oracle 假 client）：不是研究結果，不能寫進 README。**", ""]
    if not a["baseline"]["complete"]:
        lines += ["> **baseline 不完整**（缺漏見「結論」）。", ""]
    lines += ["本報告是 Study 3 的事先登記分析；不做 Core Gate 的判定。結論的措辭依 decision 036 §2、§6 事先寫定，數字由分析程式機械填入。",
              "", "## 結論", "", *conclusions(summary), "",
              "分級檢定（RQ2）的偵測力見凍結紀錄（decision 036 §2、§9；只用於報告，不改變判定）。", ""]
    body = s3.render_markdown(a)
    # 分析的 markdown 從第一個二級標題開始（標題與標示已在上面）；分析摘要在這裡是 summary.json 的 "analysis"
    lines += [body[body.index("\n## ") + 1:].replace("analysis.json", "summary.json").rstrip("\n"), ""]
    lines += [*_composition(a), *_limitations(), *_settings(summary), "", "曲線圖：[recoverability.svg](recoverability.svg)。數字：[summary.json](summary.json)。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 曲線圖


def _panel(kind: str, item: Mapping[str, Any], x0: float, y0: float, w: float, h: float) -> list[str]:
    px0, px1, py0, py1 = x0 + 26, x0 + w - 8, y0 + 18, y0 + h - 18
    family, _, seed = _PARTS[item["task_id"]]
    if kind == "curve":
        label = f"{family} s{seed}（{s2.CLASS_TEXT[item['class']]}）"
        points = [(e["k"], e["successes"], e["n"], e["agent_errors"], e["max_turns"]) for e in item["estimates"] if e["n"]]
        span = max(item["n_turns"] - 1, 1)
    else:
        label = f"{family} s{seed}（未完成，不計入 RQ1）"
        points = [(p["k"], p["successes"], p["n"], p["agent_errors"], None) for p in item["points"] if p["n"]]
        span = max([k for k, *_ in points] + [1])
    out = ['<g class="panel">', f'<text class="t2" x="{px0}" y="{y0 + 11}">{html.escape(label)}</text>']
    for level in (0.0, 1.0):
        yy = py1 - (py1 - py0) * level
        out.append(f'<line class="grid" x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}"/>')
        out.append(f'<text class="t3" x="{px0 - 4}" y="{yy + 3:.1f}" text-anchor="end">{level:g}</text>')

    def sx(k: float) -> float:
        return px0 + (px1 - px0) * k / span

    def sy(p: float) -> float:
        return py1 - (py1 - py0) * p

    if kind == "curve":
        if item["class"].startswith("cliff") and item["largest_drop_k"] is not None and item["next_k"] is not None:
            xk = sx((item["largest_drop_k"] + item["next_k"]) / 2)
            out.append(f'<line class="cliff" x1="{xk:.1f}" y1="{py0}" x2="{xk:.1f}" y2="{py1}"/>')
        if len(points) > 1:
            out.append('<path class="line" d="' + " ".join(f'{"M" if i == 0 else "L"}{sx(k):.1f},{sy(s / n):.1f}'
                                                          for i, (k, s, n, *_) in enumerate(points)) + '"/>')
        for point in item["success_and_safe"]:
            if point["n"]:
                out.append(f'<circle class="safe" cx="{sx(point["k"]):.1f}" cy="{sy(point["success_and_safe"] / point["n"]):.1f}" r="3"/>')
    for k, s, n, errors, max_turns in points:
        tip = f"k = {k}：{s}/{n} 成功（agent_error {errors}" + ("" if max_turns is None else f"、max_turns {max_turns}") + "）"
        out.append(f'<circle class="{"dot" if kind == "curve" else "pending"}" cx="{sx(k):.1f}" cy="{sy(s / n):.1f}" r="4">'
                   f'<title>{html.escape(tip)}</title></circle>')
    return [*out, "</g>"]


def recoverability_svg(summary: Mapping[str, Any]) -> str:
    """每條 sweep 過的 source 一格（完成的前綴，加上做到一半的那一條），依級別分段、段內依凍結順序。實心點 = P(success | k)，
    空心點 = P(success ∧ safe | k)；做到一半的 source 只畫已完成的點（灰色），不計入 RQ1。"""
    name = variant(summary)
    a = summary["analysis"]
    sw, r1, curves = a["sweep"], a["rq1"], a["curves"]
    cols, panel_w, panel_h, gap, left, top = 4, 220, 104, 14, 16, 100
    width = left * 2 + cols * (panel_w + gap)
    panels: dict[str, list[tuple[str, Mapping[str, Any]]]] = {lv: [("curve", c) for c in curves if c["level"] == lv] for lv in LEVELS}
    if sw["partial"] is not None:
        panels[_PARTS[sw["partial"]["task_id"]][1]].append(("partial", sw["partial"]))
    height = top + sum(22 + (math.ceil(len(p) / cols) * (panel_h + gap) if p else 24) for p in panels.values()) + 16
    title = (("[dry-run] " if a["provenance"]["dry_run"] else "") + f"Study 3 recoverability（{AGENT_SHORT[VARIANTS[name]['agent']]}）"
             "：P(success | k)")
    verdict = OUTCOME_SHORT[r1["outcome"]] + (CAP_TEXT if r1["stopped_mid_source"] else "")
    if r1["outcome"] in _REPORTED:
        verdict += f"；cliff {r1['k']} / {r1['n']}" + (f"；{r1['small_n']}" if r1["small_n"] else "")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        'aria-labelledby="t d" font-family="-apple-system, BlinkMacSystemFont, \'Noto Sans TC\', \'Segoe UI\', sans-serif">',
        f'<title id="t">{html.escape(title)}</title>',
        f'<desc id="d">{len(curves)} 條完成的曲線（只 sweep 失敗的 source），依級別分段、段內依凍結順序。實心點：P(success | k)；'
        f'空心點：P(success ∧ safe | k)。RQ1：{html.escape(verdict)}。數值見 summary.json。</desc>',
        "<style>",
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e4e3df;--series:#2a78d6;}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e88;--grid:#383835;"
        "--series:#3987e5;}}",
        ".bg{fill:var(--surface)}.t1{fill:var(--ink);font-size:15px;font-weight:600}.t2{fill:var(--ink2);font-size:11px}"
        ".t3{fill:var(--muted);font-size:10px}.grid{stroke:var(--grid);stroke-width:1}.line{fill:none;stroke:var(--series);"
        "stroke-width:2}.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}.safe{fill:var(--surface);stroke:var(--series);"
        "stroke-width:1.5}.pending{fill:var(--muted);stroke:var(--surface);stroke-width:2}.cliff{stroke:var(--ink2);stroke-width:1.5;"
        "stroke-dasharray:4 3}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
        f'<text class="t1" x="{left}" y="26">{html.escape(title)}</text>',
        f'<text class="t2" x="{left}" y="44">只 sweep 失敗的 source（固定順序的前綴），依級別分段、段內依凍結順序。x：checkpoint k；y：0–1。</text>',
        f'<text class="t2" x="{left}" y="60">實心點 P(success | k)；空心點 P(success ∧ safe | k)；虛線：cliff；灰點：做到一半的 source'
        '（不計入 RQ1）。</text>',
        f'<text class="t2" x="{left}" y="76">RQ1：{html.escape(verdict)}'
        + ("" if a["baseline"]["complete"] else "（baseline 不完整）") + "。</text>",
    ]
    y = top
    for lv in LEVELS:
        n_curves = sum(1 for kind, _ in panels[lv] if kind == "curve")
        out.append(f'<text class="t1" x="{left}" y="{y + 14}">{lv}（失敗的 source {a["sources"]["by_level"][lv]["n_failed"]} 條，'
                   f'完成的曲線 {n_curves} 條）</text>')
        y += 22
        if not panels[lv]:
            out.append(f'<text class="t3" x="{left}" y="{y + 12}">沒有完成的曲線</text>')
            y += 24
            continue
        for i, (kind, item) in enumerate(panels[lv]):
            out += _panel(kind, item, left + (i % cols) * (panel_w + gap), y + (i // cols) * (panel_h + gap), panel_w, panel_h)
        y += math.ceil(len(panels[lv]) / cols) * (panel_h + gap)
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- README


def render_readme_block(summary: Mapping[str, Any]) -> str:
    name = variant(summary)
    a = summary["analysis"]
    if a["provenance"]["dry_run"]:
        raise ValueError("dry-run outputs are not results and never go into the README")
    v = VARIANTS[name]
    lines = [v["start"], "", v["heading"], "", INTRO + _flags(a), "",
             "| 級別 | success | 成功但不安全 | agent_error | 以 max_turns 結束 |", "|---|---|---|---|---|"]
    for lv in LEVELS:
        m = a["baseline"]["levels"][lv]
        lines.append(f"| {lv} | {s3._fmt(m['success_rate'])} | {s3._fmt(m['success_unsafe_rate'])} | {m['agent_error']} | "
                     f"{s3._fmt(m['max_turns_rate'])} |")
    return "\n".join([*lines, "", *conclusions(summary), "", f"完整結果：[{v['page']}]({v['page']})。", "", v["end"]])


def _verdict_cell(summary: Mapping[str, Any] | None) -> str:
    if summary is None:
        return "尚未有正式結果"
    r1 = summary["analysis"]["rq1"]
    return (OUTCOME_SHORT[r1["outcome"]] + (CAP_TEXT if r1["stopped_mid_source"] else "")
            + (f"；{r1['small_n']}" if r1["small_n"] else "") + ("；baseline 不完整" if r1["baseline_incomplete"] else ""))


def _ratio_cell(summary: Mapping[str, Any] | None) -> str:
    if summary is None or summary["analysis"]["rq1"]["outcome"] not in _REPORTED:
        return "—"
    r1 = summary["analysis"]["rq1"]
    return f"{r1['k']} / {r1['n']}" + (f"（Wilson 95% CI [{r1['low']:.3f}, {r1['high']:.3f}]）" if r1["n"] else "")


def render_compare_block(local: Mapping[str, Any] | None, haiku: Mapping[str, Any] | None) -> str:
    """RQ1-X（§2，只做描述）：兩個 agent 的判定與 cliff 比例並列，依級別與題型列出分母的組成，並註明差異不能歸因於 agent。
    欄位依 (suite, agent)；沒有的一方標示「尚未有正式結果」；dry-run 或兩份凍結紀錄不同 → ValueError。"""
    sides = {"local": local, "haiku": haiku}
    for agent, summary in sides.items():
        if summary is None:
            continue
        name = variant(summary)
        if VARIANTS[name]["agent"] != agent:
            raise ValueError(f"the {agent} column got the {name} summary; the columns follow (suite, agent)")
        if summary["analysis"]["provenance"]["dry_run"]:
            raise ValueError("dry-run outputs are not results and never go into the README")
    present = [s for s in sides.values() if s is not None]
    if not present:
        raise ValueError("the RQ1-X table needs at least one agent's formal summary")
    if len({s["analysis"]["provenance"]["freeze_record_sha256"] for s in present}) > 1:
        raise ValueError("the two summaries come from different freeze records; both agents share one (decision 036 §9)")
    x = {agent: None if s is None else s3.rq1x_inputs(s["analysis"]) for agent, s in sides.items()}
    lines = [COMPARE["start"], "", COMPARE["heading"], "",
             f"**{COMPARE_NOTE}** 不合併資料，也不做 agent 之間的檢定（decision 009 §4）；各自的判定與措辭見各自的段落。", "",
             f"| | {AGENT_SHORT['local']} | {AGENT_SHORT['haiku']} |", "|---|---|---|",
             f"| RQ1 的判定 | {_verdict_cell(local)} | {_verdict_cell(haiku)} |",
             f"| cliff 比例（可判定的失敗曲線） | {_ratio_cell(local)} | {_ratio_cell(haiku)} |", "",
             "分母的組成（可判定 = 不是無法判定、也不是 max_turns 混淆；cliff 為可判定曲線中的條數）：", "",
             "| 組 | agent | 失敗的 source | 完成的曲線 | 可判定 | 無法判定 | max_turns 混淆 | cliff |", "|---|---|---|---|---|---|---|---|"]
    for key, groups in (("by_level", LEVELS), ("by_family", FAMILIES)):
        for group in groups:
            for agent in ("local", "haiku"):
                if x[agent] is None:
                    lines.append(f"| {group} | {AGENT_SHORT[agent]} | — | — | — | — | — | — |")
                    continue
                c = x[agent][key][group]
                lines.append(f"| {group} | {AGENT_SHORT[agent]} | {c['failed_sources']} | {c['curves']} | {c['decidable']} | "
                             f"{c['undecidable']} | {c['confounded']} | {c['cliffs']} |")
    return "\n".join([*lines, "", COMPARE["end"]])


def embed_block(readme: str, block: str) -> str:
    """已有該段落的標記 → 取代（``study2_report.embed``：剛好一對）；沒有 → 放在依 BLOCK_ORDER 前一個已存在的 Study 3 段落之後，
    都沒有時放在 Study 2 段落（README_ANCHOR）之後；連它也沒有 → 加在最後。"""
    start, end = block.split("\n")[0], block.split("\n")[-1]
    if start in readme or end in readme:
        return rep2.embed(readme, block)
    name = next(n for n, markers in _MARKERS.items() if markers == (start, end))
    anchors = [_MARKERS[n][1] for n in reversed(BLOCK_ORDER[: BLOCK_ORDER.index(name)])] + [README_ANCHOR]
    for anchor in anchors:
        if readme.count(anchor) == 1:
            at = readme.index(anchor) + len(anchor)
            return readme[:at] + "\n\n" + block + readme[at:]
    return rep2.embed(readme, block)


# ---------------------------------------------------------------- 寫入


def _check_out(out: Path, own: str, *, dry_run: bool) -> None:
    """目錄的守衛（模組 docstring）；在寫任何檔案之前。"""
    if rep2._is_round1_path(out):
        raise ValueError("docs/results holds the sealed round-1 results; write Study 3 to docs/results/study3 or study3-haiku "
                         "(decision 036 §6)")
    resolved = out.resolve()
    chain = (resolved, *resolved.parents)
    for name, directory in rep2.RESULT_DIRS.items():
        if any(rep2._same_dir(p, directory) for p in chain):
            raise ValueError(f"{directory} holds the {name} report (Study 2 / decision 035); a Study 3 report never goes there "
                             "(decision 036 §6)")
    for name, directory in RESULT_DIRS.items():
        if any(rep2._same_dir(p, directory) for p in chain):
            if name != own:
                raise ValueError(f"{directory} holds the {name} report; this report is {own} (write it to {RESULT_DIRS[own]})")
            if dry_run:
                raise ValueError(f"dry-run outputs are not results and never go into {directory} (decision 036 §6)")
    existing = out / "summary.json"
    if existing.exists():
        try:
            same = variant(json.loads(existing.read_text(encoding="utf-8"))) == own
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            same = False
        if not same:
            raise ValueError(f"{existing} is not a {own} report summary; it is not overwritten (decision 036 §6)")


def _load(path: Path, expected: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} cannot be read") from exc
    if variant(data) != expected:
        raise ValueError(f"{path} is not the {expected} summary")
    return data


def write(summary: Mapping[str, Any], out: str | Path, *, readme: str | Path | None = None) -> dict[str, Any]:
    """把一份摘要寫成報告（所有檢查在寫任何檔案之前完成）。readme：同時更新該 agent 的段落與 study3-compare 段落。"""
    own = variant(summary)
    out = Path(out)
    prov = summary["analysis"]["provenance"]
    _check_out(out, own, dry_run=prov["dry_run"])
    if not prov["dry_run"]:
        if prov["replicates"] != s2.BOOTSTRAP_REPLICATES:
            raise ValueError(f"a formal Study 3 report uses B = {s2.BOOTSTRAP_REPLICATES} (decision 036 §2)")
        if not prov["freeze_record_sha256"]:
            raise ValueError("a formal Study 3 report needs the freeze record (decision 036 §9)")
        if not s3.is_runs_root(prov["runs_root"]):
            raise ValueError(f"a formal Study 3 report enumerates every directory under {s3.RUNS_ROOT} (decision 036 §6); this summary "
                             f"was analysed under {prov['runs_root']}")
    new_readme = None
    if readme is not None:
        path = Path(readme)
        if rep2._under_round1(path.parent):
            raise ValueError("the README of the sealed round-1 results is never modified (decision 034 §1)")
        block = render_readme_block(summary)
        if not rep2._same_dir(out, RESULT_DIRS[own]):
            raise ValueError(f"the README links to {VARIANTS[own]['page']}: write the {own} report to {RESULT_DIRS[own]} (decision 036 §6)")
        if not path.is_file():
            raise ValueError(f"{path} does not exist")
        sides: dict[str, Mapping[str, Any]] = {VARIANTS[own]["agent"]: summary}
        for name, directory in RESULT_DIRS.items():
            if name != own and (directory / "summary.json").is_file():
                sides[VARIANTS[name]["agent"]] = _load(directory / "summary.json", name)
        compare = render_compare_block(sides.get("local"), sides.get("haiku"))
        new_readme = embed_block(embed_block(path.read_text(encoding="utf-8"), block), compare)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "README.md").write_text(render_page(summary), encoding="utf-8")
    (out / "recoverability.svg").write_text(recoverability_svg(summary), encoding="utf-8")
    if new_readme is not None:
        Path(readme).write_text(new_readme, encoding="utf-8")  # type: ignore[arg-type]
    return dict(summary)


def write_report(agent: str, runs_root: str | Path | None, sweep_runs: Sequence[str | Path], out: str | Path, *,
                 readme: str | Path | None = None, dry_run: bool = False, replicates: int = s2.BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    if agent not in OF_AGENT:
        raise ValueError(f"agent must be one of {tuple(OF_AGENT)}")
    _check_out(Path(out), OF_AGENT[agent], dry_run=dry_run)  # 便宜的檢查在分析之前
    return write(build(agent, runs_root, sweep_runs, dry_run=dry_run, replicates=replicates), out, readme=readme)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=tuple(OF_AGENT), required=True)
    parser.add_argument("--runs-root", default=None, help="只供 --dry-run：列舉 baseline 目錄的根目錄（正式報告一律是 repo 的 runs/，"
                                                         "decision 036 §6）")
    parser.add_argument("--sweep-run", action="append", default=[], help="sweep 目錄（每一段都要給；可重複）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--readme", default=None)
    parser.add_argument("--dry-run", action="store_true", help="dry-run 的輸出（只供測試與演練；不是研究結果，不能寫進 README）")
    parser.add_argument("--replicates", type=int, default=s2.BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)
    try:
        write_report(args.agent, args.runs_root, args.sweep_run, args.out, readme=args.readme, dry_run=args.dry_run,
                     replicates=args.replicates)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"wrote {args.out}" + (f" and {args.readme}" if args.readme else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
