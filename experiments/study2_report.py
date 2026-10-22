"""Study 2 的結果報告（decision 034 §1）：``docs/results/study2/`` 與 README 中 ``<!-- study2:start -->`` / ``<!-- study2:end -->`` 之間的段落。
Claude Haiku 5.5 的跨模型比較（decision 035）寫到 ``docs/results/study2-haiku/`` 與 ``<!-- study2-haiku:start -->`` 段落（``VARIANTS``）。

    uv run python -m experiments.study2_report --baseline-run runs/<eval> [--baseline-run runs/<eval-rerun> ...] \\
        [--sweep-run runs/<sweep-1> --sweep-run runs/<sweep-2> ...] --out docs/results/study2 [--readme README.md]

每個旗標只接一個目錄，可重複給。

- 分析一律呼叫 ``experiments.study2_analysis.analyze``（事先登記的估計量與結論規則）；這裡不另外計算任何判定。
- 輸出：``summary.json``（分析結果與溯源）、``README.md``（結果頁）、``recoverability.svg``（每個 source 一格，依級別分段）。
- **不做 Core Gate 的判定**，輸出中也不會出現「Advanced」（有測試）。
- 第一輪封存：拒絕寫入 repo 的 ``docs/results/`` 本身（第一輪的五個檔案在那裡），也拒絕把 ``--readme`` 指到 ``docs/results/`` 之下；
  比較時不分大小寫（macOS 的檔案系統不分大小寫），也不依賴目前的工作目錄。README 只改 study2 的段落。
- baseline 或 sweep 任一是 dry-run，報告就標示「不是研究結果」，而且不能寫進 README。資料不完整時照常寫入，並標示「資料不完整」。
- 所有檢查（含 README 的標記）都在寫任何檔案之前完成；正式結果必須使用 B = 10,000。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from agents.baseline import ANTHROPIC_POLICY, haiku_agent_config
from agents.ollama import local_agent_config
from experiments.call_log import reconcile
from experiments.study2_analysis import BOOTSTRAP_REPLICATES, CLASS_TEXT, RQ1_TEXT, SKIP_TEXT, Study2Analysis, analyze, render_markdown
from replay.aborts import read_manifest
from replay.trajectory import Trajectory
from tasks.generator_v2 import EVAL_SEEDS, FAMILIES, LEVELS

README_START = "<!-- study2:start -->"
README_END = "<!-- study2:end -->"
ROUND1_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"

# 兩個 agent 的報告：本機模型（decision 034）與 Claude Haiku 5.5（decision 035）各有自己的目錄與 README 段落。
VARIANTS: dict[str, dict[str, str]] = {
    "study2": {
        "start": README_START, "end": README_END,
        "heading": "### Study 2：分級的多步驟任務（decision 034）",
        "intro": "獨立的後續研究；第一輪的結果與判定不變，本段也不做 Core Gate 的判定。",
        "page": "docs/results/study2/README.md",
    },
    "study2-haiku": {
        "start": "<!-- study2-haiku:start -->", "end": "<!-- study2-haiku:end -->",
        "heading": "### Study 2 跨模型比較：Claude Haiku 5.5（decision 035）",
        "intro": "與 Study 2 相同的 36 題與協定，agent 換成 Claude Haiku 5.5；第一輪與 Study 2 的結果與判定不變，本段也不做 Core Gate 的判定。",
        "page": "docs/results/study2-haiku/README.md",
    },
}
RESULT_DIRS: dict[str, Path] = {name: ROUND1_RESULTS / Path(v["page"]).parent.name for name, v in VARIANTS.items()}


def variant(summary: dict[str, Any]) -> str:
    """報告屬於哪個 agent：provenance 的 config_id（baseline 目錄 run.json 的設定）是凍結的本機 agent（decision 034）→ "study2"；
    是 decision 035 的 Haiku agent → "study2-haiku"；其他 → ValueError（不猜測，不寫進任何一份報告）。"""
    config_id = summary["provenance"].get("config_id")
    if config_id == local_agent_config().config_id:
        return "study2"
    if config_id == haiku_agent_config().config_id:
        return "study2-haiku"
    raise ValueError(f"the baseline agent config {config_id!r} is neither the frozen local agent (decision 034) nor the Haiku "
                     "agent (decision 035)")


def _spent(run_dirs: Sequence[Path]) -> float:
    return math.fsum(row.cost_usd for d in run_dirs for row in read_manifest(d / "manifest.jsonl"))


def _run_config_id(baseline_dirs: Sequence[Path]) -> str | None:
    """baseline 目錄 run.json 的 config_id（每個目錄都有，不依賴是否有完成的 trajectory）；不只一種 → ValueError。"""
    ids = set()
    for directory in baseline_dirs:
        try:
            ids.add(json.loads((directory / "run.json").read_text(encoding="utf-8")).get("config_id"))
        except (OSError, ValueError, AttributeError):
            ids.add(None)
    if len(ids) > 1:
        raise ValueError("the baseline directories were produced by different agent configs")
    return next(iter(ids), None)


def _provenance(baseline_dirs: Sequence[Path], sweep_dirs: Sequence[Path], analysis: Study2Analysis) -> dict[str, Any]:
    trajectories = sorted(p for d in baseline_dirs for p in (d / "trajectories").glob("*.json"))
    agent = Trajectory.load(trajectories[0]).agent if trajectories else None
    mc = {} if agent is None else agent.model_config
    run_config_id = _run_config_id(baseline_dirs)
    return {
        "baseline_runs": [d.name for d in baseline_dirs],
        "sweep_runs": [d.name for d in sweep_dirs],
        "dry_run": analysis.dry_run,  # baseline 或 sweep 任一是 dry-run
        "data_complete": analysis.data_complete,
        "replicates": None if analysis.baseline["level_check"]["l1_minus_l3"] is None
        else analysis.baseline["level_check"]["l1_minus_l3"]["replicates"],
        "config_id": run_config_id if agent is None else agent.config_id,
        "model": mc.get("model"),
        "model_digest": mc.get("model_digest"),
        "ollama_version": mc.get("ollama_version"),
        "calls_baseline": reconcile(list(baseline_dirs)),
        "calls_sweep": reconcile(list(sweep_dirs)) if sweep_dirs else None,
        # 付費的 agent（decision 035）才有的欄位；本機模型的 summary 不變
        **({"provider": "anthropic", "spent_usd": {"baseline": _spent(baseline_dirs), "sweep": _spent(sweep_dirs)}}
           if (agent is not None and agent.policy == ANTHROPIC_POLICY) or run_config_id == haiku_agent_config().config_id else {}),
    }


def build(baseline_runs: Sequence[str | Path], sweep_runs: Sequence[str | Path] = (), *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    baseline_dirs, sweep_dirs = [Path(p) for p in baseline_runs], [Path(p) for p in sweep_runs]
    analysis = analyze(baseline_dirs, sweep_dirs, replicates=replicates)
    return {"format": "agent-gym/study2-report/v1", "provenance": _provenance(baseline_dirs, sweep_dirs, analysis),
            "analysis": analysis.to_dict(), "markdown": render_markdown(analysis)}


def _data_complete(summary: dict[str, Any]) -> bool:
    return bool(summary["provenance"]["data_complete"])


def render_page(summary: dict[str, Any]) -> str:
    prov = summary["provenance"]
    markdown = summary["markdown"].rstrip("\n")  # dry-run 與資料不完整的標示在分析的 markdown 中
    if variant(summary) == "study2-haiku":
        first, _, rest = markdown.partition("\n")
        markdown = "# Study 2 跨模型比較：Claude Haiku 5.5（decision 035；分析規則同 decision 034）\n" + rest
        spent = prov["spent_usd"]
        model_lines = ["- 模型：`claude-haiku-5-5`（Anthropic API；agent 設定同 decision 009，只換模型）",
                       f"- 列出的目錄的實際花費（manifest 合計）：eval baseline ${spent['baseline']:.4f}、sweep ${spent['sweep']:.4f}，"
                       f"合計 ${spent['baseline'] + spent['sweep']:.4f}。全部付費呼叫（含中止、中斷與沒有列出的目錄）以總帳為準，"
                       "見 decision 035（總預算 $70）"]
    else:
        model_lines = [f"- 模型：`{prov['model']}`，digest `{prov['model_digest']}`，Ollama {prov['ollama_version']}"]
    lines = [markdown]
    lines += ["", "## 設定與紀錄", "", *model_lines,
              f"- config_id：`{prov['config_id']}`",
              f"- baseline 目錄：{', '.join(prov['baseline_runs'])}；sweep 目錄：{', '.join(prov['sweep_runs']) or '—'}"]
    for name in ("calls_baseline", "calls_sweep"):
        calls = prov[name]
        if calls:
            lines.append(f"- {name.split('_')[1]} 的模型呼叫：{calls['calls']} 次（失敗 {calls['failed']}、有 seed {calls['seeds']}）；"
                         f"逐次嘗試核對：{'完整' if calls['complete'] else '不完整'}")
    lines += ["", "曲線圖：[recoverability.svg](recoverability.svg)。數字：[summary.json](summary.json)。", ""]
    return "\n".join(lines)


def recoverability_svg(summary: dict[str, Any]) -> str:
    """每個 source 一格：依級別分段，段內每列一個題型、每欄一個 seed。單一序列：點 = P(success | k)，空心點 = P(success ∧ safe | k)。"""
    sweep = summary["analysis"]["sweep"]
    curves = [] if sweep is None else sweep["curves"]
    seeds = list(EVAL_SEEDS)  # 固定的欄位：未完成的 seed 區塊也照實畫出（格子標示「沒有曲線」）
    panel_w, panel_h, gap = 220, 104, 14
    left, top = 16, 76
    width = max(left * 2 + len(seeds) * (panel_w + gap), 720)
    rows = [(lv, f) for lv in LEVELS for f in FAMILIES]
    height = top + len(rows) * (panel_h + gap) + len(LEVELS) * 22 + 30
    dry = summary["provenance"]["dry_run"]
    title = ("[dry-run] " if dry else "") + "Study 2 recoverability" + (
        "（Claude Haiku 5.5）" if variant(summary) == "study2-haiku" else "") + "：P(success | k)"
    rq1 = "沒有 sweep" if sweep is None else RQ1_TEXT[sweep["rq1"]["outcome"]]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        'aria-labelledby="t d" font-family="-apple-system, BlinkMacSystemFont, \'Noto Sans TC\', \'Segoe UI\', sans-serif">',
        f'<title id="t">{html.escape(title)}</title>',
        f'<desc id="d">{len(curves)} 個 source，依級別分段、每列一個題型、每欄一個 seed。實心點：P(success | k)；空心點：P(success ∧ safe | k)。'
        f'RQ1：{html.escape(rq1)}。數值見 summary.json。</desc>',
        "<style>",
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e4e3df;--series:#2a78d6;}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e88;--grid:#383835;"
        "--series:#3987e5;}}",
        ".bg{fill:var(--surface)}.t1{fill:var(--ink);font-size:15px;font-weight:600}.t2{fill:var(--ink2);font-size:11px}"
        ".t3{fill:var(--muted);font-size:10px}.grid{stroke:var(--grid);stroke-width:1}.line{fill:none;stroke:var(--series);"
        "stroke-width:2}.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}.safe{fill:var(--surface);stroke:var(--series);"
        "stroke-width:1.5}.cliff{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:4 3}",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
        f'<text class="t1" x="{left}" y="26">{html.escape(title)}</text>',
        f'<text class="t2" x="{left}" y="44">x：checkpoint k；y：0–1。實心點 P(success | k)；空心點 P(success ∧ safe | k)；虛線：cliff。</text>',
        f'<text class="t2" x="{left}" y="60">RQ1：{html.escape(rq1)}'
        + ("" if _data_complete(summary) else "（資料不完整）") + "。</text>",
    ]
    by_key = {(c["level"], c["family"], c["seed"]): c for c in curves}
    y = top
    for lv in LEVELS:
        out.append(f'<text class="t1" x="{left}" y="{y + 14}">{lv}</text>')
        y += 22
        for family in FAMILIES:
            for col, seed in enumerate(seeds):
                c = by_key.get((lv, family, seed))
                x0 = left + col * (panel_w + gap)
                px0, px1, py0, py1 = x0 + 26, x0 + panel_w - 8, y + 18, y + panel_h - 18
                label = f"{family} s{seed}" + ("" if c is None else f"（{CLASS_TEXT[c['class']]}）")
                out.append(f'<text class="t2" x="{px0}" y="{y + 11}">{html.escape(label)}</text>')
                for level in (0.0, 1.0):
                    yy = py1 - (py1 - py0) * level
                    out.append(f'<line class="grid" x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}"/>')
                    out.append(f'<text class="t3" x="{px0 - 4}" y="{yy + 3:.1f}" text-anchor="end">{level:g}</text>')
                if c is None:
                    out.append(f'<text class="t3" x="{px0}" y="{(py0 + py1) / 2:.1f}">沒有曲線（未完成或沒有 source）</text>')
                    continue
                span = max(c["n_turns"] - 1, 1)
                sx = lambda k, px0=px0, px1=px1, span=span: px0 + (px1 - px0) * k / span  # noqa: E731
                sy = lambda p, py0=py0, py1=py1: py1 - (py1 - py0) * p  # noqa: E731
                pts = [(e["k"], e["successes"] / e["n"]) for e in c["estimates"] if e["n"]]
                if c["class"].startswith("cliff") and c["largest_drop_k"] is not None and c["next_k"] is not None:
                    xk = sx((c["largest_drop_k"] + c["next_k"]) / 2)
                    out.append(f'<line class="cliff" x1="{xk:.1f}" y1="{py0}" x2="{xk:.1f}" y2="{py1}"/>')
                if len(pts) > 1:
                    out.append('<path class="line" d="' + " ".join(f'{"M" if i == 0 else "L"}{sx(k):.1f},{sy(p):.1f}'
                                                                  for i, (k, p) in enumerate(pts)) + '"/>')
                for point in c["success_and_safe"]:
                    if point["n"]:
                        out.append(f'<circle class="safe" cx="{sx(point["k"]):.1f}" cy="{sy(point["success_and_safe"] / point["n"]):.1f}" r="3"/>')
                for e in c["estimates"]:
                    if e["n"]:
                        tip = f"k = {e['k']}：{e['successes']}/{e['n']} 成功（max_turns {e['max_turns']}）"
                        out.append(f'<circle class="dot" cx="{sx(e["k"]):.1f}" cy="{sy(e["successes"] / e["n"]):.1f}" r="4">'
                                   f'<title>{html.escape(tip)}</title></circle>')
            y += panel_h + gap
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_readme_block(summary: dict[str, Any]) -> str:
    if summary["provenance"]["dry_run"]:
        raise ValueError("dry-run outputs are not results and never go into the README")
    a = summary["analysis"]
    levels = a["baseline"]["levels"]
    check = a["baseline"]["level_check"]
    sweep = a["sweep"]
    v = VARIANTS[variant(summary)]
    lines = [v["start"], "", v["heading"], "",
             v["intro"] + ("" if _data_complete(summary) else "**資料不完整。**"), "",
             "| 級別 | success | 成功但不安全 |", "|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    for lv in LEVELS:
        lines.append(f"| {lv} | {fmt(levels[lv]['success_rate'])} | {fmt(levels[lv]['success_unsafe_rate'])} |")
    valid_text = "無法計算" if check["valid"] is None else ("有效" if check["valid"] else "無效")
    lines += ["", f"- 分級：{valid_text}（L1 − L3 的 95% CI 下限 "
                  f"{fmt(None if check['l1_minus_l3'] is None else check['l1_minus_l3']['low'])}）",
              "- RQ1：" + ("沒有 sweep（" + SKIP_TEXT[a["sweep_skipped"]] + "）" if sweep is None else _rq1_line(sweep["rq1"])),
              "", f"完整結果：[{v['page']}]({v['page']})。", "", v["end"]]
    return "\n".join(lines)


def _rq1_line(rq1: dict[str, Any]) -> str:
    """RQ1 的結論附上分母、CI 與失敗 source 的組成（可判定的曲線很少時，讀者要看得到樣本數）。"""
    ci = f"，95% CI [{rq1['low']:.3f}, {rq1['high']:.3f}]" if rq1["n"] else ""
    return (f"{RQ1_TEXT[rq1['outcome']]}：可判定的失敗曲線中 cliff {rq1['k']} / {rq1['n']}{ci}"
            f"（失敗的 source {rq1['n_failed_sources']} 條，無法判定 {rq1['n_undecidable']} 條、max_turns 混淆 "
            f"{rq1['n_max_turns_confounded']} 條）")


def embed(readme: str, block: str) -> str:
    """README 中剛好一對該段落的標記（block 的第一行與最後一行）→ 取代其間的內容；沒有標記 → 加在最後；不只一對 → ValueError。"""
    start, end = block.split("\n")[0], block.split("\n")[-1]
    starts, ends = readme.count(start), readme.count(end)
    if starts == 0 and ends == 0:
        return readme.rstrip("\n") + "\n\n" + block + "\n"
    if starts != 1 or ends != 1 or readme.index(start) > readme.index(end):
        raise ValueError(f"the README must contain exactly one {start} … {end} block")
    before, rest = readme.split(start)
    _, after = rest.split(end)
    return before + block + after


def _is_round1_path(path: Path) -> bool:
    """path 是（或位在）封存的 docs/results 本身：不分大小寫，也不依賴工作目錄。docs/results/study2 不算。"""
    target = os.path.normcase(str(ROUND1_RESULTS)).casefold()
    resolved = Path(path).resolve()
    if resolved.exists() and ROUND1_RESULTS.exists() and os.path.samefile(resolved, ROUND1_RESULTS):
        return True
    return os.path.normcase(str(resolved)).casefold() == target


def _same_dir(path: Path, target: Path) -> bool:
    resolved, target = Path(path).resolve(), Path(target).resolve()
    if resolved.exists() and target.exists() and os.path.samefile(resolved, target):
        return True
    return os.path.normcase(str(resolved)).casefold() == os.path.normcase(str(target)).casefold()


def _under_round1(path: Path) -> bool:
    resolved = Path(path).resolve()
    return any(_is_round1_path(parent) for parent in (resolved, *resolved.parents))


def write_report(baseline_runs: Sequence[str | Path], sweep_runs: Sequence[str | Path], out: str | Path, *,
                 readme: str | Path | None = None, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    out = Path(out)
    if _is_round1_path(out):
        raise ValueError("docs/results holds the sealed round-1 results; write Study 2 to docs/results/study2 (decision 034 §1)")
    if readme is not None and _under_round1(Path(readme).parent):
        raise ValueError("the README of the sealed round-1 results is never modified (decision 034 §1)")
    summary = build(baseline_runs, sweep_runs, replicates=replicates)
    own = variant(summary)
    for name, directory in RESULT_DIRS.items():  # 一個 agent 的報告不寫進另一個 agent 的目錄或其子目錄（decision 035 §6）
        resolved = Path(out).resolve()
        if name != own and any(_same_dir(p, directory) for p in (resolved, *resolved.parents)):
            raise ValueError(f"{directory} holds the {name} report; this report is {own} (write it to {RESULT_DIRS[own]})")
    if not summary["provenance"]["dry_run"] and replicates != BOOTSTRAP_REPLICATES:
        raise ValueError(f"a formal Study 2 report uses B = {BOOTSTRAP_REPLICATES} (decision 034 §7)")
    new_readme = None
    if readme is not None:  # 在寫任何檔案之前完成所有檢查
        block = render_readme_block(summary)
        path = Path(readme)
        if not path.is_file():
            raise ValueError(f"{path} does not exist")
        new_readme = embed(path.read_text(encoding="utf-8"), block)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "README.md").write_text(render_page(summary), encoding="utf-8")
    (out / "recoverability.svg").write_text(recoverability_svg(summary), encoding="utf-8")
    if new_readme is not None:
        Path(readme).write_text(new_readme, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--sweep-run", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--readme", default=None)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)
    try:
        write_report(args.baseline_run, args.sweep_run, args.out, readme=args.readme, replicates=args.replicates)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"wrote {args.out}" + (f" and {args.readme}" if args.readme else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
