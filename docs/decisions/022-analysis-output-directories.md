# 022 — 分析輸出目錄不覆寫、dry-run 曲線的標示（第十二輪審查後；在任何 LLM 結果之前）

日期：2026-10-09　狀態：採用（補足 decision 021 §1；不改主要 metric、資料切分、模型、SweepPlan 或 criterion）

## 背景

第十二輪對抗式審查（decision 021 的實作）確認一個缺陷：`analyze_sweep --out` 沿用既有目錄時，只覆寫這次有的曲線，
不清除舊檔。重現情境如下：

1. 先在 `runs/sweep-analysis` 做了 dry-run 預演的分析，產生 15 條曲線。
2. 付費分析因為有缺漏的 source（rollout 0 是 agent_error）或因預算停止，只有 14 條曲線。
3. 付費分析使用同一個 `--out`，於是目錄裡留下一張 dry-run 的 oracle 曲線。

這張 SVG 本身沒有任何 dry-run 標示，與付費曲線無從區分。NEXT B.4 又要把曲線放進 README 的結果段，
因此違反 021「假 client 的輸出不得以任何途徑進入曲線」。

同一輪判定「不是缺陷」的一項：以 Python 直接呼叫 `analyze(load_rollouts(...))` 時沒有模式資訊。
這屬於 021 §1 已記錄的「直接以 Python 呼叫」限制；CLI 與 `analyze_runs` 都正確。

## 決策

1. `analyze_baseline` 與 `analyze_sweep` 的 CLI 一律拒絕寫入已存在且不是空的 `--out`，在寫入任何檔案之前報錯。
   這比照 run 目錄的規則（decision 012 §1：不覆寫既有輸出）。
2. `curve_svg(curve, *, dry_run=False)`：dry_run 為 True 時，標題開頭加上「[dry-run] 」。
   `analyze_sweep` 依分析的 dry_run 傳入，所以單獨複製出去的 SVG 也能分辨。

## 為什麼

- 分析輸出是報告的直接來源（decision 016：報告由程式機械產生），不得混入上一次分析的殘留檔案。
- 曲線最可能被單獨取用（README、簡報），標示必須跟著檔案本身。

## 放棄的替代方案

- 寫入前自動清空 `--out`：刪除既有檔案屬於破壞性操作，而且可能刪到使用者放在那裡的其他檔案。
- 只在 analysis.md 加註：SVG 被單獨取用時看不到。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。
- 每次分析都要用新的 `--out`（或空目錄）。
