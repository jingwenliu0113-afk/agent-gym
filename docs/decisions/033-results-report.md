# 033 — 正式結果的報告：docs/results 與 README 的結果段落

日期：2026-10-16　狀態：採用（只新增報告工具；不改任何分析規則、判定規則、主要 metric 或資料）

## 背景

使用者在確認凍結時（decision 030）要求：完成正式實驗後，完善可重現的 Demo、分析圖表、失敗案例及 README。
預先登記的分析工具（`analyze_baseline` / `analyze_sweep`）把輸出寫在 `runs/`，不進 repo。README 需要可以檢視的結果與圖表。

## 決策

- `experiments/results_report.py`：從正式 run 目錄產生報告。分析一律呼叫預先登記的 `analyze_runs` / `analyze_sweep_runs`
  （載入時完整核對 manifest、summary、trajectory 與續跑血緣），不另外計算任何指標。輸出完全決定性。報告包含下列內容：
  - **設定與紀錄**：模型、digest、Ollama 版本、config_id、推論參數、run 目錄、日期；`calls.jsonl` 的呼叫數、失敗數、seed 數，
    以及成功的呼叫數是否等於 manifest 的計費數（decision 032）。
  - **eval baseline**：主指標與 95% CI、per-family、結果與嘗試計數、tool 錯誤的分類。
  - **sweep**：plan、驟降判定、rerun 計數、cliff 比例、平均最大下降、Gate 的機械化判定（decision 025 §4）與說明、每個 source 的
    摘要，以及每個 checkpoint 的數值表（也就是表格版的圖）。
  - **recoverability 圖**（`recoverability.svg`）：15 個 source 的 small multiples，每個 family 一列，x 軸共用。單一序列
    （P(success | k) 與 95% CI），深色模式有自己的顏色；每個點有 `<title>`，整張圖有 `<title>` / `<desc>`。
    顏色經過 dataviz 的 validator 檢查（淺色 / 深色皆通過）。
  - **一個正式 trajectory 的完整步驟**（可重現的 Demo）：最長的 source 的每個 tool call、結果、狀態變更與 grader 的判定，作為首頁
    synthetic 示範的真實對照。
  - **失敗與異常案例**：
    - baseline 沒有成功的 rollout；
    - sweep 中每個沒有成功的 rerun：重新評分，列出沒有通過的檢查與最後一則訊息的開頭，最多列 30 個，總數照實報告；
    - 被模型自己修正的 tool 錯誤（例子）。
- 進 repo 的只有 `docs/results/` 的五個小檔案（合計約 80 KB）：`summary.json`（數字與短文字，不含 trajectory）、`README.md`、
  `recoverability.svg`、`baseline-analysis.md`、`sweep-analysis.md`，
  以及 README 中 `<!-- results:start -->` 與 `<!-- results:end -->` 之間的段落。
  - 原始的 run 目錄（trajectory、manifest、`calls.jsonl`、server log）照舊留在 `runs/`，不進 repo（使用者的規則：不 push `runs/`
    與大型產出）。
  - `docs/results/` 與既有的 `docs/demo/` 同一類：README 引用的小型文件，由程式決定性產生，並以測試確認與 README 同步。
- dry-run 的輸出可以產生報告（頁首標示「不是研究結果」），但**不能**寫進 README。

### 審查後的修正（2026-10-16，同一輪審查）

- **完整報告原樣附上**：預先登記的分析工具的完整報告（`render_markdown`）原樣寫成 `docs/results/baseline-analysis.md` 與
  `sweep-analysis.md`。下列項目都以登記的工具自己的呈現為準：
  - 所有指標（含排除的題目）、安全指標（含 forbidden attempts、constraint violation）；
  - 成功但不安全的案例、中止嘗試的部分安全指標；
  - 中斷與續跑、資料是否完整。

  結果頁只是摘要，不取代它們。
- **CI 的標示**：每個 checkpoint 的區間是**逐點 95% Wilson CI**，沒有做多重比較校正；cliff 判定使用 bonferroni95 的較寬區間。
  原本誤標為 Bonferroni。
- **Gate**：
  - `data_complete` 為假 → 結果頁、圖與 README 都標「資料不完整」，而且 **README 不更新**；
  - Gate 沒有評估（response_model 不只一種）→ 標「不評估」，README 不更新；
  - 沒有任何失敗時，明確寫出「所有 rerun 都成功、天花板效應」，不套用 `gradual` 的固定說法「失敗是逐步累積」。
- **失敗案例**：
  - sweep 的 rerun 依 manifest 中 completed 的列找到保存它的目錄，並核對 trajectory_id，不再取第一個檔名相符的目錄；
  - baseline 沒有成功的 rollout 改由 `load_runs` 的紀錄列出，包括中斷的目錄；
  - 另外回報沒有進入曲線的部分 sweep、中斷的目錄、因預算停止與續跑的次數。
- **tool 錯誤**：每個例子標明模型是否在下一步修正（下一個 call 是同一個 tool 而且成功）。
- **其他**：
  - walkthrough 依 `source_trajectory_id` 找到 sweep 實際用的那條 trajectory；
  - R 從 run.json 取得，不再寫死；
  - 圖的最小寬度 720 px；
  - dry-run 必須加 `--allow-dry-run` 才會產生報告，而且在寫任何檔案之前就拒絕寫入 README。
- **測試的有效性**：
  - 凍結後，`verify_server` 的 tools 能力與 cloud 模型檢查的測試改用凍結的身分，只讓被測的那一項拒絕；
  - 新增正式 sweep CLI 的 `calls.jsonl` 檢查，以及「草案設定、非 dry-run 的 baseline 不能當正式 sweep 的 source」的測試；
  - 新增 cliff 分支與 CI 位置的圖形測試、逐次核對能找出缺漏或沒有 seed 的紀錄的測試。

## 為什麼

- 結果必須能從原始紀錄重新產生並核對；報告只是預先登記的分析的呈現，不新增任何判定。
- 失敗案例與 Gate 的判定照實呈現：沒有 cliff 就寫沒有 cliff，Gate 不通過就寫不進 Advanced。

## 對實驗的影響

無。分析方法、判定規則與 Gate 都沒有改變。
