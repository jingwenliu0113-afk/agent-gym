# 021 — dry-run 與付費輸出的區分、祖先目錄必須齊全（第十一輪審查後；在任何 LLM 結果之前）

日期：2026-10-09　狀態：採用（補足 decision 012、019 §1 / §3 與 020 §3；不改主要 metric、資料切分、模型、SweepPlan 或 criterion）

## 背景

第十一輪對抗式審查（decision 020 的實作與 keyless Core 完整性）確認兩個缺陷：

1. **dry-run 輸出與付費輸出無從區分**。run.json、manifest、summary 都沒有記錄 client 是不是 OracleFakeClient，
   而 OracleFakeClient 的回應依真實價格計價，response.model 也與真實的相同。實際後果：
   - 以 `--dry-run` 預演付費續跑之後，付費續跑會被 020 §3 的後代檢查擋下，錯誤訊息還說預演目錄「已計費」、要求一起傳入。照著做，
     付費 run 會略過只被假 client「完成」的 source，而分析會把假的曲線與成本併入正式結果。
   - baseline 有同樣的問題，而且更嚴重：019 §1 取所有目錄中「最早結束」的 rollout 0。如果使用者把較早做的 dry-run 預演目錄也當成
     「所有 eval baseline 目錄」之一傳入，付費 sweep 的 source 會是 oracle 的假軌跡。analyze_baseline 也會把假的 rollout 併進主指標。
2. **沒有完成任何 sweep 的祖先目錄可以被漏掉**。019 §3 的檢查只看 skipped，所以當祖先目錄因為預算很小、在第一個 source 完成前就
   停止時，它不會出現在任何 skipped 中，續跑與分析都可以略過它。它已計費的成本與中止嘗試就從印出的先前花費和報告中消失，
   而報告看起來是完整的。這也破壞了 020 §3 依賴的前提「每個後代的 resumed_from 都含所有祖先」。

## 決策

### 1. dry-run 標記

- **run_baseline**：`run_split(..., dry_run=False)`。在第一次嘗試之前寫入 `run.json`，內容為 format、split、rollouts、config_id、
  dry_run。baseline 格式升為 `agent-gym/baseline-run/v2`（run.json 與 summary.json 共用）。CLI 的 `--dry-run` 傳入 `dry_run=True`。
- **load_runs**（analyze_baseline 與 select_sources 共用）：
  - 每個目錄都必須有 v2 的 run.json，key 與 format 正確，split = eval，rollouts 為 int >= 1，config_id 與 manifest 相同，dry_run 為 bool。
  - 有 summary 時，summary 的 split、rollouts、config_id 必須與 run.json 相同。
  - 所有目錄的 dry_run 必須相同：dry-run 與付費的輸出不得合併。
- **baseline 分析**：記錄 dry_run，dry-run 時報告開頭標示「不是研究結果」。分析格式升為 `agent-gym/baseline-analysis/v3`。
- **run_sweeps**：`dry_run=False` 參數寫入 sweep 的 run.json。sweep 格式升為 `agent-gym/sweep-run/v6`。CLI 的 `--dry-run` 傳入 True。
- **run_sweep CLI**（以下檢查都在建立目錄與任何 call 之前）：
  - 付費 sweep 的 baseline 必須是付費的。
  - 續跑時，`--resume-from` 的目錄必須與這次是同一種模式：付費不能接 dry-run，dry-run 也不能接付費。預演續跑要用 dry-run 鏈。
  - 020 §3 的後代檢查只看與這次同一種模式的目錄。不同模式的目錄不可能是合法的後代。錯誤訊息改為「已有嘗試紀錄」。
- **analyze_sweep**：所有 sweep 目錄必須是同一種模式；付費 sweep 的 baseline 必須是付費的；dry-run sweep 可以用任一種 baseline，
  因為它本來就不是研究結果。記錄 dry_run，dry-run 時報告開頭加上標示。分析格式升為 `agent-gym/sweep-analysis/v4`。
- **標記由 CLI 決定**。直接以 Python 呼叫 run_split / run_sweeps 時預設 dry_run=False，測試替身因此代表真實 run。
  這是已知限制，記在 NEXT。

### 2. 祖先目錄必須齊全

- **續跑**：給定目錄的 resumed_from 中，每個 run_id 都必須屬於給定的目錄之一。否則拒絕執行，不建立目錄、不發出任何 call，
  並列出缺少的 run_id；如果在相關目錄的上層目錄中找得到對應目錄，也列出它的路徑。
- **analyze_sweep**：每個目錄的 resumed_from 中，每個 run_id 都必須屬於一起分析的目錄之一，否則 ValueError。
  這取代 018 §1 / 019 對「只分析部分目錄」的容許。原本這種情況只把 skipped 的 source 標成 not_completed；現在只要已知有祖先目錄
  沒有傳入，成本與中止嘗試就不完整，因此直接拒絕，不再只是標示。

## 為什麼

- 假 client 的輸出沒有研究意義（decision 012），不得以任何途徑進入 source、主指標、曲線或成本。
- 用 dry-run 預演是自然的預防措施，不應該因此擋下付費續跑或誘導合併。
- 所有已計費目錄的成本都要進報告（018 的理由）；已知的祖先目錄沒有傳入時，報告不得看起來是完整的。

## 放棄的替代方案

- 以 trajectory 的 request_id / signature 前綴（`req_dry_`、`dry-run-`）判斷 dry-run：中止的嘗試沒有 trajectory，而且這依賴假 client
  的實作細節。
- 允許 dry-run 從付費目錄續跑（預演付費續跑）：這樣的預演目錄無法與任何目錄一起分析，還會讓血緣混雜。
- 在 run_split / run_sweeps 依 client 的型別自動判定：包裝過的 client 與測試替身無法可靠判斷。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。
- dry-run 的輸出永遠與付費輸出分開，不能一起分析，也不能互相續跑。
- 目前還沒有任何真實資料，格式升版不影響既有結果。
