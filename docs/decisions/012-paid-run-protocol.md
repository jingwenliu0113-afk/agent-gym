# 012 — 付費 LLM run 的執行協定（runner、成本上限、source 選取、Gate 判定）

日期：2026-10-08　狀態：**提案，第一次付費 run 之前須經使用者確認**（模型、成本、eval rollout 數、SweepPlan、Gate 判定都屬於會影響研究結論的決策）

## 背景

Week 1 #5 與 SPEC §6 的 keyless 部分已完成（decision 009 / 010 / 011）。剩下必須真正呼叫 LLM 的步驟：dev sanity、eval baseline、
eval 上的 failure localization sweep。依使用者指示，付費 run 之前要先確認模型與成本，credential 由環境變數提供。

## 提案

### 1. 執行工具（已實作、已以 dry-run 驗證）

- `experiments/run_baseline.py`：`--dry-run`（OracleFakeClient，不連網）或 `--confirm-paid-run --max-cost-usd X`。
  付費模式**必須**給成本上限；每個 episode 開始前檢查已花費（所有**已計費**的 model call：成功回傳的，以及 API 已回傳但被 adapter
  拒絕的，例如 max_tokens 截斷；含之後中止的嘗試，decision 014），達上限、或出現成本無法量測的已計費 call，即停止排程。
  有上限時模型必須在 PRICING 中，否則拒絕執行。manifest 每列記錄該次嘗試的已計費 call 數與成本，可逐列稽核。
  credential 完全由 SDK 從環境變數解析，程式不讀取、不檢查、不記錄任何 key。
- 輸出到 gitignored 的 `runs/<name>/`：manifest（每次嘗試一列）、每條 trajectory、summary；拒絕寫入非空目錄（不覆寫既有 run）。
- 排程：rollout_index 外層、task 內層（同一輪所有 task 跑完才進下一輪），降低時間效應。
- `experiments/run_sweep.py`：讀 eval baseline 的 summary 依第 3 節選 source，對每個 source 執行 `replay.localization.sweep`
  （共用一個 manifest 與一個成本帳）；SweepPlan **沒有預設值**，必須明確給定。達到成本上限時在下一次嘗試**開始之前**停止
  （不留下多餘的嘗試紀錄），已完成的 source 照常彙總。彙總：完成的 source 以 family 分層 task-level bootstrap 估計 cliff 比例與
  平均最大下降（任一 family 沒有完成的 source → 不做 bootstrap，只列個別結果）。
- 不支援並行；被手動中斷時，進行中的嘗試不會進 manifest，重跑必須用新目錄，報告時列出所有目錄（decision 009 審查修正）。

### 2. 順序與預算（估算見 `experiments/cost_estimate.py`）

估算方法：以 dry-run 量測 adapter 實際送出的 request 大小（字元 / 3.5 ≈ tokens），不計 prompt caching 折扣（偏高）；
output tokens 每次 call 取低 500 / 高 4000 兩組假設（adaptive thinking + tool call，真實值未知），高情境另假設 turn 數為 oracle 的 2 倍。

| 步驟 | 內容 | 估算成本 | 提案上限 |
|---|---|---|---|
| a. dev sanity | dev 5 題 × 1 rollout；只驗證 API 相容性（betas、block_binding、thinking 回傳）與實際 tokens / call，不當結果報告 | $0.4 – $6.8 | `--max-cost-usd 10` |
| b. eval baseline | eval 15 題 × R rollout（提案 R = 3；主要指標依 decision 005 / 009） | 每輪 $1.2 – $18.5，R = 3 共 $3.5 – $56 | 依 a 的實測重新估算後由使用者決定 |
| c. eval sweep | 每題一條 source × SweepPlan | 上限（所有 checkpoint 都 refine）n_fine = 24：$89 – $2,790；n_fine = 32：$119 – $3,719 | 依 a、b 實測與 decision 011 的 plan 重新估算後由使用者決定 |

a 完成後以實測的 tokens / call 取代假設，重新估算 b、c 再請使用者確認；不得在沒有新的確認下直接接著跑 b、c。

### 3. Sweep 的 source 選取（看結果前固定）

- 每個 eval task 取 eval baseline 中 **rollout_index = 0 的那一條完成的 trajectory**（不論成功與否、不挑選）。
- rollout_index 0 未完成（agent_error / aborted）→ 該 task 沒有 source，報告中列為缺漏；不得改用其他 rollout 補上。
- 成功的 source 也做 sweep：其曲線用於區分「一直都可恢復」與「中途出現不可恢復點」。

### 4. Gate 判定的操作化（提案）

SPEC Gate：「至少一種跨多條 trajectory 重複出現的 recoverability 驟降模式」。提案在看結果前固定為：

- 驟降 = sweep 的 `is_cliff`（decision 010 / 011 的判定與 plan）；位置 k*。
- 模式 = k* 那個 turn（第 k* 個 turn 的行動）的 (tool 名稱, grader 對該步的歸類：forbidden attempt / constraint 相關變更 / unnecessary mutation / 其他)。
- Gate 滿足 ⇔ 同一模式出現在 **至少 2 個不同 family 的 task** 的 source 上。
- 否則依 SPEC「收尾」：若多數 source 為 gradual 且彼此不同，正式報告「失敗是逐步累積」，不進 Advanced。
- 15 題 eval 只能當 infrastructure validation / preliminary observation：Gate 的結論必須附上這個限制，不得宣稱統計上的泛化。

## 為什麼

- 成本上限與分段確認：模型與費用是使用者明確保留的決策；實測 tokens 之前的估算誤差可達一個數量級。
- source 選取不挑選：避免挑出「剛好有 cliff」的軌跡。
- Gate 操作化先寫下：避免看到 sweep 結果後才決定什麼算「同一種模式」。

## 待使用者確認的事項

1. 模型 `claude-opus-5-5`（decision 009 固定的設定）與 dev sanity 的 $10 上限。
2. 在環境變數提供 credential（本專案程式不讀取、不保存），並確認帳號可用 `thinking-binding-controls-2026-08-01` beta。
3. eval rollout 數 R（提案 3）與 b、c 的預算（a 完成後重估）。
4. SweepPlan：decision 011 依預先登記的門檻**沒有任何候選 plan 合格**，需要使用者在其中列出的選項中決定。
5. 第 4 節 Gate 判定的操作化。
