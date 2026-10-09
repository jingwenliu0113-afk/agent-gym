# 025 — 使用者確認的付費 run 設計：模型、預算分段、R、SweepPlan、Gate 判定（在任何 LLM 結果之前）

日期：2026-10-09　狀態：採用（使用者於 2026-10-09 回覆「照你建議」；確認 decision 009 / 011 / 012 §2–§4 中待決的項目）

## 背景

decision 009、011、012 留下五個使用者決策：模型與預算、eval rollout 數 R、SweepPlan 與驟降判定、Gate 判定的操作化
（含 SPEC「收尾」分支中「彼此不同」的定義）。依預先登記的規則，SweepPlan 沒有合格的候選（decision 011），必須由使用者選擇。
2026-10-09 使用者採用 NEXT 中列出的全部建議。以下決定都在任何 LLM 結果之前寫下。

## 決策

### 1. 模型與預算

- 模型：維持 decision 009 的 `claude-opus-5-5` 與其 AgentConfig，不變更。
- 預算分段核准（decision 012 §2）：
  - 先核准 dev sanity，上限 $10；
  - eval baseline 與 sweep 的上限，在 dev sanity 以實測 tokens 重估之後，由使用者另行核准。
- 執行條件：使用者在環境變數提供 credential（程式不讀取、不保存），並明確指示開始；Mac 不在自動電力模式。
  在這之前不執行任何付費 API call。

### 2. eval rollout 數

- R = 3。
- 付費的 eval baseline 必須以 `--rollouts 3` 執行：run_baseline 的 CLI 在 `--confirm-paid-run --split eval` 時，
  R 不是 3 就拒絕執行（在任何 call 之前）。dev sanity 與 dry-run 不受限制。

### 3. SweepPlan 與驟降判定（decision 011 選項 A）

- 判定：`bonferroni95`（decision 015 的可選判定；取代 decision 010 原本的 Wilson 判定，作為正式 sweep 的判定）。
- plan：(coarse_stride, n_coarse, n_fine, drop_threshold) = (2, 10, 48, 0.3)。
  - 模擬：cliff 偵測 0.83、gradual 誤判 0.03，是唯一兩個門檻都達到的設定。
  - 估算：$213 – $3,241（dev sanity 後重估）。
- 付費 sweep 必須使用這組設定：run_sweep 的 CLI 在 `--confirm-paid-run` 時，plan 或 criterion 不同就拒絕執行（在任何 call 之前）。
  dry-run 不受限制。

### 4. Gate 判定（確認 decision 012 §4 的提案，並固定「收尾」的定義）

分析只看完成 sweep 的 source（曲線）；n = 曲線數。「驟降」= `is_cliff`（依上面的判定與 plan；max_turns 混淆的下降不算，decision 006 /
017 §1）。「模式」= 驟降那個 turn 的 (tool, grader 類別) 組合（decision 016 B3–B4）。三種結果互斥：

| 結果 | 條件 | 結論 |
|---|---|---|
| `advanced` | 同一個模式出現在至少 2 個不同 family 的 cliff source 上 | Gate 成立：可以進 Advanced |
| `gradual` | 不是 advanced；沒有 cliff 的曲線超過 n 的一半；而且沒有任何模式出現在 2 個以上的 cliff source 上（不論 family） | 正式結論「失敗是逐步累積」（SPEC 收尾），不進 Advanced，不加第二個 app |
| `no_repeated_pattern` | 其他情況（n >= 1） | 報告「有驟降但沒有跨 family 重複的模式」，不進 Advanced，也不寫「逐步累積」的結論 |

- n = 0 時結果為 `no_data`。
- 有缺漏或未完成的 source 時，照樣算出結果，但標示「資料不完整」（017 §5.3）。
- response_model 不只一種時，不評估 Gate（009 §4）。
- 所有結論都附上「eval 只有 15 題，只能當 infrastructure validation / 初步觀察」。
- 實作：`assess_gate` 新增 `outcome`，rule 字串改為引用本 decision，sweep 分析格式升為 `agent-gym/sweep-analysis/v5`。

## 為什麼

- 這些都是會影響研究結論的決策，依 CLAUDE.md 必須由使用者決定，並在看到結果之前寫下。
- 把已確認的設定寫進 CLI 檢查，付費 run 就不可能因為打錯參數而偏離登記的協定。
- 「收尾」原本只有文字描述；把三種結果機械化，看到結果之後就無法再改變「什麼算逐步累積」。

## 放棄的替代方案

- 選項 B / C（較便宜）：gradual 誤判約 14–15%，15 題中可能有 2 題左右被誤判為 cliff，容易湊出假的 Gate 結果。
- Gate 要求至少 3 個 family：eval 只有 15 題，幾乎不可能達到。
- 「彼此不同」只要求跨 family 不重複：同一個 family 內重複出現的驟降，不應該被寫成「逐步累積」。

## 對實驗的影響

- 正式的 eval sweep 用 `--plan 2,10,48,0.3 --criterion bonferroni95`；eval baseline 用 `--rollouts 3`。
- 付費 run 仍需使用者提供 credential 並明確指示開始。
