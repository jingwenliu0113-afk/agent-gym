# 032 — 逐次模型呼叫紀錄：每個 seed 都可稽核（calls.jsonl）

日期：2026-10-16　狀態：採用（只新增稽核紀錄；不改模型、prompt、tool schema、推論參數、預算、SweepPlan、判定規則或主要 metric）

## 背景

- 使用者確認凍結（decision 030）時要求：「每次隨機 seed 必須完整記錄，確保實驗可稽核與重現」。
- 原本的記錄方式（decision 026 §5）：每次呼叫的 seed 記在 trajectory 每個 turn 的 `meta.seed`。但 trajectory 只保存**完成的**嘗試。
  下列呼叫的 seed 都沒有留在任何檔案裡：
  - infrastructure 失敗而重試的嘗試；
  - agent_output 失敗的嘗試，例如輸出被截斷；
  - 被 Ctrl-C 或服務中斷的嘗試。

## 決策

- 本機模型的正式 run（`run_baseline --confirm-local-run`、`run_sweep --confirm-local-run`）在 run 目錄寫 `calls.jsonl`，
  **每次 chat 一行**，格式 `agent-gym/call/v1`（canonical JSON）。每一行記錄：
  - 這個目錄的第幾次呼叫、送出時間；
  - task_id、rollout_index、attempt、k（sweep 的 checkpoint；baseline 為 null）；
  - request 中已有的 assistant 訊息數、這次呼叫的 seed、request 的 sha256；
  - 呼叫的秒數；
  - 成功時：模型的回覆訊息與計數（model、done_reason、prompt_eval_count、eval_count、total_duration）；
  - 失敗時：例外的種類與訊息。
- 寫入方式：每行寫入後 flush + fsync，中斷時已發生的呼叫不會遺失。檔案在 run 開始時建立（空的），不覆寫既有檔案。
- 實作：`experiments/call_log.py` 的 `CallLog` 包住 client，每次嘗試以 task、rollout、attempt、k 綁定。
  其他屬性（version / tags / show、decision 028 的連續失敗計數）直接取自原本的 client，adapter 的行為不變。
- `local_probe --summarize` 回報紀錄的呼叫數、失敗數、有 seed 的呼叫數，以及「成功的呼叫數是否等於計費的呼叫數」。
- dry-run 不寫這個紀錄（不是正式結果）。

## 為什麼

- 每個實際發生的呼叫都可以對應到它的 seed 與 request，包括沒有成為最終結果的嘗試。
- 完成的嘗試可以由 trajectory（conversation）加上 seed 重送同一個 request。未完成的嘗試則由 request 的 sha256 與回覆稽核。

## 放棄的替代方案

- 把未完成嘗試的部分 trajectory 存檔：要改 manifest 與 trajectory 的格式和續跑規則（decision 018–024）。逐次紀錄已足夠稽核。
- 記錄完整的 request body：每次呼叫數 KB，sweep 的上萬次呼叫會很大；完成嘗試的 conversation 已在 trajectory 中。

## 對實驗的影響

不影響任何估計方法、判定規則、主要 metric 或資料切分。run 目錄多一個檔案；分析工具不讀它（與 summary / manifest 的交叉核對不變）。
