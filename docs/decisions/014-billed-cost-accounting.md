# 014 — 已計費但被拒絕的回應必須記帳；manifest 升 attempt/v2

日期：2026-10-08　狀態：採用（任何付費 run 之前；修正 decision 009 / 012）

## 背景

§6 與執行腳本的對抗式審查確認一個高嚴重度問題：`AnthropicPolicy.next_message` 在 API **已回傳、已計費**之後，才由 `parse_response`
拋出 AdapterError（例如 `max_tokens` 截斷、未知 block）或 ReplayIntegrityError。`CostMeter` 與 `CountingPolicy` 只在 next_message
正常回傳時記帳，因此：成本上限可被繞過（審查重現：上限 $1，實際計費約 $10）、`spent_usd` 與 manifest 都低估，而這類回應正是最貴的
（max_tokens = 16000 個 output tokens ≈ $0.32 / 次）。另外：模型不在 PRICING 時成本是 None，成本上限會靜默失效。

## 決策

1. `agents.baseline.billed_usage(response, model)`：從 response 的 usage 計算 `{"input_tokens", "output_tokens", "cost_usd"}`，
   規則與 `parse_response` 相同，但**不拋例外**（usage 不合法 → 三者皆 None；模型不在 PRICING → cost_usd None）。
2. `AnthropicPolicy.next_message`：client 回傳之後，`parse_response` 拋出的 AdapterError / ReplayIntegrityError 附加屬性
   `billed_usage`（同一個例外物件原樣往外拋）。client 本身的例外（沒有 response）不附加，視為未計費。
3. 計費的 call = 成功回傳的 next_message + 帶 `billed_usage` 的例外。`CountingPolicy` 與 `CostMeter` 都依此記帳
   （cost 為 None → unknown_cost_calls + 1）。
4. manifest 升 **`agent-gym/attempt/v2`**：AttemptRecord 新增 `billed_calls`（int）、`cost_usd`（這次嘗試已知成本的總和，float）、
   `unknown_cost_calls`（int）。只計這次嘗試的 call（不含 checkpoint 前綴）。此時還沒有任何真實的 v1 manifest。
5. `run_split` / `run_sweeps`：給了 `max_cost_usd` 時，模型必須在 PRICING 中（否則 ValueError，不執行）；執行中只要出現
   成本無法量測的已計費 call，就在下一次嘗試開始前停止排程（`stopped_by_budget = True`），寧可保守。

## 對實驗的影響

- 付費 run 的成本上限現在涵蓋所有已計費的 call；報告的成本可由 manifest 逐列加總稽核（含中止的嘗試）。
- harness 中止時 summary 不寫（decision 009 / 012 不變），但 manifest 已有每次嘗試的成本。
