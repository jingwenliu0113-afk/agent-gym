# 027 — Ollama adapter 的強化：遠端斷線不得變成 rollout 結果、context 防護優先、無法記錄的參數（decision 026 審查後）

日期：2026-10-11　狀態：採用（補足 decision 026 §1；不改模型、prompt、tool schema、推論參數、SweepPlan、判定規則或主要 metric）

## 背景

decision 026 的對抗式審查（以 127.0.0.1 上的測試 server 與假 client 重現，不連任何遠端）確認了三類缺陷：

1. **遠端斷線會變成 rollout 的最終結果**（高嚴重度）。通道或 Ollama 中斷時，每次呼叫都會得到 OllamaTransportError
   （infrastructure）。依 decision 009 重試一次之後，該 rollout 記為 aborted，下一個 rollout 也一樣。結果在幾毫秒內，其餘所有 rollout
   都變成 aborted，而且 run 正常結束（exit 0）。後果：
   - 正式 baseline：rollout 0 是 aborted 的 task 永久失去 source（012 §3 / 019 §1 不以其他 rollout 補上）；
   - 正式 sweep：source 被記為「完成」，但每個 k 的 rerun 都是 aborted；分析報告資料完整、cliff 0/15、Gate 判為 gradual；
   - 斷線恢復後，續跑會顯示「nothing left to run」：aborted 是最終結果（023），所以資料無法在協定內補回。

   Anthropic SDK 本身有 4 次重試（decision 009），本機 client 沒有；而「整個推論服務不見了」與「一次請求失敗」是不同的情況。
2. **`done_reason` 為 `length` 時跳過了 context 防護**（中嚴重度）：這個分支在兩個 context 檢查之前就丟出 AdapterError
   （agent_output），也沒有比較 eval_count 與 num_predict。結果是用盡 `num_ctx` 的回覆被記成模型自己的失敗，而不是停止實驗。
3. **無法記錄的 tool call 參數漏出未分類的例外**（中嚴重度）：
   - 物件參數巢狀 65–124 層 → ToolCall 的 ValueError；更深 → OllamaProtocolError；兩者都被分類為 harness，整個 run 停止。
   - JSON 字串參數含 NaN / Infinity / 1e400 → ValueError；極深的字串 → RecursionError；
   - 極深的回覆本文 → RecursionError。

   decision 009 規定：模型的 tool input 無法成為合法 ToolCall 時，屬於 agent_output（AnthropicPolicy 就是這樣處理的）。

同一輪判定不是缺陷：
- HTTP 5xx 的內容提到 context：026 依 status 分類，而 Ollama 以哪個 status 回報 context 錯誤沒有文件；
- 執行中途 server 被換掉：GPU lock 期間不會發生，每次執行開始時都會檢查；
- 續跑時沒有印出先前的呼叫次數：019 §3 規定上限只限這一次執行。

## 決策

1. **服務不可用 ≠ 一次失敗的請求**：`OllamaPolicy.next_message` 遇到 OllamaTransportError 時，立刻以 `client.version()` 探測。
   - 探測也失敗 → 丟出 `OllamaServerUnavailable`（harness 類，不是 infrastructure）。run_logged 會記錄這次嘗試（harness 中止、
     **不是**最終結果），然後 run 停止。
   - 探測成功 → 原本的 OllamaTransportError（infrastructure）照常重試一次。

   結果：服務中斷時，run 在第一次失敗就停止，不會把其餘 rollout 記為 aborted；中斷的那個 rollout 不是最終結果
   （baseline 為 interrupted、sweep 為非最終），服務恢復後依 017 / 019 / 023 的規則以新目錄重跑或續跑即可補回。
   client 必須提供 `version()`（真正的 client 與 dry-run 假 client 都有）。
2. **context 防護在 length 之前**：先檢查 `prompt_eval_count` 存在且 >= 1，再檢查 `prompt_eval_count + eval_count < num_ctx`，最後才看
   `done_reason`。`length` 只有在 `eval_count >= num_predict` 時才是輸出截斷（AdapterError，agent_output）；輸出少於 `num_predict`
   卻停在 length → `OllamaContextError`（harness）。
3. **無法記錄的參數**：
   - JSON 字串參數以 strict 解析（拒絕 NaN / Infinity、超出 float 範圍、重複 key、過深）；解析失敗、或解析後不是可記錄的物件
     → 原字串交給 env（記為 invalid_arguments）；
   - 物件參數或其他回覆內容無法成為合法的 ToolCall / ModelMessage（過深、非有限數、無法編碼的字串）→ `AdapterError`
     （agent_output，附 billed_usage）；
   - message 的結構型別錯誤（role、content 型別、tool_calls 形狀）仍是協定錯誤（harness）；
   - HTTP 回覆本文以 strict JSON 解析，過深或不合法 → `OllamaProtocolError`（不再漏出 RecursionError）。

## 為什麼

- 已登記的協定把 aborted 當成最終結果；只有在模型或單次請求真的失敗時才應該產生它。服務中斷屬於基礎設施的停機，
  必須停下來、保留可續跑的狀態。
- context 用盡代表 prompt 可能沒有完整進入模型，必須停止，不能算成模型的失敗。
- 單一 rollout 中模型的怪異輸出只應讓那個 rollout 失敗，不應停掉整個實驗；Anthropic 路徑早就是這樣處理的。

## 放棄的替代方案

- 在 client 層加入多次重試與退避：無法區分「服務不見了」與「一次請求失敗」，長時間中斷時仍然會把 rollout 記為 aborted。
- 每次嘗試之前都探測服務：多一次往返，而且失敗後的探測已能涵蓋同樣的情況。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。
- 中斷的本機 run 一律可以補回：baseline 以新目錄重跑，sweep 以 `--resume-from` 續跑。
