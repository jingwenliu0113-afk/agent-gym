# 028 — 推論路徑失敗、模型自己用盡 context、開發題彙整的完整性（decision 027 審查後）

日期：2026-10-11　狀態：採用（補足 decision 026 / 027；不改模型、prompt、tool schema、推論參數、SweepPlan、判定規則或主要 metric）

## 背景

decision 027 的對抗式審查（127.0.0.1 測試 server、假 client、PATH 上的假 ssh；不連任何遠端）確認了四項缺陷：

1. **`/api/version` 有回應、但 `/api/chat` 一直失敗**（高嚴重度）。例如 runner 當掉或 CUDA OOM 回 500、推論卡住到 600 秒逾時、
   佇列滿回 503。027 的探測只看 `version()`，所以每次失敗都被當成「一次失敗的請求」並重試一次，之後記為 aborted。重現的結果：
   - baseline 45/45 aborted、run 正常結束；
   - sweep 被記為 15/15 完成，每個 rerun 都是 aborted，Gate 判為 gradual；
   - 失敗的 call 不計費，所以呼叫次數上限永遠不會觸發；卡住的情況下會持有 GPU lock 長達 rollout 數 × 2 × 600 秒。

   這正是 027 要防止的後果。
2. **模型自己把對話拉長時，送出前的 fit check 停掉整個 run**（中嚴重度）。這個檢查估計 token 數 = request bytes ÷ 2：
   - 草案設定下，request 達 57,345 bytes 就會觸發，而 base request 只有約 3.7 KB；
   - 模型反覆讀取（每 turn 2–3 個 call）在 max_turns 之前就會觸發，原本應該以 max_turns 失敗結束；
   - 觸發時屬於 harness，run 停止、這個 rollout 變成 interrupted；依 019 §1 重跑時會換成另一個樣本（rollout 0 的偏差）；
   - 在 sweep 中，同一個 (k, i) 每次續跑都會再停一次，永遠無法完成。
3. **`local_probe --summarize` 漏掉失敗嘗試的呼叫**（中嚴重度）：只讀已完成的 trajectory。結果：
   - 達到 num_predict 的呼叫（4096 tokens、95 秒）、沒有名稱的 tool call、重試前的逾時都不在統計中；
   - 呼叫數少於 manifest 的計費次數，tool call 錯誤顯示為零；
   - 缺少 summary 的中斷目錄會讓它崩潰。

   這份報告是凍結設定的依據（026 §5）。
4. **wrapper 的本機通道 port 被占用時回報為「無法連線」**（低嚴重度）：例如另一個 session 的 wrapper 正在使用 11435。
   OpenSSH 在 bind 失敗時以 255 結束、遠端的 flock 根本沒有執行，wrapper 卻回報 exit 69（遠端 blocked），應該是 75（忙碌）。

同一輪判定不是缺陷，但一併處理（成本很低）：
- 彙整報告沒有標示 dry-run；
- 在 wrapper 的 cleanup 期間再按一次 Ctrl-C 會留下暫存檔（lock 本身仍會釋放）。

## 決策

1. **推論路徑連續失敗 → 服務不可用**：`OllamaHTTPClient` 記錄連續的 chat 傳輸失敗次數（成功時歸零）。`OllamaPolicy` 遇到
   OllamaTransportError 時：
   - `version()` 連不上 → `OllamaServerUnavailable`（027 §1）；
   - 這次失敗前一次的 chat 也失敗（連續 >= 2 次）→ `OllamaServerUnavailable`；
   - 其他 → 原例外（infrastructure，重試一次）。

   因此一次暫時性失敗仍會重試；同一個 rollout 的重試也失敗時，run 停止、該 rollout 不是最終結果，服務恢復後補回。
   沒有這個計數的 client（例如假 client）不受影響。
2. **模型自己用盡 context 預算 → agent_output**：送出前的 fit check 不通過時，分兩種情況：
   - 固定部分（system、user 與 tools，不含任何 assistant / tool 訊息）本身就放不下 → `OllamaContextError`（harness），
     設定有問題，必須停止；
   - 否則多出來的是對話中的 turn（模型自己拉長的，或 sweep 的前綴加上 rerun）→ `AdapterError`（agent_output，不計費）。
     該 rollout 記為 agent_error，就像以 max_turns 結束的失敗一樣計入結果，run 繼續。

   source 是完成的 baseline rollout，它在每個 turn 都通過了同一個檢查，所以 rerun 的第一個 request（前綴相同）不會因此失敗。
   估計方式（bytes ÷ 2）不變；開發題測試量到的實際 bytes / token 比例，作為凍結 num_ctx 的依據。
3. **開發題彙整的完整性**（`local_probe --summarize`）：
   - 讀取 manifest，回報所有計費的呼叫數，以及每一種中止的 (error_class, exception_type, message) 計數；
   - 每次呼叫的統計標示涵蓋範圍（只有已完成的 trajectory），並回報被排除的呼叫數；
   - 回報 run.json 的 dry_run；
   - 缺少 summary 的中斷目錄改由 manifest 彙整，不再崩潰。
4. **wrapper**：
   - ssh 因本機通道的 port 被占用而失敗（`Could not request local forwarding`）→ exit 75，訊息註明可能有另一個 wrapper 正在執行；
   - cleanup 開始時忽略後續的 INT / TERM / HUP，確保 kill / wait / 刪除暫存目錄一定完成。

## 為什麼

- aborted 是最終結果，只能來自模型或單次請求；推論服務持續失敗屬於停機，必須停止並保留可續跑的狀態。
- 模型自己造成的對話過長是模型的行為，應該計入結果；把它當成 harness 會讓重跑換樣本，造成偏差。
- 凍結的依據必須涵蓋失敗的呼叫，否則最需要的極端值會被漏掉。

## 放棄的替代方案

- 以空的 `/api/chat` 探測推論路徑：會以探測的參數載入模型；而連續失敗的計數不需要額外的呼叫。
- 為過長的對話新增一種 termination（例如 context_exhausted）：要改 trajectory 格式與 grader；agent_error 已經是「沒有完成」的既有類別。
- 放寬 fit check 的估計（例如 bytes ÷ 3）：在量到實際比例之前沒有依據。

## 對實驗的影響

- 不影響任何估計方法、判定規則或主要 metric。模型自己用盡 context 的 rollout 記為 agent_error（main success rate 中是失敗；
  在 success_rate_completed 中排除，decision 016）。
