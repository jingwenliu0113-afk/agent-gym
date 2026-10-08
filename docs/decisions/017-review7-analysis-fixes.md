# 017 — 補上已登記但漏實作的規則；分析工具改以 manifest 與 eval pool 為準（在任何 LLM 結果之前）

日期：2026-10-08　狀態：採用（實作 decision 003 / 006 / 008 / 009 / 012 已登記的規則；修正 016 A8 與 009 §4 的矛盾；
其餘為 016 沒寫清楚的操作化細節。沒有任何 LLM 結果，也沒有改主要 metric、資料切分、模型或 SweepPlan）

## 背景

第七輪對抗式審查（分析工具、README 示範、Core 完整性）確認：幾條**已採用**的規則在程式中沒有實作，分析工具也有幾處只信任
summary、不比對 manifest 與 eval pool。具體如下：

- decision 006：「若驟降同時伴隨 max_turns 比例上升，不得認定為 recoverability cliff」，`detect_cliff` 沒有實作。重現：rerun 從 k ≥ 1
  起都用完 turn 預算，該 source 仍被判為 cliff，Gate 也因此被判為滿足。
- decision 009：「每個 rollout 記錄 UTC 時間與 response.model」。sweep 的 rerun trajectory 從未存檔，response.model 因此遺失；
  付費 sweep 跑完後無法再取得每個 rollout 的步驟。
- decision 009 §4：「response.model 不一致的 rollout 不得合併」。016 A8 卻只要求「標示」，analyze_baseline 照 A8 把不同 model 的
  rollout 併入同一個 pooled 估計。016 開頭明說不改 009 的規則，所以以 009 為準。
- decision 003 / SPEC §4：報告至少分開呈現 outcome_correct、tool errors、steps、usage；analyze_baseline 沒有這些欄位。
- decision 008：要標示被擋下（effective == False）的 forbidden attempt；009 §4：要報告 multi-tool turn 數。兩者都沒有。
- decision 016 A1、009 / 012：「中斷後重跑，所有目錄的 rollout 都計入、所有目錄都列出」。但中斷的 run 沒有 summary.json，
  兩支分析工具都拒絕讀取。
- decision 012 §1 / §3：缺漏的 source 要列出；任一 family 沒有完成的 source 就不做 bootstrap。analyze_sweep 只照抄 summary，
  因此 baseline 因預算停止時，缺了整個 family 仍會報出 cliff 比例。
- 016 A5：每個中止的嘗試都要列出部分安全指標。程式只列計數 > 0 的，還把部分評分失敗（None，安全性未知）當成 0。

## 決策

### 1. max_turns 混淆（實作 decision 006）

- `replay.localization.assess_drop(estimates, drop_threshold, criterion)` 回傳 `DropAssessment`，包含 largest_drop、largest_drop_k、
  `separated`（下降 >= 門檻，且兩點的 CI 依 criterion 不重疊，也就是原本的 is_cliff）、`max_turns_confounded`、`is_cliff`。
- **max_turns_confounded** = separated，且最大下降的後一點的 max_turns 比例**嚴格大於**前一點的比例
  （以精確有理數 `max_turns_terminations / n` 比較）。
- **is_cliff = separated 且非 max_turns_confounded**。仍然只看最大下降；被混淆時**不改看第二大的下降**，避免事後搜尋。
- `detect_cliff` 維持原本的回傳形式 (largest_drop, largest_drop_k, is_cliff)，is_cliff 依上面的新定義。
- `SweepResult` 新增 `max_turns_confounded`；sweep 輸出格式升為 `agent-gym/sweep-run/v2`（目前還沒有任何 v1 的真實資料）。
- 被混淆的 source 在報告中另外列出（「若非 max_turns 會是 cliff」），不計入 cliff 比例，也不進 Gate。
- decision 011 的偵測力模擬與 README 示範都不受影響：synthetic rollout 一律以 final_answer 結束，max_turns 比例恆為 0。

### 2. sweep rollout 的 trajectory 存檔（實作 decision 009）

- `sweep(..., rollout_dir=...)`：每個完成的 rerun trajectory 存成 `rollout_dir/k<k>-r<i>.json`。`run_sweeps` 使用
  `runs/<name>/rollouts/<task_id>/`。檔案的 trajectory_id 必須與 manifest 中該 rollout 最後一次嘗試的 trajectory_id 相同。
- 一個 sweep rollout 的 response.model = 該 trajectory 中 **k 之後的 turns**（rerun 自己產生的 turns）的 response_model。

### 3. source 選取以 eval pool 為準（012 §3 的實作細節）

- `select_sources(baseline_run_dir)` 回傳**全部 15 個 eval task**，依 eval pool 的順序。沒有 rollout 0，或 rollout 0 未完成 → None。
- 讀取方式與 analyze_baseline 相同（summary、manifest、trajectory 三者交叉核對），任何不一致 → ValueError。
  例如：未知的 outcome，或 rollout 0 標為 agent_error、但 manifest 顯示已完成。
- 結果：baseline 因預算停止時，沒跑到的 task 會列為缺漏，它們的 family 也會讓 run_sweeps 不做 bootstrap。

### 4. eval baseline 分析（修正與補充 016 A）

1. **中斷的 run（沒有 summary.json）**：由 manifest 與 `trajectories/` 重建每個 rollout。最後一次嘗試完成 → 由 trajectory 分類。
   中止的情況再分三種：
   - 最後一次嘗試是 agent_output 中止 → agent_error；
   - infrastructure 中止且已達 MAX_EPISODE_ATTEMPTS → aborted；
   - infrastructure 中止但還沒用完嘗試 → **interrupted**（重試在中斷時進行中）。

   interrupted 的 rollout 不計入任何 rollout 指標，但會另外列出，它的成本與部分安全指標照常計入。報告中標示該目錄「中斷（沒有 summary）」。
   有 summary 的目錄不可能出現 interrupted（出現 → ValueError）。
2. **目錄識別**：同一個目錄（resolve 後相同）傳兩次 → ValueError。run 標籤用目錄名稱；名稱重複時改用傳入的路徑字串。
3. **summary 的每個欄位都要與 manifest / trajectory 相符**：完成的 rollout 的 termination 要等於 trajectory 與 manifest 的 termination；
   未完成的 rollout 的 outcome 要依最後一次嘗試的 error_class 判定，且 stopped_without_action 為 False。
   task_id 必須是 str、rollout_index 必須是 int >= 0。不符 → ValueError。
4. **中止嘗試的部分安全指標（016 A5 的補充）**：**每一個** aborted 嘗試都各列一筆，包括之後重試成功的較早嘗試，以及 interrupted rollout 的嘗試。
   每筆附 partial_steps，None 原樣保留：partial_steps 為 None 表示「沒有部分軌跡」；計數為 None 而 partial_steps 不是 None 表示「未知（部分評分失敗）」。
5. **response.model 不得合併（以 009 §4 為準，取代 016 A8 的「標示」）**：所有完成的 rollout 中出現的 response_model
   不只一種時，所有 pooled 指標、CI、per-family、branch、時間窗估計一律不報，只列計數、清單與成本，並說明原因。
   需要比較時，由使用者把不同 model 的目錄分開分析。只有一種 model 時照常報告，並列出該 model。
6. **補上 decision 003 / SPEC §4 的指標**：只在完成的 rollout 上計算，task 等權，95% CI 用同一個 family 分層 bootstrap：
   - outcome_correct_rate；
   - tool_errors、steps；
   - input_tokens、output_tokens、latency_s（每個 rollout 的總和）。

   某個 rollout 的 token 或 latency 為 None（無法量測）時，該 rollout 不適用這項指標，由 016 A3 的排除規則處理。
   成本維持 016 A10（manifest 加總）。
7. **multi-tool turn 與被擋下的 forbidden attempt**：列出 tool_use_blocks > 1 的 turn（run、task、rollout、turn）與總數。
   另外列出 effective == False 的 forbidden attempt（run、task、rollout、step、rule），範圍是完成的 rollout。

### 5. sweep 分析（修正與補充 016 B）

1. **輸入**：一或多個 sweep 目錄（中斷後重跑），加上 source 所屬的 baseline 目錄。每個目錄都必須有 manifest。
   summary.json 可以沒有（中斷）；有的話逐欄驗證型別，並與檔案核對：format、plan、criterion、config_id、completed 等於實際的 sweep 檔，
   sources 與 missing_sources 等於 select_sources 的結果。
   所有目錄的 plan 與 criterion 必須相同。同一個 source 在多個目錄都完成 → ValueError（不挑選）。
2. **完成的 sweep 一律重算核對**：
   - rollouts 必須與 manifest 中每個 (k, rollout_index) 的最後一次嘗試相符（嘗試數、完成與否、error_class 決定的標籤）；
   - 完成的 rollout 以存檔的 trajectory 重新 verify 與 classify_rollout；
   - estimates 以 estimate_checkpoint 重算；largest_drop、k、max_turns_confounded、is_cliff 以 assess_drop 重算。

   不符 → ValueError。summary 中的 cliff_rate 等彙總值不採用，一律由分析重算。
3. **以 eval pool 為準**：
   - missing_sources = select_sources 為 None 的 eval task；
   - not_completed = 有 source 但沒有任何目錄完成 sweep 的 task；
   - cliff 比例與平均最大下降：只要有任一 eval family 沒有完成的 sweep，就不做 bootstrap（012 §1）；
   - Gate 照常在已完成的曲線上評估，但有缺漏或未完成時標示「資料不完整」。
4. **response.model**：source 的所有 turns 與每個 sweep rollout（k 之後的 turns）的 response_model 只能有一種。
   否則不報彙總、不評估 Gate，曲線照列並標示原因（009 §4）。
5. **時間窗**：每個 rollout 以最後一次嘗試的 utc_time 取 UTC 日期。每個日期報 rollout 數、source 數、success 比例（點估計，描述用；010 §2）。
6. **中止嘗試的部分安全指標**：sweep manifest 中每個 aborted 嘗試都列一筆，規則同 4.4。
   沒有完成的 source（中斷）的 rollout 不進曲線，但成本與部分安全指標照列。
7. **曲線表**加上 failure / refusal / agent_error / aborted 計數；max_turns 混淆的 source 另外標示。
8. **成本**：所有目錄 manifest 的 cost_usd 加總，以及 unknown_cost_calls。

## 為什麼

- 這些規則在看到結果之前就已登記，程式漏掉了。等真實結果出來才補，就會變成看結果改分析。
- 只信任 summary 的分析，會在預算停止、中斷、或檔案被改動時，把不完整或錯誤的資料當成完整結果報告。
- 部分評分失敗表示「不知道安全不安全」，不能報告成「沒有危險行為」。

## 放棄的替代方案

- max_turns 混淆改用「排除 max_turns 終止的 rollout 後重算下降」：decision 006 的文字是「比例上升就不得認定」，重算是另一種規則。
- 被混淆時改看第二大的下降：等於在結果上搜尋，可能把雜訊當成 cliff。
- response.model 不同時依 model 自動分組：中止的 rollout 沒有 trajectory，無法歸到任何 model，分組會悄悄丟掉它們。
- interrupted rollout 計為 aborted：它的重試只是因為中斷而沒有發生，算成 aborted 會把中斷誤報成 agent 或 API 的失敗。
- sweep 只在 manifest 加 response_model 欄位、不存 trajectory：仍然無法事後檢查每個 rollout 的步驟（SPEC §5 的 Recorder 要求）。

## 對實驗的影響

- 付費 sweep 每個完成的 rollout 都會留下 trajectory 檔（`runs/` 為 gitignored；每條約為 source 的大小）。
- cliff 判定排除 max_turns 混淆的下降，cliff 比例與 Gate 因此比原本保守。
- 分析報告會多出：中斷目錄的 rollout、所有中止嘗試、decision 003 的指標、multi-tool turn、被擋下的嘗試、時間窗、response.model 檢查。
- Gate 的「收尾」分支（多數 gradual 且彼此不同）仍未操作化，屬於 decision 012 §4 待使用者確認的部分，本 decision 不處理。
