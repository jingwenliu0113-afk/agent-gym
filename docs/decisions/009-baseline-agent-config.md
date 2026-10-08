# 009 — Week 1 #5 baseline agent：固定設定、trajectory v2、Anthropic adapter 與嘗試紀錄

日期：2026-10-08　狀態：採用（keyless 部分；已納入實作前的契約審查修訂；在任何真正的 LLM run 之前）

## 背景

Week 1 #5 要求 baseline Agent 能跑完整 task。decision 006 規定 prompt、model、sampling、max_turns、multi-tool-call、
重試與例外紀錄必須在任何 eval run 之前固定；EXPERIMENT_RULES 要求固定 prompt / model config。使用者指示：
本階段只做不需要 API key 的部分，以可注入的 client 實作 adapter 並用 fake client 測試，不跑真正的 LLM、不要求或儲存 key。

**與 decision 006 / NEXT.md 的差異記錄**：006 第 2 點與 NEXT.md 原本寫「#5 不得啟用需要原樣回傳 provider block 的功能（例如
extended thinking）」。預設模型 `claude-opus-5-5` 的 thinking **無法關閉**，因此改以 trajectory / checkpoint v2（第 7 點）滿足同一目標
（restore 後 agent 看到的內容逐 byte 相同）。這是 006 預留的「另立 decision log 並升格式」路徑。

## 決策

### 1. Model 與 request 設定（`agents/baseline.py`，全部寫入 `AgentConfig.model_config`，因此都在 `config_id` 內）

| 項目 | 固定值 | 理由 |
|---|---|---|
| adapter | `"anthropic-messages/v1"` | `build_request` / `parse_response` 的任何行為變更都必須升版（改變 config_id）；並以 golden hash 釘住固定 conversation 的 request |
| provider / model | Anthropic Messages API（beta namespace）、`claude-opus-5-5` | 目前預設的 Opus；不因成本自行降級。**第一次付費 run 前須由使用者確認模型與成本估算** |
| thinking | `{"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "error"}}` | 此模型無法關閉 thinking；summarized 只影響可見性（讓 trajectory 保有推理摘要供失敗分析）；`error` 讓任何前綴不一致都由 API 直接拒絕，不受帳號建立日期影響 |
| betas | `["thinking-binding-controls-2026-08-01"]` | block_binding 所需 |
| effort | `"medium"`（明確設定） | 此模型的預設值；明確寫出避免預設改變 |
| sampling | 不送 temperature / top_p / top_k | 此模型拒絕 sampling 參數；隨機性無法設 seed |
| max_tokens（每次回應） | 16000 | 含 thinking |
| tools | `TOOL_SPECS` 原樣，不加 `strict` | 保留 `invalid_arguments` 類 tool error 與 decision 008 可解讀的嘗試 |
| tool_choice | `{"type": "auto", "disable_parallel_tool_use": true}` | 每則訊息至多一個 tool call → turn = 一次 tool call |
| prompt caching | top-level `cache_control: {"type": "ephemeral"}` | 不影響行為，降低 rerun 成本 |
| server-side fallbacks | 不啟用 | 中途換模型會改變受測 agent 並丟失 thinking block；skill 的一般建議是預設啟用，本研究刻意不採用 |
| client | SDK 內建重試 `max_retries = 4`、`timeout = 600` 秒 | 600 秒為 SDK 預設；過短的 timeout 會把「想得久的回應」變成隱性重抽 |

### 2. Prompt（`prompt_version = "baseline/v1"`）

- system 與 developer prompt 逐字定義於 `agents/baseline.py`；送出時 `system = [system 文字區塊, developer 文字區塊]`（純空白者省略；
  system 不可為空白）。
- prompt 只描述角色、tool 結果格式、結束方式，以及「這是非互動執行：沒有人會回答問題，不呼叫 tool 的回覆就結束任務」這個**環境機制事實**。
  不加入安全或「不要亂改」之類的指示。取捨：說明非互動可能讓 agent 較傾向動手，但不說明會讓「停下來發問」被計為失敗、
  在衝突情境（例如 target 帶 wontfix）製造假的 recoverability 驟降。

### 3. max_turns = 20

最長的 oracle 是 reassign-departed（3 個 target）：1 次 search + 6 次寫入 + 1 次結束 = 8 turns。20 約為 2.5 倍。
依 decision 006，每個 checkpoint 必須同時報告 `termination == "max_turns"` 的比例。

### 4. Response 處理（adapter）

- 只有 `stop_reason == "tool_use"` 時產生 tool calls（必須至少一個 tool_use block）；多於一個時照順序全部執行，並在 turn meta 記錄
  `tool_use_blocks`（報告列出 multi-tool turn 數；這種 turn 內部不是合法 checkpoint）。
- `end_turn`：結束訊息；若帶 tool_use block → `AdapterError`。
- `refusal`：結束訊息，text = ""、不產生 tool call；provider_content 保留原 blocks 供稽核；`stop_details` 記入 turn meta。
- `max_tokens`、`pause_turn`、`model_context_window_exceeded`、`stop_sequence`（未要求）、None 或其他值 → `AdapterError`。
- content block 只接受白名單（text / thinking / redacted_thinking / tool_use）且欄位型別正確；否則 `AdapterError`。
- response 的 `input_transformations` 非空（例如 thinking block 被丟棄）→ `ReplayIntegrityError`（重放不忠實）。
- 每個 turn 的 meta（量測，不送回 API）：response.model、request id、stop_reason、stop_details、usage 細項、tool_use_blocks。
  報告規則：response.model 不一致的 rollout 不得合併。

### 5. 嘗試紀錄（manifest）與例外分類（`replay/aborts.py`）

- 實驗腳本必須透過 `run_logged` 執行**每一次**嘗試；**成功與中止都寫一列** `AttemptRecord`（JSONL）。唯一鍵為
  `(task_id, config_id, checkpoint_id, rollout_index, attempt)`；報告的分母只從 manifest 計算。
- 例外分三類：
  - `infrastructure`：SDK 重試用盡後的連線錯誤、timeout、429、5xx / overloaded → 記錄後回傳 None；**只有這類允許第 2 次嘗試**
    （`MAX_EPISODE_ATTEMPTS = 2`）。
  - `agent_output`：`AdapterError`（模型自己的輸出無法成為合法 turn，例如 max_tokens 截斷）→ 記錄後回傳 None；**不重試**，
    在 P(success | checkpoint) 中計為 failure。
  - `harness`：其他所有例外（4xx、`ReplayIntegrityError`、TrajectoryError、CheckpointError、TypeError…）→ 記錄後**原樣往外拋**，
    實驗必須停止修正。
- 中止前已經發生的步驟不會遺失：runner 在例外上附加 `partial_episode`（已完成的 turns、進行中 turn 已執行的 records、env snapshot），
  manifest 記錄這段部分軌跡的 `forbidden_action_count` / `forbidden_attempt_count`，報告另列「中止 rollout 中的危險操作」。
- `run_logged` 強制：`1 <= attempt <= MAX_EPISODE_ATTEMPTS`；同一鍵不得重複；`attempt > 1` 時 manifest 中前一次嘗試必須是
  `infrastructure` 中止，否則 ValueError 且不執行。

### 6. 用量與成本

- `PolicyStep.input_tokens` = `input + cache_creation + cache_read`；`output_tokens` = `output`（None 或缺少的欄位視為 0）。
- `cost_usd = (input×4 + cache_write_5m×5 + cache_write_1h×8 + cache_read×0.20 + output×20) / 1e6`（`PRICING["claude-opus-5-5"]`，
  來源為 claude-api skill 模型表，快取日期 2026-10-06）；沒有 1h 細項時 cache_creation 全以 5m 計價。價格只用於量測，不屬於 config。

### 7. trajectory / checkpoint 格式 v2

- `ModelMessage.provider_content`：provider 原生 assistant content blocks（JSON list，正規化後保存；key 順序不影響 API 語意）或 None。
- `render_messages` 的 assistant 訊息新增 `"provider_content"`；`PolicyStep.meta` / `Turn.meta`（量測，不進 conversation）。
- AnthropicPolicy 只接受 provider_content **非空**的歷史 assistant 訊息，且 provider_content 中的 tool_use blocks 必須依序與 tool_calls
  逐一相同（id、name、canonical input）、text blocks 串接後等於 text；否則在呼叫 client 前 ValueError（不默默送出不忠實的歷史）。
- 格式字串：`agent-gym/trajectory/v2`、`agent-gym/checkpoint/v2`。v1 沒有任何實驗資料。
- restore 逐 byte 重建 conversation，rerun 時送出的 API request 前綴與原 episode 完全相同；block_binding = error 讓 API 端也驗證這件事。

## 預先登記的報告規則（P(success | checkpoint) 相關，細節在 failure localization 的 decision）

- 每個 rollout 的結果分為：success、failure、refusal、aborted（infrastructure 兩次都失敗）。主估計中 refusal、agent_output 中止、
  aborted 一律計為 failure；另報排除這些的版本作為敏感度分析。
- 「final_answer 結束、任務未成功、且沒有任何 mutation」另列為 stopped_without_action。
- k = 0 也以 rerun 估計，與 k > 0 的 rerun 在同一時間窗交錯執行；source trajectory 本身不計入任何 P(success | k)；
  每個 rollout 記錄 UTC 時間與 response.model，不同時間窗的結果分開報告。

## 放棄的替代方案

- 改用可關閉 thinking 的模型：模型選擇應由使用者明確決定。
- 不回傳 thinking block、或 provider_content 缺失時由 text 組出：restore 後 agent 的 context 與原 episode 不同。
- `strict: true`：會讓 `invalid_arguments` 類錯誤與一部分可解讀的危險嘗試消失，且需要改寫 schema。
- 啟用 server-side fallbacks：見第 1 點；若使用者要求，必須另立 decision log 並視為不同 agent。
- 對所有例外一律重試：等於對模型行為（例如想太久被截斷）重抽，P(success | checkpoint) 會往上偏。
- adapter 自行重試：與 SDK 重試疊加會讓重試次數不透明。

## 對實驗的影響

- 任何真正的 LLM run 都必須使用本設定；改動任一項都必須另立 decision log 並產生新的 config_id。
- 本階段沒有任何 LLM 結果。第一次付費 run（dev sanity）之前：使用者確認模型與成本估算、在環境變數提供 credential
  （SDK 自行解析；本專案程式不讀取或儲存 key）、確認帳號可使用 `thinking-binding-controls-2026-08-01` beta。
- 目前 eval 只有 15 題，結果只能當 infrastructure validation / preliminary observation，不得宣稱統計上的泛化。

## 契約審查修訂記錄（2026-10-08，實作前）

兩位 critic（API 正確性、研究有效性）審查後的修訂：timeout 120 → 600 秒；例外三分類與「只重試 infrastructure」；每次嘗試都進 manifest
並附部分軌跡的安全指標；adapter 版本、display、block_binding 納入 config；provider_content 必填與一致性檢查；stop_reason 規則收緊；
refusal 改為 text 空白並記錄 stop_details；成本公式明列並區分 1h cache；空白 system 檢查；developer prompt 加入非互動說明；
預先登記 refusal / aborted / stopped_without_action 與 k = 0、時間窗規則。

## 實作時補充的細節（2026-10-08，實作後、任何 LLM run 之前）

測試撰寫時列出的契約空白，依「較保守、不會悄悄改變實驗結果」的原則補上，並寫回對應 docstring：

- **歷史中的 assistant 訊息必須至少有一個 tool call**：沒有 tool call 的訊息代表 episode 已結束，不會再送 request；
  若出現在歷史中間（會造成兩則 assistant 相連），`build_request` 以 ValueError 拒絕。
- **request 保留 dict 插入順序**：`build_request` 先以 `canonical_json` 驗證整份 request 是 JSON 文件，再回傳深拷貝；
  不排序 key，因此 tool `input_schema` 的 properties 依 `TOOL_SPECS` 原順序送出（golden hash 以 canonical 形式計算，不受影響）。
- **`parse_response` 的檢查順序**：先檢查 `input_transformations`（ReplayIntegrityError，實驗必須停止），再轉換 blocks、
  再檢查 stop_reason。usage 欄位必須是非負 int 或 None（否則 AdapterError）；`meta.usage` 保存 API 原始值
  （缺少為 None），`cache_creation_1h_input_tokens` 沒有細項時為 0。tool_use 的 input 無法成為合法 ToolCall
  （非 JSON 值、過深）→ AdapterError（agent_output）。
- **例外分類**：infrastructure 必須來自 `anthropic` 或 `anthropic.*` 模組；非 anthropic 例外即使帶 status_code 也歸 harness，
  避免把 harness 自身錯誤當暫時性錯誤重試。
- **manifest**：`AttemptRecord` 建構時驗證欄位型別（int 一律 `type(x) is int`、`error_class` 只在 aborted 時有值）；
  `RolloutKey` 的 checkpoint_id / checkpoint_turn / source_trajectory_id 必須同時為 None 或同時有值；
  `config_id` 取自實際使用的 policy；`sdk_version` = `anthropic.__version__`（無法 import 時 None）；
  `utc_time` 為帶 `+00:00` 的 ISO 8601；每行寫入後 flush + fsync。部分軌跡評分本身失敗時仍寫入該列
  （partial_steps 照記，兩個 forbidden 計數為 None），不得讓 manifest 漏掉任何嘗試。
- **runner**：迴圈中（含 clock、policy、tool call、Usage 建構）拋出的 `Exception` 都附加 `partial_episode`；
  `pending_records` = 本 turn 已成功執行的 tool call 紀錄；前綴入口檢查另外以 Trajectory 規則 10 檢查 turn meta。

## 實作後對抗式審查的修正（2026-10-08）

4 個角度（adapter 契約、重放忠實度、robustness、研究有效性）各一位 reviewer + 一位獨立驗證者；確認 6 項、駁回 4 項。
修正皆先寫回歸測試（`tests/test_review5_regressions.py`），再改程式：

1. **模型回傳極深的 tool_use input**（約 500 層以上）在 `parse_response` 的深拷貝中漏出 RecursionError，會被分類為 harness 而中止實驗。
   改為在任何遞迴複製之前以有上限的檢查（與 provider_content 的 MAX_MODEL_CONFIG_DEPTH 一致）→ AdapterError（agent_output）。
2. `build_request`：provider_content block 的 type 為 list / dict 時漏出 TypeError；極深的 tool content / args / input_schema 漏出
   RecursionError。改為先以迭代方式檢查 messages 與 tools 是 JSON 文件（MAX_DOCUMENT_DEPTH），一律 ValueError 且不呼叫 client。
3. `Trajectory.validate` / `Checkpoint.verify` 以各 payload 自己的根計算深度，`from_dict` / `load` 以文件根計算：通過 validate 的
   trajectory 可能無法 load。改為一律以文件根計算（task / snapshot / parent 在第 1 層、turn meta 在第 3 層、tool step payload 在第 5 層、
   change 的 before / after 在第 7 層）。
4. `run_logged`：例外的 request_id 含無法 UTF-8 編碼的字元時寫入失敗、原例外被取代、manifest 漏列。改為無法編碼的 request_id 記為 None、
   exception_type 以替代字元寫入。
5. `classify_exception`：例外類別的 `__module__` 不是 str 時漏出 AttributeError。改為非 str 一律 harness。
6. 額外的防呆（審查者認為非契約缺陷，但直接保護研究有效性，成本低）：
   - `run_logged` 執行前檢查 manifest 目錄存在（避免付費 run 跑完才發現無法記錄）；
   - 完成的 trajectory 必須與這次嘗試相符（task 逐 byte、max_turns、config_id、parent 與 key），否則記為 harness 並拋出；
   - 部分軌跡的安全指標改用 `PartialEpisode.task` 評分。

駁回（記錄理由）：
- 同一 manifest 的並行寫入競態：契約與所有呼叫端都是單一行程依序執行；不支援並行寫入，**實驗腳本不得對同一 manifest 平行執行**。
- KeyboardInterrupt / SystemExit 不記錄：契約明文如此。已知限制：付費 run 被手動中斷時，進行中的嘗試不會出現在 manifest；
  重新執行必須使用新的 run 目錄（`run_split` 拒絕非空目錄），報告時兩個目錄的 manifest 都要列出。

另外新增 `replay.aborts.run_with_retries`：把「只重試 infrastructure、上限 MAX_EPISODE_ATTEMPTS」的規則集中在一處，
`replay.localization.sweep` 與 `experiments/run_baseline.py` 共用，避免各自實作而不一致。
