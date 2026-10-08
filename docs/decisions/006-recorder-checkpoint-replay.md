# 006 — Recorder、checkpoint 與 replay 協定（Week 1 #6、#8）

日期：2026-10-08　狀態：採用（已納入實作前的契約審查修訂）

## 背景

SPEC §5 要求每一步保存 model message、tool call、args、result、state hash before / after，
且完整 checkpoint 可重建 environment state、conversation history、tool calls / results、system prompt、
developer prompt、prompt version、model config；核心 invariant 是 restore 後 agent 與 environment 看到的
observable state 與原 checkpoint 一致。SPEC §6 之後要從 checkpoint 重新 rollout 估 P(success | checkpoint)。
decision 004 也留下「前綴來源綁定」交給 Recorder。

## 決策

1. **Policy 必須對 conversation 無狀態**：`Policy.next_message(messages, tools)` 只能依輸入決定輸出
   （LLM 的抽樣隨機性除外），不得保留跨呼叫的隱藏狀態。「restore agent」= 重建 conversation。
   每個 policy 公開 `agent_config`（policy、prompt_version、system_prompt、developer_prompt、model_config）；
   **是否為同一個 agent 一律以 `config_id = sha256_of(agent_config.to_dict())` 判定**。
2. **Agent 看到的輸入是決定性的**：
   - conversation 由 `replay/trajectory.py::render_messages` 產生：`system` → `developer` → `user`（task.instruction）→
     每個 turn 的 `assistant`（text + tool_calls）→ 每個 tool call 一則 `tool`（內容為 `canonical_json(observation)`）。
     形狀定義於 `agents/policy.py` 模組 docstring。
   - ToolCall.args 與 model_config 在建構時正規化為 `json.loads(canonical_json(x))`（key 排序），因此 live run 與
     save / load 後 restore 的 conversation 以 order-sensitive 的 `json.dumps` 比較也逐 byte 相同。
   - **TOOL_SPECS 全文（含 description）屬於 `SCHEMA_VERSION`**，任何改動都必須升版；測試以 golden hash 釘住。
     env.restore 與 checkpoint 重放都會比對 schema_version，因此跨版本的 rerun 會被拒絕。
   - provider 專屬的轉換（例如 Anthropic 把 developer prompt 併入 system）屬於 Week 1 #5 的 adapter。
     trajectory/v1 的 assistant 訊息只有 text 與 tool_calls；#5 不得啟用需要原樣回傳 provider block 的功能
     （例如 extended thinking），否則須另立 decision log 並升 trajectory 格式。
3. **Turn 是 checkpoint 的粒度**：一個 turn = 一次 model 呼叫 + 該訊息所有 tool calls 的執行結果。
   checkpoint k = 完成 k 個 turns 後、下一次 model 呼叫前的決策點，**0 <= k < max_turns**。
   沒有 tool call 的 assistant 訊息結束 episode（`final_answer`）；達到 `max_turns` 則 `max_turns`。
   `max_turns` 計入整條 trajectory（含 checkpoint 前綴）。
4. **Trajectory 格式**（`agent-gym/trajectory/v1`）：完整 task、agent_config、max_turns、turns
   （model message、tool steps = env 的 ToolCallRecord、usage）、termination、final env snapshot、
   parent（rerun 時：來源 trajectory_id、checkpoint_id、turn）。`trajectory_id` = 內容的 sha256，
   是**紀錄 id**（含量測），實驗層不得以它去重或當 rollout key。
5. **量測與內容分離**：model message 與 tool steps 是決定性內容；每個 turn 的 `usage`（tokens、runner 量的 latency、
   cost）是量測。runner 每個 turn 恰好呼叫 clock 兩次（next_message 前後）；測試用常數 clock 讓重跑逐 byte 相同。
   episode usage 以內建 `sum` 依 turn 順序加總，任一 turn 未知則為 None。
6. **Checkpoint 格式**（`agent-gym/checkpoint/v1`）：task、agent_config、max_turns、turn、前綴 turns、env snapshot、
   source_trajectory_id、`checkpoint_id`（內容的 sha256）。
   - **前綴來源綁定（解決 decision 004 的「切錯前綴」）**：checkpoint 只由 `make_checkpoint(trajectory, k)` 產生，
     前綴 turns 與 env snapshot 一起保存；載入時驗證 checkpoint_id，並從 task 初始狀態重放前綴，
     要求每筆 record 與 env snapshot 都逐 byte 相同。續跑的內部函式 `_continue_episode` 不公開，入口會重放驗證前綴與 env 一致。
   - **對刻意竄改的偵測需要外部錨點**：`verify_lineage(child, source, checkpoint=...)` 以來源 trajectory 與
     **呼叫端自己發出的 checkpoint** 為錨點；child 自報的 parent.turn 不可信（契約審查發現只比 parent.turn 會被繞過）。
   - 本模組所有「相同」都以 canonical_json 比較，所有 int 都排除 bool（不重犯 decision 004 的 `True == 1 == 1.0`）。
7. **Rerun**：`rerun(checkpoint, policy)` 要求 config_id 相同，以 restore 的 env 與 conversation 繼續；
   產生的 trajectory 含完整前綴、至少一個新 turn，可直接 `grade()`（重放 integrity 驗證整條）。
8. **Week 1 不含 LLM**：提供 `ScriptedPolicy`（依 conversation 中已有的 assistant 訊息數決定播放位置，因此無狀態），
   用於 oracle 執行與 restore / rerun 的決定性驗證。LLM policy 在 Week 1 #5 以同一介面接入。
9. **存檔**：canonical_json + 換行；讀檔用 strict UTF-8，拒絕 NaN / Infinity 字面量與重複 key。

## 對實驗的影響（預先登記）

- SPEC §6 的 rerun 一律從 `make_checkpoint` 產生的 checkpoint 出發；checkpoint 屬於其 task cluster。
- 只有 `(generator_version, grader_version, schema_version, agent.config_id, max_turns)` 全部相同的結果可以合併。
- **max_turns** 在任何 eval run 前固定並記錄理由；**每個 checkpoint 的 P(success | checkpoint) 必須同時報告 rerun 中
  `termination == "max_turns"` 的比例**。若驟降同時伴隨該比例上升，不得認定為 recoverability cliff（可能只是剩餘預算不足）。
- failure localization 的解析度是 turn。#5 baseline 是否允許一則訊息多個 tool call，必須在其 decision log 中、
  任何 eval run 前固定並寫入 model_config。
- **例外中止**的 episode 不產生 trajectory，但實驗腳本必須記錄（task_id、checkpoint_id、attempt、例外型別、已完成 turn 數）
  並在報告中列出中止數，不得默默重抽；暫時性 API 錯誤的重試屬於 adapter 內部，上限在看結果前固定。
- rerun trajectory 的 usage 包含前綴 turns 的原始量測；「續跑成本」只取 index >= checkpoint.turn 的 turns。

## 為什麼

- 無狀態 policy 讓 checkpoint 只需要資料（conversation + env snapshot）就能完整重建 agent，這是
  P(success | checkpoint) 能成立的前提；LLM API 本身就是無狀態的，不增加限制。
- turn 粒度對應 agent 真正的決策點；在 turn 中間 restore 會產生 agent 從未見過的 conversation。
- `k < max_turns`：k == max_turns 時 rerun 不會呼叫 policy，是變異為 0 的退化點。
- 決定性 rerun（ScriptedPolicy + 常數 clock）讓核心 invariant 可被逐 byte 驗證到整個 episode，而不只 env 層。

## 放棄的替代方案

- 每個 tool call 都可當 checkpoint：會在 agent 未看到完整 tool results 的位置切斷 conversation。
- 序列化 policy 物件（pickle）：與 LLM 無狀態 API 不符，也無法跨 process / 版本。
- checkpoint 只存 env snapshot + 指向來源 trajectory 的參照：來源遺失就無法使用，也無法單獨驗證。
- 在 trajectory / checkpoint 另存 tool_specs hash：與「TOOL_SPECS 屬於 SCHEMA_VERSION」重複；選擇單一版本來源。
- 現在就實作 LLM adapter：需要 API key（使用者指示暫緩），介面先定好即可。

## 契約審查修訂記錄（2026-10-08，實作前）

修正 render_messages 的模組路徑（原誤寫為 `agents/conversation.py`）；新增 config_id 與 canonical 比較規則；
TOOL_SPECS 納入 SCHEMA_VERSION；`continue_episode` 改為私有並加入口檢查；checkpoint 限制 k < max_turns；
lineage 改以呼叫端的 checkpoint 為錨點；args / model_config 正規化；clock 呼叫次數、usage 加總、
strict load、例外中止與 max_turns 報告規則。env / grader docstring 中「自行切 records[:k]」的舊指引已改為指向 `rerun`。
